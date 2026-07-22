#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, enum_value
from node_editor.image_metadata import (
    compression_parameters_from_metadata,
    compression_roundtrip,
    ensure_image_extension,
    summarize_metadata,
    write_encoded_image,
)
from node_editor.util import convert_cv_to_dpg, dpg_get_value, dpg_set_value


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.2'

    _save_path_setting_key = '__save_path__'
    _last_encoded_bytes_by_node = {}
    _last_codec_by_node = {}

    node_label = 'Image Compression'
    node_tag = 'ImageCompression'

    parameters = [
        {
            'name': 'codec',
            'type': PortDataType.TEXT,
            'port': 'Input02',
            'widget': 'combo',
            'label': 'Codec',
            'items': ['JPEG', 'PNG', 'WEBP'],
            'default': 'JPEG',
            'cast': str,
        },
        {
            'name': 'quality',
            'type': PortDataType.INT,
            'port': 'Input03',
            'widget': 'slider_int',
            'label': 'Quality',
            'min': 1,
            'max': 100,
            'default': 90,
            'cast': int,
        },
        {
            'name': 'subsampling',
            'type': PortDataType.TEXT,
            'port': 'Input04',
            'widget': 'combo',
            'label': 'Subsampling',
            'items': ['Auto', '4:4:4', '4:2:2', '4:2:0'],
            'default': 'Auto',
            'cast': str,
        },
        {
            'name': 'png_compression',
            'type': PortDataType.INT,
            'port': 'Input05',
            'widget': 'slider_int',
            'label': 'PNG Level',
            'min': 0,
            'max': 9,
            'default': 3,
            'cast': int,
        },
        {
            'name': 'generation',
            'type': PortDataType.INT,
            'port': 'Input06',
            'widget': 'slider_int',
            'label': 'Generations',
            'min': 1,
            'max': 10,
            'default': 1,
            'cast': int,
        },
        {
            'name': 'progressive',
            'type': PortDataType.TEXT,
            'port': 'Input07',
            'widget': 'checkbox',
            'label': 'Progressive JPEG',
            'default': False,
            'cast': bool,
        },
        {
            'name': 'optimize',
            'type': PortDataType.TEXT,
            'port': 'Input08',
            'widget': 'checkbox',
            'label': 'Optimize JPEG',
            'default': False,
            'cast': bool,
        },
    ]

    def _ensure_declarative_port_handles(self, node_id, include_elapsed=False):
        ports = super()._ensure_declarative_port_handles(
            node_id,
            include_elapsed=include_elapsed,
        )
        if not hasattr(ports, 'metadata_input'):
            self.create_port(
                node_id,
                'metadata_input',
                InputPort(PortDataType.METADATA, index=9),
            )
        if not hasattr(ports, 'metadata'):
            self.create_port(
                node_id,
                'metadata',
                OutputPort(PortDataType.METADATA, index=3),
            )
        return self.ports(node_id)

    def build_custom_ui(self, tag_node_name, node_id, width, callback):
        del tag_node_name, callback
        ports = self.ports(node_id)
        with dpg.node_attribute(
            tag=ports.metadata_input.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Input,
        ):
            dpg.add_text('match metadata')
        save_path_value_tag = self._save_path_value_tag(node_id)
        save_status_value_tag = self._save_status_value_tag(node_id)
        save_dialog_tag = self._save_dialog_tag(node_id)

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            callback=self._callback_save_image,
            tag=save_dialog_tag,
            user_data=node_id,
        ):
            dpg.add_file_extension('JPEG (*.jpg *.jpeg){.jpg,.jpeg}')
            dpg.add_file_extension('PNG (*.png){.png}')
            dpg.add_file_extension('WebP (*.webp){.webp}')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        with dpg.node_attribute(
            tag=self._save_control_tag(node_id),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            dpg.add_input_text(
                label='Save path',
                tag=save_path_value_tag,
                default_value='',
                width=width,
            )
            dpg.add_button(
                label='Browse save...',
                width=width,
                callback=lambda: dpg.show_item(save_dialog_tag),
            )
            dpg.add_button(
                label='Save current compressed image',
                width=width,
                callback=self._save_image_button,
                user_data=node_id,
            )
            dpg.add_text(
                tag=save_status_value_tag,
                default_value='Save status: waiting for image',
            )

        with dpg.node_attribute(
            tag=ports.metadata.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Output,
        ):
            dpg.add_text(
                tag=ports.metadata.value_tag,
                default_value='compression metadata',
            )

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        ports = self._ensure_declarative_port_handles(
            node_id,
            include_elapsed=self.show_elapsed_time
            and self._opencv_setting_dict.get('use_pref_counter', False),
        )
        output_image_value_tag = ports.image.value_tag
        elapsed_value_tag = getattr(ports, 'elapsed', None)
        if elapsed_value_tag is not None:
            elapsed_value_tag = elapsed_value_tag.value_tag

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        connection_info_src = ''
        self._sync_linked_parameters(connection_list, node_result_dict)
        self._apply_linked_metadata_settings(
            node_id,
            connection_list,
            node_result_dict,
        )

        for (
            connection_info,
            source_tag,
            _,
            connection_type,
        ) in self._iter_connection_infos(connection_list):
            if enum_value(connection_type) == self.TYPE_IMAGE:
                connection_info_src = self._connection_source_node_key(
                    connection_info,
                    source_tag,
                )

        frame = node_image_dict.get(connection_info_src, None)
        parameter_values = self._get_parameter_values(node_id)
        parameter_values = self.normalize_parameter_values(
            self._node_name(node_id),
            parameter_values,
        )

        start_time = None
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frame is not None:
            frame, result = self.process(frame, **parameter_values)
        else:
            result = None

        if frame is not None and use_pref_counter and start_time is not None:
            elapsed_time = int((time.perf_counter() - start_time) * 1000)
            dpg_set_value(elapsed_value_tag, str(elapsed_time).zfill(4) + 'ms')

        if frame is not None:
            texture = convert_cv_to_dpg(frame, small_window_w, small_window_h)
            dpg_set_value(output_image_value_tag, texture)

        compression_metadata = None
        if isinstance(result, dict):
            compression_metadata = result.get('compression_metadata')
        if compression_metadata is not None:
            encoded_bytes = compression_metadata.pop('encoded_bytes', None)
            if encoded_bytes is not None:
                self._last_encoded_bytes_by_node[str(node_id)] = encoded_bytes
                self._last_codec_by_node[str(node_id)] = compression_metadata.get(
                    'codec',
                    parameter_values.get('codec', 'JPEG'),
                )
            dpg_set_value(
                self.ports(node_id).metadata.value_tag,
                summarize_metadata({
                    'format': compression_metadata.get('codec'),
                    'jpeg': compression_metadata.get('jpeg'),
                }),
            )
        return frame, result

    def get_custom_setting_dict(self, tag_node_name, node_id):
        del tag_node_name
        return {
            self._save_path_setting_key: dpg_get_value(
                self._save_path_value_tag(node_id)
            ),
        }

    def set_custom_setting_dict(self, tag_node_name, node_id, setting_dict):
        del tag_node_name
        save_path = setting_dict.get(self._save_path_setting_key, '')
        dpg_set_value(self._save_path_value_tag(node_id), save_path)

    def close(self, node_id):
        self._last_encoded_bytes_by_node.pop(str(node_id), None)
        self._last_codec_by_node.pop(str(node_id), None)

    def _save_control_tag(self, node_id):
        return self._control_tag(
            self._node_name(node_id),
            self.TYPE_TEXT,
            'SaveImage',
        )

    def _save_path_value_tag(self, node_id):
        return self._control_value_tag(
            self._node_name(node_id),
            self.TYPE_TEXT,
            'SavePath',
        )

    def _save_status_value_tag(self, node_id):
        return self._control_value_tag(
            self._node_name(node_id),
            self.TYPE_TEXT,
            'SaveStatus',
        )

    def _save_dialog_tag(self, node_id):
        return f'image_compression_save:{node_id}'

    def _callback_save_image(self, sender, app_data, user_data):
        del sender
        path = self._file_dialog_path(app_data)
        if path:
            dpg_set_value(self._save_path_value_tag(user_data), path)
        self._save_image(user_data, path)

    def _save_image_button(self, sender, app_data, user_data):
        del sender, app_data
        path = dpg_get_value(self._save_path_value_tag(user_data))
        self._save_image(user_data, path)

    def _save_image(self, node_id, path):
        status_tag = self._save_status_value_tag(node_id)
        if not path:
            dpg_set_value(status_tag, 'Save status: choose a file path')
            return None
        encoded_bytes = self._last_encoded_bytes_by_node.get(str(node_id))
        if not encoded_bytes:
            dpg_set_value(status_tag, 'Save status: no image to save yet')
            return None
        codec = self._last_codec_by_node.get(str(node_id), 'JPEG')
        output_path = ensure_image_extension(path, codec)
        try:
            write_encoded_image(output_path, encoded_bytes)
        except (OSError, ValueError) as exc:
            dpg_set_value(status_tag, f'Save failed: {exc}')
            return None
        dpg_set_value(self._save_path_value_tag(node_id), output_path)
        dpg_set_value(status_tag, f'Saved: {output_path}')
        return output_path

    @staticmethod
    def _file_dialog_path(app_data):
        if isinstance(app_data, dict):
            return (
                app_data.get('file_path_name')
                or app_data.get('current_path')
                or app_data.get('current_file')
                or ''
            )
        return app_data or ''

    def _linked_metadata(self, node_id, connection_list, node_result_dict):
        ports = self._ensure_declarative_port_handles(node_id)
        for (
            connection_info,
            source_tag,
            destination_tag,
            connection_type,
        ) in self._iter_connection_infos(connection_list):
            if destination_tag != ports.metadata_input.dpg_tag:
                continue
            if enum_value(connection_type) != self.TYPE_METADATA:
                continue
            source_node_key = self._connection_source_node_key(
                connection_info,
                source_tag,
            )
            source_result = node_result_dict.get(source_node_key)
            if not isinstance(source_result, dict):
                continue
            metadata = source_result.get('metadata') or source_result.get(
                'compression_metadata',
            )
            if metadata is not None:
                return metadata
        return None

    def _apply_linked_metadata_settings(
        self,
        node_id,
        connection_list,
        node_result_dict,
    ):
        metadata = self._linked_metadata(node_id, connection_list, node_result_dict)
        parameters = compression_parameters_from_metadata(metadata)
        if not parameters:
            return

        name_to_parameter = {
            parameter['name']: parameter
            for parameter in self.parameters
        }
        for name, value in parameters.items():
            if name == 'codec':
                value = str(value).upper()
            parameter = name_to_parameter.get(name)
            if parameter is None:
                continue
            value = self._cast_parameter_value(parameter, value)
            value = self._clamp_parameter_value(parameter, value)
            value_tag = self._parameter_port_ref(node_id, parameter).value_tag
            input_tag = None
            if parameter.get('widget') in ('slider_int', 'slider_float'):
                input_tag = self._slider_input_tag(value_tag)
            self._set_parameter_widget_values(value_tag, input_tag, value)
            self._last_parameter_values[value_tag] = value

    def process(self, frame, **parameter_values):
        output, compression_metadata = compression_roundtrip(
            frame,
            parameter_values.get('codec', 'JPEG'),
            quality=parameter_values.get('quality', 90),
            subsampling=parameter_values.get('subsampling', 'Auto'),
            progressive=parameter_values.get('progressive', False),
            optimize=parameter_values.get('optimize', False),
            png_compression=parameter_values.get('png_compression', 3),
            generation=parameter_values.get('generation', 1),
            include_encoded_bytes=True,
        )
        return output, {
            'compression_metadata': compression_metadata,
            'metadata': {
                'format': compression_metadata.get('codec'),
                'jpeg': compression_metadata.get('jpeg'),
                'compression': compression_metadata,
            },
        }
