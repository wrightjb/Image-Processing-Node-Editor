#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, enum_value
from node_editor.image_metadata import (
    compression_parameters_from_metadata,
    compression_roundtrip,
    summarize_metadata,
)
from node_editor.util import convert_cv_to_dpg, dpg_set_value


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

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
        del tag_node_name, width, callback
        ports = self.ports(node_id)
        with dpg.node_attribute(
            tag=ports.metadata_input.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Input,
        ):
            dpg.add_text('match metadata')
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
            dpg_set_value(
                self.ports(node_id).metadata.value_tag,
                summarize_metadata({
                    'format': compression_metadata.get('codec'),
                    'jpeg': compression_metadata.get('jpeg'),
                }),
            )
        return frame, result

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
        )
        return output, {
            'compression_metadata': compression_metadata,
            'metadata': {
                'format': compression_metadata.get('codec'),
                'jpeg': compression_metadata.get('jpeg'),
                'compression': compression_metadata,
            },
        }
