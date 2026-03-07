#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from abc import abstractmethod

import dearpygui.dearpygui as dpg
import numpy as np

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg, dpg_get_value, dpg_set_value


class DeclarativeImageProcessNodeBase(DpgNodeABC):
    """Common implementation for simple image process nodes."""

    _opencv_setting_dict = None

    parameters = []
    show_elapsed_time = True

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        tag_node_name = self._node_name(node_id)
        input_image_tag = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01')
        input_image_value_tag = self._value_tag(input_image_tag)
        output_image_tag = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01')
        output_image_value_tag = self._value_tag(output_image_tag)
        elapsed_tag = self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02')
        elapsed_value_tag = self._value_tag(elapsed_tag)

        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(black_image, small_window_w, small_window_h)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=output_image_value_tag,
                format=dpg.mvFormat_Float_rgb,
            )

        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            with dpg.node_attribute(
                tag=input_image_tag,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=input_image_value_tag,
                    default_value='Input BGR image',
                )

            with dpg.node_attribute(
                tag=output_image_tag,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(output_image_value_tag)

            for parameter in self.parameters:
                self._add_parameter_ui(tag_node_name, parameter, small_window_w, callback)

            self.build_custom_ui(
                tag_node_name,
                node_id,
                small_window_w,
                callback,
            )

            if self.show_elapsed_time and use_pref_counter:
                with dpg.node_attribute(
                    tag=elapsed_tag,
                    attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=elapsed_value_tag,
                        default_value='elapsed time(ms)',
                    )

        self.on_node_added(tag_node_name)
        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        del node_result_dict

        tag_node_name = self._node_name(node_id)
        output_image_value_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01')
        )
        elapsed_value_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02')
        )

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        connection_info_src = ''
        self._sync_linked_parameters(connection_list)

        for source_tag, _, connection_type in self._iter_connections(connection_list):
            if connection_type == self.TYPE_IMAGE:
                connection_info_src = self._extract_source_node_key(source_tag)

        frame = node_image_dict.get(connection_info_src, None)
        parameter_values = self._get_parameter_values(tag_node_name)
        parameter_values = self.normalize_parameter_values(tag_node_name, parameter_values)

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

        return frame, result

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        setting_dict = {
            'ver': self._ver,
            'pos': dpg.get_item_pos(tag_node_name),
        }

        for parameter in self.parameters:
            parameter_value_tag = self._value_tag(
                self._port_tag(tag_node_name, parameter['type'], parameter['port'])
            )
            setting_dict[parameter_value_tag] = dpg_get_value(parameter_value_tag)

        setting_dict.update(self.get_custom_setting_dict(tag_node_name, node_id))

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)

        for parameter in self.parameters:
            parameter_value_tag = self._value_tag(
                self._port_tag(tag_node_name, parameter['type'], parameter['port'])
            )
            if parameter_value_tag not in setting_dict:
                continue
            value = self._cast_parameter_value(parameter, setting_dict[parameter_value_tag])
            dpg_set_value(parameter_value_tag, value)

        self.set_custom_setting_dict(tag_node_name, node_id, setting_dict)

        self.on_settings_applied(tag_node_name)

    def build_custom_ui(self, tag_node_name, node_id, width, callback):
        del tag_node_name, node_id, width, callback

    def get_custom_setting_dict(self, tag_node_name, node_id):
        del tag_node_name, node_id
        return {}

    def set_custom_setting_dict(self, tag_node_name, node_id, setting_dict):
        del tag_node_name, node_id, setting_dict

    def on_node_added(self, tag_node_name):
        del tag_node_name

    def on_settings_applied(self, tag_node_name):
        del tag_node_name

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        del tag_node_name
        return parameter_values

    @abstractmethod
    def process(self, frame, **parameter_values):
        pass

    def _sync_linked_parameters(self, connection_list):
        for source_tag, destination_tag, connection_type in self._iter_connections(connection_list):
            destination_port = self._extract_port_name(destination_tag)
            parameter = self._find_parameter(connection_type, destination_port)
            if parameter is None:
                continue

            source_value = dpg_get_value(self._value_tag(source_tag))
            if source_value is None:
                continue

            value = self._cast_parameter_value(parameter, source_value)
            value = self._clamp_parameter_value(parameter, value)
            dpg_set_value(self._value_tag(destination_tag), value)

    def _get_parameter_values(self, tag_node_name):
        values = {}
        for parameter in self.parameters:
            parameter_value_tag = self._value_tag(
                self._port_tag(tag_node_name, parameter['type'], parameter['port'])
            )
            value = dpg_get_value(parameter_value_tag)
            value = self._cast_parameter_value(parameter, value)
            value = self._clamp_parameter_value(parameter, value)
            values[parameter['name']] = value
        return values

    def _add_parameter_ui(self, tag_node_name, parameter, width, callback):
        port_tag = self._port_tag(tag_node_name, parameter['type'], parameter['port'])
        value_tag = self._value_tag(port_tag)

        with dpg.node_attribute(
            tag=port_tag,
            attribute_type=parameter.get('attribute_type', dpg.mvNode_Attr_Input),
        ):
            if parameter['widget'] == 'slider_int':
                dpg.add_slider_int(
                    tag=value_tag,
                    label=parameter['label'],
                    width=width - 80,
                    default_value=parameter['default'],
                    min_value=parameter['min'],
                    max_value=parameter['max'],
                    callback=None,
                )
            elif parameter['widget'] == 'slider_float':
                dpg.add_slider_float(
                    tag=value_tag,
                    label=parameter['label'],
                    width=width - 80,
                    default_value=parameter['default'],
                    min_value=parameter['min'],
                    max_value=parameter['max'],
                    callback=None,
                )
            elif parameter['widget'] == 'input_int':
                dpg.add_input_int(
                    tag=value_tag,
                    label=parameter['label'],
                    width=width - 64,
                    default_value=parameter['default'],
                    callback=callback,
                )
            elif parameter['widget'] == 'checkbox':
                dpg.add_checkbox(
                    tag=value_tag,
                    label=parameter['label'],
                    default_value=parameter['default'],
                    callback=parameter.get('callback', None),
                    user_data=parameter.get('user_data', None),
                )
            elif parameter['widget'] == 'combo':
                items = parameter['items']
                dpg.add_combo(
                    items,
                    default_value=parameter.get('default', items[0]),
                    width=width - 40,
                    label=parameter['label'],
                    tag=value_tag,
                )

    def _find_parameter(self, value_type, port_name):
        for parameter in self.parameters:
            if parameter['type'] == value_type and parameter['port'] == port_name:
                return parameter
        return None

    def _cast_parameter_value(self, parameter, value):
        if value is None:
            return parameter.get('default', value)

        cast = parameter.get('cast', None)
        try:
            if cast is int:
                return int(value)
            if cast is float:
                value = float(value)
                precision = parameter.get('precision', None)
                if precision is not None:
                    value = round(value, precision)
                return value
            if cast is bool:
                return bool(value)
        except (TypeError, ValueError):
            return parameter.get('default', value)

        return value

    def _clamp_parameter_value(self, parameter, value):
        min_value = parameter.get('min', None)
        max_value = parameter.get('max', None)

        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value

    def _iter_connections(self, connection_list):
        for connection_info in connection_list:
            if not isinstance(connection_info, (list, tuple)) or len(connection_info) < 2:
                continue

            source_tag, destination_tag = connection_info[0], connection_info[1]
            if not isinstance(source_tag, str) or not isinstance(destination_tag, str):
                continue

            source_tokens = source_tag.split(':')
            if len(source_tokens) < 4:
                continue

            connection_type = source_tokens[2]
            yield source_tag, destination_tag, connection_type

    def _extract_source_node_key(self, source_tag):
        source_tokens = source_tag.split(':')
        if len(source_tokens) < 2:
            return ''
        return ':'.join(source_tokens[:2])

    def _extract_port_name(self, tag):
        tag_tokens = tag.split(':')
        if len(tag_tokens) < 4:
            return ''
        return tag_tokens[3]

    def _node_name(self, node_id):
        return str(node_id) + ':' + self.node_tag

    def _port_tag(self, node_name, value_type, port_name):
        return f'{node_name}:{value_type}:{port_name}'

    def _value_tag(self, port_tag):
        return f'{port_tag}Value'
