#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from abc import abstractmethod

import dearpygui.dearpygui as dpg
import numpy as np

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg, dpg_get_value, dpg_set_value


class DeclarativeImageProcessNodeBase(DpgNodeBase):
    """Common implementation for simple image process nodes."""

    _opencv_setting_dict = None

    parameters = []
    show_elapsed_time = True
    _cache_toggle_setting_key = '__cache_enabled__'
    _result_image_toggle_setting_key = '__result_image_enabled__'
    _result_large_image_toggle_setting_key = '__result_large_image_enabled__'
    _cache_enabled_by_node = {}
    _result_image_enabled_by_node = {}
    _result_large_image_enabled_by_node = {}
    _ui_callback = None
    _suspend_parameter_event_tags = set()
    _last_parameter_values = {}

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        tag_node_name = self._node_name(node_id)
        input_image_tag = self._node_port_tag(node_id, self.TYPE_IMAGE, 'Input01')
        output_image_tag = self._node_port_tag(node_id, self.TYPE_IMAGE, 'Output01')
        output_image_value_tag = self._node_value_tag(
            node_id, self.TYPE_IMAGE, 'Output01'
        )
        elapsed_tag = self._node_port_tag(node_id, self.TYPE_TIME_MS, 'Output02')
        elapsed_value_tag = self._node_value_tag(
            node_id, self.TYPE_TIME_MS, 'Output02'
        )
        cache_toggle_value_tag = self._node_value_tag(
            node_id, self.TYPE_TEXT, 'Cache'
        )
        result_image_toggle_value_tag = self._node_value_tag(
            node_id, self.TYPE_TEXT, 'ResultImage'
        )
        result_large_image_toggle_value_tag = self._node_value_tag(
            node_id, self.TYPE_TEXT, 'ResultImageLarge'
        )

        self._opencv_setting_dict = opencv_setting_dict
        self._ui_callback = callback
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
            self.add_editor_toolbar(
                node_id,
                callback=callback,
                build_extra_controls=lambda: self._add_image_toolbar_controls(
                    node_id,
                    result_image_toggle_value_tag,
                    result_large_image_toggle_value_tag,
                    cache_toggle_value_tag,
                ),
            )

            with dpg.node_attribute(
                tag=input_image_tag,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                pass

            with dpg.node_attribute(
                tag=output_image_tag,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                pass

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

    def _add_image_toolbar_controls(
        self,
        node_id,
        result_image_toggle_value_tag,
        result_large_image_toggle_value_tag,
        cache_toggle_value_tag,
    ):
        dpg.add_checkbox(
            label='R',
            tag=result_image_toggle_value_tag,
            default_value=False,
            callback=self._on_result_image_toggle,
            user_data=node_id,
        )
        self._result_image_enabled_by_node[str(node_id)] = False
        dpg.add_checkbox(
            label='RL',
            tag=result_large_image_toggle_value_tag,
            default_value=False,
            callback=self._on_result_large_image_toggle,
            user_data=node_id,
        )
        self._result_large_image_enabled_by_node[str(node_id)] = False
        dpg.add_checkbox(
            label='Cache',
            tag=cache_toggle_value_tag,
            default_value=True,
            callback=self._on_cache_toggle,
            user_data=node_id,
        )
        self._cache_enabled_by_node[str(node_id)] = True

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        del node_result_dict

        tag_node_name = self._node_name(node_id)
        output_image_value_tag = self._node_value_tag(
            node_id, self.TYPE_IMAGE, 'Output01'
        )
        elapsed_value_tag = self._node_value_tag(
            node_id, self.TYPE_TIME_MS, 'Output02'
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

    def render_cached_output(self, node_id, frame):
        if frame is None:
            return

        start_time = time.perf_counter()
        output_image_value_tag = self._node_value_tag(
            node_id, self.TYPE_IMAGE, 'Output01'
        )
        elapsed_value_tag = self._node_value_tag(
            node_id, self.TYPE_TIME_MS, 'Output02'
        )
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        texture = convert_cv_to_dpg(frame, small_window_w, small_window_h)
        dpg_set_value(output_image_value_tag, texture)
        if self.show_elapsed_time and use_pref_counter:
            elapsed_time = int((time.perf_counter() - start_time) * 1000)
            dpg_set_value(elapsed_value_tag, str(elapsed_time).zfill(4) + 'ms')

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        setting_dict = {
            'ver': self._ver,
            'pos': dpg.get_item_pos(tag_node_name),
        }

        for parameter in self.parameters:
            parameter_value_tag = self._port_value_tag(
                tag_node_name, parameter['type'], parameter['port']
            )
            setting_dict[parameter_value_tag] = dpg_get_value(parameter_value_tag)

        setting_dict.update(self.get_custom_setting_dict(tag_node_name, node_id))
        setting_dict[self._cache_toggle_setting_key] = bool(
            self._cache_enabled_by_node.get(str(node_id), True)
        )
        setting_dict[self._result_image_toggle_setting_key] = bool(
            self._result_image_enabled_by_node.get(str(node_id), False)
        )
        setting_dict[self._result_large_image_toggle_setting_key] = bool(
            self._result_large_image_enabled_by_node.get(str(node_id), False)
        )

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)

        for parameter in self.parameters:
            parameter_value_tag = self._port_value_tag(
                tag_node_name, parameter['type'], parameter['port']
            )
            if parameter_value_tag not in setting_dict:
                continue
            value = self._cast_parameter_value(parameter, setting_dict[parameter_value_tag])
            dpg_set_value(parameter_value_tag, value)
            self._last_parameter_values[parameter_value_tag] = value

        self.set_custom_setting_dict(tag_node_name, node_id, setting_dict)

        cache_toggle_value_tag = self._node_value_tag(
            node_id, self.TYPE_TEXT, 'Cache'
        )
        result_image_toggle_value_tag = self._node_value_tag(
            node_id, self.TYPE_TEXT, 'ResultImage'
        )
        result_large_image_toggle_value_tag = self._node_value_tag(
            node_id, self.TYPE_TEXT, 'ResultImageLarge'
        )
        cache_enabled = bool(setting_dict.get(self._cache_toggle_setting_key, True))
        result_image_enabled = bool(
            setting_dict.get(self._result_image_toggle_setting_key, False)
        )
        result_large_image_enabled = bool(
            setting_dict.get(self._result_large_image_toggle_setting_key, False)
        )
        self._cache_enabled_by_node[str(node_id)] = cache_enabled
        self._result_image_enabled_by_node[str(node_id)] = result_image_enabled
        self._result_large_image_enabled_by_node[str(node_id)] = result_large_image_enabled
        dpg_set_value(cache_toggle_value_tag, cache_enabled)
        dpg_set_value(result_image_toggle_value_tag, result_image_enabled)
        dpg_set_value(result_large_image_toggle_value_tag, result_large_image_enabled)
        self._last_parameter_values[cache_toggle_value_tag] = cache_enabled
        self._last_parameter_values[result_image_toggle_value_tag] = result_image_enabled
        self._last_parameter_values[result_large_image_toggle_value_tag] = result_large_image_enabled
        self._emit_result_node_toggle(node_id, 'ResultImage', result_image_enabled)
        self._emit_result_node_toggle(
            node_id, 'ResultImageLarge', result_large_image_enabled
        )

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

    def _on_cache_toggle(self, sender, app_data, user_data):
        node_id_name = self._node_name(user_data)
        before_value = self._cache_enabled_by_node.get(str(user_data), True)
        self._last_parameter_values[sender] = bool(app_data)
        self._cache_enabled_by_node[str(user_data)] = bool(app_data)
        if self._ui_callback is not None:
            self._ui_callback(
                'parameter_changed',
                {
                    'node_id_name': node_id_name,
                    'port_tag': self._port_tag(node_id_name, self.TYPE_TEXT, 'Cache'),
                    'value_tag': sender,
                    'before_value': bool(before_value),
                    'after_value': bool(app_data),
                },
            )

    def _on_result_image_toggle(self, sender, app_data, user_data):
        node_id_name = self._node_name(user_data)
        before_value = self._result_image_enabled_by_node.get(str(user_data), False)
        self._last_parameter_values[sender] = bool(app_data)
        enabled = bool(app_data)
        self._result_image_enabled_by_node[str(user_data)] = enabled
        if self._ui_callback is not None:
            self._ui_callback(
                'parameter_changed',
                {
                    'node_id_name': node_id_name,
                    'port_tag': self._port_tag(node_id_name, self.TYPE_TEXT, 'ResultImage'),
                    'value_tag': sender,
                    'before_value': bool(before_value),
                    'after_value': enabled,
                },
            )
        self._emit_result_node_toggle(user_data, 'ResultImage', enabled)

    def _on_result_large_image_toggle(self, sender, app_data, user_data):
        node_id_name = self._node_name(user_data)
        before_value = self._result_large_image_enabled_by_node.get(str(user_data), False)
        self._last_parameter_values[sender] = bool(app_data)
        enabled = bool(app_data)
        self._result_large_image_enabled_by_node[str(user_data)] = enabled
        if self._ui_callback is not None:
            self._ui_callback(
                'parameter_changed',
                {
                    'node_id_name': node_id_name,
                    'port_tag': self._port_tag(node_id_name, self.TYPE_TEXT, 'ResultImageLarge'),
                    'value_tag': sender,
                    'before_value': bool(before_value),
                    'after_value': enabled,
                },
            )
        self._emit_result_node_toggle(user_data, 'ResultImageLarge', enabled)

    def _emit_result_node_toggle(self, node_id, result_node_tag, enabled):
        if self._ui_callback is None:
            return
        self._ui_callback(
            'toggle_result_node',
            {
                'source_node_id_name': self._node_name(node_id),
                'result_node_tag': result_node_tag,
                'enabled': bool(enabled),
            },
        )

    def on_editor_parameter_value_applied(self, value_tag, value):
        if not isinstance(value_tag, str) or not value_tag.endswith('Value'):
            return False
        port_name = self._extract_port_name(value_tag[:-5])
        node_id = self._extract_node_id(value_tag)
        if not node_id:
            return False

        if port_name == 'Cache':
            self._cache_enabled_by_node[str(node_id)] = bool(value)
            return True

        if port_name == 'ResultImage':
            enabled = bool(value)
            self._result_image_enabled_by_node[str(node_id)] = enabled
            self._emit_result_node_toggle(node_id, 'ResultImage', enabled)
            return True

        if port_name == 'ResultImageLarge':
            enabled = bool(value)
            self._result_large_image_enabled_by_node[str(node_id)] = enabled
            self._emit_result_node_toggle(node_id, 'ResultImageLarge', enabled)
            return True

        return False

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
            parameter_value_tag = self._port_value_tag(
                tag_node_name, parameter['type'], parameter['port']
            )
            value = dpg_get_value(parameter_value_tag)
            value = self._cast_parameter_value(parameter, value)
            value = self._clamp_parameter_value(parameter, value)
            values[parameter['name']] = value
        return values

    def _add_parameter_ui(self, tag_node_name, parameter, width, callback):
        port_tag = self._port_tag(tag_node_name, parameter['type'], parameter['port'])
        value_tag = self._port_value_tag(
            tag_node_name, parameter['type'], parameter['port']
        )
        callback_payload = {
            'node_id_name': tag_node_name,
            'port_tag': port_tag,
            'value_tag': value_tag,
            'parameter': parameter,
            'callback': parameter.get('callback', callback),
        }
        self._last_parameter_values[value_tag] = parameter.get('default', None)

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
                    callback=self._on_parameter_widget_changed,
                    user_data=callback_payload,
                )
            elif parameter['widget'] == 'slider_float':
                dpg.add_slider_float(
                    tag=value_tag,
                    label=parameter['label'],
                    width=width - 80,
                    default_value=parameter['default'],
                    min_value=parameter['min'],
                    max_value=parameter['max'],
                    callback=self._on_parameter_widget_changed,
                    user_data=callback_payload,
                )
            elif parameter['widget'] == 'input_int':
                dpg.add_input_int(
                    tag=value_tag,
                    label=parameter['label'],
                    width=width - 64,
                    default_value=parameter['default'],
                    callback=self._on_parameter_widget_changed,
                    user_data=callback_payload,
                )
            elif parameter['widget'] == 'checkbox':
                dpg.add_checkbox(
                    tag=value_tag,
                    label=parameter['label'],
                    default_value=parameter['default'],
                    callback=self._on_parameter_widget_changed,
                    user_data=callback_payload,
                )
            elif parameter['widget'] == 'combo':
                items = parameter['items']
                dpg.add_combo(
                    items,
                    default_value=parameter.get('default', items[0]),
                    width=width - 40,
                    label=parameter['label'],
                    tag=value_tag,
                    callback=self._on_parameter_widget_changed,
                    user_data=callback_payload,
                )

    def _on_parameter_widget_changed(self, sender, app_data, user_data):
        if sender in self._suspend_parameter_event_tags:
            return
        parameter_callback = None
        if isinstance(user_data, dict):
            parameter_callback = user_data.get('callback', None)
        if callable(parameter_callback):
            parameter_callback(sender, app_data)
        if callable(self._ui_callback) and isinstance(user_data, dict):
            value_tag = str(user_data.get('value_tag'))
            before_value = self._last_parameter_values.get(value_tag, app_data)
            self._last_parameter_values[value_tag] = app_data
            self._ui_callback(
                'parameter_changed',
                {
                    'node_id_name': user_data.get('node_id_name'),
                    'port_tag': user_data.get('port_tag'),
                    'value_tag': sender,
                    'before_value': before_value,
                    'after_value': app_data,
                },
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
