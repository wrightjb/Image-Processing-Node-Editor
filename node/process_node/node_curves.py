#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves adjustment node for Image Processing Node Editor."""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.curve_interpolation import points_to_lut as _shared_points_to_lut
from node.curves_points_ui import CURVE_CHANNELS, CurvesPointsEditorMixin
from node.port_model import OutputPort, PortDataType
from node_editor.util import dpg_get_value, dpg_set_value

_BGR_INDEX_BY_CHANNEL = {
    'Blue': 0,
    'Green': 1,
    'Red': 2,
}


def _points_to_lut(points, interpolation='linear'):
    return _shared_points_to_lut(points, interpolation=interpolation)


def _apply_lut(image, table):
    try:
        return cv2.LUT(image, table)
    except AttributeError:
        return table[np.asarray(image)]


def image_process(image, curves, channel=None, interpolation='linear'):
    """Apply White first, then per-channel RGB curves."""
    helper = CurvesPointsEditorMixin()
    if channel in CURVE_CHANNELS:
        curve_set = helper._default_curve_set()
        curve_set[channel] = helper._parse_points(curves)
    else:
        curve_set = helper._normalize_curve_set(curves)
    white_lut = _points_to_lut(curve_set['White'], interpolation=interpolation)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return _apply_lut(image, white_lut)

    output = image.copy()
    color_channels = output[:, :, :3]
    for channel, channel_index in _BGR_INDEX_BY_CHANNEL.items():
        channel_lut = _points_to_lut(curve_set[channel], interpolation=interpolation)
        composed_lut = channel_lut[white_lut]
        color_channels[:, :, channel_index] = _apply_lut(
            image[:, :, channel_index],
            composed_lut,
        )
    return output


class Node(CurvesPointsEditorMixin, DeclarativeImageProcessNodeBase):
    """Curves adjustment node."""

    _ver = '0.0.5'

    parameters = [
        {
            'name': 'curves',
            'type': PortDataType.CURVE_POINTS,
            'port': 'Input02',
            'label': 'Curves',
            'widget': 'custom',
            'default': '{"curves": {}}',
        },
        {
            'name': 'interpolation',
            'type': PortDataType.TEXT,
            'port': 'Input03',
            'label': 'Interpolation',
            'widget': 'combo',
            'items': ['linear', 'spline'],
            'default': 'linear',
        },
    ]

    node_label = 'Curves'
    node_tag = 'Curves'

    _min_val = 0
    _max_val = 255
    _delete_hit_radius = 6


    def _curve_interpolation(self, node_id):
        try:
            value_tag = self._parameter_port_ref(
                node_id,
                self.parameters[1],
            ).value_tag
        except (KeyError, IndexError):
            return 'linear'
        return 'spline' if dpg_get_value(value_tag) == 'spline' else 'linear'

    def _curves_output_port_ref(self, node_id):
        try:
            return self.ports(node_id).curves_output
        except (AttributeError, KeyError):
            return self.create_port(
                node_id,
                'curves_output',
                OutputPort(PortDataType.CURVE_POINTS, index=3),
            )

    def _set_curves_output_value(self, node_id, curve_set):
        try:
            value_tag = self._curves_output_port_ref(node_id).value_tag
        except (AttributeError, KeyError):
            return
        if str(node_id) not in self._curve_editor_built_by_node:
            return
        try:
            dpg_set_value(value_tag, self._serialize_curve_set(curve_set))
        except Exception:
            return

    def _set_curves_parameter_value(self, node_id, curve_set):
        try:
            value_tag = self._parameter_port_ref(
                node_id, self.parameters[0]
            ).value_tag
        except (KeyError, IndexError):
            return
        serialized_curves = self._serialize_curve_set(curve_set)
        if str(node_id) not in self._curve_editor_built_by_node:
            self._last_parameter_values[value_tag] = serialized_curves
            return
        try:
            dpg_set_value(value_tag, serialized_curves)
        except Exception:
            return
        self._last_parameter_values[value_tag] = serialized_curves

    def _on_points_changed(self, node_id, curve_set):
        self._set_curves_parameter_value(node_id, curve_set)
        self._set_curves_output_value(node_id, curve_set)

    def _emit_points_changed(self, node_id, before_points, after_points, coalesce=False):
        if self._ui_callback is None:
            return
        if before_points == after_points:
            return
        node_id_name = self._node_name(node_id)
        value_tag = self._parameter_port_ref(node_id, self.parameters[0]).value_tag
        self._ui_callback(
            'parameter_changed',
            {
                'node_id_name': node_id_name,
                'port_tag': self._parameter_port_ref(node_id, self.parameters[0]).dpg_tag,
                'value_tag': value_tag,
                'before_value': before_points,
                'after_value': after_points,
                'coalesce': bool(coalesce),
            },
        )

    def apply_history_value(self, value_tag, value):
        if not isinstance(value_tag, str):
            return False
        if not (
            value_tag.endswith(':CurvesPointsValue')
            or value_tag.endswith(':Input02Value')
        ):
            return False
        node_id_text = value_tag.split(':', maxsplit=1)[0]
        try:
            node_id = int(node_id_text)
        except ValueError:
            return False
        self._reset_points_from_setting(node_id, value)
        return True


    def on_editor_parameter_value_applied(self, value_tag, value):
        if self.apply_history_value(value_tag, value):
            return True
        return super().on_editor_parameter_value_applied(value_tag, value)

    def build_custom_ui(self, tag_node_name, node_id, width, callback):
        del tag_node_name, width, callback
        self.build_curve_points_file_dialogs(node_id)

        curves_port = self._parameter_port_ref(node_id, self.parameters[0])
        curves_output_port = self._curves_output_port_ref(node_id)
        self.begin_curve_points_editor(node_id)

        with dpg.node_attribute(
            tag=f'{self._node_name(node_id)}:CurvesEditorHeader',
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            dpg.add_input_text(
                tag=curves_port.value_tag,
                default_value=self._serialize_curve_set(self._default_curve_set()),
                show=False,
            )
            dpg.add_input_text(
                tag=curves_output_port.value_tag,
                default_value=self._serialize_curve_set(self._default_curve_set()),
                show=False,
            )
            self.build_curve_points_channel_selector(node_id)

        with dpg.node_attribute(
            tag=curves_port.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Input,
        ):
            pass

        with dpg.node_attribute(
            tag=curves_output_port.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Output,
        ):
            pass

        with dpg.node_attribute(
            tag=f'{self._node_name(node_id)}:CurvesEditorPlot',
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            self.build_curve_points_plot_controls(node_id)

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        node_id = int(str(tag_node_name).split(':', maxsplit=1)[0])
        raw_curves = parameter_values.get('curves')
        if raw_curves is None:
            curve_set = self._curve_set(node_id)
        else:
            curve_set = self._normalize_curve_set(raw_curves)
            if curve_set != self._curve_set(node_id):
                self._reset_points_from_setting(node_id, curve_set)
        parameter_values['curves'] = curve_set
        return parameter_values

    def process(self, frame, **parameter_values):
        interpolation = parameter_values.get('interpolation', 'linear')
        if interpolation == 'spline':
            frame = image_process(
                frame,
                parameter_values['curves'],
                interpolation=interpolation,
            )
        else:
            frame = image_process(frame, parameter_values['curves'])
        return frame, None

    def get_custom_setting_dict(self, tag_node_name, node_id):
        del tag_node_name
        self._redraw_line(node_id)
        curve_set = self._curve_set(node_id)
        self._set_curves_parameter_value(node_id, curve_set)
        self._set_curves_output_value(node_id, curve_set)
        return {'curves': curve_set, 'active_channel': self._active_channel(node_id)}

    def set_custom_setting_dict(self, tag_node_name, node_id, setting_dict):
        del tag_node_name
        active_channel = setting_dict.get('active_channel', 'White')
        if active_channel in CURVE_CHANNELS:
            self._set_active_channel(node_id, active_channel)
        curves = setting_dict.get('curves', setting_dict.get('points', []))
        self._reset_points_from_setting(node_id, curves)
        self._set_curves_parameter_value(node_id, self._curve_set(node_id))
        self._set_curves_output_value(node_id, self._curve_set(node_id))
