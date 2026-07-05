#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves adjustment node for Image Processing Node Editor."""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.curves_points_ui import CurvesPointsEditorMixin
from node.port_model import PortDataType
from node_editor.util import dpg_set_value


def image_process(image, points):
    # Builds LUT from curves points
    # Assumes points are pre-sorted and between 0 and 255
    xs, ys = zip(*points)
    table = np.interp(np.arange(256), xs, ys).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image


class Node(CurvesPointsEditorMixin, DeclarativeImageProcessNodeBase):
    """Curves adjustment node."""

    _ver = '0.0.3'

    parameters = [
        {
            'name': 'points',
            'type': PortDataType.CURVE_POINTS,
            'port': 'Input02',
            'label': 'Points',
            'widget': 'custom',
            'default': '[[0, 0], [255, 255]]',
        },
    ]

    node_label = 'Curves'
    node_tag = 'Curves'

    _min_val = 0
    _max_val = 255
    _delete_hit_radius = 6

    def _set_points_parameter_value(self, node_id, points):
        try:
            value_tag = self._parameter_port_ref(
                node_id, self.parameters[0]
            ).value_tag
        except (KeyError, IndexError):
            return
        serialized_points = self._serialize_points(points)
        try:
            dpg_set_value(value_tag, serialized_points)
        except Exception:
            return
        self._last_parameter_values[value_tag] = serialized_points

    def _on_points_changed(self, node_id, points):
        self._set_points_parameter_value(node_id, points)

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
        if not isinstance(value, list):
            return False
        self._reset_points_from_setting(node_id, self._parse_points(value))
        return True

    def build_custom_ui(self, tag_node_name, node_id, width, callback):
        del tag_node_name, width, callback
        self.build_curve_points_file_dialogs(node_id)

        points_port = self._parameter_port_ref(node_id, self.parameters[0])
        with dpg.node_attribute(
            tag=points_port.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Input,
        ):
            dpg.add_input_text(
                tag=points_port.value_tag,
                default_value=self._serialize_points(self._default_points()),
                show=False,
            )
            self.build_curve_points_editor(node_id)

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        node_id = int(str(tag_node_name).split(':', maxsplit=1)[0])
        raw_points = parameter_values.get('points')
        current_points = self._get_drag_points(node_id)
        if raw_points is None:
            parameter_values['points'] = current_points
            return parameter_values

        linked_points = self._parse_points(raw_points)
        if linked_points != current_points:
            self._reset_points_from_setting(node_id, linked_points)
        parameter_values['points'] = linked_points
        return parameter_values

    def process(self, frame, **parameter_values):
        frame = image_process(frame, parameter_values['points'])
        return frame, None

    def get_custom_setting_dict(self, tag_node_name, node_id):
        del tag_node_name
        points = self._get_drag_points(node_id)
        self._set_points_parameter_value(node_id, points)
        return {'points': points}

    def set_custom_setting_dict(self, tag_node_name, node_id, setting_dict):
        del tag_node_name
        points = self._parse_points(setting_dict.get('points', []))
        self._reset_points_from_setting(node_id, points)
        self._set_points_parameter_value(node_id, points)
