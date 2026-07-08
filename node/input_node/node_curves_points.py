#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input node that emits editable Curves control points."""

import dearpygui.dearpygui as dpg

from node.curves_points_ui import CurvesPointsEditorMixin
from node.node_abc import DpgNodeBase
from node.port_model import OutputPort, PortDataType, PortSpecs
from node.process_node.node_curves import _CHANNELS, _normalize_channel
from node_editor.util import dpg_get_value, dpg_set_value


class Node(CurvesPointsEditorMixin, DpgNodeBase):
    _ver = '0.0.4'

    node_label = 'Curves Points'
    node_tag = 'CurvesPoints'

    port_specs = PortSpecs(
        value=OutputPort(PortDataType.CURVE_POINTS),
    )

    def _channel_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Channel')

    def _channel_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Channel')

    def _get_channel(self, node_id):
        return _normalize_channel(dpg_get_value(self._channel_value_tag(node_id)))

    def _serialize_output(self, node_id, points):
        return self._serialize_points({
            'channel': self._get_channel(node_id),
            'points': points,
        })

    def _set_output_value(self, node_id, points):
        value_tag = self.ports(node_id).value.value_tag
        dpg_set_value(value_tag, self._serialize_output(node_id, points))

    def _on_points_changed(self, node_id, points):
        self._set_output_value(node_id, points)

    def _on_channel_changed(self, sender, app_data, user_data):
        del sender, app_data
        self._set_output_value(user_data, self._get_drag_points(user_data))

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        del callback
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        output_port = ports.value

        self._opencv_setting_dict = opencv_setting_dict
        self.build_curve_points_file_dialogs(node_id)

        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            with dpg.node_attribute(
                tag=self._channel_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(_CHANNELS),
                    label='Channel',
                    tag=self._channel_value_tag(node_id),
                    default_value='White',
                    width=160,
                    callback=self._on_channel_changed,
                    user_data=node_id,
                )
            with dpg.node_attribute(
                tag=output_port.dpg_tag,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_text(
                    tag=output_port.value_tag,
                    default_value=self._serialize_output(node_id, self._default_points()),
                    show=False,
                )
                self.build_curve_points_editor(node_id)

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        del node_id, connection_list, node_image_dict, node_result_dict
        return None, None

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        output_value_tag = self.ports(node_id).value.value_tag
        points = self._get_drag_points(node_id)
        self._set_output_value(node_id, points)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = dpg.get_item_pos(tag_node_name)
        setting_dict[output_value_tag] = self._serialize_output(node_id, points)
        setting_dict[self._channel_value_tag(node_id)] = self._get_channel(node_id)

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        output_value_tag = self.ports(node_id).value.value_tag
        raw_value = setting_dict.get(output_value_tag)
        points = self._parse_points(raw_value)
        channel = 'White'
        if isinstance(raw_value, dict):
            channel = raw_value.get('channel', channel)
        channel = setting_dict.get(self._channel_value_tag(node_id), channel)
        dpg_set_value(self._channel_value_tag(node_id), _normalize_channel(channel))
        self._reset_points_from_setting(node_id, points)
        self._set_output_value(node_id, points)
