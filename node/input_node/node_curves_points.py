#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input node that emits editable Curves control points."""

import dearpygui.dearpygui as dpg

from node.curves_points_ui import CURVE_CHANNELS, CurvesPointsEditorMixin
from node.node_abc import DpgNodeBase
from node.port_model import OutputPort, PortDataType, PortSpecs
from node_editor.util import dpg_set_value


class Node(CurvesPointsEditorMixin, DpgNodeBase):
    _ver = '0.0.5'

    node_label = 'Curves Points'
    node_tag = 'CurvesPoints'

    port_specs = PortSpecs(
        value=OutputPort(PortDataType.CURVE_POINTS),
    )

    def _set_output_value(self, node_id, curve_set):
        value_tag = self.ports(node_id).value.value_tag
        dpg_set_value(value_tag, self._serialize_curve_set(curve_set))

    def _on_points_changed(self, node_id, curve_set):
        self._set_output_value(node_id, curve_set)

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
                tag=output_port.dpg_tag,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_text(
                    tag=output_port.value_tag,
                    default_value=self._serialize_curve_set(self._default_curve_set()),
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
        self._redraw_line(node_id)
        curve_set = self._curve_set(node_id)
        self._set_output_value(node_id, curve_set)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = dpg.get_item_pos(tag_node_name)
        setting_dict[output_value_tag] = self._serialize_curve_set(curve_set)
        setting_dict['curves'] = curve_set
        setting_dict['active_channel'] = self._active_channel(node_id)

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        output_value_tag = self.ports(node_id).value.value_tag
        raw_value = setting_dict.get(output_value_tag, setting_dict.get('curves'))
        active_channel = setting_dict.get('active_channel', 'White')
        if active_channel in CURVE_CHANNELS:
            self._set_active_channel(node_id, active_channel)
        self._reset_points_from_setting(node_id, raw_value)
        self._set_output_value(node_id, self._curve_set(node_id))
