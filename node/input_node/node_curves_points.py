#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input node that emits serialized Curves control points."""

import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node.port_model import OutputPort, PortDataType, PortSpecs


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Curves Points'
    node_tag = 'CurvesPoints'

    port_specs = PortSpecs(
        value=OutputPort(PortDataType.TEXT),
    )

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        output_port = ports.value

        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']

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
                    label='Points',
                    width=small_window_w - 54,
                    default_value='[[0, 0], [255, 255]]',
                    callback=callback,
                )

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

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = dpg.get_item_pos(tag_node_name)
        setting_dict[output_value_tag] = dpg_get_value(output_value_tag)

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        output_value_tag = self.ports(node_id).value.value_tag
        dpg_set_value(output_value_tag, setting_dict[output_value_tag])
