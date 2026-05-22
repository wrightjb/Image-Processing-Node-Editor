#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


def image_process(image):
    return image


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'ON/OFF Switch'
    node_tag = 'OnOffSwitch'

    _switch_on = 'ON'
    _switch_off = 'OFF'

    _opencv_setting_dict = None

    def __init__(self):
        pass

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        # Tag names
        tag_node_name = self._node_name(node_id)
        tag_node_input01_name = self._port_tag(tag_node_name, self.TYPE_IMAGE,
                                               'Input01')
        tag_node_input01_value_name = self._value_tag(tag_node_input01_name)
        tag_node_output01_name = self._port_tag(tag_node_name, self.TYPE_IMAGE,
                                                'Output01')
        tag_node_output01_value_name = self._value_tag(tag_node_output01_name)

        tag_switch_select_name = self._port_tag(tag_node_name, self.TYPE_TEXT,
                                             'Switch')
        tag_switch_select_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Switch'))

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = int(self._opencv_setting_dict['process_width'] / 2)
        small_window_h = int(self._opencv_setting_dict['process_height'] / 2)

        # Black image for initialization
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

        # Register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # Node
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # Input port
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )
            # Image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # ON/OFF switch
            with dpg.node_attribute(
                    tag=tag_switch_select_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_radio_button(
                    (self._switch_on, self._switch_off),
                    tag=tag_switch_select_value_name,
                    default_value=self._switch_on,
                    horizontal=True,
                )

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        tag_node_name = self._node_name(node_id)
        output_value01_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))

        tag_switch_select_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Switch'))

        small_window_w = int(self._opencv_setting_dict['process_width'] / 2)
        small_window_h = int(self._opencv_setting_dict['process_height'] / 2)

        # Get source node name for image (with ID)
        connection_info_src = ''
        for source_tag, _, connection_type in self._iter_connections(
                connection_list):
            if connection_type != self.TYPE_IMAGE:
                continue

            connection_info_src = self._extract_source_node_key(source_tag)

        # Get ON/OFF selection state
        switch_status = dpg_get_value(tag_switch_select_value_name)

        # Get image
        frame = None
        if switch_status == self._switch_on:
            frame = node_image_dict.get(connection_info_src, None)

            if frame is not None:
                frame = image_process(frame)

        # Draw
        if frame is not None:
            texture = convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, None

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
