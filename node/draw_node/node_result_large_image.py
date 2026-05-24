#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Result Image(Large)'
    node_tag = 'ResultImageLarge'

    _ratio = 2

    _opencv_setting_dict = None

    def __init__(self):
        self._viewport_size_dict = {}
        self._texture_size_dict = {}

    def _compute_display_size(self, frame, max_width, max_height):
        image_h, image_w = frame.shape[:2]
        max_width = max(1, int(max_width))
        max_height = max(1, int(max_height))
        if image_w <= 0 or image_h <= 0:
            return max_width, max_height

        scale = min(max_width / float(image_w), max_height / float(image_h))
        display_w = max(1, int(round(image_w * scale)))
        display_h = max(1, int(round(image_h * scale)))
        return display_w, display_h

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
        tag_node_input01_viewport_name = self._value_tag(
            self._port_tag(tag_node_input01_name, self.TYPE_TEXT, 'Viewport'))

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        default_max_size = self._opencv_setting_dict['result_width'] * self._ratio
        self._viewport_size_dict[node_id] = (default_max_size, default_max_size)
        initial_texture_size = (default_max_size, default_max_size)
        self._texture_size_dict[node_id] = initial_texture_size

        # Black image for initialization
        small_window_w, small_window_h = initial_texture_size
        black_image = np.zeros((small_window_h, small_window_w, 3))
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
                tag=tag_node_input01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # Node
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # Image
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                with dpg.child_window(
                        tag=tag_node_input01_viewport_name,
                        width=default_max_size,
                        height=default_max_size,
                        autosize_x=False,
                        autosize_y=False,
                ):
                    dpg.add_image(
                        tag_node_input01_value_name,
                        width=default_max_size,
                        height=default_max_size,
                    )

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        # Tag names
        tag_node_name = self._node_name(node_id)
        input_value01_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01'))
        viewport_tag = self._value_tag(
            self._port_tag(
                self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01'),
                self.TYPE_TEXT, 'Viewport'))

        # OpenCV settings
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']

        # Get source node name for image (with ID)
        node_name = ''
        connection_info_src = ''
        for source_tag, _, connection_type in self._iter_connections(
                connection_list):
            if connection_type != self.TYPE_IMAGE:
                continue

            connection_info_src = self._extract_source_node_key(source_tag)
            node_name = self._extract_source_node_key(source_tag).split(':')[1]

        # Get image
        frame = node_image_dict.get(connection_info_src, None)

        # Draw
        if frame is not None:
            if draw_info_on_result and connection_info_src != '':
                node_result = node_result_dict[connection_info_src]
                frame = draw_info(node_name, node_result, frame)

            viewport_w, viewport_h = self._viewport_size_dict.get(
                node_id, (self._opencv_setting_dict['result_width'] * self._ratio,
                          self._opencv_setting_dict['result_width'] * self._ratio))
            if dpg.does_item_exist(viewport_tag):
                viewport_w = max(1, int(dpg.get_item_width(viewport_tag)))
                viewport_h = max(1, int(dpg.get_item_height(viewport_tag)))
                self._viewport_size_dict[node_id] = (viewport_w, viewport_h)

            small_window_w, small_window_h = self._compute_display_size(
                frame,
                viewport_w,
                viewport_h,
            )

            previous_texture_size = self._texture_size_dict.get(node_id)
            if previous_texture_size != (small_window_w, small_window_h):
                if dpg.does_item_exist(input_value01_tag):
                    dpg.delete_item(input_value01_tag)
                with dpg.texture_registry(show=False):
                    dpg.add_raw_texture(
                        small_window_w,
                        small_window_h,
                        np.zeros((small_window_w * small_window_h * 3,), dtype='f'),
                        tag=input_value01_tag,
                        format=dpg.mvFormat_Float_rgb,
                    )
                dpg.configure_item(
                    input_value01_tag,
                    width=small_window_w,
                    height=small_window_h,
                )
                self._texture_size_dict[node_id] = (small_window_w,
                                                    small_window_h)

            texture = convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(input_value01_tag, texture)

        return frame, None

    def close(self, node_id):
        self._viewport_size_dict.pop(node_id, None)
        self._texture_size_dict.pop(node_id, None)

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
