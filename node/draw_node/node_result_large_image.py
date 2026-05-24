#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

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
        self._max_size_dict = {}
        self._texture_size_dict = {}

    def _compute_display_size(self, frame, max_size):
        image_h, image_w = frame.shape[:2]
        max_size = max(1, int(max_size))
        long_edge = max(image_w, image_h)
        if long_edge <= 0:
            return max_size, max_size

        scale = max_size / float(long_edge)
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
        tag_node_max_size_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'MaxSize'))

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        default_max_size = self._opencv_setting_dict['result_width'] * self._ratio
        self._max_size_dict[node_id] = default_max_size
        small_window_w = default_max_size
        small_window_h = default_max_size
        self._texture_size_dict[node_id] = (small_window_w, small_window_h)

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
                dpg.add_image(tag_node_input01_value_name)
                dpg.add_input_int(
                    tag=tag_node_max_size_name,
                    label='Max Size',
                    default_value=default_max_size,
                    min_value=1,
                    min_clamped=True,
                    width=120,
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
        max_size_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'MaxSize'))

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
            max_size = dpg_get_value(max_size_tag)
            if max_size is None:
                max_size = self._max_size_dict.get(node_id, 1)
            max_size = max(1, int(max_size))
            self._max_size_dict[node_id] = max_size

            if draw_info_on_result and connection_info_src != '':
                node_result = node_result_dict[connection_info_src]
                frame = draw_info(node_name, node_result, frame)

            small_window_w, small_window_h = self._compute_display_size(
                frame,
                max_size,
            )

            previous_texture_size = self._texture_size_dict.get(node_id)
            if previous_texture_size != (small_window_w, small_window_h):
                with dpg.texture_registry(show=False):
                    dpg.add_raw_texture(
                        small_window_w,
                        small_window_h,
                        np.zeros((small_window_w * small_window_h * 3,), dtype='f'),
                        tag=input_value01_tag,
                        format=dpg.mvFormat_Float_rgb,
                    )
                dpg.configure_item(
                    self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01'),
                    width=small_window_w,
                    height=small_window_h,
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
        self._max_size_dict.pop(node_id, None)
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
