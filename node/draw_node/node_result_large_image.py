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
        self._canvas_size_dict = {}

    def _compute_display_size(self, frame, max_size):
        image_h, image_w = frame.shape[:2]
        max_size = max(1, int(max_size))
        if image_w <= 0 or image_h <= 0:
            return max_size, max_size

        scale = max_size / float(max(image_w, image_h))
        display_w = max(1, int(round(image_w * scale)))
        display_h = max(1, int(round(image_h * scale)))
        return display_w, display_h

    def _fit_with_letterbox(self, frame, canvas_size):
        display_w, display_h = self._compute_display_size(frame, canvas_size)
        resized = cv2.resize(frame, (display_w, display_h))
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        offset_x = (canvas_size - display_w) // 2
        offset_y = (canvas_size - display_h) // 2
        canvas[offset_y:offset_y + display_h, offset_x:offset_x + display_w] = resized
        return canvas

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

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        canvas_size = self._opencv_setting_dict['result_width'] * self._ratio
        self._canvas_size_dict[node_id] = canvas_size

        # Black image for initialization
        black_image = np.zeros((canvas_size, canvas_size, 3))
        black_texture = convert_cv_to_dpg(
            black_image,
            canvas_size,
            canvas_size,
        )

        # Register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                canvas_size,
                canvas_size,
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

            canvas_size = self._canvas_size_dict.get(
                node_id,
                self._opencv_setting_dict['result_width'] * self._ratio,
            )
            frame = self._fit_with_letterbox(frame, canvas_size)

            texture = convert_cv_to_dpg(
                frame,
                canvas_size,
                canvas_size,
            )
            dpg_set_value(input_value01_tag, texture)

        return frame, None

    def close(self, node_id):
        self._canvas_size_dict.pop(node_id, None)

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
