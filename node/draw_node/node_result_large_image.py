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
        self._display_size_dict = {}
        self._texture_tag_dict = {}

    def _compute_display_size(self, frame, max_size):
        image_h, image_w = frame.shape[:2]
        max_size = max(1, int(max_size))
        if image_w <= 0 or image_h <= 0:
            return max_size, max_size

        scale = max_size / float(max(image_w, image_h))
        display_w = max(1, int(round(image_w * scale)))
        display_h = max(1, int(round(image_h * scale)))
        return display_w, display_h

    def _build_texture_tag(self, node_id):
        return f'{self.node_tag}:{node_id}:Input01:Texture'

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
        tag_node_input01_image_name = self._value_tag(tag_node_input01_name)
        texture_tag = self._build_texture_tag(node_id)

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        max_size = self._opencv_setting_dict['result_width'] * self._ratio
        self._display_size_dict[node_id] = (max_size, max_size)
        self._texture_tag_dict[node_id] = texture_tag

        # Black image for initialization
        black_image = np.zeros((max_size, max_size, 3))
        black_texture = convert_cv_to_dpg(
            black_image,
            max_size,
            max_size,
        )

        # Register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                max_size,
                max_size,
                black_texture,
                tag=texture_tag,
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
                dpg.add_image(
                    texture_tag,
                    tag=tag_node_input01_image_name,
                    width=max_size,
                    height=max_size,
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
        input_image_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01'))
        texture_tag = self._texture_tag_dict.get(
            node_id, self._build_texture_tag(node_id))

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

            max_size = self._opencv_setting_dict['result_width'] * self._ratio
            display_w, display_h = self._compute_display_size(frame, max_size)

            previous_size = self._display_size_dict.get(node_id)
            if previous_size != (display_w, display_h):
                if dpg.does_item_exist(texture_tag):
                    try:
                        dpg.delete_item(texture_tag)
                    except (SystemError, RuntimeError):
                        pass
                empty_texture = np.zeros((display_w * display_h * 3,),
                                         dtype='f')
                with dpg.texture_registry(show=False):
                    dpg.add_raw_texture(
                        display_w,
                        display_h,
                        empty_texture,
                        tag=texture_tag,
                        format=dpg.mvFormat_Float_rgb,
                    )
                dpg.configure_item(
                    input_image_tag,
                    texture_tag=texture_tag,
                    width=display_w,
                    height=display_h,
                )
                self._display_size_dict[node_id] = (display_w, display_h)

            texture = convert_cv_to_dpg(
                frame,
                display_w,
                display_h,
            )
            dpg_set_value(texture_tag, texture)

        return frame, None

    def close(self, node_id):
        texture_tag = self._texture_tag_dict.pop(node_id, None)
        self._display_size_dict.pop(node_id, None)
        if texture_tag is not None and dpg.does_item_exist(texture_tag):
            try:
                dpg.delete_item(texture_tag)
            except (SystemError, RuntimeError):
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
