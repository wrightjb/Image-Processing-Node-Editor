#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import copy

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info


def create_concat_image(frame_dict, slot_num):
    def hconcat_top(images):
        max_h = max(image.shape[0] for image in images)
        padded = []
        for image in images:
            h, w = image.shape[:2]
            canvas = np.zeros((max_h, w, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = image
            padded.append(canvas)
        return cv2.hconcat(padded)

    def vconcat_left(images):
        max_w = max(image.shape[1] for image in images)
        padded = []
        for image in images:
            h, w = image.shape[:2]
            canvas = np.zeros((h, max_w, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = image
            padded.append(canvas)
        return cv2.vconcat(padded)

    if slot_num == 1:
        frame = frame_dict[0]
        display_frame = copy.deepcopy(frame)
    elif slot_num == 2:
        frame = hconcat_top([frame_dict[0], frame_dict[1]])
        display_frame = copy.deepcopy(frame)
    elif slot_num == 3 or slot_num == 4:
        hconcat_image01 = hconcat_top([frame_dict[0], frame_dict[1]])
        hconcat_image02 = hconcat_top([frame_dict[2], frame_dict[3]])
        frame = vconcat_left([hconcat_image01, hconcat_image02])
        display_frame = copy.deepcopy(frame)
    elif slot_num == 5 or slot_num == 6:
        hconcat_image01 = hconcat_top([frame_dict[0], frame_dict[1], frame_dict[2]])
        hconcat_image02 = hconcat_top([frame_dict[3], frame_dict[4], frame_dict[5]])
        frame = vconcat_left([hconcat_image01, hconcat_image02])
        display_frame = copy.deepcopy(frame)
    elif slot_num == 7 or slot_num == 8 or slot_num == 9:
        hconcat_image01 = hconcat_top([frame_dict[0], frame_dict[1], frame_dict[2]])
        hconcat_image02 = hconcat_top([frame_dict[3], frame_dict[4], frame_dict[5]])
        hconcat_image03 = hconcat_top([frame_dict[6], frame_dict[7], frame_dict[8]])
        frame = vconcat_left([hconcat_image01, hconcat_image02, hconcat_image03])
        display_frame = copy.deepcopy(frame)

    return frame, display_frame


def resize_with_aspect_and_pad(frame, resize_width, resize_height):
    image_h, image_w = frame.shape[:2]
    if image_w <= 0 or image_h <= 0:
        return np.zeros((resize_height, resize_width, 3), dtype=np.uint8)

    scale = min(resize_width / float(image_w), resize_height / float(image_h))
    width = max(1, int(round(image_w * scale)))
    height = max(1, int(round(image_h * scale)))
    resized = cv2.resize(frame, (width, height))

    canvas = np.zeros((resize_height, resize_width, 3), dtype=np.uint8)
    offset_x = (resize_width - width) // 2
    offset_y = (resize_height - height) // 2
    canvas[offset_y:offset_y + height, offset_x:offset_x + width] = resized
    return canvas


def create_image_dict(
    slot_num,
    connection_info_src_dict,
    node_image_dict,
    node_result_dict,
    image_node_name,
    resize_width,
    resize_height,
    draw_info_on_result,
):
    frame_exist_flag = False

    # Black image for initialization
    black_image = np.zeros((resize_height, resize_width, 3)).astype(np.uint8)

    frame_dict = {}
    for index in range(slot_num - 1, -1, -1):
        node_id_name = connection_info_src_dict.get(index, None)
        frame = copy.deepcopy(node_image_dict.get(node_id_name, None))
        if frame is not None:
            if draw_info_on_result:
                node_result = node_result_dict[node_id_name]
                image_node_name = node_id_name.split(':')[1]
                frame = draw_info(image_node_name, node_result, frame)
            resize_frame = frame
            frame_dict[slot_num - index - 1] = copy.deepcopy(resize_frame)

            frame_exist_flag = True
        else:
            frame_dict[slot_num - index - 1] = copy.deepcopy(black_image)

    display_num_list = [1, 2, 4, 4, 6, 6, 9, 9, 9]
    for index in range(display_num_list[slot_num - 1]):
        if frame_dict.get(index, None) is None:
            frame_dict[index] = copy.deepcopy(black_image)

    if not frame_exist_flag:
        frame_dict = None

    return frame_dict


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Image Concat'
    node_tag = 'ImageConcat'

    _opencv_setting_dict = None

    _max_slot_number = 9
    _slot_id = {}

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
        self._value_history = {}

        # Tag names
        tag_node_name = self._node_name(node_id)
        tag_node_input00_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input00')
        tag_node_input01_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01')
        tag_node_input01_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01'))
        tag_node_output01_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']

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

        # Dictionary to store slot numbers
        if tag_node_name not in self._slot_id:
            self._slot_id[tag_node_name] = 1

        # Node
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # Image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(
                    tag_node_output01_value_name,
                )
            # Add slot button
            with dpg.node_attribute(
                    tag=tag_node_input00_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label='Add Slot',
                    width=int(small_window_w / 3),
                    callback=self._add_slot,
                    user_data=tag_node_name,
                )
            # Slot
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
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
        output_value01_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        resize_width = self._opencv_setting_dict['result_width']
        resize_height = self._opencv_setting_dict['result_height']
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']

        # Get source node name for image (with ID)
        connection_info_src = ''
        connection_info_src_dict = {}
        for source_tag, destination_tag, connection_type in self._iter_connections(
                connection_list):
            # Get slot number from tag name
            slot_number = re.sub(r'\D', '', destination_tag.split(':')[-1])
            if slot_number == '':
                continue
            slot_number = int(slot_number) - 1

            if connection_type == self.TYPE_IMAGE:
                # Get source node name for image (with ID)
                connection_info_src = self._extract_source_node_key(source_tag)
                node_name = connection_info_src.split(':')[1]

                connection_info_src_dict[slot_number] = connection_info_src

        slot_num = self._slot_id[tag_node_name]

        # Get image
        frame_dict = {}
        if len(connection_info_src_dict) > 0:
            frame_dict = create_image_dict(
                slot_num,
                connection_info_src_dict,
                node_image_dict,
                node_result_dict,
                node_name,
                resize_width,
                resize_height,
                draw_info_on_result,
            )

        # Generate concatenated image
        frame = None
        display_frame = None
        if len(connection_info_src_dict) > 0 and frame_dict is not None:
            frame, display_frame = create_concat_image(frame_dict, slot_num)

        # Draw
        if display_frame is not None:
            preview_frame = resize_with_aspect_and_pad(
                display_frame,
                small_window_w,
                small_window_h,
            )
            texture = convert_cv_to_dpg(
                preview_frame,
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
        setting_dict['slot_id'] = self._slot_id[tag_node_name]

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)

        slot_number = int(setting_dict['slot_id'])
        for _ in range(slot_number - 1):
            self._add_slot(None, None, tag_node_name)

    def _add_slot(self, sender, data, user_data):
        tag_node_name = user_data

        if self._max_slot_number > self._slot_id[tag_node_name]:
            self._slot_id[tag_node_name] += 1

            # Generate insertion destination tag name
            before_tag = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input')
            before_tag += str(self._slot_id[tag_node_name] - 1).zfill(2)

            # Generate added slot tag
            tag_node_inputXX_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input')
            tag_node_inputXX_name += str(self._slot_id[tag_node_name]).zfill(2)

            tag_node_inputXX_value_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input')
            tag_node_inputXX_value_name += str(
                self._slot_id[tag_node_name]).zfill(2) + 'Value'

            # Add slot
            with dpg.node_attribute(
                    tag=tag_node_inputXX_name,
                    attribute_type=dpg.mvNode_Attr_Input,
                    parent=tag_node_name,
                    before=before_tag,
            ):
                dpg.add_text(
                    tag=tag_node_inputXX_value_name,
                    default_value='Input BGR image',
                )
