#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import re
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info

def image_process(image1, image2, alpha_val, beta_val, gamma_val):
    image1_height, image1_width = image1.shape[:2]
    image2 = cv2.resize(image2, (image1_width, image1_height))
    image = cv2.addWeighted(image1, alpha_val, image2, beta_val, gamma_val)
    return image

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
            resize_frame = cv2.resize(frame, (resize_width, resize_height))
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


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Image Alpha Blend'
    node_tag = 'ImageAlphaBlend'
    _max_slot_number = 2
    _slot_id = {}
    _alpha_min = 0.0
    _alpha_max = 1.0
    _alpha_default = 1.0
    _beta_min = 0.0
    _beta_max = 1.0
    _beta_default = 0.3
    _gamma_min = 0
    _gamma_max = 255 
    _gamma_default = 0

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
        tag_node_input01_name_port = self.input_port(node_id, self.TYPE_IMAGE, 'Input01')
        tag_node_input01_name = tag_node_input01_name_port.dpg_tag
        tag_node_input01_value_name = tag_node_input01_name_port.value_tag
        tag_node_input02_name_port = self.input_port(node_id, self.TYPE_IMAGE, 'Input02')
        tag_node_input02_name = tag_node_input02_name_port.dpg_tag
        tag_node_input02_value_name = tag_node_input02_name_port.value_tag
        tag_node_input03_name_port = self.input_port(node_id, self.TYPE_FLOAT, 'Input03')
        tag_node_input03_name = tag_node_input03_name_port.dpg_tag
        tag_node_input03_value_name = tag_node_input03_name_port.value_tag
        tag_node_input04_name_port = self.input_port(node_id, self.TYPE_FLOAT, 'Input04')
        tag_node_input04_name = tag_node_input04_name_port.dpg_tag
        tag_node_input04_value_name = tag_node_input04_name_port.value_tag
        tag_node_input05_name_port = self.input_port(node_id, self.TYPE_INT, 'Input05')
        tag_node_input05_name = tag_node_input05_name_port.dpg_tag
        tag_node_input05_value_name = tag_node_input05_name_port.value_tag
        tag_node_output01_name_port = self.output_port(node_id, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_name = tag_node_output01_name_port.dpg_tag
        tag_node_output01_value_name = tag_node_output01_name_port.value_tag
        tag_node_output02_name_port = self.output_port(node_id, self.TYPE_TIME_MS, 'Output02')
        tag_node_output02_name = tag_node_output02_name_port.dpg_tag
        tag_node_output02_value_name = tag_node_output02_name_port.value_tag

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

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
            # Input port
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )
            # Input port
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input02_value_name,
                    default_value='Input BGR image',
                )
            # Image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # Hysteresis
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input03_value_name,
                    label="alpha val",
                    width=small_window_w - 80,
                    default_value=self._alpha_default,
                    min_value=self._alpha_min,
                    max_value=self._alpha_max,
                    callback=None,
                )
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input04_value_name,
                    label="beta val",
                    width=small_window_w - 80,
                    default_value=self._beta_default,
                    min_value=self._beta_min,
                    max_value=self._beta_max,
                    callback=None,
                )
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input05_value_name,
                    label="gamma val",
                    width=small_window_w - 80,
                    default_value=self._gamma_default,
                    min_value=self._gamma_min,
                    max_value=self._gamma_max,
                    callback=None,
                )
            # Processing time
            if use_pref_counter:
                with dpg.node_attribute(
                        tag=tag_node_output02_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=tag_node_output02_value_name,
                        default_value='elapsed time(ms)',
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
        input_value03_tag = self._node_value_tag(node_id, self.TYPE_FLOAT, 'Input03')
        input_value04_tag = self._node_value_tag(node_id, self.TYPE_FLOAT, 'Input04')
        input_value05_tag = self._node_value_tag(node_id, self.TYPE_INT, 'Input05')
        output_value01_tag = self._node_value_tag(node_id, self.TYPE_IMAGE, 'Output01')
        output_value02_tag = self._node_value_tag(node_id, self.TYPE_TIME_MS, 'Output02')

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']

        # Check connection info
        frame = None
        frame1 = None
        frame2 = None
        node_name_dict = {}
        connection_info_src = ''
        connection_info_src_dict = {}
        for (
                connection_info,
                source_tag,
                destination_tag,
                connection_type,
        ) in self._iter_connection_infos(connection_list):

            # Get slot number from tag name
            destination_port_name = self._connection_port_name(
                connection_info,
                destination_tag,
            )
            slot_number = re.sub(r'\D', '', destination_port_name)
            if slot_number == '':
                continue
            slot_number = int(slot_number) - 1
            connection_tag = destination_port_name
            if connection_type == self.TYPE_FLOAT:
                # Get connection tag
                source_value_tag = self._connection_value_tag(connection_info, 'source', source_tag)
                destination_value_tag = self._connection_value_tag(connection_info, 'destination', destination_tag)
                # Update value
                input_value = round(float(dpg_get_value(source_value_tag)), 3)
                if connection_tag == 'Input03':
                    input_value = max([self._alpha_min, input_value])
                    input_value = min([self._alpha_max, input_value])
                if connection_tag == 'Input04':
                    input_value = max([self._beta_min, input_value])
                    input_value = min([self._beta_max, input_value])
                dpg_set_value(destination_value_tag, input_value)
            if connection_type == self.TYPE_INT:
                # Get connection tag
                source_value_tag = self._connection_value_tag(connection_info, 'source', source_tag)
                destination_value_tag = self._connection_value_tag(connection_info, 'destination', destination_tag)
                # Update value
                input_value = int(dpg_get_value(source_value_tag))
                if connection_tag == 'Input05':
                    input_value = max([self._gamma_min, input_value])
                    input_value = min([self._gamma_max, input_value])
                dpg_set_value(destination_value_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # Get source node name for image (with ID)
                connection_info_src = self._connection_source_node_key(connection_info, source_tag)
                node_name = connection_info_src.split(':')[1]
                node_name_dict[slot_number] = node_name
                connection_info_src_dict[slot_number] = connection_info_src

        # Get image

        if len(connection_info_src_dict) == 1:
            connected_first_slot_no = (next(iter(connection_info_src_dict)))
            frame1 = node_image_dict.get(connection_info_src_dict[connected_first_slot_no])
            frame = frame1
        if len(connection_info_src_dict) == 2:
            frame1 = node_image_dict.get(connection_info_src_dict[0])  
            frame2 = node_image_dict.get(connection_info_src_dict[1])
            frame = frame1

        # Alpha blend
        alpha_val = float(dpg_get_value(input_value03_tag))
        beta_val = float(dpg_get_value(input_value04_tag))
        gamma_val = int(dpg_get_value(input_value05_tag))

        # Measurement start
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()
        
        if len(connection_info_src_dict) == 2:
            if frame1 is not None and frame2 is not None:
                frame = image_process(frame1, frame2, alpha_val, beta_val, gamma_val)

        # Measurement end
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

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
        input_value03_tag = self._node_value_tag(node_id, self.TYPE_INT, 'Input03')

        kernel_size = dpg_get_value(input_value03_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value03_tag] = kernel_size

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        input_value03_tag = self._node_value_tag(node_id, self.TYPE_INT, 'Input02')

        kernel_size = int(setting_dict[input_value03_tag])

        dpg_set_value(input_value03_tag, kernel_size)
