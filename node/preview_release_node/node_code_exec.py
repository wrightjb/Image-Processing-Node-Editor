#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg


def image_process(input_image, input_value, code):
    image_width = input_image.shape[1]
    font_scale = image_width / 1000
    thickness = 1 + int((image_width + 500) / 1000)

    output_image = copy.deepcopy(input_image)

    exec_local = {
        'input_image': input_image,
        'input_value': input_value,
        'output_image': output_image,
    }

    try:
        exec(code, globals(), exec_local)
        output_image = exec_local['output_image']

        if len(output_image.shape) == 2:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        output_image = copy.deepcopy(input_image)

        cv2.putText(
            output_image,
            text=str(e),
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness + 8,
        )
        cv2.putText(
            output_image,
            text=str(e),
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 255),
            thickness=thickness,
        )

    return output_image


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Exec Python Code'
    node_tag = 'ExecPythonCode'

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
        tag_node_input01_value_name = self._value_tag(tag_node_input01_name)
        tag_node_input02_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_node_output01_name_port = self.output_port(node_id, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_name = tag_node_output01_name_port.dpg_tag
        tag_node_output01_value_name = self._value_tag(tag_node_output01_name)
        tag_node_output02_name_port = self.output_port(node_id, self.TYPE_TIME_MS, 'Output02')
        tag_node_output02_name = tag_node_output02_name_port.dpg_tag
        tag_node_output02_value_name = self._value_tag(tag_node_output02_name)

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = int(self._opencv_setting_dict['process_width'] * 2.5)
        small_window_h = int(self._opencv_setting_dict['process_height'] * 2.5)
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
            # Code
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, ):
                dpg.add_input_text(
                    tag=tag_node_input02_value_name,
                    multiline=True,
                    default_value=
                    'output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)',
                    width=small_window_w,
                    height=int((small_window_h / 3) * 2),
                    tab_input=True,
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
        input_value02_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        output_value01_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))
        output_value02_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02'))

        small_window_w = int(self._opencv_setting_dict['process_width'] * 2.5)
        small_window_h = int(self._opencv_setting_dict['process_height'] * 2.5)
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Get source node name for image (with ID)
        src_node_result = None
        connection_info_src = ''
        for connection_info, source_tag, _, connection_type in self._iter_connection_infos(
                connection_list):
            if connection_type == self.TYPE_INT:
                connection_info_src = self._connection_source_node_key(connection_info, source_tag)
            if connection_type == self.TYPE_IMAGE:
                connection_info_src = self._connection_source_node_key(connection_info, source_tag)
                src_node_result = node_result_dict.get(connection_info_src,
                                                       None)

        # Get image
        frame = node_image_dict.get(connection_info_src, None)

        # Get code
        code = dpg_get_value(input_value02_tag)

        # Measurement start
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        # Execute code
        if frame is not None:
            frame = image_process(frame, src_node_result, code)

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
        tag_node_input02_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))

        pos = dpg.get_item_pos(tag_node_name)
        code = dpg_get_value(tag_node_input02_value_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_input02_value_name] = code

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        tag_node_input02_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))

        code = setting_dict[tag_node_input02_value_name]

        dpg_set_value(tag_node_input02_value_name, code)
