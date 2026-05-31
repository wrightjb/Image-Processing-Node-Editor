#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info


def image_process(
        image,
        text,
        elapsed_time_text,
        color=(0, 255, 0),
        thickness=2,
):
    pos_index = 0
    pos_list = [
        (15, 30),
        (15, 60),
    ]

    if text != '':
        image = cv2.putText(
            image,
            text,
            pos_list[pos_index],
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            thickness=thickness,
        )
        pos_index += 1
    if elapsed_time_text != '':
        image = cv2.putText(
            image,
            'Elapsed time: ' + elapsed_time_text,
            pos_list[pos_index],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            thickness=thickness,
        )
        pos_index += 1
    return image


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'PutText'
    node_tag = 'PutText'

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
        tag_node_input02_name_port = self.input_port(node_id, self.TYPE_TEXT, 'Input02')
        tag_node_input02_name = tag_node_input02_name_port.dpg_tag
        tag_node_input02_value_name = tag_node_input02_name_port.value_tag
        tag_node_input03_name_port = self.input_port(node_id, self.TYPE_TIME_MS, 'Input03')
        tag_node_input03_name = tag_node_input03_name_port.dpg_tag
        tag_node_input03_value_name = tag_node_input03_name_port.value_tag
        tag_node_output01_name_port = self.output_port(node_id, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_name = tag_node_output01_name_port.dpg_tag
        tag_node_output01_value_name = tag_node_output01_name_port.value_tag

        tag_color_edit_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'ColorEdit'))

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
            # Text input field
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        tag=tag_node_input02_value_name,
                        label='',
                        width=small_window_w - 30,
                    )
                    dpg.add_color_edit(
                        (0, 255, 0),
                        tag=tag_color_edit_value_name,
                        no_inputs=True,
                        no_alpha=True,
                    )
            # Processing time input
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input03_value_name,
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
        input_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        input_value03_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Input03'))
        output_value01_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))

        tag_color_edit_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'ColorEdit'))

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']

        # Check connection info
        node_name = ''
        connection_info_src = ''
        connect_elapsed_time_flag = False
        for (
                connection_info,
                source_tag,
                destination_tag,
                connection_type,
        ) in self._iter_connection_infos(connection_list):
            if connection_type == self.TYPE_TEXT:
                # Get connection tag
                source_value_tag = self._connection_value_tag(connection_info, 'source', source_tag)
                destination_value_tag = self._connection_value_tag(connection_info, 'destination', destination_tag)
                # Update value
                input_value = dpg_get_value(source_value_tag)
                dpg_set_value(destination_value_tag, input_value)
            if connection_type == self.TYPE_TIME_MS:
                # Get connection tag
                source_value_tag = self._connection_value_tag(connection_info, 'source', source_tag)
                destination_value_tag = self._connection_value_tag(connection_info, 'destination', destination_tag)
                # Update value
                input_value = dpg_get_value(source_value_tag)
                dpg_set_value(destination_value_tag, input_value)

                connect_elapsed_time_flag = True
            if connection_type == self.TYPE_IMAGE:
                # Get source node name for image (with ID)
                connection_info_src = self._connection_source_node_key(connection_info, source_tag)
                node_name = connection_info_src.split(':')[1]

        # Get image
        frame = node_image_dict.get(connection_info_src, None)
        if draw_info_on_result and connection_info_src != '':
            node_result = node_result_dict[connection_info_src]
            frame = draw_info(node_name, node_result, frame)

        # Text, color, and elapsed time
        text = dpg_get_value(input_value02_tag)
        color = dpg_get_value(tag_color_edit_value_name)[:3]
        color = (
            int(round(color[2], 0)),
            int(round(color[1], 0)),
            int(round(color[0], 0)),
        )
        elapsed_time_text = ''
        if connect_elapsed_time_flag:
            elapsed_time_text = dpg_get_value(input_value03_tag)

        if frame is not None:
            frame = image_process(frame, text, elapsed_time_text, color)

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
        input_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_color_edit_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'ColorEdit'))

        text = dpg_get_value(input_value02_tag)
        color = dpg_get_value(tag_color_edit_value_name)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = text
        setting_dict[tag_color_edit_value_name] = color

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        input_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_color_edit_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'ColorEdit'))

        text = setting_dict[input_value02_tag]
        color = setting_dict[tag_color_edit_value_name]

        dpg_set_value(input_value02_tag, text)
        dpg_set_value(tag_color_edit_value_name, color)
