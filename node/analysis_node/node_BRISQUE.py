#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg


def image_process(image, model_path, range_path):
    score = cv2.quality.QualityBRISQUE_compute(image, model_path, range_path)
    return score


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'BRISQUE'
    node_tag = 'BRISQUE'

    _current_path = os.path.dirname(os.path.abspath(__file__))
    _model_path = _current_path + '/BRISQUE/brisque_model_live.yml'
    _range_path = _current_path + '/BRISQUE/brisque_range_live.yml'

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
        tag_node_output02_name = self._port_tag(tag_node_name, self.TYPE_TIME_MS,
                                                'Output02')
        tag_node_output02_value_name = self._value_tag(tag_node_output02_name)

        tag_node_score_name = self._port_tag(tag_node_name, self.TYPE_TEXT,
                                            'Score')
        tag_node_score_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Score'))

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
            # Result
            with dpg.node_attribute(
                    tag=tag_node_score_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_text(
                    tag=tag_node_score_value_name,
                    default_value='BRISQUE Score',
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
        output_value01_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))
        output_value02_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02'))

        tag_node_score_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Score'))

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Check connection info
        connection_info_src = ''
        for source_tag, destination_tag, connection_type in self._iter_connections(
                connection_list):
            if connection_type == self.TYPE_FLOAT:
                # Get connection tag
                source_value_tag = self._value_tag(source_tag)
                destination_value_tag = self._value_tag(destination_tag)
                # Update value
                input_value = round(float(dpg_get_value(source_value_tag)), 3)
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_value_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # Get source node name for image (with ID)
                connection_info_src = self._extract_source_node_key(source_tag)

        # Get image
        frame = node_image_dict.get(connection_info_src, None)

        # Measurement start
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        result = {}
        if frame is not None:
            score = image_process(frame, self._model_path, self._range_path)

            dpg_set_value(tag_node_score_value_name, str('%.2f' % score[0]))
            result['score'] = score

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

        return frame, result

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
