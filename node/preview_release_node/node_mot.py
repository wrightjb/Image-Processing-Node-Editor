#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time

import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg

from node.preview_release_node.mot.motpy.motpy import Motpy
# from node.preview_release_node.mot.bytetrack.mc_bytetrack import MultiClassByteTrack
# from node.preview_release_node.mot.norfair.mc_norfair import MultiClassNorfair

from node.draw_node.draw_util.draw_util import draw_multi_object_tracking_info


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'MOT(Preview Release Version)'
    node_tag = 'MultiObjectTracking'

    _opencv_setting_dict = None

    # Model settings
    _model_class = {
        'motpy': Motpy,
        # 'ByteTrack': MultiClassByteTrack,
        # 'Norfair': MultiClassNorfair,
    }

    _model_instance = {}
    _class_name_dict = None
    _track_id_dict = {}

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
        tag_node_input02_name = self._node_port_tag(node_id, self.TYPE_TEXT, 'Input02')
        tag_node_input02_value_name = self._node_value_tag(node_id, self.TYPE_TEXT, 'Input02')
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
                    default_value='Input Detection Node',
                )
            # Image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # Algorithm
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(self._model_class.keys()),
                    default_value=list(self._model_class.keys())[0],
                    width=small_window_w,
                    tag=tag_node_input02_value_name,
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
        input_value02_tag = self._node_value_tag(node_id, self.TYPE_TEXT, 'Input02')
        output_value01_tag = self._node_value_tag(node_id, self.TYPE_IMAGE, 'Output01')
        output_value02_tag = self._node_value_tag(node_id, self.TYPE_TIME_MS, 'Output02')

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Check connection info
        src_node_name = ''
        connection_info_src = ''
        for (
                connection_info,
                source_tag,
                destination_tag,
                connection_type,
        ) in self._iter_connection_infos(connection_list):
            if connection_type == self.TYPE_INT:
                # Get connection tag
                source_value_tag = self._connection_value_tag(connection_info, 'source', source_tag)
                destination_value_tag = self._connection_value_tag(connection_info, 'destination', destination_tag)
                # Update value
                input_value = int(dpg_get_value(source_value_tag))
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_value_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # Get source node name for image (with ID)
                connection_info_src = self._connection_source_node_key(connection_info, source_tag)
                src_node_name = connection_info_src.split(':')[1]

        # Get image
        frame = node_image_dict.get(connection_info_src, None)

        # Get model info
        model_name = dpg_get_value(input_value02_tag)
        model_class = self._model_class[model_name]

        model_name_with_provider = tag_node_name + ':' + model_name

        # Get model
        if frame is not None:
            if model_name_with_provider not in self._model_instance:
                # Translated from Japanese comment
                self._model_instance[model_name_with_provider] = model_class()

        # Measurement start
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        # Translated from Japanese comment
        result = {}
        if frame is not None:
            if src_node_name == 'ObjectDetection':
                # Get object detection info
                node_result = node_result_dict.get(connection_info_src, [])
                od_bboxes = node_result.get('bboxes', [])
                od_scores = node_result.get('scores', [])
                od_class_ids = node_result.get('class_ids', [])
                od_class_names = node_result.get('class_names', [])

                track_ids, t_bboxes, t_scores, t_class_ids = [], [], [], []
                track_ids, t_bboxes, t_scores, t_class_ids = self._model_instance[
                    model_name_with_provider](
                        frame,
                        od_bboxes,
                        od_scores,
                        od_class_ids,
                    )

                if node_id not in self._track_id_dict:
                    self._track_id_dict[node_id] = {}

                # Map tracking ID to serial number
                for track_id in track_ids:
                    if track_id not in self._track_id_dict[node_id]:
                        new_id = len(self._track_id_dict[node_id])
                        self._track_id_dict[node_id][track_id] = new_id

                result['track_ids'] = track_ids
                result['bboxes'] = t_bboxes
                result['scores'] = t_scores
                result['class_ids'] = t_class_ids
                result['class_names'] = od_class_names
                result['track_id_dict'] = self._track_id_dict[node_id]

            elif src_node_name == 'Classification':
                node_result = node_result_dict.get(connection_info_src, [])
                use_object_detection = node_result.get(
                    'use_object_detection',
                    False,
                )
                if use_object_detection:
                    # Get object detection info
                    od_bboxes = node_result.get('od_bboxes', [])
                    od_scores = node_result.get('class_scores', [])
                    od_class_ids = node_result.get('class_ids', [])
                    od_class_names = node_result.get('class_names', [])

                    track_ids, t_bboxes, t_scores, t_class_ids = self._model_instance[
                        model_name_with_provider](
                            frame,
                            od_bboxes,
                            od_scores,
                            od_class_ids,
                        )

                    if node_id not in self._track_id_dict:
                        self._track_id_dict[node_id] = {}

                    # Map tracking ID to serial number
                    for track_id in track_ids:
                        if track_id not in self._track_id_dict[node_id]:
                            new_id = len(self._track_id_dict[node_id])
                            self._track_id_dict[node_id][track_id] = new_id

                    result['track_ids'] = track_ids
                    result['bboxes'] = t_bboxes
                    result['scores'] = t_scores
                    result['class_ids'] = t_class_ids
                    result['class_names'] = od_class_names
                    result['track_id_dict'] = self._track_id_dict[node_id]

        # Measurement end
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # Draw
        if frame is not None:
            if src_node_name == 'ObjectDetection' or src_node_name == 'Classification':
                # Draw
                debug_frame = copy.deepcopy(frame)
                debug_frame = draw_multi_object_tracking_info(
                    debug_frame,
                    track_ids,
                    t_bboxes,
                    t_scores,
                    t_class_ids,
                    od_class_names,
                    self._track_id_dict[node_id],
                )
            else:
                debug_frame = np.zeros((small_window_w, small_window_h, 3))
            texture = convert_cv_to_dpg(
                debug_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, result

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        input_value02_tag = self._node_value_tag(node_id, self.TYPE_TEXT, 'Input02')

        # Selected model
        model_name = dpg_get_value(input_value02_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        input_value02_tag = self._node_value_tag(node_id, self.TYPE_TEXT, 'Input02')

        model_name = setting_dict[input_value02_tag]

        dpg_set_value(input_value02_tag, model_name)
