#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import os

import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg

from node.deep_learning_node.pose_estimation.movenet.movenet import (
    MoveNetSinglePoseLightning,
    MoveNetSinglePoseThunder,
    MoveNetMultiPoseLightning,
)
from node.deep_learning_node.pose_estimation.mediapipe_hands.mediapipe_hands import (
    MediaPipeHandsComplexity0,
    MediaPipeHandsComplexity1,
)
from node.deep_learning_node.pose_estimation.mediapipe_pose.mediapipe_pose import (
    MediaPipePoseComplexity0,
    MediaPipePoseComplexity1,
    MediaPipePoseComplexity2,
)

from node.draw_node.draw_util.draw_util import draw_pose_estimation_info


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Pose Estimation'
    node_tag = 'PoseEstimation'

    _min_val = 0.0
    _max_val = 1.0

    _opencv_setting_dict = None

    # Model settings
    _model_class = {
        'MoveNet(SinglePose Lightning)': MoveNetSinglePoseLightning,
        'MoveNet(SinglePose Thunder)': MoveNetSinglePoseThunder,
        'MoveNet(MulitPose Lightning)': MoveNetMultiPoseLightning,
        'MediaPipe Hands(Complexity0)': MediaPipeHandsComplexity0,
        'MediaPipe Hands(Complexity1)': MediaPipeHandsComplexity1,
        'MediaPipe Pose(Complexity0)': MediaPipePoseComplexity0,
        'MediaPipe Pose(Complexity1)': MediaPipePoseComplexity1,
        'MediaPipe Pose(Complexity2)': MediaPipePoseComplexity2,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/pose_estimation/'
    _model_path_setting = {
        'MoveNet(SinglePose Lightning)':
        _model_base_path + 'movenet/model/movenet_singlepose_lightning_4.onnx',
        'MoveNet(SinglePose Thunder)':
        _model_base_path + 'movenet/model/movenet_singlepose_thunder_4.onnx',
        'MoveNet(MulitPose Lightning)':
        _model_base_path + 'movenet/model/movenet_multipose_lightning_1.onnx',
        'MediaPipe Hands(Complexity0)': None,
        'MediaPipe Hands(Complexity1)': None,
        'MediaPipe Pose(Complexity0)': None,
        'MediaPipe Pose(Complexity1)': None,
        'MediaPipe Pose(Complexity2)': None,
    }

    _model_instance = {}

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
        tag_node_input01_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01')
        tag_node_input01_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Input01'))
        tag_node_input02_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02')
        tag_node_input02_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_node_input03_name = self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03')
        tag_node_input03_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03'))
        tag_node_output01_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))
        tag_node_output02_name = self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02')
        tag_node_output02_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02'))

        tag_provider_select_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Provider')
        tag_provider_select_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Provider'))

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']

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
            if use_gpu:
	            # CPU/GPU switch
	            with dpg.node_attribute(
	                    tag=tag_provider_select_name,
	                    attribute_type=dpg.mvNode_Attr_Static,
	            ):
	                dpg.add_radio_button(
	                    ("CPU", "GPU"),
	                    tag=tag_provider_select_value_name,
	                    default_value='CPU',
	                    horizontal=True,
	                )
            # Score threshold
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_float(
                    tag=tag_node_input03_value_name,
                    label="score",
                    width=small_window_w - 80,
                    default_value=0.3,
                    min_value=self._min_val,
                    max_value=self._max_val,
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
        input_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        input_value03_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03'))
        output_value01_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))
        output_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02'))

        tag_provider_select_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Provider'))

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']

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

        # Score threshold
        score_th = round(float(dpg_get_value(input_value03_tag)), 3)

        # Get CPU/GPU selection state
        provider = 'CPU'
        if use_gpu:
        	provider = dpg_get_value(tag_provider_select_value_name)

        # Get model info
        model_name = dpg_get_value(input_value02_tag)
        model_path = self._model_path_setting[model_name]
        model_class = self._model_class[model_name]

        model_name_with_provider = model_name + '_' + provider

        # Get model
        if frame is not None:
            if model_name_with_provider not in self._model_instance:
                if provider == 'CPU':
                    providers = ['CPUExecutionProvider']
                    self._model_instance[
                        model_name_with_provider] = model_class(
                            model_path,
                            providers=providers,
                        )
                else:
                    self._model_instance[
                        model_name_with_provider] = model_class(model_path)

        # Measurement start
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        result = {}
        if frame is not None:
            results_list = self._model_instance[model_name_with_provider](
                frame)
            result['model_name'] = model_name
            result['score_th'] = score_th
            result['results_list'] = results_list

        # Measurement end
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # Draw
        if frame is not None:
            debug_frame = copy.deepcopy(frame)
            debug_frame = draw_pose_estimation_info(
                model_name,
                debug_frame,
                results_list,
                score_th,
            )
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
        input_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        input_value03_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03'))

        # Selected model
        model_name = dpg_get_value(input_value02_tag)
        # Score threshold
        score_th = round(float(dpg_get_value(input_value03_tag)), 3)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name
        setting_dict[input_value03_tag] = score_th

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        input_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        input_value03_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03'))

        model_name = setting_dict[input_value02_tag]
        score_th = setting_dict[input_value03_tag]

        dpg_set_value(input_value02_tag, model_name)
        dpg_set_value(input_value03_tag, score_th)
