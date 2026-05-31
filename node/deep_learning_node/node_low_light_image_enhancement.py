#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os

import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg

from node.deep_learning_node.low_light_image_enhancement.TBEFN.tbefn import TBEFN
from node.deep_learning_node.low_light_image_enhancement.SCI.sci import SCI
from node.deep_learning_node.low_light_image_enhancement.AGLLNet.agllnet import AGLLNet


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Low-Light Image Enhancement'
    node_tag = 'LLIE'

    _opencv_setting_dict = None

    # Model settings
    _model_class = {
        'TBEFN(320x180)': TBEFN,
        'TBEFN(640x360)': TBEFN,
        'SCI(320x180)': SCI,
        'SCI(640x360)': SCI,
        'AGLLNet(256x256)': AGLLNet,
        'AGLLNet(512x384)': AGLLNet,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/low_light_image_enhancement/'
    _model_path_setting = {
        'TBEFN(320x180)':
        _model_base_path + 'TBEFN/saved_model_180x320/model_float32.onnx',
        'TBEFN(640x360)':
        _model_base_path + 'TBEFN/saved_model_360x640/model_float32.onnx',
        'SCI(320x180)':
        _model_base_path + 'SCI/sci_180x320/sci_180x320.onnx',
        'SCI(640x360)':
        _model_base_path + 'SCI/sci_360x640/sci_360x640.onnx',
        'AGLLNet(256x256)':
        _model_base_path + 'AGLLNet/saved_model_256x256/model_float32.onnx',
        'AGLLNet(512x384)':
        _model_base_path + 'AGLLNet/saved_model_384x512/model_float32.onnx',
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

        tag_provider_select_name = self._node_port_tag(node_id, self.TYPE_TEXT, 'Provider')
        tag_provider_select_value_name = self._node_value_tag(node_id, self.TYPE_IMAGE, 'Provider')

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

        tag_provider_select_value_name = self._node_value_tag(node_id, self.TYPE_IMAGE, 'Provider')

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']

        # Check connection info
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

        # Get image
        frame = node_image_dict.get(connection_info_src, None)

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

        if frame is not None:
            frame = self._model_instance[model_name_with_provider](frame)

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
