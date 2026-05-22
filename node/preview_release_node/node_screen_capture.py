#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import multiprocessing as mp

import cv2
import numpy as np
from PIL import ImageGrab
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


def screen_capture_process(image_queue, request):
    while True:
        pil_image = ImageGrab.grab(all_screens=True)
        cv_image = np.array(pil_image, dtype=np.uint8)
        frame = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        if image_queue.qsize() == 0:
            image_queue.put(frame)
        time.sleep(0.001)

        # Exit process when set to 0
        if request.value == 0:
            break


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Screen Capture'
    node_tag = 'ScreenCapture'

    _opencv_setting_dict = None

    _frame_count = {}

    _image_queue = None
    _request = None
    _process = None
    _prev_frame = None

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
        tag_node_output01_name = self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))
        tag_node_output02_name = self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02')
        tag_node_output02_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02'))

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
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
            # Camera image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
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

        self._frame_count[str(node_id)] = 0

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
        output_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02'))

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Create screen capture thread
        if self._process is None:
            self._image_queue = mp.Queue(maxsize=1)
            self._request = mp.Value('i', 1)
            self._process = mp.Process(
                target=screen_capture_process,
                args=(
                    self._image_queue,
                    self._request,
                ),
            )
            self._process.start()

        # Measurement start
        if use_pref_counter:
            start_time = time.perf_counter()

        # Get image
        frame = None
        if self._image_queue is not None:
            num = self._image_queue.qsize()
            if num > 0:
                frame = self._image_queue.get()
                self._prev_frame = copy.deepcopy(frame)
            else:
                frame = copy.deepcopy(self._prev_frame)

        # Measurement end
        if use_pref_counter:
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
        if self._request is not None:
            self._request.value = 0
            self._process.terminate()

            self._image_queue = None
            self._request = None
            self._process = None

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
