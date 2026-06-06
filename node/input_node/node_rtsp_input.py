#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import multiprocessing as mp

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node.port_model import OutputPort, PortDataType, PortSpecs
from node_editor.util import convert_cv_to_dpg


def receive_image_process(rtsp_url, image_queue, request):
    rtsp_capture = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = rtsp_capture.read()

        if ret:
            if image_queue.qsize() == 0:
                image_queue.put(frame)
            time.sleep(0.001)
        else:
            # If acquisition fails, wait 1 second and reconnect
            time.sleep(1)
            rtsp_capture.release()
            rtsp_capture = cv2.VideoCapture(rtsp_url)

        # Exit process when set to 0
        if request.value == 0:
            rtsp_capture.release()
            break


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'RTSP'
    node_tag = 'RTSPInput'

    _opencv_setting_dict = None
    _start_label = 'Start'
    _stop_label = 'Stop'

    _rtsp_capture = {}

    _image_queue = {}
    _request = {}
    _process = {}

    port_specs = PortSpecs(
        image=OutputPort(PortDataType.IMAGE),
        elapsed=OutputPort(PortDataType.TIME_MS),
    )

    def __init__(self):
        self._display_size_dict = {}
        self._current_texture_tag_dict = {}
        self._texture_tags_dict = {}

    def _compute_display_size(self, frame, width):
        h, w = frame.shape[:2]
        width = max(1, int(width))
        if w <= 0 or h <= 0:
            return width, width
        return width, max(1, int(round(h * (width / float(w)))))

    def _build_texture_tag_with_size(self, node_id, width, height):
        return f'{self.node_tag}:{node_id}:Output01:Texture:{width}x{height}'

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
        ports = self.create_ports(node_id)
        tag_node_input01_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input01')
        tag_node_input01_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input01')
        )
        tag_node_output01_name_port = ports.image
        tag_node_output01_name = tag_node_output01_name_port.dpg_tag
        tag_node_output01_image_name = tag_node_output01_name_port.value_tag
        tag_node_output02_name_port = ports.elapsed
        tag_node_output02_name = tag_node_output02_name_port.dpg_tag
        tag_node_output02_value_name = tag_node_output02_name_port.value_tag

        tag_node_button_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Button')
        tag_node_button_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Button'))

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        texture_tag = self._build_texture_tag_with_size(node_id, small_window_w,
                                                        small_window_h)
        self._display_size_dict[node_id] = (small_window_w, small_window_h)
        self._current_texture_tag_dict[node_id] = texture_tag
        self._texture_tags_dict[node_id] = {texture_tag}

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
            # RTSP URL input field
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_text(
                    tag=tag_node_input01_value_name,
                    label='URL',
                    width=small_window_w - 30,
                )
            # Camera image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(texture_tag,
                              tag=tag_node_output01_image_name,
                              width=small_window_w,
                              height=small_window_h)
            # Add record/playback button
            with dpg.node_attribute(
                    tag=tag_node_button_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label=self._start_label,
                    tag=tag_node_button_value_name,
                    width=small_window_w,
                    callback=self._button,
                    user_data=tag_node_name,
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
        ports = self.ports(node_id)
        input_value01_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input01')
        )
        output_image_tag = ports.image.value_tag
        output_value02_tag = ports.elapsed.value_tag
        texture_tag = self._current_texture_tag_dict.get(node_id, None)

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Translated from Japanese comment
        use_mp = self._opencv_setting_dict['use_multiprocessing_rtsp']

        # Get RTSP URL
        rtsp_url = dpg_get_value(input_value01_tag)

        # Get VideoCapture() instance
        rtsp_capture = None
        image_queue = None
        if rtsp_url != '':
            if use_mp:
                # Translated from Japanese comment
                if rtsp_url in self._image_queue:
                    image_queue = self._image_queue[rtsp_url]
            else:
                # Translated from Japanese comment
                if rtsp_url in self._rtsp_capture:
                    rtsp_capture = self._rtsp_capture[rtsp_url]

        # Measurement start
        if rtsp_url != '' and use_pref_counter:
            start_time = time.perf_counter()

        # Get image
        frame = None
        if use_mp:
            # Translated from Japanese comment
            if image_queue is not None:
                num = image_queue.qsize()
                if num > 0:
                    frame = image_queue.get()
        else:
            # Translated from Japanese comment
            if rtsp_capture is not None:
                ret, frame = rtsp_capture.read()
                if not ret:
                    return None, None

        # Measurement end
        if rtsp_url != '' and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # Draw
        if frame is not None:
            display_w, display_h = self._compute_display_size(frame,
                                                              small_window_w)
            previous_size = self._display_size_dict.get(node_id)
            if previous_size != (display_w, display_h):
                texture_tag = self._build_texture_tag_with_size(
                    node_id, display_w, display_h)
                self._current_texture_tag_dict[node_id] = texture_tag
                if node_id not in self._texture_tags_dict:
                    self._texture_tags_dict[node_id] = set()
                if not dpg.does_item_exist(texture_tag):
                    with dpg.texture_registry(show=False):
                        dpg.add_raw_texture(
                            display_w,
                            display_h,
                            np.zeros((display_w * display_h * 3,), dtype='f'),
                            tag=texture_tag,
                            format=dpg.mvFormat_Float_rgb,
                        )
                    self._texture_tags_dict[node_id].add(texture_tag)
                dpg.configure_item(output_image_tag,
                                   texture_tag=texture_tag,
                                   width=display_w,
                                   height=display_h)
                self._display_size_dict[node_id] = (display_w, display_h)
            texture = convert_cv_to_dpg(
                frame,
                display_w,
                display_h,
            )
            dpg_set_value(texture_tag, texture)

        return frame, None

    def close(self, node_id):
        # Translated from Japanese comment
        use_mp = self._opencv_setting_dict['use_multiprocessing_rtsp']
        if use_mp:
            # Translated from Japanese comment
            for rtsp_url in self._process.keys():
                self._request[rtsp_url].value = 0
                if self._process[rtsp_url].is_alive():
                    self._process[rtsp_url].terminate()
        texture_tags = self._texture_tags_dict.pop(node_id, set())
        self._current_texture_tag_dict.pop(node_id, None)
        self._display_size_dict.pop(node_id, None)
        for texture_tag in texture_tags:
            if dpg.does_item_exist(texture_tag):
                try:
                    dpg.delete_item(texture_tag)
                except (SystemError, RuntimeError):
                    pass

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        tag_node_input01_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input01'))

        pos = dpg.get_item_pos(tag_node_name)
        rtsp_url = dpg_get_value(tag_node_input01_value_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_input01_value_name] = rtsp_url

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        tag_node_input01_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input01'))

        rtsp_url = setting_dict[tag_node_input01_value_name]

        dpg_set_value(tag_node_input01_value_name, rtsp_url)

    def _button(self, sender, data, user_data):
        tag_node_name = user_data
        input_value01_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input01'))
        tag_node_button_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Button'))

        label = dpg.get_item_label(tag_node_button_value_name)

        # Get RTSP URL
        rtsp_url = dpg_get_value(input_value01_tag)

        # Translated from Japanese comment
        use_mp = self._opencv_setting_dict['use_multiprocessing_rtsp']

        if label == self._start_label:
            if rtsp_url != '':
                if use_mp:
                    # Translated from Japanese comment
                    if not (rtsp_url in self._process):
                        self._image_queue[rtsp_url] = mp.Queue(maxsize=1)
                        self._request[rtsp_url] = mp.Value('i', 1)
                        self._process[rtsp_url] = mp.Process(
                            target=receive_image_process,
                            args=(rtsp_url, self._image_queue[rtsp_url],
                                  self._request[rtsp_url]),
                        )
                        self._process[rtsp_url].start()
                else:
                    # Translated from Japanese comment
                    if not (rtsp_url in self._rtsp_capture):
                        rtsp_capture = cv2.VideoCapture(rtsp_url)
                        self._rtsp_capture[rtsp_url] = rtsp_capture

            dpg.set_item_label(tag_node_button_value_name, self._stop_label)
        elif label == self._stop_label:
            if rtsp_url != '':
                if use_mp:
                    # Translated from Japanese comment
                    if rtsp_url in self._request:
                        self._request[rtsp_url].value = 0
                        if self._process[rtsp_url].is_alive():
                            self._process[rtsp_url].terminate()
                        self._image_queue.pop(rtsp_url)
                        self._request.pop(rtsp_url)
                        self._process.pop(rtsp_url)
                else:
                    # Translated from Japanese comment
                    if rtsp_url in self._rtsp_capture:
                        self._rtsp_capture[rtsp_url].release()
                        self._rtsp_capture.pop(rtsp_url)

            dpg.set_item_label(tag_node_button_value_name, self._start_label)
