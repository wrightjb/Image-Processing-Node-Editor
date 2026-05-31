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


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Video'
    node_tag = 'Video'

    _opencv_setting_dict = None

    _video_capture = {}
    _movie_filepath = {}
    _prev_movie_filepath = {}
    _frame_count = {}
    _playback_start_time = {}
    _playback_start_frame = {}
    _last_output_frame = {}

    _min_val = 1
    _max_val = 10

    def __init__(self):
        self._display_size_dict = {}
        self._current_texture_tag_dict = {}
        self._texture_tags_dict = {}

    def _compute_display_size(self, frame, width):
        image_h, image_w = frame.shape[:2]
        width = max(1, int(width))
        if image_w <= 0 or image_h <= 0:
            return width, width
        scale = width / float(image_w)
        height = max(1, int(round(image_h * scale)))
        return width, height

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
        tag_node_input01_name = self._port_tag(tag_node_name, self.TYPE_INT, 'Input01')
        tag_node_input02_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02')
        tag_node_input02_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_node_input03_name_port = self.input_port(node_id, self.TYPE_INT, 'Input03')
        tag_node_input03_name = tag_node_input03_name_port.dpg_tag
        tag_node_input03_value_name = tag_node_input03_name_port.value_tag
        tag_node_input04_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input04')
        tag_node_input04_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input04'))
        tag_node_input05_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input05')
        tag_node_input05_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input05'))
        tag_node_output01_name_port = self.output_port(node_id, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_name = tag_node_output01_name_port.dpg_tag
        tag_node_output01_image_name = tag_node_output01_name_port.value_tag
        tag_node_output02_name_port = self.output_port(node_id, self.TYPE_TIME_MS, 'Output02')
        tag_node_output02_name = tag_node_output02_name_port.dpg_tag
        tag_node_output02_value_name = tag_node_output02_name_port.value_tag

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

        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(small_window_h * 3),
                callback=self._callback_file_select,
                id='movie_select:' + str(node_id),
        ):
            dpg.add_file_extension('Movie (*.mp4 *.avi){.mp4,.avi}')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        # Node
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # File selection
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label='Select Movie',
                    width=small_window_w,
                    callback=lambda: dpg.show_item(
                        'movie_select:' + str(node_id), ),
                )
            # Camera image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(
                    texture_tag,
                    tag=tag_node_output01_image_name,
                    width=small_window_w,
                    height=small_window_h,
                )
            # Loop enabled
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    label='Loop',
                    tag=tag_node_input02_value_name,
                    callback=callback,
                    user_data=tag_node_name,
                    default_value=True,
                )
            # Kernel size
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input03_value_name,
                    label="Skip Rate",
                    width=small_window_w - 80,
                    default_value=1,
                    min_value=self._min_val,
                    max_value=self._max_val,
                    callback=callback,
                )
            # Playback mode
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    label='Natural FPS',
                    tag=tag_node_input04_value_name,
                    callback=callback,
                    default_value=True,
                )
            # Use cache (for source nodes)
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    label='Cache Source',
                    tag=tag_node_input05_value_name,
                    callback=callback,
                    default_value=False,
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
        tag_node_input02_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_node_input03_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_INT, 'Input03'))
        tag_node_input04_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input04'))
        tag_node_input05_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input05'))
        output_image_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_IMAGE, 'Output01'))
        output_value02_tag = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Output02'))
        texture_tag = self._current_texture_tag_dict.get(node_id, None)

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Check connection info
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

        # Create VideoCapture() instance
        movie_path = self._movie_filepath.get(str(node_id), None)
        prev_movie_path = self._prev_movie_filepath.get(str(node_id), None)
        if prev_movie_path != movie_path:
            video_capture = self._video_capture.get(str(node_id), None)
            if video_capture is not None:
                video_capture.release()
            self._video_capture[str(node_id)] = cv2.VideoCapture(movie_path)
            self._prev_movie_filepath[str(node_id)] = movie_path
            self._frame_count[str(node_id)] = 0
            self._playback_start_time[str(node_id)] = time.perf_counter()
            self._playback_start_frame[str(node_id)] = 0
            self._last_output_frame.pop(str(node_id), None)

        video_capture = self._video_capture.get(str(node_id), None)

        # Loop enabled
        loop_flag = dpg_get_value(tag_node_input02_value_name)
        # Skip ratio
        skip_rate = int(dpg_get_value(tag_node_input03_value_name))
        natural_fps_mode = bool(dpg_get_value(tag_node_input04_value_name))
        cache_source = bool(dpg_get_value(tag_node_input05_value_name))

        # Measurement start
        decode_start_time = None
        if video_capture is not None and use_pref_counter:
            decode_start_time = time.perf_counter()

        # Get image
        frame = None
        if video_capture is not None:
            total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(video_capture.get(cv2.CAP_PROP_FPS))
            if source_fps <= 0:
                source_fps = 30.0

            if natural_fps_mode:
                now = time.perf_counter()
                playback_start_time = self._playback_start_time.get(str(node_id), now)
                elapsed_sec = max(0.0, now - playback_start_time)
                elapsed_frame = int(elapsed_sec * source_fps)
                target_frame_pos = elapsed_frame * skip_rate

                if total_frame > 0 and loop_flag:
                    target_frame_pos = target_frame_pos % total_frame
                elif total_frame > 0:
                    target_frame_pos = min(target_frame_pos, total_frame - 1)

                current_frame_pos = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                if target_frame_pos != current_frame_pos:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_pos)

                ret, frame = video_capture.read()
                if not ret:
                    if loop_flag:
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = video_capture.read()
                        self._playback_start_time[str(node_id)] = now
                        self._frame_count[str(node_id)] = 0
                    else:
                        frame = self._last_output_frame.get(str(node_id), None)
            else:
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        if loop_flag:
                            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            _, frame = video_capture.read()
                        else:
                            frame = self._last_output_frame.get(str(node_id), None)
                        break

                    self._frame_count[str(node_id)] += 1
                    if (self._frame_count[str(node_id)] % skip_rate) == 0:
                        break

            if frame is not None:
                self._frame_count[str(node_id)] = int(
                    video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                )
                self._last_output_frame[str(node_id)] = frame.copy()

        # Measurement end
        if video_capture is not None and use_pref_counter and decode_start_time is not None:
            elapsed_time = time.perf_counter() - decode_start_time
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
                empty_texture = np.zeros((display_w * display_h * 3,),
                                         dtype='f')
                if not dpg.does_item_exist(texture_tag):
                    with dpg.texture_registry(show=False):
                        dpg.add_raw_texture(
                            display_w,
                            display_h,
                            empty_texture,
                            tag=texture_tag,
                            format=dpg.mvFormat_Float_rgb,
                        )
                    self._texture_tags_dict[node_id].add(texture_tag)
                dpg.configure_item(
                    output_image_tag,
                    texture_tag=texture_tag,
                    width=display_w,
                    height=display_h,
                )
                self._display_size_dict[node_id] = (display_w, display_h)
            elif texture_tag is None:
                texture_tag = self._build_texture_tag_with_size(
                    node_id, display_w, display_h)
                self._current_texture_tag_dict[node_id] = texture_tag
            texture = convert_cv_to_dpg(
                frame,
                display_w,
                display_h,
            )
            dpg_set_value(texture_tag, texture)

        stream_id = self._movie_filepath.get(str(node_id), None)
        frame_index = self._frame_count.get(str(node_id), 0)
        result = {
            '__cache_stream__': stream_id,
            '__cache_frame__': frame_index,
            '__cache_kind__': 'video_frame',
        }

        return frame, result

    def close(self, node_id):
        video_capture = self._video_capture.get(str(node_id), None)
        if video_capture is not None:
            video_capture.release()
        self._video_capture.pop(str(node_id), None)
        self._movie_filepath.pop(str(node_id), None)
        self._prev_movie_filepath.pop(str(node_id), None)
        self._frame_count.pop(str(node_id), None)
        self._playback_start_time.pop(str(node_id), None)
        self._playback_start_frame.pop(str(node_id), None)
        self._last_output_frame.pop(str(node_id), None)
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
        tag_node_input02_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_node_input03_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_INT, 'Input03'))
        tag_node_input04_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input04'))
        tag_node_input05_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input05'))

        pos = dpg.get_item_pos(tag_node_name)

        loop_flag = dpg_get_value(tag_node_input02_value_name)
        skip_rate = int(dpg_get_value(tag_node_input03_value_name))
        natural_fps_mode = bool(dpg_get_value(tag_node_input04_value_name))
        cache_source = bool(dpg_get_value(tag_node_input05_value_name))

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict['movie_path'] = self._movie_filepath.get(str(node_id), None)
        setting_dict[tag_node_input02_value_name] = loop_flag
        setting_dict[tag_node_input03_value_name] = skip_rate
        setting_dict[tag_node_input04_value_name] = natural_fps_mode
        setting_dict[tag_node_input05_value_name] = cache_source
        setting_dict['__cache_source_enabled__'] = cache_source

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        tag_node_input02_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input02'))
        tag_node_input03_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_INT, 'Input03'))
        tag_node_input04_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input04'))
        tag_node_input05_value_name = self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input05'))

        movie_path = setting_dict.get('movie_path', None)
        node_id = str(node_id)
        if movie_path is not None:
            if os.path.isfile(movie_path):
                self._movie_filepath[node_id] = movie_path
            else:
                self._movie_filepath.pop(node_id, None)
                self._prev_movie_filepath.pop(node_id, None)
                self._video_capture.pop(node_id, None)
                self._last_output_frame.pop(node_id, None)
                print(f'WARNING : Movie file not found ({movie_path})')

        loop_flag = setting_dict[tag_node_input02_value_name]
        skip_rate = int(setting_dict[tag_node_input03_value_name])
        natural_fps_mode = setting_dict.get(tag_node_input04_value_name, True)
        cache_source = bool(
            setting_dict.get(
                tag_node_input05_value_name,
                setting_dict.get('__cache_source_enabled__', False),
            )
        )

        dpg_set_value(tag_node_input02_value_name, loop_flag)
        dpg_set_value(tag_node_input03_value_name, skip_rate)
        dpg_set_value(tag_node_input04_value_name, natural_fps_mode)
        dpg_set_value(tag_node_input05_value_name, cache_source)

    def _callback_file_select(self, sender, data):
        if data['file_name'] != '.':
            node_id = sender.split(':')[1]
            self._movie_filepath[node_id] = data['file_path_name']
