#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import copy

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Video(Set Frame Position)'
    node_tag = 'VideoSetFramePos'

    _opencv_setting_dict = None

    _video_capture = {}
    _movie_filepath = {}
    _prev_movie_filepath = {}
    _prev_frame_pos = {}
    _prev_frame = {}

    _min_val = 0
    _max_val = 10000000

    _window_resize_rate = 1.5

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
        tag_node_input01_name = self._node_port_tag(node_id, self.TYPE_INT, 'Input01')
        tag_node_input02_name_port = self.input_port(node_id, self.TYPE_INT, 'Input02')
        tag_node_input02_name = tag_node_input02_name_port.dpg_tag
        tag_node_input02_value_name = tag_node_input02_name_port.value_tag
        tag_node_output01_name_port = self.output_port(node_id, self.TYPE_IMAGE, 'Output01')
        tag_node_output01_name = tag_node_output01_name_port.dpg_tag
        tag_node_output01_image_name = tag_node_output01_name_port.value_tag
        tag_node_output02_name_port = self.output_port(node_id, self.TYPE_TIME_MS, 'Output02')
        tag_node_output02_name = tag_node_output02_name_port.dpg_tag
        tag_node_output02_value_name = tag_node_output02_name_port.value_tag
        tag_node_output03_name_port = self.output_port(node_id, self.TYPE_INT, 'Output03')
        tag_node_output03_name = tag_node_output03_name_port.dpg_tag
        tag_node_output03_value_name = tag_node_output03_name_port.value_tag

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_w = int(small_window_w * self._window_resize_rate)
        small_window_h = self._opencv_setting_dict['input_window_height']
        small_window_h = int(small_window_h * self._window_resize_rate)
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
                width=int(small_window_w * 3 / self._window_resize_rate),
                height=int(small_window_h * 3 / self._window_resize_rate),
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
                dpg.add_image(texture_tag,
                              tag=tag_node_output01_image_name,
                              width=small_window_w,
                              height=small_window_h)
            # Seek
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_slider_int(
                    tag=tag_node_input02_value_name,
                    label='',
                    width=small_window_w,
                    default_value=1,
                    min_value=self._min_val,
                    max_value=self._max_val,
                    format='',
                    callback=None,
                )
            # Frame position
            with dpg.node_attribute(
                    tag=tag_node_output03_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    '0',
                    tag=tag_node_output03_value_name,
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
        tag_node_input02_value_name = self._node_value_tag(node_id, self.TYPE_INT, 'Input02')
        output_image_tag = self._node_value_tag(node_id, self.TYPE_IMAGE, 'Output01')
        output_value02_tag = self._node_value_tag(node_id, self.TYPE_TIME_MS, 'Output02')
        output_value03_tag = self._node_value_tag(node_id, self.TYPE_INT, 'Output03')
        texture_tag = self._current_texture_tag_dict.get(node_id, None)

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_w = int(small_window_w * self._window_resize_rate)
        small_window_h = self._opencv_setting_dict['input_window_height']
        small_window_h = int(small_window_h * self._window_resize_rate)
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Check connection info
        seek_input_value = None
        for connection_info, source_tag, _, connection_type in self._iter_connection_infos(
                connection_list):
            if connection_type == self.TYPE_INT:
                # Get connection tag
                source_value_tag = self._connection_value_tag(connection_info, 'source', source_tag)
                # Get value
                seek_input_value = int(dpg_get_value(source_value_tag))
                seek_input_value = max([self._min_val, seek_input_value])
                seek_input_value = min([self._max_val, seek_input_value])

        # Create VideoCapture() instance
        update_flag = False
        movie_path = self._movie_filepath.get(str(node_id), None)
        prev_movie_path = self._prev_movie_filepath.get(str(node_id), None)
        if prev_movie_path != movie_path:
            video_capture = self._video_capture.get(str(node_id), None)
            if video_capture is not None:
                video_capture.release()
            self._video_capture[str(node_id)] = cv2.VideoCapture(movie_path)
            self._prev_movie_filepath[str(node_id)] = movie_path

            # Reset seek position
            self._video_capture[str(node_id)].set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Reset frame count
            dpg_set_value(tag_node_input02_value_name, 0)
            dpg_set_value(output_value03_tag, str(0))
            update_flag = True

        video_capture = self._video_capture.get(str(node_id), None)

        # Seek position
        seek_value = int(dpg_get_value(tag_node_input02_value_name))

        # Measurement start
        if video_capture is not None and use_pref_counter:
            start_time = time.perf_counter()

        # Get image
        frame = None
        if video_capture is not None:
            # Calculate frame number from seek position
            total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if seek_input_value is None:
                frame_pos = int(total_frame * (seek_value / self._max_val))
            else:
                frame_pos = seek_input_value

            # Range check
            if frame_pos < 0:
                frame_pos = 0
            if total_frame <= frame_pos:
                frame_pos = total_frame - 1

            # If seek position is input from another node, update seek bar position
            if seek_input_value is not None:
                seek_set_value = int(self._max_val * (frame_pos / total_frame))
                dpg_set_value(tag_node_input02_value_name, seek_set_value)

            if str(node_id) in self._prev_frame_pos:
                # Get image if frame position changed
                if self._prev_frame_pos[str(node_id)] != frame_pos:
                    update_flag = True
                else:
                    frame = copy.deepcopy(self._prev_frame[str(node_id)])
            else:
                # Get first image
                update_flag = True

            if update_flag:
                # Seek
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                # Get image
                _, frame = video_capture.read()
                # Store position and image when acquired
                self._prev_frame_pos[str(node_id)] = frame_pos
                self._prev_frame[str(node_id)] = copy.deepcopy(frame)
                # Frame count display
                dpg_set_value(output_value03_tag, str(frame_pos))

        # Measurement end
        if video_capture is not None and use_pref_counter:
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
        tag_node_input02_value_name = self._node_value_tag(node_id, self.TYPE_INT, 'Input02')

        pos = dpg.get_item_pos(tag_node_name)

        seek_value = int(dpg_get_value(tag_node_input02_value_name))

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_input02_value_name] = seek_value

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)
        tag_node_input02_value_name = self._node_value_tag(node_id, self.TYPE_INT, 'Input02')

        seek_value = setting_dict[tag_node_input02_value_name]

        dpg_set_value(tag_node_input02_value_name, seek_value)

    def _callback_file_select(self, sender, data):
        if data['file_name'] != '.':
            node_id = sender.split(':')[1]
            self._movie_filepath[node_id] = data['file_path_name']
