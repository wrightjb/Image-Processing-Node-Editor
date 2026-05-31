#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
import dearpygui.dearpygui as dpg


from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node_editor.util import convert_cv_to_dpg


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'WebCam'
    node_tag = 'WebCam'

    _opencv_setting_dict = None

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
        tag_node_input01_value_name = self._node_value_tag(node_id, self.TYPE_INT, 'Input01')
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
        device_no_list = self._opencv_setting_dict['device_no_list']
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
            # Camera number selection combo box
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    device_no_list,
                    width=small_window_w - 100,
                    label="Device No",
                    tag=tag_node_input01_value_name,
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
        input_value01_tag = self._node_value_tag(node_id, self.TYPE_INT, 'Input01')
        output_image_tag = self._node_value_tag(node_id, self.TYPE_IMAGE, 'Output01')
        output_value02_tag = self._node_value_tag(node_id, self.TYPE_TIME_MS, 'Output02')
        texture_tag = self._current_texture_tag_dict.get(node_id, None)

        device_no_list = self._opencv_setting_dict['device_no_list']
        camera_capture_list = self._opencv_setting_dict['camera_capture_list']
        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # Get camera number
        camera_no = dpg_get_value(input_value01_tag)

        # Get VideoCapture() instance
        camera_capture = None
        if camera_no != '':
            camera_no = int(camera_no)
            camera_index = device_no_list.index(camera_no)
            camera_capture = camera_capture_list[camera_index]

        # Measurement start
        if camera_no != '' and use_pref_counter:
            start_time = time.perf_counter()

        # Get image
        frame = None
        if camera_capture is not None:
            ret, frame = camera_capture.read()
            if not ret:
                return

        # Measurement end
        if camera_no != '' and use_pref_counter:
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
                dpg.configure_item(
                    output_image_tag,
                    texture_tag=texture_tag,
                    width=display_w,
                    height=display_h,
                )
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

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
