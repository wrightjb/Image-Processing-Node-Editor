#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves adjustment node for Image Processing Node Editor."""
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, convert_cv_to_dpg
from node.node_abc import DpgNodeABC


def image_process(image, points):
    # Builds LUT from curves points
    # Assumes points are pre-sorted and between 0 and 255
    xs, ys = zip(*points)
    table = np.interp(np.arange(256), xs, ys).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image


class Node(DpgNodeABC):
    """Curves adjustment node."""

    _ver = "0.0.1"

    node_label = "Curves"
    node_tag = "Curves"

    _min_val = 0
    _max_val = 255

    _opencv_setting_dict = None

    def __init__(self):
        pass

    def _get_tag_node_name(self, node_id):
        return f"{node_id}:{self.node_tag}"

    def _get_tag_plot_name(self, node_id):
        return f"{node_id}:{self.node_tag}:plot"

    def _get_tag_plot_series_name(self, node_id):
        return f"{node_id}:{self.node_tag}:line"

    def _get_tag_drag_point_handler_name(self, node_id):
        return f"{node_id}:{self.node_tag}:drag_point_handler"

    def _get_drag_points(self, node_id):
        plot_tag = self._get_tag_plot_name(node_id)
        # Drag points should be only children in slot 0
        point_items = dpg.get_item_children(plot_tag, slot=0)
        return sorted([dpg.get_value(tag) for tag in point_items])

    def _callback_add_point(self, sender, app_data, user_data):
        node_id = user_data[0]
        static_x = user_data[1]
        plot_tag = self._get_tag_plot_name(node_id)
        point_handler_tag = self._get_tag_drag_point_handler_name(node_id)

        if static_x is not None:
            # For endpoints only, which initially have x = y
            y = x = static_x
        else:
            # Click must be within plot, so no need to clip to range(?)
            x, y = dpg.get_plot_mouse_pos()

        point_tag = dpg.add_drag_point(
            parent=plot_tag,
            label="",
            default_value=[x, y],
            delayed=True,
            callback=self._callback_moved_point,
            user_data=(node_id, static_x),
        )
        dpg.bind_item_handler_registry(point_tag, point_handler_tag)
        dpg.set_item_user_data(point_tag, (node_id, static_x, point_tag))
        self._redraw_line(node_id)

    def _callback_moved_point(self, sender, app_data, user_data):
        node_id = user_data[0]
        static_x = user_data[1]

        x, y = dpg.get_value(sender)
        if static_x is not None:
            x = static_x
        # Clip y to range (should be 0-255)
        y = max(self._min_val, min(self._max_val, int(y)))
        dpg.set_value(sender, [x, y])
        self._redraw_line(node_id)

    def _callback_delete_point(self, sender, app_data, user_data):
        node_id, static_x, point_tag = user_data

        if static_x is not None:
            return

        if dpg.does_item_exist(point_tag):
            dpg.delete_item(point_tag)
        self._redraw_line(node_id)

    def _redraw_line(
        self, 
        node_id
    ):
        # Rebuild curve line series based on current drag points.
        plot_tag = self._get_tag_plot_name(node_id)
        points = self._get_drag_points(node_id)
        x, y = zip(*points)
        series_tag = self._get_tag_plot_series_name(node_id)
        dpg.set_value(series_tag, [x, y])

    def add_node(
        self, 
        parent, 
        node_id, 
        pos=[0, 0], 
        opencv_setting_dict=None, 
        callback=None
    ):
        # tag names
        tag_node_name = self._get_tag_node_name(node_id)
        tag_plot_name = self._get_tag_plot_name(node_id)
        tag_plot_series_name = self._get_tag_plot_series_name(node_id)
        tag_node_input_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Input01"
        tag_node_input_value_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Input01Value"
        tag_node_output_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Output01"
        tag_node_output_value_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Output01Value"
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict["process_width"]
        small_window_h = self._opencv_setting_dict["process_height"]
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # initial black image
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(
            black_image, 
            small_window_w, 
            small_window_h
        )

        # texture registration
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=tag_node_output_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # Add node
        with dpg.node(
            tag=tag_node_name, 
            parent=parent, 
            label=self.node_label, 
            pos=pos
        ):
            # Add input port
            with dpg.node_attribute(
                tag=tag_node_input_name, 
                attribute_type=dpg.mvNode_Attr_Input
            ):
                dpg.add_text(
                    tag=tag_node_input_value_name, 
                    default_value="Input BGR image"
                )
            # Add image
            with dpg.node_attribute(
                tag=tag_node_output_name, 
                attribute_type=dpg.mvNode_Attr_Output
            ):
                dpg.add_image(tag_node_output_value_name)
            # Add curve editor
            with dpg.node_attribute(
                attribute_type=dpg.mvNode_Attr_Static
            ):
                with dpg.plot(
                    width=240, 
                    height=180, 
                    tag=tag_plot_name, 
                    no_menus=True
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis, 
                        tag=f"{tag_node_name}:plot_x"
                    )
                    dpg.set_axis_limits(
                        dpg.last_item(), 
                        self._min_val, 
                        self._max_val
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis, 
                        tag=f"{tag_node_name}:plot_y"
                    )
                    dpg.set_axis_limits(
                        dpg.last_item(), 
                        self._min_val, 
                        self._max_val
                    )
                    dpg.add_line_series(
                        x=[self._min_val, self._max_val],
                        y=[self._min_val, self._max_val],
                        parent=f"{tag_node_name}:plot_y",
                        tag=tag_plot_series_name,
                    )
                    handler = dpg.add_item_handler_registry()
                    dpg.add_item_clicked_handler(
                        callback=self._callback_add_point,
                        user_data=(node_id, None),
                        button=dpg.mvMouseButton_Left,
                        parent=handler,
                    )
                    point_handler_tag = self._get_tag_drag_point_handler_name(
                        node_id
                    )
                    with dpg.item_handler_registry(tag=point_handler_tag):
                        dpg.add_item_clicked_handler(
                            callback=self._callback_delete_point,
                            button=dpg.mvMouseButton_Right,
                        )
                    # Add bottom right point
                    self._callback_add_point(
                        sender=None, 
                        app_data=None,
                        user_data=(node_id, self._min_val)
                    )
                    # Add top right point
                    self._callback_add_point(
                        sender=None, 
                        app_data=None,
                        user_data=(node_id, self._max_val)
                    )
                    dpg.bind_item_handler_registry(tag_plot_name, handler)
            # processing time
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
        node_result_dict
    ):
        node_id = int(node_id)
        tag_node_name = f"{node_id}:{self.node_tag}"
        output_value01_tag = f"{tag_node_name}:{self.TYPE_IMAGE}:Output01Value"
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        plot_tag = f"{tag_node_name}:plot"

        small_window_w = self._opencv_setting_dict["process_width"]
        small_window_h = self._opencv_setting_dict["process_height"]
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # connection check
        connection_info_src = ''
        for connection_info in connection_list:
            if connection_info[0].split(':')[2] == self.TYPE_IMAGE:
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)

        # get image
        frame = node_image_dict.get(connection_info_src, None)

        # get points from curves chart
        points = self._get_drag_points(node_id)

        # start timer
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        # process image
        if frame is not None:
            frame = image_process(frame, points)

        # stop timer
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # set display image
        if frame is not None:
            texture = convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, None

    def close(self, node_id):
        node_id = int(node_id)
        # Clean here

    def get_setting_dict(self, node_id):
        node_id = int(node_id)
        tag_node_name = f"{node_id}:{self.node_tag}"
        pos = dpg.get_item_pos(tag_node_name)
        points = self._get_drag_points(node_id)
        setting_dict = {
            "ver": self._ver,
            "pos": pos,
            "points": points,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        plot_tag = self._get_tag_plot_name(node_id)
        point_handler_tag = self._get_tag_drag_point_handler_name(node_id)
        for pt in setting_dict.get("points", []):
            static_x = pt[0] if pt[0] in [self._min_val, self._max_val] else None
            point_tag = dpg.add_drag_point(
                parent=plot_tag,
                label="",
                default_value=pt,
                callback=self._callback_moved_point,
                user_data=(node_id, static_x)
            )
            dpg.bind_item_handler_registry(point_tag, point_handler_tag)
            dpg.set_item_user_data(point_tag, (node_id, static_x, point_tag))
        self._redraw_line(node_id)
