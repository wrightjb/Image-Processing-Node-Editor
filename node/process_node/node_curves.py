#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves adjustment node for Image Processing Node Editor."""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, convert_cv_to_dpg
from node.node_abc import DpgNodeABC


class Node(DpgNodeABC):
    """Curves adjustment node."""

    _ver = "0.0.1"

    node_label = "Curves"
    node_tag = "Curves"

    _opencv_setting_dict = None

    # drag point tag list per node id
    _drag_points = {}
    _line_series = {}

    def __init__(self):
        pass

    def _callback_add_point(self, sender, app_data, user_data):
        node_id = user_data[0]
        plot_tag = user_data[1]
        # app_data from the clicked handler is the mouse button, not position.
        # Use DearPyGui to fetch the mouse position over the plot instead.
        x, y = dpg.get_plot_mouse_pos()
        # clip to range 0-255
        x = max(0, min(255, int(x)))
        y = max(0, min(255, int(y)))
        tag = f"{node_id}:drag_point_{len(self._drag_points.get(node_id, []))}"
        drag_tag = dpg.add_drag_point(
            parent=plot_tag,
            label="",
            default_value=[x, y],
            callback=self._callback_moved_point,
            user_data=node_id,
        )
        self._drag_points.setdefault(node_id, []).append(drag_tag)
        self._redraw_line(node_id, plot_tag)

    def _callback_moved_point(self, sender, app_data, user_data):
        node_id = user_data
        plot_tag = f"{node_id}:{self.node_tag}:plot"
        self._redraw_line(node_id, plot_tag)

    def _redraw_line(self, node_id, plot_tag):
        """Rebuild curve line series based on current drag points."""
        points = [dpg.get_value(tag) for tag in self._drag_points.get(node_id, [])]
        points.append([0, 0])
        points.append([255, 255])
        # sort by x
        points = sorted(points, key=lambda p: p[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        series_tag = self._line_series.get(node_id)
        if series_tag is None:
            series_tag = dpg.add_line_series(x, y, parent=f"{plot_tag}_y")
            self._line_series[node_id] = series_tag
        else:
            dpg_set_value(series_tag, [x, y])

    def add_node(self, parent, node_id, pos=[0, 0], opencv_setting_dict=None, callback=None):
        tag_node_name = f"{node_id}:{self.node_tag}"
        tag_node_input_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Input01"
        tag_node_input_value_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Input01Value"
        tag_node_output_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Output01"
        tag_node_output_value_name = f"{tag_node_name}:{self.TYPE_IMAGE}:Output01Value"

        self._opencv_setting_dict = opencv_setting_dict
        small_w = self._opencv_setting_dict["process_width"]
        small_h = self._opencv_setting_dict["process_height"]

        black_image = np.zeros((small_w, small_h, 3))
        black_texture = convert_cv_to_dpg(black_image, small_w, small_h)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_w,
                small_h,
                black_texture,
                tag=tag_node_output_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        with dpg.node(tag=tag_node_name, parent=parent, label=self.node_label, pos=pos):
            with dpg.node_attribute(tag=tag_node_input_name, attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text(tag=tag_node_input_value_name, default_value="Input BGR image")
            with dpg.node_attribute(tag=tag_node_output_name, attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_image(tag_node_output_value_name)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                with dpg.plot(width=240, height=240, tag=f"{tag_node_name}:plot", no_menus=True):
                    dpg.add_plot_axis(dpg.mvXAxis, tag=f"{tag_node_name}:plot_x")
                    dpg.set_axis_limits(dpg.last_item(), 0, 255)
                    dpg.add_plot_axis(dpg.mvYAxis, tag=f"{tag_node_name}:plot_y")
                    dpg.set_axis_limits(dpg.last_item(), 0, 255)
                    self._line_series[node_id] = dpg.add_line_series(
                        [0, 255],
                        [0, 255],
                        parent=f"{tag_node_name}:plot_y",
                        tag=f"{tag_node_name}:line",
                    )
                    handler = dpg.add_item_handler_registry()
                    dpg.add_item_clicked_handler(
                        callback=self._callback_add_point,
                        user_data=(node_id, f"{tag_node_name}:plot"),
                        parent=handler,
                    )
                    dpg.bind_item_handler_registry(f"{tag_node_name}:plot", handler)
        return tag_node_name

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        tag_node_name = f"{node_id}:{self.node_tag}"
        output_value_tag = f"{tag_node_name}:{self.TYPE_IMAGE}:Output01Value"
        plot_tag = f"{tag_node_name}:plot"

        small_w = self._opencv_setting_dict["process_width"]
        small_h = self._opencv_setting_dict["process_height"]

        connection_info_src = ''
        for connection_info in connection_list:
            if connection_info[0].split(':')[2] == self.TYPE_IMAGE:
                connection_info_src = connection_info[0]
                connection_info_src = ':'.join(connection_info_src.split(':')[:2])

        frame = node_image_dict.get(connection_info_src, None)

        # build LUT from drag points
        points = [[0, 0], [255, 255]]
        for tag in self._drag_points.get(node_id, []):
            pt = dpg.get_value(tag)
            pt[0] = max(0, min(255, pt[0]))
            pt[1] = max(0, min(255, pt[1]))
            points.append(pt)
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        table = np.interp(np.arange(256), xs, ys).astype(np.uint8)

        if frame is not None:
            result = cv2.LUT(frame, table)
            texture = convert_cv_to_dpg(result, small_w, small_h)
            dpg_set_value(output_value_tag, texture)
            # redraw line
            self._redraw_line(node_id, plot_tag)
            return result, None
        return None, None

    def close(self, node_id):
        # clean stored points
        self._drag_points.pop(node_id, None)
        self._line_series.pop(node_id, None)

    def get_setting_dict(self, node_id):
        tag_node_name = f"{node_id}:{self.node_tag}"
        pos = dpg.get_item_pos(tag_node_name)
        point_list = [dpg.get_value(tag) for tag in self._drag_points.get(node_id, [])]
        setting_dict = {
            "ver": self._ver,
            "pos": pos,
            "points": point_list,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        plot_tag = f"{node_id}:{self.node_tag}:plot"
        self._drag_points[node_id] = []
        for pt in setting_dict.get("points", []):
            drag_tag = dpg.add_drag_point(
                parent=plot_tag,
                label="",
                default_value=pt,
                callback=self._callback_moved_point,
                user_data=node_id,
            )
            self._drag_points[node_id].append(drag_tag)
        self._redraw_line(node_id, plot_tag)
