#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves adjustment node for Image Processing Node Editor."""
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, convert_cv_to_dpg
from node.node_abc import DpgNodeABC


def _dedupe_points(points):
    deduped = {}
    for x, y in points:
        x = float(np.clip(x, 0, 255))
        y = float(np.clip(y, 0, 255))
        deduped[x] = y
    return sorted(deduped.items())


def _linear_curve(points, samples=256):
    points = _dedupe_points(points)
    xs, ys = zip(*points)
    sample_x = np.linspace(0, 255, samples)
    sample_y = np.interp(sample_x, xs, ys)
    return sample_x, np.clip(sample_y, 0, 255)


def _spline_curve(points, samples=4096, fit_spline=False):
    """Build a monotone cubic Hermite spline through the curve points."""
    points = _dedupe_points(points)
    if len(points) < 3:
        return _linear_curve(points, samples)

    xs = np.array([point[0] for point in points], dtype=np.float64)
    ys = np.array([point[1] for point in points], dtype=np.float64)
    dx = np.diff(xs)
    dy = np.diff(ys)
    slopes = dy / dx

    tangents = np.zeros_like(ys)
    tangents[0] = slopes[0]
    tangents[-1] = slopes[-1]
    for index in range(1, len(ys) - 1):
        if slopes[index - 1] * slopes[index] <= 0:
            tangents[index] = 0
        elif fit_spline:
            weight_prev = 2 * dx[index] + dx[index - 1]
            weight_next = dx[index] + 2 * dx[index - 1]
            tangents[index] = (
                (weight_prev + weight_next) /
                ((weight_prev / slopes[index - 1]) +
                 (weight_next / slopes[index]))
            )
        else:
            tangents[index] = (slopes[index - 1] + slopes[index]) / 2

    if fit_spline:
        for index, slope in enumerate(slopes):
            if slope == 0:
                tangents[index] = 0
                tangents[index + 1] = 0
                continue
            alpha = tangents[index] / slope
            beta = tangents[index + 1] / slope
            distance = alpha * alpha + beta * beta
            if distance > 9:
                tau = 3 / np.sqrt(distance)
                tangents[index] = tau * alpha * slope
                tangents[index + 1] = tau * beta * slope

    sample_x = np.linspace(0, 255, samples)
    sample_y = np.empty_like(sample_x)
    segment_indexes = np.searchsorted(xs, sample_x, side="right") - 1
    segment_indexes = np.clip(segment_indexes, 0, len(xs) - 2)

    for segment_index in range(len(xs) - 1):
        mask = segment_indexes == segment_index
        if not np.any(mask):
            continue
        x0 = xs[segment_index]
        x1 = xs[segment_index + 1]
        y0 = ys[segment_index]
        y1 = ys[segment_index + 1]
        h = x1 - x0
        t = (sample_x[mask] - x0) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        sample_y[mask] = (
            h00 * y0 + h10 * h * tangents[segment_index] +
            h01 * y1 + h11 * h * tangents[segment_index + 1]
        )
    return sample_x, np.clip(sample_y, 0, 255)


def image_process(image, points, use_spline=False, fit_spline=False):
    if use_spline:
        curve_x, curve_y = _spline_curve(points, fit_spline=fit_spline)
    else:
        curve_x, curve_y = _linear_curve(points)
    table = np.interp(np.arange(256), curve_x, curve_y).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image


class Node(DpgNodeABC):
    """Curves adjustment node."""

    _ver = "0.0.2"

    node_label = "Curves"
    node_tag = "Curves"

    _min_val = 0
    _max_val = 255
    _default_plot_width = 520
    _default_plot_height = 420
    _fine_step = 0.1

    _opencv_setting_dict = None

    def __init__(self):
        self._last_touched_point = {}

    def _get_tag_node_name(self, node_id):
        return f"{node_id}:{self.node_tag}"

    def _get_tag_plot_name(self, node_id):
        return f"{node_id}:{self.node_tag}:plot"

    def _get_tag_plot_series_name(self, node_id):
        return f"{node_id}:{self.node_tag}:line"

    def _get_tag_use_spline_name(self, node_id):
        return f"{node_id}:{self.node_tag}:use_spline"

    def _get_tag_fit_spline_name(self, node_id):
        return f"{node_id}:{self.node_tag}:fit_spline"

    def _get_curve_options(self, node_id):
        return (
            bool(dpg_get_value(self._get_tag_use_spline_name(node_id))),
            bool(dpg_get_value(self._get_tag_fit_spline_name(node_id))),
        )

    def _get_drag_points(self, node_id):
        plot_tag = self._get_tag_plot_name(node_id)
        point_items = dpg.get_item_children(plot_tag, slot=0)
        return _dedupe_points([dpg.get_value(tag) for tag in point_items])

    def _callback_add_point(self, sender, app_data, user_data):
        node_id = user_data[0]
        static_x = user_data[1]
        plot_tag = self._get_tag_plot_name(node_id)
        if static_x is not None:
            y = x = static_x
        else:
            x, y = dpg.get_plot_mouse_pos()
        point_tag = dpg.add_drag_point(
            parent=plot_tag,
            label="",
            default_value=[float(x), float(y)],
            delayed=False,
            callback=self._callback_moved_point,
            user_data=(node_id, static_x),
        )
        self._last_touched_point[node_id] = point_tag
        self._redraw_line(node_id)

    def _callback_moved_point(self, sender, app_data, user_data):
        node_id = user_data[0]
        static_x = user_data[1]
        self._last_touched_point[node_id] = sender
        x, y = dpg.get_value(sender)
        if static_x is not None:
            x = static_x
        y = float(np.clip(y, self._min_val, self._max_val))
        x = float(np.clip(x, self._min_val, self._max_val))
        dpg.set_value(sender, [x, y])
        self._redraw_line(node_id)

    def _callback_toggle_curve(self, sender, app_data, user_data):
        self._redraw_line(user_data)

    def _callback_nudge_point(self, sender, app_data, user_data):
        node_id, dx, dy = user_data
        point_tag = self._last_touched_point.get(node_id)
        if not point_tag or not dpg.does_item_exist(point_tag):
            return
        point_user_data = dpg.get_item_user_data(point_tag)
        static_x = point_user_data[1]
        x, y = dpg.get_value(point_tag)
        if static_x is None:
            x = float(np.clip(x + dx, self._min_val, self._max_val))
        y = float(np.clip(y + dy, self._min_val, self._max_val))
        dpg.set_value(point_tag, [x, y])
        self._redraw_line(node_id)

    def _redraw_line(self, node_id):
        points = self._get_drag_points(node_id)
        use_spline, fit_spline = self._get_curve_options(node_id)
        if use_spline:
            x, y = _spline_curve(points, fit_spline=fit_spline)
        else:
            x, y = _linear_curve(points)
        dpg.set_value(
            self._get_tag_plot_series_name(node_id),
            [x.tolist(), y.tolist()],
        )

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
                    width=self._default_plot_width,
                    height=self._default_plot_height,
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
                        parent=handler,
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

                dpg.add_checkbox(
                    label="Use spline interpolation",
                    tag=self._get_tag_use_spline_name(node_id),
                    default_value=False,
                    callback=self._callback_toggle_curve,
                    user_data=node_id,
                )
                dpg.add_checkbox(
                    label="Fit spline tangents",
                    tag=self._get_tag_fit_spline_name(node_id),
                    default_value=True,
                    callback=self._callback_toggle_curve,
                    user_data=node_id,
                )
                dpg.add_text(
                    "Arrow keys nudge the last touched point "
                    "(0.1 value units per key press)."
                )
                key_handler = dpg.add_handler_registry()
                dpg.add_key_press_handler(
                    key=dpg.mvKey_Left,
                    callback=self._callback_nudge_point,
                    user_data=(node_id, -self._fine_step, 0),
                    parent=key_handler,
                )
                dpg.add_key_press_handler(
                    key=dpg.mvKey_Right,
                    callback=self._callback_nudge_point,
                    user_data=(node_id, self._fine_step, 0),
                    parent=key_handler,
                )
                dpg.add_key_press_handler(
                    key=dpg.mvKey_Up,
                    callback=self._callback_nudge_point,
                    user_data=(node_id, 0, self._fine_step),
                    parent=key_handler,
                )
                dpg.add_key_press_handler(
                    key=dpg.mvKey_Down,
                    callback=self._callback_nudge_point,
                    user_data=(node_id, 0, -self._fine_step),
                    parent=key_handler,
                )
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
            use_spline, fit_spline = self._get_curve_options(node_id)
            frame = image_process(frame, points, use_spline, fit_spline)

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
            "use_spline": dpg_get_value(self._get_tag_use_spline_name(node_id)),
            "fit_spline": dpg_get_value(self._get_tag_fit_spline_name(node_id)),
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        dpg_set_value(
            self._get_tag_use_spline_name(node_id),
            setting_dict.get("use_spline", False),
        )
        dpg_set_value(
            self._get_tag_fit_spline_name(node_id),
            setting_dict.get("fit_spline", True),
        )
        plot_tag = self._get_tag_plot_name(node_id)
        existing_points = dpg.get_item_children(plot_tag, slot=0)
        for point_tag in existing_points:
            dpg.delete_item(point_tag)
        for pt in setting_dict.get("points", []):
            static_x = None
            if pt[0] in (self._min_val, self._max_val):
                static_x = pt[0]
            point_tag = dpg.add_drag_point(
                parent=plot_tag,
                label="",
                default_value=pt,
                callback=self._callback_moved_point,
                user_data=(node_id, static_x)
            )
            self._last_touched_point[node_id] = point_tag
        self._redraw_line(node_id)
