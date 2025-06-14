#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value, convert_cv_to_dpg
from node.node_abc import DpgNodeABC


class Node(DpgNodeABC):

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
        x, y = app_data
        # clip to 0-255 range
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
        """Rebuild the curve line based on current drag points."""
        points = [dpg_get_value(tag) for tag in self._drag_points.get(node_id, [])]
        points.append([0, 0])
        points.append([255, 255])
        # sort by x value
        points = sorted(points, key=lambda p: p[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        series_tag = self._line_series.get(node_id)
        if series_tag is None:
            series_tag = dpg.add_line_series(x, y, parent=f"{plot_tag}_y")
            self._line_series[node_id] = series_tag
        else:
            dpg_set_value(series_tag, [x, y])

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        # tag names
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # initial black image
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

        # texture registration
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # node
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # input port
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )
            # image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # curve editor
            with dpg.node_attribute(
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                with dpg.plot(
                        width=240,
                        height=240,
                        tag=f'{tag_node_name}:plot',
                        no_menus=True,
                ):
                    dpg.add_plot_axis(dpg.mvXAxis, tag=f'{tag_node_name}:plot_x')
                    dpg.set_axis_limits(dpg.last_item(), 0, 255)
                    dpg.add_plot_axis(dpg.mvYAxis, tag=f'{tag_node_name}:plot_y')
                    dpg.set_axis_limits(dpg.last_item(), 0, 255)
                    self._line_series[node_id] = dpg.add_line_series(
                        [0, 255],
                        [0, 255],
                        parent=f'{tag_node_name}:plot_y',
                        tag=f'{tag_node_name}:line',
                    )
                    handler = dpg.add_item_handler_registry()
                    dpg.add_item_clicked_handler(
                        callback=self._callback_add_point,
                        user_data=(node_id, f'{tag_node_name}:plot'),
                    )
                    dpg.bind_item_handler_registry(f'{tag_node_name}:plot', handler)
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
        node_result_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        plot_tag = f'{tag_node_name}:plot'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # connection check
        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_IMAGE:
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)

        # get image
        frame = node_image_dict.get(connection_info_src, None)

        # build LUT from drag points
        points = [[0, 0], [255, 255]]
        for tag in self._drag_points.get(node_id, []):
            pt = dpg_get_value(tag)
            pt[0] = max(0, min(255, pt[0]))
            pt[1] = max(0, min(255, pt[1]))
            points.append(pt)
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        table = np.interp(np.arange(256), xs, ys).astype(np.uint8)

        # start timer
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frame is not None:
            result = cv2.LUT(frame, table)

        # stop timer
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag, str(elapsed_time).zfill(4) + 'ms')

        if frame is not None:
            texture = convert_cv_to_dpg(
                result,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)
            self._redraw_line(node_id, plot_tag)
            return result, None

        return None, None

    def close(self, node_id):
        # clean up stored points
        self._drag_points.pop(node_id, None)
        self._line_series.pop(node_id, None)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag

        pos = dpg.get_item_pos(tag_node_name)
        point_list = [dpg_get_value(tag) for tag in self._drag_points.get(node_id, [])]

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict['points'] = point_list

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        plot_tag = f'{node_id}:{self.node_tag}:plot'
        self._drag_points[node_id] = []

        for pt in setting_dict.get('points', []):
            drag_tag = dpg.add_drag_point(
                parent=plot_tag,
                label='',
                default_value=pt,
                callback=self._callback_moved_point,
                user_data=node_id,
            )
            self._drag_points[node_id].append(drag_tag)

        self._redraw_line(node_id, plot_tag)
