#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves adjustment node for Image Processing Node Editor."""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node_editor.util import dpg_get_value, dpg_get_item_children


def image_process(image, points):
    # Builds LUT from curves points
    # Assumes points are pre-sorted and between 0 and 255
    xs, ys = zip(*points)
    table = np.interp(np.arange(256), xs, ys).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image


class Node(DeclarativeImageProcessNodeBase):
    """Curves adjustment node."""

    _ver = '0.0.2'

    node_label = 'Curves'
    node_tag = 'Curves'

    _min_val = 0
    _max_val = 255
    _delete_hit_radius = 6

    def _get_tag_plot_name(self, node_id):
        return f'{self._node_name(node_id)}:plot'

    def _get_tag_plot_series_name(self, node_id):
        return f'{self._node_name(node_id)}:line'

    def _get_drag_points(self, node_id):
        plot_tag = self._get_tag_plot_name(node_id)
        if not dpg.does_item_exist(plot_tag):
            return self._default_points()

        point_items = dpg_get_item_children(plot_tag, slot=0)
        points = []
        for point_tag in point_items:
            value = dpg_get_value(point_tag)
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and value[0] is not None
                and value[1] is not None
            ):
                points.append([value[0], value[1]])

        if len(points) < 2:
            return self._default_points()

        return sorted(points)

    def _default_points(self):
        return [
            [self._min_val, self._min_val],
            [self._max_val, self._max_val],
        ]

    def _callback_add_point(self, sender, app_data, user_data):
        del sender, app_data
        node_id, static_x = user_data
        before_points = self._get_drag_points(node_id)
        plot_tag = self._get_tag_plot_name(node_id)
        if static_x is not None:
            y = x = static_x
        else:
            x, y = dpg.get_plot_mouse_pos()

        dpg.add_drag_point(
            parent=plot_tag,
            label='',
            default_value=[x, y],
            delayed=True,
            callback=self._callback_moved_point,
            user_data=(node_id, static_x),
        )
        self._redraw_line(node_id)
        self._emit_points_changed(node_id, before_points, self._get_drag_points(node_id))

    def _callback_moved_point(self, sender, app_data, user_data):
        del app_data
        node_id, static_x = user_data
        before_points = self._get_drag_points(node_id)

        point_value = dpg_get_value(sender)
        if not isinstance(point_value, (list, tuple)) or len(point_value) != 2:
            return

        x, y = point_value
        if static_x is not None:
            x = static_x
        y = max(self._min_val, min(self._max_val, int(y)))
        dpg.set_value(sender, [x, y])
        self._redraw_line(node_id)
        self._emit_points_changed(node_id, before_points, self._get_drag_points(node_id))

    def _callback_delete_point(self, sender, app_data, user_data):
        del sender, app_data
        node_id = user_data[0]
        before_points = self._get_drag_points(node_id)
        plot_tag = self._get_tag_plot_name(node_id)
        point_items = dpg_get_item_children(plot_tag, slot=0)
        mouse_x, mouse_y = dpg.get_plot_mouse_pos()

        closest_point_tag = None
        closest_distance_sq = float('inf')
        hit_radius_sq = self._delete_hit_radius**2

        for point_tag in point_items:
            point_user_data = dpg.get_item_user_data(point_tag)
            if point_user_data is None:
                continue

            _, static_x = point_user_data
            if static_x is not None:
                continue

            point_value = dpg_get_value(point_tag)
            if not isinstance(point_value, (list, tuple)) or len(point_value) != 2:
                continue

            px, py = point_value
            distance_sq = ((px - mouse_x) ** 2) + ((py - mouse_y) ** 2)
            if distance_sq > hit_radius_sq:
                continue

            if distance_sq < closest_distance_sq:
                closest_distance_sq = distance_sq
                closest_point_tag = point_tag

        if closest_point_tag is not None:
            dpg.delete_item(closest_point_tag)
            self._redraw_line(node_id)
            self._emit_points_changed(
                node_id,
                before_points,
                self._get_drag_points(node_id),
            )

    def _redraw_line(self, node_id):
        points = self._get_drag_points(node_id)
        x_values, y_values = zip(*points)
        series_tag = self._get_tag_plot_series_name(node_id)
        dpg.set_value(series_tag, [x_values, y_values])

    def _reset_points_from_setting(self, node_id, setting_points):
        plot_tag = self._get_tag_plot_name(node_id)
        existing_points = dpg_get_item_children(plot_tag, slot=0)
        for point_tag in existing_points:
            dpg.delete_item(point_tag)

        points_to_add = setting_points if setting_points else self._default_points()

        for point in points_to_add:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x, y = int(point[0]), int(point[1])
            y = max(self._min_val, min(self._max_val, y))
            static_x = x if x in [self._min_val, self._max_val] else None
            dpg.add_drag_point(
                parent=plot_tag,
                label='',
                default_value=[x, y],
                delayed=True,
                callback=self._callback_moved_point,
                user_data=(node_id, static_x),
            )

        self._redraw_line(node_id)

    def _emit_points_changed(self, node_id, before_points, after_points):
        if self._ui_callback is None:
            return
        if before_points == after_points:
            return
        node_id_name = self._node_name(node_id)
        value_tag = f'{node_id_name}:Text:CurvesPointsValue'
        self._ui_callback(
            'parameter_changed',
            {
                'node_id_name': node_id_name,
                'port_tag': f'{node_id_name}:Text:CurvesPoints',
                'value_tag': value_tag,
                'before_value': before_points,
                'after_value': after_points,
            },
        )

    def build_custom_ui(self, tag_node_name, node_id, width, callback):
        del tag_node_name, width, callback

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            plot_tag = self._get_tag_plot_name(node_id)
            series_tag = self._get_tag_plot_series_name(node_id)
            with dpg.plot(width=240, height=180, tag=plot_tag, no_menus=True):
                dpg.add_plot_axis(dpg.mvXAxis, tag=f'{self._node_name(node_id)}:plot_x')
                dpg.set_axis_limits(dpg.last_item(), self._min_val, self._max_val)
                dpg.add_plot_axis(dpg.mvYAxis, tag=f'{self._node_name(node_id)}:plot_y')
                dpg.set_axis_limits(dpg.last_item(), self._min_val, self._max_val)
                dpg.add_line_series(
                    x=[self._min_val, self._max_val],
                    y=[self._min_val, self._max_val],
                    parent=f'{self._node_name(node_id)}:plot_y',
                    tag=series_tag,
                )
                handler = dpg.add_item_handler_registry()
                dpg.add_item_clicked_handler(
                    callback=self._callback_add_point,
                    user_data=(node_id, None),
                    button=dpg.mvMouseButton_Left,
                    parent=handler,
                )
                dpg.add_item_clicked_handler(
                    callback=self._callback_delete_point,
                    user_data=(node_id,),
                    button=dpg.mvMouseButton_Right,
                    parent=handler,
                )
                dpg.bind_item_handler_registry(plot_tag, handler)

            self._reset_points_from_setting(node_id, self._default_points())

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        node_id = int(str(tag_node_name).split(':', maxsplit=1)[0])
        parameter_values['points'] = self._get_drag_points(node_id)
        return parameter_values

    def process(self, frame, **parameter_values):
        frame = image_process(frame, parameter_values['points'])
        return frame, None

    def get_custom_setting_dict(self, tag_node_name, node_id):
        del tag_node_name
        return {'points': self._get_drag_points(node_id)}

    def set_custom_setting_dict(self, tag_node_name, node_id, setting_dict):
        del tag_node_name
        self._reset_points_from_setting(node_id, setting_dict.get('points', []))
