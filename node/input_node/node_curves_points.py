#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input node that emits editable Curves control points."""

import ast
import json

import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_item_children, dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node.port_model import OutputPort, PortDataType, PortSpecs


class Node(DpgNodeBase):
    _ver = '0.0.2'

    node_label = 'Curves Points'
    node_tag = 'CurvesPoints'

    port_specs = PortSpecs(
        value=OutputPort(PortDataType.TEXT),
    )

    _min_val = 0
    _max_val = 255
    _delete_hit_radius = 6

    def _get_tag_plot_name(self, node_id):
        return f'{self._node_name(node_id)}:plot'

    def _get_tag_plot_series_name(self, node_id):
        return f'{self._node_name(node_id)}:line'

    def _default_points(self):
        return [
            [self._min_val, self._min_val],
            [self._max_val, self._max_val],
        ]

    def _serialize_points(self, points):
        return json.dumps(points)

    def _parse_points(self, value):
        if value is None or value == '':
            return self._default_points()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    return self._default_points()
        if not isinstance(value, (list, tuple)):
            return self._default_points()

        points = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x = int(point[0])
                y = int(point[1])
            except (TypeError, ValueError):
                continue
            x = max(self._min_val, min(self._max_val, x))
            y = max(self._min_val, min(self._max_val, y))
            points.append([x, y])

        if len(points) < 2:
            return self._default_points()

        points = sorted(points)
        if points[0][0] != self._min_val:
            points.insert(0, [self._min_val, points[0][1]])
        if points[-1][0] != self._max_val:
            points.append([self._max_val, points[-1][1]])
        return points

    def _get_drag_points(self, node_id):
        plot_tag = self._get_tag_plot_name(node_id)
        if not dpg.does_item_exist(plot_tag):
            return self._default_points()

        points = []
        for point_tag in dpg_get_item_children(plot_tag, slot=0):
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

    def _redraw_line(self, node_id):
        points = self._get_drag_points(node_id)
        x_values, y_values = zip(*points)
        dpg.set_value(self._get_tag_plot_series_name(node_id), [x_values, y_values])

    def _set_output_value(self, node_id, points):
        value_tag = self.ports(node_id).value.value_tag
        dpg_set_value(value_tag, self._serialize_points(points))

    def _reset_points(self, node_id, setting_points):
        plot_tag = self._get_tag_plot_name(node_id)
        for point_tag in dpg_get_item_children(plot_tag, slot=0):
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
        self._set_output_value(node_id, self._get_drag_points(node_id))

    def _callback_add_point(self, sender, app_data, user_data):
        del sender, app_data
        node_id, static_x = user_data
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
        self._set_output_value(node_id, self._get_drag_points(node_id))

    def _callback_moved_point(self, sender, app_data, user_data):
        del app_data
        node_id, static_x = user_data
        point_value = dpg_get_value(sender)
        if not isinstance(point_value, (list, tuple)) or len(point_value) != 2:
            return

        x, y = point_value
        if static_x is not None:
            x = static_x
        y = max(self._min_val, min(self._max_val, int(y)))
        dpg.set_value(sender, [x, y])
        self._redraw_line(node_id)
        self._set_output_value(node_id, self._get_drag_points(node_id))

    def _callback_delete_point(self, sender, app_data, user_data):
        del sender, app_data
        node_id = user_data[0]
        plot_tag = self._get_tag_plot_name(node_id)
        mouse_x, mouse_y = dpg.get_plot_mouse_pos()
        closest_point_tag = None
        closest_distance_sq = float('inf')
        hit_radius_sq = self._delete_hit_radius**2

        for point_tag in dpg_get_item_children(plot_tag, slot=0):
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
            self._set_output_value(node_id, self._get_drag_points(node_id))

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        del callback
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        output_port = ports.value

        self._opencv_setting_dict = opencv_setting_dict

        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            with dpg.node_attribute(
                tag=output_port.dpg_tag,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_text(
                    tag=output_port.value_tag,
                    default_value=self._serialize_points(self._default_points()),
                    show=False,
                )
                plot_tag = self._get_tag_plot_name(node_id)
                series_tag = self._get_tag_plot_series_name(node_id)
                with dpg.plot(width=240, height=180, tag=plot_tag, no_menus=True):
                    dpg.add_plot_axis(dpg.mvXAxis, tag=f'{tag_node_name}:plot_x')
                    dpg.set_axis_limits(dpg.last_item(), self._min_val, self._max_val)
                    dpg.add_plot_axis(dpg.mvYAxis, tag=f'{tag_node_name}:plot_y')
                    dpg.set_axis_limits(dpg.last_item(), self._min_val, self._max_val)
                    dpg.add_line_series(
                        x=[self._min_val, self._max_val],
                        y=[self._min_val, self._max_val],
                        parent=f'{tag_node_name}:plot_y',
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

                self._reset_points(node_id, self._default_points())

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        del node_id, connection_list, node_image_dict, node_result_dict
        return None, None

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        output_value_tag = self.ports(node_id).value.value_tag
        points = self._get_drag_points(node_id)
        self._set_output_value(node_id, points)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = dpg.get_item_pos(tag_node_name)
        setting_dict[output_value_tag] = self._serialize_points(points)

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        output_value_tag = self.ports(node_id).value.value_tag
        points = self._parse_points(setting_dict.get(output_value_tag))
        self._reset_points(node_id, points)
