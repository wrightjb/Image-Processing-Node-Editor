#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared DearPyGui curve-points editor helpers."""

import ast
import json

import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_item_children, dpg_get_value, dpg_set_value

CURVE_CHANNELS = ('White', 'Red', 'Green', 'Blue')
CURVE_CHANNEL_COLORS = {
    'White': (235, 235, 235, 255),
    'Red': (255, 80, 80, 255),
    'Green': (80, 220, 100, 255),
    'Blue': (80, 140, 255, 255),
}
CURVE_GHOST_COLORS = {
    'White': (180, 180, 180, 80),
    'Red': (255, 80, 80, 70),
    'Green': (80, 220, 100, 70),
    'Blue': (80, 140, 255, 70),
}


class CurvesPointsEditorMixin:
    _min_val = 0
    _max_val = 255
    _delete_hit_radius = 6
    _plot_width = 520
    _plot_height = 360
    _keyboard_nudge_step = 0.1
    _last_touched_point_by_node = {}
    _curve_sets_by_node = {}
    _active_curve_channel_by_node = {}
    _curve_editor_built_by_node = set()

    def _get_tag_plot_name(self, node_id):
        return f'{self._node_name(node_id)}:plot'

    def _get_tag_plot_series_name(self, node_id):
        return f'{self._node_name(node_id)}:line'

    def _get_tag_plot_channel_series_name(self, node_id, channel):
        return f'{self._node_name(node_id)}:line:{channel}'

    def _get_tag_points_display_name(self, node_id):
        return f'{self._node_name(node_id)}:points_display'

    def _get_tag_active_channel_name(self, node_id):
        return f'{self._node_name(node_id)}:active_channel'

    def _remember_touched_point(self, node_id, point_tag):
        self._last_touched_point_by_node[str(node_id)] = point_tag

    def _default_points(self):
        return [[self._min_val, self._min_val], [self._max_val, self._max_val]]

    def _default_curve_set(self):
        return {channel: self._default_points() for channel in CURVE_CHANNELS}

    def _serialize_points(self, points):
        return json.dumps(points, indent=2)

    def _serialize_curve_set(self, curve_set):
        return json.dumps({'curves': self._normalize_curve_set(curve_set)}, indent=2)

    def _decode_serialized_value(self, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    return value
        return value

    def _parse_points(self, value):
        if value is None or value == '':
            return self._default_points()
        value = self._decode_serialized_value(value)
        if isinstance(value, dict):
            value = value.get('points')
        if not isinstance(value, (list, tuple)):
            return self._default_points()

        points = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
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

    def _normalize_curve_set(self, value):
        value = self._decode_serialized_value(value)
        curve_set = self._default_curve_set()
        if isinstance(value, dict):
            curves_value = value.get('curves')
            if isinstance(curves_value, dict):
                for channel in CURVE_CHANNELS:
                    if channel in curves_value:
                        curve_set[channel] = self._parse_points(curves_value[channel])
                return curve_set
            if all(channel in value for channel in CURVE_CHANNELS):
                for channel in CURVE_CHANNELS:
                    curve_set[channel] = self._parse_points(value[channel])
                return curve_set
            if 'points' in value:
                channel = value.get('channel', 'White')
                if channel not in CURVE_CHANNELS:
                    channel = 'White'
                curve_set[channel] = self._parse_points(value.get('points'))
                return curve_set
        if isinstance(value, (list, tuple)) or value in (None, ''):
            curve_set['White'] = self._parse_points(value)
        return curve_set

    def _active_channel(self, node_id):
        return self._active_curve_channel_by_node.get(str(node_id), 'White')

    def _set_active_channel(self, node_id, channel):
        if channel not in CURVE_CHANNELS:
            channel = 'White'
        self._active_curve_channel_by_node[str(node_id)] = channel
        if str(node_id) in self._curve_editor_built_by_node:
            dpg_set_value(self._get_tag_active_channel_name(node_id), channel)

    def _curve_set(self, node_id):
        return self._curve_sets_by_node.setdefault(str(node_id), self._default_curve_set())

    def _set_curve_set(self, node_id, curve_set):
        self._curve_sets_by_node[str(node_id)] = self._normalize_curve_set(curve_set)

    def _get_drag_points(self, node_id):
        plot_tag = self._get_tag_plot_name(node_id)
        if str(node_id) not in self._curve_editor_built_by_node:
            return self._curve_set(node_id).get(
                self._active_channel(node_id),
                self._default_points(),
            )

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

    def _store_active_points(self, node_id, points):
        curve_set = self._curve_set(node_id)
        curve_set[self._active_channel(node_id)] = points
        return curve_set

    def _line_values(self, points):
        x_values, y_values = zip(*points)
        return [x_values, y_values]

    def _set_line_value_if_exists(self, tag, points):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, self._line_values(points))

    def _redraw_line(self, node_id):
        active_channel = self._active_channel(node_id)
        points = self._get_drag_points(node_id)
        curve_set = self._curve_set(node_id)
        curve_set[active_channel] = points
        if str(node_id) not in self._curve_editor_built_by_node:
            return
        active_line_tag = self._get_tag_plot_series_name(node_id)
        self._bind_line_theme(active_line_tag, CURVE_CHANNEL_COLORS[active_channel])
        self._set_line_value_if_exists(active_line_tag, points)
        for channel in CURVE_CHANNELS:
            self._set_line_value_if_exists(
                self._get_tag_plot_channel_series_name(node_id, channel),
                curve_set[channel],
            )
        display_tag = self._get_tag_points_display_name(node_id)
        if dpg.does_item_exist(display_tag):
            dpg.set_value(display_tag, self._serialize_curve_set(curve_set))

    def _delete_drag_points(self, node_id):
        if str(node_id) not in self._curve_editor_built_by_node:
            return
        plot_tag = self._get_tag_plot_name(node_id)
        for point_tag in dpg_get_item_children(plot_tag, slot=0):
            dpg.delete_item(point_tag)

    def _load_active_drag_points(self, node_id, points):
        if str(node_id) not in self._curve_editor_built_by_node:
            return
        plot_tag = self._get_tag_plot_name(node_id)
        self._delete_drag_points(node_id)
        points_to_add = points if points else self._default_points()
        for point in points_to_add:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x, y = float(point[0]), float(point[1])
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

    def _reset_points_from_setting(self, node_id, setting_points):
        curve_set = self._normalize_curve_set(setting_points)
        self._set_curve_set(node_id, curve_set)
        active_channel = self._active_channel(node_id)
        self._load_active_drag_points(node_id, curve_set[active_channel])
        self._redraw_line(node_id)
        self._on_points_reset(node_id, curve_set)

    def _on_points_reset(self, node_id, curve_set):
        self._on_points_changed(node_id, curve_set)

    def _on_points_changed(self, node_id, curve_set):
        del node_id, curve_set

    def _callback_add_point(self, sender, app_data, user_data):
        del sender, app_data
        node_id, static_x = user_data
        before_points = self._get_drag_points(node_id)
        plot_tag = self._get_tag_plot_name(node_id)
        if static_x is not None:
            y = x = static_x
        else:
            x, y = dpg.get_plot_mouse_pos()

        point_tag = dpg.add_drag_point(
            parent=plot_tag,
            label='',
            default_value=[x, y],
            delayed=True,
            callback=self._callback_moved_point,
            user_data=(node_id, static_x),
        )
        self._remember_touched_point(node_id, point_tag)
        self._redraw_line(node_id)
        points = self._get_drag_points(node_id)
        self._on_points_changed(node_id, self._curve_set(node_id))
        self._emit_points_changed(node_id, before_points, points, coalesce=False)

    def _callback_moved_point(self, sender, app_data, user_data):
        del app_data
        node_id, static_x = user_data
        self._remember_touched_point(node_id, sender)
        before_points = self._get_drag_points(node_id)
        point_value = dpg_get_value(sender)
        if not isinstance(point_value, (list, tuple)) or len(point_value) != 2:
            return

        x, y = point_value
        if static_x is None and (
            x < self._min_val
            or x > self._max_val
            or y < self._min_val
            or y > self._max_val
        ):
            dpg.delete_item(sender)
            self._last_touched_point_by_node.pop(str(node_id), None)
            self._redraw_line(node_id)
            points = self._get_drag_points(node_id)
            self._on_points_changed(node_id, self._store_active_points(node_id, points))
            self._emit_points_changed(
                node_id,
                before_points,
                points,
                coalesce=False,
            )
            return

        if static_x is not None:
            x = static_x
        y = max(self._min_val, min(self._max_val, float(y)))
        dpg.set_value(sender, [x, y])
        self._redraw_line(node_id)
        points = self._get_drag_points(node_id)
        self._on_points_changed(node_id, self._store_active_points(node_id, points))
        self._emit_points_changed(node_id, before_points, points, coalesce=True)

    def _callback_nudge_last_point(self, sender, app_data, user_data):
        del sender
        node_id, dx, dy = user_data
        point_tag = self._last_touched_point_by_node.get(str(node_id))
        if not point_tag or not dpg.does_item_exist(point_tag):
            return
        before_points = self._get_drag_points(node_id)
        point_value = dpg_get_value(point_tag)
        if not isinstance(point_value, (list, tuple)) or len(point_value) != 2:
            return
        point_user_data = dpg.get_item_user_data(point_tag)
        static_x = point_user_data[1] if point_user_data is not None else None
        x = static_x if static_x is not None else float(point_value[0]) + dx
        y = float(point_value[1]) + dy
        if static_x is None:
            x = max(self._min_val, min(self._max_val, x))
        y = max(self._min_val, min(self._max_val, y))
        dpg.set_value(point_tag, [x, y])
        self._redraw_line(node_id)
        points = self._get_drag_points(node_id)
        self._on_points_changed(node_id, self._store_active_points(node_id, points))
        self._emit_points_changed(node_id, before_points, points, coalesce=False)

    def _callback_delete_point(self, sender, app_data, user_data):
        del sender, app_data
        node_id = user_data[0]
        before_points = self._get_drag_points(node_id)
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
            if self._last_touched_point_by_node.get(str(node_id)) == closest_point_tag:
                self._last_touched_point_by_node.pop(str(node_id), None)
            self._redraw_line(node_id)
            points = self._get_drag_points(node_id)
            self._on_points_changed(node_id, self._store_active_points(node_id, points))
            self._emit_points_changed(node_id, before_points, points, coalesce=False)


    def _callback_clear_channel(self, sender, app_data, user_data):
        del sender, app_data
        node_id = user_data
        curve_set = self._curve_set(node_id)
        active_channel = self._active_channel(node_id)
        curve_set[active_channel] = self._default_points()
        self._load_active_drag_points(node_id, curve_set[active_channel])
        self._redraw_line(node_id)
        self._on_points_changed(node_id, curve_set)

    def _callback_clear_all(self, sender, app_data, user_data):
        del sender, app_data
        node_id = user_data
        curve_set = self._default_curve_set()
        self._set_curve_set(node_id, curve_set)
        self._load_active_drag_points(node_id, curve_set[self._active_channel(node_id)])
        self._redraw_line(node_id)
        self._on_points_changed(node_id, curve_set)

    def _callback_channel_changed(self, sender, app_data, user_data):
        del sender
        node_id = user_data
        previous_channel = self._active_channel(node_id)
        curve_set = self._curve_set(node_id)
        curve_set[previous_channel] = self._get_drag_points(node_id)
        channel = app_data if app_data in CURVE_CHANNELS else 'White'
        self._set_active_channel(node_id, channel)
        self._load_active_drag_points(node_id, curve_set[channel])
        self._redraw_line(node_id)
        self._on_points_changed(node_id, curve_set)

    def _emit_points_changed(self, node_id, before_points, after_points, coalesce=False):
        del node_id, before_points, after_points, coalesce

    def _export_dialog_tag(self, node_id):
        return f'{self._node_name(node_id)}:CurvesPointsExportDialog'

    def _import_dialog_tag(self, node_id):
        return f'{self._node_name(node_id)}:CurvesPointsImportDialog'

    def _callback_copy_points(self, sender, app_data, user_data):
        del sender, app_data
        self._redraw_line(user_data)
        dpg.set_clipboard_text(self._serialize_curve_set(self._curve_set(user_data)))

    def _callback_show_export_dialog(self, sender, app_data, user_data):
        del sender, app_data
        dpg.show_item(self._export_dialog_tag(user_data))

    def _callback_show_import_dialog(self, sender, app_data, user_data):
        del sender, app_data
        dpg.show_item(self._import_dialog_tag(user_data))

    def _callback_export_points(self, sender, app_data, user_data):
        del sender
        node_id = user_data
        file_path = app_data.get('file_path_name') if isinstance(app_data, dict) else None
        if not file_path:
            return
        self._redraw_line(node_id)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump({'curves': self._curve_set(node_id)}, file, indent=2)

    def _callback_import_points(self, sender, app_data, user_data):
        del sender
        node_id = user_data
        file_path = app_data.get('file_path_name') if isinstance(app_data, dict) else None
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return
        self._reset_points_from_setting(node_id, payload)

    def build_curve_points_file_dialogs(self, node_id):
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            callback=self._callback_export_points,
            tag=self._export_dialog_tag(node_id),
            user_data=node_id,
        ):
            dpg.add_file_extension('Curves (*.json){.json}')
            dpg.add_file_extension('', color=(150, 255, 150, 255))
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            callback=self._callback_import_points,
            tag=self._import_dialog_tag(node_id),
            user_data=node_id,
        ):
            dpg.add_file_extension('Curves (*.json){.json}')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

    def _bind_line_theme(self, item, color):
        try:
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        color,
                        category=dpg.mvThemeCat_Plots,
                    )
            dpg.bind_item_theme(item, theme)
        except Exception:
            return

    def _add_curve_line_series(self, node_id, channel, parent, active=False):
        tag = (
            self._get_tag_plot_series_name(node_id)
            if active
            else self._get_tag_plot_channel_series_name(node_id, channel)
        )
        dpg.add_line_series(
            x=[self._min_val, self._max_val],
            y=[self._min_val, self._max_val],
            parent=parent,
            tag=tag,
        )
        color = CURVE_CHANNEL_COLORS[channel] if active else CURVE_GHOST_COLORS[channel]
        self._bind_line_theme(tag, color)

    def begin_curve_points_editor(self, node_id):
        self._curve_editor_built_by_node.add(str(node_id))
        self._set_active_channel(node_id, self._active_channel(node_id))

    def build_curve_points_channel_selector(self, node_id):
        dpg.add_combo(
            list(CURVE_CHANNELS),
            label='Edit Channel',
            tag=self._get_tag_active_channel_name(node_id),
            default_value=self._active_channel(node_id),
            width=160,
            callback=self._callback_channel_changed,
            user_data=node_id,
        )

    def build_curve_points_plot_controls(self, node_id):
        plot_tag = self._get_tag_plot_name(node_id)
        y_axis_tag = f'{self._node_name(node_id)}:plot_y'
        with dpg.plot(
            width=self._plot_width,
            height=self._plot_height,
            tag=plot_tag,
            no_menus=True,
        ):
            dpg.add_plot_axis(dpg.mvXAxis, tag=f'{self._node_name(node_id)}:plot_x')
            dpg.set_axis_limits(dpg.last_item(), self._min_val, self._max_val)
            dpg.add_plot_axis(dpg.mvYAxis, tag=y_axis_tag)
            dpg.set_axis_limits(dpg.last_item(), self._min_val, self._max_val)
            for channel in CURVE_CHANNELS:
                self._add_curve_line_series(node_id, channel, y_axis_tag, active=False)
            self._add_curve_line_series(
                node_id,
                self._active_channel(node_id),
                y_axis_tag,
                active=True,
            )
            handler = dpg.add_item_handler_registry()
            dpg.add_item_clicked_handler(
                callback=self._callback_add_point,
                user_data=(node_id, None),
                button=dpg.mvMouseButton_Left,
                parent=handler,
            )
            for key, delta in (
                (dpg.mvKey_Left, (-self._keyboard_nudge_step, 0.0)),
                (dpg.mvKey_Right, (self._keyboard_nudge_step, 0.0)),
                (dpg.mvKey_Up, (0.0, self._keyboard_nudge_step)),
                (dpg.mvKey_Down, (0.0, -self._keyboard_nudge_step)),
            ):
                dpg.add_key_press_handler(
                    key=key,
                    callback=self._callback_nudge_last_point,
                    user_data=(node_id, delta[0], delta[1]),
                    parent=handler,
                )
            dpg.add_item_clicked_handler(
                callback=self._callback_delete_point,
                user_data=(node_id,),
                button=dpg.mvMouseButton_Right,
                parent=handler,
            )
            dpg.bind_item_handler_registry(plot_tag, handler)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label='Import',
                width=58,
                callback=self._callback_show_import_dialog,
                user_data=node_id,
            )
            dpg.add_button(
                label='Export',
                width=58,
                callback=self._callback_show_export_dialog,
                user_data=node_id,
            )
            dpg.add_button(
                label='Copy Curves',
                width=112,
                callback=self._callback_copy_points,
                user_data=node_id,
            )
        with dpg.group(horizontal=True):
            dpg.add_button(
                label='Clear Channel',
                width=128,
                callback=self._callback_clear_channel,
                user_data=node_id,
            )
            dpg.add_button(
                label='Clear All',
                width=82,
                callback=self._callback_clear_all,
                user_data=node_id,
            )
        dpg.add_text(
            '',
            tag=self._get_tag_points_display_name(node_id),
            show=False,
        )
        self._reset_points_from_setting(node_id, self._default_curve_set())

    def build_curve_points_editor(self, node_id):
        self.begin_curve_points_editor(node_id)
        self.build_curve_points_channel_selector(node_id)
        self.build_curve_points_plot_controls(node_id)
