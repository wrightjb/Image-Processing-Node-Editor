#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from auto_tune.curves import DEFAULT_MAX_POINTS, DEFAULT_REFINEMENT_ITERATIONS
from auto_tune.curves import tune_curve_set, tune_curves
from node.curves_points_ui import CURVE_CHANNELS, CurvesPointsEditorMixin
from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs
from node_editor.image_metadata import match_metadata_compression
from node_editor.util import dpg_get_value, dpg_set_value


class Node(DpgNodeBase):
    _ver = '0.0.3'

    def __init__(self):
        self._run_requested_node_ids = set()

    node_label = 'Auto Tune (Curves)'
    node_tag = 'AutoTuneCurves'

    port_specs = PortSpecs(
        source_image=InputPort(PortDataType.IMAGE, index=1),
        target_image=InputPort(PortDataType.IMAGE, index=2),
        points=OutputPort(PortDataType.CURVE_POINTS, index=1),
        best_score=OutputPort(PortDataType.FLOAT, index=2),
    )

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        source_image_port = ports.source_image
        target_image_port = ports.target_image
        points_port = ports.points
        best_score_port = ports.best_score
        source_image = source_image_port.dpg_tag
        target_image = target_image_port.dpg_tag
        points = points_port.dpg_tag
        best_score = best_score_port.dpg_tag
        self._opencv_setting_dict = opencv_setting_dict

        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            self.add_editor_toolbar(
                node_id,
                callback=callback,
                build_extra_controls=lambda: dpg.add_button(
                    label='Run Tune',
                    width=80,
                    callback=self._on_run_button,
                    user_data=node_id,
                ),
            )
            with dpg.node_attribute(
                tag=self._status_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_text('idle', tag=self._status_value_tag(node_id))
            with dpg.node_attribute(
                tag=self._max_points_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_int(
                    tag=self._max_points_value_tag(node_id),
                    label='Max Points',
                    default_value=DEFAULT_MAX_POINTS,
                    min_value=2,
                    min_clamped=True,
                    width=120,
                    callback=callback,
                )
            with dpg.node_attribute(
                tag=self._metric_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    ('balanced_huber', 'balanced_mae', 'mae', 'mse'),
                    label='Metric',
                    tag=self._metric_value_tag(node_id),
                    default_value='balanced_huber',
                    width=140,
                    callback=callback,
                )
            with dpg.node_attribute(
                tag=self._refinement_iterations_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_int(
                    tag=self._refinement_iterations_value_tag(node_id),
                    label='Refine Rounds',
                    default_value=DEFAULT_REFINEMENT_ITERATIONS,
                    min_value=0,
                    min_clamped=True,
                    width=120,
                    callback=callback,
                )
            with dpg.node_attribute(
                tag=self._channel_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    ['All'] + list(CURVE_CHANNELS),
                    label='Tune',
                    tag=self._channel_value_tag(node_id),
                    default_value='All',
                    width=140,
                    callback=callback,
                )
            with dpg.node_attribute(
                tag=self._interpolation_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    ('linear', 'spline'),
                    label='Interpolation',
                    tag=self._interpolation_value_tag(node_id),
                    default_value='linear',
                    width=140,
                    callback=callback,
                )
            with dpg.node_attribute(
                tag=self._match_compression_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    label='Match target compression',
                    tag=self._match_compression_value_tag(node_id),
                    default_value=False,
                    callback=callback,
                )
            with dpg.node_attribute(
                tag=source_image,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text('source image')
            with dpg.node_attribute(
                tag=target_image,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text('target image')
            with dpg.node_attribute(
                tag=points,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_text(
                    tag=points_port.value_tag,
                    default_value=CurvesPointsEditorMixin()._serialize_curve_set(
                        CurvesPointsEditorMixin()._default_curve_set(),
                    ),
                    show=False,
                )
                dpg.add_button(
                    label='Copy Points',
                    width=120,
                    callback=self._copy_points_to_clipboard,
                    user_data=points_port.value_tag,
                )
            with dpg.node_attribute(
                tag=best_score,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_float(
                    tag=best_score_port.value_tag,
                    label='score',
                    default_value=0.0,
                    width=120,
                    readonly=True,
                )

        return tag_node_name

    def _channel_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Channel')

    def _channel_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Channel')

    def _channel_value(self, node_id):
        value = dpg_get_value(self._channel_value_tag(node_id))
        if value == 'All' or value in CURVE_CHANNELS:
            return value
        return 'All'

    def _interpolation_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Interpolation')

    def _interpolation_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Interpolation')

    def _interpolation_value(self, node_id):
        value = dpg_get_value(self._interpolation_value_tag(node_id))
        return 'spline' if value == 'spline' else 'linear'

    def _match_compression_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'MatchCompression')

    def _match_compression_value_tag(self, node_id):
        return self._node_control_value_tag(
            node_id,
            self.TYPE_TEXT,
            'MatchCompression',
        )

    def _copy_points_to_clipboard(self, sender, app_data, user_data):
        del sender, app_data
        dpg.set_clipboard_text(str(dpg_get_value(user_data) or ''))

    def _metric_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Metric')

    def _metric_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Metric')

    def _max_points_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'MaxPoints')

    def _max_points_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'MaxPoints')

    def _refinement_iterations_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'RefineRounds')

    def _refinement_iterations_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'RefineRounds')

    def _status_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Status')

    def _status_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Status')

    def _set_status(self, node_id, message):
        dpg_set_value(self._status_value_tag(node_id), message)

    def _on_run_button(self, sender, app_data, user_data):
        del sender, app_data
        print(f'AutoTuneCurves: Run Tune requested for node {user_data}.')
        self._run_requested_node_ids.add(str(user_data))
        self._set_status(user_data, 'queued')

    def _linked_image_result(
        self,
        port_ref,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        for (
            connection_info,
            source_tag,
            destination_tag,
            _connection_type,
        ) in self._iter_connection_infos(connection_list):
            if destination_tag != port_ref.dpg_tag:
                continue
            source_node_key = self._connection_source_node_key(
                connection_info,
                source_tag,
            )
            return (
                node_image_dict.get(source_node_key),
                node_result_dict.get(source_node_key),
            )
        return None, None

    def _linked_image(self, port_ref, connection_list, node_image_dict):
        for (
            connection_info,
            source_tag,
            destination_tag,
            _connection_type,
        ) in self._iter_connection_infos(connection_list):
            if destination_tag != port_ref.dpg_tag:
                continue
            source_node_key = self._connection_source_node_key(
                connection_info,
                source_tag,
            )
            return node_image_dict.get(source_node_key)
        return None

    def _int_setting(self, tag, fallback, minimum):
        value = dpg_get_value(tag)
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return fallback

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        node_id_key = str(node_id)
        if node_id_key not in self._run_requested_node_ids:
            return None, {'__auto_tune_ready__': False}
        self._run_requested_node_ids.discard(node_id_key)

        ports = self.ports(node_id)
        source = self._linked_image(ports.source_image, connection_list, node_image_dict)
        target, target_result = self._linked_image_result(
            ports.target_image,
            connection_list,
            node_image_dict,
            node_result_dict,
        )
        if source is None or target is None:
            self._set_status(node_id, 'missing source/target')
            return None, {'__auto_tune_ready__': False}

        max_points = self._int_setting(
            self._max_points_value_tag(node_id),
            DEFAULT_MAX_POINTS,
            2,
        )
        refinement_iterations = self._int_setting(
            self._refinement_iterations_value_tag(node_id),
            DEFAULT_REFINEMENT_ITERATIONS,
            0,
        )
        metric_name = dpg_get_value(self._metric_value_tag(node_id)) or 'balanced_huber'
        dpg_set_value(self._max_points_value_tag(node_id), max_points)
        dpg_set_value(self._refinement_iterations_value_tag(node_id), refinement_iterations)
        self._set_status(node_id, 'running')

        def _progress(update):
            if update.get('phase') == 'refine':
                message = (
                    f"refine round {update['round_index']} "
                    f"radius={update['radius']}\n"
                    f"points={update['point_count']} "
                    f"score={update['score']:.6g}"
                )
            elif update.get('phase') == 'spline_add':
                message = (
                    f"spline add\n"
                    f"points={update['point_count']} "
                    f"bins={update['observed_bins']}\n"
                    f"score={update['score']:.6g}"
                )
            elif update.get('phase') == 'spline_inflection':
                message = (
                    f"spline inflections\n"
                    f"points={update['point_count']} "
                    f"bins={update['observed_bins']}\n"
                    f"candidates={update['candidate_count']} "
                    f"inflect={update['inflection_candidates']} "
                    f"curve={update['curvature_candidates']}\n"
                    f"score={update['score']:.6g}"
                )
            else:
                message = (
                    f"candidate {update['candidate_index']}/{update['candidate_count']}\n"
                    f"points={update['point_count']} bins={update['observed_bins']}\n"
                    f"score={update['score']:.6g}"
                )
            if 'channel' in update:
                message = f"{update['channel']} | {message}"
            if 'image_score' in update:
                message = f"{message}\nimage={update['image_score']:.6g}"
            print(f'AutoTuneCurves: {message}')
            self._set_status(node_id, message)

        match_target_compression = (
            dpg_get_value(self._match_compression_value_tag(node_id)) is True
        )
        target_metadata = None
        if isinstance(target_result, dict):
            target_metadata = target_result.get('metadata') or target_result.get(
                'compression_metadata',
            )

        def _score_image_transform(image):
            matched_image, compression_metadata = match_metadata_compression(
                image,
                target_metadata,
            )
            return matched_image, compression_metadata

        score_image_transform = None
        if match_target_compression:
            if target_metadata is None:
                self._set_status(node_id, 'running: no target compression metadata')
            else:
                score_image_transform = _score_image_transform

        interpolation = self._interpolation_value(node_id)
        tune_channel = self._channel_value(node_id)
        if tune_channel == 'All':
            result = tune_curve_set(
                source,
                target,
                max_points=max_points,
                metric_name=metric_name,
                refinement_iterations=refinement_iterations,
                progress_callback=_progress,
                score_image_transform=score_image_transform,
                interpolation=interpolation,
            )
            curves_payload = {'curves': result.best_parameters['curves']}
            status_detail = '4 curves'
        else:
            result = tune_curves(
                source,
                target,
                channel=tune_channel,
                max_points=max_points,
                metric_name=metric_name,
                refinement_iterations=refinement_iterations,
                progress_callback=_progress,
                score_image_transform=score_image_transform,
                interpolation=interpolation,
            )
            helper = CurvesPointsEditorMixin()
            curve_set = helper._default_curve_set()
            curve_set[tune_channel] = result.best_parameters['points']
            curves_payload = {'curves': curve_set}
            status_detail = f'{len(result.best_parameters["points"])} points'
        compression_metadata = result.best_parameters.get('compression_metadata')
        if match_target_compression and compression_metadata is not None:
            status_detail = f'{status_detail}, target compression matched'

        dpg_set_value(ports.points.value_tag, str(curves_payload))
        dpg_set_value(ports.best_score.value_tag, float(result.best_score))
        self._set_status(node_id, f'done: {status_detail}')
        return result.best_image, {
            '__auto_tune_ready__': True,
            'tune_result': result,
            'points': curves_payload,
            'compression_metadata': compression_metadata,
            'match_target_compression': match_target_compression,
        }

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        ports = self.ports(node_id)
        return {
            'ver': self._ver,
            'pos': dpg.get_item_pos(tag_node_name),
            ports.points.value_tag: dpg_get_value(ports.points.value_tag),
            ports.best_score.value_tag: dpg_get_value(ports.best_score.value_tag),
            self._max_points_value_tag(node_id): dpg_get_value(
                self._max_points_value_tag(node_id),
            ),
            self._metric_value_tag(node_id): dpg_get_value(self._metric_value_tag(node_id)),
            self._refinement_iterations_value_tag(node_id): dpg_get_value(
                self._refinement_iterations_value_tag(node_id),
            ),
            self._channel_value_tag(node_id): self._channel_value(node_id),
            self._interpolation_value_tag(node_id): self._interpolation_value(node_id),
            self._match_compression_value_tag(node_id): dpg_get_value(
                self._match_compression_value_tag(node_id),
            ),
            '__cache_enabled__': False,
        }

    def set_setting_dict(self, node_id, setting_dict):
        ports = self.ports(node_id)
        for value_tag in (
            ports.points.value_tag,
            ports.best_score.value_tag,
            self._max_points_value_tag(node_id),
            self._metric_value_tag(node_id),
            self._refinement_iterations_value_tag(node_id),
            self._channel_value_tag(node_id),
            self._interpolation_value_tag(node_id),
            self._match_compression_value_tag(node_id),
        ):
            if value_tag in setting_dict:
                dpg_set_value(value_tag, setting_dict[value_tag])
