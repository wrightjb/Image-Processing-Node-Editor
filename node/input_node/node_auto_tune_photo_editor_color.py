#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from auto_tune.photo_editor_color import (
    DEFAULT_GLOBAL_ITERATIONS,
    DEFAULT_MAX_DIMENSION,
    DEFAULT_PARAMETERS,
    DEFAULT_POPULATION_SIZE,
    PARAMETER_NAMES,
    tune_photo_editor_color,
)
from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs
from node.process_node.node_photo_editor_color import Node as PhotoEditorColorNode
from node_editor.image_metadata import match_metadata_compression
from node_editor.util import dpg_get_value, dpg_set_value


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Auto Tune (Photo Editor Color)'
    node_tag = 'AutoTunePhotoEditorColor'

    _specs = {
        'source_image': InputPort(PortDataType.IMAGE, index=1),
        'target_image': InputPort(PortDataType.IMAGE, index=2),
        'mode': OutputPort(PortDataType.TEXT, index=1),
    }
    for _index, _name in enumerate(PARAMETER_NAMES, start=1):
        _specs[_name] = OutputPort(PortDataType.INT, index=_index)
    _specs['best_score'] = OutputPort(
        PortDataType.FLOAT,
        index=len(PARAMETER_NAMES) + 1,
    )
    port_specs = PortSpecs(**_specs)

    def __init__(self):
        self._run_requested_node_ids = set()

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        del opencv_setting_dict
        ports = self.create_ports(node_id)
        source_image_port = ports.source_image
        target_image_port = ports.target_image
        source_image = source_image_port.dpg_tag
        target_image = target_image_port.dpg_tag
        with dpg.node(
            tag=self._node_name(node_id),
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
            self._add_text_control(node_id, 'Status', 'idle')
            self._add_combo_control(
                node_id,
                'Mode',
                ('MacGyver parity', 'Standard'),
                'MacGyver parity',
                callback,
            )
            self._add_int_control(
                node_id,
                'GlobalIterations',
                'Global iterations',
                DEFAULT_GLOBAL_ITERATIONS,
                0,
                callback,
            )
            self._add_int_control(
                node_id,
                'PopulationSize',
                'Population size',
                DEFAULT_POPULATION_SIZE,
                1,
                callback,
            )
            self._add_int_control(
                node_id,
                'MaxDimension',
                'Search dimension',
                DEFAULT_MAX_DIMENSION,
                16,
                callback,
            )
            self._add_checkbox_control(
                node_id,
                'LocalRefinement',
                'Powell refinement',
                True,
                callback,
            )
            self._add_checkbox_control(
                node_id,
                'MatchCompression',
                'Match target compression',
                False,
                callback,
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
            self._add_text_output(ports.mode, 'mode', 'MacGyver parity')
            for name in PARAMETER_NAMES:
                self._add_parameter_output(
                    node_id,
                    getattr(ports, name),
                    name,
                    DEFAULT_PARAMETERS[name],
                    callback,
                )
            self._add_float_output(ports.best_score, 'score', 0.0)
        return self._node_name(node_id)

    def _control_attr_tag(self, node_id, name, value_type=None):
        return self._node_control_tag(node_id, value_type or self.TYPE_INT, name)

    def _control_value_tag(self, node_id, name, value_type=None):
        return self._node_control_value_tag(
            node_id,
            value_type or self.TYPE_INT,
            name,
        )

    def _add_text_control(self, node_id, name, default):
        with dpg.node_attribute(
            tag=self._control_attr_tag(node_id, name, self.TYPE_TEXT),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            dpg.add_text(
                default,
                tag=self._control_value_tag(node_id, name, self.TYPE_TEXT),
            )

    def _add_combo_control(self, node_id, name, items, default, callback):
        with dpg.node_attribute(
            tag=self._control_attr_tag(node_id, name, self.TYPE_TEXT),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            dpg.add_combo(
                items,
                tag=self._control_value_tag(node_id, name, self.TYPE_TEXT),
                label=name.lower(),
                default_value=default,
                width=140,
                callback=callback,
            )

    def _add_int_control(
        self, node_id, name, label, default, minimum, callback,
    ):
        with dpg.node_attribute(
            tag=self._control_attr_tag(node_id, name),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            dpg.add_input_int(
                tag=self._control_value_tag(node_id, name),
                label=label,
                default_value=default,
                min_value=minimum,
                min_clamped=True,
                width=120,
                callback=callback,
            )

    def _add_checkbox_control(
        self, node_id, name, label, default, callback,
    ):
        with dpg.node_attribute(
            tag=self._control_attr_tag(node_id, name),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            dpg.add_checkbox(
                tag=self._control_value_tag(node_id, name),
                label=label,
                default_value=default,
                callback=callback,
            )

    def _add_parameter_output(
        self, node_id, port, name, default, callback,
    ):
        parameter = next(
            parameter for parameter in PhotoEditorColorNode.parameters
            if parameter.get('name') == name
        )
        with dpg.node_attribute(
            tag=port.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Output,
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    tag=self._control_value_tag(
                        node_id,
                        f'Tune{name.title()}',
                    ),
                    default_value=True,
                    callback=callback,
                )
                dpg.add_input_int(
                    tag=port.value_tag,
                    default_value=default,
                    width=70,
                    callback=callback,
                )
                dpg.add_button(
                    label='-',
                    width=24,
                    callback=self._nudge_parameter,
                    user_data=(
                        port.value_tag,
                        -1,
                        int(parameter['min']),
                        int(parameter['max']),
                    ),
                )
                dpg.add_button(
                    label='+',
                    width=24,
                    callback=self._nudge_parameter,
                    user_data=(
                        port.value_tag,
                        1,
                        int(parameter['min']),
                        int(parameter['max']),
                    ),
                )
                dpg.add_text(name)

    @staticmethod
    def _nudge_parameter(sender, app_data, user_data):
        del sender, app_data
        value_tag, delta, minimum, maximum = user_data
        value = dpg_get_value(value_tag)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        dpg_set_value(value_tag, max(minimum, min(maximum, value + delta)))

    @staticmethod
    def _add_text_output(port, label, default):
        with dpg.node_attribute(
            tag=port.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Output,
        ):
            dpg.add_input_text(
                tag=port.value_tag,
                label=label,
                default_value=default,
                width=140,
                readonly=True,
            )

    @staticmethod
    def _add_float_output(port, label, default):
        with dpg.node_attribute(
            tag=port.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Output,
        ):
            dpg.add_input_float(
                tag=port.value_tag,
                label=label,
                default_value=default,
                width=120,
                readonly=True,
            )

    def _set_status(self, node_id, message):
        dpg_set_value(
            self._control_value_tag(node_id, 'Status', self.TYPE_TEXT),
            message,
        )

    def _on_run_button(self, sender, app_data, user_data):
        del sender, app_data
        self._run_requested_node_ids.add(str(user_data))
        self._set_status(user_data, 'queued')

    def _enabled_parameters(self, node_id):
        return [
            name for name in PARAMETER_NAMES
            if bool(dpg_get_value(
                self._control_value_tag(node_id, f'Tune{name.title()}')
            ))
        ]

    def _linked_image_result(
        self, port_ref, connection_list, node_image_dict, node_result_dict,
    ):
        for connection, source_tag, destination_tag, _kind in (
            self._iter_connection_infos(connection_list)
        ):
            if destination_tag != port_ref.dpg_tag:
                continue
            key = self._connection_source_node_key(connection, source_tag)
            return node_image_dict.get(key), node_result_dict.get(key)
        return None, None

    @staticmethod
    def _safe_int(value, fallback, minimum):
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return fallback

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        if str(node_id) not in self._run_requested_node_ids:
            return None, {'__auto_tune_ready__': False}
        self._run_requested_node_ids.discard(str(node_id))
        ports = self.ports(node_id)
        source, _source_result = self._linked_image_result(
            ports.source_image,
            connection_list,
            node_image_dict,
            node_result_dict,
        )
        target, target_result = self._linked_image_result(
            ports.target_image,
            connection_list,
            node_image_dict,
            node_result_dict,
        )
        if source is None or target is None:
            self._set_status(node_id, 'missing source/target')
            return None, {'__auto_tune_ready__': False}

        mode = dpg_get_value(
            self._control_value_tag(node_id, 'Mode', self.TYPE_TEXT),
        )
        if mode not in ('MacGyver parity', 'Standard'):
            mode = 'MacGyver parity'
        dpg_set_value(ports.mode.value_tag, mode)
        enabled = self._enabled_parameters(node_id)
        current = {
            name: dpg_get_value(getattr(ports, name).value_tag)
            for name in PARAMETER_NAMES
        }
        iterations = self._safe_int(
            dpg_get_value(self._control_value_tag(node_id, 'GlobalIterations')),
            DEFAULT_GLOBAL_ITERATIONS,
            0,
        )
        population = self._safe_int(
            dpg_get_value(self._control_value_tag(node_id, 'PopulationSize')),
            DEFAULT_POPULATION_SIZE,
            1,
        )
        max_dimension = self._safe_int(
            dpg_get_value(self._control_value_tag(node_id, 'MaxDimension')),
            DEFAULT_MAX_DIMENSION,
            16,
        )
        local_refinement = dpg_get_value(
            self._control_value_tag(node_id, 'LocalRefinement')
        ) is True
        match_compression = dpg_get_value(
            self._control_value_tag(node_id, 'MatchCompression')
        ) is True
        target_metadata = None
        if isinstance(target_result, dict):
            target_metadata = target_result.get('metadata') or target_result.get(
                'compression_metadata'
            )

        def _score_image_transform(image):
            return match_metadata_compression(image, target_metadata)

        score_transform = (
            _score_image_transform
            if match_compression and target_metadata is not None
            else None
        )
        self._set_status(node_id, 'running')

        def _progress(update):
            message = (
                f"{update['phase']} {update['candidate_index']}\n"
                f"score={update['score']:.6g} "
                f"best={update['best_score']:.6g}"
            )
            print(f'AutoTunePhotoEditorColor: {message}')
            self._set_status(node_id, message)

        result = tune_photo_editor_color(
            source,
            target,
            current_parameters=current,
            enabled_parameters=enabled,
            mode=mode,
            max_dimension=max_dimension,
            global_iterations=iterations,
            population_size=population,
            local_refinement=local_refinement,
            progress_callback=_progress,
            score_image_transform=score_transform,
        )
        for name in PARAMETER_NAMES:
            dpg_set_value(
                getattr(ports, name).value_tag,
                int(result.best_parameters[name]),
            )
        dpg_set_value(ports.best_score.value_tag, float(result.best_score))
        self._set_status(node_id, f'done: {result.evaluated_count} candidates')
        return result.best_image, {
            '__auto_tune_ready__': True,
            'tune_result': result,
            'mode': mode,
            'compression_metadata': result.best_parameters.get(
                'compression_metadata'
            ),
            'match_target_compression': match_compression,
        }

    def close(self, node_id):
        self._run_requested_node_ids.discard(str(node_id))

    def get_setting_dict(self, node_id):
        ports = self.ports(node_id)
        setting = {
            'ver': self._ver,
            'pos': dpg.get_item_pos(self._node_name(node_id)),
            '__cache_enabled__': False,
        }
        for port in ports.as_dict().values():
            if getattr(port, 'value_tag', None):
                setting[port.value_tag] = dpg_get_value(port.value_tag)
        control_names = [
            'GlobalIterations', 'PopulationSize', 'MaxDimension',
            'LocalRefinement', 'MatchCompression',
        ] + [f'Tune{name.title()}' for name in PARAMETER_NAMES]
        for name in control_names:
            tag = self._control_value_tag(node_id, name)
            setting[tag] = dpg_get_value(tag)
        mode_tag = self._control_value_tag(node_id, 'Mode', self.TYPE_TEXT)
        setting[mode_tag] = dpg_get_value(mode_tag)
        return setting

    def set_setting_dict(self, node_id, setting_dict):
        ports = self.ports(node_id)
        for port in ports.as_dict().values():
            tag = getattr(port, 'value_tag', None)
            if tag in setting_dict:
                dpg_set_value(tag, setting_dict[tag])
        for tag, value in setting_dict.items():
            if tag != 'pos' and dpg.does_item_exist(tag):
                dpg_set_value(tag, value)
