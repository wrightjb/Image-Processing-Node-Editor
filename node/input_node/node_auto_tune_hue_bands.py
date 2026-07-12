#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from auto_tune.hue_bands import DEFAULT_REFINEMENT_ITERATIONS, tune_hue_bands
from node.process_node.node_hue_saturation_adjustment import _BANDS
from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs
from node_editor.util import dpg_get_value, dpg_set_value


class Node(DpgNodeBase):
    _ver = '0.0.1'

    def __init__(self):
        self._run_requested_node_ids = set()

    node_label = 'Auto Tune (Hue Bands)'
    node_tag = 'AutoTuneHueBands'

    _specs = {
        'source_image': InputPort(PortDataType.IMAGE, index=1),
        'target_image': InputPort(PortDataType.IMAGE, index=2),
        'blend': OutputPort(PortDataType.FLOAT, index=1),
    }
    for _index, (_band_name, _center) in enumerate(_BANDS):
        _specs[f'{_band_name}_hue_shift'] = OutputPort(
            PortDataType.INT, index=(_index * 2) + 2,
        )
        _specs[f'{_band_name}_saturation'] = OutputPort(
            PortDataType.INT, index=(_index * 2) + 3,
        )
    _specs['best_score'] = OutputPort(PortDataType.FLOAT, index=(len(_BANDS) * 2) + 2)
    port_specs = PortSpecs(**_specs)

    def add_node(self, parent, node_id, pos=[0, 0], opencv_setting_dict=None, callback=None):
        del opencv_setting_dict
        ports = self.create_ports(node_id)
        source_image_port = ports.source_image
        target_image_port = ports.target_image
        source_image = source_image_port.dpg_tag
        target_image = target_image_port.dpg_tag
        with dpg.node(tag=self._node_name(node_id), parent=parent, label=self.node_label, pos=pos):
            self.add_editor_toolbar(
                node_id,
                callback=callback,
                build_extra_controls=lambda: dpg.add_button(
                    label='Run Tune', width=80, callback=self._on_run_button, user_data=node_id,
                ),
            )
            with dpg.node_attribute(tag=self._status_attr_tag(node_id), attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_text('idle', tag=self._status_value_tag(node_id))
            with dpg.node_attribute(tag=self._refinement_iterations_attr_tag(node_id), attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(
                    tag=self._refinement_iterations_value_tag(node_id), label='Refine Rounds',
                    default_value=DEFAULT_REFINEMENT_ITERATIONS, min_value=0, min_clamped=True, width=120,
                )
            with dpg.node_attribute(tag=self._tune_blend_attr_tag(node_id), attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_checkbox(
                    tag=self._tune_blend_value_tag(node_id),
                    label='Tune Blend',
                    default_value=False,
                )
            with dpg.node_attribute(tag=self._fixed_blend_attr_tag(node_id), attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_slider_float(
                    tag=self._fixed_blend_value_tag(node_id),
                    label='Fixed Blend',
                    default_value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    width=120,
                )
            with dpg.node_attribute(tag=source_image, attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text('source image')
            with dpg.node_attribute(tag=target_image, attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text('target image')
            self._add_float_output(ports.blend, 'blend', 0.0)
            for band_name, _center in _BANDS:
                self._add_int_output(
                    getattr(ports, f'{band_name}_hue_shift'),
                    f'{band_name[:3]} hue',
                    0,
                )
                self._add_int_output(
                    getattr(ports, f'{band_name}_saturation'),
                    f'{band_name[:3]} sat',
                    0,
                )
            self._add_float_output(ports.best_score, 'score', 0.0)
        return self._node_name(node_id)

    def _add_int_output(self, port, label, default):
        with dpg.node_attribute(tag=port.dpg_tag, attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_input_int(tag=port.value_tag, label=label, default_value=default, width=120, readonly=True)

    def _add_float_output(self, port, label, default):
        with dpg.node_attribute(tag=port.dpg_tag, attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_input_float(tag=port.value_tag, label=label, default_value=default, width=120, readonly=True)

    def _status_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Status')

    def _status_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Status')

    def _refinement_iterations_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'RefineRounds')

    def _refinement_iterations_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'RefineRounds')

    def _tune_blend_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'TuneBlend')

    def _tune_blend_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'TuneBlend')

    def _tune_blend_value(self, node_id):
        return bool(dpg_get_value(self._tune_blend_value_tag(node_id)))

    def _fixed_blend_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_FLOAT, 'FixedBlend')

    def _fixed_blend_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_FLOAT, 'FixedBlend')

    def _fixed_blend_value(self, node_id):
        value = dpg_get_value(self._fixed_blend_value_tag(node_id))
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _set_status(self, node_id, message):
        dpg_set_value(self._status_value_tag(node_id), message)

    def _on_run_button(self, sender, app_data, user_data):
        del sender, app_data
        self._run_requested_node_ids.add(str(user_data))
        self._set_status(user_data, 'queued')

    def _linked_image(self, port_ref, connection_list, node_image_dict):
        for connection_info, source_tag, destination_tag, _connection_type in self._iter_connection_infos(connection_list):
            if destination_tag != port_ref.dpg_tag:
                continue
            source_node_key = self._connection_source_node_key(connection_info, source_tag)
            return node_image_dict.get(source_node_key)
        return None

    def _int_setting(self, tag, fallback, minimum):
        value = dpg_get_value(tag)
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return fallback

    def _current_output_parameters(self, ports):
        parameters = {}
        for name in ['blend'] + [f'{b}_{kind}' for b, _ in _BANDS for kind in ('hue_shift', 'saturation')]:
            value = dpg_get_value(getattr(ports, name).value_tag)
            if value is not None:
                parameters[name] = float(value) if name == 'blend' else int(value)
        return parameters

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        del node_result_dict
        node_id_key = str(node_id)
        if node_id_key not in self._run_requested_node_ids:
            return None, {'__auto_tune_ready__': False}
        self._run_requested_node_ids.discard(node_id_key)
        ports = self.ports(node_id)
        source = self._linked_image(ports.source_image, connection_list, node_image_dict)
        target = self._linked_image(ports.target_image, connection_list, node_image_dict)
        if source is None or target is None:
            self._set_status(node_id, 'missing source/target')
            return None, {'__auto_tune_ready__': False}
        refinement_iterations = self._int_setting(
            self._refinement_iterations_value_tag(node_id), DEFAULT_REFINEMENT_ITERATIONS, 0,
        )
        dpg_set_value(self._refinement_iterations_value_tag(node_id), refinement_iterations)
        self._set_status(node_id, 'running')

        def _progress(update):
            band = update.get('band') or 'blend'
            message = (
                f"{update['phase']} {band}\n"
                f"candidate {update['candidate_index']}/{update['candidate_count']} "
                f"total {update['total_evaluated']}\n"
                f"score={update['score']:.6g} best={update['best_score']:.6g}"
            )
            print(f'AutoTuneHueBands: {message}')
            self._set_status(node_id, message)

        tune_blend = self._tune_blend_value(node_id)
        fixed_blend = self._fixed_blend_value(node_id)
        dpg_set_value(self._tune_blend_value_tag(node_id), tune_blend)
        dpg_set_value(self._fixed_blend_value_tag(node_id), fixed_blend)
        result = tune_hue_bands(
            source,
            target,
            current_parameters=self._current_output_parameters(ports),
            refinement_iterations=refinement_iterations,
            progress_callback=_progress,
            tune_blend=tune_blend,
            fixed_blend=fixed_blend,
        )
        for name, value in result.best_parameters.items():
            port = getattr(ports, name, None)
            if port is not None:
                dpg_set_value(port.value_tag, float(value) if name == 'blend' else int(value))
        dpg_set_value(ports.best_score.value_tag, float(result.best_score))
        self._set_status(node_id, f'done: {result.evaluated_count} candidates')
        return result.best_image, {'__auto_tune_ready__': True, 'tune_result': result}

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        ports = self.ports(node_id)
        setting = {'ver': self._ver, 'pos': dpg.get_item_pos(self._node_name(node_id)), '__cache_enabled__': False}
        for port in ports.as_dict().values():
            if getattr(port, 'value_tag', None):
                setting[port.value_tag] = dpg_get_value(port.value_tag)
        setting[self._refinement_iterations_value_tag(node_id)] = dpg_get_value(self._refinement_iterations_value_tag(node_id))
        setting[self._tune_blend_value_tag(node_id)] = dpg_get_value(
            self._tune_blend_value_tag(node_id)
        )
        setting[self._fixed_blend_value_tag(node_id)] = dpg_get_value(
            self._fixed_blend_value_tag(node_id)
        )
        return setting

    def set_setting_dict(self, node_id, setting_dict):
        ports = self.ports(node_id)
        for port in ports.as_dict().values():
            if getattr(port, 'value_tag', None) in setting_dict:
                dpg_set_value(port.value_tag, setting_dict[port.value_tag])
        refine_tag = self._refinement_iterations_value_tag(node_id)
        if refine_tag in setting_dict:
            dpg_set_value(refine_tag, setting_dict[refine_tag])
        tune_blend_tag = self._tune_blend_value_tag(node_id)
        if tune_blend_tag in setting_dict:
            dpg_set_value(tune_blend_tag, setting_dict[tune_blend_tag])
        fixed_blend_tag = self._fixed_blend_value_tag(node_id)
        if fixed_blend_tag in setting_dict:
            dpg_set_value(fixed_blend_tag, setting_dict[fixed_blend_tag])
