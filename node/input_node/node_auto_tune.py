#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from auto_tune.gaussian_blur import DEFAULT_KERNEL_MAX, DEFAULT_KERNEL_MIN
from auto_tune.gaussian_blur import tuning_plan, tune_gaussian_blur
from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs
from node_editor.util import dpg_get_value, dpg_set_value


class Node(DpgNodeBase):
    _ver = '0.0.3'

    def __init__(self):
        self._run_requested_node_ids = set()

    node_label = 'Auto Tune (Gaussian Blur)'
    node_tag = 'AutoTuneGaussianBlur'

    port_specs = PortSpecs(
        source_image=InputPort(PortDataType.IMAGE, index=1),
        target_image=InputPort(PortDataType.IMAGE, index=2),
        kernel_size=OutputPort(PortDataType.INT, index=1),
        sigma=OutputPort(PortDataType.FLOAT, index=2),
        best_score=OutputPort(PortDataType.FLOAT, index=3),
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
        kernel_size_port = ports.kernel_size
        sigma_port = ports.sigma
        best_score_port = ports.best_score
        source_image = source_image_port.dpg_tag
        target_image = target_image_port.dpg_tag
        kernel_size = kernel_size_port.dpg_tag
        sigma = sigma_port.dpg_tag
        best_score = best_score_port.dpg_tag
        status_value_tag = self._status_value_tag(node_id)
        auto_sigma_value_tag = self._auto_sigma_value_tag(node_id)
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
                dpg.add_text(
                    'idle',
                    tag=status_value_tag,
                )
            with dpg.node_attribute(
                tag=self._auto_sigma_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    label='Auto Sigma',
                    tag=auto_sigma_value_tag,
                    default_value=True,
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
                tag=kernel_size,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_int(
                    tag=kernel_size_port.value_tag,
                    label='kernel',
                    default_value=5,
                    width=120,
                    readonly=True,
                )
            with dpg.node_attribute(
                tag=sigma,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_float(
                    tag=sigma_port.value_tag,
                    label='sigma',
                    default_value=0.0,
                    width=120,
                    readonly=True,
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

    def _auto_sigma_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'AutoSigma')

    def _auto_sigma_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'AutoSigma')

    def _status_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Status')

    def _status_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Status')

    def _set_status(self, node_id, message):
        dpg_set_value(self._status_value_tag(node_id), message)

    def _on_run_button(self, sender, app_data, user_data):
        del sender, app_data
        print(
            'AutoTuneGaussianBlur: Run Tune requested for '
            f'node {user_data}; queued for the next graph update tick.'
        )
        self._run_requested_node_ids.add(str(user_data))
        self._set_status(user_data, 'queued')

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


    def _target_gaussian_parameters(self, port_ref, connection_list):
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
            if not source_node_key.endswith(':GaussianBlur'):
                return {}

            kernel = dpg_get_value(
                self._port_value_tag(source_node_key, self.TYPE_INT, 'Input02')
            )
            sigma = dpg_get_value(
                self._port_value_tag(source_node_key, self.TYPE_FLOAT, 'Input03')
            )
            auto_sigma = dpg_get_value(
                self._port_value_tag(source_node_key, self.TYPE_INT, 'Input04')
            )
            parameters = {}
            if kernel is not None:
                parameters['kernel_size'] = int(kernel)
            if sigma is not None:
                parameters['sigma'] = float(sigma)
            if auto_sigma is not None:
                parameters['auto_sigma'] = bool(auto_sigma)
            return parameters
        return {}

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        del node_result_dict
        node_id_key = str(node_id)
        if node_id_key not in self._run_requested_node_ids:
            return None, {'__auto_tune_ready__': False}
        self._run_requested_node_ids.discard(node_id_key)

        ports = self.ports(node_id)
        source = self._linked_image(
            ports.source_image,
            connection_list,
            node_image_dict,
        )
        target = self._linked_image(
            ports.target_image,
            connection_list,
            node_image_dict,
        )
        if source is None or target is None:
            print(
                'AutoTuneGaussianBlur: Run Tune skipped; source and target '
                'images must both be connected and available.'
            )
            self._set_status(node_id, 'missing source/target')
            return None, {'__auto_tune_ready__': False}

        target_parameters = self._target_gaussian_parameters(
            ports.target_image,
            connection_list,
        )
        auto_sigma_value = dpg_get_value(self._auto_sigma_value_tag(node_id))
        fallback_auto_sigma = (
            True if auto_sigma_value is None else bool(auto_sigma_value)
        )
        current_parameters = {'auto_sigma': fallback_auto_sigma}
        current_parameters.update(target_parameters)
        auto_sigma = bool(current_parameters.get('auto_sigma', True))
        dpg_set_value(self._auto_sigma_value_tag(node_id), auto_sigma)
        plan = tuning_plan(source, DEFAULT_KERNEL_MIN, DEFAULT_KERNEL_MAX)
        print(
            'AutoTuneGaussianBlur: tuning started; '
            f'evaluating {plan["scaled_candidates"]} scaled candidates '
            f'from {plan["original_candidates"]} original odd kernels '
            f'({DEFAULT_KERNEL_MIN}..{DEFAULT_KERNEL_MAX}), '
            f'downscale step={plan["downscale_step"]}, '
            f'auto_sigma={auto_sigma}, '
            f'start={current_parameters}.'
        )

        def _progress(update):
            parameters = update['parameters']
            message = (
                f"pass {update['pass_index']}/{update['pass_count']} \n"
                f"candidate {update['candidate_index']}/"
                f"{update['candidate_count']} "
                f"total {update['total_evaluated']}\n"
                f"kernel={parameters['kernel_size']} "
                f"sigma={parameters.get('sigma', 0.0)} \n"
                f"score={update['score']:.6g} "
                f"best={update['best_score']:.6g}"
            )
            print(f'AutoTuneGaussianBlur: {message}')
            self._set_status(node_id, message)

        self._set_status(node_id, 'running')
        result = tune_gaussian_blur(
            source,
            target,
            current_parameters=current_parameters,
            progress_callback=_progress,
        )
        dpg_set_value(
            ports.kernel_size.value_tag,
            int(result.best_parameters['kernel_size']),
        )
        dpg_set_value(
            ports.sigma.value_tag,
            float(result.best_parameters['sigma']),
        )
        dpg_set_value(ports.best_score.value_tag, float(result.best_score))
        self._set_status(
            node_id,
            f'done: {result.evaluated_count} candidates',
        )
        print(
            'AutoTuneGaussianBlur: tuning finished; '
            f'evaluated {result.evaluated_count} candidates, '
            f'kernel={result.best_parameters["kernel_size"]}, '
            f'sigma={result.best_parameters["sigma"]}, '
            f'score={result.best_score}'
        )
        return result.best_image, {
            '__auto_tune_ready__': True,
            'tune_result': result,
        }

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        ports = self.ports(node_id)
        setting_dict = {
            'ver': self._ver,
            'pos': dpg.get_item_pos(tag_node_name),
            ports.kernel_size.value_tag: dpg_get_value(ports.kernel_size.value_tag),
            ports.sigma.value_tag: dpg_get_value(ports.sigma.value_tag),
            ports.best_score.value_tag: dpg_get_value(ports.best_score.value_tag),
            self._auto_sigma_value_tag(node_id): dpg_get_value(
                self._auto_sigma_value_tag(node_id),
            ),
            '__cache_enabled__': False,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        ports = self.ports(node_id)
        for value_tag in (
            ports.kernel_size.value_tag,
            ports.sigma.value_tag,
            ports.best_score.value_tag,
            self._auto_sigma_value_tag(node_id),
        ):
            if value_tag in setting_dict:
                dpg_set_value(value_tag, setting_dict[value_tag])
