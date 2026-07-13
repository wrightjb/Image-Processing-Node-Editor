#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from auto_tune.gaussian_blur import DEFAULT_KERNEL_MAX, DEFAULT_KERNEL_MIN
from auto_tune.gaussian_blur import DEFAULT_REFINEMENT_ITERATIONS
from auto_tune.gaussian_blur import tuning_plan, tune_gaussian_blur
from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs
from node_editor.util import dpg_get_value, dpg_set_value


class Node(DpgNodeBase):
    _ver = '0.0.4'

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
        auto_kernel=OutputPort(PortDataType.INT, index=4),
        kernel_factor=OutputPort(PortDataType.FLOAT, index=5),
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
        auto_kernel_port = ports.auto_kernel
        kernel_factor_port = ports.kernel_factor
        source_image = source_image_port.dpg_tag
        target_image = target_image_port.dpg_tag
        kernel_size = kernel_size_port.dpg_tag
        sigma = sigma_port.dpg_tag
        best_score = best_score_port.dpg_tag
        auto_kernel = auto_kernel_port.dpg_tag
        kernel_factor = kernel_factor_port.dpg_tag
        status_value_tag = self._status_value_tag(node_id)
        auto_sigma_value_tag = self._auto_sigma_value_tag(node_id)
        auto_kernel_value_tag = self._auto_kernel_value_tag(node_id)
        kernel_factor_value_tag = self._kernel_factor_value_tag(node_id)
        metric_value_tag = self._metric_value_tag(node_id)
        refinement_iterations_value_tag = self._refinement_iterations_value_tag(node_id)
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
                tag=self._auto_kernel_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_checkbox(
                    label='Auto Kernel',
                    tag=auto_kernel_value_tag,
                    default_value=False,
                )
            with dpg.node_attribute(
                tag=self._kernel_factor_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_float(
                    tag=kernel_factor_value_tag,
                    label='kernel factor',
                    default_value=3.0,
                    min_value=0.1,
                    min_clamped=True,
                    width=120,
                )
            with dpg.node_attribute(
                tag=self._metric_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    ('mse', 'smoothness', 'local_smoothness'),
                    label='Metric',
                    tag=metric_value_tag,
                    default_value='local_smoothness',
                    width=120,
                )
            with dpg.node_attribute(
                tag=self._refinement_iterations_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_int(
                    tag=refinement_iterations_value_tag,
                    label='Refine Rounds',
                    default_value=DEFAULT_REFINEMENT_ITERATIONS,
                    min_value=1,
                    min_clamped=True,
                    width=120,
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
            with dpg.node_attribute(
                tag=auto_kernel,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_checkbox(
                    tag=auto_kernel_port.value_tag,
                    label='auto kernel',
                    default_value=False,
                    enabled=False,
                )
            with dpg.node_attribute(
                tag=kernel_factor,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_float(
                    tag=kernel_factor_port.value_tag,
                    label='kernel factor',
                    default_value=3.0,
                    width=120,
                    readonly=True,
                )

        return tag_node_name

    def _metric_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Metric')

    def _metric_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Metric')

    def _refinement_iterations_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'RefineRounds')

    def _refinement_iterations_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'RefineRounds')

    def _auto_sigma_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'AutoSigma')

    def _auto_sigma_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'AutoSigma')

    def _auto_kernel_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_INT, 'AutoKernel')

    def _auto_kernel_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_INT, 'AutoKernel')

    def _kernel_factor_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_FLOAT, 'KernelFactor')

    def _kernel_factor_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_FLOAT, 'KernelFactor')

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

    def _refinement_iterations(self, node_id):
        value = dpg_get_value(self._refinement_iterations_value_tag(node_id))
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return DEFAULT_REFINEMENT_ITERATIONS

    def _current_output_parameters(self, ports):
        parameters = {}
        kernel = dpg_get_value(ports.kernel_size.value_tag)
        sigma = dpg_get_value(ports.sigma.value_tag)
        auto_kernel = dpg_get_value(ports.auto_kernel.value_tag)
        kernel_factor = dpg_get_value(ports.kernel_factor.value_tag)
        if kernel is not None:
            parameters['kernel_size'] = int(kernel)
        if sigma is not None:
            parameters['sigma'] = float(sigma)
        if auto_kernel is not None:
            parameters['auto_kernel'] = bool(auto_kernel)
        if kernel_factor is not None:
            parameters['kernel_factor'] = float(kernel_factor)
        return parameters

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

        output_parameters = self._current_output_parameters(ports)
        auto_sigma_value = dpg_get_value(self._auto_sigma_value_tag(node_id))
        auto_kernel_value = dpg_get_value(self._auto_kernel_value_tag(node_id))
        kernel_factor_value = dpg_get_value(self._kernel_factor_value_tag(node_id))
        fallback_auto_sigma = (
            True if auto_sigma_value is None else bool(auto_sigma_value)
        )
        fallback_auto_kernel = (
            False if auto_kernel_value is None else bool(auto_kernel_value)
        )
        fallback_kernel_factor = (
            3.0 if kernel_factor_value is None else float(kernel_factor_value)
        )
        current_parameters = {
            'auto_sigma': fallback_auto_sigma,
            'auto_kernel': fallback_auto_kernel,
            'kernel_factor': fallback_kernel_factor,
        }
        current_parameters.update(output_parameters)
        auto_kernel = bool(current_parameters.get('auto_kernel', False))
        auto_sigma = bool(current_parameters.get('auto_sigma', True)) and not auto_kernel
        kernel_factor = float(current_parameters.get('kernel_factor', 3.0))
        current_parameters['auto_sigma'] = auto_sigma
        current_parameters['auto_kernel'] = auto_kernel
        current_parameters['kernel_factor'] = kernel_factor
        dpg_set_value(self._auto_sigma_value_tag(node_id), auto_sigma)
        dpg_set_value(self._auto_kernel_value_tag(node_id), auto_kernel)
        dpg_set_value(self._kernel_factor_value_tag(node_id), kernel_factor)
        metric_name = dpg_get_value(self._metric_value_tag(node_id)) or 'local_smoothness'
        refinement_iterations = self._refinement_iterations(node_id)
        dpg_set_value(
            self._refinement_iterations_value_tag(node_id),
            refinement_iterations,
        )
        plan = tuning_plan(
            source,
            DEFAULT_KERNEL_MIN,
            DEFAULT_KERNEL_MAX,
            refinement_iterations=refinement_iterations,
        )
        print(
            'AutoTuneGaussianBlur: tuning started; '
            f'evaluating {plan["scaled_candidates"]} scaled candidates '
            f'from {plan["original_candidates"]} original odd kernels '
            f'({DEFAULT_KERNEL_MIN}..{DEFAULT_KERNEL_MAX}), '
            f'downscale step={plan["downscale_step"]}, '
            f'auto_sigma={auto_sigma}, '
            f'auto_kernel={auto_kernel}, '
            f'kernel_factor={kernel_factor}, '
            f'metric={metric_name}, '
            f'refine_rounds={refinement_iterations}, '
            f'planned_passes={plan["planned_passes"]}, '
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
            if 'candidate_smoothness' in update:
                message = (
                    f"{message}\n"
                    f"smooth={update['candidate_smoothness']:.6g} "
                    f"target={update['target_smoothness']:.6g}"
                )
            print(f'AutoTuneGaussianBlur: {message}')
            self._set_status(node_id, message)

        self._set_status(node_id, 'running')
        result = tune_gaussian_blur(
            source,
            target,
            current_parameters=current_parameters,
            progress_callback=_progress,
            metric_name=metric_name,
            refinement_iterations=refinement_iterations,
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
        dpg_set_value(
            ports.auto_kernel.value_tag,
            bool(result.best_parameters.get('auto_kernel', False)),
        )
        dpg_set_value(
            ports.kernel_factor.value_tag,
            float(result.best_parameters.get('kernel_factor', kernel_factor)),
        )
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
            ports.auto_kernel.value_tag: dpg_get_value(ports.auto_kernel.value_tag),
            ports.kernel_factor.value_tag: dpg_get_value(ports.kernel_factor.value_tag),
            self._auto_sigma_value_tag(node_id): dpg_get_value(
                self._auto_sigma_value_tag(node_id),
            ),
            self._auto_kernel_value_tag(node_id): dpg_get_value(
                self._auto_kernel_value_tag(node_id),
            ),
            self._kernel_factor_value_tag(node_id): dpg_get_value(
                self._kernel_factor_value_tag(node_id),
            ),
            self._metric_value_tag(node_id): dpg_get_value(
                self._metric_value_tag(node_id),
            ),
            self._refinement_iterations_value_tag(node_id): dpg_get_value(
                self._refinement_iterations_value_tag(node_id),
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
            ports.auto_kernel.value_tag,
            ports.kernel_factor.value_tag,
            self._auto_sigma_value_tag(node_id),
            self._auto_kernel_value_tag(node_id),
            self._kernel_factor_value_tag(node_id),
            self._metric_value_tag(node_id),
            self._refinement_iterations_value_tag(node_id),
        ):
            if value_tag in setting_dict:
                dpg_set_value(value_tag, setting_dict[value_tag])
