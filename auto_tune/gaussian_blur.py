#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gaussian Blur-specific auto-tuning helpers."""

from auto_tune.service import EvaluationPlan, ParameterSpec, TuneRequest, grid_search
from node.process_node.node_gaussian_blur import image_process


def odd_kernel_values(min_value=1, max_value=51):
    min_value = max(1, int(min_value))
    max_value = max(min_value, int(max_value))
    if min_value % 2 == 0:
        min_value += 1
    return tuple(range(min_value, max_value + 1, 2))


def sigma_values(min_value=0.1, max_value=10.0, step=0.1):
    min_value = float(min_value)
    max_value = float(max_value)
    step = float(step)
    if step <= 0:
        raise ValueError('sigma step must be positive')
    values = []
    current = min_value
    while current <= max_value + (step / 10.0):
        values.append(round(current, 3))
        current += step
    return tuple(values)


def tune_gaussian_blur(
    source_image,
    target_image,
    current_parameters=None,
    kernel_min=1,
    kernel_max=51,
    sigma_min=0.1,
    sigma_max=10.0,
    sigma_step=0.1,
):
    """Tune Gaussian Blur kernel size and, when enabled, sigma.

    The first MVP honors the node's ``auto_sigma`` behavior: when auto sigma is
    true, sigma is fixed to ``0.0`` and only odd kernel sizes are searched.
    """
    current_parameters = dict(current_parameters or {})
    auto_sigma = bool(current_parameters.get('auto_sigma', True))
    specs = [
        ParameterSpec('kernel_size', odd_kernel_values(kernel_min, kernel_max)),
    ]
    fixed = {'auto_sigma': auto_sigma}
    if auto_sigma:
        fixed['sigma'] = 0.0
    else:
        specs.append(ParameterSpec('sigma', sigma_values(sigma_min, sigma_max, sigma_step)))

    def _evaluate(parameters):
        return image_process(
            source_image.copy(),
            int(parameters['kernel_size']),
            float(parameters['sigma']),
        )

    request = TuneRequest(
        source_image=source_image,
        target_image=target_image,
        parameter_specs=specs,
        evaluation_plan=EvaluationPlan(_evaluate),
        fixed_parameters=fixed,
    )
    return grid_search(request)
