#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gaussian Blur-specific auto-tuning helpers."""

from dataclasses import replace
import math

from auto_tune.service import (
    EvaluationPlan,
    ParameterSpec,
    TuneRequest,
    grid_search,
)
from node.process_node.node_gaussian_blur import Node as GaussianBlurNode
from node.process_node.node_gaussian_blur import image_process


def _gaussian_parameter(name):
    for parameter in GaussianBlurNode.parameters:
        if parameter.get('name') == name:
            return parameter
    return {}


def _parameter_bound(name, bound_name, fallback):
    return _gaussian_parameter(name).get(bound_name, fallback)


DEFAULT_KERNEL_MIN = int(_parameter_bound('kernel_size', 'min', 1))
DEFAULT_KERNEL_MAX = int(_parameter_bound('kernel_size', 'max', 501))
DEFAULT_SIGMA_MIN = float(_parameter_bound('sigma', 'min', 0.1))
DEFAULT_SIGMA_MAX = float(_parameter_bound('sigma', 'max', 100.0))
DEFAULT_SIGMA_STEP = 0.1
DEFAULT_MAX_DIMENSION = 512
DEFAULT_REFINEMENT_DIMENSIONS = (128, 256, 512, None)


def odd_kernel_values(min_value=DEFAULT_KERNEL_MIN, max_value=DEFAULT_KERNEL_MAX):
    min_value = max(1, int(min_value))
    max_value = max(min_value, int(max_value))
    if min_value % 2 == 0:
        min_value += 1
    return tuple(range(min_value, max_value + 1, 2))


def sigma_values(
    min_value=DEFAULT_SIGMA_MIN,
    max_value=DEFAULT_SIGMA_MAX,
    step=DEFAULT_SIGMA_STEP,
):
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


def _downscale_for_tuning(image, max_dimension=DEFAULT_MAX_DIMENSION):
    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if max_dimension is None or largest_dimension <= max_dimension:
        return image, 1

    step = max(1, int(math.ceil(largest_dimension / float(max_dimension))))
    return image[::step, ::step].copy(), step


def _scale_odd_kernel(kernel_size, downscale_step):
    scaled = max(1, int(round(int(kernel_size) / float(downscale_step))))
    if scaled % 2 == 0:
        scaled += 1
    return scaled


def _unscale_odd_kernel(scaled_kernel, downscale_step, kernel_min, kernel_max):
    original = int(round(int(scaled_kernel) * float(downscale_step)))
    original = max(int(kernel_min), min(int(kernel_max), original))
    if original % 2 == 0:
        if original < int(kernel_max):
            original += 1
        else:
            original -= 1
    return max(1, original)


def _scaled_kernel_candidates(kernel_min, kernel_max, downscale_step):
    return sorted({
        _scale_odd_kernel(original_kernel, downscale_step)
        for original_kernel in odd_kernel_values(kernel_min, kernel_max)
    })


def _refined_kernel_bounds(best_kernel, downscale_step, kernel_min, kernel_max):
    radius = max(8, int(downscale_step) * 6)
    refined_min = max(int(kernel_min), int(best_kernel) - radius)
    refined_max = min(int(kernel_max), int(best_kernel) + radius)
    return refined_min, refined_max


def tuning_plan(
    source_image,
    kernel_min=DEFAULT_KERNEL_MIN,
    kernel_max=DEFAULT_KERNEL_MAX,
    max_dimension=DEFAULT_MAX_DIMENSION,
):
    _scaled_source, downscale_step = _downscale_for_tuning(
        source_image,
        max_dimension,
    )
    original_kernels = odd_kernel_values(kernel_min, kernel_max)
    scaled_kernels = _scaled_kernel_candidates(
        kernel_min,
        kernel_max,
        downscale_step,
    )
    return {
        'original_candidates': len(original_kernels),
        'scaled_candidates': len(scaled_kernels),
        'downscale_step': downscale_step,
        'max_dimension': max_dimension,
    }


def _tune_gaussian_blur_pass(
    source_image,
    target_image,
    auto_sigma,
    kernel_min,
    kernel_max,
    sigma_min,
    sigma_max,
    sigma_step,
    max_dimension,
    pass_index,
    pass_count,
    total_evaluated_before,
    progress_callback=None,
):
    source_for_score, downscale_step = _downscale_for_tuning(
        source_image,
        max_dimension,
    )
    target_for_score, _ = _downscale_for_tuning(target_image, max_dimension)
    scaled_kernels = _scaled_kernel_candidates(
        kernel_min,
        kernel_max,
        downscale_step,
    )

    specs = [
        ParameterSpec(
            'kernel_size',
            tuple(scaled_kernels),
        ),
    ]
    fixed = {'auto_sigma': auto_sigma}
    if auto_sigma:
        fixed['sigma'] = 0.0
    else:
        specs.append(
            ParameterSpec(
                'sigma',
                sigma_values(sigma_min, sigma_max, sigma_step),
            )
        )

    def _evaluate(parameters):
        return image_process(
            source_for_score.copy(),
            int(parameters['kernel_size']),
            float(parameters['sigma']),
        )

    request = TuneRequest(
        source_image=source_for_score,
        target_image=target_for_score,
        parameter_specs=specs,
        evaluation_plan=EvaluationPlan(_evaluate),
        fixed_parameters=fixed,
    )
    def _progress(update):
        if progress_callback is None:
            return
        update = dict(update)
        update.update({
            'pass_index': pass_index,
            'pass_count': pass_count,
            'downscale_step': downscale_step,
            'max_dimension': max_dimension,
            'total_evaluated': (
                total_evaluated_before + update['candidate_index']
            ),
        })
        progress_callback(update)

    result = grid_search(request, progress_callback=_progress)
    original_kernel = _unscale_odd_kernel(
        result.best_parameters['kernel_size'],
        downscale_step,
        kernel_min,
        kernel_max,
    )
    best_parameters = dict(result.best_parameters)
    best_parameters['kernel_size'] = original_kernel
    return replace(result, best_parameters=best_parameters), downscale_step


def _refinement_dimensions_for_image(image, max_dimension):
    if max_dimension not in DEFAULT_REFINEMENT_DIMENSIONS:
        return (max_dimension,)

    largest_dimension = max(image.shape[:2])
    dimensions = tuple(
        dimension
        for dimension in DEFAULT_REFINEMENT_DIMENSIONS
        if dimension is not None and largest_dimension > dimension
    )
    return dimensions + (None,)


def tune_gaussian_blur(
    source_image,
    target_image,
    current_parameters=None,
    kernel_min=DEFAULT_KERNEL_MIN,
    kernel_max=DEFAULT_KERNEL_MAX,
    sigma_min=DEFAULT_SIGMA_MIN,
    sigma_max=DEFAULT_SIGMA_MAX,
    sigma_step=DEFAULT_SIGMA_STEP,
    max_dimension=DEFAULT_MAX_DIMENSION,
    progress_callback=None,
):
    """Tune Gaussian Blur kernel size and, when enabled, sigma.

    The tuner uses coarse-to-fine grid refinement over odd kernel sizes.
    When ``auto_sigma`` is true, sigma is fixed to ``0.0`` to match the
    Gaussian Blur node's auto-sigma behavior.
    """
    current_parameters = dict(current_parameters or {})
    auto_sigma = bool(current_parameters.get('auto_sigma', True))
    refinement_dimensions = _refinement_dimensions_for_image(
        source_image,
        max_dimension,
    )

    current_kernel_min = kernel_min
    current_kernel_max = kernel_max
    total_evaluated_count = 0
    result = None
    pass_count = len(refinement_dimensions)
    for pass_offset, refinement_dimension in enumerate(refinement_dimensions):
        result, downscale_step = _tune_gaussian_blur_pass(
            source_image,
            target_image,
            auto_sigma,
            current_kernel_min,
            current_kernel_max,
            sigma_min,
            sigma_max,
            sigma_step,
            refinement_dimension,
            pass_offset + 1,
            pass_count,
            total_evaluated_count,
            progress_callback,
        )
        total_evaluated_count += result.evaluated_count
        current_kernel_min, current_kernel_max = _refined_kernel_bounds(
            result.best_parameters['kernel_size'],
            downscale_step,
            kernel_min,
            kernel_max,
        )

    best_parameters = dict(result.best_parameters)
    best_image = image_process(
        source_image.copy(),
        int(best_parameters['kernel_size']),
        float(best_parameters['sigma']),
    )
    return replace(
        result,
        best_parameters=best_parameters,
        best_image=best_image,
        evaluated_count=total_evaluated_count,
    )
