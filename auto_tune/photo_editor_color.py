#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SciPy-based tuning helpers for the Photo Editor Color node."""

from dataclasses import replace

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize

from auto_tune.service import TuneResult, mean_absolute_error
from node.process_node.node_photo_editor_color import Node as PhotoEditorColorNode
from node.process_node.node_photo_editor_color import image_process


DEFAULT_MAX_DIMENSION = 240
DEFAULT_GLOBAL_ITERATIONS = 20
DEFAULT_POPULATION_SIZE = 8
DEFAULT_SEED = 0
PARAMETER_NAMES = (
    'exposure',
    'brightness',
    'contrast',
    'saturation',
    'temperature',
    'tint',
    'hue',
)


def _parameter_definition(name):
    for parameter in PhotoEditorColorNode.parameters:
        if parameter.get('name') == name:
            return parameter
    raise KeyError(name)


PARAMETER_BOUNDS = {
    name: (
        float(_parameter_definition(name)['min']),
        float(_parameter_definition(name)['max']),
    )
    for name in PARAMETER_NAMES
}
DEFAULT_PARAMETERS = {
    name: int(_parameter_definition(name)['default'])
    for name in PARAMETER_NAMES
}


def _resize_pair(source, target, max_dimension):
    source = np.asarray(source)
    target = np.asarray(target)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError('source and target must be color images')
    if target.shape[:2] != source.shape[:2]:
        target = cv2.resize(
            target,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if max_dimension is None:
        return source, target
    largest = max(source.shape[:2])
    if largest <= int(max_dimension):
        return source, target
    step = max(1, int(np.ceil(largest / float(max_dimension))))
    return source[::step, ::step].copy(), target[::step, ::step].copy()


def _normalized_parameters(parameters, mode):
    normalized = {'mode': mode}
    for name in PARAMETER_NAMES:
        minimum, maximum = PARAMETER_BOUNDS[name]
        value = parameters.get(name, DEFAULT_PARAMETERS[name])
        normalized[name] = int(np.clip(np.rint(float(value)), minimum, maximum))
    return normalized


def _candidate_image(source, parameters):
    return image_process(source.copy(), **parameters)


def _score_candidate(image, target, score_image_transform):
    scored_image = image
    compression_metadata = None
    if score_image_transform is not None:
        scored_image, compression_metadata = score_image_transform(image)
    candidate_bgr = np.asarray(scored_image)[:, :, :3]
    target_bgr = np.asarray(target)[:, :, :3]
    return (
        mean_absolute_error(candidate_bgr, target_bgr),
        scored_image,
        compression_metadata,
    )


def tune_photo_editor_color(
    source_image,
    target_image,
    current_parameters=None,
    enabled_parameters=None,
    mode='MacGyver parity',
    max_dimension=DEFAULT_MAX_DIMENSION,
    global_iterations=DEFAULT_GLOBAL_ITERATIONS,
    population_size=DEFAULT_POPULATION_SIZE,
    seed=DEFAULT_SEED,
    local_refinement=True,
    progress_callback=None,
    score_image_transform=None,
):
    """Tune enabled color controls using global and local SciPy searches."""
    if mode not in ('MacGyver parity', 'Standard'):
        raise ValueError(f'unsupported Photo Editor Color mode: {mode}')
    current = _normalized_parameters(current_parameters or {}, mode)
    requested_parameters = (
        PARAMETER_NAMES if enabled_parameters is None else enabled_parameters
    )
    enabled = tuple(
        name for name in requested_parameters
        if name in PARAMETER_NAMES
    )
    source, target = _resize_pair(source_image, target_image, max_dimension)
    evaluated_count = 0
    best_score = None
    best_parameters = None
    best_image = None
    best_compression_metadata = None

    def _evaluate_vector(vector, phase):
        nonlocal evaluated_count, best_score, best_parameters
        nonlocal best_image, best_compression_metadata
        candidate = dict(current)
        for name, value in zip(enabled, vector):
            candidate[name] = value
        candidate = _normalized_parameters(candidate, mode)
        image = _candidate_image(source, candidate)
        score, scored_image, compression_metadata = _score_candidate(
            image,
            target,
            score_image_transform,
        )
        evaluated_count += 1
        if best_score is None or score < best_score:
            best_score = score
            best_parameters = candidate
            best_image = scored_image
            best_compression_metadata = compression_metadata
        if progress_callback is not None:
            progress_callback({
                'phase': phase,
                'candidate_index': evaluated_count,
                'candidate_count': None,
                'score': float(score),
                'best_score': float(best_score),
                'best_parameters': dict(best_parameters),
            })
        return score

    if not enabled:
        _evaluate_vector((), 'fixed')
    else:
        bounds = [PARAMETER_BOUNDS[name] for name in enabled]
        global_result = differential_evolution(
            lambda vector: _evaluate_vector(vector, 'global'),
            bounds,
            maxiter=max(0, int(global_iterations)),
            popsize=max(1, int(population_size)),
            seed=int(seed),
            polish=False,
            updating='immediate',
            workers=1,
        )
        refined_vector = np.asarray(global_result.x, dtype=np.float64)
        if local_refinement:
            local_result = minimize(
                lambda vector: _evaluate_vector(vector, 'local'),
                refined_vector,
                method='Powell',
                bounds=bounds,
                options={'maxiter': max(20, len(enabled) * 12), 'xtol': 0.25},
            )
            refined_vector = np.asarray(local_result.x, dtype=np.float64)
        _evaluate_vector(refined_vector, 'verify')

        rounded = dict(best_parameters)
        for name in enabled:
            center = rounded[name]
            minimum, maximum = PARAMETER_BOUNDS[name]
            for value in range(
                max(int(minimum), center - 1),
                min(int(maximum), center + 1) + 1,
            ):
                neighbor = dict(rounded)
                neighbor[name] = value
                _evaluate_vector([neighbor[field] for field in enabled], 'integer')

    full_parameters = dict(best_parameters)
    full_source, full_target = _resize_pair(source_image, target_image, None)
    full_image = _candidate_image(full_source, full_parameters)
    full_score, full_scored_image, full_compression_metadata = _score_candidate(
        full_image,
        full_target,
        score_image_transform,
    )
    if full_compression_metadata is not None:
        full_parameters['compression_metadata'] = full_compression_metadata
    result = TuneResult(
        best_parameters=full_parameters,
        best_score=float(full_score),
        best_image=full_scored_image,
        evaluated_count=evaluated_count,
    )
    if best_compression_metadata is not None and full_compression_metadata is None:
        result = replace(result, best_parameters={
            **result.best_parameters,
            'compression_metadata': best_compression_metadata,
        })
    return result
