#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-tuning helpers for the Hue Bands node."""

import cv2
import numpy as np

from auto_tune.service import TuneResult, mean_squared_error
from node.process_node.node_hue_saturation_adjustment import _BANDS, image_process

DEFAULT_REFINEMENT_ITERATIONS = 2
SATURATION_CANDIDATES = (-100, -75, -50, -25, 0, 25, 50, 75, 100)
HUE_CANDIDATES = (-180, -120, -60, -30, 0, 30, 60, 120, 180)


def _resize_pair(source, target, max_size=160):
    source = np.asarray(source)
    target = np.asarray(target)
    if source.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (source.shape[1], source.shape[0]))
    height, width = source.shape[:2]
    scale = min(1.0, float(max_size) / float(max(height, width)))
    if scale >= 1.0:
        return source, target
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return (
        cv2.resize(source, new_size, interpolation=cv2.INTER_AREA),
        cv2.resize(target, new_size, interpolation=cv2.INTER_AREA),
    )


def _score(image, target):
    return mean_squared_error(image, target)


def _band_parameter_names(band_name):
    return f'{band_name}_hue_shift', f'{band_name}_saturation'


def _clamp_int(value, minimum, maximum):
    return int(max(minimum, min(maximum, round(float(value)))))


def _local_values(center, radius, minimum, maximum):
    values = {
        _clamp_int(center + offset, minimum, maximum)
        for offset in (-radius, -radius // 2, 0, radius // 2, radius)
    }
    return sorted(values)


def _initial_parameters(current_parameters):
    parameters = {'blend': float(current_parameters.get('blend', 1.0))}
    for band_name, _center in _BANDS:
        hue_name, sat_name = _band_parameter_names(band_name)
        parameters[hue_name] = _clamp_int(current_parameters.get(hue_name, 0), -180, 180)
        parameters[sat_name] = _clamp_int(current_parameters.get(sat_name, 0), -100, 100)
    return parameters


def tune_hue_bands(
    source,
    target,
    current_parameters=None,
    refinement_iterations=DEFAULT_REFINEMENT_ITERATIONS,
    progress_callback=None,
):
    """Tune Hue Bands with per-band coordinate search.

    The search intentionally optimizes one hue band at a time because the node's
    bands are mostly separated by the hue-weight lookup table. A small joint
    blend pass and local refinements keep the search tractable while still
    improving interactions at band boundaries.
    """
    if source is None or target is None:
        raise ValueError('source and target images are required')
    current_parameters = current_parameters or {}
    best_parameters = _initial_parameters(current_parameters)
    work_source, work_target = _resize_pair(source, target)
    best_image = image_process(work_source, **best_parameters)
    best_score = _score(best_image, work_target)
    evaluated_count = 1

    def evaluate(parameters, phase, band_name=None, candidate_index=0, candidate_count=0):
        nonlocal best_image, best_parameters, best_score, evaluated_count
        candidate_image = image_process(work_source, **parameters)
        candidate_score = _score(candidate_image, work_target)
        evaluated_count += 1
        if candidate_score < best_score:
            best_score = candidate_score
            best_parameters = dict(parameters)
            best_image = candidate_image
        if progress_callback is not None:
            progress_callback({
                'phase': phase,
                'band': band_name,
                'candidate_index': candidate_index,
                'candidate_count': candidate_count,
                'total_evaluated': evaluated_count,
                'parameters': dict(parameters),
                'score': float(candidate_score),
                'best_score': float(best_score),
                'best_parameters': dict(best_parameters),
            })

    for band_name, _center in _BANDS:
        hue_name, sat_name = _band_parameter_names(band_name)
        candidates = [(hue, sat) for hue in HUE_CANDIDATES for sat in SATURATION_CANDIDATES]
        for index, (hue_value, sat_value) in enumerate(candidates, start=1):
            candidate = dict(best_parameters)
            candidate[hue_name] = hue_value
            candidate[sat_name] = sat_value
            evaluate(candidate, 'coarse', band_name, index, len(candidates))

    for round_index in range(max(0, int(refinement_iterations))):
        hue_radius = max(4, 30 // (round_index + 1))
        sat_radius = max(5, 25 // (round_index + 1))
        for band_name, _center in _BANDS:
            hue_name, sat_name = _band_parameter_names(band_name)
            hue_values = _local_values(best_parameters[hue_name], hue_radius, -180, 180)
            sat_values = _local_values(best_parameters[sat_name], sat_radius, -100, 100)
            candidates = [(hue, sat) for hue in hue_values for sat in sat_values]
            for index, (hue_value, sat_value) in enumerate(candidates, start=1):
                candidate = dict(best_parameters)
                candidate[hue_name] = hue_value
                candidate[sat_name] = sat_value
                evaluate(candidate, f'refine {round_index + 1}', band_name, index, len(candidates))

    for index, blend in enumerate((0.0, 0.25, 0.5, 0.75, 1.0), start=1):
        candidate = dict(best_parameters)
        candidate['blend'] = blend
        evaluate(candidate, 'blend', None, index, 5)

    full_best_image = image_process(np.asarray(source), **best_parameters)
    full_score = _score(full_best_image, np.asarray(target) if np.asarray(target).shape == np.asarray(source).shape else cv2.resize(np.asarray(target), (np.asarray(source).shape[1], np.asarray(source).shape[0])))
    return TuneResult(
        best_parameters=best_parameters,
        best_score=float(full_score),
        best_image=full_best_image,
        evaluated_count=evaluated_count,
    )
