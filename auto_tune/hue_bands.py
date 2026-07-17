#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-tuning helpers for the Hue Bands node."""

import cv2
import numpy as np

from auto_tune.service import TuneResult, mean_squared_error, normalize_image_for_metric
from node.process_node.node_hue_saturation_adjustment import (
    HUE_SHIFT_MAX,
    HUE_SHIFT_MIN,
    _BANDS,
    _get_blend_weight_lut,
    image_process,
)

DEFAULT_REFINEMENT_ITERATIONS = 2
BLEND_CANDIDATES = (1.0, 0.75, 0.5, 0.25, 0.0)
SIMPLIFY_RELATIVE_TOLERANCE = 0.01
SIMPLIFY_ABSOLUTE_TOLERANCE = 1.0e-6


def _resize_pair(source, target, max_size=240):
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


def _match_target_shape(source, target):
    source = np.asarray(source)
    target = np.asarray(target)
    if target.shape == source.shape:
        return target
    return cv2.resize(target, (source.shape[1], source.shape[0]))


def _score(image, target):
    return mean_squared_error(image, target)


def _score_with_transform(image, target, score_image_transform=None):
    scored_image = image
    compression_metadata = None
    if score_image_transform is not None:
        scored_image, compression_metadata = score_image_transform(image)
    return _score(scored_image, target), scored_image, compression_metadata


def _weighted_score(image, target, weight_map):
    candidate = normalize_image_for_metric(image)
    normalized_target = normalize_image_for_metric(target)
    if candidate.shape != normalized_target.shape:
        raise ValueError(
            'candidate and target images must have matching shapes: '
            f'{candidate.shape} != {normalized_target.shape}'
        )
    weights = np.asarray(weight_map, dtype=np.float32)
    if weights.ndim != 2 or weights.shape != candidate.shape[:2]:
        raise ValueError('weight map must match image height and width')
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1.0e-6:
        return _score(image, target)
    pixel_error = np.mean((candidate - normalized_target) ** 2, axis=2)
    return float(np.sum(pixel_error * weights) / weight_sum)


def _band_weight_map(source, band_name, blend):
    band_index = {name: index for index, (name, _center) in enumerate(_BANDS)}[band_name]
    source_hsv = cv2.cvtColor(
        np.asarray(source)[:, :, :3],
        cv2.COLOR_BGR2HSV,
    ).astype(np.float32)
    source_hue = np.clip(source_hsv[:, :, 0].astype(np.int16), 0, 179)
    source_sat = source_hsv[:, :, 1] / 255.0
    return _get_blend_weight_lut(blend)[source_hue][:, :, band_index] * (source_sat ** 2)


def _band_parameter_names(band_name):
    return f'{band_name}_hue_shift', f'{band_name}_saturation'


def _quantize_to_step(value, step):
    step = float(step)
    value = float(value)
    return np.sign(value) * np.floor((abs(value) / step) + 0.5) * step


def _clamp_to_step(value, minimum, maximum, step=0.5):
    value = _quantize_to_step(value, step)
    value = max(float(minimum), min(float(maximum), value))
    return round(value, 1)


def _local_values(center, radius, minimum, maximum, include_zero=True):
    values = {
        _clamp_to_step(center + offset, minimum, maximum)
        for offset in (-radius, -radius / 2.0, 0, radius / 2.0, radius)
    }
    if include_zero and minimum <= 0 <= maximum:
        values.add(0.0)
    return sorted(values)


def _initial_parameters(current_parameters):
    parameters = {'blend': float(current_parameters.get('blend', 1.0))}
    for band_name, _center in _BANDS:
        hue_name, sat_name = _band_parameter_names(band_name)
        parameters[hue_name] = _clamp_to_step(
            current_parameters.get(hue_name, 0),
            HUE_SHIFT_MIN,
            HUE_SHIFT_MAX,
        )
        parameters[sat_name] = _clamp_to_step(
            current_parameters.get(sat_name, 0),
            -100,
            100,
        )
    return parameters


def _parameter_penalty(parameters):
    penalty = abs(float(parameters.get('blend', 1.0)) - 1.0) * 0.01
    for band_name, _center in _BANDS:
        hue_name, sat_name = _band_parameter_names(band_name)
        penalty += abs(float(parameters.get(hue_name, 0))) / 180.0
        penalty += abs(float(parameters.get(sat_name, 0))) / 100.0
    return penalty * 1.0e-7


def _hsv_pair(source, target):
    source_bgr = np.asarray(source)[:, :, :3]
    target_bgr = np.asarray(target)[:, :, :3]
    return (
        cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV).astype(np.float32),
        cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV).astype(np.float32),
    )


def _weighted_least_squares(matrix, observed, sample_weight, ridge=1.0e-3):
    valid = sample_weight > 1.0e-6
    if int(np.count_nonzero(valid)) < matrix.shape[1]:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    weighted_matrix = matrix[valid] * sample_weight[valid, None]
    weighted_observed = observed[valid] * sample_weight[valid]
    if ridge > 0.0:
        regularizer = np.sqrt(float(ridge)) * np.eye(
            matrix.shape[1],
            dtype=np.float32,
        )
        weighted_matrix = np.vstack((weighted_matrix, regularizer))
        weighted_observed = np.concatenate((
            weighted_observed,
            np.zeros(matrix.shape[1], dtype=np.float32),
        ))
    solution, *_unused = np.linalg.lstsq(
        weighted_matrix,
        weighted_observed,
        rcond=None,
    )
    return solution.astype(np.float32)


def _estimate_parameters_from_hsv(source, target, blend):
    source_hsv, target_hsv = _hsv_pair(source, target)
    source_hue = np.clip(source_hsv[:, :, 0].astype(np.int16), 0, 179)
    source_sat = source_hsv[:, :, 1].astype(np.float32)
    target_sat = target_hsv[:, :, 1].astype(np.float32)
    weights = _get_blend_weight_lut(blend)[source_hue].reshape(-1, len(_BANDS))

    hue_delta = target_hsv[:, :, 0] - source_hsv[:, :, 0]
    hue_delta = ((hue_delta + 90.0) % 180.0) - 90.0
    hue_observed = hue_delta.reshape(-1)
    hue_sample_weight = (source_sat.reshape(-1) / 255.0) ** 2
    hue_solution = _weighted_least_squares(
        weights,
        hue_observed,
        hue_sample_weight,
    )

    sat_ratio = (target_sat + 1.0) / (source_sat + 1.0) - 1.0
    sat_observed = sat_ratio.reshape(-1)
    sat_sample_weight = np.maximum(source_sat, target_sat).reshape(-1) / 255.0
    sat_solution = _weighted_least_squares(
        weights,
        sat_observed,
        sat_sample_weight,
    )

    parameters = {'blend': float(blend)}
    for index, (band_name, _center) in enumerate(_BANDS):
        hue_name, sat_name = _band_parameter_names(band_name)
        parameters[hue_name] = _clamp_to_step(
            hue_solution[index],
            HUE_SHIFT_MIN,
            HUE_SHIFT_MAX,
        )
        parameters[sat_name] = _clamp_to_step(sat_solution[index] * 100.0, -100, 100)
    return parameters


def _estimate_candidate_parameters(
    source,
    target,
    tune_blend=False,
    fixed_blend=0.0,
):
    candidates = []
    seen = set()
    fixed_blend = float(np.clip(fixed_blend, 0.0, 1.0))
    blend_candidates = BLEND_CANDIDATES if tune_blend else (fixed_blend,)
    for blend in blend_candidates:
        parameters = _estimate_parameters_from_hsv(source, target, blend)
        key = tuple(parameters.items())
        if key not in seen:
            seen.add(key)
            candidates.append(parameters)
    return candidates


def _neutral_parameter_value(parameter_name, neutral_blend=1.0):
    if parameter_name == 'blend':
        return float(neutral_blend)
    return 0


def _tunable_parameter_names():
    names = ['blend']
    for band_name, _center in _BANDS:
        hue_name, sat_name = _band_parameter_names(band_name)
        names.extend((hue_name, sat_name))
    return names


def _parameter_bounds(parameter_name):
    if parameter_name.endswith('_hue_shift'):
        return HUE_SHIFT_MIN, HUE_SHIFT_MAX
    if parameter_name.endswith('_saturation'):
        return -100, 100
    if parameter_name == 'blend':
        return 0.0, 1.0
    raise KeyError(f'unknown hue bands parameter: {parameter_name}')


def _coordinate_candidate_values(current, step, minimum, maximum):
    values = []
    for direction in (-1, 1):
        value = _clamp_to_step(current + direction * step, minimum, maximum)
        if value != current and value not in values:
            values.append(value)
    return values


def _polish_parameters_full_image(
    parameters,
    source,
    target,
    max_passes=3,
    progress_callback=None,
    score_image_transform=None,
    step_sizes=(1.0, 0.5),
    active_only=True,
):
    """Greedily polish half-step parameters against the full-image score.

    This intentionally mirrors the manual Image Diff workflow: nudge one Hue
    Bands output up or down, keep the direction only when the full-image error
    improves, then repeat at small step sizes.  Earlier refinement uses per-band
    weighted scores for speed, but those local scores can miss cross-band
    interactions and leave active outputs a few integers away from the true
    full-image optimum.  Keep this pass deliberately narrow because every
    accepted or rejected nudge renders the Hue Bands node again.
    """
    polished = dict(parameters)
    best_image = image_process(source, **polished)
    best_score, best_scored_image, _compression_metadata = _score_with_transform(
        best_image,
        target,
        score_image_transform,
    )
    best_image = best_scored_image
    evaluated_count = 0
    names = [name for name in _tunable_parameter_names() if name != 'blend']
    if active_only:
        active_names = [name for name in names if float(polished.get(name, 0.0)) != 0.0]
        if active_names:
            names = active_names

    for step in step_sizes:
        for _pass_index in range(max(0, int(max_passes))):
            improved_in_pass = False
            for name in names:
                minimum, maximum = _parameter_bounds(name)
                parameter_improved = True
                while parameter_improved:
                    parameter_improved = False
                    current = float(polished.get(name, 0.0))
                    candidate_values = _coordinate_candidate_values(
                        current,
                        step,
                        minimum,
                        maximum,
                    )
                    for candidate_index, value in enumerate(candidate_values, start=1):
                        candidate = dict(polished)
                        candidate[name] = value
                        candidate_image = image_process(source, **candidate)
                        candidate_score, candidate_scored_image, _compression_metadata = (
                            _score_with_transform(
                                candidate_image,
                                target,
                                score_image_transform,
                            )
                        )
                        evaluated_count += 1
                        if progress_callback is not None:
                            progress_callback({
                                'phase': f'polish step {step}',
                                'band': name,
                                'candidate_index': candidate_index,
                                'candidate_count': len(candidate_values),
                                'total_evaluated': evaluated_count,
                                'parameters': dict(candidate),
                                'score': float(candidate_score),
                                'best_score': float(best_score),
                                'best_parameters': dict(polished),
                            })
                        if candidate_score + 1.0e-12 < best_score:
                            polished = candidate
                            best_image = candidate_scored_image
                            best_score = candidate_score
                            improved_in_pass = True
                            parameter_improved = True
                            break
            if not improved_in_pass:
                break
    return polished, best_score, best_image, evaluated_count


def _simplify_parameters(
    parameters,
    source,
    target,
    original_score,
    best_score,
    score_callback=None,
    neutral_blend=1.0,
    score_image_transform=None,
):
    """Prune parameters whose visual benefit is too small to justify them."""
    simplified = dict(parameters)
    current_score = float(best_score)
    improvement = max(0.0, float(original_score) - float(best_score))
    tolerance = max(
        SIMPLIFY_ABSOLUTE_TOLERANCE,
        improvement * SIMPLIFY_RELATIVE_TOLERANCE,
    )

    names = sorted(
        _tunable_parameter_names(),
        key=lambda name: abs(float(simplified.get(name, 0.0)))
        if name != 'blend' else abs(float(simplified.get(name, 1.0)) - 1.0),
        reverse=True,
    )
    for name in names:
        neutral_value = _neutral_parameter_value(name, neutral_blend)
        if simplified.get(name, neutral_value) == neutral_value:
            continue
        candidate = dict(simplified)
        candidate[name] = neutral_value
        candidate_image = image_process(source, **candidate)
        candidate_score, _scored_image, _compression_metadata = _score_with_transform(
            candidate_image,
            target,
            score_image_transform,
        )
        if candidate_score <= current_score + tolerance:
            simplified = candidate
            current_score = candidate_score
            if score_callback is not None:
                score_callback(name, current_score)
    return simplified, current_score


def tune_hue_bands(
    source,
    target,
    current_parameters=None,
    refinement_iterations=DEFAULT_REFINEMENT_ITERATIONS,
    progress_callback=None,
    tune_blend=False,
    fixed_blend=0.0,
    score_image_transform=None,
    refine_only=False,
):
    """Tune Hue Bands, optionally refining only the supplied parameters.

    ``refine_only`` skips HSV estimation, blend search, and simplification. It
    starts from ``current_parameters`` and runs the local and full-image refine
    passes, which makes repeated manual-style refinement inexpensive.
    """
    if source is None or target is None:
        raise ValueError('source and target images are required')
    current_parameters = current_parameters or {}
    fixed_blend = float(np.clip(fixed_blend, 0.0, 1.0))
    work_source, work_target = _resize_pair(source, target)

    best_parameters = _initial_parameters(current_parameters)
    best_image = image_process(work_source, **best_parameters)
    best_visual_score, best_scored_image, _compression_metadata = (
        _score_with_transform(best_image, work_target, score_image_transform)
    )
    best_image = best_scored_image
    original_visual_score = best_visual_score
    best_objective = best_visual_score + _parameter_penalty(best_parameters)
    evaluated_count = 1

    def evaluate(parameters, phase, band_name=None, candidate_index=0, candidate_count=0):
        nonlocal best_image, best_parameters, best_visual_score, best_objective
        nonlocal evaluated_count
        candidate_image = image_process(work_source, **parameters)
        candidate_visual_score, candidate_scored_image, _compression_metadata = (
            _score_with_transform(
                candidate_image,
                work_target,
                score_image_transform,
            )
        )
        candidate_objective = candidate_visual_score + _parameter_penalty(parameters)
        evaluated_count += 1
        if candidate_objective < best_objective:
            best_objective = candidate_objective
            best_visual_score = candidate_visual_score
            best_parameters = dict(parameters)
            best_image = candidate_scored_image
        if progress_callback is not None:
            progress_callback({
                'phase': phase,
                'band': band_name,
                'candidate_index': candidate_index,
                'candidate_count': candidate_count,
                'total_evaluated': evaluated_count,
                'parameters': dict(parameters),
                'score': float(candidate_visual_score),
                'best_score': float(best_visual_score),
                'best_parameters': dict(best_parameters),
            })

    if not refine_only:
        estimate_candidates = _estimate_candidate_parameters(
            work_source,
            work_target,
            tune_blend=tune_blend,
            fixed_blend=fixed_blend,
        )
        for index, parameters in enumerate(estimate_candidates, start=1):
            evaluate(parameters, 'estimate', None, index, len(estimate_candidates))

    for round_index in range(max(0, int(refinement_iterations))):
        hue_radius = max(2, 12 // (round_index + 1))
        sat_radius = max(2, 12 // (round_index + 1))
        for band_name, _center in _BANDS:
            hue_name, sat_name = _band_parameter_names(band_name)
            hue_values = _local_values(
                best_parameters[hue_name],
                hue_radius,
                HUE_SHIFT_MIN,
                HUE_SHIFT_MAX,
            )
            sat_values = _local_values(best_parameters[sat_name], sat_radius, -100, 100)
            candidates = [(hue, sat) for hue in hue_values for sat in sat_values]
            band_weights = _band_weight_map(
                work_source,
                band_name,
                best_parameters.get('blend', 1.0),
            )
            local_best_parameters = dict(best_parameters)
            local_best_image = best_image
            local_best_visual_score = best_visual_score
            local_best_score = _weighted_score(best_image, work_target, band_weights)
            local_best_objective = (
                local_best_score + _parameter_penalty(local_best_parameters)
            )
            for index, (hue_value, sat_value) in enumerate(candidates, start=1):
                candidate = dict(best_parameters)
                candidate[hue_name] = hue_value
                candidate[sat_name] = sat_value
                candidate_image = image_process(work_source, **candidate)
                candidate_visual_score, candidate_scored_image, _compression_metadata = (
                    _score_with_transform(
                        candidate_image,
                        work_target,
                        score_image_transform,
                    )
                )
                candidate_score = _weighted_score(
                    candidate_scored_image,
                    work_target,
                    band_weights,
                )
                candidate_objective = candidate_score + _parameter_penalty(candidate)
                if candidate_objective < local_best_objective:
                    local_best_objective = candidate_objective
                    local_best_score = candidate_score
                    local_best_parameters = dict(candidate)
                    local_best_image = candidate_scored_image
                    local_best_visual_score = candidate_visual_score
                if progress_callback is not None:
                    progress_callback({
                        'phase': f'refine {round_index + 1}',
                        'band': band_name,
                        'candidate_index': index,
                        'candidate_count': len(candidates),
                        'total_evaluated': evaluated_count,
                        'parameters': dict(candidate),
                        'score': float(candidate_visual_score),
                        'band_score': float(candidate_score),
                        'best_score': float(best_visual_score),
                        'best_parameters': dict(best_parameters),
                    })
                evaluated_count += 1
            if local_best_parameters != best_parameters:
                best_parameters = local_best_parameters
                best_image = local_best_image
                best_visual_score = local_best_visual_score
                best_objective = best_visual_score + _parameter_penalty(best_parameters)

    if not refine_only:
        if tune_blend:
            for index, blend in enumerate(BLEND_CANDIDATES, start=1):
                candidate = dict(best_parameters)
                candidate['blend'] = blend
                evaluate(candidate, 'blend', None, index, len(BLEND_CANDIDATES))
        else:
            best_parameters['blend'] = fixed_blend

    full_source = np.asarray(source)
    full_target = _match_target_shape(full_source, target)
    original_full_image = image_process(
        full_source,
        **_initial_parameters(current_parameters),
    )
    original_full_score, _scored_image, _compression_metadata = _score_with_transform(
        original_full_image,
        full_target,
        score_image_transform,
    )
    def _progress_polish(update):
        if progress_callback is not None:
            progress_callback(update)

    (
        best_parameters,
        best_visual_score,
        best_image,
        polish_evaluated_count,
    ) = _polish_parameters_full_image(
        best_parameters,
        full_source,
        full_target,
        progress_callback=_progress_polish,
        score_image_transform=score_image_transform,
    )
    evaluated_count += polish_evaluated_count

    def _progress_simplify(parameter_name, simplified_score):
        if progress_callback is None:
            return
        progress_callback({
            'phase': 'simplify',
            'band': parameter_name,
            'candidate_index': 0,
            'candidate_count': 0,
            'total_evaluated': evaluated_count,
            'parameters': dict(best_parameters),
            'score': float(simplified_score),
            'best_score': float(simplified_score),
            'best_parameters': dict(best_parameters),
        })

    if not refine_only:
        best_parameters, best_visual_score = _simplify_parameters(
            best_parameters,
            full_source,
            full_target,
            original_full_score,
            best_visual_score,
            score_callback=_progress_simplify,
            neutral_blend=fixed_blend if not tune_blend else 1.0,
            score_image_transform=score_image_transform,
        )

    full_best_image = image_process(full_source, **best_parameters)
    full_score, full_scored_image, compression_metadata = _score_with_transform(
        full_best_image,
        full_target,
        score_image_transform,
    )
    best_parameters = dict(best_parameters)
    best_parameters['compression_metadata'] = compression_metadata
    return TuneResult(
        best_parameters=best_parameters,
        best_score=float(full_score),
        best_image=full_scored_image,
        evaluated_count=evaluated_count,
    )
