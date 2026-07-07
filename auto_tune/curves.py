#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves node auto-tuning helpers."""

from dataclasses import dataclass

import numpy as np

from auto_tune.service import TuneResult, mean_squared_error
from node.process_node.node_curves import image_process

DEFAULT_MAX_POINTS = 20
DEFAULT_REFINEMENT_ITERATIONS = 2
DEFAULT_COMPLEXITY_PENALTY = 0.25
_LUT_X = np.arange(256, dtype=np.float32)


@dataclass(frozen=True)
class ObservedCurveLut:
    """Robust source-brightness to target-brightness observations."""

    values: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    observed_values: np.ndarray
    observed_mask: np.ndarray


def _as_uint8_values(image):
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError('image must not be empty')
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _filled_values(observed_values):
    filled = np.asarray(observed_values, dtype=np.float32).copy()
    observed = np.flatnonzero(~np.isnan(filled))
    if observed.size == 0:
        raise ValueError('at least one source brightness must be observed')
    if observed.size == 1:
        filled[:] = filled[observed[0]]
    else:
        missing = np.flatnonzero(np.isnan(filled))
        filled[missing] = np.interp(missing, observed, filled[observed])
    return np.clip(filled, 0, 255).astype(np.float32)


def estimate_curve_lut(source_image, target_image):
    """Estimate a 256-entry curves LUT from matching source/target pixels."""
    source = _as_uint8_values(source_image)
    target = _as_uint8_values(target_image)
    if source.shape != target.shape:
        raise ValueError(
            'source and target images must have matching shapes: '
            f'{source.shape} != {target.shape}'
        )

    source_flat = source.reshape(-1)
    target_flat = target.reshape(-1).astype(np.float32)
    observed_values = np.full(256, np.nan, dtype=np.float32)
    counts = np.bincount(source_flat, minlength=256).astype(np.float32)

    for source_value in np.flatnonzero(counts):
        observed_values[source_value] = np.median(
            target_flat[source_flat == source_value],
        )

    observed_mask = counts > 0
    values = _filled_values(observed_values)
    return ObservedCurveLut(
        values=values,
        weights=np.sqrt(counts).astype(np.float32),
        counts=counts,
        observed_values=observed_values,
        observed_mask=observed_mask,
    )


def _normalize_points(points):
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        normalized.append([max(0.0, min(255.0, x)), max(0.0, min(255.0, y))])
    if len(normalized) < 2:
        normalized = [[0.0, 0.0], [255.0, 255.0]]

    by_x = {}
    for x, y in sorted(normalized):
        by_x[round(x, 6)] = y
    normalized = [[float(x), float(y)] for x, y in sorted(by_x.items())]
    if normalized[0][0] != 0.0:
        normalized.insert(0, [0.0, normalized[0][1]])
    if normalized[-1][0] != 255.0:
        normalized.append([255.0, normalized[-1][1]])
    return normalized


def _format_points(points):
    formatted = []
    for x, y in _normalize_points(points):
        rounded_x = round(float(x), 3)
        rounded_y = round(float(y), 3)
        if rounded_x.is_integer():
            rounded_x = int(rounded_x)
        if rounded_y.is_integer():
            rounded_y = int(rounded_y)
        formatted.append([rounded_x, rounded_y])
    return formatted


def points_to_lut(points, quantize=False):
    """Build the LUT shape that the Curves node will interpolate."""
    normalized = _normalize_points(points)
    xs, ys = zip(*normalized)
    lut = np.interp(_LUT_X, xs, ys).astype(np.float32)
    lut = np.clip(lut, 0, 255)
    if quantize:
        return lut.astype(np.uint8).astype(np.float32)
    return lut


def _huber_loss(difference, delta=8.0):
    absolute = np.abs(difference)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic * quadratic + delta * linear


def score_curve_lut(
    candidate_lut,
    observed_lut,
    weights=None,
    metric='balanced_huber',
    observed_mask=None,
):
    """Score a candidate LUT against observed brightness-band targets."""
    candidate = np.asarray(candidate_lut, dtype=np.float32)
    observed = np.asarray(observed_lut, dtype=np.float32)
    if candidate.shape != (256,) or observed.shape != (256,):
        raise ValueError('candidate and observed LUTs must each have 256 entries')

    if observed_mask is None:
        observed_mask = ~np.isnan(observed)
    else:
        observed_mask = np.asarray(observed_mask, dtype=bool)
    if not np.any(observed_mask):
        observed_mask = np.ones(256, dtype=bool)
    observed = np.where(np.isnan(observed), candidate, observed)

    difference = candidate - observed
    if metric in ('mae', 'balanced_mae'):
        loss = np.abs(difference)
    elif metric == 'mse':
        loss = difference * difference
    else:
        loss = _huber_loss(difference)

    if weights is None:
        weights = np.ones(256, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.where(observed_mask, weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = observed_mask.astype(np.float32)

    weighted = float(np.sum(loss * weights) / max(float(np.sum(weights)), 1.0))
    if metric.startswith('balanced_') or metric == 'balanced_huber':
        equal_band = float(np.mean(loss[observed_mask]))
        return (0.7 * weighted) + (0.3 * equal_band)
    return weighted


def _rdp_error(values, weights, start, end):
    if end <= start + 1:
        return None
    x0 = float(start)
    x1 = float(end)
    y0 = float(values[start])
    y1 = float(values[end])
    indices = np.arange(start + 1, end, dtype=np.float32)
    line = y0 + ((indices - x0) / (x1 - x0)) * (y1 - y0)
    errors = np.abs(values[start + 1:end] - line) * (1.0 + weights[start + 1:end])
    if errors.size == 0:
        return None
    offset = int(np.argmax(errors))
    return float(errors[offset]), start + 1 + offset


def rdp_curve_x_positions(values, weights=None, max_points=DEFAULT_MAX_POINTS):
    """Choose important x positions from a 256-entry helper LUT."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (256,):
        raise ValueError('values must contain exactly 256 entries')
    if weights is None:
        weights = np.ones(256, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    max_points = max(2, int(max_points))

    selected = {0, 255}
    while len(selected) < max_points:
        ordered = sorted(selected)
        best = None
        for start, end in zip(ordered, ordered[1:]):
            candidate = _rdp_error(values, weights, start, end)
            if candidate is None:
                continue
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or best[1] in selected:
            break
        selected.add(best[1])
    return sorted(selected)


def _basis_for_x_positions(x_positions, sample_x):
    x_positions = np.asarray(x_positions, dtype=np.float32)
    sample_x = np.asarray(sample_x, dtype=np.float32)
    basis = np.zeros((sample_x.size, x_positions.size), dtype=np.float32)
    for row, x in enumerate(sample_x):
        if x <= x_positions[0]:
            basis[row, 0] = 1.0
            continue
        if x >= x_positions[-1]:
            basis[row, -1] = 1.0
            continue
        right = int(np.searchsorted(x_positions, x, side='right'))
        left = right - 1
        span = x_positions[right] - x_positions[left]
        if span <= 0:
            basis[row, left] = 1.0
            continue
        t = (x - x_positions[left]) / span
        basis[row, left] = 1.0 - t
        basis[row, right] = t
    return basis


def fit_curve_y_values(
    x_positions,
    observed_values,
    weights=None,
    observed_mask=None,
    prior_values=None,
    prior_weight=1e-4,
):
    """Fit point y-values for fixed x positions against observed bins."""
    x_positions = sorted({int(round(float(x))) for x in x_positions})
    if x_positions[0] != 0:
        x_positions.insert(0, 0)
    if x_positions[-1] != 255:
        x_positions.append(255)

    observed_values = np.asarray(observed_values, dtype=np.float32)
    if observed_mask is None:
        observed_mask = ~np.isnan(observed_values)
    else:
        observed_mask = np.asarray(observed_mask, dtype=bool)
    sample_x = np.flatnonzero(observed_mask).astype(np.float32)
    if sample_x.size == 0:
        sample_x = _LUT_X.copy()
        observed_y = _filled_values(observed_values)
    else:
        observed_y = observed_values[observed_mask]

    basis = _basis_for_x_positions(x_positions, sample_x)
    if weights is None:
        sample_weights = np.ones(sample_x.size, dtype=np.float32)
    else:
        sample_weights = np.asarray(weights, dtype=np.float32)[sample_x.astype(int)]
        if float(np.sum(sample_weights)) <= 0.0:
            sample_weights = np.ones(sample_x.size, dtype=np.float32)

    weighted_basis = basis * np.sqrt(sample_weights)[:, None]
    weighted_y = observed_y * np.sqrt(sample_weights)

    if prior_values is None:
        prior_values = _filled_values(observed_values)
    prior_lut = np.asarray(prior_values, dtype=np.float32)
    prior_y = np.interp(x_positions, _LUT_X, prior_lut).astype(np.float32)
    if prior_weight > 0:
        prior_basis = np.eye(len(x_positions), dtype=np.float32) * np.sqrt(prior_weight)
        weighted_basis = np.vstack([weighted_basis, prior_basis])
        weighted_y = np.concatenate([weighted_y, prior_y * np.sqrt(prior_weight)])

    fitted_y, *_ = np.linalg.lstsq(weighted_basis, weighted_y, rcond=None)
    fitted_y = np.clip(fitted_y, 0, 255)
    return _format_points(zip(x_positions, fitted_y))


def _score_points(points, observed, metric='balanced_huber'):
    lut = points_to_lut(points, quantize=True)
    return score_curve_lut(
        lut,
        observed.observed_values,
        observed.weights,
        metric=metric,
        observed_mask=observed.observed_mask,
    )


def simplify_curve_points_weighted_rdp(values, weights=None, max_points=DEFAULT_MAX_POINTS):
    """Compress a 256-entry LUT into editable curve points."""
    x_positions = rdp_curve_x_positions(values, weights, max_points=max_points)
    observed_mask = np.ones(256, dtype=bool)
    return fit_curve_y_values(
        x_positions,
        values,
        weights=weights,
        observed_mask=observed_mask,
        prior_values=values,
    )


def _refit_points(points, observed):
    return fit_curve_y_values(
        [point[0] for point in points],
        observed.observed_values,
        weights=observed.weights,
        observed_mask=observed.observed_mask,
        prior_values=observed.values,
    )


def refine_curve_points_local_search(
    points,
    observed_lut,
    weights=None,
    metric='balanced_huber',
    iterations=DEFAULT_REFINEMENT_ITERATIONS,
    observed_mask=None,
):
    """Locally refine interior point x positions, refitting y-values each time."""
    observed = ObservedCurveLut(
        values=_filled_values(observed_lut),
        weights=np.ones(256, dtype=np.float32) if weights is None else weights,
        counts=np.ones(256, dtype=np.float32) if weights is None else weights * weights,
        observed_values=np.asarray(observed_lut, dtype=np.float32),
        observed_mask=(~np.isnan(observed_lut) if observed_mask is None else observed_mask),
    )
    best_points = _refit_points(points, observed)
    best_score = _score_points(best_points, observed, metric=metric)
    radii = (64, 32, 16, 8, 4, 2, 1)
    for _round in range(max(0, int(iterations))):
        improved = False
        for radius in radii:
            for index in range(1, len(best_points) - 1):
                current_x = int(round(float(best_points[index][0])))
                min_x = int(round(float(best_points[index - 1][0]))) + 1
                max_x = int(round(float(best_points[index + 1][0]))) - 1
                candidates = {
                    current_x,
                    max(min_x, current_x - radius),
                    min(max_x, current_x + radius),
                }
                for candidate_x in sorted(candidates):
                    if candidate_x == current_x or not (min_x <= candidate_x <= max_x):
                        continue
                    candidate = [point[:] for point in best_points]
                    candidate[index][0] = candidate_x
                    candidate = _refit_points(candidate, observed)
                    score = _score_points(candidate, observed, metric=metric)
                    if score < best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
        if not improved:
            break
    return best_points


def prune_curve_points(
    points,
    observed,
    metric='balanced_huber',
    complexity_penalty=DEFAULT_COMPLEXITY_PENALTY,
    min_points=2,
):
    """Remove points that do not improve observed-fit enough to justify complexity."""
    best_points = _refit_points(points, observed)
    best_score = _score_points(best_points, observed, metric=metric)
    while len(best_points) > max(2, int(min_points)):
        best_removal = None
        for index in range(1, len(best_points) - 1):
            candidate = [point[:] for i, point in enumerate(best_points) if i != index]
            candidate = _refit_points(candidate, observed)
            score = _score_points(candidate, observed, metric=metric)
            score_increase = score - best_score
            if best_removal is None or score_increase < best_removal[0]:
                best_removal = (score_increase, score, candidate)
        if best_removal is None or best_removal[0] > complexity_penalty:
            break
        _increase, best_score, best_points = best_removal
    return best_points


def tune_curves(
    source_image,
    target_image,
    max_points=DEFAULT_MAX_POINTS,
    metric_name='balanced_huber',
    refinement_iterations=DEFAULT_REFINEMENT_ITERATIONS,
    progress_callback=None,
    complexity_penalty=DEFAULT_COMPLEXITY_PENALTY,
):
    """Recover Curves-node points from source/target images."""
    observed = estimate_curve_lut(source_image, target_image)
    x_positions = rdp_curve_x_positions(
        observed.values,
        observed.weights,
        max_points=max_points,
    )
    initial_points = fit_curve_y_values(
        x_positions,
        observed.observed_values,
        weights=observed.weights,
        observed_mask=observed.observed_mask,
        prior_values=observed.values,
    )
    initial_score = _score_points(initial_points, observed, metric=metric_name)
    if progress_callback is not None:
        progress_callback({
            'candidate_index': 1,
            'candidate_count': 3,
            'parameters': {'points': initial_points},
            'score': float(initial_score),
            'best_score': float(initial_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
            'point_count': len(initial_points),
        })

    refined_points = refine_curve_points_local_search(
        initial_points,
        observed.observed_values,
        observed.weights,
        metric=metric_name,
        iterations=refinement_iterations,
        observed_mask=observed.observed_mask,
    )
    refined_points = _refit_points(refined_points, observed)
    refined_score = _score_points(refined_points, observed, metric=metric_name)
    if progress_callback is not None:
        progress_callback({
            'candidate_index': 2,
            'candidate_count': 3,
            'parameters': {'points': refined_points},
            'score': float(refined_score),
            'best_score': float(refined_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
            'point_count': len(refined_points),
        })

    pruned_points = prune_curve_points(
        refined_points,
        observed,
        metric=metric_name,
        complexity_penalty=complexity_penalty,
    )
    lut_score = _score_points(pruned_points, observed, metric=metric_name)
    source_uint8 = _as_uint8_values(source_image).copy()
    try:
        best_image = image_process(source_uint8, pruned_points)
    except AttributeError:
        best_image = points_to_lut(pruned_points, quantize=True).astype(np.uint8)[source_uint8]
    image_score = mean_squared_error(best_image, target_image)
    if progress_callback is not None:
        progress_callback({
            'candidate_index': 3,
            'candidate_count': 3,
            'parameters': {'points': pruned_points},
            'score': float(lut_score),
            'best_score': float(lut_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
            'point_count': len(pruned_points),
            'image_score': float(image_score),
        })

    return TuneResult(
        best_parameters={
            'points': pruned_points,
            'lut_score': float(lut_score),
            'image_score': float(image_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
        },
        best_score=float(lut_score),
        best_image=best_image,
        evaluated_count=3,
    )
