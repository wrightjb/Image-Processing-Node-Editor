#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves node auto-tuning helpers."""

from dataclasses import dataclass

import numpy as np

from auto_tune.service import TuneResult, mean_squared_error
from node.process_node.node_curves import image_process

DEFAULT_MAX_POINTS = 20
DEFAULT_REFINEMENT_ITERATIONS = 2
_LUT_X = np.arange(256, dtype=np.float32)


@dataclass(frozen=True)
class ObservedCurveLut:
    """Robust source-brightness to target-brightness observations."""

    values: np.ndarray
    weights: np.ndarray
    counts: np.ndarray


def _as_uint8_values(image):
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError('image must not be empty')
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


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
    values = np.full(256, np.nan, dtype=np.float32)
    counts = np.bincount(source_flat, minlength=256).astype(np.float32)

    for source_value in np.flatnonzero(counts):
        values[source_value] = np.median(target_flat[source_flat == source_value])

    observed = np.flatnonzero(~np.isnan(values))
    if observed.size == 0:
        raise ValueError('at least one source brightness must be observed')
    if observed.size == 1:
        values[:] = values[observed[0]]
    else:
        missing = np.flatnonzero(np.isnan(values))
        values[missing] = np.interp(missing, observed, values[observed])

    return ObservedCurveLut(
        values=np.clip(values, 0, 255).astype(np.float32),
        weights=np.sqrt(counts).astype(np.float32),
        counts=counts,
    )


def points_to_lut(points):
    """Build the exact LUT shape that the Curves node will interpolate."""
    normalized = _normalize_points(points)
    xs, ys = zip(*normalized)
    return np.interp(_LUT_X, xs, ys).astype(np.float32)


def _normalize_points(points):
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x = int(round(point[0]))
        y = int(round(point[1]))
        normalized.append([max(0, min(255, x)), max(0, min(255, y))])
    if len(normalized) < 2:
        normalized = [[0, 0], [255, 255]]
    by_x = {}
    for x, y in sorted(normalized):
        by_x[x] = y
    normalized = [[x, y] for x, y in sorted(by_x.items())]
    if normalized[0][0] != 0:
        normalized.insert(0, [0, normalized[0][1]])
    if normalized[-1][0] != 255:
        normalized.append([255, normalized[-1][1]])
    return normalized


def _huber_loss(difference, delta=8.0):
    absolute = np.abs(difference)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic * quadratic + delta * linear


def score_curve_lut(candidate_lut, observed_lut, weights=None, metric='balanced_huber'):
    """Score a candidate LUT against observed brightness-band targets."""
    candidate = np.asarray(candidate_lut, dtype=np.float32)
    observed = np.asarray(observed_lut, dtype=np.float32)
    if candidate.shape != (256,) or observed.shape != (256,):
        raise ValueError('candidate and observed LUTs must each have 256 entries')
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
    observed_mask = weights > 0
    if not np.any(observed_mask):
        observed_mask = np.ones(256, dtype=bool)
        weights = np.ones(256, dtype=np.float32)

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


def simplify_curve_points_weighted_rdp(values, weights=None, max_points=DEFAULT_MAX_POINTS):
    """Compress a 256-entry LUT into editable curve points."""
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

    return [[int(x), int(round(float(values[x])))] for x in sorted(selected)]


def _neighbor_points(points, index):
    current_x, current_y = points[index]
    min_x = points[index - 1][0] + 1
    max_x = points[index + 1][0] - 1
    x_values = {current_x}
    if min_x <= current_x - 1:
        x_values.add(current_x - 1)
    if current_x + 1 <= max_x:
        x_values.add(current_x + 1)
    y_values = {
        max(0, min(255, current_y + delta))
        for delta in (-2, -1, 0, 1, 2)
    }
    return sorted(x_values), sorted(y_values)


def refine_curve_points_local_search(
    points,
    observed_lut,
    weights=None,
    metric='balanced_huber',
    iterations=DEFAULT_REFINEMENT_ITERATIONS,
):
    """Locally refine interior point coordinates against the observed LUT."""
    best_points = _normalize_points(points)
    best_score = score_curve_lut(
        points_to_lut(best_points),
        observed_lut,
        weights,
        metric=metric,
    )
    for _round in range(max(0, int(iterations))):
        improved = False
        for index in range(1, len(best_points) - 1):
            for x in _neighbor_points(best_points, index)[0]:
                for y in _neighbor_points(best_points, index)[1]:
                    candidate = [point[:] for point in best_points]
                    candidate[index] = [x, y]
                    candidate = _normalize_points(candidate)
                    if len(candidate) != len(best_points):
                        continue
                    score = score_curve_lut(
                        points_to_lut(candidate),
                        observed_lut,
                        weights,
                        metric=metric,
                    )
                    if score < best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
        if not improved:
            break
    return best_points


def tune_curves(
    source_image,
    target_image,
    max_points=DEFAULT_MAX_POINTS,
    metric_name='balanced_huber',
    refinement_iterations=DEFAULT_REFINEMENT_ITERATIONS,
    progress_callback=None,
):
    """Recover Curves-node points from source/target images."""
    observed = estimate_curve_lut(source_image, target_image)
    initial_points = simplify_curve_points_weighted_rdp(
        observed.values,
        observed.weights,
        max_points=max_points,
    )
    initial_lut = points_to_lut(initial_points)
    initial_score = score_curve_lut(
        initial_lut,
        observed.values,
        observed.weights,
        metric=metric_name,
    )
    if progress_callback is not None:
        progress_callback({
            'candidate_index': 1,
            'candidate_count': 2,
            'parameters': {'points': initial_points},
            'score': float(initial_score),
            'best_score': float(initial_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
            'point_count': len(initial_points),
        })

    refined_points = refine_curve_points_local_search(
        initial_points,
        observed.values,
        observed.weights,
        metric=metric_name,
        iterations=refinement_iterations,
    )
    refined_lut = points_to_lut(refined_points)
    lut_score = score_curve_lut(
        refined_lut,
        observed.values,
        observed.weights,
        metric=metric_name,
    )
    source_uint8 = _as_uint8_values(source_image).copy()
    try:
        best_image = image_process(source_uint8, refined_points)
    except AttributeError:
        best_image = points_to_lut(refined_points).astype(np.uint8)[source_uint8]
    image_score = mean_squared_error(best_image, target_image)
    if progress_callback is not None:
        progress_callback({
            'candidate_index': 2,
            'candidate_count': 2,
            'parameters': {'points': refined_points},
            'score': float(lut_score),
            'best_score': float(lut_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
            'point_count': len(refined_points),
            'image_score': float(image_score),
        })

    return TuneResult(
        best_parameters={
            'points': refined_points,
            'lut_score': float(lut_score),
            'image_score': float(image_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
        },
        best_score=float(lut_score),
        best_image=best_image,
        evaluated_count=2,
    )
