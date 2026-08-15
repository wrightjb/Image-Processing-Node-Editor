#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curves node auto-tuning helpers."""

from dataclasses import dataclass

import numpy as np

from auto_tune.service import TuneResult, mean_squared_error
from node.curves_points_ui import CURVE_CHANNELS, CurvesPointsEditorMixin
from node.curve_interpolation import INTERPOLATION_LINEAR
from node.curve_interpolation import INTERPOLATION_PARAMETRIC_SPLINE
from node.curve_interpolation import normalize_interpolation
from node.curve_interpolation import points_to_lut as node_points_to_lut
from node.process_node.node_curves import image_process

DEFAULT_MAX_POINTS = 20
DEFAULT_REFINEMENT_ITERATIONS = 2
DEFAULT_COMPLEXITY_PENALTY = 0.25
DEFAULT_POINT_PRECISION = 3
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


def _normalize_channel(channel):
    if isinstance(channel, str) and channel in CURVE_CHANNELS:
        return channel
    return 'White'


def _single_channel_curve_set(channel, points):
    helper = CurvesPointsEditorMixin()
    curve_set = helper._default_curve_set()
    curve_set[_normalize_channel(channel)] = points
    return curve_set


def _select_channel(image, channel='White'):
    channel = _normalize_channel(channel)
    array = np.asarray(image)
    if channel == 'White' or array.ndim != 3 or array.shape[2] < 3:
        return image
    return array[:, :, {'Blue': 0, 'Green': 1, 'Red': 2}[channel]]


def estimate_curve_lut(source_image, target_image, channel='White'):
    """Estimate a 256-entry curves LUT from matching source/target pixels."""
    source = _as_uint8_values(_select_channel(source_image, channel))
    target = _as_uint8_values(_select_channel(target_image, channel))
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


def _format_points(points, precision=DEFAULT_POINT_PRECISION):
    formatted = []
    precision = max(0, int(precision))
    for x, y in _normalize_points(points):
        rounded_x = round(float(x), precision)
        rounded_y = round(float(y), precision)
        if rounded_x.is_integer():
            rounded_x = int(rounded_x)
        if rounded_y.is_integer():
            rounded_y = int(rounded_y)
        formatted.append([rounded_x, rounded_y])
    return formatted


def points_to_lut(points, quantize=False, interpolation=INTERPOLATION_LINEAR):
    """Build the LUT shape that the Curves node will interpolate."""
    normalized = _normalize_points(points)
    interpolation = normalize_interpolation(interpolation)
    if interpolation != INTERPOLATION_LINEAR:
        lut = node_points_to_lut(
            normalized,
            interpolation=interpolation,
        ).astype(np.float32)
    else:
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
    precision=DEFAULT_POINT_PRECISION,
):
    """Fit point y-values for fixed x positions against observed bins."""
    x_positions = sorted({round(float(x), 6) for x in x_positions})
    if x_positions[0] != 0.0:
        x_positions.insert(0, 0.0)
    if x_positions[-1] != 255.0:
        x_positions.append(255.0)

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
    return _format_points(zip(x_positions, fitted_y), precision=precision)


def _score_points(points, observed, metric='balanced_huber', interpolation=INTERPOLATION_LINEAR):
    lut = points_to_lut(points, quantize=True, interpolation=interpolation)
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


def optimize_spline_y_values(
    points,
    observed,
    metric='balanced_huber',
    precision=DEFAULT_POINT_PRECISION,
    max_sweeps=1,
    interpolation=INTERPOLATION_PARAMETRIC_SPLINE,
):
    """Coordinate-refine y-values using the spline-rendered LUT as objective."""
    interpolation = normalize_interpolation(interpolation)
    best_points = _format_points(points, precision=precision)
    best_score = _score_points(
        best_points,
        observed,
        metric=metric,
        interpolation=interpolation,
    )
    radii = (32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25)
    for radius in radii:
        for _sweep in range(max(1, int(max_sweeps))):
            improved = False
            for index in range(len(best_points)):
                current_y = float(best_points[index][1])
                for candidate_y in (
                    max(0.0, current_y - radius),
                    min(255.0, current_y + radius),
                ):
                    if candidate_y == current_y:
                        continue
                    candidate = [point[:] for point in best_points]
                    candidate[index][1] = candidate_y
                    candidate = _format_points(candidate, precision=precision)
                    score = _score_points(
                        candidate,
                        observed,
                        metric=metric,
                        interpolation=interpolation,
                    )
                    if score < best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
                        break
            if not improved:
                break
    return best_points


def _refit_points(
    points,
    observed,
    precision=DEFAULT_POINT_PRECISION,
    metric='balanced_huber',
    interpolation=INTERPOLATION_LINEAR,
):
    fitted = fit_curve_y_values(
        [point[0] for point in points],
        observed.observed_values,
        weights=observed.weights,
        observed_mask=observed.observed_mask,
        prior_values=observed.values,
        precision=precision,
    )
    if interpolation != INTERPOLATION_LINEAR:
        return optimize_spline_y_values(
            fitted,
            observed,
            metric=metric,
            precision=precision,
            interpolation=interpolation,
        )
    return fitted


def fit_spline_points_additive(
    observed,
    max_points=DEFAULT_MAX_POINTS,
    metric='balanced_huber',
    complexity_penalty=DEFAULT_COMPLEXITY_PENALTY,
    precision=DEFAULT_POINT_PRECISION,
    progress_callback=None,
    interpolation=INTERPOLATION_PARAMETRIC_SPLINE,
):
    """Build spline controls additively by inserting the worst residual x bin."""
    interpolation = normalize_interpolation(interpolation)
    max_points = max(2, int(max_points))
    selected_x = [0.0, 255.0]
    best_points = _refit_points(
        [[0.0, 0.0], [255.0, 255.0]],
        observed,
        precision=precision,
        metric=metric,
        interpolation=interpolation,
    )
    best_score = _score_points(
        best_points,
        observed,
        metric=metric,
        interpolation=interpolation,
    )
    while len(selected_x) < max_points:
        lut = points_to_lut(
            best_points,
            quantize=True,
            interpolation=interpolation,
        )
        residual = np.abs(lut - observed.values)
        candidate_mask = observed.observed_mask.copy()
        candidate_indexes = np.arange(256)
        for x_value in selected_x:
            candidate_mask &= np.abs(candidate_indexes - int(round(x_value))) >= 8
        if not np.any(candidate_mask):
            break

        weighted_residual = np.where(
            candidate_mask,
            residual * (1.0 + observed.weights),
            -1.0,
        )
        candidate_order = np.argsort(weighted_residual)[::-1]
        candidate_x_values = [
            int(candidate_x)
            for candidate_x in candidate_order[:min(8, np.count_nonzero(candidate_mask))]
            if weighted_residual[int(candidate_x)] >= 0.0
        ]
        best_candidate = None
        for candidate_x in candidate_x_values:
            next_x = sorted({*selected_x, float(candidate_x)})
            candidate = _refit_points(
                [[x, float(np.interp(x, _LUT_X, observed.values))] for x in next_x],
                observed,
                precision=precision,
                metric=metric,
                interpolation=interpolation,
            )
            score = _score_points(
                candidate,
                observed,
                metric=metric,
                interpolation=interpolation,
            )
            if best_candidate is None or score < best_candidate[0]:
                best_candidate = (score, next_x, candidate)
        if best_candidate is None:
            break
        score, next_x, candidate = best_candidate
        improvement = best_score - score
        if improvement <= complexity_penalty and len(best_points) >= 3:
            break
        selected_x = next_x
        best_points = candidate
        best_score = score
        if progress_callback is not None:
            progress_callback({
                'phase': 'spline_add',
                'parameters': {'points': best_points},
                'score': float(best_score),
                'best_score': float(best_score),
                'observed_bins': int(np.count_nonzero(observed.counts)),
                'point_count': len(best_points),
            })
    return best_points


def fit_spline_points_from_dense_reconstruction(
    observed,
    max_points=DEFAULT_MAX_POINTS,
    metric='balanced_huber',
    complexity_penalty=DEFAULT_COMPLEXITY_PENALTY,
    precision=DEFAULT_POINT_PRECISION,
    progress_callback=None,
    interpolation=INTERPOLATION_PARAMETRIC_SPLINE,
    select_by_complexity=True,
):
    """Seed spline controls from inflections in the dense reconstructed LUT.

    When ``select_by_complexity`` is false, ``max_points`` is the requested
    point count rather than merely an upper bound. This is the behavior users
    select by disabling the Auto Tune node's ``Prune Points`` option.
    """
    interpolation = normalize_interpolation(interpolation)
    max_points = max(2, int(max_points))
    dense_lut = np.asarray(observed.values, dtype=np.float32)
    smooth_kernel = np.ones(9, dtype=np.float32) / 9.0
    smoothed_lut = np.convolve(
        np.pad(dense_lut, (4, 4), mode='edge'),
        smooth_kernel,
        mode='valid',
    ).astype(np.float32)
    first_derivative = np.gradient(smoothed_lut)
    second_derivative = np.gradient(first_derivative)
    curvature = np.abs(second_derivative)
    observed_curvature = curvature[observed.observed_mask]
    if observed_curvature.size:
        threshold = max(float(np.percentile(observed_curvature, 80)), 0.25)
    else:
        threshold = 0.25

    sign = np.sign(second_derivative)
    sign[sign == 0] = 1
    inflection_indexes = np.flatnonzero(np.diff(sign) != 0) + 1
    slope_sign = np.sign(first_derivative)
    slope_sign[slope_sign == 0] = 1
    extrema_indexes = np.flatnonzero(np.diff(slope_sign) != 0) + 1
    extrema_candidates = [
        int(index)
        for index in extrema_indexes
        if observed.observed_mask[int(index)]
    ]
    local_curvature_maxima = [
        index
        for index in range(1, 255)
        if (
            curvature[index] >= threshold
            and curvature[index] >= curvature[index - 1]
            and curvature[index] >= curvature[index + 1]
        )
    ]
    candidates = {
        int(index)
        for index in inflection_indexes
        if observed.observed_mask[int(index)] and curvature[int(index)] >= threshold
    }
    rdp_candidates = [
        int(index)
        for index in rdp_curve_x_positions(
            dense_lut,
            observed.weights,
            max_points=max_points,
        )
        if int(index) not in (0, 255)
    ]

    ranked_candidates = []
    for candidate in sorted(extrema_candidates):
        if candidate not in ranked_candidates:
            ranked_candidates.append(candidate)
    inflection_ranked_candidates = sorted(candidates)
    for candidate in inflection_ranked_candidates:
        if candidate not in ranked_candidates:
            ranked_candidates.append(candidate)
    for candidate in rdp_candidates[:max(0, max_points - 2)]:
        if candidate not in ranked_candidates:
            ranked_candidates.append(candidate)
    selected = [0, 255]
    best_points = _refit_points(
        [[0, float(dense_lut[0])], [255, float(dense_lut[255])]],
        observed,
        precision=precision,
        metric=metric,
        interpolation=interpolation,
    )
    best_score = _score_points(
        best_points,
        observed,
        metric=metric,
        interpolation=interpolation,
    )
    min_spacing = 8
    for candidate in ranked_candidates:
        if len(selected) >= max_points:
            break
        if (
            select_by_complexity
            and any(abs(candidate - existing) < min_spacing for existing in selected)
        ):
            continue
        candidate_x = sorted({*selected, int(candidate)})
        candidate_points = _refit_points(
            [[x, float(dense_lut[int(x)])] for x in candidate_x],
            observed,
            precision=precision,
            metric=metric,
            interpolation=interpolation,
        )
        candidate_score = _score_points(
            candidate_points,
            observed,
            metric=metric,
            interpolation=interpolation,
        )
        improvement = best_score - candidate_score
        required_improvement = max(complexity_penalty, best_score * 0.01)
        if not select_by_complexity or improvement >= required_improvement:
            selected = candidate_x
            best_points = candidate_points
            best_score = candidate_score

    if len(selected) == 2 and max_points > 2:
        for candidate in rdp_curve_x_positions(
            dense_lut,
            observed.weights,
            max_points=3,
        ):
            if candidate in (0, 255):
                continue
            selected = sorted({*selected, int(candidate)})
            best_points = _refit_points(
                [[x, float(dense_lut[int(x)])] for x in selected],
                observed,
                precision=precision,
                metric=metric,
                interpolation=interpolation,
            )
            best_score = _score_points(
                best_points,
                observed,
                metric=metric,
                interpolation=interpolation,
            )
            break

    if progress_callback is not None:
        progress_callback({
            'phase': 'spline_inflection',
            'parameters': {'points': best_points},
            'score': float(best_score),
            'best_score': float(best_score),
            'observed_bins': int(np.count_nonzero(observed.counts)),
            'inflection_candidates': len(inflection_indexes),
            'curvature_candidates': len(local_curvature_maxima),
            'candidate_count': len(ranked_candidates),
            'point_count': len(best_points),
        })
    return best_points


def refine_curve_points_local_search(
    points,
    observed_lut,
    weights=None,
    metric='balanced_huber',
    iterations=DEFAULT_REFINEMENT_ITERATIONS,
    observed_mask=None,
    precision=DEFAULT_POINT_PRECISION,
    progress_callback=None,
    interpolation=INTERPOLATION_LINEAR,
):
    """Locally refine interior point x positions, refitting y-values each time."""
    observed = ObservedCurveLut(
        values=_filled_values(observed_lut),
        weights=np.ones(256, dtype=np.float32) if weights is None else weights,
        counts=np.ones(256, dtype=np.float32) if weights is None else weights * weights,
        observed_values=np.asarray(observed_lut, dtype=np.float32),
        observed_mask=(~np.isnan(observed_lut) if observed_mask is None else observed_mask),
    )
    interpolation = normalize_interpolation(interpolation)
    best_points = _refit_points(
        points,
        observed,
        precision=precision,
        metric=metric,
        interpolation=interpolation,
    )
    best_score = _score_points(
        best_points,
        observed,
        metric=metric,
        interpolation=interpolation,
    )
    radii = (64, 32, 16, 8, 4, 2, 1)
    for _round in range(max(0, int(iterations))):
        improved = False
        for radius in radii:
            for index in range(1, len(best_points) - 1):
                current_x = float(best_points[index][0])
                min_x = float(best_points[index - 1][0]) + 0.001
                max_x = float(best_points[index + 1][0]) - 0.001
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
                    candidate = _refit_points(
                        candidate,
                        observed,
                        precision=precision,
                        metric=metric,
                        interpolation=interpolation,
                    )
                    score = _score_points(
                        candidate,
                        observed,
                        metric=metric,
                        interpolation=interpolation,
                    )
                    if score < best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
                        if progress_callback is not None:
                            progress_callback({
                                'phase': 'refine',
                                'round_index': _round + 1,
                                'radius': radius,
                                'parameters': {'points': best_points},
                                'score': float(best_score),
                                'best_score': float(best_score),
                                'point_count': len(best_points),
                            })
        if not improved:
            break
    return best_points


def prune_curve_points(
    points,
    observed,
    metric='balanced_huber',
    complexity_penalty=DEFAULT_COMPLEXITY_PENALTY,
    min_points=2,
    precision=DEFAULT_POINT_PRECISION,
    interpolation=INTERPOLATION_LINEAR,
):
    """Remove points that do not improve observed-fit enough to justify complexity."""
    interpolation = normalize_interpolation(interpolation)
    best_points = _refit_points(
        points,
        observed,
        precision=precision,
        metric=metric,
        interpolation=interpolation,
    )
    best_score = _score_points(
        best_points,
        observed,
        metric=metric,
        interpolation=interpolation,
    )
    while len(best_points) > max(2, int(min_points)):
        best_removal = None
        for index in range(1, len(best_points) - 1):
            candidate = [point[:] for i, point in enumerate(best_points) if i != index]
            candidate = _refit_points(
                candidate,
                observed,
                precision=precision,
                metric=metric,
                interpolation=interpolation,
            )
            score = _score_points(
                candidate,
                observed,
                metric=metric,
                interpolation=interpolation,
            )
            score_increase = score - best_score
            if best_removal is None or score_increase < best_removal[0]:
                best_removal = (score_increase, score, candidate)
        if best_removal is None or best_removal[0] > complexity_penalty:
            break
        _increase, best_score, best_points = best_removal
    return best_points


def prune_close_curve_points(
    points,
    observed,
    metric='balanced_huber',
    min_spacing=2.0,
    complexity_penalty=DEFAULT_COMPLEXITY_PENALTY,
    precision=DEFAULT_POINT_PRECISION,
    interpolation=INTERPOLATION_LINEAR,
):
    """Remove nearly overlaid interior points when they do not materially help."""
    interpolation = normalize_interpolation(interpolation)
    best_points = _refit_points(
        points,
        observed,
        precision=precision,
        metric=metric,
        interpolation=interpolation,
    )
    best_score = _score_points(
        best_points,
        observed,
        metric=metric,
        interpolation=interpolation,
    )
    while len(best_points) > 2:
        close_indexes = [
            index
            for index in range(1, len(best_points) - 1)
            if (
                abs(float(best_points[index][0]) - float(best_points[index - 1][0]))
                < min_spacing
                or abs(float(best_points[index + 1][0]) - float(best_points[index][0]))
                < min_spacing
            )
        ]
        if not close_indexes:
            break
        best_removal = None
        for index in close_indexes:
            candidate = [point[:] for i, point in enumerate(best_points) if i != index]
            candidate = _refit_points(
                candidate,
                observed,
                precision=precision,
                metric=metric,
                interpolation=interpolation,
            )
            score = _score_points(
                candidate,
                observed,
                metric=metric,
                interpolation=interpolation,
            )
            score_increase = score - best_score
            if best_removal is None or score_increase < best_removal[0]:
                best_removal = (score_increase, score, candidate)
        if best_removal is None or best_removal[0] > (complexity_penalty * 4.0):
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
    point_precision=DEFAULT_POINT_PRECISION,
    channel='White',
    score_image_transform=None,
    interpolation=INTERPOLATION_LINEAR,
    prune_points=True,
):
    """Recover Curves-node points from source/target images."""
    channel = _normalize_channel(channel)
    observed = estimate_curve_lut(source_image, target_image, channel=channel)
    interpolation = normalize_interpolation(interpolation)
    if interpolation != INTERPOLATION_LINEAR:
        initial_points = fit_spline_points_from_dense_reconstruction(
            observed,
            max_points=max_points,
            metric=metric_name,
            complexity_penalty=complexity_penalty,
            precision=point_precision,
            progress_callback=progress_callback,
            interpolation=interpolation,
            select_by_complexity=prune_points,
        )
    else:
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
            precision=point_precision,
        )
    initial_score = _score_points(
        initial_points,
        observed,
        metric=metric_name,
        interpolation=interpolation,
    )
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
        precision=point_precision,
        progress_callback=progress_callback,
        interpolation=interpolation,
    )
    refined_points = _refit_points(
        refined_points,
        observed,
        precision=point_precision,
        metric=metric_name,
        interpolation=interpolation,
    )
    refined_score = _score_points(
        refined_points,
        observed,
        metric=metric_name,
        interpolation=interpolation,
    )
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

    if interpolation != INTERPOLATION_LINEAR or not prune_points:
        pruned_points = refined_points
    else:
        refined_points = prune_close_curve_points(
            refined_points,
            observed,
            metric=metric_name,
            complexity_penalty=complexity_penalty,
            precision=point_precision,
            interpolation=interpolation,
        )
        pruned_points = prune_curve_points(
            refined_points,
            observed,
            metric=metric_name,
            complexity_penalty=complexity_penalty,
            precision=point_precision,
            interpolation=interpolation,
        )
    lut_score = _score_points(
        pruned_points,
        observed,
        metric=metric_name,
        interpolation=interpolation,
    )
    source_uint8 = _as_uint8_values(source_image).copy()
    try:
        best_image = image_process(
            source_uint8,
            _single_channel_curve_set(channel, pruned_points),
            interpolation=interpolation,
        )
    except AttributeError:
        lut = points_to_lut(
            pruned_points,
            quantize=True,
            interpolation=interpolation,
        ).astype(np.uint8)
        best_image = lut[source_uint8]
    scored_image = best_image
    compression_metadata = None
    if score_image_transform is not None:
        scored_image, compression_metadata = score_image_transform(best_image)
    image_score = mean_squared_error(scored_image, target_image)
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
            'compression_metadata': compression_metadata,
            'interpolation': interpolation,
            'observed_bins': int(np.count_nonzero(observed.counts)),
            'prune_points': bool(prune_points),
        },
        best_score=float(lut_score),
        best_image=scored_image,
        evaluated_count=3,
    )


def tune_curve_set(
    source_image,
    target_image,
    max_points=DEFAULT_MAX_POINTS,
    metric_name='balanced_huber',
    refinement_iterations=DEFAULT_REFINEMENT_ITERATIONS,
    progress_callback=None,
    complexity_penalty=DEFAULT_COMPLEXITY_PENALTY,
    point_precision=DEFAULT_POINT_PRECISION,
    score_image_transform=None,
    interpolation=INTERPOLATION_LINEAR,
    prune_points=True,
):
    """Recover a White-first, then RGB, Curves-node curve set."""

    def _channel_progress(channel):
        def _progress(update):
            if progress_callback is not None:
                update = dict(update)
                update['channel'] = channel
                progress_callback(update)
        return _progress

    white_result = tune_curves(
        source_image,
        target_image,
        max_points=max_points,
        metric_name=metric_name,
        refinement_iterations=refinement_iterations,
        progress_callback=_channel_progress('White'),
        complexity_penalty=complexity_penalty,
        point_precision=point_precision,
        channel='White',
        interpolation=interpolation,
        prune_points=prune_points,
    )
    helper = CurvesPointsEditorMixin()
    curve_set = helper._default_curve_set()
    curve_set['White'] = white_result.best_parameters['points']
    white_adjusted_source = image_process(source_image, curve_set)
    channel_results = {'White': white_result}

    for channel in ('Red', 'Green', 'Blue'):
        result = tune_curves(
            white_adjusted_source,
            target_image,
            max_points=max_points,
            metric_name=metric_name,
            refinement_iterations=refinement_iterations,
            progress_callback=_channel_progress(channel),
            complexity_penalty=complexity_penalty,
            point_precision=point_precision,
            channel=channel,
            interpolation=interpolation,
            prune_points=prune_points,
        )
        curve_set[channel] = result.best_parameters['points']
        channel_results[channel] = result

    best_image = image_process(source_image, curve_set, interpolation=interpolation)
    scored_image = best_image
    compression_metadata = None
    if score_image_transform is not None:
        scored_image, compression_metadata = score_image_transform(best_image)
    image_score = mean_squared_error(scored_image, target_image)
    best_score = float(np.mean([result.best_score for result in channel_results.values()]))
    return TuneResult(
        best_parameters={
            'curves': curve_set,
            'channel_results': channel_results,
            'image_score': float(image_score),
            'compression_metadata': compression_metadata,
            'prune_points': bool(prune_points),
        },
        best_score=best_score,
        best_image=scored_image,
        evaluated_count=sum(result.evaluated_count for result in channel_results.values()),
    )
