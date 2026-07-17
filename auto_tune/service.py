#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small, GUI-independent auto-tuning primitives."""

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Search domain for one tunable parameter."""

    name: str
    values: Sequence[object]

    def __post_init__(self):
        if not self.values:
            raise ValueError(f'Parameter {self.name} must have candidates')


@dataclass(frozen=True)
class EvaluationPlan:
    """Pure evaluation callback for one candidate parameter dictionary."""

    evaluate: Callable[[Mapping[str, object]], np.ndarray]


@dataclass(frozen=True)
class TuneRequest:
    source_image: np.ndarray
    target_image: np.ndarray
    parameter_specs: Sequence[ParameterSpec]
    evaluation_plan: EvaluationPlan
    fixed_parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TuneResult:
    best_parameters: dict
    best_score: float
    best_image: np.ndarray
    evaluated_count: int


def normalize_image_for_metric(image):
    """Return an image as float32 in the 0..1 range for scoring."""
    if image is None:
        raise ValueError('image must not be None')
    normalized = np.asarray(image, dtype=np.float32)
    if normalized.size == 0:
        raise ValueError('image must not be empty')
    if normalized.max(initial=0.0) > 1.0:
        normalized = normalized / 255.0
    return normalized


def mean_squared_error(candidate, target):
    candidate = normalize_image_for_metric(candidate)
    target = normalize_image_for_metric(target)
    if candidate.shape != target.shape:
        raise ValueError(
            'candidate and target images must have matching shapes: '
            f'{candidate.shape} != {target.shape}'
        )
    difference = candidate - target
    return float(np.mean(difference * difference))


def mean_absolute_error(candidate, target):
    """Return mean absolute channel error for normalized images."""
    candidate = normalize_image_for_metric(candidate)
    target = normalize_image_for_metric(target)
    if candidate.shape != target.shape:
        raise ValueError(
            'candidate and target images must have matching shapes: '
            f'{candidate.shape} != {target.shape}'
        )
    return float(np.mean(np.abs(candidate - target)))


def _candidate_dicts(parameter_specs: Iterable[ParameterSpec]):
    specs = list(parameter_specs)
    names = [spec.name for spec in specs]
    for values in product(*(spec.values for spec in specs)):
        yield dict(zip(names, values))


def grid_search(
    request,
    metric=mean_squared_error,
    progress_callback=None,
    early_stop_score=None,
):
    """Evaluate every candidate and return the lowest-scoring result."""
    if not isinstance(request, TuneRequest):
        raise TypeError('request must be a TuneRequest')

    target = normalize_image_for_metric(request.target_image)
    best_score = None
    best_parameters = None
    best_image = None
    evaluated_count = 0

    candidates = list(_candidate_dicts(request.parameter_specs))
    total_count = len(candidates)

    for candidate in candidates:
        parameters = dict(request.fixed_parameters)
        parameters.update(candidate)
        image = request.evaluation_plan.evaluate(parameters)
        score = metric(image, target)
        evaluated_count += 1
        if best_score is None or score < best_score:
            best_score = score
            best_parameters = dict(parameters)
            best_image = image
        if progress_callback is not None:
            progress_callback({
                'candidate_index': evaluated_count,
                'candidate_count': total_count,
                'parameters': dict(parameters),
                'score': float(score),
                'best_score': float(best_score),
                'best_parameters': dict(best_parameters),
            })
        if early_stop_score is not None and score <= early_stop_score:
            break

    if evaluated_count == 0:
        raise ValueError('at least one candidate must be evaluated')

    return TuneResult(
        best_parameters=best_parameters,
        best_score=float(best_score),
        best_image=best_image,
        evaluated_count=evaluated_count,
    )
