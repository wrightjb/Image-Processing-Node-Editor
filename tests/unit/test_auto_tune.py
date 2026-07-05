import numpy as np

from auto_tune.gaussian_blur import odd_kernel_values, sigma_values, tune_gaussian_blur
from auto_tune.service import (
    EvaluationPlan,
    ParameterSpec,
    TuneRequest,
    grid_search,
    mean_squared_error,
)
import node.process_node.node_gaussian_blur as gaussian_blur_module


def test_grid_search_returns_lowest_scoring_candidate():
    target = np.full((2, 2, 1), 3, dtype=np.uint8)

    def _evaluate(parameters):
        return np.full((2, 2, 1), parameters['value'], dtype=np.uint8)

    result = grid_search(
        TuneRequest(
            source_image=np.zeros((2, 2, 1), dtype=np.uint8),
            target_image=target,
            parameter_specs=[ParameterSpec('value', [1, 3, 5])],
            evaluation_plan=EvaluationPlan(_evaluate),
        )
    )

    assert result.best_parameters == {'value': 3}
    assert result.best_score == 0.0
    assert result.evaluated_count == 3


def test_metric_rejects_shape_mismatch():
    candidate = np.zeros((2, 2, 3), dtype=np.uint8)
    target = np.zeros((3, 2, 3), dtype=np.uint8)

    try:
        mean_squared_error(candidate, target)
    except ValueError as exc:
        assert 'matching shapes' in str(exc)
    else:
        raise AssertionError('shape mismatch should fail')


def test_gaussian_blur_candidate_helpers_use_valid_domains():
    assert odd_kernel_values(2, 8) == (3, 5, 7)
    assert sigma_values(0.1, 0.3, 0.1) == (0.1, 0.2, 0.3)


def test_tune_gaussian_blur_searches_odd_kernels_and_auto_sigma(monkeypatch):
    calls = []
    source = np.zeros((4, 4, 1), dtype=np.uint8)
    target = np.full((4, 4, 1), 5, dtype=np.uint8)

    def _gaussian_stub(image, kernel, sigma):
        calls.append((kernel, sigma))
        return np.full_like(image, kernel[0])

    monkeypatch.setattr(gaussian_blur_module.cv2, 'GaussianBlur', _gaussian_stub, raising=False)

    result = tune_gaussian_blur(
        source,
        target,
        current_parameters={'auto_sigma': True},
        kernel_min=1,
        kernel_max=9,
    )

    assert result.best_parameters['kernel_size'] == 5
    assert result.best_parameters['sigma'] == 0.0
    assert result.evaluated_count == 5
    assert calls == [((1, 1), 0.0), ((3, 3), 0.0), ((5, 5), 0.0), ((7, 7), 0.0), ((9, 9), 0.0)]


def test_tune_gaussian_blur_tunes_sigma_when_auto_sigma_disabled(monkeypatch):
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 2, dtype=np.uint8)

    def _gaussian_stub(image, kernel, sigma):
        del kernel
        return np.full_like(image, int(round(sigma * 10)))

    monkeypatch.setattr(gaussian_blur_module.cv2, 'GaussianBlur', _gaussian_stub, raising=False)

    result = tune_gaussian_blur(
        source,
        target,
        current_parameters={'auto_sigma': False},
        kernel_min=1,
        kernel_max=1,
        sigma_min=0.1,
        sigma_max=0.3,
        sigma_step=0.1,
    )

    assert result.best_parameters['kernel_size'] == 1
    assert result.best_parameters['sigma'] == 0.2
    assert result.evaluated_count == 3
