import numpy as np

from auto_tune.gaussian_blur import (
    DEFAULT_KERNEL_MAX,
    DEFAULT_KERNEL_MIN,
    odd_kernel_values,
    sigma_values,
    tune_gaussian_blur,
)
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
    assert DEFAULT_KERNEL_MIN == 1
    assert DEFAULT_KERNEL_MAX == 501
    assert len(odd_kernel_values()) == 251
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


def test_auto_tune_node_waits_for_run_button(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    node.create_ports(6)

    called = False

    def _tune_stub(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(auto_tune_node_module, 'tune_gaussian_blur', _tune_stub)

    assert node.update(6, [], {}, {}) == (None, {'__auto_tune_ready__': False})
    assert called is False


def test_auto_tune_node_run_accepts_legacy_tuple_connections(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    ports = node.create_ports(6)
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 3, dtype=np.uint8)
    best_image = np.full((2, 2, 1), 1, dtype=np.uint8)
    set_values = []

    class _TuneResult:
        best_parameters = {'kernel_size': 3, 'sigma': 0.0}
        best_score = 2.5
        evaluated_count = 4

    _TuneResult.best_image = best_image

    def _tune_stub(source_image, target_image, current_parameters):
        assert source_image is source
        assert target_image is target
        assert current_parameters == {'auto_sigma': True}
        return _TuneResult()

    monkeypatch.setattr(auto_tune_node_module, 'tune_gaussian_blur', _tune_stub)
    monkeypatch.setattr(
        auto_tune_node_module,
        'dpg_set_value',
        lambda tag, value: set_values.append((tag, value)),
    )

    node._on_run_button(None, None, 6)
    image, result = node.update(
        6,
        [
            ('1:Source:Image:Output01', ports.source_image.dpg_tag),
            ('2:Target:Image:Output01', ports.target_image.dpg_tag),
        ],
        {
            '1:Source': source,
            '2:Target': target,
        },
        {},
    )

    assert image is best_image
    assert result['__auto_tune_ready__'] is True
    assert result['tune_result'].best_score == 2.5
    assert set_values == [
        (ports.kernel_size.value_tag, 3),
        (ports.sigma.value_tag, 0.0),
        (ports.best_score.value_tag, 2.5),
    ]
    assert node.update(6, [], {}, {}) == (None, {'__auto_tune_ready__': False})


def test_auto_tune_node_marks_settings_uncacheable(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    node.create_ports(6)
    monkeypatch.setattr(auto_tune_node_module.dpg, 'get_item_pos', lambda tag: [0, 0])
    monkeypatch.setattr(auto_tune_node_module, 'dpg_get_value', lambda tag: 0)

    assert node.get_setting_dict(6)['__cache_enabled__'] is False


def test_declarative_nodes_skip_stale_auto_tune_parameter_values():
    from node.process_node.node_gaussian_blur import Node as GaussianBlurNode

    node = GaussianBlurNode()
    auto_tune_source = '6:AutoTuneGaussianBlur:Int:Output01'

    assert node._source_allows_parameter_sync(auto_tune_source, {}) is False
    assert node._source_allows_parameter_sync(
        auto_tune_source,
        {'6:AutoTuneGaussianBlur': {'__auto_tune_ready__': False}},
    ) is False
    assert node._source_allows_parameter_sync(
        auto_tune_source,
        {'6:AutoTuneGaussianBlur': {'__auto_tune_ready__': True}},
    ) is True
    assert node._source_allows_parameter_sync('1:IntValue:Int:Output01', {}) is True
