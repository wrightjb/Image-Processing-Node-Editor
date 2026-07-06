import numpy as np

from auto_tune.gaussian_blur import (
    DEFAULT_KERNEL_MAX,
    DEFAULT_KERNEL_MIN,
    tuning_plan,
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


def test_grid_search_reports_progress_for_each_candidate():
    target = np.full((1, 1, 1), 2, dtype=np.uint8)
    updates = []

    def _evaluate(parameters):
        return np.full((1, 1, 1), parameters['value'], dtype=np.uint8)

    grid_search(
        TuneRequest(
            source_image=np.zeros((1, 1, 1), dtype=np.uint8),
            target_image=target,
            parameter_specs=[ParameterSpec('value', [1, 2, 3])],
            evaluation_plan=EvaluationPlan(_evaluate),
        ),
        progress_callback=updates.append,
    )

    assert [update['candidate_index'] for update in updates] == [1, 2, 3]
    assert all(update['candidate_count'] == 3 for update in updates)
    assert updates[-1]['best_parameters'] == {'value': 2}


def test_metric_rejects_shape_mismatch():
    candidate = np.zeros((2, 2, 3), dtype=np.uint8)
    target = np.zeros((3, 2, 3), dtype=np.uint8)

    try:
        mean_squared_error(candidate, target)
    except ValueError as exc:
        assert 'matching shapes' in str(exc)
    else:
        raise AssertionError('shape mismatch should fail')


def test_gaussian_blur_candidate_helpers_use_node_metadata_domains():
    assert DEFAULT_KERNEL_MIN == gaussian_blur_module.Node.parameters[0]['min']
    assert DEFAULT_KERNEL_MAX == gaussian_blur_module.Node.parameters[0]['max']
    assert len(odd_kernel_values()) == 251
    assert odd_kernel_values(2, 8) == (3, 5, 7)
    assert sigma_values(0.1, 0.3, 0.1) == (0.1, 0.2, 0.3)


def test_gaussian_blur_tuning_plan_downscales_large_images():
    source = np.zeros((1200, 800, 1), dtype=np.uint8)

    plan = tuning_plan(source, max_dimension=500)

    assert plan == {
        'original_candidates': 251,
        'scaled_candidates': 84,
        'downscale_step': 3,
        'max_dimension': 500,
    }


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
    assert result.evaluated_count == 3
    assert ((5, 5), 0.0) in calls
    assert calls[-1] == ((5, 5), 0.0)


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
    assert result.evaluated_count == 1


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

    def _tune_stub(
        source_image,
        target_image,
        current_parameters,
        progress_callback=None,
    ):
        assert source_image is source
        assert target_image is target
        assert current_parameters == {'auto_sigma': True}
        if progress_callback is not None:
            progress_callback({
                'pass_index': 1,
                'pass_count': 1,
                'candidate_index': 1,
                'candidate_count': 1,
                'total_evaluated': 1,
                'parameters': {'kernel_size': 3},
                'score': 2.5,
                'best_score': 2.5,
            })
        return _TuneResult()

    monkeypatch.setattr(auto_tune_node_module, 'tune_gaussian_blur', _tune_stub)
    monkeypatch.setattr(auto_tune_node_module, 'dpg_get_value', lambda tag: True)
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
    parameter_values = [
        (tag, value)
        for tag, value in set_values
        if (
            not tag.endswith(':Text:StatusValue')
            and not tag.endswith(':Int:AutoSigmaValue')
        )
    ]
    status_values = [
        value
        for tag, value in set_values
        if tag.endswith(':Text:StatusValue')
    ]
    assert parameter_values == [
        (ports.kernel_size.value_tag, 3),
        (ports.sigma.value_tag, 0.0),
        (ports.best_score.value_tag, 2.5),
    ]
    assert status_values[-1] == 'done: 4 candidates'
    assert any('candidate 1/1' in value for value in status_values)
    assert any('\n' in value for value in status_values)
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


def test_tune_gaussian_blur_refines_downscaled_kernel_to_original_scale(monkeypatch):
    source = np.zeros((600, 600, 1), dtype=np.uint8)
    target = np.full((600, 600, 1), 117, dtype=np.uint8)

    def _gaussian_stub(image, kernel, sigma):
        del sigma
        original_scale = source.shape[0] / image.shape[0]
        value = int(round(kernel[0] * original_scale))
        return np.full_like(image, min(value, 255))

    monkeypatch.setattr(
        gaussian_blur_module.cv2,
        'GaussianBlur',
        _gaussian_stub,
        raising=False,
    )

    result = tune_gaussian_blur(
        source,
        target,
        current_parameters={'auto_sigma': True},
    )

    assert result.best_parameters['kernel_size'] == 117
    assert result.best_image.shape == source.shape


def test_auto_tune_node_can_disable_auto_sigma(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    ports = node.create_ports(6)
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 3, dtype=np.uint8)

    class _TuneResult:
        best_parameters = {'kernel_size': 3, 'sigma': 0.2}
        best_score = 1.0
        evaluated_count = 2

    _TuneResult.best_image = source

    def _tune_stub(source_image, target_image, current_parameters, **kwargs):
        del source_image, target_image, kwargs
        assert current_parameters == {'auto_sigma': False}
        return _TuneResult()

    monkeypatch.setattr(auto_tune_node_module, 'tune_gaussian_blur', _tune_stub)
    monkeypatch.setattr(auto_tune_node_module, 'dpg_get_value', lambda tag: False)
    monkeypatch.setattr(auto_tune_node_module, 'dpg_set_value', lambda tag, value: None)

    node._on_run_button(None, None, 6)
    _image, result = node.update(
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

    assert result['tune_result'].best_parameters['sigma'] == 0.2


def test_auto_tune_node_uses_target_gaussian_parameters_as_start(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    ports = node.create_ports(6)
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 3, dtype=np.uint8)

    class _TuneResult:
        best_parameters = {'kernel_size': 117, 'sigma': 2.5}
        best_score = 0.0
        evaluated_count = 1

    _TuneResult.best_image = source

    def _get_value(tag):
        if tag == '2:GaussianBlur:Int:Input02Value':
            return 117
        if tag == '2:GaussianBlur:Float:Input03Value':
            return 2.5
        if tag == '2:GaussianBlur:Int:Input04Value':
            return False
        if tag.endswith(':Text:StatusValue'):
            return None
        return True

    def _tune_stub(source_image, target_image, current_parameters, **kwargs):
        del source_image, target_image, kwargs
        assert current_parameters == {
            'auto_sigma': False,
            'kernel_size': 117,
            'sigma': 2.5,
        }
        return _TuneResult()

    monkeypatch.setattr(auto_tune_node_module, 'tune_gaussian_blur', _tune_stub)
    monkeypatch.setattr(auto_tune_node_module, 'dpg_get_value', _get_value)
    monkeypatch.setattr(auto_tune_node_module, 'dpg_set_value', lambda tag, value: None)

    node._on_run_button(None, None, 6)
    _image, result = node.update(
        6,
        [
            ('1:Source:Image:Output01', ports.source_image.dpg_tag),
            ('2:GaussianBlur:Image:Output01', ports.target_image.dpg_tag),
        ],
        {
            '1:Source': source,
            '2:GaussianBlur': target,
        },
        {},
    )

    assert result['tune_result'].best_parameters['sigma'] == 2.5


def test_tune_gaussian_blur_manual_sigma_avoids_cartesian_grid(monkeypatch):
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 25, dtype=np.uint8)

    def _gaussian_stub(image, kernel, sigma):
        del kernel
        return np.full_like(image, int(round(sigma * 10)))

    monkeypatch.setattr(
        gaussian_blur_module.cv2,
        'GaussianBlur',
        _gaussian_stub,
        raising=False,
    )

    result = tune_gaussian_blur(
        source,
        target,
        current_parameters={
            'auto_sigma': False,
            'kernel_size': 117,
            'sigma': 2.5,
        },
    )

    assert result.best_parameters['sigma'] == 2.5
    assert result.evaluated_count < 100
