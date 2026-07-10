import numpy as np
import pytest

from auto_tune.gaussian_blur import (
    DEFAULT_KERNEL_MAX,
    DEFAULT_KERNEL_MIN,
    DEFAULT_REFINEMENT_ITERATIONS,
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
        'refinement_iterations': DEFAULT_REFINEMENT_ITERATIONS,
        'planned_passes': DEFAULT_REFINEMENT_ITERATIONS,
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
        metric_name='mse',
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
        **kwargs,
    ):
        assert source_image is source
        assert target_image is target
        assert current_parameters['auto_sigma'] is True
        del kwargs
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
            and not tag.endswith(':Int:RefineRoundsValue')
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
    hue_bands_source = '7:AutoTuneHueBands:Int:Output02'
    assert node._source_allows_parameter_sync(hue_bands_source, {}) is False
    assert node._source_allows_parameter_sync(
        hue_bands_source,
        {'7:AutoTuneHueBands': {'__auto_tune_ready__': True}},
    ) is True
    assert node._source_allows_parameter_sync('1:IntValue:Int:Output01', {}) is True


def test_tune_gaussian_blur_does_not_constrain_first_run_to_prior_kernel(monkeypatch):
    source = np.zeros((4, 4, 1), dtype=np.uint8)
    target = np.full((4, 4, 1), 117, dtype=np.uint8)

    def _gaussian_stub(image, kernel, sigma):
        del sigma
        return np.full_like(image, min(kernel[0], 255))

    monkeypatch.setattr(
        gaussian_blur_module.cv2,
        'GaussianBlur',
        _gaussian_stub,
        raising=False,
    )

    result = tune_gaussian_blur(
        source,
        target,
        current_parameters={'auto_sigma': True, 'kernel_size': 5},
        kernel_min=1,
        kernel_max=201,
        metric_name='mse',
        refinement_iterations=1,
    )

    assert result.best_parameters['kernel_size'] == 117


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
        metric_name='mse',
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
        assert current_parameters['auto_sigma'] is False
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


def test_auto_tune_node_reuses_previous_output_values_as_start(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    ports = node.create_ports(6)
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 3, dtype=np.uint8)

    class _TuneResult:
        best_parameters = {'kernel_size': 87, 'sigma': 0.4}
        best_score = 0.5
        evaluated_count = 3

    _TuneResult.best_image = source

    def _get_value(tag):
        if tag == ports.kernel_size.value_tag:
            return 85
        if tag == ports.sigma.value_tag:
            return 0.3
        return True

    def _tune_stub(source_image, target_image, current_parameters, **kwargs):
        del source_image, target_image, kwargs
        assert current_parameters == {
            'auto_sigma': True,
            'kernel_size': 85,
            'sigma': 0.3,
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
            ('2:Target:Image:Output01', ports.target_image.dpg_tag),
        ],
        {
            '1:Source': source,
            '2:Target': target,
        },
        {},
    )

    assert result['tune_result'].best_parameters['kernel_size'] == 87


def test_tune_gaussian_blur_manual_sigma_can_recover_from_bad_prior(monkeypatch):
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 60, dtype=np.uint8)

    def _gaussian_stub(image, kernel, sigma):
        del kernel
        return np.full_like(image, int(round(sigma)))

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
            'sigma': 2.0,
        },
        metric_name='mse',
    )

    assert abs(result.best_parameters['sigma'] - 60.0) <= 0.5
    assert result.evaluated_count < 100


def test_blur_smoothness_error_ignores_constant_intensity_shift():
    from auto_tune.gaussian_blur import blur_smoothness_error

    candidate = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    target = np.clip(candidate + 40, 0, 255).astype(np.uint8)

    assert blur_smoothness_error(candidate, target) < 1e-6


def test_auto_tune_node_passes_selected_metric(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    ports = node.create_ports(6)
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 3, dtype=np.uint8)

    class _TuneResult:
        best_parameters = {'kernel_size': 3, 'sigma': 0.0}
        best_score = 0.0
        evaluated_count = 1

    _TuneResult.best_image = source

    def _get_value(tag):
        if tag.endswith(':Text:MetricValue'):
            return 'local_smoothness'
        return True

    def _tune_stub(source_image, target_image, current_parameters, **kwargs):
        del source_image, target_image, current_parameters
        assert kwargs['metric_name'] == 'local_smoothness'
        return _TuneResult()

    monkeypatch.setattr(auto_tune_node_module, 'tune_gaussian_blur', _tune_stub)
    monkeypatch.setattr(auto_tune_node_module, 'dpg_get_value', _get_value)
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

    assert result['tune_result'].best_score == 0.0


def test_local_smoothness_compares_neighbor_contrast_locations():
    from auto_tune.gaussian_blur import (
        blur_smoothness_error,
        local_blur_smoothness_error,
    )

    candidate = np.array(
        [[0, 255, 255], [0, 255, 255]],
        dtype=np.uint8,
    )
    target = np.array(
        [[255, 255, 0], [255, 255, 0]],
        dtype=np.uint8,
    )

    assert blur_smoothness_error(candidate, target) == 0.0
    assert local_blur_smoothness_error(candidate, target) > 0.0


def test_smoothness_metric_reports_per_candidate_diagnostics(monkeypatch):
    updates = []
    source = np.zeros((4, 4, 1), dtype=np.uint8)
    target = np.tile(np.arange(4, dtype=np.uint8), (4, 1)).reshape(4, 4, 1)

    def _gaussian_stub(image, kernel, sigma):
        del sigma
        return np.full_like(image, kernel[0])

    monkeypatch.setattr(
        gaussian_blur_module.cv2,
        'GaussianBlur',
        _gaussian_stub,
        raising=False,
    )

    tune_gaussian_blur(
        source,
        target,
        current_parameters={'auto_sigma': True},
        kernel_min=1,
        kernel_max=9,
        metric_name='smoothness',
        refinement_iterations=1,
        progress_callback=updates.append,
    )

    assert updates
    assert 'candidate_smoothness' in updates[0]
    assert 'target_smoothness' in updates[0]


def test_auto_tune_node_passes_refinement_iterations(monkeypatch):
    import node.input_node.node_auto_tune as auto_tune_node_module

    node = auto_tune_node_module.Node()
    ports = node.create_ports(6)
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.full((2, 2, 1), 3, dtype=np.uint8)

    class _TuneResult:
        best_parameters = {'kernel_size': 3, 'sigma': 0.0}
        best_score = 0.0
        evaluated_count = 1

    _TuneResult.best_image = source

    def _get_value(tag):
        if tag.endswith(':Int:RefineRoundsValue'):
            return 5
        if tag.endswith(':Text:MetricValue'):
            return 'mse'
        return True

    def _tune_stub(source_image, target_image, current_parameters, **kwargs):
        del source_image, target_image, current_parameters
        assert kwargs['refinement_iterations'] == 5
        return _TuneResult()

    monkeypatch.setattr(auto_tune_node_module, 'tune_gaussian_blur', _tune_stub)
    monkeypatch.setattr(auto_tune_node_module, 'dpg_get_value', _get_value)
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

    assert result['tune_result'].best_score == 0.0


def test_tune_curves_recovers_solar_points():
    from auto_tune.curves import tune_curves
    from auto_tune.curves import points_to_lut

    source = np.tile(np.arange(256, dtype=np.uint8), (12, 1))
    solar_points = [
        [0, 0],
        [32, 245],
        [64, 10],
        [96, 235],
        [128, 20],
        [160, 225],
        [192, 30],
        [224, 215],
        [255, 40],
    ]
    target = points_to_lut(solar_points).astype(np.uint8)[source]

    result = tune_curves(
        source,
        target,
        max_points=len(solar_points),
        refinement_iterations=1,
    )

    assert len(result.best_parameters['points']) <= len(solar_points)
    assert result.best_score < 80.0
    assert result.best_parameters['image_score'] < 0.002



def test_tune_curve_set_returns_all_channels_for_white_then_rgb_workflow():
    from auto_tune.curves import tune_curve_set
    from node.process_node.node_curves import image_process

    source = np.tile(np.arange(0, 256, 16, dtype=np.uint8), (4, 1))
    source = np.dstack((source, source, source))
    curve_set = {
        'White': [[0, 0], [255, 200]],
        'Red': [[0, 0], [200, 180], [255, 255]],
        'Green': [[0, 0], [200, 160], [255, 255]],
        'Blue': [[0, 0], [200, 140], [255, 255]],
    }
    target = image_process(source, {'curves': curve_set})

    result = tune_curve_set(source, target, max_points=4, refinement_iterations=0)

    assert set(result.best_parameters['curves']) == {'White', 'Red', 'Green', 'Blue'}
    assert result.best_parameters['image_score'] < 0.02

def test_auto_tune_curves_node_waits_for_run_button():
    import node.input_node.node_auto_tune_curves as auto_tune_curves_node_module

    node = auto_tune_curves_node_module.Node()

    assert node.update(7, [], {}, {}) == (None, {'__auto_tune_ready__': False})


def test_image_diff_reports_metrics_and_absolute_visualization():
    from node.draw_node.node_image_diff import image_process

    image_a = np.array([[[10, 20, 30], [100, 100, 100]]], dtype=np.uint8)
    image_b = np.array([[[15, 10, 30], [90, 120, 130]]], dtype=np.uint8)

    diff, metrics = image_process(image_a, image_b, amplify=2.0)

    assert diff.tolist() == [[[10, 20, 0], [20, 40, 60]]]
    assert metrics['mae'] == 12.5
    assert metrics['mse'] == pytest.approx(254.1666667)
    assert metrics['max_abs'] == 30.0
    assert metrics['changed_pixels'] == 2


def test_tune_curves_refits_sparse_endpoint_segments_and_prunes_extra_points():
    from auto_tune.curves import points_to_lut, tune_curves

    source = np.tile(np.arange(35, 256, dtype=np.uint8), (10, 1))
    points = [[0, 0], [78, 240], [188, 34], [255, 255]]
    target = points_to_lut(points, quantize=True).astype(np.uint8)[source]

    result = tune_curves(
        source,
        target,
        max_points=10,
        refinement_iterations=2,
    )

    recovered = result.best_parameters['points']
    assert len(recovered) == 4
    assert recovered[0][1] < 5
    assert recovered[1][0] == 78
    assert recovered[2][0] == 188
    assert result.best_parameters['image_score'] < 1e-4


def test_tune_curves_defaults_to_practical_point_precision():
    from auto_tune.curves import DEFAULT_POINT_PRECISION, points_to_lut, tune_curves

    source = np.tile(np.arange(35, 256, dtype=np.uint8), (4, 1))
    points = [[0, 0], [81.324324, 224], [173.675676, 31], [255, 255]]
    target = points_to_lut(points, quantize=True).astype(np.uint8)[source]

    result = tune_curves(source, target, max_points=4, refinement_iterations=1)

    assert DEFAULT_POINT_PRECISION == 3
    assert all(
        not isinstance(value, float)
        or len(str(value).split('.')[-1]) <= DEFAULT_POINT_PRECISION
        for point in result.best_parameters['points']
        for value in point
    )


def test_tune_hue_bands_recovers_simple_band_adjustment():
    import cv2
    if not hasattr(cv2, 'cvtColor'):
        pytest.skip('cv2 stub does not implement HSV conversion')

    from auto_tune.hue_bands import tune_hue_bands
    from node.process_node.node_hue_saturation_adjustment import image_process

    source = np.zeros((24, 24, 3), dtype=np.uint8)
    source[:, :, 2] = 255
    target = image_process(source, red_hue_shift=30, red_saturation=25)

    result = tune_hue_bands(
        source,
        target,
        current_parameters={},
        refinement_iterations=0,
    )

    assert result.evaluated_count > 1
    assert result.best_score < 0.01
    assert result.best_parameters['red_hue_shift'] == 30
    assert result.best_parameters['red_saturation'] == 25


def test_auto_tune_hue_bands_node_declares_all_parameter_outputs():
    import node.input_node.node_auto_tune_hue_bands as auto_tune_hue_bands_module

    node = auto_tune_hue_bands_module.Node()
    ports = node.create_ports(42)

    assert ports.source_image.dpg_tag == '42:AutoTuneHueBands:Image:Input01'
    assert ports.target_image.dpg_tag == '42:AutoTuneHueBands:Image:Input02'
    assert ports.blend.dpg_tag == '42:AutoTuneHueBands:Float:Output01'
    assert ports.red_hue_shift.dpg_tag == '42:AutoTuneHueBands:Int:Output02'
    assert ports.magenta_saturation.dpg_tag == '42:AutoTuneHueBands:Int:Output13'
    assert ports.best_score.dpg_tag == '42:AutoTuneHueBands:Float:Output14'


def test_hue_bands_simplification_prunes_low_value_parameters(monkeypatch):
    import auto_tune.hue_bands as hue_bands

    def fake_image_process(source, **parameters):
        del source
        return parameters

    def fake_score(parameters, target):
        del target
        score = 0.1
        if parameters.get('blue_hue_shift') == 70:
            score = 0.0095 if parameters.get('blend') == 1.0 else 0.009
        return score

    monkeypatch.setattr(hue_bands, 'image_process', fake_image_process)
    monkeypatch.setattr(hue_bands, '_score', fake_score)

    simplified, simplified_score = hue_bands._simplify_parameters(
        {
            'blend': 0.75,
            'blue_hue_shift': 70,
            'magenta_hue_shift': 120,
        },
        source=None,
        target=None,
        original_score=0.1,
        best_score=0.009,
    )

    assert simplified['blue_hue_shift'] == 70
    assert simplified['magenta_hue_shift'] == 0
    assert simplified['blend'] == 1.0
    assert simplified_score == 0.0095


def test_hue_bands_estimate_candidates_do_not_tune_blend_by_default(monkeypatch):
    import auto_tune.hue_bands as hue_bands

    seen_blends = []

    def fake_estimate(source, target, blend):
        del source, target
        seen_blends.append(blend)
        return {'blend': blend, 'blue_hue_shift': int(blend * 10)}

    monkeypatch.setattr(hue_bands, '_estimate_parameters_from_hsv', fake_estimate)

    default_candidates = hue_bands._estimate_candidate_parameters(None, None)
    tuned_candidates = hue_bands._estimate_candidate_parameters(
        None,
        None,
        tune_blend=True,
    )

    assert [candidate['blend'] for candidate in default_candidates] == [1.0]
    assert [candidate['blend'] for candidate in tuned_candidates] == list(
        hue_bands.BLEND_CANDIDATES
    )
    assert seen_blends == [1.0] + list(hue_bands.BLEND_CANDIDATES)


def test_auto_tune_hue_bands_tune_blend_defaults_false():
    import node.input_node.node_auto_tune_hue_bands as auto_tune_hue_bands_module

    node = auto_tune_hue_bands_module.Node()

    assert node._tune_blend_value_tag(42) == '42:AutoTuneHueBands:Int:TuneBlendValue'
