#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, Mock

import numpy as np

from auto_tune.photo_editor_color import (
    DEFAULT_PARAMETERS,
    PARAMETER_NAMES,
    tune_photo_editor_color,
)
from node.input_node.node_auto_tune_photo_editor_color import Node
import node.input_node.node_auto_tune_photo_editor_color as tuner_node_module
from node.process_node.node_photo_editor_color import image_process
from node.process_node.node_photo_editor_color import Node as PhotoEditorColorNode


def _color_fixture():
    return np.array(
        [
            [[12, 35, 80], [220, 170, 90], [45, 210, 160]],
            [[245, 235, 225], [110, 70, 190], [25, 125, 240]],
            [[75, 15, 145], [180, 200, 35], [135, 115, 95]],
        ],
        dtype=np.uint8,
    )


def test_photo_editor_color_tuner_recovers_exposure():
    source = _color_fixture()
    target = image_process(source, exposure=37)

    result = tune_photo_editor_color(
        source,
        target,
        enabled_parameters=('exposure',),
        global_iterations=8,
        population_size=5,
        seed=4,
        local_refinement=True,
    )

    assert result.best_parameters['mode'] == 'MacGyver parity'
    assert abs(result.best_parameters['exposure'] - 37) <= 1
    assert result.best_score <= (1.0 / 255.0)
    assert result.evaluated_count > 0


def test_photo_editor_color_tuner_respects_fixed_parameters():
    source = _color_fixture()
    parameters = dict(DEFAULT_PARAMETERS)
    parameters.update({'temperature': 8200, 'tint': -22})
    target = image_process(source, **parameters)

    result = tune_photo_editor_color(
        source,
        target,
        current_parameters=parameters,
        enabled_parameters=(),
    )

    assert result.best_score == 0.0
    assert result.best_parameters['temperature'] == 8200
    assert result.best_parameters['tint'] == -22
    assert np.array_equal(result.best_image, target)


def test_photo_editor_color_tuner_node_defaults_cover_all_controls():
    output_names = {spec.key for spec in Node.port_specs}

    assert Node.node_tag == 'AutoTunePhotoEditorColor'
    assert set(PARAMETER_NAMES).issubset(output_names)
    assert {'source_image', 'target_image', 'mode', 'best_score'}.issubset(
        output_names
    )


def test_photo_editor_color_tuner_honors_integer_checkbox_values(monkeypatch):
    node = Node()

    def _value(tag):
        return 0 if 'TuneBrightness' in tag or 'TuneTint' in tag else 1

    monkeypatch.setattr(
        'node.input_node.node_auto_tune_photo_editor_color.dpg_get_value',
        _value,
    )

    enabled = node._enabled_parameters(3)

    assert 'brightness' not in enabled
    assert 'tint' not in enabled
    assert set(enabled) == set(PARAMETER_NAMES) - {'brightness', 'tint'}


def test_photo_editor_color_tuner_value_field_hides_builtin_step_buttons(
    monkeypatch,
):
    dpg = MagicMock()
    monkeypatch.setattr(tuner_node_module, 'dpg', dpg)
    port = Mock(dpg_tag='3:Tuner:Int:Output01', value_tag='exposure-value')

    Node()._add_parameter_output(3, port, 'exposure', 0, None)

    assert dpg.add_input_int.call_args.kwargs['step'] == 0
    assert [call.kwargs['label'] for call in dpg.add_button.call_args_list] == [
        '-',
        '+',
    ]


def test_photo_editor_color_add_tuner_button_requests_spawn():
    node = PhotoEditorColorNode()
    callback = Mock()
    node._ui_callback = callback

    node._add_tuner_callback(None, None, 7)

    callback.assert_called_once_with(
        'spawn_photo_editor_color_tuner_requested',
        {'node_id_name': '7:PhotoEditorColor'},
    )
