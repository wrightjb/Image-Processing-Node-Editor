#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from auto_tune.photo_editor_color import (
    DEFAULT_PARAMETERS,
    PARAMETER_NAMES,
    tune_photo_editor_color,
)
from node.input_node.node_auto_tune_photo_editor_color import Node
from node.process_node.node_photo_editor_color import image_process


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
