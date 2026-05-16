#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


_BANDS = (
    ('red', 0.0),
    ('yellow', 30.0),
    ('green', 60.0),
    ('cyan', 90.0),
    ('blue', 120.0),
    ('magenta', 150.0),
)
_BAND_HALF_WIDTH = 30.0
_BAND_NAME_TO_INDEX = {band_name: index for index, (band_name, _) in enumerate(_BANDS)}


def _build_band_weight_lut():
    hue_values = np.arange(180, dtype=np.float32)[:, None]
    centers = np.array([center for _, center in _BANDS], dtype=np.float32)[None, :]

    delta = np.abs(hue_values - centers)
    distance = np.minimum(delta, 180.0 - delta)
    weights = np.clip(1.0 - (distance / _BAND_HALF_WIDTH), 0.0, 1.0)

    total = np.sum(weights, axis=1, keepdims=True)
    total[total == 0.0] = 1.0
    return weights / total


_BAND_WEIGHT_LUT = _build_band_weight_lut()


def _active_adjustments(adjustments):
    active = []
    for band_name, index in _BAND_NAME_TO_INDEX.items():
        hue_delta_degrees = float(adjustments.get(f'{band_name}_hue_shift', 0))
        saturation_delta = float(adjustments.get(f'{band_name}_saturation', 0))
        if hue_delta_degrees == 0.0 and saturation_delta == 0.0:
            continue
        active.append((index, hue_delta_degrees, saturation_delta))
    return active


def image_process(image, **adjustments):
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return image

    active_adjustments = _active_adjustments(adjustments)
    if not active_adjustments:
        return image

    bgr_image = image[:, :, :3]
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else None

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue_channel = hsv_image[:, :, 0]
    sat_channel = hsv_image[:, :, 1]

    hue_indices = np.clip(hue_channel.astype(np.int16), 0, 179)
    weights = _BAND_WEIGHT_LUT[hue_indices]

    hue_shift = np.zeros_like(hue_channel, dtype=np.float32)
    saturation_scale = np.ones_like(sat_channel, dtype=np.float32)

    for index, hue_delta_degrees, saturation_delta in active_adjustments:
        band_weight = weights[:, :, index]
        hue_shift += band_weight * (hue_delta_degrees / 2.0)
        saturation_scale += band_weight * (saturation_delta / 100.0)

    hsv_image[:, :, 0] = np.mod(hue_channel + hue_shift, 180.0)
    hsv_image[:, :, 1] = np.clip(sat_channel * saturation_scale, 0.0, 255.0)

    adjusted_bgr = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if alpha_channel is not None:
        return cv2.merge(
            (
                adjusted_bgr[:, :, 0],
                adjusted_bgr[:, :, 1],
                adjusted_bgr[:, :, 2],
                alpha_channel,
            )
        )

    return adjusted_bgr


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Hue/Saturation Adjustment'
    node_tag = 'HueSaturationAdjustment'

    parameters = []
    for index, (band_name, _) in enumerate(_BANDS):
        parameters.extend([
            {
                'name': f'{band_name}_hue_shift',
                'type': DeclarativeImageProcessNodeBase.TYPE_INT,
                'port': f'Input{(index * 2) + 2:02d}',
                'widget': 'slider_int',
                'label': f'{band_name[:3]} hue',
                'default': 0,
                'min': -180,
                'max': 180,
                'cast': int,
            },
            {
                'name': f'{band_name}_saturation',
                'type': DeclarativeImageProcessNodeBase.TYPE_INT,
                'port': f'Input{(index * 2) + 3:02d}',
                'widget': 'slider_int',
                'label': f'{band_name[:3]} sat',
                'default': 0,
                'min': -100,
                'max': 100,
                'cast': int,
            },
        ])

    def process(self, frame, **parameter_values):
        frame = image_process(frame, **parameter_values)
        return frame, None
