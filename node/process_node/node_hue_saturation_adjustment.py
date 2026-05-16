#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import dearpygui.dearpygui as dpg
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
_HUE_BAND_INDEX_LUT = np.argmax(_BAND_WEIGHT_LUT, axis=1).astype(np.int16)


def _active_adjustments(adjustments):
    active = []
    for band_name, index in _BAND_NAME_TO_INDEX.items():
        hue_delta_degrees = float(adjustments.get(f'{band_name}_hue_shift', 0))
        saturation_delta = float(adjustments.get(f'{band_name}_saturation', 0))
        if hue_delta_degrees == 0.0 and saturation_delta == 0.0:
            continue
        active.append((index, hue_delta_degrees, saturation_delta))
    return active


def image_process(image, hue_blend=1.0, **adjustments):
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
    soft_saturation_scale = np.ones_like(sat_channel, dtype=np.float32)

    hue_delta_by_band = np.zeros(len(_BANDS), dtype=np.float32)
    saturation_delta_by_band = np.zeros(len(_BANDS), dtype=np.float32)

    for index, hue_delta_degrees, saturation_delta in active_adjustments:
        band_weight = weights[:, :, index]
        hue_delta_by_band[index] = hue_delta_degrees / 2.0
        saturation_delta_by_band[index] = saturation_delta / 100.0
        soft_saturation_scale += band_weight * saturation_delta_by_band[index]

    hue_band_indices = _HUE_BAND_INDEX_LUT[hue_indices]
    hard_hue_shift = hue_delta_by_band[hue_band_indices]

    soft_hue_shift = np.zeros_like(hue_channel, dtype=np.float32)
    for index, _, _ in active_adjustments:
        soft_hue_shift += weights[:, :, index] * hue_delta_by_band[index]

    hard_saturation_scale = 1.0 + saturation_delta_by_band[hue_band_indices]

    hue_blend = float(np.clip(hue_blend, 0.0, 1.0))
    hue_shift = (hue_blend * soft_hue_shift) + ((1.0 - hue_blend) * hard_hue_shift)
    saturation_scale = (hue_blend * soft_saturation_scale) + ((1.0 - hue_blend) * hard_saturation_scale)

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
    _ver = '0.0.2'

    node_label = 'Hue/Saturation Adjustment'
    node_tag = 'HueSaturationAdjustment'

    _last_touched_slider_tag_by_node = {}

    parameters = [
        {
            'name': 'hue_blend',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input02',
            'widget': 'slider_float',
            'label': 'hue blend',
            'default': 1.0,
            'min': 0.0,
            'max': 1.0,
            'cast': float,
            'precision': 2,
        },
    ]
    for index, (band_name, _) in enumerate(_BANDS):
        parameters.extend([
            {
                'name': f'{band_name}_hue_shift',
                'type': DeclarativeImageProcessNodeBase.TYPE_INT,
                'port': f'Input{(index * 2) + 3:02d}',
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
                'port': f'Input{(index * 2) + 4:02d}',
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

    def build_custom_ui(self, tag_node_name, node_id, width, callback):
        del callback
        with dpg.node_attribute(
            tag=self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input99'),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            dpg.add_button(
                label='Reset all',
                width=width - 80,
                callback=self._reset_all_callback,
                user_data=node_id,
            )

    def on_node_added(self, tag_node_name):
        node_id = int(tag_node_name.split(':')[0])
        if not dpg.does_item_exist('_hsa_arrow_keys'):
            with dpg.handler_registry(tag='_hsa_arrow_keys'):
                dpg.add_key_press_handler(dpg.mvKey_Left, callback=self._nudge_slider, user_data=-1)
                dpg.add_key_press_handler(dpg.mvKey_Right, callback=self._nudge_slider, user_data=1)

        for parameter in self.parameters:
            slider_tag = self._value_tag(
                self._port_tag(tag_node_name, parameter['type'], parameter['port'])
            )
            dpg.configure_item(slider_tag, callback=self._slider_touched_callback, user_data=node_id)


    def _slider_touched_callback(self, sender, app_data, user_data):
        del app_data
        self._last_touched_slider_tag_by_node[user_data] = dpg.get_item_alias(sender)

    def _nudge_slider(self, sender, app_data, user_data):
        del sender, app_data
        step = int(user_data)
        for slider_tag in self._last_touched_slider_tag_by_node.values():
            if not slider_tag or not dpg.does_item_exist(slider_tag):
                continue
            item_conf = dpg.get_item_configuration(slider_tag)
            current = dpg.get_value(slider_tag)
            if isinstance(current, float):
                next_value = round(current + (0.01 * step), 2)
            else:
                next_value = int(current) + step
            min_value = item_conf.get('min_value', next_value)
            max_value = item_conf.get('max_value', next_value)
            dpg.set_value(slider_tag, max(min_value, min(max_value, next_value)))

    def _reset_all_callback(self, sender, app_data, user_data):
        del sender, app_data
        tag_node_name = self._node_name(user_data)
        for parameter in self.parameters:
            parameter_value_tag = self._value_tag(
                self._port_tag(tag_node_name, parameter['type'], parameter['port'])
            )
            dpg.set_value(parameter_value_tag, parameter['default'])
