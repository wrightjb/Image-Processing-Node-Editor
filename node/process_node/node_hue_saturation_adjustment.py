#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node_editor.util import dpg_get_value


_BANDS = (
    ('red', 0.0),
    ('orange', 15.0),
    ('yellow', 30.0),
    ('green', 60.0),
    ('cyan', 90.0),
    ('blue', 120.0),
    ('purple', 135.0),
    ('magenta', 150.0),
)
_BAND_ANCHORS_DEGREES = np.array(
    [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0, 360.0],
    dtype=np.float32,
)
_BAND_NAME_TO_INDEX = {band_name: index for index, (band_name, _) in enumerate(_BANDS)}
HUE_SHIFT_MIN = -90
HUE_SHIFT_MAX = 90
LUMINANCE_MIN = -100
LUMINANCE_MAX = 100
ACHROMATIC_POLISH_PARITY = 'Polish parity'
ACHROMATIC_STANDARD = 'Standard'


def _smoothstep_with_blend(t, blend):
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend == 0.0:
        return np.where(t >= 0.5, 1.0, 0.0).astype(np.float32)

    t_low = 0.5 - (blend / 2.0)
    t_high = 0.5 + (blend / 2.0)
    x = np.clip((t - t_low) / (t_high - t_low), 0.0, 1.0)
    return (x * x * (3.0 - (2.0 * x))).astype(np.float32)


def _build_polish_band_weight_lut(blend):
    hue_degrees = np.arange(180, dtype=np.float32) * 2.0
    weights = np.zeros((180, len(_BANDS)), dtype=np.float32)

    for lower_index in range(len(_BAND_ANCHORS_DEGREES) - 1):
        lower_anchor = _BAND_ANCHORS_DEGREES[lower_index]
        upper_anchor = _BAND_ANCHORS_DEGREES[lower_index + 1]
        if lower_index == len(_BANDS) - 1:
            upper_band_index = 0
        else:
            upper_band_index = lower_index + 1
        lower_band_index = lower_index

        if lower_index == len(_BAND_ANCHORS_DEGREES) - 2:
            in_interval = (hue_degrees >= lower_anchor) & (hue_degrees < upper_anchor)
        else:
            in_interval = (hue_degrees >= lower_anchor) & (hue_degrees <= upper_anchor)
        if not np.any(in_interval):
            continue

        t = (hue_degrees[in_interval] - lower_anchor) / (upper_anchor - lower_anchor)
        upper_weight = _smoothstep_with_blend(t, blend)
        weights[in_interval, lower_band_index] = 1.0 - upper_weight
        weights[in_interval, upper_band_index] = upper_weight

    return weights


_BAND_WEIGHT_LUT = _build_polish_band_weight_lut(1.0)
_BLEND_WEIGHT_LUT_CACHE = {}


def _get_blend_weight_lut(blend):
    blend = float(np.clip(blend, 0.0, 1.0))
    key = int(round(blend * 200.0))
    if key not in _BLEND_WEIGHT_LUT_CACHE:
        _BLEND_WEIGHT_LUT_CACHE[key] = _build_polish_band_weight_lut(
            key / 200.0,
        )
    return _BLEND_WEIGHT_LUT_CACHE[key]


def _active_adjustments(adjustments):
    active = []
    for band_name, index in _BAND_NAME_TO_INDEX.items():
        hue_delta = float(adjustments.get(f'{band_name}_hue_shift', 0))
        saturation_delta = float(adjustments.get(f'{band_name}_saturation', 0))
        luminance_delta = float(adjustments.get(f'{band_name}_luminance', 0))
        if hue_delta == 0.0 and saturation_delta == 0.0 and luminance_delta == 0.0:
            continue
        active.append((index, hue_delta, saturation_delta, luminance_delta))
    return active


def image_process(
    image,
    blend=0.0,
    achromatic_mode=ACHROMATIC_POLISH_PARITY,
    **adjustments,
):
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
    value_channel = hsv_image[:, :, 2]

    hue_indices = np.clip(hue_channel.astype(np.int16), 0, 179)
    hue_delta_by_band = np.zeros(len(_BANDS), dtype=np.float32)
    saturation_delta_by_band = np.zeros(len(_BANDS), dtype=np.float32)
    luminance_delta_by_band = np.zeros(len(_BANDS), dtype=np.float32)

    for index, hue_delta, saturation_delta, luminance_delta in active_adjustments:
        hue_delta_by_band[index] = hue_delta
        saturation_delta_by_band[index] = saturation_delta / 100.0
        luminance_delta_by_band[index] = luminance_delta / 100.0

    blend_weights = _get_blend_weight_lut(blend)[hue_indices]
    if achromatic_mode == ACHROMATIC_STANDARD:
        blend_weights = blend_weights * (sat_channel > 0.0)[..., None]

    hue_shift = np.sum(blend_weights * hue_delta_by_band[None, None, :], axis=2)
    saturation_scale = 1.0 + np.sum(
        blend_weights * saturation_delta_by_band[None, None, :],
        axis=2,
    )

    luminance_scale = 1.0 + np.sum(
        blend_weights * luminance_delta_by_band[None, None, :], axis=2
    )

    hsv_image[:, :, 0] = np.mod(hue_channel + hue_shift, 180.0)
    hsv_image[:, :, 1] = np.clip(sat_channel * saturation_scale, 0.0, 255.0)
    hsv_image[:, :, 2] = np.clip(value_channel * luminance_scale, 0.0, 255.0)

    hsv_for_bgr = hsv_image.astype(np.float32, copy=True)
    hsv_for_bgr[:, :, 0] *= 2.0
    hsv_for_bgr[:, :, 1:] /= 255.0
    adjusted_bgr_float = cv2.cvtColor(hsv_for_bgr, cv2.COLOR_HSV2BGR)
    adjusted_bgr = np.clip(
        np.rint(adjusted_bgr_float * 255.0),
        0,
        255,
    ).astype(np.uint8)

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
    _ver = '0.0.4'

    node_label = 'Hue Bands'
    node_tag = 'HueSaturationAdjustment'

    _last_touched_slider_tag_by_node = {}
    _last_touched_node_id = None
    _expanded_bands_by_node = {}

    parameters = [
        {
            'name': 'blend',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input02',
            'widget': 'slider_float',
            'label': 'blend',
            'default': 0.0,
            'min': 0.0,
            'max': 1.0,
            'cast': float,
            'precision': 2,
        },
        {
            'name': 'achromatic_mode',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input27',
            'widget': 'combo',
            'label': 'achromatic pixels',
            'items': [ACHROMATIC_POLISH_PARITY, ACHROMATIC_STANDARD],
            'default': ACHROMATIC_POLISH_PARITY,
            'cast': str,
        },
    ]
    for index, (band_name, _) in enumerate(_BANDS):
        parameters.extend([
            {
                'name': f'{band_name}_hue_shift',
                'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
                'port': f'Input{(index * 2) + 3:02d}',
                'widget': 'slider_float',
                'label': f'{band_name[:3]} hue',
                'default': 0.0,
                'min': HUE_SHIFT_MIN,
                'max': HUE_SHIFT_MAX,
                'cast': float,
                'precision': 1,
                'quantize': 0.5,
                'step': 0.5,
            },
            {
                'name': f'{band_name}_saturation',
                'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
                'port': f'Input{(index * 2) + 4:02d}',
                'widget': 'slider_float',
                'label': f'{band_name[:3]} sat',
                'default': 0.0,
                'min': -100,
                'max': 100,
                'cast': float,
                'precision': 1,
                'quantize': 0.5,
                'step': 0.5,
            },
            {
                'name': f'{band_name}_luminance',
                'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
                'port': f'Input{index + 19:02d}',
                'widget': 'slider_float',
                'label': f'{band_name[:3]} lum',
                'default': 0.0,
                'min': LUMINANCE_MIN,
                'max': LUMINANCE_MAX,
                'cast': float,
                'precision': 1,
                'quantize': 0.5,
                'step': 0.5,
            },
        ])

    def _add_band_controls(self, node_id, width):
        self._expanded_bands_by_node[str(node_id)] = set()
        with dpg.node_attribute(
            tag=self._band_controls_tag(node_id),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            with dpg.group(horizontal=True):
                button_width = max(104, (width - 12) // 2)
                dpg.add_button(
                    label='Expand all',
                    width=button_width,
                    callback=self._set_all_bands_callback,
                    user_data=(node_id, True),
                )
                dpg.add_button(
                    label='Collapse all',
                    width=button_width,
                    callback=self._set_all_bands_callback,
                    user_data=(node_id, False),
                )

    def _add_parameter_ui(self, node_id, parameter, width, callback):
        parameter_name = parameter['name']
        if parameter_name == 'achromatic_mode':
            super()._add_parameter_ui(node_id, parameter, width, callback)
            self._add_band_controls(node_id, width)
            return
        band_name = next(
            (
                name
                for name, _center in _BANDS
                if parameter_name == f'{name}_hue_shift'
            ),
            None,
        )
        if band_name is not None:
            with dpg.node_attribute(
                tag=self._band_header_tag(node_id, band_name),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    tag=self._band_button_tag(node_id, band_name),
                    label=self._band_summary(node_id, band_name),
                    width=(width * 2) - 40,
                    callback=self._toggle_band_callback,
                    user_data=(node_id, band_name),
                )
        super()._add_parameter_ui(node_id, parameter, width, callback)

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
                label='Add tuner',
                width=width - 80,
                callback=self._add_tuner_callback,
                user_data=node_id,
            )
            dpg.add_button(
                label='Reset all',
                width=width - 80,
                callback=self._reset_all_callback,
                user_data=node_id,
            )

    def on_node_added(self, tag_node_name):
        node_id = int(tag_node_name.split(':')[0])
        for band_name, _center in _BANDS:
            self._set_band_visibility(node_id, band_name, False)
        if not dpg.does_item_exist('_hsa_arrow_keys'):
            with dpg.handler_registry(tag='_hsa_arrow_keys'):
                dpg.add_key_press_handler(dpg.mvKey_Left, callback=self._nudge_slider, user_data=-1)
                dpg.add_key_press_handler(dpg.mvKey_Right, callback=self._nudge_slider, user_data=1)

        for parameter in self.parameters:
            if not parameter['widget'].startswith('slider_'):
                continue
            slider_tag = self._value_tag(
                self._port_tag(tag_node_name, parameter['type'], parameter['port'])
            )
            dpg.configure_item(slider_tag, callback=self._slider_touched_callback, user_data=node_id)


    def _slider_touched_callback(self, sender, app_data, user_data):
        slider_tag = dpg.get_item_alias(sender)
        self._last_touched_slider_tag_by_node[user_data] = slider_tag
        self._last_touched_node_id = user_data
        if self._ui_callback is not None and slider_tag:
            node_id_name = ':'.join(slider_tag.split(':')[:2])
            before_value = self._last_parameter_values.get(slider_tag, app_data)
            self._last_parameter_values[slider_tag] = app_data
            self._ui_callback(
                'parameter_changed',
                {
                    'node_id_name': node_id_name,
                    'port_tag': slider_tag[:-5],
                    'value_tag': slider_tag,
                    'before_value': before_value,
                    'after_value': app_data,
                },
            )
        self._refresh_band_summary_for_value_tag(user_data, slider_tag)

    def _on_parameter_widget_changed(self, sender, app_data, user_data):
        super()._on_parameter_widget_changed(sender, app_data, user_data)
        value_tag = user_data.get('value_tag') if isinstance(user_data, dict) else None
        if not value_tag:
            return
        node_id_name = user_data.get('node_id_name', '')
        try:
            node_id = int(node_id_name.split(':', 1)[0])
        except (TypeError, ValueError):
            return
        self._refresh_band_summary_for_value_tag(node_id, value_tag)

    def _band_controls_tag(self, node_id):
        return self._control_tag(
            self._node_name(node_id), self.TYPE_TEXT, 'BandControls'
        )

    def _band_header_tag(self, node_id, band_name):
        return self._control_tag(
            self._node_name(node_id),
            self.TYPE_TEXT,
            f'Band{band_name.title()}',
        )

    def _band_button_tag(self, node_id, band_name):
        return f'{self._band_header_tag(node_id, band_name)}:Button'

    def _band_parameter(self, band_name, suffix):
        return next(
            parameter
            for parameter in self.parameters
            if parameter['name'] == f'{band_name}_{suffix}'
        )

    def _band_summary(self, node_id, band_name):
        values = []
        for short_label, suffix in (
            ('H', 'hue_shift'),
            ('S', 'saturation'),
            ('L', 'luminance'),
        ):
            parameter = self._band_parameter(band_name, suffix)
            value_tag = self._parameter_port_ref(node_id, parameter).value_tag
            value = dpg_get_value(value_tag)
            if value is None:
                value = parameter['default']
            values.append(f'{short_label} {float(value):+.1f}')
        expanded = band_name in self._expanded_bands_by_node.get(
            str(node_id), set()
        )
        marker = '[-]' if expanded else '[+]'
        return f'{marker} {band_name.title()}    ' + '   '.join(values)

    def _set_band_visibility(self, node_id, band_name, expanded):
        expanded_bands = self._expanded_bands_by_node.setdefault(
            str(node_id), set()
        )
        if expanded:
            expanded_bands.add(band_name)
        else:
            expanded_bands.discard(band_name)
        for suffix in ('hue_shift', 'saturation', 'luminance'):
            parameter = self._band_parameter(band_name, suffix)
            value_tag = self._parameter_port_ref(node_id, parameter).value_tag
            dpg.configure_item(
                self._slider_group_tag(value_tag),
                show=expanded,
            )
        dpg.configure_item(
            self._band_button_tag(node_id, band_name),
            label=self._band_summary(node_id, band_name),
        )

    def _toggle_band_callback(self, sender, app_data, user_data):
        del sender, app_data
        node_id, band_name = user_data
        expanded = band_name not in self._expanded_bands_by_node.get(
            str(node_id), set()
        )
        self._set_band_visibility(node_id, band_name, expanded)

    def _set_all_bands_callback(self, sender, app_data, user_data):
        del sender, app_data
        node_id, expanded = user_data
        for band_name, _center in _BANDS:
            self._set_band_visibility(node_id, band_name, expanded)

    def _refresh_band_summary_for_value_tag(self, node_id, value_tag):
        for band_name, _center in _BANDS:
            for suffix in ('hue_shift', 'saturation', 'luminance'):
                parameter = self._band_parameter(band_name, suffix)
                if self._parameter_port_ref(node_id, parameter).value_tag == value_tag:
                    dpg.configure_item(
                        self._band_button_tag(node_id, band_name),
                        label=self._band_summary(node_id, band_name),
                    )
                    return

    def _nudge_slider(self, sender, app_data, user_data):
        del sender, app_data
        step = int(user_data)
        slider_tag = self._last_touched_slider_tag_by_node.get(
            self._last_touched_node_id
        )
        if not slider_tag or not dpg.does_item_exist(slider_tag):
            return
        item_conf = dpg.get_item_configuration(slider_tag)
        current = dpg.get_value(slider_tag)
        if isinstance(current, float):
            next_value = round(current + (0.5 * step), 1)
        else:
            next_value = int(current) + step
        min_value = item_conf.get('min_value', next_value)
        max_value = item_conf.get('max_value', next_value)
        updated_value = max(min_value, min(max_value, next_value))
        dpg.set_value(slider_tag, updated_value)
        if self._ui_callback is not None:
            node_id_name = ':'.join(slider_tag.split(':')[:2])
            self._ui_callback(
                'parameter_changed',
                {
                    'node_id_name': node_id_name,
                    'port_tag': slider_tag[:-5],
                    'value_tag': slider_tag,
                    'before_value': current,
                    'after_value': updated_value,
                },
            )

    def _add_tuner_callback(self, sender, app_data, user_data):
        del sender, app_data
        if self._ui_callback is None:
            return
        self._ui_callback(
            'spawn_hue_bands_tuner_requested',
            {'node_id_name': self._node_name(user_data)},
        )

    def _reset_all_callback(self, sender, app_data, user_data):
        del sender, app_data
        tag_node_name = self._node_name(user_data)
        batch_changes = []
        for parameter in self.parameters:
            parameter_value_tag = self._value_tag(
                self._port_tag(tag_node_name, parameter['type'], parameter['port'])
            )
            before_value = dpg.get_value(parameter_value_tag)
            after_value = parameter['default']
            dpg.set_value(parameter_value_tag, after_value)
            self._last_parameter_values[parameter_value_tag] = after_value
            if before_value != after_value:
                batch_changes.append(
                    {
                        'value_tag': parameter_value_tag,
                        'before_value': before_value,
                        'after_value': after_value,
                    }
                )
        if self._ui_callback is not None and batch_changes:
            self._ui_callback(
                'parameter_batch_changed',
                {
                    'node_id_name': tag_node_name,
                    'changes': batch_changes,
                },
            )
