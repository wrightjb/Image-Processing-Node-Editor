#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize and measure pixel differences between two images."""

import time

import dearpygui.dearpygui as dpg
import numpy as np

from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs
from node_editor.util import convert_cv_to_dpg, dpg_get_value, dpg_set_value


_MODE_ABSOLUTE = 'absolute'
_MODE_HEATMAP = 'heatmap'
_MODE_SIGNED = 'signed'
_MODES = (_MODE_ABSOLUTE, _MODE_HEATMAP, _MODE_SIGNED)


def _as_float_image(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    return array


def _match_shape(image_a, image_b):
    a = _as_float_image(image_a)
    b = _as_float_image(image_b)
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    channels = min(a.shape[2], b.shape[2])
    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError('images must have non-empty overlapping dimensions')
    return a[:height, :width, :channels], b[:height, :width, :channels]


def _ensure_three_channels(image):
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[:, :, :3]
    padding = np.zeros((*image.shape[:2], 3 - image.shape[2]), dtype=image.dtype)
    return np.concatenate([image, padding], axis=2)


def image_process(image_a, image_b, amplify=1.0, mode=_MODE_ABSOLUTE):
    """Return a visual diff image and numeric diff metrics."""
    a, b = _match_shape(image_a, image_b)
    difference = b - a
    absolute = np.abs(difference)
    amplified = np.clip(absolute * max(0.0, float(amplify)), 0, 255)

    if mode == _MODE_SIGNED:
        signed = np.zeros((*difference.shape[:2], 3), dtype=np.float32)
        positive = np.clip(difference, 0, 255).mean(axis=2)
        negative = np.clip(-difference, 0, 255).mean(axis=2)
        signed[:, :, 1] = positive * max(0.0, float(amplify))
        signed[:, :, 2] = negative * max(0.0, float(amplify))
        diff_image = signed
    elif mode == _MODE_HEATMAP:
        gray = np.clip(amplified.mean(axis=2), 0, 255)
        diff_image = np.zeros((*gray.shape, 3), dtype=np.float32)
        diff_image[:, :, 0] = np.clip((gray - 128) * 2, 0, 255)
        diff_image[:, :, 1] = np.clip(gray * 2, 0, 255)
        diff_image[:, :, 2] = np.clip(255 - (gray * 2), 0, 255)
        diff_image[gray <= 0] = 0
    else:
        diff_image = _ensure_three_channels(amplified)

    metrics = {
        'mae': float(np.mean(absolute)),
        'mse': float(np.mean(difference * difference)),
        'max_abs': float(np.max(absolute)),
        'changed_pixels': int(np.count_nonzero(np.max(absolute, axis=2))),
    }
    return np.clip(diff_image, 0, 255).astype(np.uint8), metrics


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Image Diff'
    node_tag = 'ImageDiff'

    port_specs = PortSpecs(
        image_a=InputPort(PortDataType.IMAGE, index=1),
        image_b=InputPort(PortDataType.IMAGE, index=2),
        amplify=InputPort(PortDataType.FLOAT, index=3),
        image=OutputPort(PortDataType.IMAGE, index=1),
        mae=OutputPort(PortDataType.FLOAT, index=2),
        mse=OutputPort(PortDataType.FLOAT, index=3),
        max_abs=OutputPort(PortDataType.FLOAT, index=4),
        elapsed=OutputPort(PortDataType.TIME_MS, index=5),
    )

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        image_a_port = ports.image_a
        image_b_port = ports.image_b
        amplify_port = ports.amplify
        image_port = ports.image
        mae_port = ports.mae
        mse_port = ports.mse
        max_abs_port = ports.max_abs
        elapsed_port = ports.elapsed
        image_a = image_a_port.dpg_tag
        image_b = image_b_port.dpg_tag
        amplify = amplify_port.dpg_tag
        image = image_port.dpg_tag
        mae = mae_port.dpg_tag
        mse = mse_port.dpg_tag
        max_abs = max_abs_port.dpg_tag
        elapsed = elapsed_port.dpg_tag

        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        black_image = np.zeros((small_window_h, small_window_w, 3), dtype=np.uint8)
        black_texture = convert_cv_to_dpg(black_image, small_window_w, small_window_h)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=image_port.value_tag,
                format=dpg.mvFormat_Float_rgb,
            )

        with dpg.node(tag=tag_node_name, parent=parent, label=self.node_label, pos=pos):
            self.add_editor_toolbar(node_id, callback=callback)
            with dpg.node_attribute(tag=image_a, attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text(tag=image_a_port.value_tag, default_value='image A')
            with dpg.node_attribute(tag=image_b, attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text(tag=image_b_port.value_tag, default_value='image B')
            with dpg.node_attribute(tag=amplify, attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_slider_float(
                    tag=amplify_port.value_tag,
                    label='amplify',
                    default_value=10.0,
                    min_value=1.0,
                    max_value=255.0,
                    width=small_window_w - 80,
                    callback=callback,
                )
            with dpg.node_attribute(
                tag=self._mode_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    _MODES,
                    label='mode',
                    tag=self._mode_value_tag(node_id),
                    default_value=_MODE_ABSOLUTE,
                    width=140,
                    callback=callback,
                )
            with dpg.node_attribute(tag=image, attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_image(image_port.value_tag)
            with dpg.node_attribute(tag=mae, attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(
                    tag=mae_port.value_tag,
                    label='MAE',
                    default_value=0.0,
                    readonly=True,
                    width=120,
                )
            with dpg.node_attribute(tag=mse, attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(
                    tag=mse_port.value_tag,
                    label='MSE',
                    default_value=0.0,
                    readonly=True,
                    width=120,
                )
            with dpg.node_attribute(tag=max_abs, attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(
                    tag=max_abs_port.value_tag,
                    label='max abs',
                    default_value=0.0,
                    readonly=True,
                    width=120,
                )
            if use_pref_counter:
                with dpg.node_attribute(
                    tag=elapsed,
                    attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(tag=elapsed_port.value_tag, default_value='0000ms')

        return tag_node_name

    def _mode_attr_tag(self, node_id):
        return self._node_control_tag(node_id, self.TYPE_TEXT, 'Mode')

    def _mode_value_tag(self, node_id):
        return self._node_control_value_tag(node_id, self.TYPE_TEXT, 'Mode')


    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        del node_result_dict
        ports = self.ports(node_id)
        output_value_tag = ports.image.value_tag
        elapsed_value_tag = ports.elapsed.value_tag
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        frame_a = None
        frame_b = None
        for (
            connection_info,
            source_tag,
            destination_tag,
            connection_type,
        ) in self._iter_connection_infos(connection_list):
            if connection_type == self.TYPE_FLOAT:
                source_value_tag = self._connection_value_tag(
                    connection_info,
                    'source',
                    source_tag,
                )
                destination_value_tag = self._connection_value_tag(
                    connection_info,
                    'destination',
                    destination_tag,
                )
                dpg_set_value(destination_value_tag, float(dpg_get_value(source_value_tag)))
            if connection_type != self.TYPE_IMAGE:
                continue
            destination_port = self._connection_port_name(connection_info, destination_tag)
            source_node_key = self._connection_source_node_key(connection_info, source_tag)
            if destination_port == 'Input01':
                frame_a = node_image_dict.get(source_node_key)
            elif destination_port == 'Input02':
                frame_b = node_image_dict.get(source_node_key)

        if frame_a is None or frame_b is None:
            return None, None

        start_time = time.perf_counter()
        amplify = float(dpg_get_value(ports.amplify.value_tag) or 1.0)
        mode = dpg_get_value(self._mode_value_tag(node_id)) or _MODE_ABSOLUTE
        frame, metrics = image_process(frame_a, frame_b, amplify=amplify, mode=mode)

        if use_pref_counter:
            elapsed_time = int((time.perf_counter() - start_time) * 1000)
            dpg_set_value(elapsed_value_tag, str(elapsed_time).zfill(4) + 'ms')

        dpg_set_value(ports.mae.value_tag, metrics['mae'])
        dpg_set_value(ports.mse.value_tag, metrics['mse'])
        dpg_set_value(ports.max_abs.value_tag, metrics['max_abs'])
        texture = convert_cv_to_dpg(frame, small_window_w, small_window_h)
        dpg_set_value(output_value_tag, texture)
        return frame, metrics

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        ports = self.ports(node_id)
        return {
            'ver': self._ver,
            'pos': dpg.get_item_pos(tag_node_name),
            ports.amplify.value_tag: dpg_get_value(ports.amplify.value_tag),
            ports.mae.value_tag: dpg_get_value(ports.mae.value_tag),
            ports.mse.value_tag: dpg_get_value(ports.mse.value_tag),
            ports.max_abs.value_tag: dpg_get_value(ports.max_abs.value_tag),
            self._mode_value_tag(node_id): dpg_get_value(self._mode_value_tag(node_id)),
        }

    def set_setting_dict(self, node_id, setting_dict):
        ports = self.ports(node_id)
        for value_tag in (
            ports.amplify.value_tag,
            ports.mae.value_tag,
            ports.mse.value_tag,
            ports.max_abs.value_tag,
            self._mode_value_tag(node_id),
        ):
            if value_tag in setting_dict:
                dpg_set_value(value_tag, setting_dict[value_tag])
