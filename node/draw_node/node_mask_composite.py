#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from node.node_abc import DpgNodeBase
from node.port_model import (
    InputPort,
    OutputPort,
    PortDataType,
    PortSpecs,
    enum_value,
)
from node_editor.util import convert_cv_to_dpg, dpg_get_value, dpg_set_value


def _resize_like(image, reference):
    reference_height, reference_width = reference.shape[:2]
    if image.shape[:2] == (reference_height, reference_width):
        return image
    return cv2.resize(image, (reference_width, reference_height))


def _ensure_three_channels(image):
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def _mask_to_bool(mask, reference):
    mask = _resize_like(mask, reference)
    if mask.ndim == 3:
        mask = _ensure_three_channels(mask).max(axis=2)
    return mask > 0


def image_process(image_a, mask, image_b=None, invert_mask=False):
    image_a = _ensure_three_channels(image_a)
    mask_bool = _mask_to_bool(mask, image_a)

    if invert_mask:
        mask_bool = np.logical_not(mask_bool)

    if image_b is None:
        image_b = np.zeros_like(image_a)
    else:
        image_b = _ensure_three_channels(_resize_like(image_b, image_a))

    return np.where(mask_bool[:, :, None], image_a, image_b).astype(image_a.dtype)


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Mask Composite'
    node_tag = 'MaskComposite'

    port_specs = PortSpecs(
        image_a=InputPort(PortDataType.IMAGE, index=1),
        mask=InputPort(PortDataType.IMAGE, index=2),
        image_b=InputPort(PortDataType.IMAGE, index=3),
        invert_mask=InputPort(PortDataType.TEXT, index=4),
        image=OutputPort(PortDataType.IMAGE, index=1),
        elapsed=OutputPort(PortDataType.TIME_MS, index=2),
    )

    _opencv_setting_dict = None

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
        image_a = image_a_port.dpg_tag
        image_a_value = image_a_port.value_tag
        mask_port = ports.mask
        mask = mask_port.dpg_tag
        mask_value = mask_port.value_tag
        image_b_port = ports.image_b
        image_b = image_b_port.dpg_tag
        image_b_value = image_b_port.value_tag
        invert_mask_port = ports.invert_mask
        invert_mask = invert_mask_port.dpg_tag
        invert_mask_value = invert_mask_port.value_tag
        image_port = ports.image
        image = image_port.dpg_tag
        image_value = image_port.value_tag
        elapsed_port = ports.elapsed
        elapsed = elapsed_port.dpg_tag
        elapsed_value = elapsed_port.value_tag
        status_attr = self._status_attr_tag(node_id)
        status_value = self._status_value_tag(node_id)
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
                tag=image_value,
                format=dpg.mvFormat_Float_rgb,
            )

        with dpg.node(tag=tag_node_name, parent=parent, label=self.node_label, pos=pos):
            self.add_editor_toolbar(node_id, callback=callback)
            with dpg.node_attribute(
                tag=status_attr,
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_text(
                    'waiting for Image A and Mask',
                    tag=status_value,
                )
            with dpg.node_attribute(
                tag=image_a,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(tag=image_a_value, default_value='Image A (white)')
            with dpg.node_attribute(
                tag=mask,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(tag=mask_value, default_value='Mask')
            with dpg.node_attribute(
                tag=image_b,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(tag=image_b_value, default_value='Image B / black')
            with dpg.node_attribute(
                tag=invert_mask,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_checkbox(
                    tag=invert_mask_value,
                    label='invert mask',
                    default_value=False,
                )
            with dpg.node_attribute(
                tag=image,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(image_value)
            if use_pref_counter:
                with dpg.node_attribute(
                    tag=elapsed,
                    attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(elapsed_value, default_value='elapsed time(ms)')

        return tag_node_name

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        del node_result_dict
        ports = self.ports(node_id)
        output_value_tag = ports.image.value_tag
        elapsed_value_tag = ports.elapsed.value_tag
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        image_a_port_name = ports.image_a.port_name
        mask_port_name = ports.mask.port_name
        image_b_port_name = ports.image_b.port_name
        status_value_tag = self._status_value_tag(node_id)

        images_by_port = {}
        for (
            connection_info,
            source_tag,
            destination_tag,
            connection_type,
        ) in self._iter_connection_infos(connection_list):
            if enum_value(connection_type) != self.TYPE_IMAGE:
                continue
            destination_port_name = self._connection_port_name(
                connection_info, destination_tag
            )
            source_node_key = self._connection_source_node_key(
                connection_info, source_tag
            )
            images_by_port[destination_port_name] = node_image_dict.get(source_node_key)

        image_a = images_by_port.get(image_a_port_name)
        mask = images_by_port.get(mask_port_name)
        image_b = images_by_port.get(image_b_port_name)
        frame = image_a
        result = None

        if image_a is None:
            dpg_set_value(status_value_tag, 'waiting for Image A')
        elif mask is None:
            dpg_set_value(status_value_tag, 'waiting for Mask; passing Image A through')
        else:
            start_time = time.perf_counter() if use_pref_counter else None
            invert_mask = bool(dpg_get_value(ports.invert_mask.value_tag))
            mask_bool = _mask_to_bool(mask, image_a)
            mask_white_ratio = float(np.mean(mask_bool)) * 100.0
            frame = image_process(image_a, mask, image_b, invert_mask)
            dpg_set_value(
                status_value_tag,
                f'composited; mask white {mask_white_ratio:.1f}%',
            )
            if use_pref_counter and start_time is not None:
                elapsed_time = int((time.perf_counter() - start_time) * 1000)
                dpg_set_value(elapsed_value_tag, str(elapsed_time).zfill(4) + 'ms')

        if frame is not None:
            texture = convert_cv_to_dpg(frame, small_window_w, small_window_h)
            dpg_set_value(output_value_tag, texture)

        return frame, result

    def _status_attr_tag(self, node_id):
        return f'{self._node_name(node_id)}:StatusAttr'

    def _status_value_tag(self, node_id):
        return f'{self._node_name(node_id)}:StatusValue'

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        invert_value_tag = self.ports(node_id).invert_mask.value_tag
        return {
            'ver': self._ver,
            'pos': dpg.get_item_pos(tag_node_name),
            invert_value_tag: dpg_get_value(invert_value_tag),
        }

    def set_setting_dict(self, node_id, setting_dict):
        invert_value_tag = self.ports(node_id).invert_mask.value_tag
        if invert_value_tag in setting_dict:
            dpg_set_value(invert_value_tag, bool(setting_dict[invert_value_tag]))
