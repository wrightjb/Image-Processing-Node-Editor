#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


_MODES = ('MacGyver parity', 'Standard')


def _clip_uint8(float_image):
    return np.clip(np.rint(float_image), 0, 255).astype(np.uint8)


def _split_bgr_alpha(image):
    bgr_image = image[:, :, :3]
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else None
    return bgr_image, alpha_channel


def _merge_alpha(bgr_image, alpha_channel):
    if alpha_channel is None:
        return bgr_image
    return np.dstack((bgr_image, alpha_channel)).astype(bgr_image.dtype, copy=False)


def _apply_macgyver_rgb(
    rgb_image, exposure, brightness, contrast, saturation, temperature, tint, hue,
):
    rgb = rgb_image.astype(np.float32)

    if exposure != 0:
        rgb *= 2.0 ** (float(exposure) / 127.0)
        rgb = np.clip(rgb, 0.0, 255.0)

    if brightness != 0:
        brightness = float(brightness)
        if brightness > 0:
            rgb += brightness * ((255.0 - rgb) / 255.0)
        else:
            rgb += brightness * (rgb / 255.0)
        rgb = np.clip(rgb, 0.0, 255.0)

    if contrast != 0:
        slope = 1.0 + (float(contrast) / 126.0)
        rgb = 127.0 + (rgb - 127.0) * slope
        rgb = np.clip(rgb, 0.0, 255.0)

    if temperature != 6500:
        dt = float(temperature) - 6500.0
        factor = dt / 3500.0
        if dt > 0:
            scales = np.array(
                [1.0 - factor * 0.094, 1.0 - factor * 0.016, 1.0 + factor * 0.148],
                dtype=np.float32,
            )
        else:
            scales = np.array(
                [1.0 - factor * 0.250, 1.0 + factor * 0.133, 1.0 + factor * 0.469],
                dtype=np.float32,
            )
        rgb *= scales
        rgb = np.clip(rgb, 0.0, 255.0)

    if tint != 0:
        factor = float(tint) / 100.0
        scales = np.array(
            [1.0 - factor * 0.242, 1.0 + factor * 0.195, 1.0 - factor * 0.242],
            dtype=np.float32,
        )
        rgb *= scales
        rgb = np.clip(rgb, 0.0, 255.0)

    if saturation != 100:
        luma = (
            0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]
        )
        sat_factor = float(saturation) / 100.0
        rgb = luma[:, :, None] + (rgb - luma[:, :, None]) * sat_factor
        rgb = np.clip(rgb, 0.0, 255.0)

    if hue != 0:
        theta = np.deg2rad(float(hue))
        c = np.cos(theta)
        s = np.sin(theta)
        matrix = np.array(
            [
                [
                    0.213 + 0.787 * c - 0.213 * s,
                    0.715 - 0.715 * c - 0.715 * s,
                    0.072 - 0.072 * c + 0.928 * s,
                ],
                [
                    0.213 - 0.213 * c + 0.143 * s,
                    0.715 + 0.285 * c + 0.140 * s,
                    0.072 - 0.072 * c - 0.283 * s,
                ],
                [
                    0.213 - 0.213 * c - 0.787 * s,
                    0.715 - 0.715 * c + 0.715 * s,
                    0.072 + 0.928 * c + 0.072 * s,
                ],
            ],
            dtype=np.float32,
        )
        rgb = np.tensordot(rgb, matrix.T, axes=([2], [0]))
        rgb = np.clip(rgb, 0.0, 255.0)

    return rgb


def _apply_standard_bgr(
    bgr_image, exposure, brightness, contrast, saturation, temperature, tint, hue,
):
    bgr = bgr_image.astype(np.float32)

    if exposure != 0:
        bgr *= 2.0 ** (float(exposure) / 127.0)
    if brightness != 0:
        bgr += float(brightness)
    if contrast != 0:
        slope = 1.0 + (float(contrast) / 127.0)
        bgr = 127.5 + (bgr - 127.5) * slope
    bgr = np.clip(bgr, 0.0, 255.0)

    if temperature != 6500 or tint != 0:
        lab = cv2.cvtColor(_clip_uint8(bgr), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] += (float(temperature) - 6500.0) / 70.0
        lab[:, :, 1] += float(tint)
        bgr = cv2.cvtColor(_clip_uint8(lab), cv2.COLOR_LAB2BGR).astype(np.float32)

    if saturation != 100 or hue != 0:
        hsv = cv2.cvtColor(_clip_uint8(bgr), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + (float(hue) / 2.0), 180.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (float(saturation) / 100.0), 0.0, 255.0)
        bgr = cv2.cvtColor(_clip_uint8(hsv), cv2.COLOR_HSV2BGR).astype(np.float32)

    return bgr


def image_process(
    image,
    mode='MacGyver parity',
    exposure=0,
    brightness=0,
    contrast=0,
    saturation=100,
    temperature=6500,
    tint=0,
    hue=0,
):
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return image

    bgr_image, alpha_channel = _split_bgr_alpha(image)
    if mode == 'Standard':
        adjusted_bgr = _clip_uint8(
            _apply_standard_bgr(
                bgr_image, exposure, brightness, contrast, saturation, temperature, tint, hue,
            )
        )
    else:
        rgb_image = bgr_image[:, :, ::-1]
        adjusted_rgb = _clip_uint8(
            _apply_macgyver_rgb(
                rgb_image, exposure, brightness, contrast, saturation, temperature, tint, hue,
            )
        )
        adjusted_bgr = adjusted_rgb[:, :, ::-1]

    return _merge_alpha(adjusted_bgr, alpha_channel)


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Photo Editor Color'
    node_tag = 'PhotoEditorColor'

    parameters = [
        {
            'name': 'mode', 'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input02', 'widget': 'combo', 'label': 'mode',
            'items': list(_MODES), 'default': 'MacGyver parity',
        },
        {
            'name': 'exposure', 'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03', 'widget': 'slider_int', 'label': 'exposure',
            'default': 0, 'min': -127, 'max': 127, 'cast': int,
        },
        {
            'name': 'brightness', 'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input04', 'widget': 'slider_int', 'label': 'brightness',
            'default': 0, 'min': -127, 'max': 127, 'cast': int,
        },
        {
            'name': 'contrast', 'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input05', 'widget': 'slider_int', 'label': 'contrast',
            'default': 0, 'min': -127, 'max': 127, 'cast': int,
        },
        {
            'name': 'saturation', 'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input06', 'widget': 'slider_int', 'label': 'saturation',
            'default': 100, 'min': 0, 'max': 200, 'cast': int,
        },
        {
            'name': 'temperature', 'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input07', 'widget': 'slider_int', 'label': 'temperature',
            'default': 6500, 'min': 3000, 'max': 17000, 'cast': int,
        },
        {
            'name': 'tint', 'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input08', 'widget': 'slider_int', 'label': 'tint',
            'default': 0, 'min': -100, 'max': 200, 'cast': int,
        },
        {
            'name': 'hue', 'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input09', 'widget': 'slider_int', 'label': 'hue',
            'default': 0, 'min': -180, 'max': 180, 'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        return image_process(frame, **parameter_values), None
