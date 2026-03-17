#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


_COLOR_SPACES = ('HSV', 'LAB', 'LUV', 'RGB')


def _rotate_hue_hsv(bgr_image, hue_shift_degrees):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hue_shift = int(round(hue_shift_degrees / 2.0))
    hsv_image[:, :, 0] = (
        (hsv_image[:, :, 0].astype(np.int16) + hue_shift) % 180
    ).astype(np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def _rotate_hue_lab(bgr_image, hue_shift_degrees):
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

    ab_image = lab_image[:, :, 1:3].astype(np.float32) - 128.0
    theta = np.deg2rad(float(hue_shift_degrees))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    a_rot = ab_image[:, :, 0] * cos_theta - ab_image[:, :, 1] * sin_theta
    b_rot = ab_image[:, :, 0] * sin_theta + ab_image[:, :, 1] * cos_theta

    lab_image[:, :, 1] = np.clip(a_rot + 128.0, 0, 255).astype(np.uint8)
    lab_image[:, :, 2] = np.clip(b_rot + 128.0, 0, 255).astype(np.uint8)

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)




def _rotate_hue_luv(bgr_image, hue_shift_degrees):
    luv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LUV)

    uv_image = luv_image[:, :, 1:3].astype(np.float32) - 128.0
    theta = np.deg2rad(float(hue_shift_degrees))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    u_rot = uv_image[:, :, 0] * cos_theta - uv_image[:, :, 1] * sin_theta
    v_rot = uv_image[:, :, 0] * sin_theta + uv_image[:, :, 1] * cos_theta

    luv_image[:, :, 1] = np.clip(u_rot + 128.0, 0, 255).astype(np.uint8)
    luv_image[:, :, 2] = np.clip(v_rot + 128.0, 0, 255).astype(np.uint8)

    return cv2.cvtColor(luv_image, cv2.COLOR_LUV2BGR)


def _rotate_hue_rgb(bgr_image, hue_shift_degrees):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32)

    theta = np.deg2rad(float(hue_shift_degrees))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    axis = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)

    kx, ky, kz = axis
    k_mat = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float32,
    )
    identity = np.eye(3, dtype=np.float32)
    outer = np.outer(axis, axis).astype(np.float32)
    rotation = cos_theta * identity + sin_theta * k_mat + (1.0 - cos_theta) * outer

    rotated_rgb = np.tensordot(rgb_image, rotation.T, axes=([2], [0]))
    rotated_rgb = np.clip(rotated_rgb, 0, 255).astype(np.uint8)

    return cv2.cvtColor(rotated_rgb, cv2.COLOR_RGB2BGR)

def image_process(image, hue_shift_degrees, color_space='HSV'):
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return image

    color_space = color_space if color_space in _COLOR_SPACES else 'HSV'

    bgr_image = image[:, :, :3]
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else None

    if color_space == 'LAB':
        rotated_bgr = _rotate_hue_lab(bgr_image, hue_shift_degrees)
    elif color_space == 'LUV':
        rotated_bgr = _rotate_hue_luv(bgr_image, hue_shift_degrees)
    elif color_space == 'RGB':
        rotated_bgr = _rotate_hue_rgb(bgr_image, hue_shift_degrees)
    else:
        rotated_bgr = _rotate_hue_hsv(bgr_image, hue_shift_degrees)

    if alpha_channel is not None:
        return cv2.merge(
            (
                rotated_bgr[:, :, 0],
                rotated_bgr[:, :, 1],
                rotated_bgr[:, :, 2],
                alpha_channel,
            )
        )

    return rotated_bgr


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.3'

    node_label = 'HueRotation'
    node_tag = 'HueRotation'

    parameters = [
        {
            'name': 'hue_shift_degrees',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'degree',
            'default': 0,
            'min': -180,
            'max': 180,
            'cast': int,
        },
        {
            'name': 'color_space',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input03',
            'widget': 'combo',
            'label': 'space',
            'items': list(_COLOR_SPACES),
            'default': 'HSV',
        },
    ]

    def process(self, frame, **parameter_values):
        hue_shift_degrees = parameter_values['hue_shift_degrees']
        color_space = parameter_values['color_space']
        frame = image_process(frame, hue_shift_degrees, color_space)
        return frame, None
