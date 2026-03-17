#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, hue_shift_degrees):
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return image

    bgr_image = image[:, :, :3]
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else None

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hue_shift = int(round(hue_shift_degrees / 2.0))
    hsv_image[:, :, 0] = ((hsv_image[:, :, 0].astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    rotated_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    if alpha_channel is not None:
        return cv2.merge((rotated_bgr[:, :, 0], rotated_bgr[:, :, 1], rotated_bgr[:, :, 2], alpha_channel))

    return rotated_bgr


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

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
    ]

    def process(self, frame, **parameter_values):
        hue_shift_degrees = parameter_values['hue_shift_degrees']
        frame = image_process(frame, hue_shift_degrees)
        return frame, None
