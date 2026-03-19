#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, warmth, tint):
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return image

    bgr_image = image[:, :, :3]
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else None

    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    lab_image = lab_image.astype(np.int16)

    lab_image[:, :, 1] = np.clip(lab_image[:, :, 1] + int(tint), 0, 255)
    lab_image[:, :, 2] = np.clip(lab_image[:, :, 2] + int(warmth), 0, 255)

    balanced_bgr = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if alpha_channel is not None:
        return cv2.merge(
            (
                balanced_bgr[:, :, 0],
                balanced_bgr[:, :, 1],
                balanced_bgr[:, :, 2],
                alpha_channel,
            )
        )

    return balanced_bgr


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Warmth & Tint'
    node_tag = 'WarmthTint'

    parameters = [
        {
            'name': 'warmth',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'warmth',
            'default': 0,
            'min': -100,
            'max': 100,
            'cast': int,
        },
        {
            'name': 'tint',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03',
            'widget': 'slider_int',
            'label': 'tint',
            'default': 0,
            'min': -100,
            'max': 100,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        warmth = parameter_values['warmth']
        tint = parameter_values['tint']
        frame = image_process(frame, warmth, tint)
        return frame, None
