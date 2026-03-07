#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, gamma):
    table = (np.arange(256) / 255) ** gamma * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Gamma Correction'
    node_tag = 'GammaCorrection'

    parameters = [
        {
            'name': 'gamma',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input02',
            'widget': 'slider_float',
            'label': 'gamma',
            'default': 1.0,
            'min': 0.01,
            'max': 4.0,
            'cast': float,
            'precision': 3,
        },
    ]

    def process(self, frame, **parameter_values):
        gamma = parameter_values['gamma']
        frame = image_process(frame, gamma)
        return frame, None
