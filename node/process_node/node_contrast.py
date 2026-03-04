#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Contrast'
    node_tag = 'Contrast'

    parameters = [
        {
            'name': 'alpha',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input02',
            'widget': 'slider_float',
            'label': 'alpha',
            'default': 1.0,
            'min': 0.0,
            'max': 4.0,
            'cast': float,
            'precision': 3,
        },
    ]

    def process(self, frame, **parameter_values):
        alpha = parameter_values['alpha']
        frame = image_process(frame, alpha)
        return frame, None
