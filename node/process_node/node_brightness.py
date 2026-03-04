#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, beta):
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Brightness'
    node_tag = 'Brightness'

    parameters = [
        {
            'name': 'beta',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'beta',
            'default': 0,
            'min': 0,
            'max': 255,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        beta = parameter_values['beta']
        frame = image_process(frame, beta)
        return frame, None
