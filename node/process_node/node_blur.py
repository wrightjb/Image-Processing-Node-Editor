#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Blur'
    node_tag = 'Blur'

    parameters = [
        {
            'name': 'kernel_size',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'kernel',
            'default': 5,
            'min': 1,
            'max': 128,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        kernel_size = parameter_values['kernel_size']
        frame = image_process(frame, kernel_size)
        return frame, None
