#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.process_node.node_gaussian_blur import image_process as gaussian_blur
from node.stack_blur import stack_blur


BLUR_METHODS = ('Box Blur', 'Gaussian Blur', 'Stack Blur')


def image_process(image, kernel_size, method='Box Blur', sigma=0.0, radius=150):
    if method == 'Gaussian Blur':
        return gaussian_blur(image, kernel_size, sigma)
    if method == 'Stack Blur':
        return stack_blur(image, radius)
    return cv2.blur(image, (kernel_size, kernel_size))


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.1.0'

    node_label = 'Blur'
    node_tag = 'Blur'

    parameters = [
        {
            'name': 'method',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input03',
            'widget': 'combo',
            'label': 'method',
            'items': BLUR_METHODS,
            'default': 'Box Blur',
        },
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
        {
            'name': 'sigma',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input04',
            'widget': 'slider_float',
            'label': 'Gaussian sigma',
            'default': 0.0,
            'min': 0.0,
            'max': 200.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'radius',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input05',
            'widget': 'slider_int',
            'label': 'Stack radius',
            'default': 150,
            'min': 0,
            'max': 254,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        frame = image_process(
            frame,
            parameter_values['kernel_size'],
            parameter_values['method'],
            parameter_values['sigma'],
            parameter_values['radius'],
        )
        return frame, None
