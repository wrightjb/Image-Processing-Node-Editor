#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.process_node.node_gaussian_blur import image_process as gaussian_blur
from node.smart_blur import smart_blur
from node.stack_blur import MAX_RADIUS, stack_blur


BLUR_METHODS = ('Box Blur', 'Gaussian Blur', 'Stack Blur', 'Smart Blur')


def image_process(
    image,
    kernel_size,
    method='Box Blur',
    sigma=0.0,
    radius=150,
    threshold=10,
    auto_sigma=True,
    auto_kernel=False,
    kernel_factor=3.0,
):
    if method == 'Gaussian Blur':
        if auto_sigma and not auto_kernel:
            sigma = 0.0
        return gaussian_blur(
            image,
            kernel_size,
            sigma,
            auto_kernel=auto_kernel,
            kernel_factor=kernel_factor,
        )
    if method == 'Stack Blur':
        return stack_blur(image, radius)
    if method == 'Smart Blur':
        return smart_blur(image, radius, threshold)
    return cv2.blur(image, (kernel_size, kernel_size))


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.2.0'

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
            'max': MAX_RADIUS,
            'cast': int,
        },
        {
            'name': 'threshold',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input09',
            'widget': 'slider_int',
            'label': 'Smart threshold',
            'default': 10,
            'min': 1,
            'max': 255,
            'cast': int,
        },
        {
            'name': 'auto_sigma',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input06',
            'widget': 'checkbox',
            'label': 'Auto Sigma',
            'default': True,
            'cast': bool,
        },
        {
            'name': 'auto_kernel',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input07',
            'widget': 'checkbox',
            'label': 'Auto Kernel',
            'default': False,
            'cast': bool,
        },
        {
            'name': 'kernel_factor',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input08',
            'widget': 'slider_float',
            'label': 'Gaussian kernel factor',
            'default': 3.0,
            'min': 0.1,
            'max': 10.0,
            'cast': float,
            'precision': 2,
        },
    ]

    def process(self, frame, **parameter_values):
        frame = image_process(
            frame,
            parameter_values['kernel_size'],
            parameter_values['method'],
            parameter_values['sigma'],
            parameter_values['radius'],
            parameter_values.get('threshold', 10),
            parameter_values.get('auto_sigma', True),
            parameter_values.get('auto_kernel', False),
            parameter_values.get('kernel_factor', 3.0),
        )
        return frame, None
