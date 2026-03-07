#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, x0y0, x1y0, x2y0, x0y1, x1y1, x2y1, x0y2, x1y2, x2y2, k):
    kernel = np.array(
        [
            [x0y0, x1y0, x2y0],
            [x0y1, x1y1, x2y1],
            [x0y2, x1y2, x2y2],
        ]
    ) * k
    image = cv2.filter2D(image, -1, kernel)
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Simple Filter'
    node_tag = 'SimpleFilter'

    parameters = [
        {
            'name': 'x0y0',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input02',
            'widget': 'slider_float',
            'label': 'x-1,y-1',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x1y0',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input03',
            'widget': 'slider_float',
            'label': 'x, y-1',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x2y0',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input04',
            'widget': 'slider_float',
            'label': 'x+1, y-1',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x0y1',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input05',
            'widget': 'slider_float',
            'label': 'x-1, y',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x1y1',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input06',
            'widget': 'slider_float',
            'label': 'x, y',
            'default': 1.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x2y1',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input07',
            'widget': 'slider_float',
            'label': 'x+1, y',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x0y2',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input08',
            'widget': 'slider_float',
            'label': 'x-1, y+1',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x1y2',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input09',
            'widget': 'slider_float',
            'label': 'x, y+1',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'x2y2',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input10',
            'widget': 'slider_float',
            'label': 'x+1, y+1',
            'default': 0.0,
            'min': -1.0,
            'max': 1.0,
            'cast': float,
            'precision': 3,
        },
        {
            'name': 'k',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input11',
            'widget': 'slider_float',
            'label': 'K',
            'default': 1.0,
            'min': 0.0,
            'max': 10.0,
            'cast': float,
            'precision': 3,
        },
    ]

    def process(self, frame, **parameter_values):
        frame = image_process(
            frame,
            parameter_values['x0y0'],
            parameter_values['x1y0'],
            parameter_values['x2y0'],
            parameter_values['x0y1'],
            parameter_values['x1y1'],
            parameter_values['x2y1'],
            parameter_values['x0y2'],
            parameter_values['x1y2'],
            parameter_values['x2y2'],
            parameter_values['k'],
        )
        return frame, None
