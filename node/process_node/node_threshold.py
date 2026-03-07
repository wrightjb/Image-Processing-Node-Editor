#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, threshold_type, binary_threshold):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(
        image,
        binary_threshold,
        255,
        threshold_type,
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Threshold'
    node_tag = 'Threshold'

    _threshold_types = {
        'THRESH_BINARY': getattr(cv2, 'THRESH_BINARY', 0),
        'THRESH_BINARY_INV': getattr(cv2, 'THRESH_BINARY_INV', 1),
        'THRESH_TRUNC': getattr(cv2, 'THRESH_TRUNC', 2),
        'THRESH_TOZERO': getattr(cv2, 'THRESH_TOZERO', 3),
        'THRESH_TOZERO_INV': getattr(cv2, 'THRESH_TOZERO_INV', 4),
        'THRESH_OTSU': getattr(cv2, 'THRESH_BINARY', 0) + getattr(cv2, 'THRESH_OTSU', 8),
    }

    parameters = [
        {
            'name': 'threshold_type',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input02',
            'widget': 'combo',
            'label': 'type',
            'items': list(_threshold_types.keys()),
            'default': 'THRESH_BINARY',
        },
        {
            'name': 'binary_threshold',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03',
            'widget': 'slider_int',
            'label': 'threshold',
            'default': 127,
            'min': 0,
            'max': 255,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        threshold_name = parameter_values['threshold_type']
        threshold_type = self._threshold_types.get(
            threshold_name,
            self._threshold_types['THRESH_BINARY'],
        )
        binary_threshold = parameter_values['binary_threshold']
        frame = image_process(frame, threshold_type, binary_threshold)
        return frame, None
