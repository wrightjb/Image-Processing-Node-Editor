#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lossy image compression node for matching saved JPEG artifacts."""

import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, quality):
    quality = max(1, min(100, int(quality)))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode('.jpg', image, encode_params)
    if not success:
        return image.copy()

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return image.copy()

    return decoded


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Image Compression'
    node_tag = 'ImageCompression'

    parameters = [
        {
            'name': 'quality',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'quality',
            'default': 75,
            'min': 1,
            'max': 100,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        quality = parameter_values['quality']
        frame = image_process(frame, quality)
        return frame, None
