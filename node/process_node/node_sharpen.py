#!/usr/bin/env python
# -*- coding: utf-8 -*-

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.sharpen import photo_editor_sharpen, polish_sharpen


SHARPEN_METHODS = ('Photo Editor', 'Polish')


def image_process(image, method, strength, preview_width=1080):
    if method == 'Polish':
        return polish_sharpen(image, strength, preview_width)
    return photo_editor_sharpen(image, strength)


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.1.0'

    node_label = 'Sharpen'
    node_tag = 'Sharpen'

    parameters = [
        {
            'name': 'method',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input02',
            'widget': 'combo',
            'label': 'method',
            'items': SHARPEN_METHODS,
            'default': 'Photo Editor',
        },
        {
            'name': 'strength',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03',
            'widget': 'slider_int',
            'label': 'strength',
            'default': 0,
            'min': 0,
            'max': 500,
            'cast': int,
        },
        {
            'name': 'preview_width',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input04',
            'widget': 'input_int',
            'label': 'Polish preview width',
            'default': 1080,
            'min': 1,
            'max': 8192,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        return image_process(
            frame,
            parameter_values['method'],
            parameter_values['strength'],
            parameter_values.get('preview_width', 1080),
        ), None
