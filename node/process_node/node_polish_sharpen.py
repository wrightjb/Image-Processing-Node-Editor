#!/usr/bin/env python
# -*- coding: utf-8 -*-

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.sharpen import polish_sharpen


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Polish Sharpen'
    node_tag = 'PolishSharpen'

    parameters = [
        {
            'name': 'strength',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'strength',
            'default': 0,
            'min': 0,
            'max': 100,
            'cast': int,
        },
        {
            'name': 'preview_width',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03',
            'widget': 'input_int',
            'label': 'preview width',
            'default': 1080,
            'min': 1,
            'max': 8192,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        return polish_sharpen(
            frame,
            parameter_values['strength'],
            parameter_values.get('preview_width', 1080),
        ), None
