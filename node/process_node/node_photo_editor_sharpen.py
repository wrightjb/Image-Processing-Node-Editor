#!/usr/bin/env python
# -*- coding: utf-8 -*-

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.sharpen import photo_editor_sharpen


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Photo Editor Sharpen'
    node_tag = 'PhotoEditorSharpen'

    parameters = [
        {
            'name': 'strength',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'strength',
            'default': 0,
            'min': 0,
            'max': 500,
            'cast': int,
        },
    ]

    def process(self, frame, **parameter_values):
        return photo_editor_sharpen(frame, parameter_values['strength']), None
