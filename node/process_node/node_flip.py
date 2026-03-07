#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, hflip_flag, vflip_flag):
    flipcode = None
    if hflip_flag and vflip_flag:
        flipcode = 0
    elif hflip_flag:
        flipcode = 1
    elif vflip_flag:
        flipcode = -1

    if flipcode is not None:
        image = cv2.flip(image, flipcode)

    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Flip'
    node_tag = 'Flip'

    parameters = [
        {
            'name': 'hflip_flag',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input02',
            'widget': 'checkbox',
            'label': 'Horizontal flip',
            'default': False,
            'cast': bool,
        },
        {
            'name': 'vflip_flag',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input03',
            'widget': 'checkbox',
            'label': 'Vertical flip',
            'default': False,
            'cast': bool,
        },
    ]

    def process(self, frame, **parameter_values):
        frame = image_process(
            frame,
            parameter_values['hflip_flag'],
            parameter_values['vflip_flag'],
        )
        return frame, None
