#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, colormap):
    image = cv2.applyColorMap(image, colormap)
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'ApplyColorMap'
    node_tag = 'ApplyColorMap'

    _colormap_types = {
        'COLORMAP_AUTUMN': cv2.COLORMAP_AUTUMN,
        'COLORMAP_BONE': cv2.COLORMAP_BONE,
        'COLORMAP_JET': cv2.COLORMAP_JET,
        'COLORMAP_WINTER': cv2.COLORMAP_WINTER,
        'COLORMAP_RAINBOW': cv2.COLORMAP_RAINBOW,
        'COLORMAP_OCEAN': cv2.COLORMAP_OCEAN,
        'COLORMAP_SUMMER': cv2.COLORMAP_SUMMER,
        'COLORMAP_SPRING': cv2.COLORMAP_SPRING,
        'COLORMAP_COOL': cv2.COLORMAP_COOL,
        'COLORMAP_HSV': cv2.COLORMAP_HSV,
        'COLORMAP_PINK': cv2.COLORMAP_PINK,
        'COLORMAP_HOT': cv2.COLORMAP_HOT,
        'COLORMAP_PARULA': cv2.COLORMAP_PARULA,
        'COLORMAP_MAGMA': cv2.COLORMAP_MAGMA,
        'COLORMAP_INFERNO': cv2.COLORMAP_INFERNO,
        'COLORMAP_PLASMA': cv2.COLORMAP_PLASMA,
        'COLORMAP_VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'COLORMAP_CIVIDIS': cv2.COLORMAP_CIVIDIS,
        'COLORMAP_TWILIGHT': cv2.COLORMAP_TWILIGHT,
        'COLORMAP_TWILIGHT_SHIFTED': cv2.COLORMAP_TWILIGHT_SHIFTED,
        'COLORMAP_TURBO': cv2.COLORMAP_TURBO,
        'COLORMAP_DEEPGREEN': cv2.COLORMAP_DEEPGREEN,
    }

    parameters = [
        {
            'name': 'colormap_type',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input02',
            'widget': 'combo',
            'label': 'type',
            'items': list(_colormap_types.keys()),
            'default': 'COLORMAP_JET',
        },
    ]

    def process(self, frame, **parameter_values):
        colormap = self._colormap_types[parameter_values['colormap_type']]
        frame = image_process(frame, colormap)
        return frame, None
