#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node_editor.util import dpg_set_value


def image_process(image, min_x, max_x, min_y, max_y):
    if max_x < min_x:
        max_x = min_x + 0.01
    if max_y < min_y:
        max_y = min_y + 0.01

    image_height, image_width = image.shape[0], image.shape[1]
    min_x_ = int(min_x * image_width)
    max_x_ = int(max_x * image_width)
    min_y_ = int(min_y * image_height)
    max_y_ = int(max_y * image_height)
    image = image[min_y_:max_y_, min_x_:max_x_]
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Crop'
    node_tag = 'Crop'

    parameters = [
        {
            'name': 'min_x',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input02',
            'widget': 'slider_float',
            'label': 'min x',
            'default': 0.0,
            'min': 0.0,
            'max': 0.99,
            'cast': float,
        },
        {
            'name': 'max_x',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input03',
            'widget': 'slider_float',
            'label': 'max x',
            'default': 1.0,
            'min': 0.01,
            'max': 1.0,
            'cast': float,
        },
        {
            'name': 'min_y',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input04',
            'widget': 'slider_float',
            'label': 'min y',
            'default': 0.0,
            'min': 0.0,
            'max': 0.99,
            'cast': float,
        },
        {
            'name': 'max_y',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input05',
            'widget': 'slider_float',
            'label': 'max y',
            'default': 1.0,
            'min': 0.01,
            'max': 1.0,
            'cast': float,
        },
    ]

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        min_x = parameter_values['min_x']
        max_x = parameter_values['max_x']
        min_y = parameter_values['min_y']
        max_y = parameter_values['max_y']

        if min_x > max_x:
            min_x, max_x = max_x - 0.01, min_x + 0.01
            dpg_set_value(
                self._port_value_tag(tag_node_name, self.TYPE_FLOAT, 'Input02'),
                min_x,
            )
            dpg_set_value(
                self._port_value_tag(tag_node_name, self.TYPE_FLOAT, 'Input03'),
                max_x,
            )

        if min_y > max_y:
            min_y, max_y = max_y - 0.01, min_y + 0.01
            dpg_set_value(
                self._port_value_tag(tag_node_name, self.TYPE_FLOAT, 'Input04'),
                min_y,
            )
            dpg_set_value(
                self._port_value_tag(tag_node_name, self.TYPE_FLOAT, 'Input05'),
                max_y,
            )

        parameter_values['min_x'] = min_x
        parameter_values['max_x'] = max_x
        parameter_values['min_y'] = min_y
        parameter_values['max_y'] = max_y

        return parameter_values

    def process(self, frame, **parameter_values):
        frame = image_process(
            frame,
            parameter_values['min_x'],
            parameter_values['max_x'],
            parameter_values['min_y'],
            parameter_values['max_y'],
        )
        return frame, None
