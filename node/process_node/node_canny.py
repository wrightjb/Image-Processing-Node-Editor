#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node_editor.util import dpg_set_value


def image_process(image, min_val, max_val):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, min_val, max_val)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Canny'
    node_tag = 'Canny'

    parameters = [
        {
            'name': 'min_val',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'min val',
            'default': 100,
            'min': 1,
            'max': 254,
            'cast': int,
        },
        {
            'name': 'max_val',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03',
            'widget': 'slider_int',
            'label': 'max val',
            'default': 200,
            'min': 2,
            'max': 255,
            'cast': int,
        },
    ]

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        min_val = parameter_values['min_val']
        max_val = parameter_values['max_val']

        if min_val > max_val:
            min_val, max_val = max_val - 1, min_val + 1
            dpg_set_value(
                self._port_value_tag(tag_node_name, self.TYPE_INT, 'Input02'),
                min_val,
            )
            dpg_set_value(
                self._port_value_tag(tag_node_name, self.TYPE_INT, 'Input03'),
                max_val,
            )

        parameter_values['min_val'] = min_val
        parameter_values['max_val'] = max_val
        return parameter_values

    def process(self, frame, **parameter_values):
        frame = image_process(frame, parameter_values['min_val'], parameter_values['max_val'])
        return frame, None
