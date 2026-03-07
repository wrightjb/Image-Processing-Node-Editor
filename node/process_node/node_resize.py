#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node_editor.util import dpg_set_value


def image_process(image, width, height, interpolation_flag):
    image = cv2.resize(
        image,
        dsize=(width, height),
        interpolation=interpolation_flag,
    )
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Resize'
    node_tag = 'Resize'

    _interpolation = {
        'INTER_LINEAR': getattr(cv2, 'INTER_LINEAR', 1),
        'INTER_NEAREST': getattr(cv2, 'INTER_NEAREST', 0),
        'INTER_AREA': getattr(cv2, 'INTER_AREA', 3),
        'INTER_CUBIC': getattr(cv2, 'INTER_CUBIC', 2),
        'INTER_LANCZOS4': getattr(cv2, 'INTER_LANCZOS4', 4),
        'INTER_NEAREST_EXACT': getattr(cv2, 'INTER_NEAREST_EXACT', 6),
    }

    parameters = [
        {
            'name': 'interpolation_text',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input04',
            'widget': 'combo',
            'label': '',
            'items': list(_interpolation.keys()),
            'default': 'INTER_LINEAR',
        },
        {
            'name': 'width',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'input_int',
            'label': 'Width',
            'default': 960,
            'min': 1,
            'max': 4096,
            'cast': int,
        },
        {
            'name': 'height',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03',
            'widget': 'input_int',
            'label': 'Height',
            'default': 540,
            'min': 1,
            'max': 4096,
            'cast': int,
        },
    ]

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        width = parameter_values['width']
        height = parameter_values['height']
        interpolation_text = parameter_values['interpolation_text']

        width = max(1, min(4096, width))
        height = max(1, min(4096, height))

        dpg_set_value(
            self._value_tag(self._port_tag(tag_node_name, self.TYPE_INT, 'Input02')),
            width,
        )
        dpg_set_value(
            self._value_tag(self._port_tag(tag_node_name, self.TYPE_INT, 'Input03')),
            height,
        )

        if interpolation_text not in self._interpolation:
            interpolation_text = 'INTER_LINEAR'
            dpg_set_value(
                self._value_tag(self._port_tag(tag_node_name, self.TYPE_TEXT, 'Input04')),
                interpolation_text,
            )

        parameter_values['width'] = width
        parameter_values['height'] = height
        parameter_values['interpolation_text'] = interpolation_text
        return parameter_values

    def process(self, frame, **parameter_values):
        interpolation_flag = self._interpolation[parameter_values['interpolation_text']]
        frame = image_process(
            frame,
            parameter_values['width'],
            parameter_values['height'],
            interpolation_flag,
        )
        return frame, None
