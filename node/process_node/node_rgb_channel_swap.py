#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


_CHANNEL_ORDERS = ('RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR')
_BGR_INDEX_BY_RGB_NAME = {
    'R': 2,
    'G': 1,
    'B': 0,
}


def image_process(image, channel_order):
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return image

    if channel_order not in _CHANNEL_ORDERS:
        channel_order = 'RGB'

    bgr_image = image[:, :, :3]
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else None

    output_bgr = bgr_image.copy()
    # ``channel_order`` is expressed in user-facing RGB order. For example,
    # "BGR" means output red receives source blue, output green receives
    # source green, and output blue receives source red.
    for output_rgb_name, source_rgb_name in zip('RGB', channel_order):
        output_bgr_index = _BGR_INDEX_BY_RGB_NAME[output_rgb_name]
        source_bgr_index = _BGR_INDEX_BY_RGB_NAME[source_rgb_name]
        output_bgr[:, :, output_bgr_index] = bgr_image[:, :, source_bgr_index]

    if alpha_channel is not None:
        return np.dstack((output_bgr, alpha_channel))

    return output_bgr


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'RGB Channel Swap'
    node_tag = 'RGBChannelSwap'

    parameters = [
        {
            'name': 'channel_order',
            'type': DeclarativeImageProcessNodeBase.TYPE_TEXT,
            'port': 'Input02',
            'widget': 'combo',
            'label': 'RGB order',
            'items': list(_CHANNEL_ORDERS),
            'default': 'RGB',
        },
    ]

    def process(self, frame, **parameter_values):
        channel_order = parameter_values['channel_order']
        frame = image_process(frame, channel_order)
        return frame, None
