#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'EqualizeHist'
    node_tag = 'EqualizeHist'

    parameters = []

    def process(self, frame, **parameter_values):
        del parameter_values
        frame = image_process(frame)
        return frame, None
