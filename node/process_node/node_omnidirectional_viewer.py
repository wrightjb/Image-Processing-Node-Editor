#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def image_process(image, phi, theta):
    image = remap_image(image, phi, theta)
    return image


def create_rotation_matrix(roll, pitch, yaw):
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180

    matrix01 = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)],
    ])

    matrix02 = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)],
    ])

    matrix03 = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    matrix = np.dot(matrix03, np.dot(matrix02, matrix01))

    return matrix


def calculate_phi_and_theta(
    viewpoint,
    imagepoint,
    sensor_width,
    sensor_height,
    output_width,
    output_height,
    rotation_matrix,
):
    width = np.arange(
        (-1) * sensor_width,
        sensor_width,
        sensor_width * 2 / output_width,
    )
    height = np.arange(
        (-1) * sensor_height,
        sensor_height,
        sensor_height * 2 / output_height,
    )

    ww, hh = np.meshgrid(width, height)

    point_distance = (imagepoint - viewpoint)
    if point_distance == 0:
        point_distance = 0.1

    a1 = ww / point_distance
    a2 = hh / point_distance
    b1 = -a1 * viewpoint
    b2 = -a2 * viewpoint

    a = 1 + (a1**2) + (a2**2)
    b = 2 * ((a1 * b1) + (a2 * b2))
    c = (b1**2) + (b2**2) - 1

    d = ((b**2) - (4 * a * c))**(1 / 2)

    x = (-b + d) / (2 * a)
    y = (a1 * x) + b1
    z = (a2 * x) + b2

    xd = rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z
    yd = rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z
    zd = rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z

    phi = np.arcsin(zd)
    theta = np.arcsin(yd / np.cos(phi))

    xd[xd > 0] = 0
    xd[xd < 0] = 1
    yd[yd > 0] = np.pi
    yd[yd < 0] = -np.pi

    offset = yd * xd
    gain = -2 * xd + 1
    theta = gain * theta + offset

    return phi, theta


def remap_image(image, phi, theta):
    input_height, input_width = image.shape[:2]

    phi = (phi * input_height / np.pi + input_height / 2)
    phi = phi.astype(np.float32)
    theta = (theta * input_width / (2 * np.pi) + input_width / 2)
    theta = theta.astype(np.float32)

    output_image = cv2.remap(image, theta, phi, cv2.INTER_CUBIC)

    return output_image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.2'

    node_label = 'Omnidirectional Viewer'
    node_tag = 'OmnidirectionalViewer'

    _sensor_size = 0.561
    _output_width = 960
    _output_height = 540

    parameters = [
        {
            'name': 'pitch',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'pitch',
            'default': 0,
            'min': 0,
            'max': 359,
            'cast': int,
        },
        {
            'name': 'yaw',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input03',
            'widget': 'slider_int',
            'label': 'yaw',
            'default': 0,
            'min': 0,
            'max': 359,
            'cast': int,
        },
        {
            'name': 'roll',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input04',
            'widget': 'slider_int',
            'label': 'roll',
            'default': 0,
            'min': 0,
            'max': 359,
            'cast': int,
        },
        {
            'name': 'imagepoint',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input05',
            'widget': 'slider_float',
            'label': 'image point',
            'default': 0.0,
            'min': -1.0,
            'max': 3.0,
            'cast': float,
            'precision': 3,
        },
    ]

    def __init__(self):
        self._params = {}

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        node_id = int(str(tag_node_name).split(':', maxsplit=1)[0])
        parameter_values['_node_id'] = node_id
        return parameter_values

    def process(self, frame, **parameter_values):
        node_id = parameter_values.pop('_node_id')

        pitch = parameter_values['pitch']
        yaw = parameter_values['yaw']
        roll = parameter_values['roll']
        imagepoint = parameter_values['imagepoint']

        change_param_flag = False
        if node_id not in self._params:
            change_param_flag = True
        else:
            prev_pitch, prev_yaw, prev_roll, prev_imagepoint = self._params[node_id][:4]
            if prev_pitch != pitch:
                change_param_flag = True
            if prev_yaw != yaw:
                change_param_flag = True
            if prev_roll != roll:
                change_param_flag = True
            if prev_imagepoint != imagepoint:
                change_param_flag = True

        if change_param_flag:
            sensor_width = self._sensor_size
            sensor_height = self._sensor_size * (self._output_height / self._output_width)

            rotation_matrix = create_rotation_matrix(
                roll,
                pitch,
                yaw,
            )

            phi, theta = calculate_phi_and_theta(
                -1.0,
                imagepoint,
                sensor_width,
                sensor_height,
                self._output_width,
                self._output_height,
                rotation_matrix,
            )

            self._params[node_id] = [pitch, yaw, roll, imagepoint, phi, theta]

        phi, theta = self._params[node_id][4], self._params[node_id][5]
        frame = image_process(frame, phi, theta)

        return frame, None

    def close(self, node_id):
        node_id = int(node_id)
        if node_id in self._params:
            del self._params[node_id]
