#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def auto_kernel_size(sigma, kernel_factor=3.0):
    kernel_size = int(round(float(sigma) * float(kernel_factor)) * 2 + 1)
    return max(1, kernel_size)


def image_process(image, kernel_size, sigma, auto_kernel=False, kernel_factor=3.0):
    if auto_kernel:
        kernel_size = auto_kernel_size(sigma, kernel_factor)
    elif kernel_size % 2 == 0:
        kernel_size += 1
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return image


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.3'

    node_label = 'Gaussian Blur'
    node_tag = 'GaussianBlur'

    parameters = [
        {
            'name': 'kernel_size',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input02',
            'widget': 'slider_int',
            'label': 'kernel',
            'default': 5,
            'min': 1,
            'max': 501,
            'cast': int,
        },
        {
            'name': 'auto_sigma',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input04',
            'widget': 'checkbox',
            'label': 'Auto Sigma',
            'default': True,
            'cast': bool,
        },
        {
            'name': 'auto_kernel',
            'type': DeclarativeImageProcessNodeBase.TYPE_INT,
            'port': 'Input05',
            'widget': 'checkbox',
            'label': 'Auto Kernel',
            'default': False,
            'cast': bool,
        },
        {
            'name': 'kernel_factor',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input06',
            'widget': 'slider_float',
            'label': 'kernel factor',
            'default': 3.0,
            'min': 0.1,
            'max': 10.0,
            'cast': float,
            'precision': 2,
        },
        {
            'name': 'sigma',
            'type': DeclarativeImageProcessNodeBase.TYPE_FLOAT,
            'port': 'Input03',
            'widget': 'slider_float',
            'label': 'sigma',
            'default': 0.1,
            'min': 0.1,
            'max': 100.0,
            'cast': float,
            'precision': 3,
        },
    ]

    def on_node_added(self, tag_node_name):
        sigma_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03')
        )
        auto_sigma_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'Input04')
        )
        kernel_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'Input02')
        )
        auto_kernel_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'Input05')
        )

        def _toggle_sigma(_sender, app_data, user_data):
            dpg.configure_item(user_data, enabled=not app_data)
            if self._ui_callback is not None:
                before_value = self._last_parameter_values.get(auto_sigma_tag, bool(app_data))
                self._last_parameter_values[auto_sigma_tag] = bool(app_data)
                self._ui_callback(
                    'parameter_changed',
                    {
                        'node_id_name': tag_node_name,
                        'port_tag': self._port_tag(tag_node_name, self.TYPE_INT, 'Input04'),
                        'value_tag': auto_sigma_tag,
                        'before_value': bool(before_value),
                        'after_value': bool(app_data),
                    },
                )

        def _toggle_kernel(_sender, app_data, user_data):
            dpg.configure_item(user_data, enabled=not app_data)

        dpg.configure_item(auto_sigma_tag, callback=_toggle_sigma, user_data=sigma_tag)
        dpg.configure_item(auto_kernel_tag, callback=_toggle_kernel, user_data=kernel_tag)
        auto_sigma = bool(dpg.get_value(auto_sigma_tag))
        auto_kernel = bool(dpg.get_value(auto_kernel_tag))
        dpg.configure_item(sigma_tag, enabled=not auto_sigma)
        dpg.configure_item(kernel_tag, enabled=not auto_kernel)

    def on_settings_applied(self, tag_node_name):
        sigma_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03')
        )
        auto_sigma_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'Input04')
        )
        kernel_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'Input02')
        )
        auto_kernel_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_INT, 'Input05')
        )
        auto_sigma = bool(dpg.get_value(auto_sigma_tag))
        auto_kernel = bool(dpg.get_value(auto_kernel_tag))
        dpg.configure_item(sigma_tag, enabled=not auto_sigma)
        dpg.configure_item(kernel_tag, enabled=not auto_kernel)

    def normalize_parameter_values(self, tag_node_name, parameter_values):
        del tag_node_name
        if parameter_values['auto_kernel']:
            parameter_values['auto_sigma'] = False
        elif parameter_values['auto_sigma']:
            parameter_values['sigma'] = 0.0
        return parameter_values

    def process(self, frame, **parameter_values):
        frame = image_process(
            frame,
            parameter_values['kernel_size'],
            parameter_values['sigma'],
            parameter_values['auto_kernel'],
            parameter_values['kernel_factor'],
        )
        return frame, None
