#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase


def auto_kernel_size(sigma, kernel_factor=3.0):
    kernel_size = int(round(float(sigma) * float(kernel_factor)) * 2 + 1)
    return max(1, kernel_size)


def auto_sigma_value(kernel_size):
    if int(kernel_size) % 2 == 0:
        kernel_size = int(kernel_size) + 1
    return ((int(kernel_size) - 1) * 0.5 - 1) * 0.3 + 0.8


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
            'max': 200.0,
            'cast': float,
            'precision': 3,
        },
    ]

    def _set_parameter_value(self, value_tag, value):
        dpg.set_value(value_tag, value)
        input_tag = self._slider_input_tag(value_tag)
        if dpg.does_item_exist(input_tag):
            dpg.set_value(input_tag, value)

    def _refresh_auto_kernel_value(self, kernel_tag, sigma_tag, kernel_factor_tag):
        sigma = dpg.get_value(sigma_tag)
        kernel_factor = dpg.get_value(kernel_factor_tag)
        self._set_parameter_value(
            kernel_tag,
            auto_kernel_size(sigma, kernel_factor),
        )

    def _refresh_auto_sigma_value(self, kernel_tag, sigma_tag):
        self._set_parameter_value(
            sigma_tag,
            round(auto_sigma_value(dpg.get_value(kernel_tag)), 3),
        )

    def _chain_parameter_callback(self, value_tag, callback):
        previous_callback = dpg.get_item_callback(value_tag)
        previous_user_data = dpg.get_item_user_data(value_tag)

        def _callback(sender, app_data, user_data):
            if callable(previous_callback):
                previous_callback(sender, app_data, previous_user_data)
            callback(sender, app_data, user_data)

        dpg.configure_item(value_tag, callback=_callback)

    def _chain_slider_callbacks(self, value_tag, callback):
        self._chain_parameter_callback(value_tag, callback)
        input_tag = self._slider_input_tag(value_tag)
        if dpg.does_item_exist(input_tag):
            self._chain_parameter_callback(input_tag, callback)

    def _gaussian_tags(self, tag_node_name):
        return {
            'kernel': self._value_tag(
                self._port_tag(tag_node_name, self.TYPE_INT, 'Input02')
            ),
            'sigma': self._value_tag(
                self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input03')
            ),
            'auto_sigma': self._value_tag(
                self._port_tag(tag_node_name, self.TYPE_INT, 'Input04')
            ),
            'auto_kernel': self._value_tag(
                self._port_tag(tag_node_name, self.TYPE_INT, 'Input05')
            ),
            'kernel_factor': self._value_tag(
                self._port_tag(tag_node_name, self.TYPE_FLOAT, 'Input06')
            ),
        }

    def _refresh_auto_parameter_display(self, tag_node_name):
        tags = self._gaussian_tags(tag_node_name)
        auto_kernel = bool(dpg.get_value(tags['auto_kernel']))
        auto_sigma = bool(dpg.get_value(tags['auto_sigma'])) and not auto_kernel
        if auto_kernel:
            self._refresh_auto_kernel_value(
                tags['kernel'],
                tags['sigma'],
                tags['kernel_factor'],
            )
        elif auto_sigma:
            self._refresh_auto_sigma_value(tags['kernel'], tags['sigma'])
        dpg.configure_item(tags['sigma'], enabled=not auto_sigma)
        dpg.configure_item(tags['kernel'], enabled=not auto_kernel)

    def on_node_added(self, tag_node_name):
        tags = self._gaussian_tags(tag_node_name)

        def _toggle_sigma(_sender, app_data, _user_data):
            auto_sigma = bool(app_data)
            if auto_sigma and dpg.get_value(tags['auto_kernel']):
                dpg.set_value(tags['auto_kernel'], False)
            dpg.configure_item(tags['sigma'], enabled=not auto_sigma)
            dpg.configure_item(tags['kernel'], enabled=True)
            if auto_sigma:
                self._refresh_auto_sigma_value(tags['kernel'], tags['sigma'])

        def _toggle_kernel(_sender, app_data, _user_data):
            auto_kernel = bool(app_data)
            if auto_kernel and dpg.get_value(tags['auto_sigma']):
                dpg.set_value(tags['auto_sigma'], False)
            dpg.configure_item(tags['kernel'], enabled=not auto_kernel)
            dpg.configure_item(tags['sigma'], enabled=True)
            if auto_kernel:
                self._refresh_auto_kernel_value(
                    tags['kernel'],
                    tags['sigma'],
                    tags['kernel_factor'],
                )

        def _source_changed(_sender, _app_data, _user_data):
            self._refresh_auto_parameter_display(tag_node_name)

        self._chain_parameter_callback(tags['auto_sigma'], _toggle_sigma)
        self._chain_parameter_callback(tags['auto_kernel'], _toggle_kernel)
        self._chain_slider_callbacks(tags['kernel'], _source_changed)
        self._chain_slider_callbacks(tags['sigma'], _source_changed)
        self._chain_slider_callbacks(tags['kernel_factor'], _source_changed)
        self._refresh_auto_parameter_display(tag_node_name)

    def on_settings_applied(self, tag_node_name):
        tags = self._gaussian_tags(tag_node_name)
        auto_kernel = bool(dpg.get_value(tags['auto_kernel']))
        auto_sigma = bool(dpg.get_value(tags['auto_sigma'])) and not auto_kernel
        if auto_kernel and dpg.get_value(tags['auto_sigma']):
            dpg.set_value(tags['auto_sigma'], False)
        dpg.configure_item(tags['sigma'], enabled=not auto_sigma)
        dpg.configure_item(tags['kernel'], enabled=not auto_kernel)
        self._refresh_auto_parameter_display(tag_node_name)

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
