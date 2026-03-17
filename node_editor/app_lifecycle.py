#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

import dearpygui.dearpygui as dpg

from node_editor.util import check_camera_connection, check_serial_connection


def load_opencv_settings(setting_path):
    with open(setting_path) as fp:
        return json.load(fp)


def initialize_camera_resources(opencv_setting_dict):
    import cv2

    webcam_width = opencv_setting_dict['webcam_width']
    webcam_height = opencv_setting_dict['webcam_height']

    print('**** Check Camera Connection ********')
    device_no_list = check_camera_connection()
    camera_capture_list = []
    for device_no in device_no_list:
        video_capture = cv2.VideoCapture(device_no)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        camera_capture_list.append(video_capture)

    opencv_setting_dict['device_no_list'] = device_no_list
    opencv_setting_dict['camera_capture_list'] = camera_capture_list
    return camera_capture_list


def initialize_serial_resources(opencv_setting_dict):
    serial_device_no_list = []
    serial_connection_list = []

    if opencv_setting_dict.get('use_serial') is True:
        import serial

        print('**** Check Serial Device Connection ********')
        serial_device_no_list = check_serial_connection()
        for serial_device_no in serial_device_no_list:
            serial_connection_list.append(serial.Serial(serial_device_no, 115200))

    opencv_setting_dict['serial_device_no_list'] = serial_device_no_list
    opencv_setting_dict['serial_connection_list'] = serial_connection_list
    return serial_connection_list


def setup_dearpygui(editor_width, editor_height, font_path):
    print('**** DearPyGui Setup ********')
    dpg.create_context()
    dpg.setup_dearpygui()
    dpg.create_viewport(
        title='Image Processing Node Editor',
        width=editor_width,
        height=editor_height,
    )

    with dpg.font_registry():
        with dpg.font(font_path, 16) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
    dpg.bind_font(default_font)


def shutdown_runtime(node_editor, camera_capture_list, serial_connection_list, event_loop=None):
    print('**** Terminate process ********')
    print('**** Close All Node ********')
    for node_id_name in node_editor.get_node_list():
        node_id, node_name = node_id_name.split(':')
        node_instance = node_editor.get_node_instance(node_name)
        node_instance.close(node_id)

    print('**** Release All VideoCapture ********')
    for camera_capture in camera_capture_list:
        camera_capture.release()

    print('**** Release All Serial Connections ********')
    for serial_connection in serial_connection_list:
        serial_connection.close()

    print('**** Stop Event Loop ********')
    node_editor.set_terminate_flag()
    if event_loop is not None:
        event_loop.stop()

    print('**** Destroy DearPyGui Context ********')
    dpg.destroy_context()
