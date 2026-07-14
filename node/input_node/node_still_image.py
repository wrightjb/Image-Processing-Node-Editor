#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node.port_model import OutputPort, PortDataType, PortSpecs
from node_editor.image_metadata import inspect_image_file, summarize_metadata
from node_editor.util import convert_cv_to_dpg


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Image'
    node_tag = 'Image'

    _opencv_setting_dict = None

    _image = {}
    _image_filepath = {}
    _prev_image_filepath = {}
    _metadata = {}

    port_specs = PortSpecs(
        image=OutputPort(PortDataType.IMAGE, index=1),
        metadata=OutputPort(PortDataType.METADATA, index=2),
    )

    def __init__(self):
        self._display_size_dict = {}
        self._current_texture_tag_dict = {}
        self._texture_tags_dict = {}

    def _compute_display_size(self, frame, width):
        image_h, image_w = frame.shape[:2]
        width = max(1, int(width))
        if image_w <= 0 or image_h <= 0:
            return width, width
        scale = width / float(image_w)
        height = max(1, int(round(image_h * scale)))
        return width, height

    def _build_texture_tag_with_size(self, node_id, width, height):
        return f'{self.node_tag}:{node_id}:Output01:Texture:{width}x{height}'

    def _get_image_file_signature(self, image_path):
        if image_path is None:
            return None
        try:
            stat_result = os.stat(image_path)
        except OSError:
            return (image_path, None)
        return (
            image_path,
            stat_result.st_mtime_ns,
            stat_result.st_size,
        )

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        # Tag names
        tag_node_name = self._node_name(node_id)
        tag_node_input01_name = self._port_tag(
            tag_node_name, self.TYPE_INT, 'Input01'
        )
        ports = self.create_ports(node_id)
        tag_node_output01_name_port = ports.image
        tag_node_output01_name = tag_node_output01_name_port.dpg_tag
        tag_node_output01_image_name = tag_node_output01_name_port.value_tag
        tag_node_output02_name_port = ports.metadata
        tag_node_output02_name = tag_node_output02_name_port.dpg_tag
        tag_node_output02_value_name = tag_node_output02_name_port.value_tag

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']
        texture_tag = self._build_texture_tag_with_size(node_id, small_window_w,
                                                        small_window_h)
        self._display_size_dict[node_id] = (small_window_w, small_window_h)
        self._current_texture_tag_dict[node_id] = texture_tag
        self._texture_tags_dict[node_id] = {texture_tag}

        # Black image for initialization
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

        # Register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=texture_tag,
                format=dpg.mvFormat_Float_rgb,
            )

        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(small_window_h * 3),
                callback=self._callback_file_select,
                id='image_select:' + str(node_id),
        ):
            dpg.add_file_extension(
                'Image (*.bmp *.jpg *.png *.gif){.bmp,.jpg,.png,.gif}')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        # Node
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            self.add_editor_toolbar(node_id, callback=callback)
            # File selection
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label='Select Image',
                    width=small_window_w,
                    callback=lambda: dpg.show_item(
                        'image_select:' + str(node_id), ),
                )
            # Camera image
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(
                    texture_tag,
                    tag=tag_node_output01_image_name,
                    width=small_window_w,
                    height=small_window_h,
                )
            with dpg.node_attribute(
                    tag=tag_node_output02_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=tag_node_output02_value_name,
                    default_value='Metadata: none',
                    wrap=small_window_w,
                )

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        ports = self.ports(node_id)
        output_image_tag = ports.image.value_tag
        output_metadata_tag = ports.metadata.value_tag
        texture_tag = self._current_texture_tag_dict.get(node_id, None)

        small_window_w = self._opencv_setting_dict['input_window_width']
        small_window_h = self._opencv_setting_dict['input_window_height']

        # Create VideoCapture() instance
        image_path = self._image_filepath.get(str(node_id), None)
        prev_image_path = self._prev_image_filepath.get(str(node_id), None)
        if prev_image_path != image_path:
            self._image[str(node_id)] = cv2.imread(image_path)
            metadata = inspect_image_file(image_path)
            self._metadata[str(node_id)] = metadata
            dpg_set_value(
                output_metadata_tag,
                summarize_metadata(metadata, multiline=True),
            )
            self._prev_image_filepath[str(node_id)] = image_path

        # Get image
        frame = self._image.get(str(node_id), None)

        # Draw
        if frame is not None:
            display_w, display_h = self._compute_display_size(frame,
                                                              small_window_w)
            previous_size = self._display_size_dict.get(node_id)
            if previous_size != (display_w, display_h):
                texture_tag = self._build_texture_tag_with_size(
                    node_id, display_w, display_h)
                self._current_texture_tag_dict[node_id] = texture_tag
                if node_id not in self._texture_tags_dict:
                    self._texture_tags_dict[node_id] = set()
                empty_texture = np.zeros((display_w * display_h * 3,),
                                         dtype='f')
                if not dpg.does_item_exist(texture_tag):
                    with dpg.texture_registry(show=False):
                        dpg.add_raw_texture(
                            display_w,
                            display_h,
                            empty_texture,
                            tag=texture_tag,
                            format=dpg.mvFormat_Float_rgb,
                        )
                    self._texture_tags_dict[node_id].add(texture_tag)
                dpg.configure_item(
                    output_image_tag,
                    texture_tag=texture_tag,
                    width=display_w,
                    height=display_h,
                )
                self._display_size_dict[node_id] = (display_w, display_h)
            elif texture_tag is None:
                texture_tag = self._build_texture_tag_with_size(
                    node_id, display_w, display_h)
                self._current_texture_tag_dict[node_id] = texture_tag
            texture = convert_cv_to_dpg(
                frame,
                display_w,
                display_h,
            )
            dpg_set_value(texture_tag, texture)

        metadata = self._metadata.get(str(node_id), None)
        if metadata is not None:
            dpg_set_value(
                output_metadata_tag,
                summarize_metadata(metadata, multiline=True),
            )
            
        result = {
            'metadata': metadata,
            '__cache_kind__': 'still_image',
            '__cache_source__': image_path,
            '__cache_file_signature__': self._get_image_file_signature(image_path),
        }

        return frame, result

    def close(self, node_id):
        texture_tags = self._texture_tags_dict.pop(node_id, set())
        self._current_texture_tag_dict.pop(node_id, None)
        self._display_size_dict.pop(node_id, None)
        for texture_tag in texture_tags:
            if dpg.does_item_exist(texture_tag):
                try:
                    dpg.delete_item(texture_tag)
                except (SystemError, RuntimeError):
                    pass

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        pos = dpg.get_item_pos(tag_node_name)
        image_path = self._image_filepath.get(str(node_id), None)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict['image_path'] = image_path
        setting_dict['image_file_signature'] = self._get_image_file_signature(
            image_path
        )
        setting_dict['__cache_source_enabled__'] = True

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        image_path = setting_dict.get('image_path', None)

        if image_path is None:
            return

        node_id = str(node_id)
        if os.path.isfile(image_path):
            self._image_filepath[node_id] = image_path
            return

        self._image_filepath.pop(node_id, None)
        self._prev_image_filepath.pop(node_id, None)
        self._image.pop(node_id, None)
        self._metadata.pop(node_id, None)
        print(f'WARNING : Image file not found ({image_path})')

    def _callback_file_select(self, sender, data):
        if data['file_name'] != '.':
            node_id = sender.split(':')[1]
            self._image_filepath[node_id] = data['file_path_name']
