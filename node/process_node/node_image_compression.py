#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg

from node.base.declarative_node_base import DeclarativeImageProcessNodeBase
from node.port_model import OutputPort, PortDataType
from node_editor.image_metadata import (
    compression_roundtrip,
    summarize_metadata,
)
from node_editor.util import dpg_set_value


class Node(DeclarativeImageProcessNodeBase):
    _ver = '0.0.1'

    node_label = 'Image Compression'
    node_tag = 'ImageCompression'

    parameters = [
        {
            'name': 'codec',
            'type': PortDataType.TEXT,
            'port': 'Input02',
            'widget': 'combo',
            'label': 'Codec',
            'items': ['JPEG', 'PNG', 'WEBP'],
            'default': 'JPEG',
            'cast': str,
        },
        {
            'name': 'quality',
            'type': PortDataType.INT,
            'port': 'Input03',
            'widget': 'slider_int',
            'label': 'Quality',
            'min': 1,
            'max': 100,
            'default': 90,
            'cast': int,
        },
        {
            'name': 'subsampling',
            'type': PortDataType.TEXT,
            'port': 'Input04',
            'widget': 'combo',
            'label': 'Subsampling',
            'items': ['Auto', '4:4:4', '4:2:2', '4:2:0'],
            'default': 'Auto',
            'cast': str,
        },
        {
            'name': 'png_compression',
            'type': PortDataType.INT,
            'port': 'Input05',
            'widget': 'slider_int',
            'label': 'PNG Level',
            'min': 0,
            'max': 9,
            'default': 3,
            'cast': int,
        },
        {
            'name': 'generation',
            'type': PortDataType.INT,
            'port': 'Input06',
            'widget': 'slider_int',
            'label': 'Generations',
            'min': 1,
            'max': 10,
            'default': 1,
            'cast': int,
        },
        {
            'name': 'progressive',
            'type': PortDataType.TEXT,
            'port': 'Input07',
            'widget': 'checkbox',
            'label': 'Progressive JPEG',
            'default': False,
            'cast': bool,
        },
        {
            'name': 'optimize',
            'type': PortDataType.TEXT,
            'port': 'Input08',
            'widget': 'checkbox',
            'label': 'Optimize JPEG',
            'default': False,
            'cast': bool,
        },
    ]

    def _ensure_declarative_port_handles(self, node_id, include_elapsed=False):
        ports = super()._ensure_declarative_port_handles(
            node_id,
            include_elapsed=include_elapsed,
        )
        if not hasattr(ports, 'metadata'):
            self.create_port(
                node_id,
                'metadata',
                OutputPort(PortDataType.METADATA, index=3),
            )
        return self.ports(node_id)

    def build_custom_ui(self, tag_node_name, node_id, width, callback):
        del tag_node_name, width, callback
        ports = self.ports(node_id)
        with dpg.node_attribute(
            tag=ports.metadata.dpg_tag,
            attribute_type=dpg.mvNode_Attr_Output,
        ):
            dpg.add_text(
                tag=ports.metadata.value_tag,
                default_value='compression metadata',
            )

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        frame, result = super().update(
            node_id,
            connection_list,
            node_image_dict,
            node_result_dict,
        )
        compression_metadata = None
        if isinstance(result, dict):
            compression_metadata = result.get('compression_metadata')
        if compression_metadata is not None:
            dpg_set_value(
                self.ports(node_id).metadata.value_tag,
                summarize_metadata({
                    'format': compression_metadata.get('codec'),
                    'jpeg': compression_metadata.get('jpeg'),
                }),
            )
        return frame, result

    def process(self, frame, **parameter_values):
        output, compression_metadata = compression_roundtrip(
            frame,
            parameter_values.get('codec', 'JPEG'),
            quality=parameter_values.get('quality', 90),
            subsampling=parameter_values.get('subsampling', 'Auto'),
            progressive=parameter_values.get('progressive', False),
            optimize=parameter_values.get('optimize', False),
            png_compression=parameter_values.get('png_compression', 3),
            generation=parameter_values.get('generation', 1),
        )
        return output, {
            'compression_metadata': compression_metadata,
            'metadata': {
                'format': compression_metadata.get('codec'),
                'jpeg': compression_metadata.get('jpeg'),
                'compression': compression_metadata,
            },
        }
