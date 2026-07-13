#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from node.node_abc import DpgNodeBase
from node.port_model import (
    InputPort,
    OutputPort,
    PortDataType,
    PortSpecs,
    enum_value,
)
from node_editor.image_metadata import format_metadata_report, summarize_metadata
from node_editor.util import dpg_set_value


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Compression Inspect'
    node_tag = 'CompressionInspect'

    port_specs = PortSpecs(
        metadata_input=InputPort(PortDataType.METADATA, index=1),
        metadata=OutputPort(PortDataType.METADATA, index=1),
    )

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        del opencv_setting_dict
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        metadata_input_port = ports.metadata_input
        metadata_input = metadata_input_port.dpg_tag
        metadata_port = ports.metadata
        metadata = metadata_port.dpg_tag
        metadata_value = metadata_port.value_tag
        report_tag = self._report_value_tag(node_id)

        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            self.add_editor_toolbar(node_id, callback=callback)
            with dpg.node_attribute(
                tag=metadata_input,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text('metadata')
            with dpg.node_attribute(
                tag=self._report_attr_tag(node_id),
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_text(
                    tag=report_tag,
                    default_value='Connect image metadata.',
                    wrap=320,
                )
            with dpg.node_attribute(
                tag=metadata,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=metadata_value,
                    default_value='metadata',
                )

        return tag_node_name

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        del node_image_dict
        metadata = None
        for connection_info, source_tag, _, connection_type in self._iter_connection_infos(
            connection_list
        ):
            if enum_value(connection_type) != self.TYPE_METADATA:
                continue
            source_node_key = self._connection_source_node_key(
                connection_info,
                source_tag,
            )
            source_result = node_result_dict.get(source_node_key)
            if isinstance(source_result, dict):
                metadata = source_result.get('metadata') or source_result.get(
                    'compression_metadata'
                )
                if metadata is not None:
                    break

        if metadata is None:
            report = 'No metadata available.'
            summary = 'metadata'
        else:
            report = format_metadata_report(metadata)
            summary = summarize_metadata(metadata)

        dpg_set_value(self._report_value_tag(node_id), report)
        dpg_set_value(self.ports(node_id).metadata.value_tag, summary)
        return None, {'metadata': metadata}

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        return {
            'ver': self._ver,
            'pos': dpg.get_item_pos(self._node_name(node_id)),
        }

    def set_setting_dict(self, node_id, setting_dict):
        del node_id, setting_dict

    def _report_attr_tag(self, node_id):
        return f'{self._node_name(node_id)}:Report'

    def _report_value_tag(self, node_id):
        return f'{self._node_name(node_id)}:ReportValue'
