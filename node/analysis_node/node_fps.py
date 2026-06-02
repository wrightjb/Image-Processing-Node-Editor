#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from collections import deque

import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'FPS'
    node_tag = 'FPS'

    _opencv_setting_dict = None

    _buffer_len = 10
    _value_history = None

    _max_slot_number = 10
    _slot_id = {}

    port_specs = PortSpecs(
        elapsed_input=InputPort(PortDataType.TIME_MS),
        elapsed=OutputPort(PortDataType.TIME_MS, index=2),
    )

    def __init__(self):
        pass

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        self._value_history = {}

        # Tag names
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        tag_node_input00_name = self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Input00')
        tag_node_input01_name_port = ports.elapsed_input
        tag_node_input01_name = tag_node_input01_name_port.dpg_tag
        tag_node_input01_value_name = tag_node_input01_name_port.value_tag
        tag_node_output01_name = self._port_tag(tag_node_name, self.TYPE_TEXT, 'Output01')
        tag_node_output01_value_name = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Output01')
        )
        tag_node_output02_name_port = ports.elapsed
        tag_node_output02_name = tag_node_output02_name_port.dpg_tag
        tag_node_output02_value_name = tag_node_output02_name_port.value_tag

        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['result_width']

        # Dictionary to store slot numbers
        if tag_node_name not in self._slot_id:
            self._slot_id[tag_node_name] = 1

        # Node
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # FPS display
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_text(
                    tag=tag_node_output01_value_name,
                    default_value='FPS:',
                )
            # Total time display
            with dpg.node_attribute(
                    tag=tag_node_output02_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=tag_node_output02_value_name,
                    default_value='Total time(ms)',
                )
            # Add slot button
            with dpg.node_attribute(
                    tag=tag_node_input00_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label='Add Slot',
                    width=int(small_window_w / 3),
                    callback=self._add_slot,
                    user_data=tag_node_name,
                )
            # Slot
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='elapsed time(ms)',
                )

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        tag_node_name = self._node_name(node_id)
        output_value01_tag = self._value_tag(
            self._port_tag(tag_node_name, self.TYPE_TEXT, 'Output01')
        )
        output_value02_tag = self.ports(node_id).elapsed.value_tag

        total_elapsed_time = 0

        # Get source node name for image (with ID)
        for (
                connection_info,
                source_tag,
                destination_tag,
                connection_type,
        ) in self._iter_connection_infos(connection_list):
            if connection_type == self.TYPE_TIME_MS:
                # Get connection tag
                source_value_tag = self._connection_value_tag(connection_info, 'source', source_tag)
                destination_value_tag = self._connection_value_tag(connection_info, 'destination', destination_tag)

                # Update value
                input_value = dpg_get_value(source_value_tag)

                # Extract numeric value only
                input_value = re.sub(r'\D', '', input_value)
                if input_value != '':
                    input_value = int(input_value)

                    # Add elapsed time to queue
                    if source_value_tag not in self._value_history:
                        self._value_history[source_value_tag] = deque(
                            maxlen=self._buffer_len)
                        self._value_history[source_value_tag].append(input_value)
                    else:
                        self._value_history[source_value_tag].append(input_value)

                    # Average processing time
                    average_elapsed_time = sum(
                        self._value_history[source_value_tag]) / len(
                            self._value_history[source_value_tag])

                    # Calculate FPS
                    fps = 0
                    if average_elapsed_time > 0:
                        fps = 1000.0 / average_elapsed_time

                    # Generate display text
                    text = 'FPS:'
                    if fps > 1:
                        fps = int(fps)
                        text += '{:.0f}'.format(fps).zfill(3)
                    else:
                        text += '{:.2f}'.format(fps).zfill(3)
                    text += ' (' + '{:.0f}'.format(input_value).zfill(
                        4) + 'ms)'

                    # Update text
                    dpg_set_value(destination_value_tag, text)

                    # Total time of all slots
                    total_elapsed_time += average_elapsed_time

        # Calculate FPS from total time of all slots
        if total_elapsed_time > 0:
            fps = 1000.0 / total_elapsed_time
            text = 'FPS:'
            if fps > 1:
                fps = int(fps)
                text += '{:.0f}'.format(fps).zfill(3)
            else:
                text += '{:.2f}'.format(fps).zfill(3)
            # text += ' (' + '{:.0f}'.format(total_elapsed_time).zfill(4) + 'ms)'

            dpg_set_value(output_value01_tag, text)
            dpg_set_value(
                output_value02_tag,
                '{:.0f}'.format(total_elapsed_time).zfill(4) + 'ms',
            )

        return None, None

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict['slot_id'] = self._slot_id[tag_node_name]

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = self._node_name(node_id)

        slot_number = int(setting_dict['slot_id'])
        for _ in range(slot_number - 1):
            self._add_slot(None, None, tag_node_name)

    def _add_slot(self, sender, data, user_data):
        tag_node_name = user_data

        if self._max_slot_number > self._slot_id[tag_node_name]:
            self._slot_id[tag_node_name] += 1

            # Generate insertion destination tag name
            before_tag = self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Input')
            before_tag += str(self._slot_id[tag_node_name] - 1).zfill(2)

            # Generate added slot tag
            tag_node_inputXX_name = self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Input')
            tag_node_inputXX_name += str(self._slot_id[tag_node_name]).zfill(2)

            tag_node_inputXX_value_name = self._port_tag(tag_node_name, self.TYPE_TIME_MS, 'Input')
            tag_node_inputXX_value_name += str(
                self._slot_id[tag_node_name]).zfill(2) + 'Value'

            # Add slot
            with dpg.node_attribute(
                    tag=tag_node_inputXX_name,
                    attribute_type=dpg.mvNode_Attr_Input,
                    parent=tag_node_name,
                    before=before_tag,
            ):
                dpg.add_text(
                    tag=tag_node_inputXX_value_name,
                    default_value='elapsed time(ms)',
                )
