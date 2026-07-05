#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from auto_tune.gaussian_blur import tune_gaussian_blur
from node.node_abc import DpgNodeBase
from node.port_model import InputPort, OutputPort, PortDataType, PortSpecs
from node_editor.util import dpg_get_value, dpg_set_value


class Node(DpgNodeBase):
    _ver = '0.0.1'

    node_label = 'Auto Tune (Gaussian Blur)'
    node_tag = 'AutoTuneGaussianBlur'

    port_specs = PortSpecs(
        source_image=InputPort(PortDataType.IMAGE, index=1),
        target_image=InputPort(PortDataType.IMAGE, index=2),
        kernel_size=OutputPort(PortDataType.INT, index=1),
        sigma=OutputPort(PortDataType.FLOAT, index=2),
        best_score=OutputPort(PortDataType.FLOAT, index=3),
    )

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        del callback
        tag_node_name = self._node_name(node_id)
        ports = self.create_ports(node_id)
        source_image_port = ports.source_image
        target_image_port = ports.target_image
        kernel_size_port = ports.kernel_size
        sigma_port = ports.sigma
        best_score_port = ports.best_score
        source_image = source_image_port.dpg_tag
        target_image = target_image_port.dpg_tag
        kernel_size = kernel_size_port.dpg_tag
        sigma = sigma_port.dpg_tag
        best_score = best_score_port.dpg_tag
        self._opencv_setting_dict = opencv_setting_dict

        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            with dpg.node_attribute(
                tag=source_image,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text('source image')
            with dpg.node_attribute(
                tag=target_image,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text('target image')
            with dpg.node_attribute(
                tag=kernel_size,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_int(
                    tag=kernel_size_port.value_tag,
                    label='kernel',
                    default_value=5,
                    width=120,
                    readonly=True,
                )
            with dpg.node_attribute(
                tag=sigma,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_float(
                    tag=sigma_port.value_tag,
                    label='sigma',
                    default_value=0.0,
                    width=120,
                    readonly=True,
                )
            with dpg.node_attribute(
                tag=best_score,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_float(
                    tag=best_score_port.value_tag,
                    label='score',
                    default_value=0.0,
                    width=120,
                    readonly=True,
                )

        return tag_node_name

    def _linked_image(self, port_ref, connection_list, node_image_dict):
        for connection_info in self._iter_connection_infos(connection_list):
            if connection_info is None:
                continue
            source_tag, destination_tag = connection_info.legacy_pair
            if destination_tag != port_ref.dpg_tag:
                continue
            source_node_key = self._connection_source_node_key(
                connection_info,
                source_tag,
            )
            return node_image_dict.get(source_node_key)
        return None

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        del node_result_dict
        ports = self.ports(node_id)
        source = self._linked_image(ports.source_image, connection_list, node_image_dict)
        target = self._linked_image(ports.target_image, connection_list, node_image_dict)
        if source is None or target is None:
            return None, None

        result = tune_gaussian_blur(
            source,
            target,
            current_parameters={'auto_sigma': True},
        )
        dpg_set_value(ports.kernel_size.value_tag, int(result.best_parameters['kernel_size']))
        dpg_set_value(ports.sigma.value_tag, float(result.best_parameters['sigma']))
        dpg_set_value(ports.best_score.value_tag, float(result.best_score))
        return result.best_image, result

    def close(self, node_id):
        del node_id

    def get_setting_dict(self, node_id):
        tag_node_name = self._node_name(node_id)
        ports = self.ports(node_id)
        setting_dict = {
            'ver': self._ver,
            'pos': dpg.get_item_pos(tag_node_name),
            ports.kernel_size.value_tag: dpg_get_value(ports.kernel_size.value_tag),
            ports.sigma.value_tag: dpg_get_value(ports.sigma.value_tag),
            ports.best_score.value_tag: dpg_get_value(ports.best_score.value_tag),
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        ports = self.ports(node_id)
        for value_tag in (
            ports.kernel_size.value_tag,
            ports.sigma.value_tag,
            ports.best_score.value_tag,
        ):
            if value_tag in setting_dict:
                dpg_set_value(value_tag, setting_dict[value_tag])
