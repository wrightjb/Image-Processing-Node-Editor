#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import json
import platform
import datetime
import re
from glob import glob
from collections import OrderedDict
from importlib import import_module
from pathlib import Path
from dataclasses import dataclass

import dearpygui.dearpygui as dpg


@dataclass(frozen=True)
class NodeRef:
    node_id: str
    node_tag: str

    @property
    def node_id_name(self):
        return f'{self.node_id}:{self.node_tag}'


@dataclass(frozen=True)
class PortRef:
    node_ref: NodeRef
    direction: str
    data_type: str
    index: int
    dpg_tag: str


from node_editor.history import (
    AddNodeCommand,
    DeleteNodesCommand,
    MoveNodeCommand,
    AddLinkCommand,
    RemoveLinkCommand,
    ReplaceLinkCommand,
    CompositeCommand,
    SetParameterCommand,
    history_command_label,
)


class DpgNodeEditor(object):
    _ver = '0.0.1'

    _node_editor_tag = 'NodeEditor'
    _node_editor_label = 'Node editor'
    _window_tag = _node_editor_tag + 'Window'
    _link_feedback_tag = _node_editor_tag + 'LinkFeedback'
    _insert_link_popup_tag = _node_editor_tag + 'InsertLinkPopup'
    _insert_link_popup_anchor_tag = _insert_link_popup_tag + 'Anchor'
    _add_node_popup_tag = _node_editor_tag + 'AddNodePopup'
    _add_node_popup_anchor_tag = _add_node_popup_tag + 'Anchor'
    _history_window_tag = _node_editor_tag + 'HistoryWindow'
    _history_undo_text_tag = _node_editor_tag + 'HistoryUndoText'
    _history_redo_text_tag = _node_editor_tag + 'HistoryRedoText'
    _node_close_attr_suffix = ':CloseAttr'
    _node_close_button_suffix = ':CloseButton'

    _node_id = 0
    _node_instance_list = {}
    _node_list = []
    _node_link_list = []
    _link_view_id_map = {}

    _last_pos = None

    _terminate_flag = False

    _opencv_setting_dict = None

    _use_debug_print = False

    def __init__(
        self,
        width=1280,
        height=720,
        pos=[0, 0],
        opencv_setting_dict=None,
        node_dir='node',
        menu_dict=None,
        use_debug_print=False,
    ):
        self._mdl_init(opencv_setting_dict, use_debug_print)
        self._cntrl_init(node_dir, menu_dict)
        self._vw_init(width, height, pos)

    # -------------------------------------------------------------------------
    # Model functions
    def _mdl_init(self, opencv_setting_dict=None, use_debug_print=False):
        self._node_id = 0
        self._node_instance_list = {}
        self._node_list = []
        self._node_link_list = []
        self._link_view_id_map = {}
        self._node_connection_dict = OrderedDict([])
        self._pending_insert_link_dpg_id = None
        self._pending_add_from_output_tag = None
        self._pending_add_to_input_tag = None
        self._node_port_capabilities = {}
        self._node_registry = {}
        self._port_registry = {}
        self._link_registry = {}
        self._link_by_dest_port = {}
        self._use_debug_print = use_debug_print
        self._terminate_flag = False
        self._opencv_setting_dict = opencv_setting_dict
        self._undo_stack = []
        self._redo_stack = []
        self._move_start_positions = {}
        self._node_position_cache = {}
        self._parameter_last_values = {}
        self._parameter_last_coalesce_hint = {}
        self._parameter_drag_stream_active = set()
        self._suspend_parameter_history = False
        self._history_node_id_remap = {}

    def _mdl_add_node(self, node_tag):
        self._node_id += 1
        new_node_id_name = f"{self._node_id}:{node_tag}"
        return self._node_id, new_node_id_name

    def _mdl_register_node_ref(self, node_ref):
        self._node_registry[node_ref.node_id_name] = node_ref

    def _mdl_get_node_ref(self, node_id_name):
        return self._node_registry.get(node_id_name)

    def _cntrl_parse_port_tag(self, port_tag):
        if not isinstance(port_tag, str):
            return None
        if port_tag in self._port_registry:
            return self._port_registry[port_tag]
        parts = port_tag.split(':')
        if len(parts) < 4:
            return None
        node_ref = NodeRef(parts[0], parts[1])
        port_name = parts[3]
        if port_name.startswith('Input'):
            direction = 'Input'
            index_text = port_name[len('Input'):]
        elif port_name.startswith('Output'):
            direction = 'Output'
            index_text = port_name[len('Output'):]
        else:
            return None
        try:
            index = int(index_text)
        except ValueError:
            return None
        parsed = PortRef(node_ref, direction, parts[2], index, port_tag)
        self._port_registry[port_tag] = parsed
        if node_ref.node_id_name not in self._node_registry:
            self._mdl_register_node_ref(node_ref)
        return parsed

    def _mdl_add_link(self, source_tag, dest_tag):
        source_port = self._cntrl_parse_port_tag(source_tag)
        dest_port = self._cntrl_parse_port_tag(dest_tag)
        if source_port is None or dest_port is None:
            return False
        if dest_tag in self._link_by_dest_port:
            return False
        self._node_link_list.append([source_tag, dest_tag])
        self._link_registry[(source_tag, dest_tag)] = (source_port, dest_port)
        self._link_by_dest_port[dest_tag] = source_tag
        return True

    def _mdl_get_link_by_destination(self, dest_tag):
        source_tag = self._link_by_dest_port.get(dest_tag)
        if source_tag is not None:
            return [source_tag, dest_tag]
        return None

    def _mdl_get_export_settings(self):
        setting_dict = {}
        setting_dict['node_list'] = self._node_list
        setting_dict['link_list'] = self._node_link_list
        for node_id_name in self._node_list:
            node_id, node_name = node_id_name.split(':')
            node = self._node_instance_list[node_name]
            setting = node.get_setting_dict(node_id)
            setting_dict[node_id_name] = {
                'id': str(node_id),
                'name': str(node_name),
                'setting': setting
            }
        return setting_dict

    def _mdl_delete_node(self, node_id_name):
        node_id, node_name = node_id_name.split(':')
        if node_name == 'ExecPythonCode':
            return
        node_instance = self.get_node_instance(node_name)
        node_instance.close(node_id)
        self._node_list.remove(node_id_name)

        copy_node_link_list = copy.deepcopy(self._node_link_list)
        for link_info in copy_node_link_list:
            source_port = self._port_registry.get(link_info[0])
            dest_port = self._port_registry.get(link_info[1])
            source_node = source_port.node_ref.node_id_name if source_port else ''
            destination_node = dest_port.node_ref.node_id_name if dest_port else ''
            if not source_node and isinstance(link_info[0], str):
                source_parts = link_info[0].split(':')
                if len(source_parts) >= 2:
                    source_node = f'{source_parts[0]}:{source_parts[1]}'
            if not destination_node and isinstance(link_info[1], str):
                dest_parts = link_info[1].split(':')
                if len(dest_parts) >= 2:
                    destination_node = f'{dest_parts[0]}:{dest_parts[1]}'
            if source_node == node_id_name or destination_node == node_id_name:
                self._link_registry.pop((link_info[0], link_info[1]), None)
                self._link_by_dest_port.pop(link_info[1], None)
                self._node_link_list.remove(link_info)
        self._node_registry.pop(node_id_name, None)
        self._mdl_sort_node_graph()

    def _mdl_delete_link(self, link):
        source_tag, dest_tag = link
        self._link_registry.pop((source_tag, dest_tag), None)
        self._link_by_dest_port.pop(dest_tag, None)
        self._node_link_list.remove(link)
        self._mdl_sort_node_graph()

    def _mdl_sort_node_graph(self):
        node_list = self._node_list
        node_link_list = self._node_link_list

        node_id_dict = OrderedDict({})
        node_connection_dict = OrderedDict({})

        # Organize node IDs and node links as dictionaries
        for source, destination in node_link_list:
            source_id = int(source.split(':')[0])
            destination_id = int(destination.split(':')[0])

            if destination_id not in node_id_dict:
                node_id_dict[destination_id] = [source_id]
            else:
                node_id_dict[destination_id].append(source_id)

            split_destination = destination.split(':')
            node_name = split_destination[0] + ':' + split_destination[1]
            if node_name not in node_connection_dict:
                node_connection_dict[node_name] = [[source, destination]]
            else:
                node_connection_dict[node_name].append([source, destination])

        node_id_list = list(node_id_dict.items())
        node_connection_list = list(node_connection_dict.items())

        # Reorder processing from input to output
        index = 0
        guard = 0
        max_guard = max(1, len(node_id_list) * len(node_id_list) * 4)
        while index < len(node_id_list):
            guard += 1
            if guard > max_guard:
                # Cyclic graphs can cause endless swapping here. Keep current
                # ordering as a safe fallback instead of hanging the app.
                break
            swap_flag = False
            for check_id in node_id_list[index][1]:
                for check_index in range(index + 1, len(node_id_list)):
                    if node_id_list[check_index][0] == check_id:
                        node_id_list[check_index], node_id_list[
                            index] = node_id_list[index], node_id_list[
                                check_index]
                        node_connection_list[
                            check_index], node_connection_list[
                                index] = node_connection_list[
                                    index], node_connection_list[check_index]

                        swap_flag = True
                        break
            if not swap_flag:
                index += 1

        # Add nodes not in the link list (e.g., input nodes)
        index = 0
        unfinded_id_dict = {}
        while index < len(node_id_list):
            for check_id in node_id_list[index][1]:
                check_index = 0
                find_flag = False
                while check_index < len(node_id_list):
                    if check_id == node_id_list[check_index][0]:
                        find_flag = True
                        break
                    check_index += 1
                if not find_flag:
                    for _, node_id_name in enumerate(node_list):
                        node_id, _ = node_id_name.split(':')
                        if int(node_id) == check_id:
                            unfinded_id_dict[check_id] = node_id_name
                            break
            index += 1

        for unfinded_value in unfinded_id_dict.values():
            node_connection_list.insert(0, (unfinded_value, []))

        self._node_connection_dict = OrderedDict(node_connection_list)

    # -------------------------------------------------------------------------
    # View functions
    def _vw_init(self, width, height, pos):
        self._vw_create_file_dialogs(height)
        self._vw_create_main_window(width, height, pos)

    def _vw_create_file_dialogs(self, height):
        datetime_now = datetime.datetime.now()
        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(height / 2),
                default_filename=datetime_now.strftime('%Y%m%d'),
                callback=self._cntrl_file_export,
                id='file_export',
        ):
            dpg.add_file_extension('.json')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(height / 2),
                callback=self._cntrl_file_import,
                id='file_import',
        ):
            dpg.add_file_extension('.json')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

    def _vw_create_main_window(self, width, height, pos):
        with dpg.window(
                tag=self._window_tag,
                label=self._node_editor_label,
                width=width,
                height=height,
                pos=pos,
                menubar=True,
                on_close=self._cntrl_close_window,
        ):
            with dpg.menu_bar(label='MenuBar'):
                with dpg.menu(label='File'):
                    dpg.add_menu_item(
                        tag='Menu_File_Export',
                        label='Export',
                        callback=self._cntrl_file_export_menu,
                        user_data='Menu_File_Export',
                    )
                    dpg.add_menu_item(
                        tag='Menu_File_Import',
                        label='Import',
                        callback=self._cntrl_file_import_menu,
                        user_data='Menu_File_Import',
                    )
                self._vw_create_node_menus()
                self._vw_create_insert_link_menu()
            dpg.add_text(
                default_value='',
                tag=self._link_feedback_tag,
                color=(255, 180, 80, 255),
                wrap=0,
            )
            with dpg.node_editor(
                    tag=self._node_editor_tag,
                    callback=self._cntrl_link,
                    minimap=True,
                    minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            ):
                pass
            dpg.add_button(
                tag=self._insert_link_popup_anchor_tag,
                label='',
                width=1,
                height=1,
                show=False,
            )
            with dpg.popup(
                    self._insert_link_popup_anchor_tag,
                    mousebutton=dpg.mvMouseButton_Right,
                    tag=self._insert_link_popup_tag,
            ):
                self._vw_create_insert_link_popup_menu()
            dpg.add_button(
                tag=self._add_node_popup_anchor_tag,
                label='',
                width=1,
                height=1,
                show=False,
            )
            with dpg.popup(
                    self._add_node_popup_anchor_tag,
                    mousebutton=dpg.mvMouseButton_Right,
                    tag=self._add_node_popup_tag,
            ):
                self._vw_create_add_node_popup_menu()
        self._vw_create_history_window()
        dpg.set_primary_window(self._window_tag, True)

    def _vw_create_node_menus(self):
        for menu_label, nodes in self._menu_nodes.items():
            with dpg.menu(label=menu_label):
                for node_info in nodes:
                    dpg.add_menu_item(
                        tag='Menu_' + node_info['tag'],
                        label=node_info['label'],
                        callback=self._cntrl_add_node,
                        user_data=node_info['tag'],
                    )

    def _vw_create_insert_link_menu(self):
        with dpg.menu(label='Edit'):
            dpg.add_menu_item(
                tag='Menu_Edit_Undo',
                label='Undo',
                callback=self._cntrl_undo,
                enabled=False,
            )
            dpg.add_menu_item(
                tag='Menu_Edit_Redo',
                label='Redo',
                callback=self._cntrl_redo,
                enabled=False,
            )
            dpg.add_separator()
            dpg.add_menu_item(
                tag='Menu_Edit_History',
                label='History...',
                callback=self._cntrl_toggle_history_window,
            )
            dpg.add_separator()
            with dpg.menu(label='Insert'):
                for menu_label, nodes in self._menu_nodes.items():
                    with dpg.menu(label=menu_label):
                        for node_info in nodes:
                            dpg.add_menu_item(
                                tag='Menu_InsertLink_' + node_info['tag'],
                                label=node_info['label'],
                                callback=self._cntrl_insert_node_into_selected_link,
                                user_data=node_info['tag'],
                            )

    def _vw_create_insert_link_popup_menu(self):
        dpg.add_text('Insert Node')
        dpg.add_separator()

    def _vw_create_add_node_popup_menu(self):
        dpg.add_text('Add Node')
        dpg.add_separator()

    def _vw_create_history_window(self):
        with dpg.window(
            tag=self._history_window_tag,
            label='Undo/Redo History',
            width=460,
            height=360,
            pos=[40, 40],
            show=False,
        ):
            dpg.add_text('Undo stack (top first):')
            dpg.add_text(default_value='(empty)', tag=self._history_undo_text_tag)
            dpg.add_separator()
            dpg.add_text('Redo stack (top first):')
            dpg.add_text(default_value='(empty)', tag=self._history_redo_text_tag)

    def _vw_clear_popup_menu_items(self, popup_tag):
        popup_children = dpg.get_item_children(popup_tag, 1)
        if not popup_children:
            return
        for child in popup_children[2:]:
            dpg.delete_item(child)

    def _cntrl_extract_node_port_capabilities(self, node, node_source_path):
        capabilities = {'input_types': set(), 'output_types': set()}

        # Declarative nodes expose parameter metadata directly.
        if hasattr(node, 'parameters') and isinstance(node.parameters, list):
            capabilities['input_types'].add('Image')
            capabilities['output_types'].add('Image')
            if getattr(node, 'show_elapsed_time', True):
                capabilities['output_types'].add('TimeMS')
            for parameter in node.parameters:
                if not isinstance(parameter, dict):
                    continue
                port_name = str(parameter.get('port', ''))
                port_type = str(parameter.get('type', ''))
                if not port_name.startswith('Input'):
                    continue
                if port_type in ('Int', 'Float', 'Image', 'Text', 'TimeMS'):
                    capabilities['input_types'].add(port_type)

        try:
            source_text = Path(node_source_path).read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return capabilities

        pattern = re.compile(
            r"_port_tag\(\s*tag_node_name\s*,\s*self\.(TYPE_[A-Z_]+)\s*,\s*'((?:Input|Output)\d{2})'"
        )
        type_map = {
            'TYPE_INT': 'Int',
            'TYPE_FLOAT': 'Float',
            'TYPE_IMAGE': 'Image',
            'TYPE_TIME_MS': 'TimeMS',
            'TYPE_TEXT': 'Text',
        }
        for type_token, port_name in pattern.findall(source_text):
            mapped = type_map.get(type_token)
            if mapped is None:
                continue
            if port_name.startswith('Input'):
                capabilities['input_types'].add(mapped)
            else:
                capabilities['output_types'].add(mapped)
        return capabilities

    def _cntrl_is_node_append_compatible(self, node_tag, source_type):
        caps = self._node_port_capabilities.get(node_tag, {})
        return source_type in caps.get('input_types', set())

    def _cntrl_is_node_insert_compatible(self, node_tag, link_type):
        caps = self._node_port_capabilities.get(node_tag, {})
        input_types = caps.get('input_types', set())
        output_types = caps.get('output_types', set())
        return link_type in input_types and link_type in output_types

    def _vw_populate_append_popup_menu(self, source_type):
        self._vw_clear_popup_menu_items(self._add_node_popup_tag)
        added = False
        for menu_label, nodes in self._menu_nodes.items():
            compatible_nodes = [
                node_info for node_info in nodes
                if self._cntrl_is_node_append_compatible(node_info['tag'], source_type)
            ]
            if not compatible_nodes:
                continue
            added = True
            with dpg.menu(label=menu_label, parent=self._add_node_popup_tag):
                for node_info in compatible_nodes:
                    dpg.add_menu_item(
                        label=node_info['label'],
                        callback=self._cntrl_add_node_from_output_port,
                        user_data=node_info['tag'],
                    )
        if not added:
            dpg.add_text('No compatible nodes', parent=self._add_node_popup_tag)

    def _vw_populate_prepend_popup_menu(self, dest_type):
        self._vw_clear_popup_menu_items(self._add_node_popup_tag)
        added = False
        for menu_label, nodes in self._menu_nodes.items():
            compatible_nodes = [
                node_info for node_info in nodes
                if dest_type in self._node_port_capabilities.get(
                    node_info['tag'], {}
                ).get('output_types', set())
            ]
            if not compatible_nodes:
                continue
            added = True
            with dpg.menu(label=menu_label, parent=self._add_node_popup_tag):
                for node_info in compatible_nodes:
                    dpg.add_menu_item(
                        label=node_info['label'],
                        callback=self._cntrl_add_node_to_input_port,
                        user_data=node_info['tag'],
                    )
        if not added:
            dpg.add_text('No compatible nodes', parent=self._add_node_popup_tag)

    def _vw_populate_insert_popup_menu(self, link_type):
        self._vw_clear_popup_menu_items(self._insert_link_popup_tag)
        added = False
        for menu_label, nodes in self._menu_nodes.items():
            compatible_nodes = [
                node_info for node_info in nodes
                if self._cntrl_is_node_insert_compatible(node_info['tag'], link_type)
            ]
            if not compatible_nodes:
                continue
            added = True
            with dpg.menu(label=menu_label, parent=self._insert_link_popup_tag):
                for node_info in compatible_nodes:
                    dpg.add_menu_item(
                        label=node_info['label'],
                        callback=self._cntrl_insert_node_into_selected_link,
                        user_data=node_info['tag'],
                    )
        if not added:
            dpg.add_text('No compatible nodes', parent=self._insert_link_popup_tag)

    def _vw_add_node(self, node_tag, new_id, pos):
        node = self._node_instance_list[node_tag]
        node_view_tag = node.add_node(
            self._node_editor_tag,
            new_id,
            pos=pos,
            opencv_setting_dict=self._opencv_setting_dict,
            callback=self._cntrl_node_callback,
        )

        if node_tag == 'ExecPythonCode':
            return

        close_attr_tag = node_view_tag + self._node_close_attr_suffix
        close_button_tag = node_view_tag + self._node_close_button_suffix
        if dpg.does_item_exist(close_attr_tag):
            dpg.delete_item(close_attr_tag)

        with dpg.node_attribute(
                parent=node_view_tag,
                tag=close_attr_tag,
                attribute_type=dpg.mvNode_Attr_Static,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text(' ')
                dpg.add_spacer(width=180)
                dpg.add_button(
                    tag=close_button_tag,
                    label='x',
                    width=20,
                    height=20,
                    callback=self._cntrl_delete_node_by_button,
                    user_data=node_view_tag,
                )
        node_children = dpg.get_item_children(node_view_tag, 1)
        if node_children:
            first_attribute = node_children[0]
            if first_attribute != close_attr_tag:
                dpg.move_item(close_attr_tag, parent=node_view_tag, before=first_attribute)

    def _vw_add_link(self, source, destination):
        source_id = source
        destination_id = destination

        if isinstance(source, str) and not dpg.does_item_exist(source):
            raise ValueError(f'Invalid source port tag: {source}')
        if isinstance(destination, str) and not dpg.does_item_exist(destination):
            raise ValueError(f'Invalid destination port tag: {destination}')

        return dpg.add_node_link(
            source_id,
            destination_id,
            parent=self._node_editor_tag,
        )

    def _vw_register_link(self, source_tag, dest_tag, link_dpg_id):
        self._link_view_id_map[(source_tag, dest_tag)] = link_dpg_id

    def _vw_delete_link(self, source_tag, dest_tag):
        link_dpg_id = self._link_view_id_map.pop((source_tag, dest_tag), None)
        if link_dpg_id is not None:
            self._vw_delete_item(link_dpg_id)

    def _vw_delete_links_for_node(self, node_id_name):
        delete_targets = []
        for source_tag, dest_tag in self._link_view_id_map:
            source_port = self._port_registry.get(source_tag)
            dest_port = self._port_registry.get(dest_tag)
            source_node = source_port.node_ref.node_id_name if source_port else ''
            destination_node = dest_port.node_ref.node_id_name if dest_port else ''
            if not source_node and isinstance(source_tag, str):
                source_parts = source_tag.split(':')
                if len(source_parts) >= 2:
                    source_node = f'{source_parts[0]}:{source_parts[1]}'
            if not destination_node and isinstance(dest_tag, str):
                dest_parts = dest_tag.split(':')
                if len(dest_parts) >= 2:
                    destination_node = f'{dest_parts[0]}:{dest_parts[1]}'
            if source_node == node_id_name or destination_node == node_id_name:
                delete_targets.append((source_tag, dest_tag))

        for source_tag, dest_tag in delete_targets:
            self._vw_delete_link(source_tag, dest_tag)

    def _vw_delete_item(self, item_id):
        dpg.delete_item(item_id)

    def _vw_set_node_pos(self, node_id_name, pos):
        if dpg.does_item_exist(node_id_name):
            dpg.set_item_pos(node_id_name, pos)

    def _cntrl_update_node_position_cache(self, node_id_name):
        if dpg.does_item_exist(node_id_name):
            self._node_position_cache[node_id_name] = list(dpg.get_item_pos(node_id_name))

    def _vw_show_file_export(self):
        dpg.show_item('file_export')

    def _vw_show_file_import(self):
        dpg.show_item('file_import')

    def _vw_set_link_feedback(self, message):
        dpg.set_value(self._link_feedback_tag, message)
        window_label = self._node_editor_label
        if message:
            window_label = f'{self._node_editor_label} | {message}'
        dpg.configure_item(self._window_tag, label=window_label)

    def _vw_show_insert_link_popup(self, pos):
        dpg.set_item_pos(self._insert_link_popup_tag, pos)
        dpg.show_item(self._insert_link_popup_tag)

    def _vw_hide_insert_link_popup(self):
        if dpg.is_item_shown(self._insert_link_popup_tag):
            dpg.hide_item(self._insert_link_popup_tag)

    def _vw_show_add_node_popup(self, pos):
        dpg.set_item_pos(self._add_node_popup_tag, pos)
        dpg.show_item(self._add_node_popup_tag)

    def _vw_hide_add_node_popup(self):
        if dpg.is_item_shown(self._add_node_popup_tag):
            dpg.hide_item(self._add_node_popup_tag)

    # -------------------------------------------------------------------------
    # Controller functions
    def _cntrl_init(self, node_dir, menu_dict):
        self._cntrl_discover_nodes(node_dir, menu_dict)
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self._cntrl_save_last_pos)
            dpg.add_mouse_down_handler(callback=self._cntrl_capture_move_start_positions)
            dpg.add_mouse_release_handler(callback=self._cntrl_commit_move_commands)
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right,
                callback=self._cntrl_open_insert_link_popup,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Delete,
                callback=self._cntrl_delete_selected,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Escape,
                callback=self._cntrl_close_insert_link_popup_on_escape,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Z,
                callback=self._cntrl_undo,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Y,
                callback=self._cntrl_redo,
            )

    def _cntrl_discover_nodes(self, node_dir, menu_dict):
        # Define menu items (key: menu name, value: directory containing node code)
        if menu_dict is None:
            menu_dict = OrderedDict({
                'Input Node': 'input_node',
                'Process Node': 'process_node',
                'Output Node': 'output_node'
            })

        self._menu_nodes = OrderedDict()
        for menu_label, sub_dir in menu_dict.items():
            self._menu_nodes[menu_label] = []
            node_sources_path = os.path.join(node_dir, sub_dir, '*.py')
            node_sources = glob(node_sources_path)
            for node_source in node_sources:
                import_path = os.path.splitext(
                    os.path.normpath(node_source))[0]
                if platform.system() == 'Windows':
                    import_path = import_path.replace('\\', '.')
                else:
                    import_path = import_path.replace('/', '.')
                import_path = '.'.join(import_path.split('.')[-3:])
                if import_path.endswith('__init__'):
                    continue
                module = import_module(import_path)
                node = module.Node()
                self._node_instance_list[node.node_tag] = node
                self._menu_nodes[menu_label].append({
                    'tag': node.node_tag,
                    'label': node.node_label,
                    'source_path': node_source,
                })
                self._node_port_capabilities[node.node_tag] = \
                    self._cntrl_extract_node_port_capabilities(node, node_source)

    def _cntrl_add_node(self, sender, data, user_data):
        new_id, new_node_id_name = self._mdl_add_node(user_data)
        self._mdl_register_node_ref(NodeRef(str(new_id), user_data))
        pos = [0, 0]
        if self._last_pos is not None:
            pos = [self._last_pos[0] + 30, self._last_pos[1] + 30]
        self._vw_add_node(user_data, new_id, pos)
        # Must add to list AFTER fully init because update's always async
        self._node_list.append(new_node_id_name)
        self._cntrl_update_node_position_cache(new_node_id_name)
        node_setting = self.get_node_instance(user_data).get_setting_dict(str(new_id))
        self._cntrl_push_undo_command(
            AddNodeCommand(
                new_id,
                user_data,
                list(pos),
                copy.deepcopy(node_setting),
                [],
                [],
            )
        )

        if self._use_debug_print:
            print('**** _cntrl_add_node ****')
            print(f'\tNode ID         : {self._node_id}')
            print(f'\tsender          : {sender}')
            print(f'\tdata            : {data}')
            print(f'\tuser_data       : {user_data}')
            print(f'\tself._node_list : {", ".join(self._node_list)}')
            print()

    def _cntrl_add_node_with_id(self, node_tag, node_id, pos, setting):
        self._node_id = max(self._node_id, int(node_id))
        node_id_name = f'{node_id}:{node_tag}'
        if node_id_name in self._node_list:
            return node_id_name
        self._mdl_register_node_ref(NodeRef(str(node_id), node_tag))
        self._vw_add_node(node_tag, int(node_id), pos)
        self._node_list.append(node_id_name)
        self._cntrl_update_node_position_cache(node_id_name)
        node = self.get_node_instance(node_tag)
        try:
            node.set_setting_dict(str(node_id), copy.deepcopy(setting))
        except Exception:
            pass
        return node_id_name

    def _cntrl_add_node_from_history(self, node_tag, requested_node_id, pos, setting):
        requested_node_id = int(requested_node_id)
        requested_id_name = f'{requested_node_id}:{node_tag}'
        try:
            created = self._cntrl_add_node_with_id(
                node_tag, requested_node_id, pos, setting
            )
            self._history_node_id_remap[requested_id_name] = created
            return created
        except Exception:
            new_id, _ = self._mdl_add_node(node_tag)
            created = self._cntrl_add_node_with_id(node_tag, new_id, pos, setting)
            self._history_node_id_remap[requested_id_name] = created
            return created

    def _cntrl_resolve_history_node_id_name(self, node_id_name):
        resolved = self._history_node_id_remap.get(node_id_name, node_id_name)
        return resolved

    def _cntrl_resolve_history_port_tag(self, port_tag):
        if not isinstance(port_tag, str):
            return port_tag
        parts = port_tag.split(':')
        if len(parts) < 4:
            return port_tag
        node_id_name = f'{parts[0]}:{parts[1]}'
        resolved_node_id_name = self._cntrl_resolve_history_node_id_name(node_id_name)
        if ':' not in resolved_node_id_name:
            return port_tag
        resolved_id, resolved_tag = resolved_node_id_name.split(':', 1)
        parts[0] = resolved_id
        parts[1] = resolved_tag
        return ':'.join(parts)

    def _cntrl_node_callback(self, event_name, data):
        if event_name == 'toggle_result_node':
            if not isinstance(data, dict):
                return
            self._cntrl_toggle_result_node(
                str(data.get('source_node_id_name', '')),
                str(data.get('result_node_tag', '')),
                bool(data.get('enabled', False)),
            )
            return
        if event_name == 'parameter_changed':
            if not isinstance(data, dict):
                return
            self._cntrl_record_parameter_change(
                str(data.get('node_id_name', '')),
                str(data.get('value_tag', '')),
                data.get('before_value'),
                data.get('after_value'),
                data.get('coalesce', None),
            )
            return
        if event_name == 'parameter_batch_changed':
            if not isinstance(data, dict):
                return
            node_id_name = str(data.get('node_id_name', ''))
            changes = data.get('changes', [])
            if not isinstance(changes, list) or not changes:
                return
            commands = []
            for change in changes:
                if not isinstance(change, dict):
                    continue
                value_tag = str(change.get('value_tag', ''))
                before_value = change.get('before_value')
                after_value = change.get('after_value')
                if not value_tag or before_value == after_value:
                    continue
                commands.append(
                    SetParameterCommand(
                        node_id_name=node_id_name,
                        value_tag=value_tag,
                        before_value=before_value,
                        after_value=after_value,
                    )
                )
            if commands:
                if len(commands) == 1:
                    self._cntrl_push_undo_command(commands[0])
                else:
                    self._cntrl_push_undo_command(CompositeCommand(commands))
            return
        if isinstance(event_name, str) and ':' in event_name and event_name.endswith('Value'):
            value_tag = event_name
            before_value = self._parameter_last_values.get(value_tag, dpg.get_value(value_tag))
            after_value = data
            self._parameter_last_values[value_tag] = after_value
            node_id_name = ':'.join(value_tag.split(':')[:2])
            self._cntrl_record_parameter_change(
                node_id_name,
                value_tag,
                before_value,
                after_value,
                None,
            )

    def _cntrl_set_parameter_value(self, value_tag, value):
        if dpg.does_item_exist(value_tag):
            dpg.set_value(value_tag, value)
            self._parameter_last_values[value_tag] = value
            self._cntrl_apply_parameter_side_effects(value_tag, value)
            return

        if not isinstance(value_tag, str) or ':' not in value_tag:
            return
        node_parts = value_tag.split(':')
        if len(node_parts) < 2:
            return
        node_tag = node_parts[1]
        node = self._node_instance_list.get(node_tag)
        if node is None:
            return
        if hasattr(node, 'apply_history_value') and callable(node.apply_history_value):
            applied = node.apply_history_value(value_tag, value)
            if applied:
                self._parameter_last_values[value_tag] = value

    def _cntrl_apply_parameter_side_effects(self, value_tag, value):
        if not isinstance(value_tag, str) or not value_tag.endswith('Value'):
            return
        parts = value_tag[:-5].split(':')
        if len(parts) < 4:
            return
        node_id, node_tag, _, port_name = parts[0], parts[1], parts[2], parts[3]
        node = self._node_instance_list.get(node_tag)
        if node is None:
            return
        if port_name == 'Cache' and hasattr(node, '_on_cache_toggle'):
            node._on_cache_toggle(value_tag, bool(value), node_id)
        elif port_name == 'ResultImage' and hasattr(node, '_on_result_image_toggle'):
            node._on_result_image_toggle(value_tag, bool(value), node_id)
        elif (
            port_name == 'ResultImageLarge'
            and hasattr(node, '_on_result_large_image_toggle')
        ):
            node._on_result_large_image_toggle(value_tag, bool(value), node_id)

    def _cntrl_record_parameter_change(
        self,
        node_id_name,
        value_tag,
        before_value,
        after_value,
        coalesce_hint=None,
    ):
        if self._suspend_parameter_history:
            return
        if not isinstance(value_tag, str) or ':' not in value_tag:
            return
        if before_value == after_value:
            return
        is_numeric_edit = (
            isinstance(before_value, (int, float))
            and isinstance(after_value, (int, float))
            and not isinstance(before_value, bool)
            and not isinstance(after_value, bool)
        )
        is_curves_drag_edit = (
            bool(coalesce_hint)
            and isinstance(before_value, list)
            and isinstance(after_value, list)
        )
        if is_curves_drag_edit and value_tag.endswith(':CurvesPointsValue'):
            if value_tag in self._parameter_drag_stream_active:
                for idx in range(len(self._undo_stack) - 1, -1, -1):
                    cmd = self._undo_stack[idx]
                    if not isinstance(cmd, SetParameterCommand):
                        break
                    if cmd.value_tag != value_tag:
                        continue
                    self._undo_stack[idx] = SetParameterCommand(
                        cmd.node_id_name,
                        value_tag,
                        cmd.before_value,
                        after_value,
                    )
                    self._redo_stack.clear()
                    self._cntrl_refresh_history_menu_items()
                    return
            else:
                self._parameter_drag_stream_active.add(value_tag)
        if is_curves_drag_edit:
            for idx in range(len(self._undo_stack) - 1, -1, -1):
                cmd = self._undo_stack[idx]
                if not isinstance(cmd, SetParameterCommand):
                    break
                if cmd.value_tag != value_tag:
                    continue
                if (
                    isinstance(cmd.before_value, list)
                    and isinstance(cmd.after_value, list)
                    and len(cmd.before_value) != len(cmd.after_value)
                ):
                    # Don't merge drag updates into click add/delete edits.
                    break
                self._undo_stack[idx] = SetParameterCommand(
                    cmd.node_id_name,
                    value_tag,
                    cmd.before_value,
                    after_value,
                )
                del self._undo_stack[idx + 1:]
                self._parameter_last_coalesce_hint[value_tag] = True
                self._redo_stack.clear()
                self._cntrl_refresh_history_menu_items()
                return

        if self._undo_stack and isinstance(self._undo_stack[-1], SetParameterCommand):
            last_cmd = self._undo_stack[-1]
            can_merge_curves_drag = bool(is_curves_drag_edit)
            if (
                can_merge_curves_drag
                and self._parameter_last_coalesce_hint.get(value_tag, False) is False
            ):
                can_merge_curves_drag = False
            if (
                (is_numeric_edit or can_merge_curves_drag)
                and last_cmd.value_tag == value_tag
            ):
                if last_cmd.before_value == after_value:
                    self._undo_stack.pop()
                else:
                    self._undo_stack[-1] = SetParameterCommand(
                        last_cmd.node_id_name,
                        value_tag,
                        last_cmd.before_value,
                        after_value,
                    )
                self._parameter_last_coalesce_hint[value_tag] = bool(coalesce_hint)
                self._redo_stack.clear()
                self._cntrl_refresh_history_menu_items()
                return
        self._cntrl_push_undo_command(
            SetParameterCommand(
                node_id_name,
                value_tag,
                before_value,
                after_value,
            )
        )
        self._parameter_last_coalesce_hint[value_tag] = bool(coalesce_hint)
        if not bool(coalesce_hint):
            self._parameter_drag_stream_active.discard(value_tag)

    def _cntrl_toggle_result_node(
        self,
        source_node_id_name,
        result_node_tag,
        enabled,
    ):
        if ':' not in source_node_id_name:
            return
        if result_node_tag not in self._node_instance_list:
            return

        source_output_tag = self._cntrl_find_node_port(
            source_node_id_name, 'Image', 'Output'
        )
        if source_output_tag is None:
            return

        existing_result_node = self._cntrl_find_linked_result_node(
            source_node_id_name,
            result_node_tag,
        )
        if enabled:
            if existing_result_node is not None:
                return
            source_pos = dpg.get_item_pos(source_node_id_name)
            new_id, new_node_id_name = self._mdl_add_node(result_node_tag)
            self._mdl_register_node_ref(NodeRef(str(new_id), result_node_tag))
            result_pos = [source_pos[0] + 260, source_pos[1]]
            self._vw_add_node(result_node_tag, new_id, result_pos)
            self._node_list.append(new_node_id_name)
            result_input_tag = self._cntrl_find_node_port(
                new_node_id_name, 'Image', 'Input'
            )
            if result_input_tag is None:
                self._mdl_delete_node(new_node_id_name)
                self._vw_delete_item(new_node_id_name)
                return
            if self._mdl_add_link(source_output_tag, result_input_tag):
                link_dpg_id = self._vw_add_link(source_output_tag, result_input_tag)
                self._vw_register_link(source_output_tag, result_input_tag, link_dpg_id)
            self._mdl_sort_node_graph()
        else:
            if existing_result_node is None:
                return
            self._mdl_delete_node(existing_result_node)
            self._vw_delete_links_for_node(existing_result_node)
            self._vw_delete_item(existing_result_node)

    def _cntrl_find_linked_result_node(self, source_node_id_name, result_node_tag):
        source_output_tag = self._cntrl_find_node_port(
            source_node_id_name, 'Image', 'Output'
        )
        if source_output_tag is None:
            return None
        for source_tag, dest_tag in self._node_link_list:
            if source_tag != source_output_tag:
                continue
            dest_port = self._port_registry.get(dest_tag)
            if dest_port and dest_port.node_ref.node_tag == result_node_tag:
                return dest_port.node_ref.node_id_name
        return None

    def _cntrl_get_link_from_dpg_id(self, link_dpg_id):
        link_dpg_config = dpg.get_item_configuration(link_dpg_id)
        return [
            dpg.get_item_alias(link_dpg_config['attr_1']),
            dpg.get_item_alias(link_dpg_config['attr_2']),
        ]

    def _cntrl_open_insert_link_popup(self, sender, data):
        del sender, data
        output_port_tag = self._cntrl_get_hovered_output_port_tag()
        self._pending_add_from_output_tag = output_port_tag
        input_port_tag = self._cntrl_get_hovered_input_port_tag()
        self._pending_add_to_input_tag = input_port_tag

        if output_port_tag is not None:
            self._pending_insert_link_dpg_id = None
            self._pending_add_to_input_tag = None
            self._vw_hide_insert_link_popup()
            source_port = self._cntrl_parse_port_tag(output_port_tag)
            self._vw_populate_append_popup_menu(
                source_port.data_type if source_port else ''
            )
            mouse_pos = dpg.get_mouse_pos(local=False)
            window_pos = dpg.get_item_pos(self._window_tag)
            popup_pos = [
                int(mouse_pos[0] - window_pos[0]),
                int(mouse_pos[1] - window_pos[1]),
            ]
            self._vw_show_add_node_popup(popup_pos)
            return

        if input_port_tag is not None:
            self._pending_insert_link_dpg_id = None
            self._pending_add_from_output_tag = None
            self._vw_hide_insert_link_popup()
            dest_port = self._cntrl_parse_port_tag(input_port_tag)
            self._vw_populate_prepend_popup_menu(
                dest_port.data_type if dest_port else ''
            )
            mouse_pos = dpg.get_mouse_pos(local=False)
            window_pos = dpg.get_item_pos(self._window_tag)
            popup_pos = [
                int(mouse_pos[0] - window_pos[0]),
                int(mouse_pos[1] - window_pos[1]),
            ]
            self._vw_show_add_node_popup(popup_pos)
            return

        self._vw_hide_add_node_popup()
        link_dpg_id = self._cntrl_get_target_link_for_context_insert()
        self._pending_insert_link_dpg_id = link_dpg_id
        if link_dpg_id is not None:
            source_tag, dest_tag = self._cntrl_get_link_from_dpg_id(link_dpg_id)
            source_port = self._cntrl_parse_port_tag(source_tag)
            dest_port = self._cntrl_parse_port_tag(dest_tag)
            if (
                source_port is not None
                and dest_port is not None
                and source_port.data_type == dest_port.data_type
            ):
                self._vw_populate_insert_popup_menu(source_port.data_type)
            else:
                self._vw_populate_insert_popup_menu('')
            mouse_pos = dpg.get_mouse_pos(local=False)
            window_pos = dpg.get_item_pos(self._window_tag)
            popup_pos = [
                int(mouse_pos[0] - window_pos[0]),
                int(mouse_pos[1] - window_pos[1]),
            ]
            self._vw_show_insert_link_popup(popup_pos)
        else:
            self._vw_hide_insert_link_popup()

    def _cntrl_close_insert_link_popup_on_escape(self, sender, data):
        del sender, data
        if dpg.is_item_shown(self._insert_link_popup_tag):
            self._pending_insert_link_dpg_id = None
            self._vw_hide_insert_link_popup()
        if dpg.is_item_shown(self._add_node_popup_tag):
            self._vw_hide_add_node_popup()
        self._pending_add_from_output_tag = None
        self._pending_add_to_input_tag = None

    def _cntrl_get_target_link_for_context_insert(self):
        selected_links = dpg.get_selected_links(self._node_editor_tag)
        if len(selected_links) == 1:
            return selected_links[0]

        for link_dpg_id in self._link_view_id_map.values():
            if dpg.is_item_hovered(link_dpg_id):
                return link_dpg_id
        return None


    def _cntrl_get_hovered_output_port_tag(self):
        for node_id_name in self._node_list:
            parts = node_id_name.split(':')
            if len(parts) < 2:
                continue
            node_id = parts[0]
            node_name = parts[1]
            for index in range(100):
                port_tag = f'{node_id}:{node_name}:Image:Output{index:02d}'
                if dpg.does_item_exist(port_tag) and dpg.is_item_hovered(port_tag):
                    return port_tag
                for port_type in ('Int', 'Float', 'Text', 'Time', 'TimeMs', 'TimeMS'):
                    type_port_tag = f'{node_id}:{node_name}:{port_type}:Output{index:02d}'
                    if dpg.does_item_exist(type_port_tag) and dpg.is_item_hovered(type_port_tag):
                        return type_port_tag
        return None

    def _cntrl_get_hovered_input_port_tag(self):
        for node_id_name in self._node_list:
            parts = node_id_name.split(':')
            if len(parts) < 2:
                continue
            node_id = parts[0]
            node_name = parts[1]
            for index in range(100):
                port_tag = f'{node_id}:{node_name}:Image:Input{index:02d}'
                if dpg.does_item_exist(port_tag) and dpg.is_item_hovered(port_tag):
                    return port_tag
                for port_type in ('Int', 'Float', 'Text', 'Time', 'TimeMs', 'TimeMS'):
                    type_port_tag = f'{node_id}:{node_name}:{port_type}:Input{index:02d}'
                    if dpg.does_item_exist(type_port_tag) and dpg.is_item_hovered(type_port_tag):
                        return type_port_tag
        return None

    def _cntrl_get_insert_node_pos(self, source_tag, dest_tag):
        source_node = ':'.join(source_tag.split(':')[:2])
        dest_node = ':'.join(dest_tag.split(':')[:2])
        source_pos = dpg.get_item_pos(source_node)
        dest_pos = dpg.get_item_pos(dest_node)

        return [
            int((source_pos[0] + dest_pos[0]) / 2),
            int((source_pos[1] + dest_pos[1]) / 2),
        ]

    def _cntrl_find_node_port(self, node_id_name, port_type, port_prefix):
        expected_attr_type = None
        if port_prefix == 'Input':
            expected_attr_type = dpg.mvNode_Attr_Input
        elif port_prefix == 'Output':
            expected_attr_type = dpg.mvNode_Attr_Output

        for index in range(100):
            port_tag = (
                f'{node_id_name}:{port_type}:{port_prefix}{index:02d}'
            )
            if not dpg.does_item_exist(port_tag):
                continue

            if expected_attr_type is not None:
                config = dpg.get_item_configuration(port_tag)
                if config.get('attribute_type') != expected_attr_type:
                    continue
            return port_tag
        return None


    def _cntrl_add_node_from_output_port(self, sender, data, user_data):
        del sender, data
        self._vw_hide_add_node_popup()

        source_tag = self._pending_add_from_output_tag
        self._pending_add_from_output_tag = None
        if source_tag is None:
            self._vw_set_link_feedback('Create from output requires a hovered output port.')
            return

        source_port = self._cntrl_parse_port_tag(source_tag)
        if source_port is None:
            self._vw_set_link_feedback('Cannot create from output: invalid source port tag format.')
            return

        link_type = source_port.data_type
        new_id, new_node_id_name = self._mdl_add_node(user_data)
        self._mdl_register_node_ref(NodeRef(str(new_id), user_data))
        source_node = source_port.node_ref.node_id_name
        source_pos = dpg.get_item_pos(source_node)
        new_pos = [source_pos[0] + 260, source_pos[1]]
        self._vw_add_node(user_data, new_id, new_pos)
        self._node_list.append(new_node_id_name)
        self._cntrl_update_node_position_cache(new_node_id_name)

        input_tag = self._cntrl_find_node_port(new_node_id_name, link_type, 'Input')
        if input_tag is None:
            self._mdl_delete_node(new_node_id_name)
            self._vw_delete_item(new_node_id_name)
            self._vw_set_link_feedback(
                f'Cannot connect {user_data}: it needs a {link_type} input port.'
            )
            return

        if self._mdl_add_link(source_tag, input_tag):
            link_dpg_id = self._vw_add_link(source_tag, input_tag)
            self._vw_register_link(source_tag, input_tag, link_dpg_id)
            self._mdl_sort_node_graph()
            node = self.get_node_instance(user_data)
            node_setting = node.get_setting_dict(str(new_id))
            self._cntrl_push_undo_command(
                AddNodeCommand(
                    new_id,
                    user_data,
                    list(new_pos),
                    copy.deepcopy(node_setting),
                    [(source_tag, input_tag)],
                    [],
                )
            )
            self._vw_set_link_feedback('')

    def _cntrl_add_node_to_input_port(self, sender, data, user_data):
        del sender, data
        self._vw_hide_add_node_popup()

        dest_tag = self._pending_add_to_input_tag
        self._pending_add_to_input_tag = None
        if dest_tag is None:
            self._vw_set_link_feedback('Create to input requires a hovered input port.')
            return

        dest_port = self._cntrl_parse_port_tag(dest_tag)
        if dest_port is None:
            self._vw_set_link_feedback('Cannot create to input: invalid destination port tag format.')
            return

        link_type = dest_port.data_type
        new_id, new_node_id_name = self._mdl_add_node(user_data)
        self._mdl_register_node_ref(NodeRef(str(new_id), user_data))
        dest_node = dest_port.node_ref.node_id_name
        dest_pos = dpg.get_item_pos(dest_node)
        new_pos = [dest_pos[0] - 260, dest_pos[1]]
        self._vw_add_node(user_data, new_id, new_pos)
        self._node_list.append(new_node_id_name)
        self._cntrl_update_node_position_cache(new_node_id_name)

        output_tag = self._cntrl_find_node_port(new_node_id_name, link_type, 'Output')
        if output_tag is None:
            self._mdl_delete_node(new_node_id_name)
            self._vw_delete_item(new_node_id_name)
            self._vw_set_link_feedback(
                f'Cannot connect {user_data}: it needs a {link_type} output port.'
            )
            return

        replaced_links = []
        existing_link = self._mdl_get_link_by_destination(dest_tag)
        if existing_link is not None:
            self._cntrl_remove_link_by_tags(
                existing_link[0], existing_link[1], record_history=False
            )
            replaced_links.append((existing_link[0], existing_link[1]))

        if self._mdl_add_link(output_tag, dest_tag):
            link_dpg_id = self._vw_add_link(output_tag, dest_tag)
            self._vw_register_link(output_tag, dest_tag, link_dpg_id)
            self._mdl_sort_node_graph()
            node = self.get_node_instance(user_data)
            node_setting = node.get_setting_dict(str(new_id))
            if replaced_links:
                self._cntrl_push_undo_command(
                    CompositeCommand(
                        [
                            RemoveLinkCommand(replaced_links[0][0], replaced_links[0][1]),
                            AddNodeCommand(
                                new_id,
                                user_data,
                                list(new_pos),
                                copy.deepcopy(node_setting),
                                [(output_tag, dest_tag)],
                                [],
                            ),
                        ]
                    )
                )
            else:
                self._cntrl_push_undo_command(
                    AddNodeCommand(
                        new_id,
                        user_data,
                        list(new_pos),
                        copy.deepcopy(node_setting),
                        [(output_tag, dest_tag)],
                        [],
                    )
                )
            self._vw_set_link_feedback('')

    def _cntrl_insert_node_into_selected_link(self, sender, data, user_data):
        del sender, data
        self._vw_hide_insert_link_popup()
        selected_links = dpg.get_selected_links(self._node_editor_tag)
        selected_link_dpg_id = None
        if len(selected_links) == 1:
            selected_link_dpg_id = selected_links[0]
        elif self._pending_insert_link_dpg_id is not None:
            selected_link_dpg_id = self._pending_insert_link_dpg_id

        self._pending_insert_link_dpg_id = None
        self._pending_add_from_output_tag = None
        if selected_link_dpg_id is None:
            self._vw_set_link_feedback(
                'Insert into link requires a selected or hovered link.'
            )
            return
        source_tag, dest_tag = self._cntrl_get_link_from_dpg_id(
            selected_link_dpg_id
        )
        source_port = self._cntrl_parse_port_tag(source_tag)
        dest_port = self._cntrl_parse_port_tag(dest_tag)
        if source_port is None or dest_port is None:
            self._vw_set_link_feedback(
                'Cannot insert node into link: invalid port tag format.'
            )
            return

        link_type = source_port.data_type
        if link_type != dest_port.data_type:
            self._vw_set_link_feedback(
                'Cannot insert node into link: source and destination '
                'types do not match.'
            )
            return

        new_id, new_node_id_name = self._mdl_add_node(user_data)
        self._mdl_register_node_ref(NodeRef(str(new_id), user_data))
        insert_pos = self._cntrl_get_insert_node_pos(source_tag, dest_tag)
        self._vw_add_node(user_data, new_id, insert_pos)
        self._node_list.append(new_node_id_name)
        self._cntrl_update_node_position_cache(new_node_id_name)

        input_tag = self._cntrl_find_node_port(
            new_node_id_name,
            link_type,
            'Input',
        )
        output_tag = self._cntrl_find_node_port(
            new_node_id_name,
            link_type,
            'Output',
        )

        if input_tag is None or output_tag is None:
            self._mdl_delete_node(new_node_id_name)
            self._vw_delete_item(new_node_id_name)
            self._vw_set_link_feedback(
                f'Cannot insert {user_data}: it needs both {link_type} '
                'input and output ports.'
            )
            return

        original_link = [source_tag, dest_tag]
        self._mdl_delete_link(original_link)
        self._link_view_id_map.pop(tuple(original_link), None)
        self._vw_delete_item(selected_link_dpg_id)

        for new_source, new_dest in (
            (source_tag, input_tag),
            (output_tag, dest_tag),
        ):
            if self._mdl_add_link(new_source, new_dest):
                link_dpg_id = self._vw_add_link(new_source, new_dest)
                self._vw_register_link(new_source, new_dest, link_dpg_id)

        self._mdl_sort_node_graph()
        node = self.get_node_instance(user_data)
        node_setting = node.get_setting_dict(str(new_id))
        self._cntrl_push_undo_command(
            CompositeCommand(
                [
                    RemoveLinkCommand(source_tag, dest_tag),
                    AddNodeCommand(
                        new_id,
                        user_data,
                        list(insert_pos),
                        copy.deepcopy(node_setting),
                        [
                            (source_tag, input_tag),
                            (output_tag, dest_tag),
                        ],
                        [],
                    ),
                ]
            )
        )
        self._vw_set_link_feedback('')

        if self._use_debug_print:
            print('**** _cntrl_insert_node_into_selected_link ****')
            print(f'\tselected_link_dpg_id      : {selected_link_dpg_id}')
            print(f'\tinserted_node             : {new_node_id_name}')
            print(f'\tself._node_list           : {self._node_list}')
            print(f'\tself._node_link_list      : {self._node_link_list}')
            print()

    def _cntrl_link(self, sender, data):
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            self._vw_set_link_feedback(
                'Link rejected: invalid link data from DearPyGui.'
            )
            return

        source_dpg_id, dest_dpg_id = data
        source_tag = dpg.get_item_alias(source_dpg_id)
        dest_tag = dpg.get_item_alias(dest_dpg_id)
        if source_tag is None or dest_tag is None:
            self._vw_set_link_feedback(
                'Link rejected: unable to resolve source or destination port.'
            )
            return

        source_port = self._cntrl_parse_port_tag(source_tag)
        dest_port = self._cntrl_parse_port_tag(dest_tag)
        if source_port is None or dest_port is None:
            self._vw_set_link_feedback(
                'Link rejected: invalid port tag format.'
            )
            return

        source_type = source_port.data_type
        dest_type = dest_port.data_type

        if source_type != dest_type:
            self._vw_set_link_feedback(
                f'Link rejected: {source_type} output cannot connect to '
                f'{dest_type} input.'
            )
            return

        if self._cntrl_would_create_cycle(source_tag, dest_tag):
            self._vw_set_link_feedback(
                'Link rejected: this connection would create a cycle.'
            )
            return

        existing_link = self._mdl_get_link_by_destination(dest_tag)
        if existing_link is not None:
            if existing_link[0] == source_tag:
                self._vw_set_link_feedback(
                    'Link rejected: input is already connected to that source.'
                )
                self._mdl_sort_node_graph()
                return
            self._cntrl_remove_link_by_tags(
                existing_link[0], existing_link[1], record_history=False
            )

        if self._mdl_add_link(source_tag, dest_tag):
            link_dpg_id = self._vw_add_link(source_dpg_id, dest_dpg_id)
            self._vw_register_link(source_tag, dest_tag, link_dpg_id)
            if existing_link is not None:
                self._cntrl_push_undo_command(
                    ReplaceLinkCommand(
                        existing_link[0],
                        source_tag,
                        dest_tag,
                    )
                )
            else:
                self._cntrl_push_undo_command(AddLinkCommand(source_tag, dest_tag))
            self._vw_set_link_feedback('')
        self._mdl_sort_node_graph()

        if self._use_debug_print:
            print('**** _cntrl_link ****')
            print(f'\tsender                     : {sender}')
            print(f'\tdata                       : {data}')
            print(f'\tself._node_list            : {self._node_list}')
            print(f'\tself._node_link_list       : {self._node_link_list}')
            print(f'\tself._node_connection_dict : {self._node_connection_dict}')
            print()

    def _cntrl_close_window(self, sender):
        self._vw_delete_item(sender)

    def _cntrl_file_export(self, sender, data):
        setting_dict = self._mdl_get_export_settings()
        with open(data['file_path_name'], 'w') as fp:
            json.dump(setting_dict, fp, indent=4)

        if self._use_debug_print:
            print('**** _cntrl_file_export ****')
            print(f'\tsender          : {sender}')
            print(f'\tdata            : {data}')
            print(f'\tsetting_dict    : {setting_dict}')
            print()

    def _cntrl_file_export_menu(self, sender, data, user_data):
        self._vw_show_file_export()

    def _cntrl_file_import_menu(self, sender, data, user_data):
        self._vw_show_file_import()

    def import_setting_file(self, file_path):
        """Public helper for startup/programmatic imports."""
        return self._cntrl_import_setting_file(file_path)

    def _cntrl_import_setting_file(self, file_path):
        with open(file_path) as fp:
            setting_dict = json.load(fp)
        self._cntrl_import_setting_dict(setting_dict)
        return setting_dict

    def _cntrl_import_setting_dict(self, setting_dict):
        self._suspend_parameter_history = True
        try:
            self._cntrl_import_setting_dict_body(setting_dict)
        finally:
            self._suspend_parameter_history = False

    def _cntrl_import_setting_dict_body(self, setting_dict):
        if setting_dict is None:
            return

        if 'node_list' not in setting_dict or 'link_list' not in setting_dict:
            raise KeyError('Invalid node editor setting file format.')

        id_map = {}
        new_node_list = []
        new_link_list = []

        for node_id_name in setting_dict['node_list']:
            old_id, node_name = node_id_name.split(':')
            node = self._node_instance_list[node_name]

            new_id, _ = self._mdl_add_node(node_name)
            self._mdl_register_node_ref(NodeRef(str(new_id), node_name))
            id_map[old_id] = str(new_id)

            ver = setting_dict[node_id_name]['setting']['ver']
            if ver != node._ver:
                warning_node_name = setting_dict[node_id_name]['name']
                print(f'WARNING : {warning_node_name} is different version')
                print(f'\t                 Load Version -> {ver}')
                print(f'\t                 Code Version -> {node._ver}\n')

            pos = setting_dict[node_id_name]['setting']['pos']
            self._vw_add_node(node_name, new_id, pos)
            self._node_list.append(f'{new_id}:{node_name}')

            original_setting = setting_dict[node_id_name]['setting']
            new_setting = {}
            for key, value in original_setting.items():
                if key.startswith(f'{old_id}:{node_name}'):
                    new_key = key.replace(
                        f'{old_id}:{node_name}', f'{new_id}:{node_name}', 1)
                    new_setting[new_key] = value
                else:
                    new_setting[key] = value

            node.set_setting_dict(new_id, new_setting)
            self._cntrl_prime_parameter_last_values_from_setting(new_setting)
            new_node_list.append(f'{new_id}:{node_name}')

        for link_info in setting_dict['link_list']:
            source_parts = link_info[0].split(':')
            dest_parts = link_info[1].split(':')

            if source_parts[0] in id_map and dest_parts[0] in id_map:
                source_parts[0] = id_map[source_parts[0]]
                dest_parts[0] = id_map[dest_parts[0]]
                new_source = ':'.join(source_parts)
                new_destination = ':'.join(dest_parts)
                link_dpg_id = self._vw_add_link(new_source, new_destination)
                self._vw_register_link(new_source, new_destination, link_dpg_id)
                if not self._mdl_add_link(new_source, new_destination):
                    # Preserve serialized links for compatibility with older
                    # graph formats (e.g., duplicate destination links).
                    new_link_list.append([new_source, new_destination])

        self._node_link_list.extend(new_link_list)
        self._mdl_sort_node_graph()
        self._cntrl_sync_position_cache()

    def _cntrl_sync_position_cache(self):
        self._node_position_cache = {}
        for node_id_name in self._node_list:
            self._cntrl_update_node_position_cache(node_id_name)

    def _cntrl_prime_parameter_last_values_from_setting(self, setting_dict):
        if not isinstance(setting_dict, dict):
            return
        for key, value in setting_dict.items():
            if isinstance(key, str) and key.endswith('Value'):
                self._parameter_last_values[key] = value

    def _cntrl_file_import(self, sender, data):
        if data['file_name'] == '.':
            return
        setting_dict = self._cntrl_import_setting_file(data['file_path_name'])

        if self._use_debug_print:
            print('**** _cntrl_file_import ****')
            print(f'\tsender          : {sender}')
            print(f'\tdata            : {data}')
            print(f'\tsetting_dict    : {setting_dict}')
            print()

    def _cntrl_save_last_pos(self, sender, data):
        selected_nodes = dpg.get_selected_nodes(self._node_editor_tag)
        if selected_nodes:
            self._last_pos = dpg.get_item_pos(selected_nodes[0])

    def _cntrl_add_link_by_tags(self, source_tag, dest_tag):
        source_tag = self._cntrl_resolve_history_port_tag(source_tag)
        dest_tag = self._cntrl_resolve_history_port_tag(dest_tag)
        if not self._mdl_add_link(source_tag, dest_tag):
            return False
        link_dpg_id = self._vw_add_link(source_tag, dest_tag)
        self._vw_register_link(source_tag, dest_tag, link_dpg_id)
        return True

    def _cntrl_remove_link_by_tags(self, source_tag, dest_tag, record_history=False):
        source_tag = self._cntrl_resolve_history_port_tag(source_tag)
        dest_tag = self._cntrl_resolve_history_port_tag(dest_tag)
        link = [source_tag, dest_tag]
        if link not in self._node_link_list:
            return False
        self._mdl_delete_link(link)
        self._vw_delete_link(source_tag, dest_tag)
        if record_history:
            self._cntrl_push_undo_command(RemoveLinkCommand(source_tag, dest_tag))
        return True

    def _cntrl_capture_move_start_positions(self, sender, data):
        del sender, data
        self._move_start_positions = {}
        for node_dpg_id in dpg.get_selected_nodes(self._node_editor_tag):
            node_id_name = dpg.get_item_alias(node_dpg_id)
            if isinstance(node_id_name, str) and ':' in node_id_name:
                before_pos = self._node_position_cache.get(
                    node_id_name,
                    list(dpg.get_item_pos(node_id_name)),
                )
                self._move_start_positions[node_id_name] = list(before_pos)

        if self._move_start_positions:
            return
        for node_id_name in self._node_list:
            if dpg.does_item_exist(node_id_name) and dpg.is_item_hovered(node_id_name):
                before_pos = self._node_position_cache.get(
                    node_id_name,
                    list(dpg.get_item_pos(node_id_name)),
                )
                self._move_start_positions[node_id_name] = list(before_pos)
                break

    def _cntrl_commit_move_commands(self, sender, data):
        del sender, data
        for node_id_name, before_pos in self._move_start_positions.items():
            if not dpg.does_item_exist(node_id_name):
                continue
            after_pos = list(dpg.get_item_pos(node_id_name))
            if before_pos != after_pos:
                self._cntrl_push_undo_command(
                    MoveNodeCommand(node_id_name, list(before_pos), list(after_pos))
                )
            self._node_position_cache[node_id_name] = list(after_pos)
        self._move_start_positions = {}

    def _cntrl_push_undo_command(self, command):
        self._undo_stack.append(command)
        self._redo_stack.clear()
        self._cntrl_refresh_history_menu_items()

    def _cntrl_refresh_history_menu_items(self):
        undo_label = 'Undo'
        redo_label = 'Redo'
        undo_enabled = len(self._undo_stack) > 0
        redo_enabled = len(self._redo_stack) > 0
        if undo_enabled:
            undo_label += f' ({history_command_label(self._undo_stack[-1], self._cntrl_get_parameter_label)})'
        if redo_enabled:
            redo_label += f' ({history_command_label(self._redo_stack[-1], self._cntrl_get_parameter_label)})'
        if dpg.does_item_exist('Menu_Edit_Undo'):
            dpg.configure_item(
                'Menu_Edit_Undo',
                label=undo_label,
                enabled=undo_enabled,
            )
        if dpg.does_item_exist('Menu_Edit_Redo'):
            dpg.configure_item(
                'Menu_Edit_Redo',
                label=redo_label,
                enabled=redo_enabled,
            )
        self._cntrl_refresh_history_window()

    def _cntrl_history_lines(self, stack):
        if not stack:
            return '(empty)'
        lines = []
        for index, command in enumerate(reversed(stack), start=1):
            lines.append(f'{index}. {history_command_label(command, self._cntrl_get_parameter_label)}')
        return '\n'.join(lines)

    def _cntrl_get_parameter_label(self, value_tag):
        if not dpg.does_item_exist(value_tag):
            return None
        try:
            conf = dpg.get_item_configuration(value_tag)
        except Exception:
            return None
        if not isinstance(conf, dict):
            return None
        label = conf.get('label', None)
        return str(label) if label else None

    def _cntrl_refresh_history_window(self):
        if dpg.does_item_exist(self._history_undo_text_tag):
            dpg.set_value(
                self._history_undo_text_tag,
                self._cntrl_history_lines(self._undo_stack),
            )
        if dpg.does_item_exist(self._history_redo_text_tag):
            dpg.set_value(
                self._history_redo_text_tag,
                self._cntrl_history_lines(self._redo_stack),
            )

    def _cntrl_toggle_history_window(self, sender, data):
        del sender, data
        if not dpg.does_item_exist(self._history_window_tag):
            return
        if dpg.is_item_shown(self._history_window_tag):
            dpg.hide_item(self._history_window_tag)
            return
        self._cntrl_refresh_history_window()
        dpg.show_item(self._history_window_tag)

    def _cntrl_undo(self, sender, data):
        del sender, data
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        self._suspend_parameter_history = True
        try:
            if isinstance(cmd, AddNodeCommand):
                cmd.undo(self)
            elif isinstance(cmd, DeleteNodesCommand):
                cmd.undo(self)
            elif isinstance(cmd, MoveNodeCommand):
                cmd.undo(self)
            elif isinstance(cmd, AddLinkCommand):
                cmd.undo(self)
            elif isinstance(cmd, RemoveLinkCommand):
                cmd.undo(self)
            elif isinstance(cmd, ReplaceLinkCommand):
                cmd.undo(self)
            elif isinstance(cmd, CompositeCommand):
                cmd.undo(self)
            elif isinstance(cmd, SetParameterCommand):
                cmd.undo(self)
        finally:
            self._suspend_parameter_history = False
        self._redo_stack.append(cmd)
        self._cntrl_refresh_history_menu_items()

    def _cntrl_redo(self, sender, data):
        del sender, data
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        self._suspend_parameter_history = True
        try:
            if isinstance(cmd, AddNodeCommand):
                cmd.redo(self)
            elif isinstance(cmd, DeleteNodesCommand):
                cmd.redo(self)
            elif isinstance(cmd, MoveNodeCommand):
                cmd.redo(self)
            elif isinstance(cmd, AddLinkCommand):
                cmd.redo(self)
            elif isinstance(cmd, RemoveLinkCommand):
                cmd.redo(self)
            elif isinstance(cmd, ReplaceLinkCommand):
                cmd.redo(self)
            elif isinstance(cmd, CompositeCommand):
                cmd.redo(self)
            elif isinstance(cmd, SetParameterCommand):
                cmd.redo(self)
        finally:
            self._suspend_parameter_history = False
        self._undo_stack.append(cmd)
        self._cntrl_refresh_history_menu_items()

    def _cntrl_reconnect_through_deleted_nodes(self, deleted_node_tags):
        deleted_set = set(deleted_node_tags)
        if not deleted_set:
            return []

        links_by_dest_node = {}
        outgoing_edges = []
        for source_tag, dest_tag in self._node_link_list:
            source_port = self._port_registry.get(source_tag)
            dest_port = self._port_registry.get(dest_tag)
            if source_port is None or dest_port is None:
                continue
            source_node = source_port.node_ref.node_id_name
            dest_node = dest_port.node_ref.node_id_name
            link_type = source_port.data_type
            links_by_dest_node.setdefault(dest_node, []).append(
                (source_tag, dest_tag, source_node, dest_node, link_type)
            )

            if source_node in deleted_set and dest_node not in deleted_set:
                outgoing_edges.append(
                    (source_tag, dest_tag, source_node, dest_node, link_type)
                )

        def _find_external_sources(start_deleted_node, link_type):
            stack = [start_deleted_node]
            visited = set()
            boundary_sources = set()
            while stack:
                current_node = stack.pop()
                if current_node in visited:
                    continue
                visited.add(current_node)
                for (
                    src_tag,
                    _dst_tag,
                    src_node,
                    _cur_dest,
                    cur_type,
                ) in links_by_dest_node.get(current_node, []):
                    if cur_type != link_type:
                        continue
                    if src_node in deleted_set:
                        stack.append(src_node)
                    else:
                        boundary_sources.add(src_tag)
            return boundary_sources

        reconnect_pairs = []
        for _src_tag, dest_tag, source_node, _dest_node, link_type in outgoing_edges:
            incoming_sources = _find_external_sources(source_node, link_type)
            # Heal only when there is a single unambiguous external source.
            # If there are 0 or >1 candidates, skip reconnection by design.
            if len(incoming_sources) != 1:
                continue
            source_tag = next(iter(incoming_sources))
            if source_tag == dest_tag:
                continue
            reconnect_pairs.append((source_tag, dest_tag))
        return reconnect_pairs

    def _cntrl_delete_selected(self, sender, data):
        selected_nodes = dpg.get_selected_nodes(self._node_editor_tag)
        selected_node_tags = [dpg.get_item_alias(node_dpg_id) for node_dpg_id in selected_nodes]
        selected_links = dpg.get_selected_links(self._node_editor_tag)
        self._cntrl_delete_targets(selected_node_tags, selected_links)

    def _cntrl_delete_targets(self, selected_node_tags, selected_link_ids):
        deleted_nodes_payload = []
        removed_links = []
        removed_selected_links = []
        reconnect_pairs = self._cntrl_reconnect_through_deleted_nodes(selected_node_tags)
        for node_tag in selected_node_tags:
            if node_tag not in self._node_list:
                continue
            node_id, node_name = node_tag.split(':')
            node = self.get_node_instance(node_name)
            deleted_nodes_payload.append(
                {
                    'node_id': node_id,
                    'node_tag': node_name,
                    'pos': list(dpg.get_item_pos(node_tag)),
                    'setting': copy.deepcopy(node.get_setting_dict(node_id)),
                }
            )
            for source_tag, dest_tag in list(self._node_link_list):
                if source_tag.startswith(f'{node_id}:{node_name}:') or dest_tag.startswith(f'{node_id}:{node_name}:'):
                    removed_links.append((source_tag, dest_tag))
        for node_tag in selected_node_tags:
            self._cntrl_delete_node_by_tag(node_tag)

        for source_tag, dest_tag in reconnect_pairs:
            if self._mdl_add_link(source_tag, dest_tag):
                link_dpg_id = self._vw_add_link(source_tag, dest_tag)
                self._vw_register_link(source_tag, dest_tag, link_dpg_id)

        for link_dpg_id in selected_link_ids:
            if not dpg.does_item_exist(link_dpg_id):
                continue
            try:
                link = self._cntrl_get_link_from_dpg_id(link_dpg_id)
            except Exception:
                continue
            if self._cntrl_remove_link_by_tags(link[0], link[1], record_history=False):
                removed_selected_links.append((link[0], link[1]))

        if deleted_nodes_payload:
            self._cntrl_push_undo_command(
                DeleteNodesCommand(
                    deleted_nodes_payload,
                    removed_links,
                    reconnect_pairs,
                    removed_selected_links,
                )
            )

        if self._use_debug_print:
            print('**** _cntrl_delete_selected ****')
            print(f'\tself._node_list            : {self._node_list}')
            print(f'\tself._node_link_list       : {self._node_link_list}')
            print(f'\tself._node_connection_dict : {self._node_connection_dict}')

    def _cntrl_would_create_cycle(self, source_tag, dest_tag):
        source_port = self._cntrl_parse_port_tag(source_tag)
        dest_port = self._cntrl_parse_port_tag(dest_tag)
        if source_port is None or dest_port is None:
            return False

        source_node = source_port.node_ref.node_id_name
        dest_node = dest_port.node_ref.node_id_name
        if source_node == dest_node:
            return True

        adjacency = {}
        for existing_source_tag, existing_dest_tag in self._node_link_list:
            existing_source = self._cntrl_parse_port_tag(existing_source_tag)
            existing_dest = self._cntrl_parse_port_tag(existing_dest_tag)
            if existing_source is None or existing_dest is None:
                continue
            adjacency.setdefault(existing_source.node_ref.node_id_name, set()).add(
                existing_dest.node_ref.node_id_name
            )
        adjacency.setdefault(source_node, set()).add(dest_node)

        stack = [dest_node]
        visited = set()
        while stack:
            node = stack.pop()
            if node == source_node:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency.get(node, []))
        return False

    def _cntrl_delete_node_by_tag(self, node_tag):
        if node_tag not in self._node_list:
            return

        self._mdl_delete_node(node_tag)
        self._node_position_cache.pop(node_tag, None)
        self._vw_delete_links_for_node(node_tag)
        if dpg.does_item_exist(node_tag):
            self._vw_delete_item(node_tag)

    def _cntrl_delete_node_by_button(self, sender, data, user_data):
        node_tag = user_data
        if not node_tag or not dpg.does_item_exist(node_tag):
            return
        self._cntrl_delete_targets([node_tag], [])

    # Public functions
    def get_node_list(self):
        return self._node_list

    def is_node_active(self, node_id_name):
        return node_id_name in self._node_list

    def get_sorted_node_connection(self):
        return self._node_connection_dict

    def get_node_instance(self, node_name):
        return self._node_instance_list.get(node_name, None)

    def set_terminate_flag(self, flag=True):
        self._terminate_flag = flag

    def get_terminate_flag(self):
        return self._terminate_flag
