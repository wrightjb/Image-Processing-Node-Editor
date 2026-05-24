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
        while index < len(node_id_list):
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
            if source_port is None or dest_port is None:
                continue
            if (
                source_port.node_ref.node_id_name == node_id_name
                or dest_port.node_ref.node_id_name == node_id_name
            ):
                delete_targets.append((source_tag, dest_tag))

        for source_tag, dest_tag in delete_targets:
            self._vw_delete_link(source_tag, dest_tag)

    def _vw_delete_item(self, item_id):
        dpg.delete_item(item_id)

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

        if self._use_debug_print:
            print('**** _cntrl_add_node ****')
            print(f'\tNode ID         : {self._node_id}')
            print(f'\tsender          : {sender}')
            print(f'\tdata            : {data}')
            print(f'\tuser_data       : {user_data}')
            print(f'\tself._node_list : {", ".join(self._node_list)}')
            print()

    def _cntrl_node_callback(self, event_name, data):
        if event_name == 'toggle_result_node':
            if not isinstance(data, dict):
                return
            self._cntrl_toggle_result_node(
                str(data.get('source_node_id_name', '')),
                str(data.get('result_node_tag', '')),
                bool(data.get('enabled', False)),
            )

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

        output_tag = self._cntrl_find_node_port(new_node_id_name, link_type, 'Output')
        if output_tag is None:
            self._mdl_delete_node(new_node_id_name)
            self._vw_delete_item(new_node_id_name)
            self._vw_set_link_feedback(
                f'Cannot connect {user_data}: it needs a {link_type} output port.'
            )
            return

        if self._mdl_add_link(output_tag, dest_tag):
            link_dpg_id = self._vw_add_link(output_tag, dest_tag)
            self._vw_register_link(output_tag, dest_tag, link_dpg_id)
            self._mdl_sort_node_graph()
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

        existing_link = self._mdl_get_link_by_destination(dest_tag)
        if existing_link is not None:
            if existing_link[0] == source_tag:
                self._vw_set_link_feedback(
                    'Link rejected: input is already connected to that source.'
                )
                self._mdl_sort_node_graph()
                return
            self._mdl_delete_link(existing_link)
            self._vw_delete_link(*existing_link)

        if self._mdl_add_link(source_tag, dest_tag):
            link_dpg_id = self._vw_add_link(source_dpg_id, dest_dpg_id)
            self._vw_register_link(source_tag, dest_tag, link_dpg_id)
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
                new_link_list.append([new_source, new_destination])

        self._node_link_list.extend(new_link_list)
        self._mdl_sort_node_graph()

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

    def _cntrl_reconnect_through_node(self, node_id_name):
        incoming_by_type = {}
        outgoing_by_type = {}

        for source_tag, dest_tag in self._node_link_list:
            source_port = self._port_registry.get(source_tag)
            dest_port = self._port_registry.get(dest_tag)
            if source_port is None or dest_port is None:
                continue
            if dest_port.node_ref.node_id_name == node_id_name:
                incoming_by_type.setdefault(dest_port.data_type, []).append(source_tag)
            elif source_port.node_ref.node_id_name == node_id_name:
                outgoing_by_type.setdefault(source_port.data_type, []).append(dest_tag)

        reconnect_pairs = []
        for link_type, outgoing_destinations in outgoing_by_type.items():
            incoming_sources = incoming_by_type.get(link_type, [])
            if len(incoming_sources) != 1:
                continue
            source_tag = incoming_sources[0]
            for dest_tag in outgoing_destinations:
                if source_tag == dest_tag:
                    continue
                reconnect_pairs.append((source_tag, dest_tag))

        return reconnect_pairs

    def _cntrl_delete_selected(self, sender, data):
        selected_nodes = dpg.get_selected_nodes(self._node_editor_tag)
        reconnect_pairs = []
        for node_dpg_id in selected_nodes:
            node_tag = dpg.get_item_alias(node_dpg_id)
            reconnect_pairs.extend(self._cntrl_delete_node_by_tag(node_tag))

        for source_tag, dest_tag in reconnect_pairs:
            if self._mdl_add_link(source_tag, dest_tag):
                link_dpg_id = self._vw_add_link(source_tag, dest_tag)
                self._vw_register_link(source_tag, dest_tag, link_dpg_id)

        selected_links = dpg.get_selected_links(self._node_editor_tag)
        for link_dpg_id in selected_links:
            link = self._cntrl_get_link_from_dpg_id(link_dpg_id)
            self._mdl_delete_link(link)
            self._link_view_id_map.pop(tuple(link), None)
            self._vw_delete_item(link_dpg_id)

        if self._use_debug_print:
            print('**** _cntrl_delete_selected ****')
            print(f'\tself._node_list            : {self._node_list}')
            print(f'\tself._node_link_list       : {self._node_link_list}')
            print(f'\tself._node_connection_dict : {self._node_connection_dict}')

    def _cntrl_delete_node_by_tag(self, node_tag):
        reconnect_pairs = self._cntrl_reconnect_through_node(node_tag)

        if node_tag not in self._node_list:
            return []

        self._mdl_delete_node(node_tag)
        self._vw_delete_links_for_node(node_tag)
        if dpg.does_item_exist(node_tag):
            self._vw_delete_item(node_tag)
        return reconnect_pairs

    def _cntrl_delete_node_by_button(self, sender, data, user_data):
        delete_callback = (
            lambda s=None, d=None, tag=user_data:
            self._cntrl_delete_node_deferred(s, d, tag)
        )
        dpg.set_frame_callback(
            dpg.get_frame_count() + 1,
            delete_callback,
        )

    def _cntrl_delete_node_deferred(self, sender, data, user_data):
        node_tag = user_data
        if (not node_tag) or (not dpg.does_item_exist(node_tag)):
            return

        reconnect_pairs = self._cntrl_delete_node_by_tag(node_tag)
        for source_tag, dest_tag in reconnect_pairs:
            if self._mdl_add_link(source_tag, dest_tag):
                link_dpg_id = self._vw_add_link(source_tag, dest_tag)
                self._vw_register_link(source_tag, dest_tag, link_dpg_id)

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
