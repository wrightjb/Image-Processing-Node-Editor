#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import json
import platform
import datetime
from glob import glob
from collections import OrderedDict
from importlib import import_module

import dearpygui.dearpygui as dpg


class DpgNodeEditor(object):
    _ver = '0.0.1'

    _node_editor_tag = 'NodeEditor'
    _node_editor_label = 'Node editor'
    _window_tag = _node_editor_tag + 'Window'

    _node_id = 0
    _node_instance_list = {}
    _node_list = []
    _node_link_list = []

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
        self._node_connection_dict = OrderedDict([])
        self._use_debug_print = use_debug_print
        self._terminate_flag = False
        self._opencv_setting_dict = opencv_setting_dict

    def _mdl_add_node(self, node_tag):
        self._node_id += 1
        new_node_id_name = f"{self._node_id}:{node_tag}"
        return self._node_id, new_node_id_name

    def _mdl_add_link(self, source_tag, dest_tag):
        if any(dest_tag == d_tag for _, d_tag in self._node_link_list):
            return False
        self._node_link_list.append([source_tag, dest_tag])
        return True

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
            source_node = ':'.join(link_info[0].split(':')[:2])
            destination_node = ':'.join(link_info[1].split(':')[:2])
            if source_node == node_id_name or destination_node == node_id_name:
                self._node_link_list.remove(link_info)
        self._mdl_sort_node_graph()

    def _mdl_delete_link(self, link):
        self._node_link_list.remove(link)
        self._mdl_sort_node_graph()

    def _mdl_sort_node_graph(self):
        node_list = self._node_list
        node_link_list = self._node_link_list

        node_id_dict = OrderedDict({})
        node_connection_dict = OrderedDict({})

        # ノードIDとノード接続を辞書形式で整理
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

        # 入力から出力に向かって処理順序を入れ替える
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

        # 接続リストに登場しないノードを追加する(入力ノード等)
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
            with dpg.node_editor(
                    tag=self._node_editor_tag,
                    callback=self._cntrl_link,
                    minimap=True,
                    minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            ):
                pass
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

    def _vw_add_node(self, node_tag, new_id, pos):
        node = self._node_instance_list[node_tag]
        node.add_node(
            self._node_editor_tag,
            new_id,
            pos=pos,
            opencv_setting_dict=self._opencv_setting_dict,
        )

    def _vw_add_link(self, source, destination):
        dpg.add_node_link(source, destination, parent=self._node_editor_tag)

    def _vw_delete_item(self, item_id):
        dpg.delete_item(item_id)

    def _vw_show_file_export(self):
        dpg.show_item('file_export')

    def _vw_show_file_import(self):
        dpg.show_item('file_import')

    # -------------------------------------------------------------------------
    # Controller functions
    def _cntrl_init(self, node_dir, menu_dict):
        self._cntrl_discover_nodes(node_dir, menu_dict)
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self._cntrl_save_last_pos)
            dpg.add_key_press_handler(
                dpg.mvKey_Delete,
                callback=self._cntrl_delete_selected,
            )

    def _cntrl_discover_nodes(self, node_dir, menu_dict):
        # メニュー項目定義(key：メニュー名、value：ノードのコード格納ディレクトリ名)
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
                    'label': node.node_label
                })

    def _cntrl_add_node(self, sender, data, user_data):
        new_id, new_node_id_name = self._mdl_add_node(user_data)
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

    def _cntrl_link(self, sender, data):
        source_dpg_id, dest_dpg_id = data
        source_tag = dpg.get_item_alias(source_dpg_id)
        dest_tag = dpg.get_item_alias(dest_dpg_id)
        source_type = source_tag.split(':')[2]
        dest_type = dest_tag.split(':')[2]

        if source_type != dest_type:
            return
        if self._mdl_add_link(source_tag, dest_tag):
            self._vw_add_link(source_dpg_id, dest_dpg_id)
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
                self._vw_add_link(new_source, new_destination)
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

    def _cntrl_delete_selected(self, sender, data):
        selected_nodes = dpg.get_selected_nodes(self._node_editor_tag)
        for node_dpg_id in selected_nodes:
            node_tag = dpg.get_item_alias(node_dpg_id)
            self._mdl_delete_node(node_tag)
            self._vw_delete_item(node_dpg_id)

        selected_links = dpg.get_selected_links(self._node_editor_tag)
        for link_dpg_id in selected_links:
            link_dpg_config = dpg.get_item_configuration(link_dpg_id)
            link = [
                dpg.get_item_alias(link_dpg_config['attr_1']),
                dpg.get_item_alias(link_dpg_config['attr_2'])
            ]
            self._mdl_delete_link(link)
            self._vw_delete_item(link_dpg_id)

        if self._use_debug_print:
            print('**** _cntrl_delete_selected ****')
            print(f'\tself._node_list            : {self._node_list}')
            print(f'\tself._node_link_list       : {self._node_link_list}')
            print(f'\tself._node_connection_dict : {self._node_connection_dict}')

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
