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
        self.mdl_init(opencv_setting_dict, use_debug_print)
        self.vw_init(width, height, pos, node_dir, menu_dict)
        self.cntrl_init()

    # -------------------------------------------------------------------------
    # Model functions
    def mdl_init(self, opencv_setting_dict=None, use_debug_print=False):
        self._node_id = 0
        self._node_instance_list = {}
        self._node_list = []
        self._node_link_list = []
        self._node_connection_dict = OrderedDict([])
        self._use_debug_print = use_debug_print
        self._terminate_flag = False
        self._opencv_setting_dict = opencv_setting_dict

    def mdl_add_node(self, node_tag):
        self._node_id += 1
        node_instance = self._node_instance_list[node_tag]
        return self._node_id, node_instance

    def mdl_add_link(self, source, destination):
        # 型が一致するもののみ処理
        source_type = source.split(':')[2]
        destination_type = destination.split(':')[2]
        if source_type != destination_type:
            return False

        # 入力端子に複数接続しようとしていないかチェック
        for _, dest_node in self._node_link_list:
            if destination == dest_node:
                return False

        self._node_link_list.append([source, destination])
        return True

    def mdl_get_export_settings(self):
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

    def mdl_delete_node(self, node_id_name):
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
        self.mdl_sort_node_graph()

    def mdl_delete_link(self, link):
        self._node_link_list.remove(link)
        self.mdl_sort_node_graph()

    def mdl_sort_node_graph(self):
        node_list = self._node_list
        node_link_list = self._node_link_list

        node_id_dict = OrderedDict({})
        node_connection_dict = OrderedDict({})

        # ノードIDとノード接続を辞書形式で整理
        for node_link_info in node_link_list:
            source = dpg.get_item_alias(node_link_info[0])
            destination = dpg.get_item_alias(node_link_info[1])
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
    def vw_init(self, width, height, pos, node_dir, menu_dict):
        # メニュー項目定義(key：メニュー名、value：ノードのコード格納ディレクトリ名)
        if menu_dict is None:
            menu_dict = OrderedDict({
                'Input Node': 'input_node',
                'Process Node': 'process_node',
                'Output Node': 'output_node'
            })
        self.vw_create_file_dialogs(height)
        self.vw_create_main_window(width, height, pos, node_dir, menu_dict)

    def vw_create_file_dialogs(self, height):
        datetime_now = datetime.datetime.now()
        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(height / 2),
                default_filename=datetime_now.strftime('%Y%m%d'),
                callback=self.cntrl_file_export,
                id='file_export',
        ):
            dpg.add_file_extension('.json')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                modal=True,
                height=int(height / 2),
                callback=self.cntrl_file_import,
                id='file_import',
        ):
            dpg.add_file_extension('.json')
            dpg.add_file_extension('', color=(150, 255, 150, 255))

    def vw_create_main_window(self, width, height, pos, node_dir, menu_dict):
        with dpg.window(
                tag=self._window_tag,
                label=self._node_editor_label,
                width=width,
                height=height,
                pos=pos,
                menubar=True,
                on_close=self.cntrl_close_window,
        ):
            with dpg.menu_bar(label='MenuBar'):
                with dpg.menu(label='File'):
                    dpg.add_menu_item(
                        tag='Menu_File_Export',
                        label='Export',
                        callback=self.cntrl_file_export_menu,
                        user_data='Menu_File_Export',
                    )
                    dpg.add_menu_item(
                        tag='Menu_File_Import',
                        label='Import',
                        callback=self.cntrl_file_import_menu,
                        user_data='Menu_File_Import',
                    )
                self.vw_create_node_menus(node_dir, menu_dict)
            with dpg.node_editor(
                    tag=self._node_editor_tag,
                    callback=self.cntrl_link,
                    minimap=True,
                    minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            ):
                pass
        dpg.set_primary_window(self._window_tag, True)

    def vw_create_node_menus(self, node_dir, menu_dict):
        for menu_label, sub_dir in menu_dict.items():
            with dpg.menu(label=menu_label):
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
                    dpg.add_menu_item(
                        tag='Menu_' + node.node_tag,
                        label=node.node_label,
                        callback=self.cntrl_add_node,
                        user_data=node.node_tag,
                    )
                    self._node_instance_list[node.node_tag] = node

    def vw_add_node(self, node_tag, new_id, pos):
        node = self._node_instance_list[node_tag]
        tag_name = node.add_node(
            self._node_editor_tag,
            new_id,
            pos=pos,
            opencv_setting_dict=self._opencv_setting_dict,
        )
        self._node_list.append(f"{new_id}:{node_tag}")

    def vw_add_link(self, source, destination):
        dpg.add_node_link(source, destination, parent=self._node_editor_tag)

    def vw_delete_item(self, item_id):
        dpg.delete_item(item_id)

    def vw_show_file_export(self):
        dpg.show_item('file_export')

    def vw_show_file_import(self):
        dpg.show_item('file_import')

    # -------------------------------------------------------------------------
    # Controller functions
    def cntrl_init(self):
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self.cntrl_save_last_pos)
            dpg.add_key_press_handler(
                dpg.mvKey_Delete,
                callback=self.cntrl_delete_selected,
            )

    def cntrl_add_node(self, sender, data, user_data):
        new_id, node_instance = self.mdl_add_node(user_data)
        pos = [0, 0]
        if self._last_pos is not None:
            pos = [self._last_pos[0] + 30, self._last_pos[1] + 30]
        self.vw_add_node(user_data, new_id, pos)

        if self._use_debug_print:
            print('**** cntrl_add_node ****')
            print(f'    Node ID         : {self._node_id}')
            print(f'    sender          : {sender}')
            print(f'    data            : {data}')
            print(f'    user_data       : {user_data}')
            print(f'    self._node_list : {", ".join(self._node_list)}')
            print()

    def cntrl_link(self, sender, data):
        source = dpg.get_item_alias(data[0])
        destination = dpg.get_item_alias(data[1])

        if self.mdl_add_link(source, destination):
            self.vw_add_link(source, destination)

        self.mdl_sort_node_graph()

        if self._use_debug_print:
            print('**** cntrl_link ****')
            print(f'    sender                     : {sender}')
            print(f'    data                       : {data}')
            print(f'    self._node_list            : {self._node_list}')
            print(f'    self._node_link_list       : {self._node_link_list}')
            print(f'    self._node_connection_dict : {self._node_connection_dict}')
            print()

    def cntrl_close_window(self, sender):
        self.vw_delete_item(sender)

    def cntrl_file_export(self, sender, data):
        setting_dict = self.mdl_get_export_settings()
        with open(data['file_path_name'], 'w') as fp:
            json.dump(setting_dict, fp, indent=4)

        if self._use_debug_print:
            print('**** cntrl_file_export ****')
            print(f'    sender          : {sender}')
            print(f'    data            : {data}')
            print(f'    setting_dict    : {setting_dict}')
            print()

    def cntrl_file_export_menu(self, sender, data, user_data):
        self.vw_show_file_export()

    def cntrl_file_import_menu(self, sender, data, user_data):
        self.vw_show_file_import()

    def cntrl_file_import(self, sender, data):
        if data['file_name'] == '.':
            return
        with open(data['file_path_name']) as fp:
            setting_dict = json.load(fp)

        id_map = {}
        new_node_list = []
        new_link_list = []

        for node_id_name in setting_dict['node_list']:
            old_id, node_name = node_id_name.split(':')
            node = self._node_instance_list[node_name]

            new_id, _ = self.mdl_add_node(node_name)
            id_map[old_id] = str(new_id)

            ver = setting_dict[node_id_name]['setting']['ver']
            if ver != node._ver:
                warning_node_name = setting_dict[node_id_name]['name']
                print(f'WARNING : {warning_node_name} is different version')
                print(f'                     Load Version -> {ver}')
                print(f'                     Code Version -> {node._ver}\n')

            pos = setting_dict[node_id_name]['setting']['pos']
            self.vw_add_node(node_name, new_id, pos)

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
                self.vw_add_link(new_source, new_destination)
                new_link_list.append([new_source, new_destination])

        self._node_link_list.extend(new_link_list)
        self.mdl_sort_node_graph()

        if self._use_debug_print:
            print('**** cntrl_file_import ****')
            print(f'    sender          : {sender}')
            print(f'    data            : {data}')
            print(f'    setting_dict    : {setting_dict}')
            print()

    def cntrl_save_last_pos(self, sender, data):
        selected_nodes = dpg.get_selected_nodes(self._node_editor_tag)
        if selected_nodes:
            self._last_pos = dpg.get_item_pos(selected_nodes[0])

    def cntrl_delete_selected(self, sender, data):
        selected_nodes = dpg.get_selected_nodes(self._node_editor_tag)
        for node_id in selected_nodes:
            node_id_name = dpg.get_item_alias(node_id)
            self.mdl_delete_node(node_id_name)
            self.vw_delete_item(node_id)

        selected_links = dpg.get_selected_links(self._node_editor_tag)
        for link_id in selected_links:
            config = dpg.get_item_configuration(link_id)
            link = [
                dpg.get_item_alias(config['attr_1']),
                dpg.get_item_alias(config['attr_2'])
            ]
            self.mdl_delete_link(link)
            self.vw_delete_item(link_id)

        if self._use_debug_print:
            print('**** cntrl_delete_selected ****')
            print(f'    self._node_list            : {self._node_list}')
            print(f'    self._node_link_list       : {self._node_link_list}')
            print(f'    self._node_connection_dict : {self._node_connection_dict}')

    # Public functions
    def get_node_list(self):
        return self._node_list

    def get_sorted_node_connection(self):
        return self._node_connection_dict

    def get_node_instance(self, node_name):
        return self._node_instance_list.get(node_name, None)

    def set_terminate_flag(self, flag=True):
        self._terminate_flag = flag

    def get_terminate_flag(self):
        return self._terminate_flag
