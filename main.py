#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import json
import asyncio
import argparse
import hashlib
import pickle
from collections import OrderedDict
import os

import cv2
import dearpygui.dearpygui as dpg

try:
    from .node_editor.util import check_camera_connection
    from .node_editor.node_editor import DpgNodeEditor
except ImportError:
    from node_editor.util import check_camera_connection
    from node_editor.node_editor import DpgNodeEditor


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        type=str,
        # get abs
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         'node_editor/setting/setting.json')),
    )
    parser.add_argument("--unuse_async_draw", action="store_true")
    parser.add_argument("--use_debug_print", action="store_true")
    parser.add_argument(
        "-i",
        "--import_json",
        type=str,
        default=None,
        help="Path to a node editor JSON file to import at startup.",
    )

    args = parser.parse_args()

    return args


def async_main(node_editor, use_debug_print=False):
    # 各ノードの処理結果保持用Dict
    node_image_dict = {}
    node_result_dict = {}
    node_cache_dict = {}

    # メインループ
    while not node_editor.get_terminate_flag():
        try:
            update_node_info(
                node_editor,
                node_image_dict,
                node_result_dict,
                node_cache_dict=node_cache_dict,
            )
        except Exception as e:
            print('ERROR: async_main loop exception')
            print(f"\terror                : {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue


def _freeze_cache_value(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    if isinstance(value, bytes):
        return hashlib.sha1(value).hexdigest()

    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)

    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(item) for item in value))

    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _freeze_cache_value(key),
                    _freeze_cache_value(item),
                )
                for key, item in value.items()
            )
        )

    if (
        hasattr(value, 'shape') and
        hasattr(value, 'dtype') and
        hasattr(value, 'tobytes')
    ):
        try:
            return (
                'ndarray',
                tuple(value.shape),
                str(value.dtype),
                hashlib.sha1(value.tobytes()).hexdigest(),
            )
        except TypeError:
            pass

    return repr(value)


def _build_node_signature(
    node_id,
    connection_list,
    node_image_dict,
    node_result_dict,
    node_setting,
):
    upstream_values = []
    for source_tag, _ in connection_list:
        source_node_id_name = ':'.join(source_tag.split(':')[:2])
        upstream_values.append((
            source_tag,
            _freeze_cache_value(node_image_dict.get(source_node_id_name)),
            _freeze_cache_value(node_result_dict.get(source_node_id_name)),
        ))

    signature_payload = {
        'node_id': node_id,
        'connection_list': connection_list,
        'upstream_values': upstream_values,
        'node_setting': _freeze_cache_value(node_setting),
    }
    payload_bytes = pickle.dumps(signature_payload)
    return hashlib.sha1(payload_bytes).hexdigest()


def update_node_info(
    node_editor,
    node_image_dict,
    node_result_dict,
    node_cache_dict=None,
    mode_async=True,
):
    """
    Update all nodes in topological order with optional in-memory caching.

    Cache path (connected nodes):
      1) build a signature from upstream outputs + node settings
      2) if signature matches previous run, reuse cached output and skip update()

    Non-cache path (source nodes without inbound links):
      always run update() so UI/callback-driven state changes are picked up.
    """
    if node_cache_dict is None:
        node_cache_dict = {}

    def _is_valid_connection(connection_info, valid_nodes):
        if len(connection_info) != 2:
            return False

        source_tag, dest_tag = connection_info
        source_node_id_name = ':'.join(source_tag.split(':')[:2])
        dest_node_id_name = ':'.join(dest_tag.split(':')[:2])
        if source_node_id_name not in valid_nodes:
            return False
        if dest_node_id_name not in valid_nodes:
            return False

        return True

    # ノードリスト取得
    node_list = list(node_editor.get_node_list())
    active_node_set = set(node_list)
    # Remove stale outputs for nodes deleted from the editor.
    deleted_image_node_id_name_list = [
        node_id_name for node_id_name in node_image_dict.keys()
        if node_id_name not in active_node_set
    ]
    for deleted_node_id_name in deleted_image_node_id_name_list:
        del node_image_dict[deleted_node_id_name]

    deleted_result_node_id_name_list = [
        node_id_name for node_id_name in node_result_dict.keys()
        if node_id_name not in active_node_set
    ]
    for deleted_node_id_name in deleted_result_node_id_name_list:
        del node_result_dict[deleted_node_id_name]

    # ノード接続情報取得
    sorted_node_connection_dict = node_editor.get_sorted_node_connection()

    # 各ノードの情報をアップデート
    for node_id_name in node_list:
        if node_id_name not in node_image_dict:
            node_image_dict[node_id_name] = None

        node_id, node_name = node_id_name.split(':')

        # Skip nodes that were deleted in GUI callbacks during this update tick.
        if hasattr(node_editor, 'is_node_active'):
            try:
                if not node_editor.is_node_active(node_id_name):
                    continue
            except Exception:
                pass

        connection_list = sorted_node_connection_dict.get(node_id_name, [])
        connection_list = [
            connection_info for connection_info in connection_list
            if _is_valid_connection(connection_info, active_node_set)
        ]

        # ノード名からインスタンスを取得
        node_instance = node_editor.get_node_instance(node_name)
        if node_instance is None:
            node_image_dict[node_id_name] = None
            node_result_dict[node_id_name] = None
            continue
        cache_signature = None
        # Only cache nodes that have inbound links (downstream processors).
        # Source/input nodes must keep running every frame.
        use_cache = len(connection_list) > 0
        node_setting = {}
        if use_cache and hasattr(node_instance, 'get_setting_dict'):
            if mode_async:
                try:
                    node_setting = node_instance.get_setting_dict(node_id)
                except Exception as e:
                    print(
                        'WARNING: failed to read node settings in '
                        f'update_node_info ({node_id_name}) '
                        f'{type(e).__name__}: {e}'
                    )
                    import traceback
                    traceback.print_exc()
                    use_cache = False
            else:
                node_setting = node_instance.get_setting_dict(node_id)

        if use_cache:
            # Cache-hit path: skip expensive update() when nothing relevant changed.
            cache_signature = _build_node_signature(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
                node_setting,
            )
            cached_result = node_cache_dict.get(node_id_name)
            if (
                cached_result is not None and
                cached_result.get('signature') == cache_signature
            ):
                node_image_dict[node_id_name] = copy.deepcopy(
                    cached_result['image']
                )
                node_result_dict[node_id_name] = copy.deepcopy(
                    cached_result['result']
                )
                continue

        # 指定ノードの情報を更新
        if mode_async:
            try:
                image, result = node_instance.update(
                    node_id,
                    connection_list,
                    node_image_dict,
                    node_result_dict,
                )
            except Exception as e:
                print(
                    'WARNING: node update exception '
                    f'({node_id_name}) {type(e).__name__}: {e}'
                )
                import traceback
                traceback.print_exc()
                image, result = None, None
        else:
            image, result = node_instance.update(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
            )
        # Cache-miss path (or source node path): run node update and store outputs.
        node_image_dict[node_id_name] = copy.deepcopy(image)
        node_result_dict[node_id_name] = copy.deepcopy(result)
        if use_cache:
            # Persist latest outputs for the next signature match.
            node_cache_dict[node_id_name] = {
                'signature': cache_signature,
                'image': copy.deepcopy(image),
                'result': copy.deepcopy(result),
            }
        elif node_id_name in node_cache_dict:
            # Ensure source nodes never keep stale cache entries.
            del node_cache_dict[node_id_name]

    # Remove cache entries for nodes that were deleted from the graph.
    deleted_node_id_name_list = [
        node_id_name for node_id_name in node_cache_dict.keys()
        if node_id_name not in node_list
    ]
    for deleted_node_id_name in deleted_node_id_name_list:
        del node_cache_dict[deleted_node_id_name]

    return


def main():

    args = get_args()
    setting = args.setting
    unuse_async_draw = args.unuse_async_draw
    use_debug_print = args.use_debug_print
    import_json = args.import_json

    # 動作設定
    print('**** Load Config ********')
    opencv_setting_dict = None
    with open(setting) as fp:
        opencv_setting_dict = json.load(fp)
    webcam_width = opencv_setting_dict['webcam_width']
    webcam_height = opencv_setting_dict['webcam_height']

    # 接続カメラチェック
    print('**** Check Camera Connection ********')
    device_no_list = check_camera_connection()
    camera_capture_list = []
    for device_no in device_no_list:
        video_capture = cv2.VideoCapture(device_no)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        camera_capture_list.append(video_capture)

    # カメラ設定保持
    opencv_setting_dict['device_no_list'] = device_no_list
    opencv_setting_dict['camera_capture_list'] = camera_capture_list

    # DearPyGui準備(コンテキスト生成、セットアップ、ビューポート生成)
    editor_width = opencv_setting_dict['editor_width']
    editor_height = opencv_setting_dict['editor_height']

    # Serial接続デバイスチェック
    serial_device_no_list = []
    serial_connection_list = []
    use_serial = opencv_setting_dict['use_serial']
    if use_serial == True:
        import serial
        try:
            from .node_editor.util import check_serial_connection
        except:
            from node_editor.util import check_serial_connection
        print('**** Check Serial Device Connection ********')
        serial_device_no_list = check_serial_connection()
        for serial_device_no in serial_device_no_list:
            ser = serial.Serial(serial_device_no,115200)
            serial_connection_list.append(ser)
        
    # Serial接続デバイス設定保持
    opencv_setting_dict['serial_device_no_list'] = serial_device_no_list
    opencv_setting_dict['serial_connection_list'] = serial_connection_list

    print('**** DearPyGui Setup ********')
    dpg.create_context()
    dpg.setup_dearpygui()
    dpg.create_viewport(
        title="Image Processing Node Editor",
        width=editor_width,
        height=editor_height,
    )

    # デフォルトフォント変更
    # このファイルのパスを取得
    current_path = os.path.dirname(os.path.abspath(__file__))
    with dpg.font_registry():
        with dpg.font(
                current_path +
                '/node_editor/font/YasashisaAntiqueFont/07YasashisaAntique.otf',
                16,
        ) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
    dpg.bind_font(default_font)

    # ノードエディター生成
    print('**** Create NodeEditor ********')
    menu_dict = OrderedDict({
        'InputNode': 'input_node',
        'ProcessNode': 'process_node',
        # 'DeepLearningNode': 'deep_learning_node',
        'AnalysisNode': 'analysis_node',
        'DrawNode': 'draw_node',
        'OtherNode': 'other_node',
        # 'PreviewReleaseNode': 'preview_release_node'
    })
    # print
    node_editor = DpgNodeEditor(
        width=editor_width - 15,
        height=editor_height - 40,
        opencv_setting_dict=opencv_setting_dict,
        menu_dict=menu_dict,
        use_debug_print=use_debug_print,
        node_dir=current_path + '/node',
    )

    if import_json is not None:
        print('**** Import JSON ********')
        node_editor.import_setting_file(import_json)

    # ビューポート表示
    dpg.show_viewport()

    # メインループ
    print('**** Start Main Event Loop ********')
    if not unuse_async_draw:
        event_loop = asyncio.get_event_loop()
        event_loop.run_in_executor(None, async_main, node_editor, use_debug_print)
        dpg.start_dearpygui()
    else:
        # 各ノードの処理結果保持用Dict
        node_image_dict = {}
        node_result_dict = {}
        node_cache_dict = {}
        while dpg.is_dearpygui_running():
            update_node_info(
                node_editor,
                node_image_dict,
                node_result_dict,
                node_cache_dict=node_cache_dict,
                mode_async=False,
            )
            dpg.render_dearpygui_frame()

    # 終了処理
    print('**** Terminate process ********')
    # 各ノードの終了処理
    print('**** Close All Node ********')
    node_list = node_editor.get_node_list()
    for node_id_name in node_list:
        node_id, node_name = node_id_name.split(':')
        node_instance = node_editor.get_node_instance(node_name)
        node_instance.close(node_id)
    # OpenCV関連終了処理
    print('**** Release All VideoCapture ********')
    for camera_capture in camera_capture_list:
        camera_capture.release()
    # イベントループの停止
    print('**** Stop Event Loop ********')
    node_editor.set_terminate_flag()
    event_loop.stop()
    # DearPyGuiコンテキスト破棄
    print('**** Destroy DearPyGui Context ********')
    dpg.destroy_context()


if __name__ == '__main__':
    main()
