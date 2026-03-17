#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import argparse
from collections import OrderedDict
import os

import dearpygui.dearpygui as dpg

try:
    from .node_editor.node_editor import DpgNodeEditor
    from .node_editor.graph_runtime import GraphRuntime, update_node_info
    from .node_editor.app_lifecycle import (
        load_opencv_settings,
        initialize_camera_resources,
        initialize_serial_resources,
        setup_dearpygui,
        shutdown_runtime,
    )
except ImportError:
    from node_editor.node_editor import DpgNodeEditor
    from node_editor.graph_runtime import GraphRuntime, update_node_info
    from node_editor.app_lifecycle import (
        load_opencv_settings,
        initialize_camera_resources,
        initialize_serial_resources,
        setup_dearpygui,
        shutdown_runtime,
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--setting',
        type=str,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                'node_editor/setting/setting.json',
            )
        ),
    )
    parser.add_argument('--unuse_async_draw', action='store_true')
    parser.add_argument('--use_debug_print', action='store_true')
    parser.add_argument(
        '-i',
        '--import_json',
        type=str,
        default=None,
        help='Path to a node editor JSON file to import at startup.',
    )

    return parser.parse_args()


def async_main(node_editor, runtime):
    while not node_editor.get_terminate_flag():
        try:
            runtime.step(node_editor, mode_async=True)
        except Exception as e:
            print('ERROR: async_main loop exception')
            print(f'\terror                : {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            print()
            continue


def main():
    args = get_args()
    setting = args.setting
    unuse_async_draw = args.unuse_async_draw
    use_debug_print = args.use_debug_print
    import_json = args.import_json

    print('**** Load Config ********')
    opencv_setting_dict = load_opencv_settings(setting)

    camera_capture_list = initialize_camera_resources(opencv_setting_dict)
    serial_connection_list = initialize_serial_resources(opencv_setting_dict)

    editor_width = opencv_setting_dict['editor_width']
    editor_height = opencv_setting_dict['editor_height']

    current_path = os.path.dirname(os.path.abspath(__file__))
    setup_dearpygui(
        editor_width,
        editor_height,
        os.path.join(
            current_path,
            'node_editor/font/YasashisaAntiqueFont/07YasashisaAntique.otf',
        ),
    )

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
    node_editor = DpgNodeEditor(
        width=editor_width - 15,
        height=editor_height - 40,
        opencv_setting_dict=opencv_setting_dict,
        menu_dict=menu_dict,
        use_debug_print=use_debug_print,
        node_dir=os.path.join(current_path, 'node'),
    )

    if import_json is not None:
        print('**** Import JSON ********')
        try:
            node_editor.import_setting_file(import_json)
        except Exception as e:
            print('ERROR: failed to import startup JSON file')
            print(f'\tpath                 : {import_json}')
            print(f'\terror                : {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            print()

    dpg.show_viewport()

    runtime = GraphRuntime()

    print('**** Start Main Event Loop ********')
    event_loop = None
    if not unuse_async_draw:
        event_loop = asyncio.get_event_loop()
        event_loop.run_in_executor(None, async_main, node_editor, runtime)
        dpg.start_dearpygui()
    else:
        while dpg.is_dearpygui_running():
            runtime.step(node_editor, mode_async=False)
            dpg.render_dearpygui_frame()

    shutdown_runtime(
        node_editor,
        camera_capture_list,
        serial_connection_list,
        event_loop=event_loop,
    )


if __name__ == '__main__':
    main()
