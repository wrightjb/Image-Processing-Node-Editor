#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

import dearpygui.dearpygui as dpg

try:
    from .node_editor.graph_runtime import GraphRuntime, update_node_info
    from .node_editor.runtime_controller import run_editor_main_loop
    from .node_editor.app_lifecycle import (
        load_opencv_settings,
        initialize_camera_resources,
        initialize_serial_resources,
        setup_dearpygui,
        shutdown_runtime,
    )
    from .node_editor.editor_factory import (
        create_node_editor,
        import_startup_json,
    )
except ImportError:
    from node_editor.graph_runtime import GraphRuntime, update_node_info
    from node_editor.runtime_controller import run_editor_main_loop
    from node_editor.app_lifecycle import (
        load_opencv_settings,
        initialize_camera_resources,
        initialize_serial_resources,
        setup_dearpygui,
        shutdown_runtime,
    )
    from node_editor.editor_factory import (
        create_node_editor,
        import_startup_json,
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
    node_editor = create_node_editor(
        current_path=current_path,
        editor_width=editor_width,
        editor_height=editor_height,
        opencv_setting_dict=opencv_setting_dict,
        use_debug_print=use_debug_print,
    )

    import_startup_json(node_editor, import_json)

    dpg.show_viewport()

    runtime = GraphRuntime()

    event_loop = run_editor_main_loop(
        node_editor,
        runtime,
        unuse_async_draw=unuse_async_draw,
    )

    shutdown_runtime(
        node_editor,
        camera_capture_list,
        serial_connection_list,
        event_loop=event_loop,
    )


if __name__ == '__main__':
    main()
