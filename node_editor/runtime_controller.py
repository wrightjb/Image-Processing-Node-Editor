#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio

import dearpygui.dearpygui as dpg


def async_runtime_worker(node_editor, runtime):
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


def run_editor_main_loop(node_editor, runtime, unuse_async_draw):
    print('**** Start Main Event Loop ********')

    if not unuse_async_draw:
        event_loop = asyncio.get_event_loop()
        event_loop.run_in_executor(None, async_runtime_worker, node_editor, runtime)
        dpg.start_dearpygui()
        return event_loop

    while dpg.is_dearpygui_running():
        runtime.step(node_editor, mode_async=False)
        dpg.render_dearpygui_frame()

    return None
