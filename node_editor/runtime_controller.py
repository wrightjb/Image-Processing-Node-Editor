#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import time

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


def _sleep_for_render_cap(frame_started_at, render_fps):
    if render_fps is None or render_fps <= 0:
        return

    target_frame_seconds = 1.0 / float(render_fps)
    elapsed_seconds = time.perf_counter() - frame_started_at
    remaining_seconds = target_frame_seconds - elapsed_seconds
    if remaining_seconds > 0:
        time.sleep(remaining_seconds)


def _run_capped_render_loop(render_fps, frame_callback=None):
    while dpg.is_dearpygui_running():
        frame_started_at = time.perf_counter()
        if frame_callback is not None:
            frame_callback()
        dpg.render_dearpygui_frame()
        _sleep_for_render_cap(frame_started_at, render_fps)


def run_editor_main_loop(
    node_editor,
    runtime,
    unuse_async_draw,
    render_fps=None,
):
    print('**** Start Main Event Loop ********')

    if not unuse_async_draw:
        event_loop = asyncio.get_event_loop()
        event_loop.run_in_executor(None, async_runtime_worker, node_editor, runtime)
        if render_fps is not None and render_fps > 0:
            _run_capped_render_loop(render_fps)
        else:
            dpg.start_dearpygui()
        return event_loop

    def step_runtime():
        runtime.step(node_editor, mode_async=False)

    _run_capped_render_loop(render_fps, frame_callback=step_runtime)

    return None
