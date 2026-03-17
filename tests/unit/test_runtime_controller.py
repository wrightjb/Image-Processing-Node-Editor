from unittest.mock import Mock

from node_editor import runtime_controller


def test_async_runtime_worker_continues_after_step_exception():
    node_editor = Mock()
    node_editor.get_terminate_flag.side_effect = [False, False, True]

    runtime = Mock()
    runtime.step.side_effect = [RuntimeError('boom'), None]

    runtime_controller.async_runtime_worker(node_editor, runtime)

    assert runtime.step.call_count == 2


def test_run_editor_main_loop_async_mode(monkeypatch):
    fake_loop = Mock()
    fake_asyncio = Mock()
    fake_asyncio.get_event_loop.return_value = fake_loop
    fake_dpg = Mock()

    monkeypatch.setattr(runtime_controller, 'asyncio', fake_asyncio)
    monkeypatch.setattr(runtime_controller, 'dpg', fake_dpg)

    node_editor = Mock()
    runtime = Mock()

    event_loop = runtime_controller.run_editor_main_loop(
        node_editor,
        runtime,
        unuse_async_draw=False,
    )

    fake_loop.run_in_executor.assert_called_once_with(
        None,
        runtime_controller.async_runtime_worker,
        node_editor,
        runtime,
    )
    fake_dpg.start_dearpygui.assert_called_once()
    assert event_loop is fake_loop


def test_run_editor_main_loop_sync_mode(monkeypatch):
    fake_dpg = Mock()
    fake_dpg.is_dearpygui_running.side_effect = [True, True, False]
    monkeypatch.setattr(runtime_controller, 'dpg', fake_dpg)

    node_editor = Mock()
    runtime = Mock()

    event_loop = runtime_controller.run_editor_main_loop(
        node_editor,
        runtime,
        unuse_async_draw=True,
    )

    assert event_loop is None
    assert runtime.step.call_count == 2
    runtime.step.assert_called_with(node_editor, mode_async=False)
    assert fake_dpg.render_dearpygui_frame.call_count == 2
