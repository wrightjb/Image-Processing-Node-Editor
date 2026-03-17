import sys
from unittest.mock import Mock

from node_editor import app_lifecycle


def test_initialize_camera_resources_sets_settings(monkeypatch):
    fake_cv2 = Mock()
    fake_capture_a = Mock()
    fake_capture_b = Mock()
    fake_cv2.VideoCapture.side_effect = [fake_capture_a, fake_capture_b]
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4

    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)
    monkeypatch.setattr(app_lifecycle, 'check_camera_connection', lambda: [0, 2])

    settings = {'webcam_width': 320, 'webcam_height': 240}
    cameras = app_lifecycle.initialize_camera_resources(settings)

    assert settings['device_no_list'] == [0, 2]
    assert settings['camera_capture_list'] == cameras
    assert len(cameras) == 2
    assert fake_cv2.VideoCapture.call_count == 2


def test_initialize_serial_resources_sets_settings(monkeypatch):
    fake_serial_module = Mock()
    fake_serial_module.Serial.side_effect = ['serA', 'serB']

    monkeypatch.setattr(app_lifecycle, 'check_serial_connection', lambda: ['COM1', 'COM2'])
    monkeypatch.setitem(sys.modules, 'serial', fake_serial_module)

    settings = {'use_serial': True}
    serial_connections = app_lifecycle.initialize_serial_resources(settings)

    assert settings['serial_device_no_list'] == ['COM1', 'COM2']
    assert settings['serial_connection_list'] == serial_connections
    assert serial_connections == ['serA', 'serB']


def test_shutdown_runtime_closes_resources(monkeypatch):
    fake_dpg = Mock()
    monkeypatch.setattr(app_lifecycle, 'dpg', fake_dpg)

    node = Mock()
    editor = Mock()
    editor.get_node_list.return_value = ['1:NodeA']
    editor.get_node_instance.return_value = node

    camera = Mock()
    serial_connection = Mock()
    loop = Mock()

    app_lifecycle.shutdown_runtime(
        editor,
        [camera],
        [serial_connection],
        event_loop=loop,
    )

    node.close.assert_called_once_with('1')
    camera.release.assert_called_once()
    serial_connection.close.assert_called_once()
    editor.set_terminate_flag.assert_called_once()
    loop.stop.assert_called_once()
    fake_dpg.destroy_context.assert_called_once()
