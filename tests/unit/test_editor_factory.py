from collections import OrderedDict
from unittest.mock import Mock

from node_editor import editor_factory


def test_build_default_menu_dict_returns_expected_order():
    menu_dict = editor_factory.build_default_menu_dict()

    assert isinstance(menu_dict, OrderedDict)
    assert list(menu_dict.keys()) == [
        'InputNode',
        'ProcessNode',
        'AnalysisNode',
        'DrawNode',
        'OtherNode',
    ]


def test_create_node_editor_uses_default_menu(monkeypatch):
    fake_editor_class = Mock()
    fake_editor_instance = Mock()
    fake_editor_class.return_value = fake_editor_instance
    monkeypatch.setattr(editor_factory, 'DpgNodeEditor', fake_editor_class)

    editor = editor_factory.create_node_editor(
        current_path='/tmp/project',
        editor_width=1280,
        editor_height=720,
        opencv_setting_dict={'a': 1},
        use_debug_print=True,
    )

    assert editor is fake_editor_instance
    call_kwargs = fake_editor_class.call_args.kwargs
    assert call_kwargs['width'] == 1265
    assert call_kwargs['height'] == 680
    assert call_kwargs['node_dir'] == '/tmp/project/node'
    assert isinstance(call_kwargs['menu_dict'], OrderedDict)


def test_import_startup_json_calls_import_when_provided():
    node_editor = Mock()

    editor_factory.import_startup_json(node_editor, '/tmp/sample.json')

    node_editor.import_setting_file.assert_called_once_with('/tmp/sample.json')


def test_import_startup_json_skips_when_none():
    node_editor = Mock()

    editor_factory.import_startup_json(node_editor, None)

    node_editor.import_setting_file.assert_not_called()
