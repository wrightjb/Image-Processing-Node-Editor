import numpy as np

from node.base import declarative_node_base as base_module
import node.process_node.node_blur as blur_module
import node.process_node.node_brightness as brightness_module
import node.process_node.node_contrast as contrast_module

BlurNode = blur_module.Node
BrightnessNode = brightness_module.Node
ContrastNode = contrast_module.Node


class DpgStub:
    @staticmethod
    def get_item_pos(tag):
        assert tag
        return [10, 20]


def _prepare_node(node):
    node._opencv_setting_dict = {
        'process_width': 8,
        'process_height': 8,
        'use_pref_counter': False,
    }


def test_blur_node_update_with_parameter_link(monkeypatch):
    node = BlurNode()
    _prepare_node(node)

    values = {
        '1:IntValue:Int:Output01Value': 5,
        '2:Blur:Int:Input02Value': 1,
    }
    written = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: written.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: ('texture', frame.shape, w, h))
    monkeypatch.setattr(blur_module.cv2, 'blur', lambda f, k: f, raising=False)

    frame = np.full((8, 8, 3), 127, dtype=np.uint8)
    out_frame, result = node.update(
        2,
        [
            ['1:ImageSource:Image:Output01', '2:Blur:Image:Input01'],
            ['1:IntValue:Int:Output01', '2:Blur:Int:Input02'],
        ],
        {'1:ImageSource': frame},
        {},
    )

    assert result is None
    assert out_frame.shape == frame.shape
    assert written['2:Blur:Int:Input02Value'] == 5
    assert written['2:Blur:Image:Output01Value'][0] == 'texture'


def test_brightness_node_get_set_settings(monkeypatch):
    node = BrightnessNode()

    values = {'3:Brightness:Int:Input02Value': 21}
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'dpg', DpgStub())

    setting = node.get_setting_dict(3)

    assert setting['ver'] == node._ver
    assert setting['pos'] == [10, 20]
    assert setting['3:Brightness:Int:Input02Value'] == 21

    node.set_setting_dict(3, {'3:Brightness:Int:Input02Value': 44})
    assert writes['3:Brightness:Int:Input02Value'] == 44


def test_contrast_node_clamps_and_rounds_linked_float(monkeypatch):
    node = ContrastNode()
    _prepare_node(node)

    values = {
        '9:FloatValue:Float:Output01Value': 5.9999,
        '7:Contrast:Float:Input02Value': 0.3,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(contrast_module.cv2, 'convertScaleAbs', lambda f, alpha, beta: f, raising=False)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    node.update(
        7,
        [
            ['4:ImageSource:Image:Output01', '7:Contrast:Image:Input01'],
            ['9:FloatValue:Float:Output01', '7:Contrast:Float:Input02'],
        ],
        {'4:ImageSource': frame},
        {},
    )

    assert writes['7:Contrast:Float:Input02Value'] == 4.0
