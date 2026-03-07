import pytest
import numpy as np

from node.base import declarative_node_base as base_module
import node.process_node.node_blur as blur_module
import node.process_node.node_brightness as brightness_module
import node.process_node.node_contrast as contrast_module
import node.process_node.node_flip as flip_module
import node.process_node.node_threshold as threshold_module
import node.process_node.node_canny as canny_module
import node.process_node.node_resize as resize_module
import node.process_node.node_gaussian_blur as gaussian_blur_module
import node.process_node.node_crop as crop_module
import node.process_node.node_simple_filter as simple_filter_module
import node.process_node.node_curves as curves_module

BlurNode = blur_module.Node
BrightnessNode = brightness_module.Node
ContrastNode = contrast_module.Node
FlipNode = flip_module.Node
ThresholdNode = threshold_module.Node
CannyNode = canny_module.Node
ResizeNode = resize_module.Node
GaussianBlurNode = gaussian_blur_module.Node
CropNode = crop_module.Node
SimpleFilterNode = simple_filter_module.Node
CurvesNode = curves_module.Node


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
    monkeypatch.setattr(canny_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
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


def test_flip_node_link_updates_only_target_checkbox(monkeypatch):
    node = FlipNode()
    _prepare_node(node)

    values = {
        '11:BoolSwitch:Text:Output01Value': True,
        '12:Flip:Text:Input02Value': False,
        '12:Flip:Text:Input03Value': False,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(flip_module.cv2, 'flip', lambda f, c: f, raising=False)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    node.update(
        12,
        [
            ['4:ImageSource:Image:Output01', '12:Flip:Image:Input01'],
            ['11:BoolSwitch:Text:Output01', '12:Flip:Text:Input03'],
        ],
        {'4:ImageSource': frame},
        {},
    )

    assert writes['12:Flip:Text:Input03Value'] is True
    assert '12:Flip:Text:Input02Value' not in writes


def test_threshold_node_uses_default_on_missing_combo_value(monkeypatch):
    node = ThresholdNode()
    _prepare_node(node)

    values = {
        '22:Threshold:Text:Input02Value': None,
        '22:Threshold:Int:Input03Value': None,
    }

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: None)
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)

    calls = {}

    def _threshold_stub(image, binary_threshold, max_value, threshold_type):
        calls['binary_threshold'] = binary_threshold
        calls['threshold_type'] = threshold_type
        return None, image

    monkeypatch.setattr(threshold_module.cv2, 'cvtColor', lambda image, code: image, raising=False)
    monkeypatch.setattr(threshold_module.cv2, 'threshold', _threshold_stub, raising=False)
    monkeypatch.setattr(threshold_module.cv2, 'COLOR_BGR2GRAY', 0, raising=False)
    monkeypatch.setattr(threshold_module.cv2, 'COLOR_GRAY2BGR', 1, raising=False)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    out_frame, result = node.update(
        22,
        [
            ['7:ImageSource:Image:Output01', '22:Threshold:Image:Input01'],
        ],
        {'7:ImageSource': frame},
        {},
    )

    assert result is None
    assert out_frame.shape == frame.shape
    assert calls['binary_threshold'] == 127
    assert calls['threshold_type'] == threshold_module.Node._threshold_types['THRESH_BINARY']


def test_canny_node_normalizes_crossed_thresholds(monkeypatch):
    node = CannyNode()
    _prepare_node(node)

    values = {
        '31:Canny:Int:Input02Value': 210,
        '31:Canny:Int:Input03Value': 100,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(canny_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(canny_module.cv2, 'cvtColor', lambda image, code: image, raising=False)
    monkeypatch.setattr(canny_module.cv2, 'Canny', lambda image, min_v, max_v: image, raising=False)
    monkeypatch.setattr(canny_module.cv2, 'COLOR_BGR2GRAY', 0, raising=False)
    monkeypatch.setattr(canny_module.cv2, 'COLOR_GRAY2BGR', 1, raising=False)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    node.update(
        31,
        [
            ['1:ImageSource:Image:Output01', '31:Canny:Image:Input01'],
        ],
        {'1:ImageSource': frame},
        {},
    )

    assert writes['31:Canny:Int:Input02Value'] == 99
    assert writes['31:Canny:Int:Input03Value'] == 211


def test_resize_node_fallbacks_invalid_values(monkeypatch):
    node = ResizeNode()
    _prepare_node(node)

    values = {
        '42:Resize:Int:Input02Value': 'bad',
        '42:Resize:Int:Input03Value': -5,
        '42:Resize:Text:Input04Value': 'UNKNOWN',
    }
    writes = {}
    calls = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(resize_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)

    def _resize_stub(image, dsize, interpolation):
        calls['dsize'] = dsize
        calls['interpolation'] = interpolation
        return image

    monkeypatch.setattr(resize_module.cv2, 'resize', _resize_stub, raising=False)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    node.update(
        42,
        [
            ['5:ImageSource:Image:Output01', '42:Resize:Image:Input01'],
        ],
        {'5:ImageSource': frame},
        {},
    )

    assert writes['42:Resize:Int:Input02Value'] == 960
    assert writes['42:Resize:Int:Input03Value'] == 1
    assert writes['42:Resize:Text:Input04Value'] == 'INTER_LINEAR'
    assert calls['dsize'] == (960, 1)


def test_gaussian_blur_auto_sigma_sets_zero(monkeypatch):
    node = GaussianBlurNode()
    _prepare_node(node)

    values = {
        '52:GaussianBlur:Int:Input02Value': 4,
        '52:GaussianBlur:Int:Input04Value': True,
        '52:GaussianBlur:Float:Input03Value': 3.2,
    }
    calls = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: None)
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)

    def _gaussian_stub(image, kernel, sigma):
        calls['kernel'] = kernel
        calls['sigma'] = sigma
        return image

    monkeypatch.setattr(gaussian_blur_module.cv2, 'GaussianBlur', _gaussian_stub, raising=False)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    node.update(
        52,
        [
            ['9:ImageSource:Image:Output01', '52:GaussianBlur:Image:Input01'],
        ],
        {'9:ImageSource': frame},
        {},
    )

    assert calls['kernel'] == (5, 5)
    assert calls['sigma'] == 0.0


def test_crop_node_normalizes_crossed_bounds(monkeypatch):
    node = CropNode()
    _prepare_node(node)

    values = {
        '61:Crop:Float:Input02Value': 0.9,
        '61:Crop:Float:Input03Value': 0.2,
        '61:Crop:Float:Input04Value': 0.8,
        '61:Crop:Float:Input05Value': 0.1,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(crop_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    out_frame, result = node.update(
        61,
        [
            ['6:ImageSource:Image:Output01', '61:Crop:Image:Input01'],
        ],
        {'6:ImageSource': frame},
        {},
    )

    assert result is None
    assert out_frame.shape[0] > 0
    assert out_frame.shape[1] > 0
    assert writes['61:Crop:Float:Input02Value'] == 0.19
    assert writes['61:Crop:Float:Input03Value'] == 0.91
    assert writes['61:Crop:Float:Input04Value'] == pytest.approx(0.09)
    assert writes['61:Crop:Float:Input05Value'] == pytest.approx(0.81)


def test_simple_filter_settings_include_all_kernel_values(monkeypatch):
    node = SimpleFilterNode()

    values = {
        '70:SimpleFilter:Float:Input02Value': 0.0,
        '70:SimpleFilter:Float:Input03Value': 0.1,
        '70:SimpleFilter:Float:Input04Value': 0.2,
        '70:SimpleFilter:Float:Input05Value': 0.3,
        '70:SimpleFilter:Float:Input06Value': 0.4,
        '70:SimpleFilter:Float:Input07Value': 0.5,
        '70:SimpleFilter:Float:Input08Value': 0.6,
        '70:SimpleFilter:Float:Input09Value': 0.7,
        '70:SimpleFilter:Float:Input10Value': 0.8,
        '70:SimpleFilter:Float:Input11Value': 1.9,
    }

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg', DpgStub())

    setting = node.get_setting_dict(70)

    for index in range(2, 12):
        tag = f'70:SimpleFilter:Float:Input{index:02d}Value'
        assert tag in setting


def test_simple_filter_linked_k_value_uses_k_range(monkeypatch):
    node = SimpleFilterNode()
    _prepare_node(node)

    values = {
        '88:FloatValue:Float:Output01Value': 12.0,
        '80:SimpleFilter:Float:Input02Value': 0.0,
        '80:SimpleFilter:Float:Input03Value': 0.0,
        '80:SimpleFilter:Float:Input04Value': 0.0,
        '80:SimpleFilter:Float:Input05Value': 0.0,
        '80:SimpleFilter:Float:Input06Value': 1.0,
        '80:SimpleFilter:Float:Input07Value': 0.0,
        '80:SimpleFilter:Float:Input08Value': 0.0,
        '80:SimpleFilter:Float:Input09Value': 0.0,
        '80:SimpleFilter:Float:Input10Value': 0.0,
        '80:SimpleFilter:Float:Input11Value': 1.0,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(simple_filter_module.cv2, 'filter2D', lambda image, d, k: image, raising=False)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    node.update(
        80,
        [
            ['3:ImageSource:Image:Output01', '80:SimpleFilter:Image:Input01'],
            ['88:FloatValue:Float:Output01', '80:SimpleFilter:Float:Input11'],
        ],
        {'3:ImageSource': frame},
        {},
    )

    assert writes['80:SimpleFilter:Float:Input11Value'] == 10.0


def test_curves_node_uses_custom_points_in_process(monkeypatch):
    node = CurvesNode()
    _prepare_node(node)

    monkeypatch.setattr(curves_module.Node, '_get_drag_points', lambda self, node_id: [[0, 0], [128, 200], [255, 255]])
    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: None)
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: None)
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)

    captured = {}

    def _lut_stub(image, points):
        captured['points'] = points
        return image

    monkeypatch.setattr(curves_module, 'image_process', _lut_stub)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    out_frame, result = node.update(
        91,
        [
            ['3:ImageSource:Image:Output01', '91:Curves:Image:Input01'],
        ],
        {'3:ImageSource': frame},
        {},
    )

    assert result is None
    assert out_frame.shape == frame.shape
    assert captured['points'] == [[0, 0], [128, 200], [255, 255]]


def test_curves_node_settings_include_points(monkeypatch):
    node = CurvesNode()

    monkeypatch.setattr(base_module, 'dpg', DpgStub())
    monkeypatch.setattr(curves_module.Node, '_get_drag_points', lambda self, node_id: [[0, 0], [64, 80], [255, 255]])

    setting = node.get_setting_dict(92)

    assert setting['ver'] == node._ver
    assert setting['pos'] == [10, 20]
    assert setting['points'] == [[0, 0], [64, 80], [255, 255]]
