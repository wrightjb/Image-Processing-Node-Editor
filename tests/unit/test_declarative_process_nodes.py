import pytest
import numpy as np

from node.port_model import LinkConnectionAdapter, LinkRef, NodeRef, PortRef
from node.base import declarative_node_base as base_module
from node import node_abc as node_abc_module
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
import node.process_node.node_omnidirectional_viewer as omni_module
import node.process_node.node_hue_rotation as hue_rotation_module
import node.process_node.node_hue_saturation_adjustment as hue_saturation_adjustment_module
import node.process_node.node_warmth_tint as warmth_tint_module
import node.process_node.node_rgb_channel_swap as rgb_channel_swap_module

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


def _identity_curve_set():
    return {
        'White': [[0, 0], [255, 255]],
        'Red': [[0, 0], [255, 255]],
        'Green': [[0, 0], [255, 255]],
        'Blue': [[0, 0], [255, 255]],
    }


OmniNode = omni_module.Node
HueRotationNode = hue_rotation_module.Node
HueSaturationAdjustmentNode = hue_saturation_adjustment_module.Node
WarmthTintNode = warmth_tint_module.Node
RGBChannelSwapNode = rgb_channel_swap_module.Node


class DpgStub:
    @staticmethod
    def get_item_pos(tag):
        assert tag
        return [10, 20]


def _port_ref(node_id, node_tag, direction, data_type, port_name):
    prefix = 'Input' if direction == 'Input' else 'Output'
    dpg_tag = f'{node_id}:{node_tag}:{data_type}:{port_name}'
    return PortRef(
        node_ref=NodeRef(str(node_id), node_tag),
        direction=direction,
        data_type=data_type,
        index=int(port_name[len(prefix):]),
        port_name=port_name,
        dpg_tag=dpg_tag,
        value_tag=f'{dpg_tag}Value',
    )


def _link_adapter(source, destination):
    return LinkConnectionAdapter(LinkRef(source, destination))


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
    monkeypatch.setattr(
        base_module,
        'dpg_set_value',
        lambda tag, value: written.setdefault(tag, value),
    )
    monkeypatch.setattr(
        base_module,
        'convert_cv_to_dpg',
        lambda frame, w, h: ('texture', frame.shape, w, h),
    )
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


def test_blur_node_update_with_typed_connection_adapters(monkeypatch):
    node = BlurNode()
    _prepare_node(node)

    image_output = _port_ref(1, 'ImageSource', 'Output', 'Image', 'Output01')
    image_input = _port_ref(2, 'Blur', 'Input', 'Image', 'Input01')
    int_output = _port_ref(1, 'IntValue', 'Output', 'Int', 'Output01')
    int_input = _port_ref(2, 'Blur', 'Input', 'Int', 'Input02')
    values = {
        int_output.value_tag: 7,
        int_input.value_tag: 1,
    }
    written = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(
        base_module,
        'dpg_set_value',
        lambda tag, value: written.setdefault(tag, value),
    )
    monkeypatch.setattr(
        base_module,
        'convert_cv_to_dpg',
        lambda frame, w, h: ('texture', frame.shape, w, h),
    )
    monkeypatch.setattr(blur_module.cv2, 'blur', lambda f, k: f, raising=False)

    frame = np.full((8, 8, 3), 127, dtype=np.uint8)
    out_frame, result = node.update(
        2,
        [
            _link_adapter(image_output, image_input),
            _link_adapter(int_output, int_input),
        ],
        {image_output.node_ref.node_id_name: frame},
        {},
    )

    assert result is None
    assert out_frame.shape == frame.shape
    assert written[int_input.value_tag] == 7
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
    monkeypatch.setattr(
        base_module,
        'dpg_get_value',
        lambda tag: '[[0, 0], [128, 200], [255, 255]]'
        if tag == '91:Curves:CurvePoints:Input02Value'
        else None,
    )
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: None)
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)

    captured = {}

    def _lut_stub(image, points, channel='White'):
        captured['points'] = points
        captured['channel'] = channel
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
    expected_curves = _identity_curve_set()
    expected_curves['White'] = [[0, 0], [128, 200], [255, 255]]
    assert captured['points'] == expected_curves
    assert captured['channel'] == 'White'



def test_curves_node_declares_curve_set_output_port():
    node = CurvesNode()

    port = node._curves_output_port_ref(17)

    assert port.dpg_tag == '17:Curves:CurvePoints:Output03'
    assert port.value_tag == '17:Curves:CurvePoints:Output03Value'

def test_curves_node_settings_include_points(monkeypatch):
    node = CurvesNode()

    monkeypatch.setattr(base_module, 'dpg', DpgStub())
    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: None)
    monkeypatch.setattr(curves_module, 'dpg_set_value', lambda tag, value: None)
    monkeypatch.setattr(curves_module.Node, '_get_drag_points', lambda self, node_id: [[0, 0], [64, 80], [255, 255]])

    setting = node.get_setting_dict(92)

    assert setting['ver'] == node._ver
    assert setting['pos'] == [10, 20]
    assert setting['curves']['White'] == [[0, 0], [64, 80], [255, 255]]


def test_omnidirectional_viewer_reuses_cached_maps(monkeypatch):
    node = OmniNode()
    _prepare_node(node)

    values = {
        '95:OmnidirectionalViewer:Int:Input02Value': 10,
        '95:OmnidirectionalViewer:Int:Input03Value': 20,
        '95:OmnidirectionalViewer:Int:Input04Value': 30,
        '95:OmnidirectionalViewer:Float:Input05Value': 0.2,
    }

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: None)
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)

    calls = {'calc': 0}

    monkeypatch.setattr(omni_module, 'create_rotation_matrix', lambda roll, pitch, yaw: 'rotation')

    def _calc_stub(*args, **kwargs):
        calls['calc'] += 1
        return np.zeros((2, 2)), np.ones((2, 2))

    monkeypatch.setattr(omni_module, 'calculate_phi_and_theta', _calc_stub)
    monkeypatch.setattr(omni_module, 'image_process', lambda image, phi, theta: image)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    node.update(
        95,
        [
            ['1:ImageSource:Image:Output01', '95:OmnidirectionalViewer:Image:Input01'],
        ],
        {'1:ImageSource': frame},
        {},
    )
    node.update(
        95,
        [
            ['1:ImageSource:Image:Output01', '95:OmnidirectionalViewer:Image:Input01'],
        ],
        {'1:ImageSource': frame},
        {},
    )

    assert calls['calc'] == 1


def test_omnidirectional_viewer_close_clears_cache():
    node = OmniNode()
    node._params[96] = [0, 0, 0, 0.0, np.zeros((1, 1)), np.zeros((1, 1))]

    node.close(96)

    assert 96 not in node._params


def test_hue_rotation_node_hsv_mode_and_alpha(monkeypatch):
    node = HueRotationNode()

    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_BGR2HSV', 1, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_HSV2BGR', 2, raising=False)

    def _cvt_color_stub(image, code):
        if code == hue_rotation_module.cv2.COLOR_BGR2HSV:
            hsv = np.zeros_like(image)
            hsv[:, :, 0] = 0
            hsv[:, :, 1] = 255
            hsv[:, :, 2] = 255
            return hsv
        bgr = np.zeros_like(image)
        bgr[:, :, 0] = (image[:, :, 0] * 2) % 255
        bgr[:, :, 1] = 220
        bgr[:, :, 2] = 0
        return bgr

    monkeypatch.setattr(hue_rotation_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'merge', lambda channels: np.stack(channels, axis=-1), raising=False)

    bgr_pixel = np.array([[[0, 0, 255]]], dtype=np.uint8)
    out_bgr, result = node.process(
        bgr_pixel,
        hue_shift_degrees=120,
        color_space='HSV',
    )

    assert result is None
    assert out_bgr.shape == bgr_pixel.shape
    assert out_bgr[0, 0, 0] == 120
    assert out_bgr[0, 0, 1] == 220
    assert out_bgr[0, 0, 2] == 0

    bgra_pixel = np.array([[[0, 0, 255, 77]]], dtype=np.uint8)
    out_bgra, _ = node.process(
        bgra_pixel,
        hue_shift_degrees=120,
        color_space='HSV',
    )

    assert out_bgra[0, 0, 3] == 77


def test_hue_rotation_node_lab_mode_and_invalid_space_fallback(monkeypatch):
    node = HueRotationNode()

    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_BGR2LAB', 3, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_LAB2BGR', 4, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_BGR2HSV', 1, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_HSV2BGR', 2, raising=False)

    calls = {'lab': 0, 'hsv': 0}

    def _cvt_color_stub(image, code):
        if code == hue_rotation_module.cv2.COLOR_BGR2LAB:
            calls['lab'] += 1
            lab = np.zeros_like(image)
            lab[:, :, 0] = 50
            lab[:, :, 1] = 148
            lab[:, :, 2] = 108
            return lab
        if code == hue_rotation_module.cv2.COLOR_LAB2BGR:
            out = np.zeros_like(image)
            out[:, :, 0] = image[:, :, 1]
            out[:, :, 1] = image[:, :, 2]
            out[:, :, 2] = image[:, :, 0]
            return out
        if code == hue_rotation_module.cv2.COLOR_BGR2HSV:
            calls['hsv'] += 1
            return np.zeros_like(image)
        return np.full_like(image, 17)

    monkeypatch.setattr(hue_rotation_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)

    bgr_pixel = np.array([[[0, 0, 255]]], dtype=np.uint8)

    out_lab, _ = node.process(
        bgr_pixel,
        hue_shift_degrees=90,
        color_space='LAB',
    )
    out_fallback, _ = node.process(
        bgr_pixel,
        hue_shift_degrees=90,
        color_space='UNKNOWN',
    )

    assert calls['lab'] == 1
    assert calls['hsv'] == 1
    assert out_lab.shape == bgr_pixel.shape
    assert out_fallback[0, 0, 0] == 17


def test_hue_rotation_node_luv_and_rgb_modes(monkeypatch):
    node = HueRotationNode()

    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_BGR2LUV', 5, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_LUV2BGR', 6, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_BGR2RGB', 7, raising=False)
    monkeypatch.setattr(hue_rotation_module.cv2, 'COLOR_RGB2BGR', 8, raising=False)

    calls = {'luv': 0, 'rgb': 0}

    def _cvt_color_stub(image, code):
        if code == hue_rotation_module.cv2.COLOR_BGR2LUV:
            calls['luv'] += 1
            luv = np.zeros_like(image)
            luv[:, :, 0] = 60
            luv[:, :, 1] = 144
            luv[:, :, 2] = 112
            return luv
        if code == hue_rotation_module.cv2.COLOR_LUV2BGR:
            out = np.zeros_like(image)
            out[:, :, 0] = image[:, :, 2]
            out[:, :, 1] = image[:, :, 1]
            out[:, :, 2] = image[:, :, 0]
            return out
        if code == hue_rotation_module.cv2.COLOR_BGR2RGB:
            calls['rgb'] += 1
            return image[:, :, ::-1]
        if code == hue_rotation_module.cv2.COLOR_RGB2BGR:
            return image[:, :, ::-1]
        return image

    monkeypatch.setattr(hue_rotation_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)

    bgr_pixel = np.array([[[10, 20, 200]]], dtype=np.uint8)

    out_luv, _ = node.process(
        bgr_pixel,
        hue_shift_degrees=45,
        color_space='LUV',
    )
    out_rgb, _ = node.process(
        bgr_pixel,
        hue_shift_degrees=60,
        color_space='RGB',
    )

    assert calls['luv'] == 1
    assert calls['rgb'] == 1
    assert out_luv.shape == bgr_pixel.shape
    assert out_rgb.shape == bgr_pixel.shape


def test_hue_saturation_adjustment_node_get_set_settings(monkeypatch):
    node = HueSaturationAdjustmentNode()

    values = {
        '111:HueSaturationAdjustment:Float:Input02Value': 1.0,
        '111:HueSaturationAdjustment:Int:Input03Value': 45,
        '111:HueSaturationAdjustment:Int:Input04Value': 20,
        '111:HueSaturationAdjustment:Int:Input05Value': 10,
        '111:HueSaturationAdjustment:Int:Input06Value': 0,
        '111:HueSaturationAdjustment:Int:Input07Value': 0,
        '111:HueSaturationAdjustment:Int:Input08Value': 0,
        '111:HueSaturationAdjustment:Int:Input09Value': 0,
        '111:HueSaturationAdjustment:Int:Input10Value': 0,
        '111:HueSaturationAdjustment:Int:Input11Value': 0,
        '111:HueSaturationAdjustment:Int:Input12Value': 0,
        '111:HueSaturationAdjustment:Int:Input13Value': 0,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'dpg', DpgStub())

    setting = node.get_setting_dict(111)

    assert setting['ver'] == node._ver
    assert setting['pos'] == [10, 20]
    assert setting['111:HueSaturationAdjustment:Float:Input02Value'] == 1.0
    assert setting['111:HueSaturationAdjustment:Int:Input03Value'] == 45

    node.set_setting_dict(111, {
        '111:HueSaturationAdjustment:Float:Input02Value': 0.0,
        '111:HueSaturationAdjustment:Int:Input03Value': -60,
        '111:HueSaturationAdjustment:Int:Input12Value': 50,
        '111:HueSaturationAdjustment:Int:Input13Value': 55,
    })

    assert writes['111:HueSaturationAdjustment:Float:Input02Value'] == 0.0
    assert writes['111:HueSaturationAdjustment:Int:Input03Value'] == -60
    assert writes['111:HueSaturationAdjustment:Int:Input12Value'] == 50
    assert writes['111:HueSaturationAdjustment:Int:Input13Value'] == 55


def test_hue_saturation_adjustment_process_targets_band_and_preserves_alpha(monkeypatch):
    node = HueSaturationAdjustmentNode()

    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'COLOR_BGR2HSV', 21, raising=False)
    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'COLOR_HSV2BGR', 22, raising=False)

    def _cvt_color_stub(image, code):
        if code == hue_saturation_adjustment_module.cv2.COLOR_BGR2HSV:
            hsv = np.zeros_like(image)
            hsv[:, :, 0] = 0
            hsv[:, :, 1] = 100
            hsv[:, :, 2] = 200
            return hsv
        return image

    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)
    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'merge', lambda channels: np.stack(channels, axis=-1), raising=False)

    params = {parameter['name']: 0 for parameter in node.parameters}
    params['red_hue_shift'] = 30
    params['red_saturation'] = 50

    bgr_pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
    out_bgr, result = node.process(bgr_pixel, **params)

    assert result is None
    assert out_bgr.shape == bgr_pixel.shape
    assert out_bgr[0, 0, 0] == 30
    assert out_bgr[0, 0, 1] == 150
    assert out_bgr[0, 0, 2] == 200

    bgra_pixel = np.array([[[10, 20, 30, 77]]], dtype=np.uint8)
    out_bgra, _ = node.process(bgra_pixel, **params)

    assert out_bgra.shape == bgra_pixel.shape
    assert out_bgra[0, 0, 3] == 77




def test_hue_saturation_adjustment_process_skips_when_all_zero(monkeypatch):
    node = HueSaturationAdjustmentNode()

    called = {'cvt': 0}

    def _cvt_color_stub(image, code):
        called['cvt'] += 1
        return image

    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)

    params = {parameter['name']: 0 for parameter in node.parameters}
    bgr_image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    out, result = node.process(bgr_image, **params)

    assert result is None
    assert called['cvt'] == 0
    assert out is bgr_image


def test_hue_saturation_adjustment_uses_only_active_band_weights(monkeypatch):
    node = HueSaturationAdjustmentNode()

    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'COLOR_BGR2HSV', 21, raising=False)
    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'COLOR_HSV2BGR', 22, raising=False)

    def _cvt_color_stub(image, code):
        if code == hue_saturation_adjustment_module.cv2.COLOR_BGR2HSV:
            hsv = np.zeros_like(image)
            hsv[:, :, 0] = 0
            hsv[:, :, 1] = 100
            hsv[:, :, 2] = 200
            return hsv
        return image

    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)

    params = {parameter['name']: 0 for parameter in node.parameters}
    params['red_hue_shift'] = 30

    out, _ = node.process(np.array([[[8, 9, 10]]], dtype=np.uint8), **params)

    assert out[0, 0, 0] == 30
    assert out[0, 0, 1] == 100
    assert out[0, 0, 2] == 200




def test_hue_saturation_adjustment_hue_uses_single_band_without_neighbor_bleed(monkeypatch):
    node = HueSaturationAdjustmentNode()

    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'COLOR_BGR2HSV', 21, raising=False)
    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'COLOR_HSV2BGR', 22, raising=False)

    def _cvt_color_stub(image, code):
        if code == hue_saturation_adjustment_module.cv2.COLOR_BGR2HSV:
            hsv = np.zeros_like(image)
            hsv[:, :, 0] = 0
            hsv[:, :, 1] = 180
            hsv[:, :, 2] = 200
            return hsv
        return image

    monkeypatch.setattr(hue_saturation_adjustment_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)

    params = {parameter['name']: 0 for parameter in node.parameters}
    params['yellow_hue_shift'] = 50
    params['magenta_hue_shift'] = -50

    out, _ = node.process(np.array([[[3, 4, 5]]], dtype=np.uint8), **params)

    assert out[0, 0, 0] == 0
    assert out[0, 0, 1] == 180
    assert out[0, 0, 2] == 200
def test_hue_saturation_adjustment_update_clamps_linked_values(monkeypatch):
    node = HueSaturationAdjustmentNode()
    _prepare_node(node)

    values = {
        '501:IntValue:Int:Output01Value': 999,
        '502:IntValue:Int:Output01Value': -180,
        '601:HueSaturationAdjustment:Int:Input04Value': 0,
        '601:HueSaturationAdjustment:Int:Input03Value': 0,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(hue_saturation_adjustment_module, 'image_process', lambda frame, **kwargs: frame)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    out_frame, result = node.update(
        601,
        [
            ['1:ImageSource:Image:Output01', '601:HueSaturationAdjustment:Image:Input01'],
            ['501:IntValue:Int:Output01', '601:HueSaturationAdjustment:Int:Input03'],
            ['502:IntValue:Int:Output01', '601:HueSaturationAdjustment:Int:Input04'],
        ],
        {'1:ImageSource': frame},
        {},
    )

    assert result is None
    assert out_frame.shape == frame.shape
    assert writes['601:HueSaturationAdjustment:Int:Input03Value'] == 90
    assert writes['601:HueSaturationAdjustment:Int:Input04Value'] == -100


def test_warmth_tint_node_get_set_settings(monkeypatch):
    node = WarmthTintNode()

    values = {
        '101:WarmthTint:Int:Input02Value': 15,
        '101:WarmthTint:Int:Input03Value': -12,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'dpg', DpgStub())

    setting = node.get_setting_dict(101)

    assert setting['ver'] == node._ver
    assert setting['pos'] == [10, 20]
    assert setting['101:WarmthTint:Int:Input02Value'] == 15
    assert setting['101:WarmthTint:Int:Input03Value'] == -12

    node.set_setting_dict(101, {
        '101:WarmthTint:Int:Input02Value': 24,
        '101:WarmthTint:Int:Input03Value': -18,
    })

    assert writes['101:WarmthTint:Int:Input02Value'] == 24
    assert writes['101:WarmthTint:Int:Input03Value'] == -18


def test_warmth_tint_node_process_adjusts_lab_and_preserves_alpha(monkeypatch):
    node = WarmthTintNode()

    monkeypatch.setattr(warmth_tint_module.cv2, 'COLOR_BGR2LAB', 11, raising=False)
    monkeypatch.setattr(warmth_tint_module.cv2, 'COLOR_LAB2BGR', 12, raising=False)

    calls = []

    def _cvt_color_stub(image, code):
        calls.append(code)
        if code == warmth_tint_module.cv2.COLOR_BGR2LAB:
            lab = np.zeros_like(image)
            lab[:, :, 0] = 40
            lab[:, :, 1] = 128
            lab[:, :, 2] = 128
            return lab

        out = np.zeros_like(image)
        out[:, :, 0] = image[:, :, 2]
        out[:, :, 1] = image[:, :, 1]
        out[:, :, 2] = image[:, :, 0]
        return out

    monkeypatch.setattr(warmth_tint_module.cv2, 'cvtColor', _cvt_color_stub, raising=False)
    monkeypatch.setattr(warmth_tint_module.cv2, 'merge', lambda channels: np.stack(channels, axis=-1), raising=False)

    bgr_pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
    out_bgr, result = node.process(bgr_pixel, warmth=25, tint=-15)

    assert result is None
    assert calls == [11, 12]
    assert out_bgr.shape == bgr_pixel.shape
    assert out_bgr[0, 0, 0] == 153
    assert out_bgr[0, 0, 1] == 113
    assert out_bgr[0, 0, 2] == 40

    bgra_pixel = np.array([[[10, 20, 30, 99]]], dtype=np.uint8)
    out_bgra, _ = node.process(bgra_pixel, warmth=25, tint=-15)

    assert out_bgra.shape == bgra_pixel.shape
    assert out_bgra[0, 0, 3] == 99


def test_warmth_tint_node_update_clamps_linked_values(monkeypatch):
    node = WarmthTintNode()
    _prepare_node(node)

    values = {
        '201:IntValue:Int:Output01Value': 130,
        '202:IntValue:Int:Output01Value': -150,
        '301:WarmthTint:Int:Input02Value': 0,
        '301:WarmthTint:Int:Input03Value': 0,
    }
    writes = {}

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(base_module, 'dpg_set_value', lambda tag, value: writes.setdefault(tag, value))
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(warmth_tint_module, 'image_process', lambda frame, warmth, tint: frame)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    out_frame, result = node.update(
        301,
        [
            ['1:ImageSource:Image:Output01', '301:WarmthTint:Image:Input01'],
            ['201:IntValue:Int:Output01', '301:WarmthTint:Int:Input02'],
            ['202:IntValue:Int:Output01', '301:WarmthTint:Int:Input03'],
        ],
        {'1:ImageSource': frame},
        {},
    )

    assert result is None
    assert out_frame.shape == frame.shape
    assert writes['301:WarmthTint:Int:Input02Value'] == 100
    assert writes['301:WarmthTint:Int:Input03Value'] == -100


class DpgContextRecorder:
    mvNode_Attr_Input = 'input'
    mvNode_Attr_Output = 'output'
    mvNode_Attr_Static = 'static'
    mvFormat_Float_rgb = 'float-rgb'

    def __init__(self):
        self.node_attributes = []
        self.raw_textures = []
        self.widgets = []

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def texture_registry(self, **kwargs):
        return self._Context()

    def node(self, **kwargs):
        return self._Context()

    def node_attribute(self, **kwargs):
        self.node_attributes.append(kwargs)
        return self._Context()

    def group(self, **kwargs):
        return self._Context()

    def add_raw_texture(self, *args, **kwargs):
        self.raw_textures.append((args, kwargs))

    def add_button(self, **kwargs):
        self.widgets.append(('button', kwargs))

    def add_checkbox(self, **kwargs):
        self.widgets.append(('checkbox', kwargs))

    def add_slider_int(self, **kwargs):
        self.widgets.append(('slider_int', kwargs))

    def add_slider_float(self, **kwargs):
        self.widgets.append(('slider_float', kwargs))

    def add_input_int(self, **kwargs):
        self.widgets.append(('input_int', kwargs))

    def add_input_float(self, **kwargs):
        self.widgets.append(('input_float', kwargs))

    def add_combo(self, *args, **kwargs):
        self.widgets.append(('combo', args, kwargs))

    def add_text(self, **kwargs):
        self.widgets.append(('text', kwargs))


def test_declarative_add_node_declares_typed_ports(monkeypatch):
    node = BlurNode()
    dpg_recorder = DpgContextRecorder()
    registered_ports = []
    node.set_port_registration_callback(registered_ports.append)

    monkeypatch.setattr(base_module, 'dpg', dpg_recorder)
    monkeypatch.setattr(node_abc_module, 'dpg', dpg_recorder)
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda image, w, h: image)

    tag_node_name = node.add_node(
        'NodeEditor',
        6,
        pos=[10, 20],
        opencv_setting_dict={
            'process_width': 8,
            'process_height': 8,
            'use_pref_counter': True,
        },
    )

    declared_tags = [port.dpg_tag for port in node.get_declared_port_refs(6)]
    assert tag_node_name == '6:Blur'
    assert declared_tags == [
        '6:Blur:Image:Input01',
        '6:Blur:Image:Output01',
        '6:Blur:TimeMS:Output02',
        '6:Blur:Int:Input02',
    ]
    assert registered_ports == node.get_declared_port_refs(6)
    assert [attr['tag'] for attr in dpg_recorder.node_attributes] == [
        '6:Blur:ToolbarAttr',
        '6:Blur:Image:Input01',
        '6:Blur:Image:Output01',
        '6:Blur:Int:Input02',
        '6:Blur:TimeMS:Output02',
    ]


def test_custom_parameter_ui_defers_attribute_to_node(monkeypatch):
    node = CurvesNode()
    dpg_recorder = DpgContextRecorder()

    monkeypatch.setattr(base_module, 'dpg', dpg_recorder)
    monkeypatch.setattr(node_abc_module, 'dpg', dpg_recorder)

    node._add_parameter_ui(7, node.parameters[0], 240, callback=None)

    assert dpg_recorder.node_attributes == []
    assert dpg_recorder.widgets == []
    assert node.ports(7).parameters['curves'].dpg_tag == '7:Curves:CurvePoints:Input02'


def test_slider_parameter_ui_adds_nudge_buttons_and_text_input(monkeypatch):
    node = GaussianBlurNode()
    dpg_recorder = DpgContextRecorder()

    monkeypatch.setattr(base_module, 'dpg', dpg_recorder)

    node._add_parameter_ui(7, node.parameters[0], 240, callback=None)

    widget_types = [widget[0] for widget in dpg_recorder.widgets]
    assert widget_types == ['button', 'slider_int', 'input_int', 'button', 'text']
    assert dpg_recorder.widgets[0][1]['label'] == '-'
    assert dpg_recorder.widgets[1][1]['tag'] == '7:GaussianBlur:Int:Input02Value'
    assert dpg_recorder.widgets[1][1]['label'] == ''
    assert dpg_recorder.widgets[1][1]['user_data']['input_tag'] == (
        '7:GaussianBlur:Int:Input02Value:Input'
    )
    assert dpg_recorder.widgets[2][1]['tag'] == '7:GaussianBlur:Int:Input02Value:Input'
    assert dpg_recorder.widgets[2][1]['step'] == 0
    assert dpg_recorder.widgets[3][1]['label'] == '+'
    assert dpg_recorder.widgets[4][1]['default_value'] == 'kernel'


def test_float_slider_steps_use_nice_range_based_values():
    node = GaussianBlurNode()

    assert (
        node._get_parameter_step(gaussian_blur_module.Node.parameters[2])
        == 1.0
    )
    assert node._get_parameter_step(contrast_module.Node.parameters[0]) == 0.05
    assert node._get_parameter_step(crop_module.Node.parameters[0]) == 0.01

def test_slider_nudge_uses_parameter_step_and_clamps(monkeypatch):
    node = GaussianBlurNode()
    node._last_parameter_values = {'7:GaussianBlur:Int:Input02Value': 999}
    values = {'7:GaussianBlur:Int:Input02Value': 501}
    writes = {}
    events = []

    class _Dpg:
        @staticmethod
        def does_item_exist(tag):
            return True

        @staticmethod
        def set_value(tag, value):
            writes[tag] = value
            values[tag] = value

    monkeypatch.setattr(base_module, 'dpg', _Dpg())
    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    node._ui_callback = lambda event, payload: events.append((event, payload))

    parameter = node.parameters[0]
    node._nudge_parameter_widget(
        None,
        None,
        {
            'value_tag': '7:GaussianBlur:Int:Input02Value',
            'input_tag': '7:GaussianBlur:Int:Input02Value:Input',
            'parameter': parameter,
            'direction': 1,
            'callback_payload': {
                'node_id_name': '7:GaussianBlur',
                'port_tag': '7:GaussianBlur:Int:Input02',
                'value_tag': '7:GaussianBlur:Int:Input02Value',
                'parameter': parameter,
                'callback': None,
            },
        },
    )

    assert writes == {
        '7:GaussianBlur:Int:Input02Value': 501,
        '7:GaussianBlur:Int:Input02Value:Input': 501,
    }
    assert events == [
        (
            'parameter_changed',
            {
                'node_id_name': '7:GaussianBlur',
                'port_tag': '7:GaussianBlur:Int:Input02',
                'value_tag': '7:GaussianBlur:Int:Input02Value',
                'before_value': 999,
                'after_value': 501,
            },
        )
    ]


def test_slider_change_syncs_text_input(monkeypatch):
    node = GaussianBlurNode()
    node._last_parameter_values = {'7:GaussianBlur:Int:Input02Value': 5}
    writes = {}

    class _Dpg:
        @staticmethod
        def does_item_exist(tag):
            return True

        @staticmethod
        def set_value(tag, value):
            writes[tag] = value

    monkeypatch.setattr(base_module, 'dpg', _Dpg())
    parameter = node.parameters[0]

    node._on_parameter_widget_changed(
        '7:GaussianBlur:Int:Input02Value',
        7,
        {
            'node_id_name': '7:GaussianBlur',
            'port_tag': '7:GaussianBlur:Int:Input02',
            'value_tag': '7:GaussianBlur:Int:Input02Value',
            'input_tag': '7:GaussianBlur:Int:Input02Value:Input',
            'parameter': parameter,
            'callback': None,
        },
    )

    assert writes['7:GaussianBlur:Int:Input02Value'] == 7
    assert writes['7:GaussianBlur:Int:Input02Value:Input'] == 7


def test_slider_text_input_syncs_canonical_slider_and_clamps(monkeypatch):
    node = GaussianBlurNode()
    node._last_parameter_values = {'7:GaussianBlur:Int:Input02Value': 5}
    writes = {}
    events = []

    class _Dpg:
        @staticmethod
        def does_item_exist(tag):
            return True

        @staticmethod
        def set_value(tag, value):
            writes[tag] = value

    monkeypatch.setattr(base_module, 'dpg', _Dpg())
    node._ui_callback = lambda event, payload: events.append((event, payload))
    parameter = node.parameters[0]

    node._on_parameter_widget_changed(
        '7:GaussianBlur:Int:Input02Value:Input',
        999,
        {
            'node_id_name': '7:GaussianBlur',
            'port_tag': '7:GaussianBlur:Int:Input02',
            'value_tag': '7:GaussianBlur:Int:Input02Value',
            'input_tag': '7:GaussianBlur:Int:Input02Value:Input',
            'parameter': parameter,
            'callback': None,
        },
    )

    assert writes['7:GaussianBlur:Int:Input02Value'] == 501
    assert writes['7:GaussianBlur:Int:Input02Value:Input'] == 501
    assert events[0][1]['value_tag'] == '7:GaussianBlur:Int:Input02Value'
    assert events[0][1]['after_value'] == 501


def test_curves_node_uses_linked_points_parameter(monkeypatch):
    node = CurvesNode()
    _prepare_node(node)

    values = {
        '101:CurvesPoints:CurvePoints:Output01Value': '[[0, 0], [96, 180], [255, 255]]',
        '102:Curves:CurvePoints:Input02Value': '[[0, 0], [255, 255]]',
    }
    written = {}

    def _record_write(tag, value):
        written[tag] = value
        values[tag] = value

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(
        base_module,
        'dpg_set_value',
        _record_write,
    )
    monkeypatch.setattr(
        curves_module,
        'dpg_set_value',
        _record_write,
    )
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(
        curves_module.Node,
        '_get_drag_points',
        lambda self, node_id: [[0, 0], [255, 255]],
    )
    monkeypatch.setattr(
        curves_module.Node,
        '_reset_points_from_setting',
        lambda self, node_id, points: None,
    )

    captured = {}

    def _lut_stub(image, points, channel='White'):
        captured['points'] = points
        captured['channel'] = channel
        return image

    monkeypatch.setattr(curves_module, 'image_process', _lut_stub)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    out_frame, result = node.update(
        102,
        [
            ['3:ImageSource:Image:Output01', '102:Curves:Image:Input01'],
            ['101:CurvesPoints:CurvePoints:Output01', '102:Curves:CurvePoints:Input02'],
        ],
        {'3:ImageSource': frame},
        {},
    )

    assert result is None
    assert out_frame.shape == frame.shape
    assert written['102:Curves:CurvePoints:Input02Value'] == '[[0, 0], [96, 180], [255, 255]]'
    expected_curves = _identity_curve_set()
    expected_curves['White'] = [[0, 0], [96, 180], [255, 255]]
    assert captured['points'] == expected_curves
    assert captured['channel'] == 'White'

def test_curves_image_process_applies_selected_rgb_channel_only():
    image = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
    points = [[0, 0], [255, 255]]
    points[1][1] = 0

    red = curves_module.image_process(image, points, 'Red')
    green = curves_module.image_process(image, points, 'Green')
    blue = curves_module.image_process(image, points, 'Blue')

    assert red[0, 0].tolist() == [10, 20, 0, 40]
    assert green[0, 0].tolist() == [10, 0, 30, 40]
    assert blue[0, 0].tolist() == [0, 20, 30, 40]


def test_curves_image_process_applies_white_before_rgb_curve_set():
    image = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
    curve_set = _identity_curve_set()
    curve_set['White'] = [[0, 0], [255, 255]]
    curve_set['White'][1][1] = 100
    curve_set['Red'] = [[0, 0], [100, 50], [255, 255]]
    curve_set['Green'] = [[0, 0], [100, 75], [255, 255]]
    curve_set['Blue'] = [[0, 0], [100, 25], [255, 255]]

    result = curves_module.image_process(image, {'curves': curve_set})

    assert result[0, 0].tolist() == [0, 5, 5, 40]


def test_curves_node_uses_channel_from_linked_points_payload(monkeypatch):
    node = CurvesNode()
    _prepare_node(node)

    values = {
        '201:CurvesPoints:CurvePoints:Output01Value': (
            '{"channel": "Red", "points": [[0, 0], [255, 128]]}'
        ),
        '202:Curves:CurvePoints:Input02Value': '[[0, 0], [255, 255]]',
    }

    monkeypatch.setattr(base_module, 'dpg_get_value', lambda tag: values.get(tag))
    monkeypatch.setattr(
        base_module,
        'dpg_set_value',
        lambda tag, value: values.__setitem__(tag, value),
    )
    monkeypatch.setattr(
        curves_module,
        'dpg_set_value',
        lambda tag, value: values.__setitem__(tag, value),
    )
    monkeypatch.setattr(base_module, 'convert_cv_to_dpg', lambda frame, w, h: frame)
    monkeypatch.setattr(
        curves_module.Node,
        '_get_drag_points',
        lambda self, node_id: [[0, 0], [255, 255]],
    )
    monkeypatch.setattr(
        curves_module.Node,
        '_reset_points_from_setting',
        lambda self, node_id, points: None,
    )

    captured = {}

    def _lut_stub(image, points, channel='White'):
        captured['points'] = points
        captured['channel'] = channel
        return image

    monkeypatch.setattr(curves_module, 'image_process', _lut_stub)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    node.update(
        202,
        [
            ['3:ImageSource:Image:Output01', '202:Curves:Image:Input01'],
            ['201:CurvesPoints:CurvePoints:Output01', '202:Curves:CurvePoints:Input02'],
        ],
        {'3:ImageSource': frame},
        {},
    )

    expected_curves = _identity_curve_set()
    expected_curves['Red'] = [[0, 0], [255, 128]]
    assert captured['points'] == expected_curves
    assert captured['channel'] == 'White'

def test_rgb_channel_swap_reorders_rgb_channels_and_preserves_alpha():
    bgr_pixel = [10, 20, 30]
    bgra_pixel = [10, 20, 30, 40]

    bgr_image = np.array([[bgr_pixel]], dtype=np.uint8)
    bgra_image = np.array([[bgra_pixel]], dtype=np.uint8)

    swapped_bgr = rgb_channel_swap_module.image_process(bgr_image, 'BGR')
    swapped_bgra = rgb_channel_swap_module.image_process(bgra_image, 'GBR')

    assert swapped_bgr[0, 0].tolist() == [30, 20, 10]
    assert swapped_bgra[0, 0].tolist() == [30, 10, 20, 40]


def test_rgb_channel_swap_node_falls_back_to_rgb_on_invalid_order():
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)

    result = rgb_channel_swap_module.image_process(image, 'invalid')

    assert result[0, 0].tolist() == [10, 20, 30]


def test_hue_saturation_adjustment_nudges_only_last_touched_node(monkeypatch):
    node = HueSaturationAdjustmentNode()
    node._last_touched_slider_tag_by_node = {
        1: '1:HueSaturationAdjustment:Int:Input03Value',
        2: '2:HueSaturationAdjustment:Int:Input03Value',
    }
    node._last_touched_node_id = 2
    writes = {}

    class _Dpg:
        @staticmethod
        def does_item_exist(tag):
            return True

        @staticmethod
        def get_item_configuration(tag):
            return {'min_value': -180, 'max_value': 180}

        @staticmethod
        def get_value(tag):
            return 10 if tag.startswith('2:') else 100

        @staticmethod
        def set_value(tag, value):
            writes[tag] = value

    monkeypatch.setattr(hue_saturation_adjustment_module, 'dpg', _Dpg())

    node._nudge_slider(None, None, 1)

    assert writes == {'2:HueSaturationAdjustment:Int:Input03Value': 11}


def test_hue_saturation_adjustment_blend_zero_uses_one_hot_band_weights():
    weights = hue_saturation_adjustment_module._get_blend_weight_lut(0.0)

    assert np.allclose(np.sum(weights, axis=1), 1.0)
    assert np.all(np.count_nonzero(weights, axis=1) == 1)


def test_hue_saturation_adjustment_uses_eight_photo_editor_bands():
    assert hue_saturation_adjustment_module._BANDS == (
        ('red', 0.0),
        ('orange', 15.0),
        ('yellow', 30.0),
        ('green', 60.0),
        ('cyan', 90.0),
        ('blue', 120.0),
        ('purple', 135.0),
        ('magenta', 150.0),
    )


def test_hue_saturation_adjustment_hue_shift_range_spans_full_circle():
    assert hue_saturation_adjustment_module.HUE_SHIFT_MIN == -90
    assert hue_saturation_adjustment_module.HUE_SHIFT_MAX == 90
    assert (
        hue_saturation_adjustment_module.HUE_SHIFT_MIN % 180
        == hue_saturation_adjustment_module.HUE_SHIFT_MAX % 180
    )


def test_gaussian_blur_auto_kernel_uses_configurable_factor(monkeypatch):
    node = GaussianBlurNode()
    calls = {}

    def _gaussian_stub(image, kernel, sigma):
        calls['kernel'] = kernel
        calls['sigma'] = sigma
        return image

    monkeypatch.setattr(gaussian_blur_module.cv2, 'GaussianBlur', _gaussian_stub, raising=False)

    frame = np.zeros((3, 3), dtype=np.uint8)
    result, _ = node.process(
        frame,
        kernel_size=5,
        auto_sigma=True,
        auto_kernel=True,
        kernel_factor=2.5,
        sigma=2.0,
    )

    assert result is frame
    assert calls['kernel'] == (11, 11)
    assert calls['sigma'] == 2.0


def test_gaussian_blur_auto_kernel_overrides_auto_sigma():
    node = GaussianBlurNode()
    values = {
        'kernel_size': 5,
        'auto_sigma': True,
        'auto_kernel': True,
        'kernel_factor': 3.0,
        'sigma': 1.5,
    }

    normalized = node.normalize_parameter_values('1:GaussianBlur', values)

    assert normalized['auto_sigma'] is False
    assert normalized['sigma'] == 1.5


def test_gaussian_blur_auto_kernel_toggle_updates_kernel_display(monkeypatch):
    node = GaussianBlurNode()
    values = {
        '8:GaussianBlur:Int:Input02Value': 5,
        '8:GaussianBlur:Int:Input02Value:Input': 5,
        '8:GaussianBlur:Float:Input03Value': 2.0,
        '8:GaussianBlur:Float:Input03Value:Input': 2.0,
        '8:GaussianBlur:Int:Input04Value': False,
        '8:GaussianBlur:Int:Input05Value': False,
        '8:GaussianBlur:Float:Input06Value': 2.5,
        '8:GaussianBlur:Float:Input06Value:Input': 2.5,
    }
    callbacks = {}
    configured = {}
    previous_calls = []

    class _Dpg:
        @staticmethod
        def get_value(tag):
            return values[tag]

        @staticmethod
        def set_value(tag, value):
            values[tag] = value

        @staticmethod
        def configure_item(tag, **kwargs):
            configured.setdefault(tag, {}).update(kwargs)
            if 'callback' in kwargs:
                callbacks[tag] = kwargs['callback']

        @staticmethod
        def does_item_exist(tag):
            return tag in values

        @staticmethod
        def get_item_callback(tag):
            return lambda sender, app_data, user_data: previous_calls.append(
                (tag, sender, app_data, user_data)
            )

        @staticmethod
        def get_item_user_data(tag):
            return {'tag': tag}

    monkeypatch.setattr(gaussian_blur_module, 'dpg', _Dpg())

    node.on_node_added('8:GaussianBlur')
    callbacks['8:GaussianBlur:Int:Input05Value'](
        '8:GaussianBlur:Int:Input05Value',
        True,
        None,
    )

    assert values['8:GaussianBlur:Int:Input02Value'] == 11
    assert values['8:GaussianBlur:Int:Input02Value:Input'] == 11
    assert configured['8:GaussianBlur:Int:Input02Value']['enabled'] is False
    assert configured['8:GaussianBlur:Float:Input03Value']['enabled'] is True
    assert previous_calls[-1][0] == '8:GaussianBlur:Int:Input05Value'


def test_gaussian_blur_auto_sigma_toggle_updates_sigma_display(monkeypatch):
    node = GaussianBlurNode()
    values = {
        '8:GaussianBlur:Int:Input02Value': 5,
        '8:GaussianBlur:Int:Input02Value:Input': 5,
        '8:GaussianBlur:Float:Input03Value': 2.0,
        '8:GaussianBlur:Float:Input03Value:Input': 2.0,
        '8:GaussianBlur:Int:Input04Value': False,
        '8:GaussianBlur:Int:Input05Value': False,
        '8:GaussianBlur:Float:Input06Value': 2.5,
        '8:GaussianBlur:Float:Input06Value:Input': 2.5,
    }
    callbacks = {}
    configured = {}

    class _Dpg:
        @staticmethod
        def get_value(tag):
            return values[tag]

        @staticmethod
        def set_value(tag, value):
            values[tag] = value

        @staticmethod
        def configure_item(tag, **kwargs):
            configured.setdefault(tag, {}).update(kwargs)
            if 'callback' in kwargs:
                callbacks[tag] = kwargs['callback']

        @staticmethod
        def does_item_exist(tag):
            return tag in values

        @staticmethod
        def get_item_callback(tag):
            del tag
            return None

        @staticmethod
        def get_item_user_data(tag):
            del tag
            return None

    monkeypatch.setattr(gaussian_blur_module, 'dpg', _Dpg())

    node.on_node_added('8:GaussianBlur')
    callbacks['8:GaussianBlur:Int:Input04Value'](
        '8:GaussianBlur:Int:Input04Value',
        True,
        None,
    )

    assert values['8:GaussianBlur:Int:Input05Value'] is False
    assert values['8:GaussianBlur:Float:Input03Value'] == 1.1
    assert values['8:GaussianBlur:Float:Input03Value:Input'] == 1.1
    assert configured['8:GaussianBlur:Float:Input03Value']['enabled'] is False
    assert configured['8:GaussianBlur:Int:Input02Value']['enabled'] is True
