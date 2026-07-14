import pytest

from node.analysis_node.node_BRISQUE import Node as BrisqueNode
from node.analysis_node.node_fps import Node as FpsNode
from node.analysis_node.node_rgb_histgram import Node as RgbHistgramNode
from node.deep_learning_node.node_classification import Node as ClassificationNode
from node.deep_learning_node.node_face_detection import Node as FaceDetectionNode
from node.deep_learning_node.node_low_light_image_enhancement import (
    Node as LowLightImageEnhancementNode,
)
from node.deep_learning_node.node_monocular_depth_estimation import (
    Node as MonocularDepthEstimationNode,
)
from node.deep_learning_node.node_object_detection import Node as ObjectDetectionNode
from node.deep_learning_node.node_pose_estimation import Node as PoseEstimationNode
from node.deep_learning_node.node_semantic_segmentation import (
    Node as SemanticSegmentationNode,
)
from node.draw_node.node_draw_information import Node as DrawInformationNode
from node.draw_node.node_image_alpha_blend import Node as ImageAlphaBlendNode
from node.draw_node.node_result_image import Node as ResultImageNode
from node.draw_node.node_result_large_image import Node as ResultLargeImageNode
from node.draw_node.node_image_concat import Node as ImageConcatNode
from node.draw_node.node_puttext import Node as PutTextNode
from node.input_node.node_int_value import Node as IntValueNode
from node.input_node.node_rtsp_input import Node as RtspInputNode
from node.input_node.node_video_input import Node as VideoInputNode
from node.input_node.node_video_set_frame_pos_input import (
    Node as VideoSetFramePosNode,
)
from node.input_node.node_webcam_input import Node as WebCamNode
from node.other_node.node_on_off_switch import Node as OnOffSwitchNode
from node.other_node.node_video_writer import Node as VideoWriterNode
from node.preview_release_node.node_code_exec import Node as CodeExecNode
from node.preview_release_node.node_mot import Node as MotNode
from node.preview_release_node.node_screen_capture import (
    Node as ScreenCaptureNode,
)
from node.process_node.node_brightness import Node as BrightnessNode
from node.port_serialization import port_ref_from_tag
from node.port_model import (
    InputPort,
    NodeRef,
    OutputPort,
    ParameterPort,
    PortDataType,
    PortDirection,
    PortRef,
    PortSpecs,
)


def test_port_specs_preserve_compact_tag_shape():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            value=OutputPort(PortDataType.INT),
        )

    port = SpecNode().create_ports(7).value

    assert port == PortRef(
        node_ref=NodeRef('7', 'IntValue'),
        direction=PortDirection.OUTPUT,
        data_type=PortDataType.INT,
        index=1,
        port_name='Output01',
        dpg_tag='7:IntValue:Int:Output01',
        value_tag='7:IntValue:Int:Output01Value',
        control_tag=None,
        spec_key='value',
    )


def test_port_specs_auto_number_by_node_and_direction():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            image_in=InputPort(PortDataType.IMAGE),
            int_param=ParameterPort(PortDataType.INT),
            image_out=OutputPort(PortDataType.IMAGE),
            text_in=InputPort(PortDataType.TEXT),
        )

    node = SpecNode()
    ports = node.create_ports(3)
    next_node_ports = node.create_ports(4)

    assert ports.image_in.port_name == 'Input01'
    assert ports.image_in.dpg_tag == '3:IntValue:Image:Input01'
    assert ports.image_in.control_tag is None
    assert ports.int_param.port_name == 'Input02'
    assert ports.int_param.dpg_tag == '3:IntValue:Int:Input02'
    assert ports.int_param.control_tag == '3:IntValue:Int:Input02Value'
    assert ports.image_out.port_name == 'Output01'
    assert ports.text_in.port_name == 'Input03'
    assert next_node_ports.image_in.port_name == 'Input01'


def test_explicit_port_spec_index_advances_auto_numbering():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            explicit=InputPort(PortDataType.INT, index=3),
            auto=InputPort(PortDataType.TEXT),
        )

    ports = SpecNode().create_ports(2)

    assert ports.explicit.index == 3
    assert ports.explicit.port_name == 'Input03'
    assert ports.auto.port_name == 'Input04'


def test_declared_port_refs_are_stored_by_node():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            value=OutputPort(PortDataType.INT),
        )

    node = SpecNode()
    first = node.create_ports(1).value
    second = node.create_ports(2).value

    assert node.get_declared_port_refs(1) == [first]
    assert node.get_declared_port_refs(2) == [second]
    assert node.get_declared_port_refs() == [first, second]


def test_declared_ports_use_direction_enum_values():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            value=OutputPort(PortDataType.INT),
            value_input=InputPort(PortDataType.FLOAT),
        )

    ports = SpecNode().create_ports(10)

    assert ports.value.direction is PortDirection.OUTPUT
    assert ports.value.direction == 'Output'
    assert ports.value.port_name == 'Output01'
    assert ports.value.value_tag == '10:IntValue:Int:Output01Value'
    assert ports.value_input.direction is PortDirection.INPUT
    assert ports.value_input.direction == 'Input'
    assert ports.value_input.port_name == 'Input01'


def test_port_specs_create_base_owned_attribute_handles():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            value=OutputPort(PortDataType.INT),
            image=OutputPort(PortDataType.IMAGE),
        )

    node = SpecNode()

    ports = node.create_ports(12)

    assert ports.value.direction is PortDirection.OUTPUT
    assert ports.value.data_type is PortDataType.INT
    assert ports.value.port_name == 'Output01'
    assert ports.value.spec_key == 'value'
    assert ports.image.direction is PortDirection.OUTPUT
    assert ports.image.data_type is PortDataType.IMAGE
    assert ports.image.port_name == 'Output02'
    assert node.ports(12) is ports
    assert node.get_declared_port_refs(12) == [ports.value, ports.image]


def test_port_spec_index_override_derives_legacy_name():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            elapsed=OutputPort(PortDataType.TIME_MS, index=2),
        )

    ports = SpecNode().create_ports(13)

    assert ports.elapsed.index == 2
    assert ports.elapsed.port_name == 'Output02'
    assert ports.elapsed.dpg_tag == '13:IntValue:TimeMS:Output02'


def test_create_port_adds_dynamic_handle_and_collection_entries():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            value=OutputPort(PortDataType.INT),
        )

    node = SpecNode()
    ports = node.create_ports(14)

    dynamic_slot = node.create_port(
        14,
        'slot_2',
        InputPort(PortDataType.IMAGE, index=2),
        collection='image_slots',
    )
    debug_port = node.create_port(
        14,
        'debug',
        OutputPort(PortDataType.TEXT, index=3),
    )

    assert ports.image_slots['slot_2'] is dynamic_slot
    assert dynamic_slot.direction is PortDirection.INPUT
    assert dynamic_slot.data_type is PortDataType.IMAGE
    assert dynamic_slot.port_name == 'Input02'
    assert dynamic_slot.spec_key == 'slot_2'
    assert dynamic_slot.dpg_tag == '14:IntValue:Image:Input02'
    assert ports.debug is debug_port
    assert debug_port.direction is PortDirection.OUTPUT
    assert debug_port.port_name == 'Output03'
    assert debug_port.spec_key == 'debug'
    assert node.ports(14) is ports
    assert node.get_declared_port_refs(14) == [
        ports.value,
        dynamic_slot,
        debug_port,
    ]



def test_parameter_port_spec_defaults_control_tag_to_value_tag():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            threshold=ParameterPort(PortDataType.INT, index=2),
        )

    ports = SpecNode().create_ports(15)

    assert ports.threshold.direction is PortDirection.INPUT
    assert ports.threshold.data_type is PortDataType.INT
    assert ports.threshold.port_name == 'Input02'
    assert ports.threshold.value_tag == '15:IntValue:Int:Input02Value'
    assert ports.threshold.control_tag == ports.threshold.value_tag


def test_declarative_process_base_creates_parameter_collection_handles():
    node = BrightnessNode()

    ports = node._ensure_declarative_port_handles(16, include_elapsed=True)

    assert ports.image_input.direction is PortDirection.INPUT
    assert ports.image_input.data_type is PortDataType.IMAGE
    assert ports.image_input.port_name == 'Input01'
    assert ports.image.direction is PortDirection.OUTPUT
    assert ports.image.data_type is PortDataType.IMAGE
    assert ports.image.port_name == 'Output01'
    assert ports.elapsed.direction is PortDirection.OUTPUT
    assert ports.elapsed.data_type is PortDataType.TIME_MS
    assert ports.elapsed.port_name == 'Output02'

    beta = ports.parameters['beta']
    assert beta.direction is PortDirection.INPUT
    assert beta.data_type is PortDataType.INT
    assert beta.index == 2
    assert beta.port_name == 'Input02'
    assert beta.spec_key == 'beta'
    assert beta.control_tag == beta.value_tag
    assert node._parameter_port_ref(16, node.parameters[0]) is beta


def test_dynamic_slot_nodes_create_collection_port_handles(monkeypatch):
    class NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    for node_class, collection_name, data_type in (
        (ImageConcatNode, 'image_inputs', PortDataType.IMAGE),
        (FpsNode, 'elapsed_inputs', PortDataType.TIME_MS),
    ):
        node = node_class()
        node_id = 22
        tag_node_name = node._node_name(node_id)
        node.create_ports(node_id)
        node._slot_id[tag_node_name] = 1

        module_dpg = __import__(node_class.__module__, fromlist=['dpg']).dpg
        monkeypatch.setattr(
            module_dpg,
            'node_attribute',
            lambda *args, **kwargs: NullContext(),
        )
        monkeypatch.setattr(module_dpg, 'add_text', lambda *args, **kwargs: None)

        node._add_slot(None, None, tag_node_name)

        dynamic_port = getattr(node.ports(node_id), collection_name)[2]
        assert dynamic_port.direction is PortDirection.INPUT
        assert dynamic_port.data_type is data_type
        assert dynamic_port.index == 2
        assert dynamic_port.port_name == 'Input02'
        assert dynamic_port.spec_key == '2'


def test_port_serialization_parses_legacy_tag_to_typed_ref():
    port = port_ref_from_tag('21:IntValue:Float:Input03')

    assert port.node_ref == NodeRef('21', 'IntValue')
    assert port.direction is PortDirection.INPUT
    assert port.data_type is PortDataType.FLOAT
    assert port.index == 3
    assert port.port_name == 'Input03'
    assert port.value_tag == '21:IntValue:Float:Input03Value'


def test_migrated_sink_and_source_nodes_expose_named_port_handles():
    result_node = ResultImageNode()
    result_ports = result_node.create_ports(31)

    assert result_ports.image.direction is PortDirection.INPUT
    assert result_ports.image.data_type is PortDataType.IMAGE
    assert result_ports.image.port_name == 'Input01'
    assert result_ports.image.value_tag == '31:ResultImage:Image:Input01Value'

    capture_node = ScreenCaptureNode()
    capture_ports = capture_node.create_ports(32)

    assert capture_ports.image.direction is PortDirection.OUTPUT
    assert capture_ports.image.data_type is PortDataType.IMAGE
    assert capture_ports.image.port_name == 'Output01'
    assert capture_ports.elapsed.direction is PortDirection.OUTPUT
    assert capture_ports.elapsed.data_type is PortDataType.TIME_MS
    assert capture_ports.elapsed.port_name == 'Output02'
    assert capture_ports.elapsed.value_tag == (
        '32:ScreenCapture:TimeMS:Output02Value'
    )


def test_expanded_migrated_nodes_expose_expected_port_handles():
    cases = (
        (
            BrisqueNode,
            'BRISQUE',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            ClassificationNode,
            'Classification',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            CodeExecNode,
            'ExecPythonCode',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            DrawInformationNode,
            'DrawInformation',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
            },
        ),
        (
            ImageAlphaBlendNode,
            'ImageAlphaBlend',
            {
                'image_a': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image_b': (PortDirection.INPUT, PortDataType.IMAGE, 'Input02'),
                'alpha': (PortDirection.INPUT, PortDataType.FLOAT, 'Input03'),
                'beta': (PortDirection.INPUT, PortDataType.FLOAT, 'Input04'),
                'gamma': (PortDirection.INPUT, PortDataType.INT, 'Input05'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            FaceDetectionNode,
            'FaceDetection',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'threshold': (PortDirection.INPUT, PortDataType.FLOAT, 'Input03'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            FpsNode,
            'FPS',
            {
                'elapsed_input': (PortDirection.INPUT, PortDataType.TIME_MS, 'Input01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            ImageConcatNode,
            'ImageConcat',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
            },
        ),
        (
            LowLightImageEnhancementNode,
            'LLIE',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            MonocularDepthEstimationNode,
            'MonocularDepthEstimation',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            MotNode,
            'MultiObjectTracking',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            ObjectDetectionNode,
            'ObjectDetection',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'threshold': (PortDirection.INPUT, PortDataType.FLOAT, 'Input03'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            PoseEstimationNode,
            'PoseEstimation',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'threshold': (PortDirection.INPUT, PortDataType.FLOAT, 'Input03'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            PutTextNode,
            'PutText',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'text': (PortDirection.INPUT, PortDataType.TEXT, 'Input02'),
                'elapsed_input': (PortDirection.INPUT, PortDataType.TIME_MS, 'Input03'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
            },
        ),
        (
            RgbHistgramNode,
            'RGBHistgram',
            {
                'image': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
            },
        ),
        (
            ResultLargeImageNode,
            'ResultImageLarge',
            {
                'image': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
            },
        ),
        (
            SemanticSegmentationNode,
            'SemanticSegmentation',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'threshold': (PortDirection.INPUT, PortDataType.FLOAT, 'Input03'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            VideoWriterNode,
            'VideoWriter',
            {
                'image': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
            },
        ),
        (
            RtspInputNode,
            'RTSPInput',
            {
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            VideoInputNode,
            'Video',
            {
                'skip_rate': (PortDirection.INPUT, PortDataType.INT, 'Input03'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            VideoSetFramePosNode,
            'VideoSetFramePos',
            {
                'seek': (PortDirection.INPUT, PortDataType.INT, 'Input02'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
                'frame_pos': (PortDirection.OUTPUT, PortDataType.INT, 'Output03'),
            },
        ),
        (
            WebCamNode,
            'WebCam',
            {
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
                'elapsed': (PortDirection.OUTPUT, PortDataType.TIME_MS, 'Output02'),
            },
        ),
        (
            OnOffSwitchNode,
            'OnOffSwitch',
            {
                'image_input': (PortDirection.INPUT, PortDataType.IMAGE, 'Input01'),
                'image': (PortDirection.OUTPUT, PortDataType.IMAGE, 'Output01'),
            },
        ),
    )

    for node_class, node_tag, expectations in cases:
        ports = node_class().create_ports(41)

        for handle_name, (direction, data_type, port_name) in expectations.items():
            port = getattr(ports, handle_name)

            assert port.direction is direction
            assert port.data_type is data_type
            assert port.port_name == port_name
            assert port.spec_key == handle_name
            assert port.dpg_tag == f'41:{node_tag}:{data_type.value}:{port_name}'
            assert port.value_tag == f'{port.dpg_tag}Value'


def test_port_spec_declaration_invokes_registration_callback():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            elapsed=OutputPort(PortDataType.TIME_MS, index=2),
        )

    node = SpecNode()
    registered_ports = []
    node.set_port_registration_callback(registered_ports.append)

    declared = node.create_ports(4).elapsed

    assert registered_ports == [declared]


def test_parameter_port_accepts_custom_control_tag():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            combo=ParameterPort(
                PortDataType.TEXT,
                index=2,
                control_tag='5:IntValue:Text:ComboControl',
            ),
        )

    port = SpecNode().create_ports(5).combo

    assert port.control_tag == '5:IntValue:Text:ComboControl'
    assert port.value_tag == '5:IntValue:Text:Input02Value'


def test_port_spec_rejects_non_numeric_index():
    class SpecNode(IntValueNode):
        port_specs = PortSpecs(
            image=InputPort(PortDataType.IMAGE, index='Main'),
        )

    with pytest.raises(ValueError, match="invalid literal for int"):
        SpecNode().create_ports(1)
