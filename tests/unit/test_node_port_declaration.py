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
from node.port_serialization import port_ref_from_tag
from node.port_model import (
    InputPort,
    NodeRef,
    OutputPort,
    PortDataType,
    PortDirection,
    PortRef,
    PortSpecs,
)


def test_explicit_port_declaration_preserves_compact_tag_shape():
    node = IntValueNode()

    port = node.output_port(7, node.TYPE_INT, 'Output01')

    assert port == PortRef(
        node_ref=NodeRef('7', 'IntValue'),
        direction=PortDirection.OUTPUT,
        data_type='Int',
        index=1,
        port_name='Output01',
        dpg_tag='7:IntValue:Int:Output01',
        value_tag='7:IntValue:Int:Output01Value',
        control_tag=None,
    )


def test_auto_numbered_ports_increment_by_node_and_direction():
    node = IntValueNode()

    image_in = node.input_port(3, node.TYPE_IMAGE)
    int_in = node.parameter_port(3, node.TYPE_INT)
    image_out = node.output_port(3, node.TYPE_IMAGE)
    next_node_input = node.input_port(4, node.TYPE_TEXT)

    assert image_in.port_name == 'Input01'
    assert image_in.dpg_tag == '3:IntValue:Image:Input01'
    assert image_in.control_tag is None
    assert int_in.port_name == 'Input02'
    assert int_in.dpg_tag == '3:IntValue:Int:Input02'
    assert int_in.control_tag == '3:IntValue:Int:Input02Value'
    assert image_out.port_name == 'Output01'
    assert next_node_input.port_name == 'Input01'


def test_explicit_port_declaration_advances_auto_numbering():
    node = IntValueNode()

    explicit_port = node.input_port(2, node.TYPE_INT, 'Input03')
    auto_port = node.input_port(2, node.TYPE_TEXT)

    assert explicit_port.index == 3
    assert auto_port.port_name == 'Input04'


def test_declared_port_refs_are_stored_by_node():
    node = IntValueNode()

    first = node.output_port(1, node.TYPE_INT, 'Output01')
    second = node.output_port(2, node.TYPE_INT, 'Output01')

    assert node.get_declared_port_refs(1) == [first]
    assert node.get_declared_port_refs(2) == [second]
    assert node.get_declared_port_refs() == [first, second]


def test_declared_ports_use_direction_enum_values():
    node = IntValueNode()

    output_port = node.output_port(10, node.TYPE_INT)
    input_port = node.input_port(10, node.TYPE_FLOAT)

    assert output_port.direction is PortDirection.OUTPUT
    assert output_port.direction == 'Output'
    assert output_port.port_name == 'Output01'
    assert output_port.value_tag == '10:IntValue:Int:Output01Value'
    assert input_port.direction is PortDirection.INPUT
    assert input_port.direction == 'Input'
    assert input_port.port_name == 'Input01'


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


def test_port_declaration_invokes_registration_callback():
    node = IntValueNode()
    registered_ports = []
    node.set_port_registration_callback(registered_ports.append)

    declared = node.output_port(4, node.TYPE_TIME_MS, 'Output02')

    assert registered_ports == [declared]


def test_parameter_port_accepts_custom_control_tag():
    node = IntValueNode()

    port = node.parameter_port(
        5,
        node.TYPE_TEXT,
        'Input02',
        control_tag='5:IntValue:Text:ComboControl',
    )

    assert port.control_tag == '5:IntValue:Text:ComboControl'
    assert port.value_tag == '5:IntValue:Text:Input02Value'


def test_port_declaration_rejects_mismatched_direction_prefix():
    node = IntValueNode()

    with pytest.raises(ValueError, match='must start with Output'):
        node.output_port(1, node.TYPE_IMAGE, 'Input01')


def test_port_declaration_rejects_non_numeric_index():
    node = IntValueNode()

    with pytest.raises(ValueError, match='must end with a numeric index'):
        node.input_port(1, node.TYPE_IMAGE, 'InputMain')
