import pytest

from node.draw_node.node_result_image import Node as ResultImageNode
from node.input_node.node_int_value import Node as IntValueNode
from node.preview_release_node.node_screen_capture import (
    Node as ScreenCaptureNode,
)
from node.port_serialization import port_ref_from_tag
from node.port_model import (
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
