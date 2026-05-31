import pytest

from node.input_node.node_int_value import Node as IntValueNode
from node.port_model import NodeRef, PortRef


def test_explicit_port_declaration_preserves_compact_tag_shape():
    node = IntValueNode()

    port = node.output_port(7, node.TYPE_INT, 'Output01')

    assert port == PortRef(
        node_ref=NodeRef('7', 'IntValue'),
        direction='Output',
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
