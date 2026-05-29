import pytest

from node.node_abc import DpgNodeBase
from node_editor.port_model import NodeRef, PortRef


class ConcreteNode(DpgNodeBase):
    node_tag = 'Concrete'
    node_label = 'Concrete Node'

    def add_node(
        self,
        parent,
        node_id,
        pos=None,
        opencv_setting_dict=None,
        callback=None,
    ):
        del parent, node_id, pos, opencv_setting_dict, callback

    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        del node_id, connection_list, node_image_dict, node_result_dict

    def get_setting_dict(self, node_id):
        del node_id
        return {}

    def set_setting_dict(self, node_id, setting_dict):
        del node_id, setting_dict

    def close(self, node_id):
        del node_id


def test_input_and_output_port_declarations_keep_existing_tag_shape():
    node = ConcreteNode()

    image_in = node.input_port(7, node.TYPE_IMAGE, 'Input01')
    float_out = node.output_port(7, node.TYPE_FLOAT, 'Output12')

    assert image_in == PortRef(
        NodeRef('7', 'Concrete'),
        'Input',
        'Image',
        1,
        '7:Concrete:Image:Input01',
        '7:Concrete:Image:Input01Value',
        None,
    )
    assert float_out == PortRef(
        NodeRef('7', 'Concrete'),
        'Output',
        'Float',
        12,
        '7:Concrete:Float:Output12',
        '7:Concrete:Float:Output12Value',
        None,
    )
    assert node.get_declared_port_refs(7) == [image_in, float_out]


def test_parameter_port_declares_input_with_default_control_tag():
    node = ConcreteNode()

    threshold = node.parameter_port(3, node.TYPE_INT, 'Input02')

    assert threshold.direction == 'Input'
    assert threshold.index == 2
    assert threshold.dpg_tag == '3:Concrete:Int:Input02'
    assert threshold.value_tag == '3:Concrete:Int:Input02Value'
    assert threshold.control_tag == '3:Concrete:Int:Input02Value'


def test_parameter_port_accepts_custom_control_tag():
    node = ConcreteNode()

    mode = node.parameter_port(
        '9',
        node.TYPE_TEXT,
        'Input05',
        control_tag='9:Concrete:Text:ModeCombo',
    )

    assert mode.node_ref.node_id == '9'
    assert mode.control_tag == '9:Concrete:Text:ModeCombo'
    assert mode.value_tag == '9:Concrete:Text:Input05Value'


def test_port_declaration_invokes_registration_callback():
    node = ConcreteNode()
    registered_ports = []
    node.set_port_registration_callback(registered_ports.append)

    declared = node.output_port(4, node.TYPE_TIME_MS, 'Output02')

    assert registered_ports == [declared]


def test_port_declaration_rejects_mismatched_direction_prefix():
    node = ConcreteNode()

    with pytest.raises(ValueError, match='must start with Output'):
        node.output_port(1, node.TYPE_IMAGE, 'Input01')


def test_port_declaration_rejects_non_numeric_index():
    node = ConcreteNode()

    with pytest.raises(ValueError, match='must end with a numeric index'):
        node.input_port(1, node.TYPE_IMAGE, 'InputMain')
