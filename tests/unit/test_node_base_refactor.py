from node.node_abc import DpgNodeABC, DpgNodeBase
from node.input_node.node_int_value import Node as IntValueNode
from node.process_node.node_blur import Node as BlurNode


def test_dpg_node_base_is_concrete_helper_subclass():
    assert issubclass(DpgNodeBase, DpgNodeABC)


def test_abstract_interface_no_longer_owns_concrete_helpers():
    for helper_name in (
        '_node_name',
        '_port_tag',
        '_value_tag',
        '_iter_connections',
        'add_editor_toolbar',
    ):
        assert helper_name not in DpgNodeABC.__dict__
        assert helper_name in DpgNodeBase.__dict__


def test_representative_direct_nodes_inherit_concrete_base():
    assert isinstance(IntValueNode(), DpgNodeBase)
    assert isinstance(BlurNode(), DpgNodeBase)


def test_concrete_base_preserves_existing_tag_helpers():
    node = IntValueNode()
    node_name = node._node_name(3)
    port_tag = node._port_tag(node_name, node.TYPE_INT, 'Output01')

    assert node_name == '3:IntValue'
    assert port_tag == '3:IntValue:Int:Output01'
    assert node._value_tag(port_tag) == '3:IntValue:Int:Output01Value'


def test_concrete_base_preserves_connection_iteration_guards():
    node = IntValueNode()

    assert list(node._iter_connections([
        ['1:Src:Int:Output01', '2:IntValue:Int:Input01'],
        ['malformed-source', '2:IntValue:Int:Input01'],
        ['1:Src:Int:Output02'],
        [None, '2:IntValue:Int:Input02'],
    ])) == [
        ('1:Src:Int:Output01', '2:IntValue:Int:Input01', 'Int'),
    ]
