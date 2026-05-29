from node.node_abc import DpgNodeABC, DpgNodeBase
from node.input_node.node_int_value import Node as IntValueNode
from node.process_node.node_blur import Node as BlurNode


def test_dpg_node_base_is_compatibility_subclass():
    assert issubclass(DpgNodeBase, DpgNodeABC)


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
