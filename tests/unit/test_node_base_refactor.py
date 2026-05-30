from node.node_abc import DpgNodeABC, DpgNodeBase
from node.port_model import LinkConnectionAdapter, LinkRef, NodeRef, PortRef
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


def test_concrete_base_node_tag_helpers_compose_port_and_value_tags():
    node = IntValueNode()

    assert node._node_port_tag(3, node.TYPE_INT, 'Output01') == (
        '3:IntValue:Int:Output01'
    )
    assert node._port_value_tag('3:IntValue', node.TYPE_INT, 'Output01') == (
        '3:IntValue:Int:Output01Value'
    )
    assert node._node_value_tag(3, node.TYPE_INT, 'Output01') == (
        '3:IntValue:Int:Output01Value'
    )


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


def test_concrete_base_iter_connections_accepts_typed_link_adapter():
    node = IntValueNode()
    link_ref = LinkRef(
        PortRef(
            node_ref=NodeRef('1', 'Src'),
            direction='Output',
            data_type='Int',
            index=1,
            port_name='Output01',
            dpg_tag='1:Src:Int:Output01',
        ),
        PortRef(
            node_ref=NodeRef('2', 'IntValue'),
            direction='Input',
            data_type='Int',
            index=1,
            port_name='Input01',
            dpg_tag='2:IntValue:Int:Input01',
        ),
    )

    assert list(node._iter_connections([LinkConnectionAdapter(link_ref)])) == [
        ('1:Src:Int:Output01', '2:IntValue:Int:Input01', 'Int'),
    ]


def test_concrete_base_iter_connection_infos_exposes_typed_adapter():
    node = IntValueNode()
    link_ref = LinkRef(
        PortRef(
            node_ref=NodeRef('1', 'Src'),
            direction='Output',
            data_type='Int',
            index=1,
            port_name='Output01',
            dpg_tag='1:Src:Int:Output01',
        ),
        PortRef(
            node_ref=NodeRef('2', 'IntValue'),
            direction='Input',
            data_type='Int',
            index=1,
            port_name='Input01',
            dpg_tag='2:IntValue:Int:Input01',
        ),
    )
    adapter = LinkConnectionAdapter(link_ref)

    assert list(node._iter_connection_infos([adapter])) == [
        (adapter, '1:Src:Int:Output01', '2:IntValue:Int:Input01', 'Int'),
    ]


def test_concrete_base_iter_connections_tags_preserve_typed_metadata():
    node = IntValueNode()
    source = PortRef(
        node_ref=NodeRef('1', 'Src'),
        direction='Output',
        data_type='Int',
        index=1,
        port_name='Output01',
        dpg_tag='legacy:source:Int:Output99',
        value_tag='1:Src:Int:Output01Value',
    )
    destination = PortRef(
        node_ref=NodeRef('2', 'IntValue'),
        direction='Input',
        data_type='Int',
        index=1,
        port_name='Input01',
        dpg_tag='legacy:dest:Int:Input99',
        value_tag='2:IntValue:Int:Input01Value',
    )
    source_tag, destination_tag, connection_type = next(
        node._iter_connections([LinkConnectionAdapter(LinkRef(source, destination))])
    )

    assert connection_type == 'Int'
    assert str(source_tag) == 'legacy:source:Int:Output99'
    assert node._extract_source_node_key(source_tag) == '1:Src'
    assert node._extract_node_id(source_tag) == '1'
    assert node._extract_port_name(destination_tag) == 'Input01'
    assert node._value_tag(source_tag) == '1:Src:Int:Output01Value'


def test_direct_node_add_node_graph_attributes_use_port_declarations():
    import re
    from pathlib import Path

    node_root = Path(__file__).parents[2] / 'node'
    failures = []
    for path in sorted(node_root.rglob('*')):
        if not (path.name.endswith('.py') or path.name.endswith('.py.disable')):
            continue
        source = path.read_text(encoding='utf-8')
        if 'class Node(DpgNodeBase)' not in source:
            continue
        add_node_match = re.search(r'(?m)^    def add_node\(', source)
        if add_node_match is None:
            continue
        next_method_match = re.search(
            r'(?m)^    def \w+\(',
            source[add_node_match.end():],
        )
        if next_method_match is None:
            add_node_source = source[add_node_match.start():]
        else:
            add_node_source = source[
                add_node_match.start():add_node_match.end() + next_method_match.start()
            ]
        declared_tag_variables = {
            declaration.group(1)
            for declaration in re.finditer(
                r'(\w+)_port = self\.(?:input_port|output_port)\(',
                add_node_source,
            )
        }
        lines = add_node_source.splitlines()
        index = 0
        while index < len(lines):
            if 'with dpg.node_attribute(' not in lines[index]:
                index += 1
                continue
            call_lines = [lines[index]]
            if '):' not in lines[index]:
                index += 1
                while index < len(lines):
                    call_lines.append(lines[index])
                    if lines[index].strip() == '):':
                        break
                    index += 1
            call_source = '\n'.join(call_lines)
            tag_match = re.search(r'tag=(\w+)', call_source)
            attr_match = re.search(
                r'attribute_type=dpg\.mvNode_Attr_(Input|Output)',
                call_source,
            )
            if (
                tag_match is not None
                and attr_match is not None
                and tag_match.group(1) not in declared_tag_variables
            ):
                failures.append(
                    f'{path.relative_to(node_root.parent)}: '
                    f'{attr_match.group(1)} {tag_match.group(1)}'
                )
            index += 1

    assert failures == []
