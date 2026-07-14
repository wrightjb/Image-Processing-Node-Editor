from node.port_model import LinkConnectionAdapter, LinkRef, NodeRef, PortRef
import node.analysis_node.node_compression_inspect as inspect_module
from node.analysis_node.node_compression_inspect import Node


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


def test_compression_inspect_reads_typed_metadata_connection(monkeypatch):
    node = Node()
    ports = node.create_ports('2')
    written = {}
    monkeypatch.setattr(
        inspect_module,
        'dpg_set_value',
        lambda tag, value: written.setdefault(tag, value),
    )
    source = _port_ref('1', 'Image', 'Output', 'Metadata', 'Output02')
    link = LinkConnectionAdapter(LinkRef(source, ports.metadata_input))
    metadata = {
        'format': 'JPEG',
        'filename': 'target.jpg',
        'jpeg': {
            'width': 800,
            'height': 600,
            'subsampling': '4:2:0',
            'estimated_quality': 90,
            'quality_confidence': 'heuristic',
            'progressive': False,
            'quantization_tables': {'0': {'values': []}},
        },
    }

    _, result = node.update(
        '2',
        [link],
        {},
        {'1:Image': {'metadata': metadata}},
    )

    assert result == {'metadata': metadata}
    assert 'Format: JPEG' in written[node._report_value_tag('2')]
    assert written[ports.metadata.value_tag] == 'JPEG, 800x600, 4:2:0, Q≈90, baseline'
