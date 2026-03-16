from collections import OrderedDict
from unittest.mock import Mock

from node_editor.graph_runtime import GraphRuntime


class FakeEditor:
    def __init__(self, nodes, conn_dict, inst_map):
        self.nodes = nodes
        self.conn_dict = conn_dict
        self.inst_map = inst_map

    def get_node_list(self):
        return self.nodes

    def get_sorted_node_connection(self):
        return self.conn_dict

    def get_node_instance(self, name):
        return self.inst_map[name]


def test_graph_runtime_persists_state_between_steps():
    source_node = Mock()
    source_node.update.return_value = ('src-img', {'source': 1})

    process_node = Mock()
    process_node.update.return_value = ('img1', {'v': 1})
    process_node.get_setting_dict.return_value = {'alpha': 0.5}

    nodes = ['1:SourceNode', '2:ProcessNode']
    conn_dict = OrderedDict([
        ('1:SourceNode', []),
        ('2:ProcessNode', [['1:SourceNode:Image:Output01', '2:ProcessNode:Image:Input01']]),
    ])
    editor = FakeEditor(
        nodes,
        conn_dict,
        {'SourceNode': source_node, 'ProcessNode': process_node},
    )

    runtime = GraphRuntime()
    runtime.step(editor, mode_async=False)
    runtime.step(editor, mode_async=False)

    assert source_node.update.call_count == 2
    assert process_node.update.call_count == 1
    assert runtime.node_image_dict['2:ProcessNode'] == 'img1'
    assert runtime.node_result_dict['2:ProcessNode'] == {'v': 1}
