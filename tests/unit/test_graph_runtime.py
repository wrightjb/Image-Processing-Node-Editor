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


def test_graph_runtime_policy_cache_disabled_runs_every_step():
    source_node = Mock()
    source_node.update.return_value = ('src-img', {'source': 1})

    process_node = Mock()
    process_node.update.return_value = ('img', {'v': 1})
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

    runtime = GraphRuntime(cache_enabled=False)
    runtime.step(editor, mode_async=False)
    runtime.step(editor, mode_async=False)

    assert process_node.update.call_count == 2
    assert runtime.node_cache_dict == {}


def test_graph_runtime_policy_cache_source_nodes():
    source_node = Mock()
    source_node.update.return_value = ('src-img', {'source': 1})
    source_node.get_setting_dict.return_value = {'seed': 1}

    nodes = ['1:SourceNode']
    conn_dict = OrderedDict([('1:SourceNode', [])])
    editor = FakeEditor(nodes, conn_dict, {'SourceNode': source_node})

    runtime = GraphRuntime(cache_enabled=True, cache_source_nodes=True)
    runtime.step(editor, mode_async=False)
    runtime.step(editor, mode_async=False)

    assert source_node.update.call_count == 1
    assert '1:SourceNode' in runtime.node_cache_dict
