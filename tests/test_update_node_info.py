import pytest
from collections import OrderedDict
from unittest.mock import Mock

from main import update_node_info


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


def test_update_node_info_basic():
    node1 = Mock()
    node2 = Mock()
    node1.update.return_value = ('img1', 'res1')
    node2.update.return_value = ('img2', 'res2')
    editor = FakeEditor(
        ['1:TestNode', '2:TestNode'],
        OrderedDict({
            '1:TestNode': [],
            '2:TestNode': [['1:TestNode:out', '2:TestNode:in']],
        }),
        {'TestNode': node1}
    )
    editor.inst_map['TestNode'] = node1
    # second node uses same class
    editor.inst_map['TestNode2'] = node2
    # Map each node name to instance
    def get_instance(name):
        return node1 if name == 'TestNode' else node2
    editor.get_node_instance = get_instance

    image_dict = {}
    result_dict = {}
    update_node_info(editor, image_dict, result_dict, mode_async=False)

    assert image_dict['1:TestNode'] == 'img1'
    assert image_dict['2:TestNode'] == 'img2'
    assert result_dict['1:TestNode'] == 'res1'
    assert result_dict['2:TestNode'] == 'res2'
    assert node1.update.called
    assert node2.update.called
