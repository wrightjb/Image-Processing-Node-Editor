# tests/test_update_node_info.py
# Standard library imports
from collections import OrderedDict
from unittest.mock import Mock, patch

# Third-party imports
import pytest

# Local application imports
from node_editor.graph_runtime import update_node_info


class FakeEditor:
    """
    A minimal interface stub for testing the update_node_info function.

    - get_node_list: returns list of node_id_name strings.
    - get_sorted_node_connection: returns dict mapping node_id_name to connection lists.
    - get_node_instance: returns an object with an update method.
    """
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


@pytest.mark.parametrize(
    "nodes, conn_dict, updates",
    [
        (
            ['1:TestNode'],
            OrderedDict([('1:TestNode', [])]),
            {'TestNode': ('img1', 'res1')}
        ),
        (
            ['1:TestNode', '2:TestNode2'],
            OrderedDict([
                ('1:TestNode', []),
                ('2:TestNode2', [['1:TestNode:out', '2:TestNode2:in']])
            ]),
            {'TestNode': ('img1', 'res1'), 'TestNode2': ('img2', 'res2')}
        )
    ]
)
def test_update_node_info_basic(nodes, conn_dict, updates):
    # Set up mocks based on parameterized update values
    inst_map = {}
    for node_name, (img_val, res_val) in updates.items():
        m = Mock()
        # Ensure initialization step set dict entry to None
        def make_side_effect(key, img_val, res_val):
            def side_effect(node_id, conn, img_dict, res_dict):
                assert img_dict[key] is None
                return (img_val, res_val)
            return side_effect

        # Find the full key (e.g. '1:TestNode') for this node_name
        key = next(k for k in nodes if k.split(':')[1] == node_name)
        m.update.side_effect = make_side_effect(key, img_val, res_val)
        inst_map[node_name] = m

    editor = FakeEditor(nodes, conn_dict, inst_map)
    image_dict = {}
    result_dict = {}

    # Run synchronously to avoid sys.exit
    update_node_info(editor, image_dict, result_dict, mode_async=False)

    # Verify final image and result dictionaries
    for node_name, (img_val, res_val) in updates.items():
        key = next(k for k in nodes if k.split(':')[1] == node_name)
        assert image_dict[key] == img_val
        assert result_dict[key] == res_val
        assert inst_map[node_name].update.call_count == 1


def test_update_node_info_async_exception_recovers_without_exit():
    class BadNode:
        def update(self, *args, **kwargs):
            raise ValueError("boom")

    nodes = ['1:TestNode']
    conn_dict = OrderedDict([('1:TestNode', [])])
    inst_map = {'TestNode': BadNode()}
    editor = FakeEditor(nodes, conn_dict, inst_map)

    image_dict = {}
    result_dict = {}
    with patch('sys.exit') as exit_mock:
        update_node_info(editor, image_dict, result_dict, mode_async=True)

    exit_mock.assert_not_called()
    assert image_dict['1:TestNode'] is None
    assert result_dict['1:TestNode'] is None


def test_update_node_info_sync_exception():
    """
    When mode_async=False, exceptions from node.update should propagate.
    """
    class BadNode:
        def update(self, *args, **kwargs):
            raise RuntimeError("sync boom")

    nodes = ['1:TestNode']
    conn_dict = OrderedDict([('1:TestNode', [])])
    inst_map = {'TestNode': BadNode()}
    editor = FakeEditor(nodes, conn_dict, inst_map)

    image_dict = {}
    result_dict = {}
    with pytest.raises(RuntimeError) as exc_info:
        update_node_info(editor, image_dict, result_dict, mode_async=False)
    assert "sync boom" in str(exc_info.value)



def test_update_node_info_uses_cache_when_signature_unchanged():
    source_node = Mock()
    source_node.update.return_value = ('src-img', {'source': 1})
    source_node.get_setting_dict.return_value = {'alpha': 0.1}

    process_node = Mock()
    process_node.update.return_value = ('img1', {'v': 1})
    process_node.get_setting_dict.return_value = {'alpha': 0.5}

    nodes = ['1:SourceNode', '2:TestNode']
    conn_dict = OrderedDict([
        ('1:SourceNode', []),
        ('2:TestNode', [['1:SourceNode:image:Output01', '2:TestNode:image:Input01']]),
    ])
    editor = FakeEditor(
        nodes,
        conn_dict,
        {'SourceNode': source_node, 'TestNode': process_node},
    )

    image_dict = {}
    result_dict = {}
    cache_dict = {}

    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )
    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )

    assert source_node.update.call_count == 2
    assert process_node.update.call_count == 1
    assert image_dict['2:TestNode'] == 'img1'
    assert result_dict['2:TestNode'] == {'v': 1}


def test_update_node_info_invalidates_cache_when_setting_changes():
    source_node = Mock()
    source_node.update.return_value = ('src-img', {'source': 1})
    source_node.get_setting_dict.return_value = {'alpha': 0.1}

    process_node = Mock()
    process_node.update.side_effect = [('img1', {'v': 1}), ('img2', {'v': 2})]
    process_node.get_setting_dict.side_effect = [{'alpha': 0.5}, {'alpha': 0.7}]

    nodes = ['1:SourceNode', '2:TestNode']
    conn_dict = OrderedDict([
        ('1:SourceNode', []),
        ('2:TestNode', [['1:SourceNode:image:Output01', '2:TestNode:image:Input01']]),
    ])
    editor = FakeEditor(
        nodes,
        conn_dict,
        {'SourceNode': source_node, 'TestNode': process_node},
    )

    image_dict = {}
    result_dict = {}
    cache_dict = {}

    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )
    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )

    assert process_node.update.call_count == 2
    assert image_dict['2:TestNode'] == 'img2'
    assert result_dict['2:TestNode'] == {'v': 2}


def test_update_node_info_does_not_cache_source_nodes_without_inputs():
    node = Mock()
    node.update.return_value = ('img1', {'v': 1})
    node.get_setting_dict.return_value = {'alpha': 0.5}

    nodes = ['1:TestNode']
    conn_dict = OrderedDict([('1:TestNode', [])])
    editor = FakeEditor(nodes, conn_dict, {'TestNode': node})

    image_dict = {}
    result_dict = {}
    cache_dict = {}

    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )
    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )

    assert node.update.call_count == 2
    assert cache_dict == {}


def test_update_node_info_cleans_cache_for_deleted_nodes():
    node = Mock()
    node.update.return_value = ('img1', {'v': 1})
    node.get_setting_dict.return_value = {'alpha': 0.5}

    nodes = ['1:TestNode']
    conn_dict = OrderedDict([('1:TestNode', [])])
    editor = FakeEditor(nodes, conn_dict, {'TestNode': node})

    image_dict = {}
    result_dict = {}
    cache_dict = {}

    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )

    editor.nodes = []
    editor.conn_dict = OrderedDict([])
    update_node_info(
        editor,
        image_dict,
        result_dict,
        node_cache_dict=cache_dict,
        mode_async=False,
    )

    assert cache_dict == {}

def test_update_node_info_ignores_stale_connections():
    source_node = Mock()
    source_node.update.return_value = ('src-img', None)
    source_node.get_setting_dict.return_value = {}

    process_node = Mock()
    process_node.update.return_value = ('out-img', None)
    process_node.get_setting_dict.return_value = {}

    nodes = ['2:ProcessNode']
    conn_dict = OrderedDict([
        ('2:ProcessNode', [['1:SourceNode:Image:Output01', '2:ProcessNode:Image:Input01']]),
    ])
    editor = FakeEditor(nodes, conn_dict, {'ProcessNode': process_node})

    image_dict = {}
    result_dict = {}

    update_node_info(editor, image_dict, result_dict, mode_async=False)

    process_node.update.assert_called_once_with(
        '2',
        [],
        image_dict,
        result_dict,
    )


def test_update_node_info_removes_deleted_node_outputs():
    node = Mock()
    node.update.return_value = ('img', {'ok': True})
    node.get_setting_dict.return_value = {}

    nodes = ['2:ProcessNode']
    conn_dict = OrderedDict([('2:ProcessNode', [])])
    editor = FakeEditor(nodes, conn_dict, {'ProcessNode': node})

    image_dict = {
        '1:OldNode': 'stale',
        '2:ProcessNode': None,
    }
    result_dict = {
        '1:OldNode': {'stale': True},
    }

    update_node_info(editor, image_dict, result_dict, mode_async=False)

    assert '1:OldNode' not in image_dict
    assert '1:OldNode' not in result_dict
    assert image_dict['2:ProcessNode'] == 'img'


def test_update_node_info_skips_inactive_nodes_during_tick():
    class FakeEditorWithActive(FakeEditor):
        def __init__(self, nodes, conn_dict, inst_map, active_set):
            super().__init__(nodes, conn_dict, inst_map)
            self.active_set = active_set

        def is_node_active(self, node_id_name):
            return node_id_name in self.active_set

    node = Mock()
    node.update.return_value = ('img', None)
    node.get_setting_dict.return_value = {}

    nodes = ['1:NodeA']
    conn_dict = OrderedDict([('1:NodeA', [])])
    editor = FakeEditorWithActive(nodes, conn_dict, {'NodeA': node}, active_set=set())

    image_dict = {}
    result_dict = {}

    update_node_info(editor, image_dict, result_dict, mode_async=False)

    node.update.assert_not_called()
