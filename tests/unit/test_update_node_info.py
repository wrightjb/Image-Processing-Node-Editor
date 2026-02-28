# tests/test_update_node_info.py
# Standard library imports
from collections import OrderedDict
from unittest.mock import Mock, patch

# Third-party imports
import pytest

# Local application imports
from main import update_node_info


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


def test_update_node_info_async_exception():
    class BadNode:
        def update(self, *args, **kwargs):
            raise ValueError("boom")

    nodes = ['1:TestNode']
    conn_dict = OrderedDict([('1:TestNode', [])])
    inst_map = {'TestNode': BadNode()}
    editor = FakeEditor(nodes, conn_dict, inst_map)

    with patch('sys.exit', side_effect=SystemExit) as exit_mock:
        with pytest.raises(SystemExit):
            update_node_info(editor, {}, {}, mode_async=True)
        assert exit_mock.called


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

    assert node.update.call_count == 1
    assert image_dict['1:TestNode'] == 'img1'
    assert result_dict['1:TestNode'] == {'v': 1}


def test_update_node_info_invalidates_cache_when_setting_changes():
    node = Mock()
    node.update.side_effect = [('img1', {'v': 1}), ('img2', {'v': 2})]
    node.get_setting_dict.side_effect = [{'alpha': 0.5}, {'alpha': 0.7}]

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
    assert image_dict['1:TestNode'] == 'img2'
    assert result_dict['1:TestNode'] == {'v': 2}


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
