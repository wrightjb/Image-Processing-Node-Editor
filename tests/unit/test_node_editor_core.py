# tests/test_node_editor_core.py
# Standard library imports
from collections import OrderedDict
from unittest.mock import Mock, patch

# Third-party imports
import pytest

# Local application imports
from node_editor.node_editor import DpgNodeEditor


class DummyNode:
    """
    A minimal fake node for core editor tests.

    Implements the node interface with predictable behavior:
    - `add_node` returns a string ID based on node_id and tag.
    - `update` returns deterministic image/result tuples.
    Other methods are no-ops or return fixed settings.
    """
    _ver = '1.0'
    node_tag = 'TestNode'
    node_label = 'Test Node'

    def add_node(self, parent, node_id, pos=None, opencv_setting_dict=None):
        return f"{node_id}:{self.node_tag}"

    def update(self, node_id, connection_list, img_dict, res_dict):
        return f"img{node_id}", f"res{node_id}"

    def get_setting_dict(self, node_id):
        return {"ver": self._ver, "pos": [0, 0]}

    def set_setting_dict(self, node_id, setting_dict):
        pass

    def close(self, node_id):
        pass


@pytest.fixture
def editor_and_dpg():
    with patch('node_editor.node_editor.dpg') as dpg, \
         patch('node_editor.node_editor.glob') as mock_glob, \
         patch('node_editor.node_editor.import_module') as mock_import:
        # set up DearPyGui mocks
        dpg.get_item_alias.side_effect = lambda x: x
        dpg.get_selected_nodes.return_value = []
        dpg.get_selected_links.return_value = []
        dpg.get_item_pos.return_value = [0, 0]
        dpg.add_node_link = Mock()
        dpg.delete_item = Mock()

        # load DummyNode from patched import
        mock_glob.return_value = ['node/test_node/test_node.py']
        mod = Mock(Node=DummyNode)
        mock_import.return_value = mod

        editor = DpgNodeEditor(use_debug_print=False)
        yield editor, dpg


def test_add_node_increments_id(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor._cntrl_add_node(None, None, 'TestNode')
    assert editor._node_id == 1
    assert editor._node_list == ['1:TestNode']


def test_add_node_uses_last_position(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_selected_nodes.return_value = ['1:TestNode']
    dpg.get_item_pos.return_value = [100, 200]
    editor._cntrl_save_last_pos(None, None)
    editor._cntrl_add_node(None, None, 'TestNode')
    dpg.get_item_pos.assert_called_once_with('1:TestNode')


def test_save_last_pos_no_selection(editor_and_dpg):
    editor, _ = editor_and_dpg
    # No nodes selected: last_pos should remain None
    editor._last_pos = None
    editor._cntrl_save_last_pos(None, None)
    assert editor._last_pos is None


def test_link_callback_basic(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    link = ['1:TestNode:Int:Out', '2:TestNode:Int:In']
    editor._cntrl_link('NodeEditor', link)
    assert editor._node_link_list == [link]
    dpg.add_node_link.assert_called_once_with(link[0], link[1], parent='NodeEditor')


def test_link_prevents_duplicate_dest(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    link = ['1:TestNode:Int:Out', '2:TestNode:Int:In']
    editor._cntrl_link('NodeEditor', link)
    editor._cntrl_link('NodeEditor', link)
    assert len(editor._node_link_list) == 1


def test_link_mismatched_type_ignored(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_link('NodeEditor', ['1:TestNode:Int:Out', '2:TestNode:Float:In'])
    assert editor._node_link_list == []


def test_close_window(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._cntrl_close_window('window')
    dpg.delete_item.assert_called_once_with('window')


def test_sort_node_graph(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor._node_list = ['1:TestNode', '2:TestNode', '3:TestNode', '4:TestNode']
    editor._node_link_list = [
        ['1:TestNode:out', '2:TestNode:in'],
        ['1:TestNode:out', '3:TestNode:in'],
        ['2:TestNode:out', '4:TestNode:in'],
        ['3:TestNode:out', '4:TestNode:in'],
    ]
    editor._mdl_sort_node_graph()
    result = editor.get_sorted_node_connection()
    assert list(result.keys()) == ['1:TestNode', '2:TestNode', '3:TestNode', '4:TestNode']


def test_sort_node_graph_unconnected(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor._node_list = ['1:TestNode', '2:TestNode']
    editor._node_link_list = []
    editor._mdl_sort_node_graph()
    result = editor.get_sorted_node_connection()
    assert result == OrderedDict()


def test_sort_node_graph_empty_list(editor_and_dpg):
    editor, _ = editor_and_dpg
    # Empty node list should yield empty result
    editor._node_list = []
    editor._node_link_list = []
    editor._mdl_sort_node_graph()
    result = editor.get_sorted_node_connection()
    assert result == OrderedDict()


def test_save_last_pos(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_selected_nodes.return_value = ['1:TestNode']
    dpg.get_item_pos.return_value = [50, 60]
    editor._cntrl_save_last_pos(None, None)
    assert editor._last_pos == [50, 60]


def test_delete_node(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._node_link_list = [['1:TestNode:out', '2:TestNode:in']]
    dpg.get_selected_nodes.return_value = ['1:TestNode']
    editor._cntrl_delete_selected(None, None)
    assert '1:TestNode' not in editor._node_list
    assert editor._node_link_list == []


def test_mv_key_del_no_selection(editor_and_dpg):
    editor, dpg = editor_and_dpg
    # initial state
    editor._node_list = ['1:TestNode']
    editor._node_link_list = [['1:TestNode:out', '1:TestNode:in']]
    # no nodes or links selected
    dpg.get_selected_nodes.return_value = []
    dpg.get_selected_links.return_value = []
    editor._cntrl_delete_selected(None, None)
    assert editor._node_list == ['1:TestNode']
    assert editor._node_link_list == [['1:TestNode:out', '1:TestNode:in']]


def test_set_get_terminate_flag(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor.set_terminate_flag(True)
    assert editor.get_terminate_flag() is True
    editor.set_terminate_flag(False)
    assert editor.get_terminate_flag() is False
