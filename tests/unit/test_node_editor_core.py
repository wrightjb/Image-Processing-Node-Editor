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
        dpg.get_mouse_pos.return_value = [0, 0]
        dpg.get_viewport_client_width.return_value = 1280
        dpg.get_viewport_client_height.return_value = 720
        dpg.does_item_exist.return_value = False
        dpg.is_item_hovered.return_value = False
        dpg.is_item_shown.return_value = False
        dpg.get_item_state.return_value = {'hovered': False}
        dpg.get_item_configuration.return_value = {}
        dpg.get_item_pos.return_value = [0, 0]
        dpg.add_node_link = Mock()
        dpg.configure_item = Mock()
        dpg.delete_item = Mock()
        dpg.focus_item = Mock()
        dpg.set_item_pos = Mock()
        dpg.set_value = Mock()
        dpg.show_item = Mock()
        dpg.hide_item = Mock()

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
    dpg.get_item_alias.side_effect = {101: '1:TestNode:Int:Output01', 102: '2:TestNode:Int:Input01'}.get
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    link_ids = [101, 102]
    editor._cntrl_link('NodeEditor', link_ids)
    assert editor._node_link_list == [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]
    dpg.add_node_link.assert_called_once_with(101, 102, parent='NodeEditor')


def test_link_prevents_duplicate_dest(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_item_alias.side_effect = {101: '1:TestNode:Int:Output01', 102: '2:TestNode:Int:Input01'}.get
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    link_ids = [101, 102]
    editor._cntrl_link('NodeEditor', link_ids)
    editor._cntrl_link('NodeEditor', link_ids)
    assert len(editor._node_link_list) == 1


def test_link_replaces_existing_dest(editor_and_dpg):
    editor, dpg = editor_and_dpg
    alias_map = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
        103: '3:TestNode:Int:Output01',
    }
    dpg.get_item_alias.side_effect = alias_map.get
    dpg.add_node_link.side_effect = ['link-1', 'link-2']

    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')

    editor._cntrl_link('NodeEditor', [101, 102])
    editor._cntrl_link('NodeEditor', [103, 102])

    assert editor._node_link_list == [['3:TestNode:Int:Output01', '2:TestNode:Int:Input01']]
    assert editor._link_view_id_map == {
        ('3:TestNode:Int:Output01', '2:TestNode:Int:Input01'): 'link-2'
    }
    dpg.add_node_link.assert_any_call(101, 102, parent='NodeEditor')
    dpg.add_node_link.assert_any_call(103, 102, parent='NodeEditor')
    dpg.delete_item.assert_called_once_with('link-1')
    assert dpg.set_value.call_args_list[-1].args == ('NodeEditorLinkFeedback', '')
    assert dpg.configure_item.call_args_list[-1].kwargs == {
        'label': 'Node editor'
    }


def test_link_mismatched_type_ignored(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01', 102: '2:TestNode:Float:Input01'
    }.get
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_link('NodeEditor', [101, 102])
    assert editor._node_link_list == []
    dpg.set_value.assert_called_with(
        'NodeEditorLinkFeedback',
        'Link rejected: Int output cannot connect to Float input.',
    )
    dpg.configure_item.assert_called_with(
        'NodeEditorWindow',
        label='Node editor | Link rejected: Int output cannot connect to Float input.',
    )


def test_link_duplicate_same_source_sets_feedback(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }.get
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')

    editor._cntrl_link('NodeEditor', [101, 102])
    editor._cntrl_link('NodeEditor', [101, 102])

    assert editor._node_link_list == [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]
    dpg.set_value.assert_called_with(
        'NodeEditorLinkFeedback',
        'Link rejected: input is already connected to that source.',
    )
    dpg.configure_item.assert_called_with(
        'NodeEditorWindow',
        label='Node editor | Link rejected: input is already connected to that source.',
    )


def test_link_invalid_payload_sets_feedback(editor_and_dpg):
    editor, dpg = editor_and_dpg

    editor._cntrl_link('NodeEditor', [101])

    assert editor._node_link_list == []
    dpg.set_value.assert_called_with(
        'NodeEditorLinkFeedback',
        'Link rejected: invalid link data from DearPyGui.',
    )
    dpg.configure_item.assert_called_with(
        'NodeEditorWindow',
        label='Node editor | Link rejected: invalid link data from DearPyGui.',
    )


def test_insert_node_into_selected_link(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }.get
    dpg.get_selected_links.return_value = ['existing-link']
    dpg.get_item_configuration.return_value = {'attr_1': 101, 'attr_2': 102}
    dpg.get_item_pos.side_effect = {
        '1:TestNode': [0, 0],
        '2:TestNode': [120, 60],
    }.get
    dpg.does_item_exist.side_effect = (
        lambda tag: tag in {
            '3:TestNode:Int:Input01',
            '3:TestNode:Int:Output01',
        }
    )
    dpg.add_node_link.side_effect = ['new-link-1', 'new-link-2']

    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._node_link_list = [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]
    editor._link_view_id_map = {
        ('1:TestNode:Int:Output01', '2:TestNode:Int:Input01'): 'existing-link'
    }

    editor._cntrl_insert_node_into_selected_link(None, None, 'TestNode')

    assert editor._node_list == ['1:TestNode', '2:TestNode', '3:TestNode']
    assert editor._node_link_list == [
        ['1:TestNode:Int:Output01', '3:TestNode:Int:Input01'],
        ['3:TestNode:Int:Output01', '2:TestNode:Int:Input01'],
    ]
    assert editor._link_view_id_map == {
        ('1:TestNode:Int:Output01', '3:TestNode:Int:Input01'): 'new-link-1',
        ('3:TestNode:Int:Output01', '2:TestNode:Int:Input01'): 'new-link-2',
    }
    dpg.add_node_link.assert_any_call(
        '1:TestNode:Int:Output01',
        '3:TestNode:Int:Input01',
        parent='NodeEditor',
    )
    dpg.add_node_link.assert_any_call(
        '3:TestNode:Int:Output01',
        '2:TestNode:Int:Input01',
        parent='NodeEditor',
    )
    dpg.delete_item.assert_called_once_with('existing-link')
    assert dpg.set_value.call_args_list[-1].args == ('NodeEditorLinkFeedback', '')


def test_insert_node_into_selected_link_requires_selection(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_selected_links.return_value = []

    editor._cntrl_insert_node_into_selected_link(None, None, 'TestNode')

    dpg.set_value.assert_called_with(
        'NodeEditorLinkFeedback',
        'Insert into link requires a selected or hovered link.',
    )


def test_open_insert_link_popup_on_right_click_with_single_selection(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_selected_links.return_value = ['existing-link']
    dpg.get_mouse_pos.return_value = [128.3, 255.9]

    editor._cntrl_open_insert_link_popup(None, None)

    dpg.set_item_pos.assert_called_with('NodeEditorInsertLinkPopup', [128, 255])
    dpg.show_item.assert_called_with('NodeEditorInsertLinkPopup')
    dpg.focus_item.assert_called_with('NodeEditorInsertLinkPopup')


def test_open_insert_link_popup_on_hovered_link(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._link_view_id_map = {
        ('1:TestNode:Int:Output01', '2:TestNode:Int:Input01'): 'link-1'
    }
    dpg.get_selected_links.return_value = []
    dpg.is_item_hovered.side_effect = lambda item: item == 'link-1'
    dpg.get_mouse_pos.return_value = [10, 20]

    editor._cntrl_open_insert_link_popup(None, None)

    dpg.show_item.assert_called_with('NodeEditorInsertLinkPopup')


def test_insert_node_into_selected_link_uses_pending_hovered_link(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }.get
    dpg.get_item_configuration.return_value = {'attr_1': 101, 'attr_2': 102}
    dpg.get_item_pos.side_effect = {'1:TestNode': [0, 0], '2:TestNode': [100, 0]}.get
    dpg.does_item_exist.side_effect = lambda tag: tag in {
        '3:TestNode:Int:Input01', '3:TestNode:Int:Output01'
    }
    dpg.add_node_link.side_effect = ['new-link-1', 'new-link-2']
    editor._pending_insert_link_dpg_id = 'existing-link'
    editor._node_link_list = [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]
    editor._link_view_id_map = {
        ('1:TestNode:Int:Output01', '2:TestNode:Int:Input01'): 'existing-link'
    }
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')

    editor._cntrl_insert_node_into_selected_link(None, None, 'TestNode')

    assert editor._pending_insert_link_dpg_id is None


def test_open_insert_link_popup_hides_when_selection_invalid(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_selected_links.return_value = []
    dpg.is_item_shown.return_value = True

    editor._cntrl_open_insert_link_popup(None, None)

    dpg.hide_item.assert_called_with('NodeEditorInsertLinkPopup')


def test_insert_link_popup_closes_on_outside_left_click(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._insert_link_popup_open = True
    dpg.get_item_state.return_value = {'hovered': False}
    dpg.is_item_shown.return_value = True
    editor._pending_insert_link_dpg_id = 'existing-link'

    editor._cntrl_close_insert_link_popup_on_left_click(None, None)

    assert editor._pending_insert_link_dpg_id is None
    dpg.hide_item.assert_called_with('NodeEditorInsertLinkPopup')


def test_insert_link_popup_stays_open_on_inside_left_click(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._insert_link_popup_open = True
    dpg.get_item_state.return_value = {'hovered': True}

    editor._cntrl_close_insert_link_popup_on_left_click(None, None)

    dpg.hide_item.assert_not_called()


def test_insert_link_popup_closes_on_escape(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._insert_link_popup_open = True
    dpg.is_item_shown.return_value = True
    editor._pending_insert_link_dpg_id = 'existing-link'

    editor._cntrl_close_insert_link_popup_on_escape(None, None)

    assert editor._pending_insert_link_dpg_id is None
    dpg.hide_item.assert_called_with('NodeEditorInsertLinkPopup')


def test_insert_node_into_selected_link_rejects_incompatible_node(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }.get
    dpg.get_selected_links.return_value = ['existing-link']
    dpg.get_item_configuration.return_value = {'attr_1': 101, 'attr_2': 102}
    dpg.get_item_pos.side_effect = {
        '1:TestNode': [0, 0],
        '2:TestNode': [120, 60],
    }.get
    dpg.does_item_exist.return_value = False

    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._node_link_list = [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]

    editor._cntrl_insert_node_into_selected_link(None, None, 'TestNode')

    assert editor._node_list == ['1:TestNode', '2:TestNode']
    assert editor._node_link_list == [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]
    dpg.delete_item.assert_called_once_with('3:TestNode')
    dpg.set_value.assert_called_with(
        'NodeEditorLinkFeedback',
        'Cannot insert TestNode: it needs both Int input and output ports.',
    )


def test_close_window(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._cntrl_close_window('window')
    dpg.delete_item.assert_called_once_with('window')


def test_sort_node_graph(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor._node_list = ['1:TestNode', '2:TestNode', '3:TestNode', '4:TestNode']
    editor._node_link_list = [
        ['1:TestNode:Output01', '2:TestNode:Input01'],
        ['1:TestNode:Output01', '3:TestNode:Input01'],
        ['2:TestNode:Output01', '4:TestNode:Input01'],
        ['3:TestNode:Output01', '4:TestNode:Input01'],
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
    editor._node_link_list = [['1:TestNode:Output01', '2:TestNode:Input01']]
    dpg.get_selected_nodes.return_value = ['1:TestNode']
    editor._cntrl_delete_selected(None, None)
    assert '1:TestNode' not in editor._node_list
    assert editor._node_link_list == []


def test_mv_key_del_no_selection(editor_and_dpg):
    editor, dpg = editor_and_dpg
    # initial state
    editor._node_list = ['1:TestNode']
    editor._node_link_list = [['1:TestNode:Output01', '1:TestNode:Input01']]
    # no nodes or links selected
    dpg.get_selected_nodes.return_value = []
    dpg.get_selected_links.return_value = []
    editor._cntrl_delete_selected(None, None)
    assert editor._node_list == ['1:TestNode']
    assert editor._node_link_list == [['1:TestNode:Output01', '1:TestNode:Input01']]


def test_set_get_terminate_flag(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor.set_terminate_flag(True)
    assert editor.get_terminate_flag() is True
    editor.set_terminate_flag(False)
    assert editor.get_terminate_flag() is False
