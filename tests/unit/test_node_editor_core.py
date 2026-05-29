# tests/test_node_editor_core.py
# Standard library imports
from collections import OrderedDict
from unittest.mock import Mock, patch

# Third-party imports
import pytest

# Local application imports
from node.node_abc import DpgNodeBase
from node.port_model import NodeRef, PortRef
from node_editor.node_editor import (
    AddNodeCommand,
    CompositeCommand,
    DpgNodeEditor,
    RemoveLinkCommand,
    SetParameterCommand,
)


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

    def add_node(
        self,
        parent,
        node_id,
        pos=None,
        opencv_setting_dict=None,
        callback=None,
    ):
        del callback
        return f"{node_id}:{self.node_tag}"

    def update(self, node_id, connection_list, img_dict, res_dict):
        return f"img{node_id}", f"res{node_id}"

    def get_setting_dict(self, node_id):
        return {"ver": self._ver, "pos": [0, 0]}

    def set_setting_dict(self, node_id, setting_dict):
        pass

    def close(self, node_id):
        pass


class PortDeclaringDummyNode(DpgNodeBase):
    _ver = '1.0'
    node_tag = 'TestNode'
    node_label = 'Test Node'

    def add_node(
        self,
        parent,
        node_id,
        pos=None,
        opencv_setting_dict=None,
        callback=None,
    ):
        del parent, pos, opencv_setting_dict, callback
        self.input_port(node_id, self.TYPE_IMAGE, 'Input01')
        self.output_port(node_id, self.TYPE_IMAGE, 'Output01')
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
    assert ('1:TestNode:Int:Output01', '2:TestNode:Int:Input01') in editor._link_registry
    assert editor._link_by_dest_port['2:TestNode:Int:Input01'] == '1:TestNode:Int:Output01'
    dpg.add_node_link.assert_called_once_with(101, 102, parent='NodeEditor')


def test_add_node_registers_declared_port_refs(editor_and_dpg):
    editor, _ = editor_and_dpg
    editor._node_instance_list['TestNode'] = PortDeclaringDummyNode()

    editor._cntrl_add_node(None, None, 'TestNode')

    input_port = editor._port_registry['1:TestNode:Image:Input01']
    output_port = editor._port_registry['1:TestNode:Image:Output01']
    assert input_port.node_ref.node_id_name == '1:TestNode'
    assert input_port.direction == 'Input'
    assert input_port.port_name == 'Input01'
    assert output_port.direction == 'Output'
    assert output_port.port_name == 'Output01'
    assert editor._node_registry['1:TestNode'] == input_port.node_ref


def test_add_node_keeps_dynamic_port_registration_callback(editor_and_dpg):
    editor, _ = editor_and_dpg
    node = PortDeclaringDummyNode()
    editor._node_instance_list['TestNode'] = node

    editor._cntrl_add_node(None, None, 'TestNode')
    node.input_port(1, node.TYPE_TEXT, 'Input02')

    assert '1:TestNode:Text:Input02' in editor._port_registry


def test_registered_hovered_output_port_beyond_legacy_scan_range(editor_and_dpg):
    editor, dpg = editor_and_dpg
    output_port = PortRef(
        node_ref=NodeRef('1', 'TestNode'),
        direction='Output',
        data_type='Int',
        index=125,
        port_name='Output125',
        dpg_tag='1:TestNode:Int:Output125',
        value_tag='1:TestNode:Int:Output125Value',
    )
    editor._mdl_register_port_ref(output_port)
    dpg.does_item_exist.side_effect = lambda tag: tag == output_port.dpg_tag
    dpg.is_item_hovered.side_effect = lambda tag: tag == output_port.dpg_tag

    assert editor._cntrl_get_hovered_output_port_tag() == output_port.dpg_tag


def test_find_node_port_uses_registered_port_beyond_legacy_scan_range(editor_and_dpg):
    editor, dpg = editor_and_dpg
    input_port = PortRef(
        node_ref=NodeRef('1', 'TestNode'),
        direction='Input',
        data_type='Image',
        index=125,
        port_name='Input125',
        dpg_tag='1:TestNode:Image:Input125',
        value_tag='1:TestNode:Image:Input125Value',
    )
    editor._mdl_register_port_ref(input_port)
    dpg.does_item_exist.side_effect = lambda tag: tag == input_port.dpg_tag
    dpg.get_item_configuration.side_effect = lambda tag: (
        {'attribute_type': dpg.mvNode_Attr_Input}
        if tag == input_port.dpg_tag
        else {}
    )

    assert (
        editor._cntrl_find_node_port('1:TestNode', 'Image', 'Input')
        == input_port.dpg_tag
    )


def test_find_node_port_falls_back_to_legacy_scan_and_registers(editor_and_dpg):
    editor, dpg = editor_and_dpg
    input_tag = '1:TestNode:Image:Input01'
    dpg.does_item_exist.side_effect = lambda tag: tag == input_tag
    dpg.get_item_configuration.side_effect = lambda tag: (
        {'attribute_type': dpg.mvNode_Attr_Input}
        if tag == input_tag
        else {}
    )

    assert editor._cntrl_find_node_port('1:TestNode', 'Image', 'Input') == input_tag
    assert editor._port_registry[input_tag].port_name == 'Input01'


def test_parse_port_tag_returns_typed_metadata(editor_and_dpg):
    editor, _ = editor_and_dpg
    port = editor._cntrl_parse_port_tag('7:TestNode:Float:Input03')
    assert port is not None
    assert port.node_ref.node_id == '7'
    assert port.node_ref.node_tag == 'TestNode'
    assert port.node_ref.node_id_name == '7:TestNode'
    assert port.direction == 'Input'
    assert port.data_type == 'Float'
    assert port.index == 3
    assert port.port_name == 'Input03'
    assert port.value_tag == '7:TestNode:Float:Input03Value'
    assert editor._port_registry['7:TestNode:Float:Input03'] == port


def test_delete_link_updates_typed_registries(editor_and_dpg):
    editor, _ = editor_and_dpg
    source_tag = '1:TestNode:Int:Output01'
    dest_tag = '2:TestNode:Int:Input01'
    assert editor._mdl_add_link(source_tag, dest_tag) is True
    assert editor._mdl_get_link_by_destination(dest_tag) == [source_tag, dest_tag]

    editor._mdl_delete_link([source_tag, dest_tag])

    assert editor._mdl_get_link_by_destination(dest_tag) is None
    assert (source_tag, dest_tag) not in editor._link_registry
    assert dest_tag not in editor._link_by_dest_port


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
    alias_map = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }
    dpg.get_item_alias.side_effect = lambda item: alias_map.get(item, item)
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


def test_link_cycle_is_rejected(editor_and_dpg):
    editor, dpg = editor_and_dpg
    alias_map = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
        103: '2:TestNode:Int:Output01',
        104: '1:TestNode:Int:Input01',
    }
    dpg.get_item_alias.side_effect = alias_map.get
    dpg.does_item_exist.side_effect = lambda _tag: True
    dpg.add_node_link.side_effect = ['link-a']

    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')

    editor._cntrl_link('NodeEditor', [101, 102])
    assert editor._node_link_list == [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]

    editor._cntrl_link('NodeEditor', [103, 104])

    assert editor._node_link_list == [['1:TestNode:Int:Output01', '2:TestNode:Int:Input01']]
    dpg.set_value.assert_any_call(
        'NodeEditorLinkFeedback',
        'Link rejected: this connection would create a cycle.',
    )


def test_insert_node_into_selected_link(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }.get
    dpg.get_selected_links.return_value = ['existing-link']
    def config_side_effect(tag):
        if tag == 'existing-link':
            return {'attr_1': 101, 'attr_2': 102}
        if tag == '3:TestNode:Int:Input01':
            return {'attribute_type': dpg.mvNode_Attr_Input}
        if tag == '3:TestNode:Int:Output01':
            return {'attribute_type': dpg.mvNode_Attr_Output}
        return {}

    dpg.get_item_configuration.side_effect = config_side_effect
    dpg.get_item_pos.side_effect = {
        '1:TestNode': [0, 0],
        '2:TestNode': [120, 60],
    }.get
    dpg.does_item_exist.side_effect = (
        lambda tag: tag in {
            '1:TestNode:Int:Output01',
            '2:TestNode:Int:Input01',
            '3:TestNode:Int:Input01',
            '3:TestNode:Int:Output01',
        }
    )
    dpg.add_node_link.side_effect = ['new-link-1', 'new-link-2', 'restored-link', 'redo-link-1', 'redo-link-2']

    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._mdl_add_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01')
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


def test_add_node_to_occupied_input_is_single_composite_undo(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda _tag: True
    dpg.get_item_pos.side_effect = lambda tag: [300, 200] if tag == '2:TestNode' else [0, 0]
    dpg.get_item_configuration.side_effect = lambda tag: (
        {'attribute_type': dpg.mvNode_Attr_Output}
        if tag == '4:TestNode:Int:Output01'
        else {'attribute_type': dpg.mvNode_Attr_Input}
    )

    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')

    # Existing 1 -> 2 link
    assert editor._mdl_add_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01')
    editor._vw_register_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01', 'link-old')

    editor._pending_add_to_input_tag = '2:TestNode:Int:Input01'
    editor._cntrl_add_node_to_input_port(None, None, 'TestNode')

    assert ['4:TestNode:Int:Output01', '2:TestNode:Int:Input01'] in editor._node_link_list
    assert ['1:TestNode:Int:Output01', '2:TestNode:Int:Input01'] not in editor._node_link_list

    editor._cntrl_undo(None, None)

    assert ['1:TestNode:Int:Output01', '2:TestNode:Int:Input01'] in editor._node_link_list
    assert '4:TestNode' not in editor._node_list


def test_insert_then_move_undo_redo_sequence(editor_and_dpg):
    editor, dpg = editor_and_dpg
    alias_map = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }
    dpg.get_item_alias.side_effect = lambda item: alias_map.get(item, item)
    dpg.get_selected_links.return_value = ['existing-link']
    pos_map = {
        '1:TestNode': [0, 0],
        '2:TestNode': [120, 60],
        '3:TestNode': [60, 30],
    }

    def config_side_effect(tag):
        if tag == 'existing-link':
            return {'attr_1': 101, 'attr_2': 102}
        if tag == '3:TestNode:Int:Input01':
            return {'attribute_type': dpg.mvNode_Attr_Input}
        if tag == '3:TestNode:Int:Output01':
            return {'attribute_type': dpg.mvNode_Attr_Output}
        return {}

    dpg.get_item_configuration.side_effect = config_side_effect
    dpg.get_item_pos.side_effect = lambda tag: pos_map.get(tag, [0, 0])
    dpg.set_item_pos.side_effect = lambda tag, pos: pos_map.__setitem__(tag, list(pos))
    dpg.does_item_exist.side_effect = lambda _tag: True
    dpg.add_node_link.side_effect = [
        'new-link-1', 'new-link-2', 'restored-link', 'redo-link-1', 'redo-link-2'
    ]

    editor._cntrl_add_node(None, None, 'TestNode')
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._mdl_add_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01')
    editor._link_view_id_map = {
        ('1:TestNode:Int:Output01', '2:TestNode:Int:Input01'): 'existing-link'
    }

    editor._cntrl_insert_node_into_selected_link(None, None, 'TestNode')
    dpg.get_selected_nodes.return_value = ['3:TestNode']
    editor._cntrl_capture_move_start_positions(None, None)
    pos_map['3:TestNode'] = [200, 150]
    editor._cntrl_commit_move_commands(None, None)

    editor._cntrl_undo(None, None)  # undo move
    assert pos_map['3:TestNode'] == [60, 30]

    editor._cntrl_undo(None, None)  # undo insert
    assert '3:TestNode' not in editor._node_list
    assert ['1:TestNode:Int:Output01', '2:TestNode:Int:Input01'] in editor._node_link_list

    editor._cntrl_redo(None, None)  # redo insert
    assert '3:TestNode' in editor._node_list
    assert ['1:TestNode:Int:Output01', '3:TestNode:Int:Input01'] in editor._node_link_list
    assert ['3:TestNode:Int:Output01', '2:TestNode:Int:Input01'] in editor._node_link_list

    editor._cntrl_redo(None, None)  # redo move
    assert pos_map['3:TestNode'] == [200, 150]


def test_history_menu_labels_track_undo_redo_top(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda tag: tag in {'Menu_Edit_Undo', 'Menu_Edit_Redo'}

    editor._cntrl_add_node(None, None, 'TestNode')

    dpg.configure_item.assert_any_call(
        'Menu_Edit_Undo',
        label='Undo (Add node: TestNode)',
        enabled=True,
    )
    dpg.configure_item.assert_any_call(
        'Menu_Edit_Redo',
        label='Redo',
        enabled=False,
    )

    editor._cntrl_undo(None, None)

    dpg.configure_item.assert_any_call(
        'Menu_Edit_Undo',
        label='Undo',
        enabled=False,
    )
    dpg.configure_item.assert_any_call(
        'Menu_Edit_Redo',
        label='Redo (Add node: TestNode)',
        enabled=True,
    )


def test_composite_insert_label_prefers_insert_node(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda tag: tag == 'Menu_Edit_Undo'
    editor._cntrl_push_undo_command(
        CompositeCommand([
            RemoveLinkCommand('1:TestNode:Int:Output01', '2:TestNode:Int:Input01'),
            AddNodeCommand(3, 'TestNode', [0, 0], {}, [], []),
        ])
    )
    dpg.configure_item.assert_any_call(
        'Menu_Edit_Undo',
        label='Undo (Insert node: TestNode)',
        enabled=True,
    )


class DuckHistoryCommand:
    def __init__(self):
        self.calls = []

    def undo(self, editor):
        self.calls.append(('undo', editor))

    def redo(self, editor):
        self.calls.append(('redo', editor))


def test_undo_redo_dispatches_to_command_methods_without_type_branching(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.return_value = False
    command = DuckHistoryCommand()

    editor._undo_stack.append(command)
    editor._cntrl_undo(None, None)

    assert command.calls == [('undo', editor)]
    assert editor._redo_stack == [command]

    editor._cntrl_redo(None, None)

    assert command.calls == [('undo', editor), ('redo', editor)]
    assert editor._undo_stack == [command]
    assert editor._suspend_parameter_history is False


def test_new_command_after_fully_undone_history_clears_stale_remap(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.return_value = False
    editor._redo_stack.append(DuckHistoryCommand())
    editor._history_node_id_remap['1:TestNode'] = '9:TestNode'

    editor._cntrl_push_undo_command(DuckHistoryCommand())

    assert editor._redo_stack == []
    assert editor._history_node_id_remap == {}


def _configure_shortcut_keys(dpg, pressed_keys):
    dpg.mvKey_LControl = 'LControl'
    dpg.mvKey_RControl = 'RControl'
    dpg.mvKey_LShift = 'LShift'
    dpg.mvKey_RShift = 'RShift'
    dpg.is_key_down.side_effect = lambda key: key in pressed_keys


def test_keyboard_shortcuts_use_standard_undo_redo_modifiers(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.return_value = False

    undo_command = DuckHistoryCommand()
    editor._undo_stack.append(undo_command)
    _configure_shortcut_keys(dpg, {'LControl'})

    editor._cntrl_keyboard_z_shortcut(None, None)

    assert undo_command.calls == [('undo', editor)]
    assert editor._redo_stack == [undo_command]

    _configure_shortcut_keys(dpg, {'RControl', 'RShift'})
    editor._cntrl_keyboard_z_shortcut(None, None)

    assert undo_command.calls == [('undo', editor), ('redo', editor)]
    assert editor._undo_stack == [undo_command]

    _configure_shortcut_keys(dpg, set())
    editor._cntrl_keyboard_z_shortcut(None, None)
    assert undo_command.calls == [('undo', editor), ('redo', editor)]


def test_parameter_change_coalesces_numeric_edits_and_undo_redo(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda _tag: True
    param_tag = '1:TestNode:Int:Input01Value'
    value_state = {param_tag: 5}

    def set_value_side_effect(tag, value):
        value_state[tag] = value

    def get_value_side_effect(tag):
        return value_state.get(tag)

    dpg.set_value.side_effect = set_value_side_effect
    dpg.get_value.side_effect = get_value_side_effect
    editor._cntrl_add_node(None, None, 'TestNode')
    editor._undo_stack.clear()
    editor._redo_stack.clear()

    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:TestNode',
            'value_tag': param_tag,
            'before_value': 5,
            'after_value': 6,
        },
    )
    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:TestNode',
            'value_tag': param_tag,
            'before_value': 6,
            'after_value': 7,
        },
    )

    assert len(editor._undo_stack) == 1
    assert isinstance(editor._undo_stack[-1], SetParameterCommand)
    assert editor._undo_stack[-1].before_value == 5
    assert editor._undo_stack[-1].after_value == 7

    editor._cntrl_undo(None, None)
    assert value_state[param_tag] == 5
    editor._cntrl_redo(None, None)
    assert value_state[param_tag] == 7


def test_raw_widget_callback_records_parameter_history(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda _tag: True
    param_tag = '1:FloatValue:Float:Output01Value'
    value_state = {param_tag: 1.0}

    dpg.get_value.side_effect = lambda tag: value_state.get(tag)
    dpg.set_value.side_effect = lambda tag, value: value_state.__setitem__(tag, value)

    editor._cntrl_node_callback(param_tag, 2.5)

    assert len(editor._undo_stack) == 1
    assert isinstance(editor._undo_stack[-1], SetParameterCommand)
    assert editor._undo_stack[-1].before_value == 1.0
    assert editor._undo_stack[-1].after_value == 2.5

    editor._cntrl_undo(None, None)
    assert value_state[param_tag] == 1.0


def test_toggle_parameter_undo_redo_applies_toggle_side_effects(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda _tag: True
    toggle_tag = '1:TestNode:Text:ResultImageValue'
    value_state = {toggle_tag: False}
    dpg.set_value.side_effect = lambda tag, value: value_state.__setitem__(tag, value)
    dpg.get_value.side_effect = lambda tag: value_state.get(tag)

    node = editor._node_instance_list['TestNode']
    node.on_editor_parameter_value_applied = Mock(return_value=True)

    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:TestNode',
            'value_tag': toggle_tag,
            'before_value': False,
            'after_value': True,
        },
    )

    editor._cntrl_undo(None, None)
    node.on_editor_parameter_value_applied.assert_called_with(toggle_tag, False)
    editor._cntrl_redo(None, None)
    node.on_editor_parameter_value_applied.assert_called_with(toggle_tag, True)


def test_parameter_history_label_is_human_readable(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda tag: tag == 'Menu_Edit_Undo'
    editor._cntrl_push_undo_command(
        SetParameterCommand(
            node_id_name='1:TestNode',
            value_tag='1:TestNode:Text:ResultImageValue',
            before_value=False,
            after_value=True,
        )
    )
    dpg.configure_item.assert_any_call(
        'Menu_Edit_Undo',
        label='Undo (Set parameter: 1:TestNode.ResultImage)',
        enabled=True,
    )


def test_parameter_history_prefers_widget_label_when_available(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda tag: tag in {
        'Menu_Edit_Undo',
        '1:TestNode:Text:ResultImageValue',
    }
    dpg.get_item_configuration.side_effect = lambda tag: (
        {'label': 'Result Image'} if tag == '1:TestNode:Text:ResultImageValue' else {}
    )
    editor._cntrl_push_undo_command(
        SetParameterCommand(
            node_id_name='1:TestNode',
            value_tag='1:TestNode:Text:ResultImageValue',
            before_value=False,
            after_value=True,
        )
    )
    dpg.configure_item.assert_any_call(
        'Menu_Edit_Undo',
        label='Undo (Set parameter: 1:TestNode.Result Image)',
        enabled=True,
    )


def test_curves_points_parameter_records_each_edit_and_undo_redo(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda _tag: True
    value_tag = '1:Curves:Text:CurvesPointsValue'
    value_state = {
        value_tag: [[0, 0], [255, 255]],
    }
    dpg.get_value.side_effect = lambda tag: value_state.get(tag)
    dpg.set_value.side_effect = lambda tag, value: value_state.__setitem__(tag, value)

    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:Curves',
            'value_tag': value_tag,
            'before_value': [[0, 0], [255, 255]],
            'after_value': [[0, 0], [128, 200], [255, 255]],
        },
    )
    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:Curves',
            'value_tag': value_tag,
            'before_value': [[0, 0], [128, 200], [255, 255]],
            'after_value': [[0, 0], [180, 210], [255, 255]],
        },
    )

    assert len(editor._undo_stack) == 2
    assert isinstance(editor._undo_stack[-1], SetParameterCommand)
    assert editor._undo_stack[-1].before_value == [[0, 0], [128, 200], [255, 255]]
    assert editor._undo_stack[-1].after_value == [[0, 0], [180, 210], [255, 255]]

    editor._cntrl_undo(None, None)
    assert value_state[value_tag] == [[0, 0], [128, 200], [255, 255]]
    editor._cntrl_undo(None, None)
    assert value_state[value_tag] == [[0, 0], [255, 255]]
    editor._cntrl_redo(None, None)
    assert value_state[value_tag] == [[0, 0], [128, 200], [255, 255]]
    editor._cntrl_redo(None, None)
    assert value_state[value_tag] == [[0, 0], [180, 210], [255, 255]]


def test_curves_drag_coalesces_but_clicks_stay_separate(editor_and_dpg):
    editor, _ = editor_and_dpg
    value_tag = '1:Curves:Text:CurvesPointsValue'
    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:Curves',
            'value_tag': value_tag,
            'before_value': [[0, 0], [255, 255]],
            'after_value': [[0, 0], [100, 160], [255, 255]],
            'coalesce': False,  # click add
        },
    )
    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:Curves',
            'value_tag': value_tag,
            'before_value': [[0, 0], [100, 160], [255, 255]],
            'after_value': [[0, 0], [105, 162], [255, 255]],
            'coalesce': True,  # drag update
        },
    )
    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:Curves',
            'value_tag': value_tag,
            'before_value': [[0, 0], [105, 162], [255, 255]],
            'after_value': [[0, 0], [115, 170], [255, 255]],
            'coalesce': True,  # same drag continues
        },
    )
    editor._cntrl_node_callback(
        'parameter_changed',
        {
            'node_id_name': '1:Curves',
            'value_tag': value_tag,
            'before_value': [[0, 0], [115, 170], [255, 255]],
            'after_value': [[0, 0], [115, 170], [180, 200], [255, 255]],
            'coalesce': False,  # click add again
        },
    )
    assert len(editor._undo_stack) == 3


def test_parameter_undo_uses_node_history_apply_for_custom_controls(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.side_effect = lambda _tag: False
    curves_node = Mock()
    curves_node.apply_history_value.return_value = True
    editor._node_instance_list['Curves'] = curves_node

    cmd = SetParameterCommand(
        node_id_name='1:Curves',
        value_tag='1:Curves:Text:CurvesPointsValue',
        before_value=[[0, 0], [255, 255]],
        after_value=[[0, 0], [100, 180], [255, 255]],
    )
    editor._cntrl_push_undo_command(cmd)
    editor._cntrl_undo(None, None)

    curves_node.apply_history_value.assert_called_with(
        '1:Curves:Text:CurvesPointsValue',
        [[0, 0], [255, 255]],
    )


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
    dpg.get_item_configuration.return_value = {'attr_1': 101, 'attr_2': 102}
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }.get
    dpg.get_mouse_pos.return_value = [128.3, 255.9]

    editor._cntrl_open_insert_link_popup(None, None)

    dpg.set_item_pos.assert_any_call('NodeEditorInsertLinkPopup', [128, 255])
    dpg.show_item.assert_any_call('NodeEditorInsertLinkPopup')


def test_open_add_node_popup_from_hovered_output_port(editor_and_dpg):
    editor, dpg = editor_and_dpg
    output_tag = '1:TestNode:Int:Output01'
    editor._node_list = ['1:TestNode']
    editor._vw_populate_append_popup_menu = Mock()
    dpg.does_item_exist.side_effect = lambda tag: tag == output_tag
    dpg.is_item_hovered.side_effect = lambda tag: tag == output_tag
    dpg.get_mouse_pos.return_value = [210.7, 260.2]
    dpg.is_item_shown.side_effect = lambda tag: tag == 'NodeEditorInsertLinkPopup'
    dpg.get_item_pos.side_effect = lambda tag: (
        [10, 20] if tag == 'NodeEditorWindow' else [0, 0]
    )

    editor._cntrl_open_insert_link_popup(None, None)

    assert editor._pending_add_from_output_tag == output_tag
    assert editor._pending_add_to_input_tag is None
    assert editor._pending_insert_link_dpg_id is None
    editor._vw_populate_append_popup_menu.assert_called_once_with('Int')
    dpg.set_item_pos.assert_any_call('NodeEditorAddNodePopup', [200, 240])
    dpg.show_item.assert_any_call('NodeEditorAddNodePopup')
    dpg.hide_item.assert_any_call('NodeEditorInsertLinkPopup')


def test_open_add_node_popup_from_hovered_input_port(editor_and_dpg):
    editor, dpg = editor_and_dpg
    input_tag = '1:TestNode:Int:Input01'
    editor._node_list = ['1:TestNode']
    editor._vw_populate_prepend_popup_menu = Mock()
    dpg.does_item_exist.side_effect = lambda tag: tag == input_tag
    dpg.is_item_hovered.side_effect = lambda tag: tag == input_tag
    dpg.get_mouse_pos.return_value = [310.4, 70.9]
    dpg.is_item_shown.side_effect = lambda tag: tag == 'NodeEditorInsertLinkPopup'
    dpg.get_item_pos.side_effect = lambda tag: (
        [10, 20] if tag == 'NodeEditorWindow' else [0, 0]
    )

    editor._cntrl_open_insert_link_popup(None, None)

    assert editor._pending_add_from_output_tag is None
    assert editor._pending_add_to_input_tag == input_tag
    assert editor._pending_insert_link_dpg_id is None
    editor._vw_populate_prepend_popup_menu.assert_called_once_with('Int')
    dpg.set_item_pos.assert_any_call('NodeEditorAddNodePopup', [300, 50])
    dpg.show_item.assert_any_call('NodeEditorAddNodePopup')
    dpg.hide_item.assert_any_call('NodeEditorInsertLinkPopup')


def test_add_node_from_hovered_output_creates_connected_node(editor_and_dpg):
    editor, dpg = editor_and_dpg
    source_tag = '1:TestNode:Int:Output01'
    input_tag = '2:TestNode:Int:Input01'
    editor._node_id = 1
    editor._node_list = ['1:TestNode']
    editor._pending_add_from_output_tag = source_tag
    dpg.does_item_exist.side_effect = lambda tag: tag in {source_tag, input_tag}
    dpg.get_item_configuration.side_effect = lambda tag: (
        {'attribute_type': dpg.mvNode_Attr_Input}
        if tag == input_tag
        else {}
    )
    dpg.get_item_pos.side_effect = lambda tag: (
        [100, 200] if tag == '1:TestNode' else [0, 0]
    )
    dpg.add_node_link.return_value = 'link-new'

    editor._cntrl_add_node_from_output_port(None, None, 'TestNode')

    assert editor._pending_add_from_output_tag is None
    assert editor._node_list == ['1:TestNode', '2:TestNode']
    assert editor._node_link_list == [[source_tag, input_tag]]
    assert editor._link_view_id_map[(source_tag, input_tag)] == 'link-new'
    dpg.add_node_link.assert_called_once_with(
        source_tag,
        input_tag,
        parent='NodeEditor',
    )
    dpg.set_value.assert_any_call('NodeEditorLinkFeedback', '')


def test_add_node_to_hovered_input_creates_connected_node(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dest_tag = '1:TestNode:Int:Input01'
    output_tag = '2:TestNode:Int:Output01'
    editor._node_id = 1
    editor._node_list = ['1:TestNode']
    editor._pending_add_to_input_tag = dest_tag
    dpg.does_item_exist.side_effect = lambda tag: tag in {dest_tag, output_tag}
    dpg.get_item_configuration.side_effect = lambda tag: (
        {'attribute_type': dpg.mvNode_Attr_Output}
        if tag == output_tag
        else {}
    )
    dpg.get_item_pos.side_effect = lambda tag: (
        [300, 200] if tag == '1:TestNode' else [0, 0]
    )
    dpg.add_node_link.return_value = 'link-new'

    editor._cntrl_add_node_to_input_port(None, None, 'TestNode')

    assert editor._pending_add_to_input_tag is None
    assert editor._node_list == ['1:TestNode', '2:TestNode']
    assert editor._node_link_list == [[output_tag, dest_tag]]
    assert editor._link_view_id_map[(output_tag, dest_tag)] == 'link-new'
    dpg.add_node_link.assert_called_once_with(
        output_tag,
        dest_tag,
        parent='NodeEditor',
    )
    dpg.set_value.assert_any_call('NodeEditorLinkFeedback', '')


def test_open_insert_link_popup_on_hovered_link(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._link_view_id_map = {
        ('1:TestNode:Int:Output01', '2:TestNode:Int:Input01'): 'link-1'
    }
    dpg.get_item_configuration.return_value = {'attr_1': 101, 'attr_2': 102}
    dpg.get_item_alias.side_effect = {
        101: '1:TestNode:Int:Output01',
        102: '2:TestNode:Int:Input01',
    }.get
    dpg.get_selected_links.return_value = []
    dpg.is_item_hovered.side_effect = lambda item: item == 'link-1'
    dpg.get_mouse_pos.return_value = [10, 20]

    editor._cntrl_open_insert_link_popup(None, None)

    dpg.show_item.assert_any_call('NodeEditorInsertLinkPopup')


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
    editor._mdl_add_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01')
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

    dpg.hide_item.assert_any_call('NodeEditorInsertLinkPopup')


def test_insert_link_popup_closes_on_escape(editor_and_dpg):
    editor, dpg = editor_and_dpg
    editor._insert_link_popup_open = True
    dpg.is_item_shown.return_value = True
    editor._pending_insert_link_dpg_id = 'existing-link'

    editor._cntrl_close_insert_link_popup_on_escape(None, None)

    assert editor._pending_insert_link_dpg_id is None
    dpg.hide_item.assert_any_call('NodeEditorInsertLinkPopup')


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
    editor._mdl_add_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01')
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


def test_delete_multiple_nodes_heals_external_path(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.return_value = True
    editor._cntrl_add_node(None, None, 'TestNode')  # 1
    editor._cntrl_add_node(None, None, 'TestNode')  # 2
    editor._cntrl_add_node(None, None, 'TestNode')  # 3
    editor._cntrl_add_node(None, None, 'TestNode')  # 4

    editor._mdl_add_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01')
    editor._mdl_add_link('2:TestNode:Int:Output01', '3:TestNode:Int:Input01')
    editor._mdl_add_link('3:TestNode:Int:Output01', '4:TestNode:Int:Input01')

    dpg.get_selected_nodes.return_value = ['2:TestNode', '3:TestNode']
    dpg.get_selected_links.return_value = []
    editor._cntrl_delete_selected(None, None)

    assert '2:TestNode' not in editor._node_list
    assert '3:TestNode' not in editor._node_list
    assert ['1:TestNode:Int:Output01', '4:TestNode:Int:Input01'] in editor._node_link_list


def test_delete_multiple_nodes_heals_two_independent_paths(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.does_item_exist.return_value = True
    for _ in range(8):
        editor._cntrl_add_node(None, None, 'TestNode')

    editor._mdl_add_link('1:TestNode:Int:Output01', '2:TestNode:Int:Input01')
    editor._mdl_add_link('2:TestNode:Int:Output01', '3:TestNode:Int:Input01')
    editor._mdl_add_link('3:TestNode:Int:Output01', '4:TestNode:Int:Input01')

    editor._mdl_add_link('5:TestNode:Int:Output01', '6:TestNode:Int:Input01')
    editor._mdl_add_link('6:TestNode:Int:Output01', '7:TestNode:Int:Input01')
    editor._mdl_add_link('7:TestNode:Int:Output01', '8:TestNode:Int:Input01')

    dpg.get_selected_nodes.return_value = [
        '2:TestNode', '3:TestNode', '6:TestNode', '7:TestNode'
    ]
    dpg.get_selected_links.return_value = []
    editor._cntrl_delete_selected(None, None)

    assert ['1:TestNode:Int:Output01', '4:TestNode:Int:Input01'] in editor._node_link_list
    assert ['5:TestNode:Int:Output01', '8:TestNode:Int:Input01'] in editor._node_link_list


def test_delete_selected_ignores_stale_selected_link_ids(editor_and_dpg):
    editor, dpg = editor_and_dpg
    dpg.get_selected_nodes.return_value = []
    dpg.get_selected_links.return_value = [273]
    dpg.does_item_exist.return_value = False

    editor._cntrl_delete_selected(None, None)

    assert editor._node_link_list == []
