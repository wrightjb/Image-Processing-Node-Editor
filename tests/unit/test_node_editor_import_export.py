import pytest
import json
import os
from unittest.mock import Mock, patch
from collections import OrderedDict

from node_editor.node_editor import DpgNodeEditor


def _typed_port_payload(port_tag):
    parts = str(port_tag).split(':')
    if len(parts) < 4:
        return {'dpg_tag': port_tag}
    node_id, node_tag, data_type, port_name = parts[:4]
    direction = ''
    index = 0
    if port_name.startswith('Input'):
        direction = 'Input'
        index_text = port_name[len('Input'):]
    elif port_name.startswith('Output'):
        direction = 'Output'
        index_text = port_name[len('Output'):]
    else:
        index_text = ''
    if index_text.isdigit():
        index = int(index_text)
    return {
        'node': {'id': node_id, 'tag': node_tag},
        'direction': direction,
        'data_type': data_type,
        'index': index,
        'port_name': port_name,
        'dpg_tag': port_tag,
    }


def _typed_link_refs(link_pairs):
    return [
        {
            'source': _typed_port_payload(source_tag),
            'destination': _typed_port_payload(dest_tag),
        }
        for source_tag, dest_tag in link_pairs
    ]


def _link_ref_pairs(link_refs):
    return [
        [link_ref['source']['dpg_tag'], link_ref['destination']['dpg_tag']]
        for link_ref in link_refs
    ]


def _typed_link_pairs_from_editor(editor):
    return [link_ref.legacy_pair for link_ref in editor._link_refs]


def _add_link_pairs(editor, link_pairs):
    for source_tag, dest_tag in link_pairs:
        assert editor._mdl_add_link(source_tag, dest_tag) is True


def _history_link_pairs(links):
    pairs = []
    for link in links:
        if hasattr(link, 'legacy_pair'):
            pairs.append(tuple(link.legacy_pair))
        else:
            pairs.append(tuple(link))
    return pairs


class TestDpgNodeEditorImportExport:
    """Test suite for DpgNodeEditor import/export functionality"""

    @pytest.fixture
    def mock_dpg(self):
        """Mock dearpygui module"""
        with patch('node_editor.node_editor.dpg') as mock_dpg:
            # Configure common dpg mock behaviors
            mock_dpg.get_item_alias.side_effect = lambda x: x
            mock_dpg.get_selected_nodes.return_value = []
            mock_dpg.get_selected_links.return_value = []
            yield mock_dpg

    @pytest.fixture
    def mock_node_instance(self):
        """Create a mock node instance"""
        mock_node = Mock()
        mock_node.node_tag = 'test_node'
        mock_node.node_label = 'Test Node'
        mock_node._ver = '1.0.0'
        mock_node.get_setting_dict.return_value = {
            'ver': '1.0.0',
            'pos': [100, 200],
            'param1': 'value1',
            'param2': 42
        }
        mock_node.add_node.return_value = 'test_node_tag'
        mock_node.set_setting_dict = Mock()
        return mock_node

    @pytest.fixture
    def node_editor(self, mock_dpg, mock_node_instance):
        """Create a DpgNodeEditor instance with mocked dependencies"""
        with patch('node_editor.node_editor.glob') as mock_glob, \
             patch('node_editor.node_editor.import_module') as mock_import:

            # Mock glob to return test node files
            mock_glob.return_value = ['node/test_node/test_node.py']

            # Mock import_module to return our mock node
            mock_module = Mock()
            mock_module.Node.return_value = mock_node_instance
            mock_import.return_value = mock_module

            editor = DpgNodeEditor(use_debug_print=False)
            return editor

    def test_export_empty_editor(
        self,
        node_editor,
        mock_dpg,
        tmp_path
    ):
        """Test exporting from an empty node editor"""
        temp_path = tmp_path / "empty.json"
        sender = 'file_export'
        data = {'file_path_name': str(temp_path)}

        node_editor._cntrl_file_export(sender, data)

        with open(temp_path, 'r') as f:
            exported_data = json.load(f)

        assert 'node_list' in exported_data
        assert 'link_list' not in exported_data
        assert 'link_refs' in exported_data
        assert exported_data['node_list'] == []
        assert exported_data['link_refs'] == []

    def test_export_with_nodes(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path
    ):
        """Test exporting with nodes in the editor"""
        node_editor._node_list = ['1:test_node', '2:test_node']
        _add_link_pairs(node_editor, [[
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01'
        ]])

        temp_path = tmp_path / "with_nodes.json"
        sender = 'file_export'
        data = {'file_path_name': str(temp_path)}

        node_editor._cntrl_file_export(sender, data)

        with open(temp_path, 'r') as f:
            exported_data = json.load(f)

        assert exported_data['node_list'] == ['1:test_node', '2:test_node']
        assert 'link_list' not in exported_data
        assert exported_data['link_refs'] == [
            {
                'source': {
                    'node': {'id': '1', 'tag': 'test_node'},
                    'direction': 'Output',
                    'data_type': 'Image',
                    'index': 1,
                    'port_name': 'Output01',
                    'dpg_tag': '1:test_node:Image:Output01',
                },
                'destination': {
                    'node': {'id': '2', 'tag': 'test_node'},
                    'direction': 'Input',
                    'data_type': 'Image',
                    'index': 1,
                    'port_name': 'Input01',
                    'dpg_tag': '2:test_node:Image:Input01',
                },
            }
        ]

        assert '1:test_node' in exported_data
        assert '2:test_node' in exported_data
        assert exported_data['1:test_node']['id'] == '1'
        assert exported_data['1:test_node']['name'] == 'test_node'
        assert 'setting' in exported_data['1:test_node']

        assert mock_node_instance.get_setting_dict.call_count == 2

    def test_export_menu(self, node_editor, mock_dpg):
        """Test the file export menu callback"""
        node_editor._cntrl_file_export_menu(None, None, None)
        mock_dpg.show_item.assert_called_once_with('file_export')

    @pytest.mark.parametrize("node_id", [0,1,10,100])
    def test_import_menu(
        self,
        node_editor,
        mock_dpg,
        node_id
    ):
        """Test file import menu when editor empty (should show file
           dialog)"""
        node_editor._node_id = node_id
        node_editor._cntrl_file_import_menu(None, None, None)
        mock_dpg.show_item.assert_called_once_with('file_import')

    def test_import_success(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path
    ):
        """Test successful file import"""
        test_data = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': _typed_link_refs([[
                '1:test_node:Image:Output01',
                '2:test_node:Image:Input01',
            ]]),
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [100, 200],
                    'param1': 'value1',
                    '1:test_node:Int:Input01Value': 15,
                }
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [300, 400],
                    'param2': 'value2',
                    '2:test_node:Float:Input01Value': 0.75,
                }
            }
        }
        temp_path = tmp_path / "import.json"
        with open(temp_path, 'w') as f:
            json.dump(test_data, f)

        sender = 'file_import'
        data = {'file_name': temp_path.name, 'file_path_name': str(temp_path)}
        node_editor._cntrl_file_import(sender, data)

        assert node_editor._node_list == ['1:test_node', '2:test_node']
        assert _typed_link_pairs_from_editor(node_editor) == [[
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01'
        ]]
        assert node_editor._node_id == 2
        assert mock_node_instance.add_node.call_count == 2
        assert mock_node_instance.set_setting_dict.call_count == 2
        assert node_editor._parameter_last_values['1:test_node:Int:Input01Value'] == 15
        assert node_editor._parameter_last_values['2:test_node:Float:Input01Value'] == 0.75
        mock_dpg.add_node_link.assert_called_once_with(
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01',
            parent=node_editor._node_editor_tag
        )

    def test_import_prefers_typed_link_refs(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path,
    ):
        test_data = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': [
                {
                    'source': {
                        'node': {'id': '1', 'tag': 'test_node'},
                        'direction': 'Output',
                        'data_type': 'Image',
                        'index': 1,
                        'port_name': 'Output01',
                        'dpg_tag': '1:test_node:Image:Output01',
                    },
                    'destination': {
                        'node': {'id': '2', 'tag': 'test_node'},
                        'direction': 'Input',
                        'data_type': 'Image',
                        'index': 1,
                        'port_name': 'Input01',
                        'dpg_tag': '2:test_node:Image:Input01',
                    },
                }
            ],
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [100, 200]},
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [300, 400]},
            },
        }
        temp_path = tmp_path / 'typed_import.json'
        temp_path.write_text(json.dumps(test_data))

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        assert _typed_link_pairs_from_editor(node_editor) == [[
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01',
        ]]
        link_ref = node_editor._mdl_get_link_ref_by_destination(
            '2:test_node:Image:Input01'
        )
        assert link_ref is not None
        assert link_ref.source.dpg_tag == '1:test_node:Image:Output01'
        assert link_ref.destination.dpg_tag == '2:test_node:Image:Input01'

    def test_import_is_undoable_and_redoable(
        self,
        node_editor,
        mock_dpg,
        tmp_path,
    ):
        test_data = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': _typed_link_refs([[
                '1:test_node:Image:Output01',
                '2:test_node:Image:Input01',
            ]]),
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [100, 200],
                },
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [300, 200],
                },
            },
        }
        temp_path = tmp_path / "undoable_import.json"
        temp_path.write_text(json.dumps(test_data))
        mock_dpg.does_item_exist.return_value = True
        mock_dpg.get_item_pos.return_value = [100, 200]

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        assert node_editor._node_list == ['1:test_node', '2:test_node']
        assert _typed_link_pairs_from_editor(node_editor) == [[
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01',
        ]]
        assert len(node_editor._undo_stack) == 1
        assert node_editor._redo_stack == []

        node_editor._cntrl_undo(None, None)

        assert node_editor._node_list == []
        assert _typed_link_pairs_from_editor(node_editor) == []
        assert len(node_editor._redo_stack) == 1

        node_editor._cntrl_redo(None, None)

        assert node_editor._node_list == ['1:test_node', '2:test_node']
        assert _typed_link_pairs_from_editor(node_editor) == [[
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01',
        ]]
        assert node_editor.get_sorted_node_connection() == OrderedDict([
            ('1:test_node', []),
            ('2:test_node', [[
                '1:test_node:Image:Output01',
                '2:test_node:Image:Input01',
            ]]),
        ])
        assert len(node_editor._undo_stack) == 1

    @pytest.mark.parametrize(
        "link_pairs",
        [
            [["1:test_node:output", "2:test_node:Image:Input01"]],
            [["1:test_node:Image:Output01", "2:test_node:input"]],
        ],
    )
    def test_import_rejects_malformed_link_tags_without_link_state(
        self,
        node_editor,
        mock_dpg,
        tmp_path,
        link_pairs,
    ):
        test_data = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': _typed_link_refs(link_pairs),
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [100, 200]},
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [300, 200]},
            },
        }
        temp_path = tmp_path / "malformed_link_import.json"
        temp_path.write_text(json.dumps(test_data))

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        assert _typed_link_pairs_from_editor(node_editor) == []
        assert node_editor._link_registry == {}
        assert node_editor._undo_stack[-1].links == []
        mock_dpg.add_node_link.assert_not_called()

    @pytest.mark.parametrize(
        "replacement_link",
        [
            ["1:test_node:output", "3:test_node:Image:Input01"],
            ["2:test_node:Image:Output99", "3:test_node:Image:Input01"],
        ],
    )
    def test_import_failed_duplicate_destination_replacement_keeps_existing_link(
        self,
        node_editor,
        mock_dpg,
        tmp_path,
        replacement_link,
    ):
        valid_link = [
            '1:test_node:Image:Output01',
            '3:test_node:Image:Input01',
        ]
        test_data = {
            'node_list': ['1:test_node', '2:test_node', '3:test_node'],
            'link_refs': _typed_link_refs([valid_link, replacement_link]),
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [100, 200]},
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [300, 200]},
            },
            '3:test_node': {
                'id': '3',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [500, 200]},
            },
        }
        temp_path = tmp_path / "failed_duplicate_destination_import.json"
        temp_path.write_text(json.dumps(test_data))

        existing_items = {
            '1:test_node',
            '2:test_node',
            '3:test_node',
            '1:test_node:Image:Output01',
            '2:test_node:Image:Output01',
            '3:test_node:Image:Input01',
        }
        mock_dpg.does_item_exist.side_effect = lambda tag: tag in existing_items
        mock_dpg.get_item_pos.return_value = [100, 200]
        mock_dpg.add_node_link.side_effect = [
            'valid-link',
            'redo-valid-link',
        ]

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        assert _typed_link_pairs_from_editor(node_editor) == [[
            '1:test_node:Image:Output01',
            '3:test_node:Image:Input01',
        ]]
        assert _history_link_pairs(node_editor._undo_stack[-1].links) == [(
            '1:test_node:Image:Output01',
            '3:test_node:Image:Input01',
        )]

        node_editor._cntrl_undo(None, None)
        assert _typed_link_pairs_from_editor(node_editor) == []

        node_editor._cntrl_redo(None, None)
        assert _typed_link_pairs_from_editor(node_editor) == [[
            '1:test_node:Image:Output01',
            '3:test_node:Image:Input01',
        ]]

    def test_import_rolls_back_model_when_parseable_port_is_missing_in_view(
        self,
        node_editor,
        mock_dpg,
        tmp_path,
    ):
        test_data = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': _typed_link_refs([[
                '1:test_node:Image:Output99',
                '2:test_node:Image:Input01',
            ]]),
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [100, 200]},
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [300, 200]},
            },
        }
        temp_path = tmp_path / "missing_view_port_import.json"
        temp_path.write_text(json.dumps(test_data))

        existing_items = {
            '1:test_node',
            '2:test_node',
            '2:test_node:Image:Input01',
        }
        mock_dpg.does_item_exist.side_effect = lambda tag: tag in existing_items
        mock_dpg.get_item_pos.return_value = [100, 200]

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        assert _typed_link_pairs_from_editor(node_editor) == []
        assert node_editor._link_registry == {}
        assert node_editor._link_view_id_map == {}
        assert node_editor._undo_stack[-1].links == []
        mock_dpg.add_node_link.assert_not_called()

    @pytest.mark.parametrize(
        "link_pairs",
        [
            [["1:test_node:Image:Input01", "2:test_node:Image:Input01"]],
            [["1:test_node:Int:Output01", "2:test_node:Float:Input01"]],
        ],
    )
    def test_import_rejects_semantically_invalid_links_without_link_state(
        self,
        node_editor,
        mock_dpg,
        tmp_path,
        link_pairs,
    ):
        test_data = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': _typed_link_refs(link_pairs),
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [100, 200]},
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [300, 200]},
            },
        }
        temp_path = tmp_path / "semantic_invalid_link_import.json"
        temp_path.write_text(json.dumps(test_data))
        mock_dpg.does_item_exist.return_value = True
        mock_dpg.get_item_pos.return_value = [100, 200]

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        assert _typed_link_pairs_from_editor(node_editor) == []
        assert node_editor._link_registry == {}
        assert node_editor._undo_stack[-1].links == []
        mock_dpg.add_node_link.assert_not_called()

    def test_import_duplicate_destination_link_last_link_wins(
        self,
        node_editor,
        mock_dpg,
        tmp_path,
    ):
        test_data = {
            'node_list': ['1:test_node', '2:test_node', '3:test_node'],
            'link_refs': _typed_link_refs([
                ['1:test_node:Image:Output01', '3:test_node:Image:Input01'],
                ['2:test_node:Image:Output01', '3:test_node:Image:Input01'],
            ]),
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [100, 200]},
            },
            '2:test_node': {
                'id': '2',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [300, 200]},
            },
            '3:test_node': {
                'id': '3',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [500, 200]},
            },
        }
        temp_path = tmp_path / "duplicate_destination_import.json"
        temp_path.write_text(json.dumps(test_data))
        mock_dpg.does_item_exist.return_value = True
        mock_dpg.get_item_pos.return_value = [100, 200]

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        assert _typed_link_pairs_from_editor(node_editor) == [[
            '2:test_node:Image:Output01',
            '3:test_node:Image:Input01',
        ]]
        assert _history_link_pairs(node_editor._undo_stack[-1].links) == [(
            '2:test_node:Image:Output01',
            '3:test_node:Image:Input01',
        )]

    def test_import_primes_move_cache_for_first_drag_undo(
        self,
        node_editor,
        mock_dpg,
        tmp_path,
    ):
        test_data = {
            'node_list': ['1:test_node'],
            'link_refs': [],
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [10, 20],
                }
            },
        }
        temp_path = tmp_path / "import_move_cache.json"
        with open(temp_path, 'w') as f:
            json.dump(test_data, f)

        pos_map = {'1:test_node': [10, 20]}
        mock_dpg.get_item_alias.side_effect = lambda tag: tag
        mock_dpg.get_selected_nodes.return_value = ['1:test_node']
        mock_dpg.does_item_exist.side_effect = lambda tag: tag in pos_map
        mock_dpg.get_item_pos.side_effect = lambda tag: pos_map.get(tag, [0, 0])
        mock_dpg.set_item_pos.side_effect = lambda tag, pos: pos_map.__setitem__(tag, list(pos))

        node_editor._cntrl_file_import(
            'file_import',
            {'file_name': temp_path.name, 'file_path_name': str(temp_path)},
        )

        node_editor._cntrl_capture_move_start_positions(None, None)
        pos_map['1:test_node'] = [100, 120]
        node_editor._cntrl_commit_move_commands(None, None)
        node_editor._cntrl_undo(None, None)

        assert pos_map['1:test_node'] == [10, 20]

    def test_import_version_warning(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        capsys,
        tmp_path
    ):
        """Test import with version mismatch warning"""
        mock_node_instance._ver = '2.0.0'
        test_data = {
            'node_list': ['1:test_node'],
            'link_refs': [],
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [100, 200]
                }
            }
        }
        temp_path = tmp_path / "version.json"
        with open(temp_path, 'w') as f:
            json.dump(test_data, f)

        sender = 'file_import'
        data = {'file_name': temp_path.name, 'file_path_name': str(temp_path)}
        node_editor._cntrl_file_import(sender, data)

        captured = capsys.readouterr()
        assert 'WARNING : test_node is different version' in captured.out
        assert 'Load Version -> 1.0.0' in captured.out
        assert 'Code Version -> 2.0.0' in captured.out

    def test_import_updates_node_id(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path
    ):
        """Test that import updates _node_id to highest imported ID"""
        node_editor._node_id = 1
        test_data = {
            'node_list': ['5:test_node'],
            'link_refs': [],
            '5:test_node': {
                'id': '5',
                'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [100, 200]}
            }
        }
        temp_path = tmp_path / "update_id.json"
        with open(temp_path, 'w') as f:
            json.dump(test_data, f)

        sender = 'file_import'
        data = {'file_name': temp_path.name, 'file_path_name': str(temp_path)}
        node_editor._cntrl_file_import(sender, data)

        # grab all the new IDs you handed out
        call_ids = [
            call.args[1] for call in mock_node_instance.add_node.call_args_list
            ]
        highest_assigned = max(call_ids)

        assert node_editor._node_id >= highest_assigned

    def test_import_missing_key_raises(self, node_editor, mock_dpg, tmp_path):
        """Test that importing JSON missing mandatory keys raises
           KeyError"""
        # Create JSON missing 'node_list'
        bad = {'link_refs': []}
        path = tmp_path / 'bad.json'
        path.write_text(json.dumps(bad))
        with pytest.raises(KeyError):
            node_editor._cntrl_file_import(
                None,
                {'file_name': path.name, 'file_path_name': str(path)}
            )

    def test_import_invalid_json_raises(self, node_editor, mock_dpg, tmp_path):
        """Test import with invalid JSON file"""
        temp_path = tmp_path / "invalid.json"
        with open(temp_path, 'w') as f:
            f.write("invalid json content")

        sender = 'file_import'
        data = {'file_name': temp_path.name, 'file_path_name': str(temp_path)}
        with pytest.raises(json.JSONDecodeError):
            node_editor._cntrl_file_import(sender, data)

    def test_import_missing_file_raises(self, node_editor, mock_dpg):
        """Test import with non-existent file"""
        sender = 'file_import'
        data = {
            'file_name': 'missing.json',
            'file_path_name': '/nonexistent/file.json'
        }
        with pytest.raises(FileNotFoundError):
            node_editor._cntrl_file_import(sender, data)

    @pytest.mark.parametrize("debug_print", [True, False])
    def test_import_debug_print_behavior(
        self,
        mock_dpg,
        mock_node_instance,
        debug_print,
        capsys,
        tmp_path
    ):
        """Test that debug print works correctly when
           enabled/disabled"""
        with patch('node_editor.node_editor.glob') as mock_glob, \
             patch('node_editor.node_editor.import_module') as mock_import:
            mock_glob.return_value = ['node/test_node/test_node.py']
            mock_module = Mock()
            mock_module.Node.return_value = mock_node_instance
            mock_import.return_value = mock_module

            editor = DpgNodeEditor(use_debug_print=debug_print)
            temp_path = tmp_path / "debug.json"

            sender = 'file_export'
            data = {'file_path_name': str(temp_path)}
            editor._cntrl_file_export(sender, data)

            captured = capsys.readouterr()
            if debug_print:
                assert '**** _cntrl_file_export ****' in captured.out
            else:
                assert '**** _cntrl_file_export ****' not in captured.out

    def test_export_import_round_trip(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path
    ):
        """Test that exporting then importing preserves the exact JSON
           structure and data"""

        # Arrange: seed editor with two nodes and a link between them
        node_editor._node_list = ['1:test_node', '2:test_node']
        _add_link_pairs(node_editor, [[
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01',
        ]])

        # Act: export the original state to JSON
        orig = tmp_path / 'orig.json'
        node_editor._cntrl_file_export(None, {'file_path_name': str(orig)})

        # Act: import into a fresh editor instance
        with patch('node_editor.node_editor.glob') as mg, \
             patch('node_editor.node_editor.import_module') as mi:
            mg.return_value = ['node/test_node/test_node.py']
            mi.return_value = Mock(Node=lambda: mock_node_instance)
            new_ed = DpgNodeEditor(use_debug_print=False)
            new_ed._cntrl_file_import(
                None,
                {'file_name': orig.name, 'file_path_name': str(orig)}
            )

            # Re-export the imported state to another JSON
            fresh = tmp_path / 'fresh.json'
            new_ed._cntrl_file_export(None, {'file_path_name': str(fresh)})

        # Assert: the two JSON payloads are identical
        assert json.loads(orig.read_text()) == json.loads(fresh.read_text())

    def test_export_import_complex_graph(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path
    ):
        """Test export and import on a branching graph with shared
           dependencies"""
        # Arrange: create nodes A, B, C, D and links A->B, A->C, B->D, C->D
        original_nodes = [
            '1:test_node',
            '2:test_node',
            '3:test_node',
            '4:test_node'
        ]
        original_links = [
            ['1:test_node:Image:Output01', '2:test_node:Image:Input01'],
            ['1:test_node:Image:Output01', '3:test_node:Image:Input01'],
            ['2:test_node:Image:Output01', '4:test_node:Image:Input01'],
            ['3:test_node:Image:Output01', '4:test_node:Image:Input02'],
        ]
        node_editor._node_list = original_nodes.copy()
        _add_link_pairs(node_editor, original_links)
        # Provide distinct settings
        mock_node_instance.get_setting_dict.side_effect = lambda nid: {
            'ver': '1.0.0',
            'pos': [int(nid), int(nid)]
        }

        # Act: export the graph
        export_path = tmp_path / 'complex.json'
        node_editor._cntrl_file_export(
            None,
            {'file_path_name': str(export_path)}
        )
        exported = json.loads(export_path.read_text())

        # Assert export structure is correct
        assert set(exported['node_list']) == set(original_nodes)
        assert 'link_list' not in exported
        assert (set(tuple(l) for l in _link_ref_pairs(exported['link_refs']))
                == set(tuple(l) for l in original_links))

        # Act: import into a fresh editor
        with patch('node_editor.node_editor.glob') as mg, \
             patch('node_editor.node_editor.import_module') as mi:
            mg.return_value = ['node/test_node/test_node.py']
            mi.return_value = Mock(Node=lambda: mock_node_instance)
            imported_editor = DpgNodeEditor(use_debug_print=False)
            imported_editor._cntrl_file_import(
                None,
                {
                    'file_name': export_path.name,
                    'file_path_name': str(export_path)
                }
            )

        # Assert import reconstructed the same graph
        assert set(imported_editor._node_list) == set(original_nodes)
        assert (set(tuple(l) for l in _typed_link_pairs_from_editor(imported_editor))
                == set(tuple(l) for l in original_links))

    def test_import_after_existing_nodes_non_conflicting(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path
    ):
        """Imported nodes must be renumbered so they don’t clash with
           existing IDs."""
        # Arrange: editor already has nodes 1 and 2
        node_editor._node_list = ['1:test_node', '2:test_node']
        _add_link_pairs(node_editor, [[
            '1:test_node:Image:Output01',
            '2:test_node:Image:Input01',
        ]])
        node_editor._node_id = 2

        # import JSON with the same IDs (1 & 2), so they must be remapped
        imported = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': _typed_link_refs([[
                '1:test_node:Image:Output01',
                '2:test_node:Image:Input01',
            ]]),
            '1:test_node': {
                'id': '1', 'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [10, 20]}
            },
            '2:test_node': {
                'id': '2', 'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [30, 40]}
            }
        }
        path = tmp_path / "after.json"
        path.write_text(json.dumps(imported))
        data = {'file_name': path.name, 'file_path_name': str(path)}

        # Act
        node_editor._cntrl_file_import(None, data)

        # Assert: we still have exactly 4 nodes total
        assert len(node_editor._node_list) == 4

        # old IDs remain present
        assert set(node_editor._node_list).issuperset(['1:test_node', '2:test_node'])

        # extract new IDs and verify they don’t collide and are unique
        existing_ids = {1, 2}
        all_ids = [int(n.split(':')[0]) for n in node_editor._node_list]
        new_ids = [i for i in all_ids if i not in existing_ids]
        assert len(new_ids) == 2
        assert len(set(new_ids)) == 2
        assert all(i not in existing_ids for i in new_ids)

        # ensure _node_id has advanced at least to the max assigned ID
        assert node_editor._node_id >= max(all_ids)

        # ensure add_node was called with exactly those two new IDs
        call_ids = [call.args[1] for call in mock_node_instance.add_node.call_args_list]
        assert set(call_ids) == set(new_ids)

        # grab all links that connect two imported IDs
        imported_links = [
            link for link in _typed_link_pairs_from_editor(node_editor)
            if int(link[0].split(':')[0]) in new_ids
            and int(link[1].split(':')[0]) in new_ids
        ]

        # there should be exactly one of those
        assert len(imported_links) == 1

        # and it should connect the two imported IDs
        src_id, dst_id = [int(x.split(':')[0]) for x in imported_links[0]]
        assert {src_id, dst_id} == set(new_ids)

    def test_delete_linked_node_after_import_with_missing_port_registry(
        self,
        node_editor,
        mock_dpg,
        mock_node_instance,
        tmp_path,
    ):
        imported = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_refs': _typed_link_refs([[
                '1:test_node:Image:Output01',
                '2:test_node:Image:Input01',
            ]]),
            '1:test_node': {
                'id': '1', 'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [10, 20]}
            },
            '2:test_node': {
                'id': '2', 'name': 'test_node',
                'setting': {'ver': '1.0.0', 'pos': [30, 40]}
            }
        }
        path = tmp_path / "imported_links.json"
        path.write_text(json.dumps(imported))
        node_editor._cntrl_file_import(
            None, {'file_name': path.name, 'file_path_name': str(path)}
        )

        node_editor._port_registry.clear()
        mock_dpg.get_selected_nodes.return_value = ['1:test_node']
        mock_dpg.get_selected_links.return_value = []
        node_editor._cntrl_delete_selected(None, None)

        assert '1:test_node' not in node_editor._node_list
        assert _typed_link_pairs_from_editor(node_editor) == []
        assert node_editor._link_registry == {}
