import pytest
import json
import os
from unittest.mock import Mock, patch
from collections import OrderedDict

from node_editor.node_editor import DpgNodeEditor


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

    def test_callback_file_export_empty_editor(
        self, 
        node_editor, 
        mock_dpg, 
        tmp_path
    ):
        """Test exporting from an empty node editor"""
        temp_path = tmp_path / "empty.json"
        sender = 'file_export'
        data = {'file_path_name': str(temp_path)}

        node_editor._callback_file_export(sender, data)

        with open(temp_path, 'r') as f:
            exported_data = json.load(f)

        assert 'node_list' in exported_data
        assert 'link_list' in exported_data
        assert exported_data['node_list'] == []
        assert exported_data['link_list'] == []

    def test_callback_file_export_with_nodes(
        self, 
        node_editor, 
        mock_dpg, 
        mock_node_instance, 
        tmp_path
    ):
        """Test exporting with nodes in the editor"""
        node_editor._node_list = ['1:test_node', '2:test_node']
        node_editor._node_link_list = [[
            '1:test_node:output', 
            '2:test_node:input'
        ]]

        temp_path = tmp_path / "with_nodes.json"
        sender = 'file_export'
        data = {'file_path_name': str(temp_path)}

        node_editor._callback_file_export(sender, data)

        with open(temp_path, 'r') as f:
            exported_data = json.load(f)

        assert exported_data['node_list'] == ['1:test_node', '2:test_node']
        assert exported_data['link_list'] == [[
            '1:test_node:output', 
            '2:test_node:input'
        ]]

        assert '1:test_node' in exported_data
        assert '2:test_node' in exported_data
        assert exported_data['1:test_node']['id'] == '1'
        assert exported_data['1:test_node']['name'] == 'test_node'
        assert 'setting' in exported_data['1:test_node']
        
        assert mock_node_instance.get_setting_dict.call_count == 2

    def test_callback_file_export_menu(self, node_editor, mock_dpg):
        """Test the file export menu callback"""
        node_editor._callback_file_export_menu()
        mock_dpg.show_item.assert_called_once_with('file_export')

    def test_callback_file_import_menu_empty_editor(
        self, 
        node_editor, 
        mock_dpg
    ):
        """Test file import menu when editor empty (should show file dialog)"""
        node_editor._node_id = 0
        node_editor._callback_file_import_menu()
        mock_dpg.show_item.assert_called_once_with('file_import')

    def test_callback_file_import_menu_with_nodes(self, node_editor, mock_dpg):
        """Test file import menu when editor has nodes (should show warning)"""
        node_editor._node_id = 1
        node_editor._callback_file_import_menu()
        mock_dpg.configure_item.assert_called_once_with(
            'modal_file_import', 
            show=True
        )

    def test_callback_file_import_success(
        self, 
        node_editor, 
        mock_dpg, 
        mock_node_instance, 
        tmp_path
    ):
        """Test successful file import"""
        test_data = {
            'node_list': ['1:test_node', '2:test_node'],
            'link_list': [['1:test_node:output', '2:test_node:input']],
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [100, 200],
                    'param1': 'value1'
                }
            },
            '2:test_node': {
                'id': '2', 
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [300, 400],
                    'param2': 'value2'
                }
            }
        }
        temp_path = tmp_path / "import.json"
        with open(temp_path, 'w') as f:
            json.dump(test_data, f)

        sender = 'file_import'
        data = {'file_name': temp_path.name, 'file_path_name': str(temp_path)}
        node_editor._callback_file_import(sender, data)

        assert node_editor._node_list == ['1:test_node', '2:test_node']
        assert node_editor._node_link_list == [[
            '1:test_node:output', 
            '2:test_node:input'
        ]]
        assert node_editor._node_id == 2
        assert mock_node_instance.add_node.call_count == 2
        assert mock_node_instance.set_setting_dict.call_count == 2
        mock_dpg.add_node_link.assert_called_once_with(
            '1:test_node:output',
            '2:test_node:input', 
            parent=node_editor._node_editor_tag
        )

    def test_callback_file_import_version_warning(
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
            'link_list': [],
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
        node_editor._callback_file_import(sender, data)

        captured = capsys.readouterr()
        assert 'WARNING : test_node is different version' in captured.out
        assert 'Load Version ->1.0.0' in captured.out
        assert 'Code Version ->2.0.0' in captured.out

    def test_callback_file_import_updates_node_id(
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
            'link_list': [],
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
        node_editor._callback_file_import(sender, data)

        assert node_editor._node_id == 5

    def test_callback_file_import_missing_key_raises(
        self, 
        node_editor, 
        mock_dpg, 
        tmp_path
    ):
        """Test that importing JSON missing mandatory keys raises KeyError"""
        # Create JSON missing 'node_list'
        bad = {'link_list': []}
        path = tmp_path / 'bad.json'
        path.write_text(json.dumps(bad))
        with pytest.raises(KeyError):
            node_editor._callback_file_import(
                None, 
                {'file_name': path.name, 'file_path_name': str(path)}
            )

    def test_callback_file_import_invalid_json(
        self, 
        node_editor, 
        mock_dpg, 
        tmp_path
    ):
        """Test import with invalid JSON file"""
        temp_path = tmp_path / "invalid.json"
        with open(temp_path, 'w') as f:
            f.write("invalid json content")

        sender = 'file_import'
        data = {'file_name': temp_path.name, 'file_path_name': str(temp_path)}
        with pytest.raises(json.JSONDecodeError):
            node_editor._callback_file_import(sender, data)

    def test_callback_file_import_missing_file(self, node_editor, mock_dpg):
        """Test import with non-existent file"""
        sender = 'file_import'
        data = {
            'file_name': 'missing.json', 
            'file_path_name': '/nonexistent/file.json'
        }
        with pytest.raises(FileNotFoundError):
            node_editor._callback_file_import(sender, data)

    @pytest.mark.parametrize("debug_print", [True, False])
    def test_callback_file_import_debug_print_behavior(
        self, 
        mock_dpg, 
        mock_node_instance, 
        debug_print, 
        capsys, 
        tmp_path
    ):
        """Test that debug print works correctly when enabled/disabled"""
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
            editor._callback_file_export(sender, data)

            captured = capsys.readouterr()
            if debug_print:
                assert '**** _callback_file_export ****' in captured.out
            else:
                assert '**** _callback_file_export ****' not in captured.out

    def test_callback_file_export_import_round_trip(
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
        node_editor._node_link_list = [['1:test_node:out', '2:test_node:in']]

        # Act: export the original state to JSON
        orig = tmp_path / 'orig.json'
        node_editor._callback_file_export(None, {'file_path_name': str(orig)})

        # Act: import into a fresh editor instance
        with patch('node_editor.node_editor.glob') as mg, \
             patch('node_editor.node_editor.import_module') as mi:
            mg.return_value = ['node/test_node/test_node.py']
            mi.return_value = Mock(Node=lambda: mock_node_instance)
            new_ed = DpgNodeEditor(use_debug_print=False)
            new_ed._callback_file_import(
                None,
                {'file_name': orig.name, 'file_path_name': str(orig)}
            )

            # Re-export the imported state to another JSON
            fresh = tmp_path / 'fresh.json'
            new_ed._callback_file_export(None, {'file_path_name': str(fresh)})

        # Assert: the two JSON payloads are identical
        assert json.loads(orig.read_text()) == json.loads(fresh.read_text())

    def test_callback_file_export_import_complex_graph(
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
            ['1:test_node:out', '2:test_node:in'],
            ['1:test_node:out', '3:test_node:in'],
            ['2:test_node:out', '4:test_node:in'],
            ['3:test_node:out', '4:test_node:in'],
        ]
        node_editor._node_list = original_nodes.copy()
        node_editor._node_link_list = original_links.copy()
        # Provide distinct settings
        mock_node_instance.get_setting_dict.side_effect = lambda nid: {
            'ver': '1.0.0',
            'pos': [int(nid), int(nid)]
        }

        # Act: export the graph
        export_path = tmp_path / 'complex.json'
        node_editor._callback_file_export(
            None, 
            {'file_path_name': str(export_path)}
        )
        exported = json.loads(export_path.read_text())

        # Assert export structure is correct
        assert set(exported['node_list']) == set(original_nodes)
        assert (set(tuple(l) for l in exported['link_list']) 
                == set(tuple(l) for l in original_links))

        # Act: import into a fresh editor
        with patch('node_editor.node_editor.glob') as mg, \
             patch('node_editor.node_editor.import_module') as mi:
            mg.return_value = ['node/test_node/test_node.py']
            mi.return_value = Mock(Node=lambda: mock_node_instance)
            imported_editor = DpgNodeEditor(use_debug_print=False)
            imported_editor._callback_file_import(
                None,
                {
                    'file_name': export_path.name, 
                    'file_path_name': str(export_path)
                }
            )

        # Assert import reconstructed the same graph
        assert set(imported_editor._node_list) == set(original_nodes)
        assert (set(tuple(l) for l in imported_editor._node_link_list) 
                == set(tuple(l) for l in original_links))