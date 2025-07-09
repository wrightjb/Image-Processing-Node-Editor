# test_node_editor_import_export.py
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict

# Assuming the DpgNodeEditor is in the node_editor module
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
    
    def test_callback_file_export_empty_editor(self, node_editor, mock_dpg):
        """Test exporting from an empty node editor"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Mock the callback data
            sender = 'file_export'
            data = {'file_path_name': temp_path}
            
            # Call the export callback
            node_editor._callback_file_export(sender, data)
            
            # Verify the exported file
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'node_list' in exported_data
            assert 'link_list' in exported_data
            assert exported_data['node_list'] == []
            assert exported_data['link_list'] == []
            
        finally:
            os.unlink(temp_path)
    
    def test_callback_file_export_with_nodes(self, node_editor, mock_dpg, mock_node_instance):
        """Test exporting with nodes in the editor"""
        # Setup test data
        node_editor._node_list = ['1:test_node', '2:test_node']
        node_editor._node_link_list = [['1:test_node:output', '2:test_node:input']]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            sender = 'file_export'
            data = {'file_path_name': temp_path}
            
            # Call the export callback
            node_editor._callback_file_export(sender, data)
            
            # Verify the exported file
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data['node_list'] == ['1:test_node', '2:test_node']
            assert exported_data['link_list'] == [['1:test_node:output', '2:test_node:input']]
            
            # Check node settings were exported
            assert '1:test_node' in exported_data
            assert '2:test_node' in exported_data
            assert exported_data['1:test_node']['id'] == '1'
            assert exported_data['1:test_node']['name'] == 'test_node'
            assert 'setting' in exported_data['1:test_node']
            
            # Verify get_setting_dict was called for each node
            assert mock_node_instance.get_setting_dict.call_count == 2
            
        finally:
            os.unlink(temp_path)
    
    def test_callback_file_export_menu(self, node_editor, mock_dpg):
        """Test the file export menu callback"""
        node_editor._callback_file_export_menu()
        mock_dpg.show_item.assert_called_once_with('file_export')
    
    def test_callback_file_import_menu_empty_editor(self, node_editor, mock_dpg):
        """Test file import menu when editor is empty (should show file dialog)"""
        node_editor._node_id = 0
        node_editor._callback_file_import_menu()
        mock_dpg.show_item.assert_called_once_with('file_import')
    
    def test_callback_file_import_menu_with_nodes(self, node_editor, mock_dpg):
        """Test file import menu when editor has nodes (should show warning)"""
        node_editor._node_id = 1
        node_editor._callback_file_import_menu()
        mock_dpg.configure_item.assert_called_once_with('modal_file_import', show=True)
    
    def test_callback_file_import_cancel(self, node_editor, mock_dpg):
        """Test import callback when user cancels (file_name is '.')"""
        sender = 'file_import'
        data = {'file_name': '.', 'file_path_name': '/path/to/file.json'}
        
        # Should not raise any exceptions and not process anything
        node_editor._callback_file_import(sender, data)
        
        # Verify no nodes were added
        assert node_editor._node_list == []
        assert node_editor._node_link_list == []
    
    def test_callback_file_import_success(self, node_editor, mock_dpg, mock_node_instance):
        """Test successful file import"""
        # Create test data to import
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(test_data, temp_file)
            temp_path = temp_file.name
        
        try:
            sender = 'file_import'
            data = {'file_name': 'test.json', 'file_path_name': temp_path}
            
            # Call the import callback
            node_editor._callback_file_import(sender, data)
            
            # Verify nodes were imported
            assert node_editor._node_list == ['1:test_node', '2:test_node']
            assert node_editor._node_link_list == [['1:test_node:output', '2:test_node:input']]
            
            # Verify node_id was updated
            assert node_editor._node_id == 2
            
            # Verify add_node was called for each node
            assert mock_node_instance.add_node.call_count == 2
            
            # Verify set_setting_dict was called for each node
            assert mock_node_instance.set_setting_dict.call_count == 2
            
            # Verify node links were added
            mock_dpg.add_node_link.assert_called_once_with(
                '1:test_node:output',
                '2:test_node:input', 
                parent=node_editor._node_editor_tag
            )
            
        finally:
            os.unlink(temp_path)
    
    def test_callback_file_import_version_warning(self, node_editor, mock_dpg, mock_node_instance, capsys):
        """Test import with version mismatch warning"""
        # Mock node with different version
        mock_node_instance._ver = '2.0.0'
        
        test_data = {
            'node_list': ['1:test_node'],
            'link_list': [],
            '1:test_node': {
                'id': '1',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',  # Different from node's version
                    'pos': [100, 200]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(test_data, temp_file)
            temp_path = temp_file.name
        
        try:
            sender = 'file_import'
            data = {'file_name': 'test.json', 'file_path_name': temp_path}
            
            # Call the import callback
            node_editor._callback_file_import(sender, data)
            
            # Check that warning was printed
            captured = capsys.readouterr()
            assert 'WARNING : test_node is different version' in captured.out
            assert 'Load Version ->1.0.0' in captured.out
            assert 'Code Version ->2.0.0' in captured.out
            
        finally:
            os.unlink(temp_path)
    
    def test_callback_file_import_updates_node_id(self, node_editor, mock_dpg, mock_node_instance):
        """Test that import updates _node_id to highest imported ID"""
        node_editor._node_id = 1  # Start with lower ID
        
        test_data = {
            'node_list': ['5:test_node'],  # Higher ID
            'link_list': [],
            '5:test_node': {
                'id': '5',
                'name': 'test_node',
                'setting': {
                    'ver': '1.0.0',
                    'pos': [100, 200]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(test_data, temp_file)
            temp_path = temp_file.name
        
        try:
            sender = 'file_import'
            data = {'file_name': 'test.json', 'file_path_name': temp_path}
            
            node_editor._callback_file_import(sender, data)
            
            # Verify _node_id was updated to the higher value
            assert node_editor._node_id == 5
            
        finally:
            os.unlink(temp_path)
    
    def test_callback_file_import_invalid_json(self, node_editor, mock_dpg):
        """Test import with invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write("invalid json content")
            temp_path = temp_file.name
        
        try:
            sender = 'file_import'
            data = {'file_name': 'test.json', 'file_path_name': temp_path}
            
            # Should raise JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                node_editor._callback_file_import(sender, data)
                
        finally:
            os.unlink(temp_path)
    
    def test_callback_file_import_missing_file(self, node_editor, mock_dpg):
        """Test import with non-existent file"""
        sender = 'file_import'
        data = {'file_name': 'test.json', 'file_path_name': '/nonexistent/file.json'}
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            node_editor._callback_file_import(sender, data)
    
    @pytest.mark.parametrize("debug_print", [True, False])
    def test_debug_print_behavior(self, mock_dpg, mock_node_instance, debug_print, capsys):
        """Test that debug print works correctly when enabled/disabled"""
        with patch('node_editor.node_editor.glob') as mock_glob, \
             patch('node_editor.node_editor.import_module') as mock_import:
            
            mock_glob.return_value = ['node/test_node/test_node.py']
            mock_module = Mock()
            mock_module.Node.return_value = mock_node_instance
            mock_import.return_value = mock_module
            
            editor = DpgNodeEditor(use_debug_print=debug_print)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Test export debug output
                sender = 'file_export'
                data = {'file_path_name': temp_path}
                editor._callback_file_export(sender, data)
                
                captured = capsys.readouterr()
                if debug_print:
                    assert '**** _callback_file_export ****' in captured.out
                else:
                    assert '**** _callback_file_export ****' not in captured.out
                    
            finally:
                os.unlink(temp_path)