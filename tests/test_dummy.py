# test_node_editor_integration.py
import pytest
import dearpygui.dearpygui as dpg
from unittest.mock import patch, MagicMock
import sys
import os
import warnings

from node_editor.node_editor import DpgNodeEditor


class TestNodeEditorIntegration:
    """Integration tests for DpgNodeEditor window and viewport sizing."""
    
    @pytest.fixture(autouse=True)
    def setup_dpg(self):
        """Setup DPG context for each test."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dpg.create_context()
            dpg.setup_dearpygui()

        yield # Test runs here

        # Cleanup after each test
        try:
            dpg.destroy_context()
        except:
            pass
    
    @pytest.fixture
    def mock_node_modules(self):
        """Mock node modules to avoid file system dependencies."""
        with patch('node_editor.node_editor.glob') as mock_glob, \
             patch('node_editor.node_editor.import_module') as mock_import:
            
            # Mock empty node directory
            mock_glob.return_value = []
            
            yield mock_glob, mock_import

    def test_no_nodes_on_launch(
        self, 
        mock_node_modules
    ):
        editor = DpgNodeEditor()
        nodes = editor.get_node_list()

        assert len(nodes) == 0

    def test_fail(
        self, 
        mock_node_modules
    ):
        assert False
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])