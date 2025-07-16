# test_node_editor_integration.py
import pytest
import dearpygui.dearpygui as dpg
from unittest.mock import patch, MagicMock
import sys
import os

from node_editor.node_editor import DpgNodeEditor


class TestNodeEditorIntegration:
    """Integration tests for DpgNodeEditor window and viewport sizing."""
    
    @pytest.fixture(autouse=True)
    def setup_dpg(self):
        """Setup DPG context for each test."""
        dpg.create_context()
        dpg.create_viewport()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        yield
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

    def advance_frames(self, n=5):
        for _ in range(n):
            dpg.render_dearpygui_frame()
    
    def test_window_matches_viewport_size_on_start_default(
        self, 
        mock_node_modules
    ):
        """Test that window size matches viewport size on startup with default dimensions."""
        # Set viewport size
        viewport_width, viewport_height = 1920, 1080
        dpg.set_viewport_width(viewport_width)
        dpg.set_viewport_height(viewport_height)
        
        # Create node editor with default size (should match viewport)
        editor = DpgNodeEditor()
        self.advance_frames()
        
        # Get current configuration
        window_config = dpg.get_item_configuration(editor._window_tag)   
        viewport_client_width = dpg.get_viewport_client_width()
        viewport_client_height = dpg.get_viewport_client_height()

        # Verify window size matches specified dimensions
        assert window_config['width'] == viewport_client_width
        assert window_config['height'] == viewport_client_height
    
    def test_window_matches_viewport_size_on_start_custom(
        self, 
        mock_node_modules
    ):
        """Test that window size matches specified dimensions on startup."""
        # Set viewport size
        viewport_width, viewport_height = 1920, 1080
        dpg.set_viewport_width(viewport_width)
        dpg.set_viewport_height(viewport_height)
        
        # Create node editor with custom size
        custom_width, custom_height = 1600, 900
        editor = DpgNodeEditor(width=custom_width, height=custom_height)
        self.advance_frames()
        
        # Get window configuration
        window_config = dpg.get_item_configuration(editor._window_tag)
        
        viewport_client_width = dpg.get_viewport_client_width()
        viewport_client_height = dpg.get_viewport_client_height()
        print((viewport_client_width, viewport_client_height))
        print((window_config['width'], window_config['height']))
        # Verify window size matches specified dimensions
        assert window_config['width'] == viewport_client_width
        assert window_config['height'] == viewport_client_height
    
    @pytest.mark.parametrize("initial_size,new_size", [
        ((100,100), (200,200)),
        ((800, 600), (1440, 900)),
        ((1280, 720), (1920, 1080)),
        ((1920, 1080), (1280, 720)),
        ((1920, 1080), (3840, 2160)),
    ])
    def test_window_resize_after_viewport_change(self, mock_node_modules, initial_size, new_size):
        """Test window resizing behavior after viewport size changes."""
        initial_width, initial_height = initial_size
        new_width, new_height = new_size
        
        # Set initial viewport size
        dpg.set_viewport_width(initial_width)
        dpg.set_viewport_height(initial_height)
        
        # Create node editor
        editor = DpgNodeEditor(width=initial_width, height=initial_height)
        self.advance_frames()
        
        window_config = dpg.get_item_configuration(editor._window_tag)
        viewport_client_width = dpg.get_viewport_client_width()
        viewport_client_height = dpg.get_viewport_client_height()

        # Verify initial size
        assert window_config['width'] == viewport_client_width
        assert window_config['height'] == viewport_client_height
        
        # Change viewport size
        dpg.set_viewport_width(new_width)
        dpg.set_viewport_height(new_height)
        self.advance_frames()
        
        updated_config = dpg.get_item_configuration(editor._window_tag)
        updated_viewport_client_width = dpg.get_viewport_client_width()
        updated_viewport_client_height = dpg.get_viewport_client_height()

        # Verify new size
        assert updated_config['width'] == updated_viewport_client_width
        assert updated_config['height'] == updated_viewport_client_height


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    