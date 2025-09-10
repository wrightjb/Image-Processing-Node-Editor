#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import dearpygui.dearpygui as dpg
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import cv2
import tempfile
from collections import OrderedDict

from node_editor.node_editor import DpgNodeEditor
from main import update_node_info
from node.process_node.node_blur import image_process as blur_image_process
from node.process_node.node_contrast import image_process as contrast_image_process


class TestNodeEditorIntegration:
    """Integration tests for DpgNodeEditor with basic node types."""

    node_width = 240
    node_height = 135

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

    def advance_frames(self, n=5):
        for _ in range(n):
            dpg.render_dearpygui_frame()

    def create_checkerboard_image(self, size=32, square_size=8):
        """Create a checkerboard pattern image for testing."""
        image = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = [255, 255, 255]
                else:
                    image[i:i+square_size, j:j+square_size] = [0, 0, 0]
        return image

    def create_gradient_image(self, size=32):
        """Create a gradient image for testing contrast effects."""
        image = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                image[i, j] = [i * (255 // size), j * (255 // size), 128]
        return image

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_image_blur_pipeline(self, kernel_size, debug_test):
        """Test image input -> blur processing pipeline."""
        # Create test image
        test_image = self.create_checkerboard_image()
        
        # Create temporary file for test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            image_path = tmp.name

        try:
            # Create node editor with proper menu configuration
            menu_dict = OrderedDict({
                'InputNode': 'input_node',
                'ProcessNode': 'process_node',
            })
            editor = DpgNodeEditor(
                width=800,
                height=600,
                opencv_setting_dict={
                    'input_window_width': 240,
                    'input_window_height': 135,
                    'process_width': 240,
                    'process_height': 135,
                    'use_pref_counter': False
                },
                menu_dict=menu_dict,
                node_dir='node'
            )
            self.advance_frames()

            # Add image node
            editor.cntrl_add_node(None, None, 'Image')
            image_node_id = '1:Image'
            
            # Add blur node
            editor._last_pos = [self.node_width, self.node_height] # non-overlap for debug
            editor.cntrl_add_node(None, None, 'Blur')
            blur_node_id = '2:Blur'

            # Set image path in image node
            image_node = editor.get_node_instance('Image')
            image_node._image_filepath['1'] = image_path
            image_node._image['1'] = test_image

            # Set blur kernel size
            blur_node = editor.get_node_instance('Blur')
            blur_kernel_tag = blur_node_id + ':' + blur_node.TYPE_INT + ':Input02Value'
            dpg.set_value(blur_kernel_tag, kernel_size)

            # Create node connection (image output -> blur input)
            image_output_tag = image_node_id + ':' + image_node.TYPE_IMAGE + ':Output01'
            blur_input_tag = blur_node_id + ':' + blur_node.TYPE_IMAGE + ':Input01'

            image_output_id = dpg.get_alias_id(image_output_tag)
            blur_input_id = dpg.get_alias_id(blur_input_tag)
            
            # Use editor's callback to link nodes
            editor.cntrl_link('NodeEditor', [image_output_id, blur_input_id])

            # Create node dictionaries for update
            node_image_dict = {}
            node_result_dict = {}

            # Update nodes using main.py function
            update_node_info(editor, node_image_dict, node_result_dict)

            # Debug: print node connections and results
            print("Node list:", editor.get_node_list())
            print("Node connections:", editor.get_sorted_node_connection())
            print("Node image dict keys:", list(node_image_dict.keys()))
            print("Node image dict values:", {k: v is not None for k, v in node_image_dict.items()})
            print("Available node instances:", list(editor._node_instance_list.keys()))

            # Verify results
            assert image_node_id in node_image_dict, f"Image node {image_node_id} should be in results"
            assert blur_node_id in node_image_dict, f"Blur node {blur_node_id} should be in results"
            
            image_result = node_image_dict[image_node_id]
            blur_result = node_image_dict[blur_node_id]
            
            assert image_result is not None, "Image node should return image"
            assert blur_result is not None, "Blur node should return processed image"
            
            # Check that blur actually changed the image
            assert not np.array_equal(image_result, blur_result), "Blur should modify the image"
            
            # Check that blur result has expected shape
            assert blur_result.shape == test_image.shape, "Blur should preserve image shape"

            # Check that blur result matches direct processing
            expected_blur_result = blur_image_process(test_image, kernel_size)
            np.testing.assert_allclose(blur_result, expected_blur_result, atol=1,
                                       err_msg="Blur node output differs from direct processing")

            if debug_test:
                self.advance_frames()
                breakpoint()

        finally:
            # Cleanup
            if os.path.exists(image_path):
                os.unlink(image_path)

    @pytest.mark.parametrize("contrast_alpha", [0.5, 1.0, 2.0])
    def test_image_contrast_pipeline(self, contrast_alpha, debug_test):
        """Test image input -> contrast processing pipeline."""
        # Create test image with gradient for better contrast testing
        test_image = self.create_gradient_image()
        
        # Create temporary file for test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            image_path = tmp.name

        try:
            # Create node editor with proper menu configuration
            menu_dict = OrderedDict({
                'InputNode': 'input_node',
                'ProcessNode': 'process_node',
            })
            editor = DpgNodeEditor(
                width=800,
                height=600,
                opencv_setting_dict={
                    'input_window_width': self.node_width,
                    'input_window_height': self.node_height,
                    'process_width': self.node_width,
                    'process_height': self.node_height,
                    'use_pref_counter': False
                },
                menu_dict=menu_dict,
                node_dir='node'
            )
            self.advance_frames()

            # Add image node
            editor.cntrl_add_node(None, None, 'Image')
            image_node_id = '1:Image'
            
            # Add contrast node
            editor._last_pos = [self.node_width, self.node_height] # non-overlap for debug
            editor.cntrl_add_node(None, None, 'Contrast')
            contrast_node_id = '2:Contrast'

            # Set image path in image node
            image_node = editor.get_node_instance('Image')
            image_node._image_filepath['1'] = image_path
            image_node._image['1'] = test_image

            # Set contrast alpha
            contrast_node = editor.get_node_instance('Contrast')
            contrast_alpha_tag = contrast_node_id + ':' + contrast_node.TYPE_FLOAT + ':Input02Value'
            dpg.set_value(contrast_alpha_tag, contrast_alpha)

            # Create node connection (image output -> contrast input)
            image_output_tag = image_node_id + ':' + image_node.TYPE_IMAGE + ':Output01'
            contrast_input_tag = contrast_node_id + ':' + contrast_node.TYPE_IMAGE + ':Input01'

            image_output_id = dpg.get_alias_id(image_output_tag)
            contrast_input_id = dpg.get_alias_id(contrast_input_tag)
            
            # Use editor's callback to link nodes
            editor.cntrl_link('NodeEditor', [image_output_id, contrast_input_id])
            
            # Create node dictionaries for update
            node_image_dict = {}
            node_result_dict = {}

            # Update nodes using main.py function
            update_node_info(editor, node_image_dict, node_result_dict)

            # Verify results
            assert image_node_id in node_image_dict, "Image node should be in results"
            assert contrast_node_id in node_image_dict, "Contrast node should be in results"
            
            image_result = node_image_dict[image_node_id]
            contrast_result = node_image_dict[contrast_node_id]
            
            assert image_result is not None, "Image node should return image"
            assert contrast_result is not None, "Contrast node should return processed image"
            
            # For alpha != 1.0, contrast should modify the image
            if contrast_alpha != 1.0:
                assert not np.array_equal(image_result, contrast_result), "Contrast should modify the image"
            else:
                assert np.array_equal(image_result, contrast_result), "Contrast alpha=1.0 should not modify the image"
            
            # Check that contrast result has expected shape
            assert contrast_result.shape == test_image.shape, "Contrast should preserve image shape"

            # Check that contrast result matches direct processing
            expected_contrast_result = contrast_image_process(test_image, contrast_alpha)
            np.testing.assert_allclose(contrast_result, expected_contrast_result, atol=1,
                                       err_msg="Contrast node output differs from direct processing")

            if debug_test:
                self.advance_frames()
                breakpoint()

        finally:
            # Cleanup
            if os.path.exists(image_path):
                os.unlink(image_path)

    @pytest.mark.parametrize("kernel_size,contrast_alpha", [
        (3, 0.5),
        (7, 1.0),
        (15, 2.0)
    ])
    def test_image_blur_contrast_pipeline(self, kernel_size, contrast_alpha, debug_test):
        """Test image input -> blur -> contrast processing pipeline."""
        # Create test image
        test_image = self.create_gradient_image()
        
        # Create temporary file for test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            image_path = tmp.name

        try:
            # Create node editor with proper menu configuration
            menu_dict = OrderedDict({
                'InputNode': 'input_node',
                'ProcessNode': 'process_node',
            })
            editor = DpgNodeEditor(
                width=800,
                height=600,
                opencv_setting_dict={
                    'input_window_width': self.node_width,
                    'input_window_height': self.node_height,
                    'process_width': self.node_width,
                    'process_height': self.node_height,
                    'use_pref_counter': False
                },
                menu_dict=menu_dict,
                node_dir='node'
            )
            self.advance_frames()

            # Add image node
            editor.cntrl_add_node(None, None, 'Image')
            image_node_id = '1:Image'
            
            # Add blur node
            editor._last_pos = [self.node_width, self.node_height] # non-overlap for debug
            editor.cntrl_add_node(None, None, 'Blur')
            blur_node_id = '2:Blur'
            
            # Add contrast node
            editor._last_pos = [self.node_width*2, self.node_height*2]
            editor.cntrl_add_node(None, None, 'Contrast')
            contrast_node_id = '3:Contrast'

            # Set image path in image node
            image_node = editor.get_node_instance('Image')
            image_node._image_filepath['1'] = image_path
            image_node._image['1'] = test_image

            # Set blur kernel size
            blur_node = editor.get_node_instance('Blur')
            blur_kernel_tag = blur_node_id + ':' + blur_node.TYPE_INT + ':Input02Value'
            dpg.set_value(blur_kernel_tag, kernel_size)

            # Set contrast alpha
            contrast_node = editor.get_node_instance('Contrast')
            contrast_alpha_tag = contrast_node_id + ':' + contrast_node.TYPE_FLOAT + ':Input02Value'
            dpg.set_value(contrast_alpha_tag, contrast_alpha)

            # Create node connections
            image_output_tag = image_node_id + ':' + image_node.TYPE_IMAGE + ':Output01'
            blur_input_tag = blur_node_id + ':' + blur_node.TYPE_IMAGE + ':Input01'
            blur_output_tag = blur_node_id + ':' + blur_node.TYPE_IMAGE + ':Output01'
            contrast_input_tag = contrast_node_id + ':' + contrast_node.TYPE_IMAGE + ':Input01'

            image_output_id = dpg.get_alias_id(image_output_tag)
            blur_input_id = dpg.get_alias_id(blur_input_tag)
            blur_output_id = dpg.get_alias_id(blur_output_tag)
            contrast_input_id = dpg.get_alias_id(contrast_input_tag)
            
            # Connect image -> blur
            editor.cntrl_link('NodeEditor', [image_output_id, blur_input_id])
            
            # Connect blur -> contrast
            editor.cntrl_link('NodeEditor', [blur_output_id, contrast_input_id])

            # Create node dictionaries for update
            node_image_dict = {}
            node_result_dict = {}

            # Update nodes using main.py function
            update_node_info(editor, node_image_dict, node_result_dict)

            # Verify results
            assert image_node_id in node_image_dict, "Image node should be in results"
            assert blur_node_id in node_image_dict, "Blur node should be in results"
            assert contrast_node_id in node_image_dict, "Contrast node should be in results"
            
            image_result = node_image_dict[image_node_id]
            blur_result = node_image_dict[blur_node_id]
            contrast_result = node_image_dict[contrast_node_id]
            
            assert image_result is not None, "Image node should return image"
            assert blur_result is not None, "Blur node should return processed image"
            assert contrast_result is not None, "Contrast node should return processed image"
            
            # Check that processing chain works
            assert contrast_result.shape == test_image.shape, "Final result should preserve image shape"

            # Check that final result matches direct processing chain
            expected_blur_result = blur_image_process(test_image, kernel_size)
            expected_contrast_result = contrast_image_process(expected_blur_result, contrast_alpha)
            np.testing.assert_allclose(contrast_result, expected_contrast_result, atol=1,
                                       err_msg="Chained node output differs from direct processing")

            if debug_test:
                self.advance_frames()
                breakpoint()

        finally:
            # Cleanup
            if os.path.exists(image_path):
                os.unlink(image_path)


if __name__ == "__main__":
    # pytest.main([__file__, "-v"])

    tnei = TestNodeEditorIntegration()
    tnei.test_image_blur_pipeline(kernel_size=7)