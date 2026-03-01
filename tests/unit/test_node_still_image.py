from unittest.mock import patch

from node.input_node.node_still_image import Node


class TestNodeStillImageSettings:
    def test_get_setting_dict_includes_image_path(self):
        node = Node()
        node._image_filepath['1'] = '/tmp/example.png'

        with patch('node.input_node.node_still_image.dpg') as mock_dpg:
            mock_dpg.get_item_pos.return_value = [10, 20]

            setting = node.get_setting_dict(1)

        assert setting['ver'] == node._ver
        assert setting['pos'] == [10, 20]
        assert setting['image_path'] == '/tmp/example.png'

    def test_set_setting_dict_loads_existing_image_path(self):
        node = Node()
        setting = {'image_path': '/tmp/example.png'}

        with patch('node.input_node.node_still_image.os.path.isfile', return_value=True):
            node.set_setting_dict(1, setting)

        assert node._image_filepath['1'] == '/tmp/example.png'

    def test_set_setting_dict_handles_missing_image_path_gracefully(self, capsys):
        node = Node()
        node._image_filepath['1'] = '/tmp/old.png'
        node._prev_image_filepath['1'] = '/tmp/old.png'
        node._image['1'] = object()
        setting = {'image_path': '/tmp/missing.png'}

        with patch('node.input_node.node_still_image.os.path.isfile', return_value=False):
            node.set_setting_dict(1, setting)

        captured = capsys.readouterr()
        assert 'WARNING : Image file not found (/tmp/missing.png)' in captured.out
        assert '1' not in node._image_filepath
        assert '1' not in node._prev_image_filepath
        assert '1' not in node._image
