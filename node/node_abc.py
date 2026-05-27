from abc import ABCMeta, abstractmethod


class DpgNodeABC(metaclass=ABCMeta):
    _ver = '0.0.0'

    node_label = ''
    node_tag = ''

    TYPE_INT = 'Int'
    TYPE_FLOAT = 'Float'
    TYPE_IMAGE = 'Image'
    TYPE_TIME_MS = 'TimeMS'
    TYPE_TEXT = 'Text'

    def _node_name(self, node_id):
        return f'{node_id}:{self.node_tag}'

    def _port_tag(self, node_name, value_type, port_name):
        return f'{node_name}:{value_type}:{port_name}'

    def _value_tag(self, port_tag):
        return f'{port_tag}Value'

    def _extract_source_node_key(self, source_tag):
        source_tokens = source_tag.split(':')
        if len(source_tokens) < 2:
            return ''
        return ':'.join(source_tokens[:2])

    def _extract_port_name(self, tag):
        tag_tokens = tag.split(':')
        if len(tag_tokens) < 4:
            return ''
        return tag_tokens[3]

    def _extract_node_id(self, tag):
        tag_tokens = tag.split(':')
        if len(tag_tokens) < 2:
            return ''
        return tag_tokens[0]

    def _iter_connections(self, connection_list):
        for connection_info in connection_list:
            if not isinstance(connection_info, (list, tuple)) or len(connection_info) < 2:
                continue

            source_tag, destination_tag = connection_info[0], connection_info[1]
            if not isinstance(source_tag, str) or not isinstance(destination_tag, str):
                continue

            source_tokens = source_tag.split(':')
            if len(source_tokens) < 4:
                continue

            connection_type = source_tokens[2]
            yield source_tag, destination_tag, connection_type

    @abstractmethod
    def add_node(
        self,
        parent,
        node_id,
        pos,
        width,
        height,
        opencv_setting_dict,
    ):
        pass

    @abstractmethod
    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        pass

    @abstractmethod
    def get_setting_dict(self, node_id):
        pass

    @abstractmethod
    def set_setting_dict(self, node_id, setting_dict):
        pass

    @abstractmethod
    def close(self, node_id):
        pass

    def on_editor_parameter_value_applied(self, value_tag, value):
        """Hook called when editor applies a parameter value (e.g. undo/redo).

        Returns:
            bool: True if the node handled additional side effects, else False.
        """
        del value_tag, value
        return False
