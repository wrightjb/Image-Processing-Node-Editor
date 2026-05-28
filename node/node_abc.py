from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import dearpygui.dearpygui as dpg


@dataclass(frozen=True)
class NodeEditorFeatures:
    show_delete_button: bool = True


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

    def get_editor_features(self, node_id):
        """Return per-node editor chrome capabilities.

        Nodes can override this to opt out of default editor UI affordances
        (for example the delete button).
        """
        del node_id
        return NodeEditorFeatures()

    def _editor_toolbar_attr_tag(self, node_id):
        return f'{self._node_name(node_id)}:ToolbarAttr'

    def _editor_toolbar_group_tag(self, node_id):
        return f'{self._node_name(node_id)}:ToolbarGroup'

    def _editor_delete_button_tag(self, node_id):
        return f'{self._node_name(node_id)}:CloseButton'

    def add_editor_toolbar(
        self,
        node_id,
        callback=None,
        build_extra_controls=None,
    ):
        """Add the standard node toolbar row.

        The toolbar always starts with the universal delete button. Nodes that
        need node-owned controls can append them by passing
        ``build_extra_controls``; the callable runs inside the horizontal
        toolbar group.
        """
        node_id_name = self._node_name(node_id)
        with dpg.node_attribute(
            tag=self._editor_toolbar_attr_tag(node_id),
            attribute_type=dpg.mvNode_Attr_Static,
        ):
            with dpg.group(
                horizontal=True,
                tag=self._editor_toolbar_group_tag(node_id),
            ):
                dpg.add_button(
                    tag=self._editor_delete_button_tag(node_id),
                    label='x',
                    width=20,
                    height=20,
                    callback=self._on_editor_delete_button,
                    user_data=(callback, node_id_name),
                )
                if build_extra_controls is not None:
                    build_extra_controls()

    def _on_editor_delete_button(self, sender, app_data, user_data):
        del sender, app_data
        if not isinstance(user_data, tuple) or len(user_data) != 2:
            return
        callback, node_id_name = user_data
        if callback is None:
            return
        callback('delete_node_requested', {'node_id_name': node_id_name})
