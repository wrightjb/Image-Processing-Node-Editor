from abc import ABCMeta, abstractmethod

import dearpygui.dearpygui as dpg

from node.port_model import NodeRef, PortRef


class _PortTagString(str):
    def __new__(cls, value, port_ref=None):
        tag = str.__new__(cls, value)
        tag.port_ref = port_ref
        return tag


class DpgNodeABC(metaclass=ABCMeta):
    _ver = '0.0.0'

    node_label = ''
    node_tag = ''

    TYPE_INT = 'Int'
    TYPE_FLOAT = 'Float'
    TYPE_IMAGE = 'Image'
    TYPE_TIME_MS = 'TimeMS'
    TYPE_TEXT = 'Text'

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


class DpgNodeBase(DpgNodeABC):
    """Concrete base for DearPyGui nodes.

    This base owns shared DearPyGui tag, connection parsing, and editor toolbar
    helpers. `DpgNodeABC` remains the abstract plugin contract.
    """

    def _node_name(self, node_id):
        return f'{node_id}:{self.node_tag}'

    def _port_tag(self, node_name, value_type, port_name):
        return f'{node_name}:{value_type}:{port_name}'

    def _value_tag(self, port_tag):
        port_ref = getattr(port_tag, 'port_ref', None)
        if port_ref is not None and port_ref.value_tag:
            return port_ref.value_tag
        return f'{port_tag}Value'

    def _node_port_tag(self, node_id, value_type, port_name):
        return self._port_tag(self._node_name(node_id), value_type, port_name)

    def _port_value_tag(self, node_name, value_type, port_name):
        return self._value_tag(self._port_tag(node_name, value_type, port_name))

    def _node_value_tag(self, node_id, value_type, port_name):
        return self._value_tag(self._node_port_tag(node_id, value_type, port_name))

    def node_ref(self, node_id):
        return NodeRef(str(node_id), self.node_tag)

    def set_port_registration_callback(self, callback):
        self._port_registration_callback = callback

    def get_declared_port_refs(self, node_id=None):
        self._ensure_port_declaration_state()
        if node_id is None:
            return [
                port_ref
                for node_ports in self._declared_port_refs.values()
                for port_ref in node_ports.values()
            ]
        node_ports = self._declared_port_refs.get(self._node_name(node_id), {})
        return list(node_ports.values())

    def get_declared_port_ref(
        self,
        node_id,
        data_type=None,
        port_name=None,
        direction=None,
    ):
        for port_ref in self.get_declared_port_refs(node_id):
            if data_type is not None and port_ref.data_type != data_type:
                continue
            if port_name is not None and port_ref.port_name != port_name:
                continue
            if direction is not None and port_ref.direction != direction:
                continue
            return port_ref
        return None

    def declared_port_value_tag(
        self,
        node_id,
        data_type,
        port_name,
        direction=None,
    ):
        port_ref = self.get_declared_port_ref(
            node_id,
            data_type=data_type,
            port_name=port_name,
            direction=direction,
        )
        if port_ref is not None and port_ref.value_tag:
            return port_ref.value_tag
        return self._node_value_tag(node_id, data_type, port_name)

    def input_port(self, node_id, data_type, port_name=None):
        return self._declare_port(node_id, data_type, 'Input', port_name)

    def output_port(self, node_id, data_type, port_name=None):
        return self._declare_port(node_id, data_type, 'Output', port_name)

    def parameter_port(self, node_id, data_type, port_name=None, control_tag=None):
        return self._declare_port(
            node_id,
            data_type,
            'Input',
            port_name,
            control_tag=control_tag,
            default_control_tag=True,
        )

    def _declare_port(
        self,
        node_id,
        data_type,
        direction,
        port_name=None,
        control_tag=None,
        default_control_tag=False,
    ):
        self._ensure_port_declaration_state()
        node_ref = self.node_ref(node_id)
        port_name, index = self._resolve_port_name(node_ref, direction, port_name)
        dpg_tag = self._port_tag(node_ref.node_id_name, data_type, port_name)
        value_tag = self._value_tag(dpg_tag)
        if control_tag is None and default_control_tag:
            control_tag = value_tag
        port_ref = PortRef(
            node_ref=node_ref,
            direction=direction,
            data_type=data_type,
            index=index,
            port_name=port_name,
            dpg_tag=dpg_tag,
            value_tag=value_tag,
            control_tag=control_tag,
        )
        self._remember_port_ref(port_ref)
        return port_ref

    def _resolve_port_name(self, node_ref, direction, port_name):
        counter_key = (node_ref.node_id_name, direction)
        if port_name is None:
            index = self._port_index_counters.get(counter_key, 0) + 1
            self._port_index_counters[counter_key] = index
            return f'{direction}{index:02d}', index

        index = self._port_index(port_name, direction)
        self._port_index_counters[counter_key] = max(
            self._port_index_counters.get(counter_key, 0),
            index,
        )
        return port_name, index

    def _port_index(self, port_name, direction):
        if not isinstance(port_name, str) or not port_name.startswith(direction):
            raise ValueError(
                f'{direction} port names must start with {direction}: {port_name}'
            )
        index_text = port_name[len(direction):]
        try:
            return int(index_text)
        except ValueError as exc:
            raise ValueError(
                f'{direction} port names must end with a numeric index: {port_name}'
            ) from exc

    def _remember_port_ref(self, port_ref):
        node_ports = self._declared_port_refs.setdefault(
            port_ref.node_ref.node_id_name,
            {},
        )
        node_ports[port_ref.dpg_tag] = port_ref
        callback = getattr(self, '_port_registration_callback', None)
        if callback is not None:
            callback(port_ref)

    def _ensure_port_declaration_state(self):
        if not hasattr(self, '_declared_port_refs'):
            self._declared_port_refs = {}
        if not hasattr(self, '_port_index_counters'):
            self._port_index_counters = {}
        if not hasattr(self, '_port_registration_callback'):
            self._port_registration_callback = None

    def _extract_source_node_key(self, source_tag):
        port_ref = getattr(source_tag, 'port_ref', None)
        if port_ref is not None:
            return port_ref.node_ref.node_id_name
        source_tokens = source_tag.split(':')
        if len(source_tokens) < 2:
            return ''
        return ':'.join(source_tokens[:2])

    def _extract_port_name(self, tag):
        port_ref = getattr(tag, 'port_ref', None)
        if port_ref is not None:
            return port_ref.port_name
        tag_tokens = tag.split(':')
        if len(tag_tokens) < 4:
            return ''
        return tag_tokens[3]

    def _extract_node_id(self, tag):
        port_ref = getattr(tag, 'port_ref', None)
        if port_ref is not None:
            return port_ref.node_ref.node_id
        tag_tokens = tag.split(':')
        if len(tag_tokens) < 2:
            return ''
        return tag_tokens[0]

    def _connection_source_node_key(self, connection_info, source_tag):
        source_port = getattr(connection_info, 'source', None)
        if source_port is not None:
            return source_port.node_ref.node_id_name
        return self._extract_source_node_key(source_tag)

    def _connection_port_name(self, connection_info, destination_tag):
        destination_port = getattr(connection_info, 'destination', None)
        if destination_port is not None:
            return destination_port.port_name
        return self._extract_port_name(destination_tag)

    def _connection_value_tag(self, connection_info, endpoint, fallback_tag):
        port_ref = getattr(connection_info, endpoint, None)
        if port_ref is not None and port_ref.value_tag:
            return port_ref.value_tag
        return self._value_tag(fallback_tag)

    def _iter_connection_infos(self, connection_list):
        for connection_info in connection_list:
            source_port = getattr(connection_info, 'source', None)
            destination_port = getattr(connection_info, 'destination', None)
            if hasattr(connection_info, 'legacy_pair'):
                source_tag, destination_tag = connection_info.legacy_pair
            elif (
                isinstance(connection_info, (list, tuple))
                and len(connection_info) >= 2
            ):
                source_tag, destination_tag = connection_info[0], connection_info[1]
            else:
                continue

            if not isinstance(source_tag, str) or not isinstance(destination_tag, str):
                continue

            if source_port is not None:
                connection_type = source_port.data_type
            else:
                source_tokens = source_tag.split(':')
                if len(source_tokens) < 4:
                    continue
                connection_type = source_tokens[2]
            yield (
                connection_info,
                source_tag,
                destination_tag,
                connection_type,
            )

    def _iter_connections(self, connection_list):
        for (
            connection_info,
            source_tag,
            destination_tag,
            connection_type,
        ) in self._iter_connection_infos(connection_list):
            yield (
                _PortTagString(source_tag, getattr(connection_info, 'source', None)),
                _PortTagString(
                    destination_tag,
                    getattr(connection_info, 'destination', None),
                ),
                connection_type,
            )

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
                    label='X',
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
