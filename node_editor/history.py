from dataclasses import dataclass


@dataclass(frozen=True)
class AddNodeCommand:
    node_id: int
    node_tag: str
    pos: list
    setting: dict
    created_links: list
    replaced_links: list

    def undo(self, editor):
        node_id_name = editor._cntrl_resolve_history_node_id_name(
            f'{self.node_id}:{self.node_tag}'
        )
        editor._cntrl_delete_node_by_tag(node_id_name)
        for source_tag, dest_tag in self.replaced_links:
            editor._cntrl_add_link_by_tags(source_tag, dest_tag)

    def redo(self, editor):
        recreated_node_id_name = editor._cntrl_add_node_from_history(
            self.node_tag,
            self.node_id,
            self.pos,
            self.setting,
        )
        for source_tag, dest_tag in self.replaced_links:
            editor._cntrl_remove_link_by_tags(source_tag, dest_tag)
        for source_tag, dest_tag in self.created_links:
            editor._cntrl_add_link_by_tags(source_tag, dest_tag)
        return recreated_node_id_name


@dataclass(frozen=True)
class DeleteNodesCommand:
    nodes: list
    removed_links: list
    healed_links: list
    removed_selected_links: list

    def undo(self, editor):
        for node_info in self.nodes:
            editor._cntrl_add_node_from_history(
                node_info['node_tag'],
                int(node_info['node_id']),
                node_info['pos'],
                node_info['setting'],
            )
        for source_tag, dest_tag in self.healed_links:
            editor._cntrl_remove_link_by_tags(source_tag, dest_tag)
        for source_tag, dest_tag in self.removed_links + self.removed_selected_links:
            editor._cntrl_add_link_by_tags(source_tag, dest_tag)

    def redo(self, editor):
        for source_tag, dest_tag in self.removed_selected_links:
            editor._cntrl_remove_link_by_tags(source_tag, dest_tag)
        for source_tag, dest_tag in self.removed_links:
            editor._cntrl_remove_link_by_tags(source_tag, dest_tag)
        for node_info in self.nodes:
            editor._cntrl_delete_node_by_tag(editor._cntrl_resolve_history_node_id_name(
                f"{node_info['node_id']}:{node_info['node_tag']}"
            ))
        for source_tag, dest_tag in self.healed_links:
            editor._cntrl_add_link_by_tags(source_tag, dest_tag)


@dataclass(frozen=True)
class MoveNodeCommand:
    node_id_name: str
    before_pos: list
    after_pos: list

    def undo(self, editor):
        resolved = editor._cntrl_resolve_history_node_id_name(self.node_id_name)
        editor._vw_set_node_pos(resolved, self.before_pos)
        editor._node_position_cache[resolved] = list(self.before_pos)

    def redo(self, editor):
        resolved = editor._cntrl_resolve_history_node_id_name(self.node_id_name)
        editor._vw_set_node_pos(resolved, self.after_pos)
        editor._node_position_cache[resolved] = list(self.after_pos)


@dataclass(frozen=True)
class AddLinkCommand:
    source_tag: str
    dest_tag: str

    def undo(self, editor):
        editor._cntrl_remove_link_by_tags(self.source_tag, self.dest_tag)

    def redo(self, editor):
        editor._cntrl_add_link_by_tags(self.source_tag, self.dest_tag)


@dataclass(frozen=True)
class RemoveLinkCommand:
    source_tag: str
    dest_tag: str

    def undo(self, editor):
        editor._cntrl_add_link_by_tags(self.source_tag, self.dest_tag)

    def redo(self, editor):
        editor._cntrl_remove_link_by_tags(self.source_tag, self.dest_tag)


@dataclass(frozen=True)
class ReplaceLinkCommand:
    old_source_tag: str
    new_source_tag: str
    dest_tag: str

    def undo(self, editor):
        editor._cntrl_remove_link_by_tags(self.new_source_tag, self.dest_tag)
        editor._cntrl_add_link_by_tags(self.old_source_tag, self.dest_tag)

    def redo(self, editor):
        editor._cntrl_remove_link_by_tags(self.old_source_tag, self.dest_tag)
        editor._cntrl_add_link_by_tags(self.new_source_tag, self.dest_tag)


@dataclass(frozen=True)
class CompositeCommand:
    commands: list

    def undo(self, editor):
        for command in reversed(self.commands):
            command.undo(editor)

    def redo(self, editor):
        for command in self.commands:
            command.redo(editor)


@dataclass(frozen=True)
class SetParameterCommand:
    node_id_name: str
    value_tag: str
    before_value: object
    after_value: object

    def undo(self, editor):
        resolved_tag = editor._cntrl_resolve_history_port_tag(self.value_tag)
        editor._cntrl_set_parameter_value(resolved_tag, self.before_value)

    def redo(self, editor):
        resolved_tag = editor._cntrl_resolve_history_port_tag(self.value_tag)
        editor._cntrl_set_parameter_value(resolved_tag, self.after_value)


def history_command_label(command, parameter_label_resolver=None):
    if isinstance(command, AddNodeCommand):
        return f'Add node: {command.node_tag}'
    if isinstance(command, DeleteNodesCommand):
        return f'Delete node(s): {len(command.nodes)}'
    if isinstance(command, MoveNodeCommand):
        return f'Move node: {command.node_id_name}'
    if isinstance(command, AddLinkCommand):
        return 'Add link'
    if isinstance(command, RemoveLinkCommand):
        return 'Remove link'
    if isinstance(command, ReplaceLinkCommand):
        return 'Replace link'
    if isinstance(command, CompositeCommand):
        if not command.commands:
            return 'Composite (empty)'
        add_node_command = next(
            (cmd for cmd in command.commands if isinstance(cmd, AddNodeCommand)),
            None,
        )
        if add_node_command is not None:
            return f'Insert node: {add_node_command.node_tag}'
        return f'Composite: {history_command_label(command.commands[0], parameter_label_resolver)}'
    if isinstance(command, SetParameterCommand):
        if callable(parameter_label_resolver):
            resolved_label = parameter_label_resolver(command.value_tag)
            if resolved_label:
                parts = str(command.value_tag).split(':')
                node_name = f'{parts[0]}:{parts[1]}' if len(parts) >= 2 else command.node_id_name
                return f'Set parameter: {node_name}.{resolved_label}'
        parts = str(command.value_tag).split(':')
        if len(parts) >= 4:
            node_name = f'{parts[0]}:{parts[1]}'
            parameter_name = parts[3]
            if parameter_name.endswith('Value'):
                parameter_name = parameter_name[:-5]
            return f'Set parameter: {node_name}.{parameter_name}'
        return f'Set parameter: {command.value_tag}'
    return command.__class__.__name__
