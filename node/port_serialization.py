from node.port_model import (
    NodeRef,
    PortDataType,
    PortDirection,
    PortRef,
    enum_value,
    normalize_port_data_type,
    normalize_port_direction,
)


def legacy_port_name(direction, index):
    direction = normalize_port_direction(direction)
    return f'{direction.value}{int(index):02d}'


def make_port_tag(node_id_name, data_type, port_name):
    return f'{node_id_name}:{enum_value(data_type)}:{port_name}'


def make_port_value_tag(port_tag):
    return f'{port_tag}Value'


def port_ref_to_tag(port_ref):
    return make_port_tag(
        port_ref.node_ref.node_id_name,
        port_ref.data_type,
        port_ref.port_name,
    )


def port_ref_from_tag(port_tag):
    if not isinstance(port_tag, str):
        return None
    parts = port_tag.split(':')
    if len(parts) < 4:
        return None

    node_ref = NodeRef(parts[0], parts[1])
    try:
        data_type = normalize_port_data_type(parts[2])
    except ValueError:
        return None
    port_name = parts[3]
    if port_name.startswith(PortDirection.INPUT.value):
        direction = PortDirection.INPUT
        index_text = port_name[len(PortDirection.INPUT.value):]
    elif port_name.startswith(PortDirection.OUTPUT.value):
        direction = PortDirection.OUTPUT
        index_text = port_name[len(PortDirection.OUTPUT.value):]
    else:
        return None

    try:
        index = int(index_text)
    except ValueError:
        return None

    return PortRef(
        node_ref=node_ref,
        direction=direction,
        data_type=data_type,
        index=index,
        port_name=port_name,
        dpg_tag=port_tag,
        value_tag=make_port_value_tag(port_tag),
    )
