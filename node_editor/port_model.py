from dataclasses import dataclass


@dataclass(frozen=True)
class NodeRef:
    node_id: str
    node_tag: str

    @property
    def node_id_name(self):
        return f'{self.node_id}:{self.node_tag}'


@dataclass(frozen=True)
class PortRef:
    node_ref: NodeRef
    direction: str
    data_type: str
    index: int
    dpg_tag: str
    value_tag: str = None
    control_tag: str = None
