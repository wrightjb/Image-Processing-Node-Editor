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
    port_name: str
    dpg_tag: str
    value_tag: str = None
    control_tag: str = None


@dataclass(frozen=True)
class LinkRef:
    source: PortRef
    destination: PortRef

    @property
    def source_tag(self):
        return self.source.dpg_tag

    @property
    def destination_tag(self):
        return self.destination.dpg_tag

    @property
    def legacy_pair(self):
        return [self.source_tag, self.destination_tag]

    def __iter__(self):
        yield self.source
        yield self.destination

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.source, self.destination)[index]
