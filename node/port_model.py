from dataclasses import dataclass, replace
from enum import Enum
from types import SimpleNamespace


class PortDirection(str, Enum):
    INPUT = 'Input'
    OUTPUT = 'Output'


class PortDataType(str, Enum):
    INT = 'Int'
    FLOAT = 'Float'
    IMAGE = 'Image'
    TIME_MS = 'TimeMS'
    TEXT = 'Text'


def enum_value(value):
    if isinstance(value, Enum):
        return value.value
    return value


def normalize_port_direction(direction):
    if isinstance(direction, PortDirection):
        return direction
    return PortDirection(direction)


def normalize_port_data_type(data_type):
    if isinstance(data_type, PortDataType):
        return data_type
    return PortDataType(data_type)


@dataclass(frozen=True)
class NodeRef:
    node_id: str
    node_tag: str

    @property
    def node_id_name(self):
        return f'{self.node_id}:{self.node_tag}'


@dataclass(frozen=True)
class PortSpec:
    direction: PortDirection
    data_type: PortDataType
    key: str = None
    index: int = None
    label: str = None
    control_tag: str = None
    default_control_tag: bool = False

    def with_key(self, key):
        if self.key == key:
            return self
        return replace(self, key=key)


def InputPort(
    data_type,
    index=None,
    label=None,
    control_tag=None,
    default_control_tag=False,
):
    return PortSpec(
        direction=PortDirection.INPUT,
        data_type=normalize_port_data_type(data_type),
        index=index,
        label=label,
        control_tag=control_tag,
        default_control_tag=default_control_tag,
    )


def OutputPort(data_type, index=None, label=None, control_tag=None):
    return PortSpec(
        direction=PortDirection.OUTPUT,
        data_type=normalize_port_data_type(data_type),
        index=index,
        label=label,
        control_tag=control_tag,
    )


def ParameterPort(data_type, index=None, label=None, control_tag=None):
    return InputPort(
        data_type,
        index=index,
        label=label,
        control_tag=control_tag,
        default_control_tag=True,
    )


class PortSpecs:
    def __init__(self, **specs):
        self._specs = tuple(
            spec.with_key(key) for key, spec in specs.items()
        )

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)

    def __bool__(self):
        return bool(self._specs)


class PortHandles(SimpleNamespace):
    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class PortRef:
    node_ref: NodeRef
    direction: PortDirection
    data_type: PortDataType
    index: int
    port_name: str
    dpg_tag: str
    value_tag: str = None
    control_tag: str = None
    spec_key: str = None


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


@dataclass(frozen=True)
class LinkConnectionAdapter:
    link_ref: LinkRef

    @property
    def source(self):
        return self.link_ref.source

    @property
    def destination(self):
        return self.link_ref.destination

    @property
    def source_tag(self):
        return self.link_ref.source_tag

    @property
    def destination_tag(self):
        return self.link_ref.destination_tag

    @property
    def legacy_pair(self):
        return self.link_ref.legacy_pair

    def __iter__(self):
        yield self.source_tag
        yield self.destination_tag

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.source_tag, self.destination_tag)[index]
