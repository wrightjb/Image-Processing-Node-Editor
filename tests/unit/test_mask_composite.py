import numpy as np

from node.draw_node import node_mask_composite
from node.draw_node.node_mask_composite import Node, image_process
from node.port_model import (
    LinkConnectionAdapter,
    LinkRef,
    NodeRef,
    PortDataType,
    PortDirection,
    PortRef,
)


def test_mask_composite_uses_white_mask_for_image_a_and_black_for_image_b():
    image_a = np.full((2, 2, 3), 100, dtype=np.uint8)
    image_b = np.full((2, 2, 3), 25, dtype=np.uint8)
    mask = np.array(
        [
            [[255, 255, 255], [0, 0, 0]],
            [[0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    result = image_process(image_a, mask, image_b)

    assert result[0, 0].tolist() == [100, 100, 100]
    assert result[0, 1].tolist() == [25, 25, 25]
    assert result[1, 0].tolist() == [25, 25, 25]
    assert result[1, 1].tolist() == [100, 100, 100]


def test_mask_composite_can_invert_threshold_style_mask():
    image_a = np.full((1, 2, 3), 100, dtype=np.uint8)
    image_b = np.full((1, 2, 3), 25, dtype=np.uint8)
    mask = np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8)

    result = image_process(image_a, mask, image_b, invert_mask=True)

    assert result[0, 0].tolist() == [25, 25, 25]
    assert result[0, 1].tolist() == [100, 100, 100]


def _port_ref(node_id, node_tag, direction, data_type, port_name):
    if direction == PortDirection.INPUT:
        prefix = PortDirection.INPUT.value
    else:
        prefix = PortDirection.OUTPUT.value
    return PortRef(
        node_ref=NodeRef(str(node_id), node_tag),
        direction=direction,
        data_type=data_type,
        index=int(port_name[len(prefix):]),
        port_name=port_name,
        dpg_tag=f'{node_id}:{node_tag}:{data_type.value}:{port_name}',
        value_tag=f'{node_id}:{node_tag}:{data_type.value}:{port_name}Value',
    )


def test_mask_composite_update_reads_typed_image_links(monkeypatch):
    status_updates = []
    monkeypatch.setattr(node_mask_composite, 'dpg_get_value', lambda tag: False)
    monkeypatch.setattr(
        node_mask_composite,
        'dpg_set_value',
        lambda tag, value: status_updates.append((tag, value)),
    )
    monkeypatch.setattr(
        node_mask_composite,
        'convert_cv_to_dpg',
        lambda image, width, height: image,
    )

    node = Node()
    node.create_ports('3')
    node._opencv_setting_dict = {
        'process_width': 2,
        'process_height': 2,
        'use_pref_counter': False,
    }
    image_a = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask = np.array(
        [
            [[255, 255, 255], [0, 0, 0]],
            [[0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    image_link = LinkConnectionAdapter(
        LinkRef(
            _port_ref(
                1,
                'Image',
                PortDirection.OUTPUT,
                PortDataType.IMAGE,
                'Output01',
            ),
            node.ports('3').image_a,
        )
    )
    mask_link = LinkConnectionAdapter(
        LinkRef(
            _port_ref(
                2,
                'Threshold',
                PortDirection.OUTPUT,
                PortDataType.IMAGE,
                'Output01',
            ),
            node.ports('3').mask,
        )
    )

    frame, _ = node.update(
        '3',
        [image_link, mask_link],
        {'1:Image': image_a, '2:Threshold': mask},
        {},
    )

    assert frame[0, 0].tolist() == [100, 100, 100]
    assert frame[0, 1].tolist() == [0, 0, 0]
    assert any(
        'composited; mask white 50.0%' == value
        for _, value in status_updates
    )
