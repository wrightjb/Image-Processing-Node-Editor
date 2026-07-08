import numpy as np

from node.draw_node.node_mask_composite import image_process


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
