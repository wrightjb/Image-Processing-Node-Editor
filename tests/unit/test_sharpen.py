import numpy as np

from node.process_node.node_sharpen import Node as SharpenNode
from node.process_node.node_sharpen import image_process
from node.sharpen import photo_editor_sharpen, polish_preview_offsets, polish_sharpen


def _photo_editor_reference(image, slider):
    source = image.astype(np.float64)
    height, width = source.shape[:2]
    output = np.empty_like(image)
    strength = slider / 500.0
    for y in range(height):
        for x in range(width):
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx or dy:
                        neighbors.append(
                            source[
                                min(max(y + dy, 0), height - 1),
                                min(max(x + dx, 0), width - 1),
                            ]
                        )
            value = source[y, x] + strength * (
                2.0 * source[y, x] - sum(neighbors) / 4.0
            )
            output[y, x] = np.clip(np.floor(value), 0, 255)
    return output


def test_photo_editor_matches_eight_neighbor_reference_at_verified_strengths():
    image = np.array(
        [
            [[7, 51, 203], [32, 90, 121], [250, 6, 80]],
            [[99, 100, 101], [128, 129, 130], [15, 220, 33]],
            [[0, 255, 64], [177, 43, 211], [88, 144, 199]],
        ],
        dtype=np.uint8,
    )

    for slider in (125, 250, 500):
        result = photo_editor_sharpen(image, slider)
        assert np.array_equal(result, _photo_editor_reference(image, slider))


def test_photo_editor_floors_fractional_values_and_preserves_alpha():
    image = np.zeros((3, 3, 4), dtype=np.uint8)
    image[:, :, :3] = 10
    image[1, 1, :3] = 11
    image[:, :, 3] = np.arange(9, dtype=np.uint8).reshape(3, 3)

    result = photo_editor_sharpen(image, 125)

    assert np.array_equal(result[1, 1, :3], [11, 11, 11])
    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_polish_offsets_follow_preview_coordinate_system():
    assert polish_preview_offsets((1000, 1080, 3)) == (1.0, 1.0)
    assert polish_preview_offsets((1000, 2160, 3)) == (2.0, 2.0)
    offset_x, offset_y = polish_preview_offsets((900, 1408, 3))
    assert np.isclose(offset_x, 1408 / 1080)
    assert np.isclose(offset_y, 900 / round(900 * 1080 / 1408))


def test_polish_uses_axial_neighbors_and_clamps_edges_at_1080_width():
    row = np.arange(1080, dtype=np.uint16) % 256
    image = np.repeat(row[None, :, None], 3, axis=2).astype(np.uint8)

    result = polish_sharpen(image, 100)

    source = image.astype(np.float32) / 255.0
    center = source.astype(np.float16).astype(np.float32)
    left = np.concatenate((source[:, :1], source[:, :-1]), axis=1)
    right = np.concatenate((source[:, 1:], source[:, -1:]), axis=1)
    left = left.astype(np.float16).astype(np.float32)
    right = right.astype(np.float16).astype(np.float32)
    expected = center + (2.0 * center - left - right)
    expected = np.clip(np.rint(expected * 255.0), 0, 255).astype(np.uint8)
    assert np.array_equal(result, expected)


def test_polish_bilinearly_samples_a_fractional_preview_offset():
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3) * 2

    # A six-pixel source rendered four pixels wide samples 1.5 source pixels
    # away. The inferred three-pixel preview height gives a 5/3 y offset.
    result = polish_sharpen(image, 50, preview_width=4)
    source = image.astype(np.float32) / 255.0
    center = source[2, 3].astype(np.float16).astype(np.float32)
    left = 0.5 * source[2, 1] + 0.5 * source[2, 2]
    right = 0.5 * source[2, 4] + 0.5 * source[2, 5]
    up = (2 / 3) * source[0, 3] + (1 / 3) * source[1, 3]
    down = (1 / 3) * source[3, 3] + (2 / 3) * source[4, 3]
    neighbors = [
        value.astype(np.float16).astype(np.float32)
        for value in (left, right, up, down)
    ]
    expected = center + 0.5 * (4.0 * center - sum(neighbors))
    expected = np.clip(np.rint(expected * 255.0), 0, 255).astype(np.uint8)
    assert np.array_equal(result[2, 3], expected)


def test_polish_shared_slider_supports_strengths_above_native_range():
    image = np.full((3, 3, 3), 100, dtype=np.uint8)
    image[1, 1] = 110

    assert not np.array_equal(
        polish_sharpen(image, 500),
        polish_sharpen(image, 100),
    )


def test_sharpen_node_exposes_method_dropdown_and_shared_slider_range():
    method, strength, preview_width = SharpenNode.parameters
    assert method['items'] == ('Photo Editor', 'Polish')
    assert method['default'] == 'Photo Editor'
    assert (strength['min'], strength['max']) == (0, 500)
    assert preview_width['default'] == 1080


def test_sharpen_node_dispatches_selected_method():
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)

    assert np.array_equal(
        image_process(image, 'Photo Editor', 125),
        photo_editor_sharpen(image, 125),
    )
    assert np.array_equal(
        image_process(image, 'Polish', 125, preview_width=3),
        polish_sharpen(image, 125, preview_width=3),
    )
