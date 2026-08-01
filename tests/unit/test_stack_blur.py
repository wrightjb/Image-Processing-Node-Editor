import numpy as np
import pytest

from node.stack_blur import MUL_TABLE, SHG_TABLE, stack_blur


def test_radius_150_matches_photo_editor_integer_reference():
    source = np.array(
        [
            [[0, 10, 250, 0], [64, 50, 200, 17], [128, 100, 150, 64],
             [192, 150, 100, 127], [255, 200, 50, 255]],
            [[255, 0, 40, 1], [192, 50, 80, 18], [128, 100, 120, 65],
             [64, 150, 160, 128], [0, 200, 200, 254]],
            [[7, 250, 0, 2], [40, 200, 50, 19], [90, 150, 100, 66],
             [160, 100, 150, 129], [240, 50, 255, 253]],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [[121, 126, 137, 0], [123, 126, 138, 17], [125, 126, 138, 64],
             [126, 126, 138, 127], [128, 126, 138, 255]],
            [[121, 126, 137, 1], [123, 126, 137, 18], [125, 126, 137, 65],
             [126, 126, 137, 128], [128, 126, 138, 254]],
            [[121, 127, 137, 2], [123, 127, 137, 19], [125, 127, 137, 66],
             [126, 127, 137, 129], [127, 127, 138, 253]],
        ],
        dtype=np.uint8,
    )

    result = stack_blur(source, 150)

    assert (MUL_TABLE[150], SHG_TABLE[150]) == (368, 23)
    assert result.shape == source.shape
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[:, :, 3], source[:, :, 3])


def test_stack_blur_clamps_edges_and_runs_horizontal_then_vertical():
    source = np.array([[0, 0], [0, 255]], dtype=np.uint8)

    result = stack_blur(source, 1)

    np.testing.assert_array_equal(result, np.array([[15, 47], [47, 143]], dtype=np.uint8))


def test_stack_blur_rejects_non_8_bit_input():
    with pytest.raises(TypeError, match='8-bit'):
        stack_blur(np.zeros((2, 2), dtype=np.float32), 3)


def test_stack_blur_supports_radius_1000():
    source = np.array([[0, 255]], dtype=np.uint8)

    result = stack_blur(source, 1000)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
