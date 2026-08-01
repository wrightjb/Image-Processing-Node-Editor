import numpy as np
import pytest

from node.smart_blur import smart_blur, smart_blur_edge_magnitude


SOURCE = np.array(
    [
        [[0, 0, 0, 1], [30, 60, 90, 2], [200, 10, 40, 3],
         [255, 255, 255, 4]],
        [[10, 100, 40, 5], [80, 80, 80, 6], [20, 220, 100, 7],
         [240, 30, 180, 8]],
        [[255, 0, 0, 9], [0, 255, 0, 10], [0, 0, 255, 11],
         [127, 128, 129, 12]],
    ],
    dtype=np.uint8,
)


REFERENCE_OUTPUTS = {
    (20, 1): [
        [[0, 0, 0, 1], [30, 60, 90, 2], [200, 10, 40, 3], [255, 255, 255, 4]],
        [[10, 100, 40, 5], [80, 80, 80, 6], [20, 220, 100, 7],
         [240, 30, 180, 8]],
        [[147, 79, 81, 9], [149, 87, 90, 10], [0, 0, 255, 11],
         [154, 102, 106, 12]],
    ],
    (20, 10): [
        [[0, 0, 0, 1], [30, 60, 90, 2], [200, 10, 40, 3], [255, 255, 255, 4]],
        [[10, 100, 40, 5], [80, 80, 80, 6], [20, 220, 100, 7],
         [240, 30, 180, 8]],
        [[147, 79, 81, 9], [149, 87, 90, 10], [0, 0, 255, 11],
         [154, 102, 106, 12]],
    ],
    (20, 30): [
        [[0, 0, 0, 1], [101, 80, 93, 2], [200, 10, 40, 3],
         [255, 255, 255, 4]],
        [[66, 92, 58, 5], [110, 85, 86, 6], [70, 172, 100, 7],
         [184, 78, 135, 8]],
        [[147, 79, 81, 9], [149, 87, 90, 10], [43, 27, 210, 11],
         [154, 102, 106, 12]],
    ],
    (1, 30): [
        [[0, 0, 0, 1], [49, 56, 71, 2], [200, 10, 40, 3],
         [255, 255, 255, 4]],
        [[33, 86, 35, 5], [68, 89, 74, 6], [49, 178, 109, 7],
         [199, 81, 173, 8]],
        [[150, 71, 12, 9], [59, 125, 66, 10], [13, 30, 224, 11],
         [117, 91, 160, 12]],
    ],
}


def _assert_exact_with_error_metrics(actual, expected):
    if np.array_equal(actual, expected):
        return
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    pytest.fail(
        f'maximum error={error.max()}, mean absolute error={error.mean():.6f}, '
        f'differing-pixel count={np.count_nonzero(error)}'
    )


@pytest.mark.parametrize('radius,threshold', REFERENCE_OUTPUTS)
def test_smart_blur_matches_reference_outputs(radius, threshold):
    expected = np.asarray(REFERENCE_OUTPUTS[(radius, threshold)], dtype=np.uint8)

    result = smart_blur(SOURCE, radius, threshold)

    _assert_exact_with_error_metrics(result, expected)
    np.testing.assert_array_equal(result[:, :, 3], SOURCE[:, :, 3])


def test_edge_magnitude_isolated_forward_difference_combination():
    expected = np.array(
        [[60, 23, 172, 105], [35, 33, 37, 22], [0, 0, 43, 0]],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(smart_blur_edge_magnitude(SOURCE), expected)
