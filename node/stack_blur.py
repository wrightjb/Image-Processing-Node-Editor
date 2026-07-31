#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


MAX_RADIUS = 1000


def _normalization_for_radius(radius):
    """Return the classic Stack Blur reciprocal multiplier and shift."""
    divisor = (radius + 1) ** 2
    shift = max(0, divisor.bit_length() + 8)
    multiplier = ((1 << shift) + divisor - 1) // divisor
    while multiplier > 512:
        shift -= 1
        multiplier = ((1 << shift) + divisor - 1) // divisor
    return multiplier, shift


_NORMALIZATION_TABLE = tuple(
    _normalization_for_radius(radius) for radius in range(MAX_RADIUS + 1)
)
MUL_TABLE = tuple(value[0] for value in _NORMALIZATION_TABLE)
SHG_TABLE = tuple(value[1] for value in _NORMALIZATION_TABLE)


def _stack_blur_axis(source, radius, axis, multiplier, shift):
    """Run one integer Stack Blur pass over all scanlines on an axis."""
    pixels = np.swapaxes(source, 0, axis).astype(np.int64, copy=False)
    length = pixels.shape[0]
    result = np.empty_like(pixels, dtype=np.uint8)

    edge_weight = (radius + 1) + (radius * (radius + 1) // 2)
    weighted_sum = pixels[0] * edge_weight
    incoming_sum = np.zeros_like(pixels[0])
    outgoing_sum = pixels[0] * (radius + 1)
    for offset in range(1, radius + 1):
        sample = pixels[min(offset, length - 1)]
        weighted_sum += sample * (radius + 1 - offset)
        incoming_sum += sample

    for position in range(length):
        result[position] = (weighted_sum * multiplier) >> shift

        weighted_sum -= outgoing_sum
        outgoing_index = max(position - radius, 0)
        outgoing_sum -= pixels[outgoing_index]

        incoming_index = min(position + radius + 1, length - 1)
        incoming_sum += pixels[incoming_index]
        weighted_sum += incoming_sum

        center_index = min(position + 1, length - 1)
        outgoing_sum += pixels[center_index]
        incoming_sum -= pixels[center_index]

    return np.swapaxes(result, 0, axis)


def stack_blur(image, radius):
    """Apply Mario Klingemann's two-pass, 8-bit integer Stack Blur."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError('Stack Blur requires an 8-bit NumPy image')
    if image.ndim not in (2, 3):
        raise ValueError('Stack Blur requires a 2D image or channel-last image')

    radius = int(radius)
    if radius < 0 or radius > MAX_RADIUS:
        raise ValueError(f'Stack Blur radius must be between 0 and {MAX_RADIUS}')
    if radius == 0 or image.size == 0:
        return image.copy()

    alpha = None
    channels = image
    if image.ndim == 3 and image.shape[2] == 4:
        channels = image[:, :, :3]
        alpha = image[:, :, 3].copy()

    multiplier = MUL_TABLE[radius]
    shift = SHG_TABLE[radius]
    horizontal = _stack_blur_axis(channels, radius, 1, multiplier, shift)
    blurred = _stack_blur_axis(horizontal, radius, 0, multiplier, shift)

    if alpha is None:
        return blurred
    return np.dstack((blurred, alpha))
