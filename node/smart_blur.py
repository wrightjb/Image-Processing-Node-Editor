#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from node.stack_blur import stack_blur


def smart_blur_edge_magnitude(image):
    """Return Photo Editor-style max(right, below) RGB edge magnitudes."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('Smart Blur requires an image with at least 3 channels')

    # OpenCV images are BGR(A), but an unweighted average is channel-order
    # independent. Integer division matches the app's 8-bit intensity path.
    intensity = image[:, :, :3].astype(np.int32).sum(axis=2) // 3
    right = np.empty_like(intensity)
    below = np.empty_like(intensity)
    right[:, :-1] = intensity[:, 1:]
    right[:, -1] = intensity[:, -1]
    below[:-1, :] = intensity[1:, :]
    below[-1, :] = intensity[-1, :]
    return np.maximum(np.abs(intensity - right), np.abs(intensity - below))


def _divide_toward_zero(numerator, denominator):
    magnitude = np.abs(numerator) // denominator
    return np.where(numerator < 0, -magnitude, magnitude)


def smart_blur(image, radius, threshold):
    """Stack-blur an image and restore edges detected in the sharp source."""
    threshold = int(threshold)
    if threshold <= 0:
        raise ValueError('Smart Blur threshold must be greater than zero')

    blurred = stack_blur(image, radius)
    edge = smart_blur_edge_magnitude(image)
    denominator = 2 * threshold
    edge = np.minimum(edge, denominator).astype(np.int64)

    original_color = image[:, :, :3].astype(np.int64)
    blurred_color = blurred[:, :, :3].astype(np.int64)
    delta = (original_color - blurred_color) * edge[:, :, None]
    restored = blurred_color + _divide_toward_zero(delta, denominator)

    result = blurred.copy()
    result[:, :, :3] = restored.astype(np.uint8)
    return result
