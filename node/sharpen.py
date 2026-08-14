#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sharpen implementations matching the Photo Editor and Polish apps."""

import numpy as np


def _color_and_alpha(image):
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None, None
    color = image[:, :, :3]
    alpha = image[:, :, 3:] if image.shape[2] > 3 else None
    return color, alpha


def _restore_extra_channels(color, extra_channels):
    if extra_channels is None:
        return color
    return np.concatenate((color, extra_channels), axis=2)


def photo_editor_sharpen(image, slider):
    """Apply Photo Editor's eight-neighbor sharpen at strength ``0..500``."""
    color, extra_channels = _color_and_alpha(image)
    if color is None:
        return image

    slider = np.clip(float(slider), 0.0, 500.0)
    if slider == 0.0:
        return image.copy()

    source = color.astype(np.float64)
    padded = np.pad(source, ((1, 1), (1, 1), (0, 0)), mode='edge')
    neighbor_sum = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    strength = slider / 500.0
    sharpened = source + strength * (2.0 * source - 0.25 * neighbor_sum)
    result = np.clip(np.floor(sharpened), 0, 255).astype(np.uint8)
    return _restore_extra_channels(result, extra_channels)


def polish_preview_offsets(image_shape, preview_width=1080):
    """Return source-pixel offsets for one pixel in Polish's preview space."""
    height, width = image_shape[:2]
    preview_width = max(1, int(preview_width))
    preview_height = max(1, round(height * preview_width / width))
    return width / preview_width, height / preview_height


def _bilinear_sample(source, x_coordinates, y_coordinates):
    """Sample normalized RGB with clamp-to-edge addressing."""
    height, width = source.shape[:2]
    x = np.clip(x_coordinates, 0.0, width - 1.0)
    y = np.clip(y_coordinates, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
    bottom = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _mediump(value):
    """Approximate a GLSL mediump texture value with IEEE binary16."""
    return value.astype(np.float16).astype(np.float32)


def polish_sharpen(image, slider, preview_width=1080):
    """Apply Polish's preview-space four-neighbor sharpen.

    Polish's native slider ends at 100, but values through 500 are accepted so
    the shared Sharpen node can use one strength range for both methods.
    """
    color, extra_channels = _color_and_alpha(image)
    if color is None:
        return image

    slider = np.clip(float(slider), 0.0, 500.0)
    if slider == 0.0:
        return image.copy()

    source = color.astype(np.float32) / 255.0
    height, width = source.shape[:2]
    offset_x, offset_y = polish_preview_offsets(source.shape, preview_width)
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)

    center = _mediump(source)
    left = _mediump(_bilinear_sample(source, grid_x - offset_x, grid_y))
    right = _mediump(_bilinear_sample(source, grid_x + offset_x, grid_y))
    up = _mediump(_bilinear_sample(source, grid_x, grid_y - offset_y))
    down = _mediump(_bilinear_sample(source, grid_x, grid_y + offset_y))
    strength = np.float32(slider / 100.0)
    sharpened = center + strength * (4.0 * center - left - right - up - down)
    result = np.clip(np.rint(sharpened * 255.0), 0, 255).astype(np.uint8)
    return _restore_extra_channels(result, extra_channels)
