#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curve interpolation helpers shared by Curves UI, processing, and tuning."""

import numpy as np


def points_to_lut(points, interpolation='linear'):
    """Return a 256-entry uint8 LUT for editable curve control points."""
    xs, ys = zip(*points)
    x_values = np.arange(256, dtype=np.float32)
    if interpolation != 'spline' or len(points) < 3:
        return np.interp(x_values, xs, ys).astype(np.uint8)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    result = np.interp(x_values, xs, ys).astype(np.float32)
    for index in range(len(xs) - 1):
        mask = (x_values >= xs[index]) & (x_values <= xs[index + 1])
        if not np.any(mask):
            continue
        x0 = xs[max(0, index - 1)]
        x1 = xs[index]
        x2 = xs[index + 1]
        x3 = xs[min(len(xs) - 1, index + 2)]
        y0 = ys[max(0, index - 1)]
        y1 = ys[index]
        y2 = ys[index + 1]
        y3 = ys[min(len(ys) - 1, index + 2)]
        if x2 <= x1:
            continue
        t = (x_values[mask] - x1) / (x2 - x1)
        m1 = 0.0 if x2 == x0 else (y2 - y0) * (x2 - x1) / (x2 - x0)
        m2 = 0.0 if x3 == x1 else (y3 - y1) * (x2 - x1) / (x3 - x1)
        h00 = (2 * t**3) - (3 * t**2) + 1
        h10 = t**3 - (2 * t**2) + t
        h01 = (-2 * t**3) + (3 * t**2)
        h11 = t**3 - t**2
        result[mask] = (h00 * y1) + (h10 * m1) + (h01 * y2) + (h11 * m2)
    return np.clip(result, 0, 255).astype(np.uint8)
