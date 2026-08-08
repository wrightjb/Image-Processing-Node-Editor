#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curve interpolation helpers shared by Curves UI, processing, and tuning."""

import numpy as np
from scipy.interpolate import CubicSpline, PPoly


def points_to_lut(points, interpolation="linear"):
    """Return a 256-entry uint8 LUT from editable curve control points."""
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    xs = points[:, 0]
    ys = points[:, 1]
    x_values = np.arange(256, dtype=np.float64)

    if interpolation != "spline" or len(points) < 3:
        result = np.interp(x_values, xs, ys)
        return np.rint(np.clip(result, 0, 255)).astype(np.uint8)

    # Uniform parameterization by control-point order.
    t = np.arange(len(points), dtype=np.float64)

    # One parametric natural cubic:
    # curve(t) returns [x(t), y(t)].
    curve = CubicSpline(
        t,
        points,
        axis=0,
        bc_type="natural",
        extrapolate=False,
    )

    # Extract x(t) as a scalar piecewise polynomial so it can be inverted.
    x_curve = PPoly(
        curve.c[:, :, 0],
        curve.x,
        extrapolate=False,
    )

    result = np.empty(256, dtype=np.float64)

    for target_x in range(256):
        roots = x_curve.solve(
            y=float(target_x),
            discontinuity=False,
            extrapolate=False,
        )

        # solve() may return NaN for an identically matching segment.
        roots = roots[np.isfinite(roots)]
        roots = roots[
            (roots >= t[0] - 1e-9)
            & (roots <= t[-1] + 1e-9)
        ]

        if roots.size:
            roots = np.clip(roots, t[0], t[-1])

            # Remove duplicated roots at spline-segment boundaries.
            roots = np.unique(np.round(roots, decimals=12))

            candidate_y = curve(roots)[:, 1]

            # Branch rule: use the greatest output value at this x.
            result[target_x] = np.max(candidate_y)
        else:
            # Normally unreachable when the endpoints span x=0 through 255.
            result[target_x] = np.interp(target_x, xs, ys)

    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)