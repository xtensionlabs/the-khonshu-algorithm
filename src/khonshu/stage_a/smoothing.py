"""Spline and polyline sampling for an already selected centerline route."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .config import StageAConfig


def smooth_route(
    route_points: np.ndarray,
    config: StageAConfig,
    cubic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a route at fixed spacing and calculate signed curvature."""

    keep = [0]
    for index in range(1, len(route_points)):
        if float(np.linalg.norm(route_points[index] - route_points[keep[-1]])) >= 0.25:
            keep.append(index)
    route_points = route_points[keep]
    if len(route_points) < 2:
        return np.empty(0), np.empty((0, 2)), np.empty(0)
    segment_lengths = np.linalg.norm(np.diff(route_points, axis=0), axis=1)
    control_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cubic and len(route_points) >= 3:
        spline = CubicSpline(control_s, route_points, axis=0, bc_type="natural")
        sample_s = np.arange(0.0, control_s[-1], config.spline_sample_spacing_m)
        if not len(sample_s) or control_s[-1] - sample_s[-1] > 1e-6:
            sample_s = np.append(sample_s, control_s[-1])
        xy = spline(sample_s)
        first = spline(sample_s, 1)
        second = spline(sample_s, 2)
        denominator = np.maximum(np.sum(first * first, axis=1) ** 1.5, 1e-9)
        curvature = (
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        ) / denominator
        return sample_s, xy, curvature
    sample_s = np.arange(0.0, control_s[-1], config.spline_sample_spacing_m)
    if not len(sample_s) or control_s[-1] - sample_s[-1] > 1e-6:
        sample_s = np.append(sample_s, control_s[-1])
    xy = np.column_stack(
        [np.interp(sample_s, control_s, route_points[:, axis]) for axis in range(2)]
    )
    return sample_s, xy, np.zeros(len(sample_s), dtype=np.float64)
