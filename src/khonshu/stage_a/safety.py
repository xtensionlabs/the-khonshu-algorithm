"""Vehicle-footprint and track-section safety checks for Stage A paths."""

from __future__ import annotations

import math

import numpy as np

from .config import StageAConfig
from .models import CrossTrackPair


def segment_footprint_clearance(
    first: np.ndarray,
    second: np.ndarray,
    cone_xy: np.ndarray,
    config: StageAConfig,
) -> float:
    """Return clearance from the car swept along one straight path segment."""

    first_x, first_y = float(first[0]), float(first[1])
    delta_x = float(second[0]) - first_x
    delta_y = float(second[1]) - first_y
    length = math.hypot(delta_x, delta_y)
    if length > 1e-9:
        forward_x, forward_y = delta_x / length, delta_y / length
    else:
        forward_x, forward_y = 1.0, 0.0
    lateral_x, lateral_y = -forward_y, forward_x
    half_length = 0.5 * config.vehicle_length_m
    half_width = 0.5 * config.vehicle_width_m
    minimum_distance = math.inf
    for cone in cone_xy:
        relative_x = float(cone[0]) - first_x
        relative_y = float(cone[1]) - first_y
        along = relative_x * forward_x + relative_y * forward_y
        lateral = relative_x * lateral_x + relative_y * lateral_y
        excess_along = max(
            -half_length - along,
            along - (length + half_length),
            0.0,
        )
        excess_lateral = max(abs(lateral) - half_width, 0.0)
        distance = math.hypot(excess_along, excess_lateral)
        minimum_distance = min(minimum_distance, distance)
    return minimum_distance - config.cone_radius_m - config.collision_margin_m


def path_footprint_clearance(
    path_xy: np.ndarray,
    cone_xy: np.ndarray,
    config: StageAConfig,
) -> float:
    """Check the car body, not just its centerline, along a sampled path."""

    origin = np.zeros(2, dtype=np.float64)
    initial_clearance = segment_footprint_clearance(origin, path_xy[0], cone_xy, config)
    if initial_clearance < 0.0:
        return initial_clearance
    outer_radius = math.hypot(
        0.5 * config.vehicle_length_m,
        0.5 * config.vehicle_width_m,
    ) + config.cone_radius_m + config.collision_margin_m
    point_clearances = np.linalg.norm(
        path_xy[:, None, :] - cone_xy[None, :, :], axis=2
    )
    minimum_point_clearance = float(point_clearances.min())
    interpolation_allowance = 0.5 * config.spline_sample_spacing_m
    if minimum_point_clearance >= outer_radius + interpolation_allowance:
        return min(
            initial_clearance,
            minimum_point_clearance - outer_radius - interpolation_allowance,
        )
    return min(
        initial_clearance,
        min(
            segment_footprint_clearance(first, second, cone_xy, config)
            for first, second in zip(path_xy[:-1], path_xy[1:])
        ),
    )


def within_cross_sections(
    path_xy: np.ndarray,
    cross_pairs: tuple[CrossTrackPair, ...] | list[CrossTrackPair],
    cone_xy: np.ndarray,
    config: StageAConfig,
    pair_indices: tuple[int, ...] | None = None,
) -> bool:
    """Ensure path samples stay between and near supported track sections."""

    selected_pairs = (
        cross_pairs
        if pair_indices is None
        else [cross_pairs[index] for index in pair_indices]
    )
    if not selected_pairs:
        return False
    local_sections = []
    for first_idx, second_idx, midpoint, _quality, _width in selected_pairs:
        local_sections.append((cone_xy[first_idx], cone_xy[second_idx], midpoint))
    mids = np.asarray([section[2] for section in local_sections])
    endpoints_a = np.asarray([section[0] for section in local_sections])
    segments = np.asarray([section[1] - section[0] for section in local_sections])
    max_support = max(4.0, config.maximum_boundary_pair_distance_m)
    offsets = path_xy[:, None, :] - mids[None, :, :]
    nearest = np.argmin(np.sum(offsets * offsets, axis=2), axis=1)
    nearest_distance = np.sqrt(
        np.sum(offsets[np.arange(len(path_xy)), nearest] ** 2, axis=1)
    )
    if np.any(nearest_distance > max_support):
        return False
    path_offsets = path_xy - endpoints_a[nearest]
    selected_segments = segments[nearest]
    numerator = np.sum(path_offsets * selected_segments, axis=1)
    denominator = np.maximum(np.sum(selected_segments * selected_segments, axis=1), 1e-9)
    projection = numerator / denominator
    return bool(np.all((projection >= -0.12) & (projection <= 1.12)))
