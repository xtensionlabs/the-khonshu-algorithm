"""Deterministic 2D scenes for branch-to-branch planner comparisons.

These fixtures exercise software behavior only. They are not simulator or
vehicle safety evidence.
"""

from __future__ import annotations

import bisect
import math
from typing import Dict, Tuple

from .types import Cone, ConeColor, VehicleState

Scenario = Tuple[Tuple[Cone, ...], VehicleState]


def _corridor(
    turn: int = 0,
    missing_inside_boundary: bool = False,
    unlabeled: bool = False,
    false_positives: bool = False,
) -> Scenario:
    cones = []
    radius_m = 20.0

    for distance_m in range(3, 31, 3):
        if turn == 0:
            center_x, center_y = float(distance_m), 0.0
            tangent_x, tangent_y = 1.0, 0.0
        else:
            angle = distance_m / radius_m
            center_x = radius_m * math.sin(angle)
            center_y = turn * radius_m * (1.0 - math.cos(angle))
            tangent_x = math.cos(angle)
            tangent_y = turn * math.sin(angle)

        left_normal_x, left_normal_y = -tangent_y, tangent_x
        left_color = ConeColor.UNKNOWN if unlabeled else ConeColor.BLUE
        right_color = ConeColor.UNKNOWN if unlabeled else ConeColor.YELLOW

        omit_left = missing_inside_boundary and turn > 0 and distance_m in {12, 15, 18, 21}
        if not omit_left:
            cones.append(
                Cone(
                    center_x + 2.0 * left_normal_x,
                    center_y + 2.0 * left_normal_y,
                    left_color,
                )
            )
        cones.append(
            Cone(
                center_x - 2.0 * left_normal_x,
                center_y - 2.0 * left_normal_y,
                right_color,
            )
        )

    if false_positives:
        cones.extend(
            (
                Cone(9.0, 7.0, ConeColor.UNKNOWN),
                Cone(18.0, -7.0, ConeColor.UNKNOWN),
                Cone(24.0, 8.0, ConeColor.UNKNOWN),
            )
        )

    return tuple(cones), VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)


def synthetic_scenarios() -> Dict[str, Scenario]:
    """Return fresh deterministic scenarios in the shared REP-103 map frame."""

    return {
        "straight": _corridor(),
        "left_turn": _corridor(turn=1),
        "right_turn": _corridor(turn=-1),
        "missing_inside_boundary": _corridor(turn=1, missing_inside_boundary=True),
        "unlabeled_cones": _corridor(unlabeled=True),
        "false_positives": _corridor(false_positives=True),
    }


def synthetic_full_circuit() -> tuple[
    tuple[Cone, ...],
    tuple[VehicleState, ...],
    tuple[tuple[float, float], ...],
]:
    """Build a closed dummy circuit with bends, chicanes, and planning poses.

    The course is a planar, smooth loop. Each cone pair marks the two track
    boundaries, while the returned vehicle poses let the local planner cover
    the whole circuit using repeated, realistic forward-looking observations.
    """

    def xy_at(parameter: float) -> tuple[float, float]:
        return (
            28.0 * math.cos(parameter) + 4.0 * math.sin(3.0 * parameter),
            20.0 * math.sin(parameter) + 2.0 * math.sin(2.0 * parameter),
        )

    def tangent_at(parameter: float) -> tuple[float, float]:
        return (
            -28.0 * math.sin(parameter) + 12.0 * math.cos(3.0 * parameter),
            20.0 * math.cos(parameter) + 4.0 * math.cos(2.0 * parameter),
        )

    dense_count = 4096
    dense_parameters = [2.0 * math.pi * index / dense_count for index in range(dense_count + 1)]
    dense_points = [xy_at(parameter) for parameter in dense_parameters]
    cumulative_distance = [0.0]
    for first, second in zip(dense_points, dense_points[1:]):
        cumulative_distance.append(
            cumulative_distance[-1] + math.hypot(second[0] - first[0], second[1] - first[1])
        )
    circuit_length = cumulative_distance[-1]

    def parameter_at_distance(distance: float) -> float:
        wrapped_distance = distance % circuit_length
        index = min(
            bisect.bisect_right(cumulative_distance, wrapped_distance) - 1,
            dense_count - 1,
        )
        segment_start = cumulative_distance[index]
        segment_length = cumulative_distance[index + 1] - segment_start
        fraction = (wrapped_distance - segment_start) / max(segment_length, 1e-12)
        return dense_parameters[index] + fraction * (
            dense_parameters[index + 1] - dense_parameters[index]
        )

    cone_count = max(48, round(circuit_length / 3.2))
    cone_spacing = circuit_length / cone_count
    cones: list[Cone] = []
    for index in range(cone_count):
        parameter = parameter_at_distance(index * cone_spacing)
        x, y = xy_at(parameter)
        tangent_x, tangent_y = tangent_at(parameter)
        tangent_norm = math.hypot(tangent_x, tangent_y)
        normal_x = -tangent_y / tangent_norm
        normal_y = tangent_x / tangent_norm
        cones.extend(
            (
                Cone(x + 2.0 * normal_x, y + 2.0 * normal_y, ConeColor.BLUE),
                Cone(x - 2.0 * normal_x, y - 2.0 * normal_y, ConeColor.YELLOW),
            )
        )

    pose_count = max(48, round(circuit_length / 3.0))
    pose_spacing = circuit_length / pose_count
    poses: list[VehicleState] = []
    reference_centerline: list[tuple[float, float]] = []
    for index in range(pose_count):
        parameter = parameter_at_distance(index * pose_spacing)
        x, y = xy_at(parameter)
        tangent_x, tangent_y = tangent_at(parameter)
        yaw = math.atan2(tangent_y, tangent_x)
        poses.append(VehicleState(x, y, yaw, 0.0, float(index)))
        reference_centerline.append((x, y))

    return tuple(cones), tuple(poses), tuple(reference_centerline)
