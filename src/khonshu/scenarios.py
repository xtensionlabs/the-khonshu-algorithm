"""Deterministic 2D scenes for branch-to-branch planner comparisons.

These fixtures exercise software behavior only. They are not simulator or
vehicle safety evidence.
"""

from __future__ import annotations

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
