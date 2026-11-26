"""Stage 1: Delaunay triangulation and heading-aware centreline extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial import Delaunay
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from khonshu_planner.core.types import Cone, ConeColor, TrackEstimate


@dataclass(frozen=True)
class DelaunayConfig:
    min_edge_length: float = 2.0
    max_edge_length: float = 6.5
    max_step_dist: float = 7.5
    max_turn_angle_rad: float = math.radians(75.0)
    first_point_max_angle_rad: float = math.radians(65.0)
    angle_weight: float = 2.5
    min_point_spacing: float = 0.6
    orange_lateral_limit: float = 6.0


def _wrap_angle(angle: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _cone_side(cone: Cone, config: DelaunayConfig) -> str | None:
    """Classify a cone as Left ('L') or Right ('R') in car-local base_footprint."""
    if cone.color == ConeColor.BLUE:
        return "L"
    if cone.color == ConeColor.YELLOW:
        return "R"
    if cone.color in (ConeColor.ORANGE, ConeColor.BIG_ORANGE):
        if cone.x > -2.0 and abs(cone.y) <= config.orange_lateral_limit:
            return "L" if cone.y >= 0.0 else "R"
    return None


def _order_midpoints(
    raw_midpoints: list[tuple[float, float]],
    config: DelaunayConfig,
) -> list[tuple[float, float]]:
    """Order midpoints forward from the car at (0, 0) facing +x (heading = 0)."""
    if not raw_midpoints:
        return []

    used = [False] * len(raw_midpoints)
    ordered: list[tuple[float, float]] = []
    cur_x, cur_y = 0.0, 0.0
    heading = 0.0

    while True:
        best_idx = -1
        best_cost = float("inf")
        is_first = len(ordered) == 0
        max_angle = (
            config.first_point_max_angle_rad
            if is_first
            else config.max_turn_angle_rad
        )

        for idx, (mx, my) in enumerate(raw_midpoints):
            if used[idx]:
                continue
            if is_first and mx <= 0.0:
                continue

            dx = mx - cur_x
            dy = my - cur_y
            dist = math.hypot(dx, dy)
            if dist < config.min_point_spacing or dist > config.max_step_dist:
                continue

            step_heading = math.atan2(dy, dx)
            dtheta = abs(_wrap_angle(step_heading - heading))
            if dtheta > max_angle:
                continue

            cost = dist + config.angle_weight * dtheta
            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        if best_idx == -1:
            break

        bx, by = raw_midpoints[best_idx]
        heading = math.atan2(by - cur_y, bx - cur_x)
        cur_x, cur_y = bx, by
        ordered.append((bx, by))
        used[best_idx] = True

        # Suppress clustered midpoints from adjacent diagonal Delaunay edges
        for idx, (mx, my) in enumerate(raw_midpoints):
            if not used[idx] and math.hypot(mx - bx, my - by) < config.min_point_spacing:
                used[idx] = True

    return ordered


def estimate_track(
    cones: Sequence[Cone],
    config: DelaunayConfig = DelaunayConfig(),
) -> TrackEstimate:
    """Run Delaunay triangulation and extract an ordered centreline from cones."""
    pts_list: list[tuple[float, float]] = []
    sides: list[str] = []

    for cone in cones:
        side = _cone_side(cone, config)
        if side is not None:
            pts_list.append((cone.x, cone.y))
            sides.append(side)

    if len(pts_list) < 3:
        return TrackEstimate(midpoints=(), cross_edges=(), triangles=())

    pts = np.asarray(pts_list, dtype=float)
    try:
        tri = Delaunay(pts)
    except QhullError:
        return TrackEstimate(midpoints=(), cross_edges=(), triangles=())

    triangles: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    cross_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    raw_midpoints: list[tuple[float, float]] = []
    seen_edges: set[tuple[int, int]] = set()

    for a, b, c in tri.simplices:
        pa = (float(pts[a, 0]), float(pts[a, 1]))
        pb = (float(pts[b, 0]), float(pts[b, 1]))
        pc = (float(pts[c, 0]), float(pts[c, 1]))
        triangles.append((pa, pb, pc))

        for i, j in ((a, b), (b, c), (c, a)):
            edge_key = (int(min(i, j)), int(max(i, j)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            if sides[i] == sides[j]:
                continue

            length = math.hypot(
                float(pts[j, 0] - pts[i, 0]),
                float(pts[j, 1] - pts[i, 1]),
            )
            if config.min_edge_length <= length <= config.max_edge_length:
                p1 = (float(pts[i, 0]), float(pts[i, 1]))
                p2 = (float(pts[j, 0]), float(pts[j, 1]))
                cross_edges.append((p1, p2))
                raw_midpoints.append(((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5))

    ordered = _order_midpoints(raw_midpoints, config)
    return TrackEstimate(
        midpoints=tuple(ordered),
        cross_edges=tuple(cross_edges),
        triangles=tuple(triangles),
    )
