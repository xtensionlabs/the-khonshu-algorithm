"""Stage 2: Spline smoothing, curvature estimation, and velocity profiling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from khonshu_planner.core.types import Waypoint


@dataclass(frozen=True)
class SmoothingConfig:
    sample_spacing: float = 0.5     # Distance between output waypoints (m)
    smooth_passes: int = 2          # Pre-smoothing passes on noisy interior midpoints
    max_speed: float = 5.0          # Conservative Phase 1 cap (m/s)
    min_speed: float = 1.5          # Minimum cruising speed through tight corners (m/s)
    max_lat_accel: float = 4.0      # Lateral grip limit (m/s^2)
    max_accel: float = 2.5          # Longitudinal acceleration limit (m/s^2)
    max_brake: float = 4.0          # Longitudinal braking limit (m/s^2)


def _smooth_anchor_points(
    points: np.ndarray,
    passes: int,
) -> np.ndarray:
    """Apply light [0.25, 0.5, 0.25] smoothing to interior points, keeping endpoints fixed."""
    pts = points.copy()
    if len(pts) < 3 or passes <= 0:
        return pts
    for _ in range(passes):
        updated = pts.copy()
        updated[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
        pts = updated
    return pts


def _deduplicate_chain(points: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Ensure consecutive points have strictly positive distance for CubicSpline."""
    kept = [points[0]]
    for pt in points[1:]:
        if math.hypot(float(pt[0] - kept[-1][0]), float(pt[1] - kept[-1][1])) > eps:
            kept.append(pt)
    return np.asarray(kept, dtype=float)


def _compute_velocity_profile(
    s_vals: np.ndarray,
    curvatures: np.ndarray,
    current_speed: float,
    config: SmoothingConfig,
) -> np.ndarray:
    """Compute feasible speeds via corner limit + forward accel + backward brake passes."""
    n = len(s_vals)
    if n == 0:
        return np.array([], dtype=float)

    # 1. Lateral grip (cornering) speed cap
    speeds = np.empty(n, dtype=float)
    for i in range(n):
        abs_k = abs(float(curvatures[i]))
        if abs_k < 1e-6:
            v_corner = config.max_speed
        else:
            v_corner = min(config.max_speed, math.sqrt(config.max_lat_accel / abs_k))
        speeds[i] = max(config.min_speed, v_corner)

    # 2. Forward acceleration pass from current car speed
    v0 = max(current_speed, config.min_speed)
    speeds[0] = min(speeds[0], v0)
    for i in range(1, n):
        ds = max(float(s_vals[i] - s_vals[i - 1]), 0.0)
        v_reach = math.sqrt(speeds[i - 1] ** 2 + 2.0 * config.max_accel * ds)
        speeds[i] = min(speeds[i], v_reach)

    # 3. Backward braking pass so the car slows down BEFORE tight corners
    for i in range(n - 2, -1, -1):
        ds = max(float(s_vals[i + 1] - s_vals[i]), 0.0)
        v_brake = math.sqrt(speeds[i + 1] ** 2 + 2.0 * config.max_brake * ds)
        speeds[i] = min(speeds[i], v_brake)

    return speeds


def smooth_and_profile(
    midpoints: Sequence[tuple[float, float]],
    current_speed: float = 0.0,
    config: SmoothingConfig = SmoothingConfig(),
) -> tuple[Waypoint, ...]:
    """Fit a smooth trajectory from (0, 0) through midpoints and assign target speeds."""
    if not midpoints:
        return ()

    raw = np.asarray([(0.0, 0.0), *midpoints], dtype=float)
    pts = _deduplicate_chain(_smooth_anchor_points(raw, config.smooth_passes))
    if len(pts) < 2:
        return ()

    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    s_knots = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = float(s_knots[-1])
    if total_len < 1e-3:
        return ()

    num_samples = max(2, int(math.ceil(total_len / config.sample_spacing)) + 1)
    s_eval = np.linspace(0.0, total_len, num_samples)

    if len(pts) == 2:
        # Straight line fallback when only one midpoint is available
        xs = np.interp(s_eval, s_knots, pts[:, 0])
        ys = np.interp(s_eval, s_knots, pts[:, 1])
        heading = math.atan2(float(pts[1, 1] - pts[0, 1]), float(pts[1, 0] - pts[0, 0]))
        headings = np.full(num_samples, heading, dtype=float)
        curvatures = np.zeros(num_samples, dtype=float)
    else:
        cs_x = CubicSpline(s_knots, pts[:, 0], bc_type="natural")
        cs_y = CubicSpline(s_knots, pts[:, 1], bc_type="natural")

        xs = cs_x(s_eval)
        ys = cs_y(s_eval)
        dx = cs_x(s_eval, 1)
        dy = cs_y(s_eval, 1)
        ddx = cs_x(s_eval, 2)
        ddy = cs_y(s_eval, 2)

        headings = np.arctan2(dy, dx)
        denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
        curvatures = (dx * ddy - dy * ddx) / denom

    speeds = _compute_velocity_profile(s_eval, curvatures, current_speed, config)

    return tuple(
        Waypoint(
            x=float(xs[i]),
            y=float(ys[i]),
            heading=float(headings[i]),
            curvature=float(curvatures[i]),
            speed=float(speeds[i]),
        )
        for i in range(num_samples)
    )
