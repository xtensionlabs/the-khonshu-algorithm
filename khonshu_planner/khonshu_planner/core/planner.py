"""Top-level pure-Python Khonshu path planner combining Stage 1, Stage 2, and failsafe."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Sequence

from khonshu_planner.core.delaunay import DelaunayConfig, estimate_track
from khonshu_planner.core.smoothing import SmoothingConfig, smooth_and_profile
from khonshu_planner.core.types import Cone, TrackEstimate, Trajectory, Waypoint


@dataclass(frozen=True)
class PlannerConfig:
    delaunay: DelaunayConfig = field(default_factory=DelaunayConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    min_waypoints: int = 2


@dataclass(frozen=True)
class PlanResult:
    trajectory: Trajectory
    track_estimate: TrackEstimate


class KhonshuPlanner:
    """ROS-independent v0.1 path planner."""

    def __init__(self, config: PlannerConfig = PlannerConfig()) -> None:
        self.config = config

    def plan(
        self,
        cones: Sequence[Cone],
        current_speed: float = 0.0,
        frame_id: str = "base_footprint",
    ) -> PlanResult:
        t0 = time.perf_counter()
        estimate = estimate_track(cones, self.config.delaunay)
        waypoints = smooth_and_profile(
            estimate.midpoints,
            current_speed=current_speed,
            config=self.config.smoothing,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if len(waypoints) < self.config.min_waypoints:
            failsafe = Trajectory(
                waypoints=(Waypoint(0.0, 0.0, 0.0, 0.0, 0.0),),
                frame_id=frame_id,
                is_failsafe=True,
                compute_time_ms=dt_ms,
            )
            return PlanResult(trajectory=failsafe, track_estimate=estimate)

        trajectory = Trajectory(
            waypoints=waypoints,
            frame_id=frame_id,
            is_failsafe=False,
            compute_time_ms=dt_ms,
        )
        return PlanResult(trajectory=trajectory, track_estimate=estimate)
