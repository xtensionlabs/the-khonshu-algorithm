"""Environment-independent planner inputs and outputs.

Coordinates use the ROS REP-103 right-handed convention: x forward, y left,
meters, and yaw counter-clockwise from +x. All positions in one planning call
must share the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple


class ConeColor(str, Enum):
    UNKNOWN = "unknown"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE_SMALL = "orange_small"
    ORANGE_BIG = "orange_big"


@dataclass(frozen=True)
class Cone:
    x: float
    y: float
    color: ConeColor

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("cone coordinates must be finite meters")
        if not isinstance(self.color, ConeColor):
            object.__setattr__(self, "color", ConeColor(self.color))


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw_rad: float
    speed_mps: float
    timestamp_s: float
    frame_id: str = "map"

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.yaw_rad, self.speed_mps, self.timestamp_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vehicle state values must be finite")
        if self.speed_mps < 0.0:
            raise ValueError("vehicle speed must be non-negative")
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")


@dataclass(frozen=True)
class TrajectoryPoint:
    s_m: float
    x: float
    y: float
    curvature_per_m: float
    target_speed_mps: float

    def __post_init__(self) -> None:
        values = (self.s_m, self.x, self.y, self.curvature_per_m, self.target_speed_mps)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("trajectory point values must be finite")
        if self.s_m < 0.0:
            raise ValueError("path distance must be non-negative")
        if self.target_speed_mps < 0.0:
            raise ValueError("target speed must be non-negative")


@dataclass(frozen=True)
class Trajectory:
    frame_id: str
    timestamp_s: float
    points: Tuple[TrajectoryPoint, ...]
    valid: bool
    emergency_stop: bool
    reason: str = ""
    compute_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        if not math.isfinite(self.timestamp_s):
            raise ValueError("trajectory timestamp must be finite")
        if not math.isfinite(self.compute_time_ms) or self.compute_time_ms < 0.0:
            raise ValueError("compute time must be a finite non-negative value")
        if self.valid == self.emergency_stop:
            raise ValueError("valid trajectories run normally; invalid trajectories request a stop")
        if not self.valid and not self.reason:
            raise ValueError("a stop trajectory must include a failure reason")
        if not self.valid and self.points:
            raise ValueError("a failed plan must not contain an unverified path")
