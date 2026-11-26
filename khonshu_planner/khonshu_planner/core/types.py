"""ROS-independent data types for the Khonshu path planner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConeColor(Enum):
    BLUE = "blue"             # Left track boundary
    YELLOW = "yellow"         # Right track boundary
    ORANGE = "orange"         # Small orange (start/finish or chicane)
    BIG_ORANGE = "big_orange" # Large orange (start/finish gate)
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Cone:
    x: float
    y: float
    color: ConeColor
    # 2x2 covariance flattened [xx, xy, yx, yy] matching EUFS ConeWithCovariance
    covariance: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float                # Heading in radians
    speed: float              # Forward speed in m/s
    frame_id: str = "map"


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    heading: float = 0.0      # Tangent angle along path (rad)
    curvature: float = 0.0    # kappa = 1 / turn_radius (1/m)
    speed: float = 0.0        # Target speed at this waypoint (m/s)


@dataclass(frozen=True)
class TrackEstimate:
    """Intermediate geometry from Stage 1 for debugging and RViz visualization."""
    midpoints: tuple[tuple[float, float], ...]
    cross_edges: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    triangles: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...]


@dataclass(frozen=True)
class Trajectory:
    waypoints: tuple[Waypoint, ...]
    frame_id: str = "base_footprint"
    is_failsafe: bool = False
    compute_time_ms: float = 0.0
