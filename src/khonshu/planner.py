"""Local path-planning adapter around the vendored FaSTTUBe baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .types import Cone, ConeColor, Trajectory, TrajectoryPoint, VehicleState


@dataclass(frozen=True)
class PlannerConfig:
    """Conservative v0.1 settings; distances are meters and speed is m/s."""

    safe_speed_mps: float = 2.0
    maximum_cycle_time_ms: float = 30.0
    minimum_cone_clearance_m: float = 0.75
    minimum_input_cones: int = 3
    minimum_sorted_boundary_cones: int = 2
    maximum_start_distance_m: float = 5.0
    mission: str = "trackdrive"

    def __post_init__(self) -> None:
        if not math.isfinite(self.safe_speed_mps) or self.safe_speed_mps <= 0.0:
            raise ValueError("safe_speed_mps must be finite and greater than zero")
        if (
            not math.isfinite(self.maximum_cycle_time_ms)
            or self.maximum_cycle_time_ms <= 0.0
        ):
            raise ValueError("maximum_cycle_time_ms must be finite and positive")
        if (
            not math.isfinite(self.minimum_cone_clearance_m)
            or self.minimum_cone_clearance_m <= 0.0
        ):
            raise ValueError("minimum_cone_clearance_m must be finite and positive")
        if self.minimum_input_cones < 1:
            raise ValueError("minimum_input_cones must be positive")
        if self.minimum_sorted_boundary_cones < 1:
            raise ValueError("minimum_sorted_boundary_cones must be positive")
        if (
            not math.isfinite(self.maximum_start_distance_m)
            or self.maximum_start_distance_m <= 0.0
        ):
            raise ValueError("maximum_start_distance_m must be finite and positive")
        if self.mission not in {"acceleration", "skidpad", "autocross", "trackdrive"}:
            raise ValueError("mission must be acceleration, skidpad, autocross, or trackdrive")


def cones_by_vendor_type(cones: Sequence[Cone]) -> list[np.ndarray]:
    """Convert typed cones to the five Nx2 arrays expected by FaSTTUBe."""

    rows: dict[ConeColor, list[tuple[float, float]]] = {
        color: [] for color in ConeColor
    }
    for cone in cones:
        rows[cone.color].append((cone.x, cone.y))

    # FaSTTUBe 0.4.3.2 ConeTypes indices: unknown, right/yellow, left/blue,
    # orange-small, orange-big. Keep this conversion free of vendor imports so
    # translation itself cannot trigger package/JIT startup in a live cycle.
    vendor_index = {
        ConeColor.UNKNOWN: 0,
        ConeColor.YELLOW: 1,
        ConeColor.BLUE: 2,
        ConeColor.ORANGE_SMALL: 3,
        ConeColor.ORANGE_BIG: 4,
    }
    arrays = [np.empty((0, 2), dtype=np.float64) for _ in range(5)]
    for color, index in vendor_index.items():
        arrays[index] = np.asarray(rows[color], dtype=np.float64).reshape((-1, 2))
    return arrays


def _make_vendor_backend(mission: str) -> Any:
    """Import upstream lazily so data types remain usable without ROS or SciPy."""

    from fsd_path_planning import MissionTypes, PathPlanner

    return PathPlanner(MissionTypes[mission])


class LocalPlanner:
    """Plan a constant-speed local trajectory from cones and vehicle state.

    ``backend`` is injectable for tests and for comparing planner versions. The
    default backend is a long-lived FaSTTUBe planner instance because upstream
    retains its previous path for continuity and failure recovery.
    """

    def __init__(self, config: PlannerConfig | None = None, backend: Any = None):
        self.config = config or PlannerConfig()
        self._owns_backend = backend is None
        self._backend = backend if backend is not None else _make_vendor_backend(self.config.mission)

    def plan(self, cones: Sequence[Cone], state: VehicleState) -> Trajectory:
        started = perf_counter()

        def stop(reason: str) -> Trajectory:
            return Trajectory(
                frame_id=state.frame_id,
                timestamp_s=state.timestamp_s,
                points=(),
                valid=False,
                emergency_stop=True,
                reason=reason,
                compute_time_ms=(perf_counter() - started) * 1000.0,
            )

        if len(cones) < self.config.minimum_input_cones:
            return stop("insufficient_cones")

        try:
            vendor_cones = cones_by_vendor_type(cones)
            position = np.asarray((state.x, state.y), dtype=np.float64)
            direction = np.asarray(
                (math.cos(state.yaw_rad), math.sin(state.yaw_rad)), dtype=np.float64
            )
            result = self._backend.calculate_path_in_global_frame(
                vendor_cones,
                position,
                direction,
                return_intermediate_results=True,
            )
            path, sorted_left, sorted_right, *_ = result
            path_array = np.asarray(path, dtype=np.float64)
            sorted_left_array = np.asarray(sorted_left)
            sorted_right_array = np.asarray(sorted_right)
        except Exception as error:  # upstream errors must become an explicit safe stop
            return stop(f"planner_error:{type(error).__name__}")

        if (
            sorted_left_array.ndim != 2
            or sorted_right_array.ndim != 2
            or sorted_left_array.shape[1] != 2
            or sorted_right_array.shape[1] != 2
        ):
            return stop("invalid_boundary_shape")
        if max(len(sorted_left_array), len(sorted_right_array)) < self.config.minimum_sorted_boundary_cones:
            return stop("insufficient_boundary_evidence")
        if path_array.ndim != 2 or path_array.shape[1] != 4 or len(path_array) < 2:
            return stop("invalid_path_shape")
        if not np.isfinite(path_array).all():
            return stop("non_finite_path")
        if np.any(np.diff(path_array[:, 0]) <= 0.0):
            return stop("non_monotonic_path_distance")

        observed_cones = np.asarray([(cone.x, cone.y) for cone in cones], dtype=np.float64)
        path_to_cones = path_array[:, None, 1:3] - observed_cones[None, :, :]
        minimum_clearance = float(np.linalg.norm(path_to_cones, axis=2).min())
        if minimum_clearance < self.config.minimum_cone_clearance_m:
            return stop("cone_clearance_violation")

        distances_to_vehicle = np.linalg.norm(path_array[:, 1:3] - position, axis=1)
        if float(distances_to_vehicle.min()) > self.config.maximum_start_distance_m:
            return stop("path_too_far_from_vehicle")

        points = tuple(
            TrajectoryPoint(
                s_m=float(s),
                x=float(x),
                y=float(y),
                curvature_per_m=float(curvature),
                target_speed_mps=self.config.safe_speed_mps,
            )
            for s, x, y, curvature in path_array
        )
        if (perf_counter() - started) * 1000.0 > self.config.maximum_cycle_time_ms:
            return stop("planning_deadline_exceeded")

        return Trajectory(
            frame_id=state.frame_id,
            timestamp_s=state.timestamp_s,
            points=points,
            valid=True,
            emergency_stop=False,
            compute_time_ms=(perf_counter() - started) * 1000.0,
        )

    def warm_up(self) -> None:
        """Compile/cache representative upstream paths before driving.

        Each shared deterministic scenario is run while stationary so branch-
        specific JIT compilation does not land on a live planning cycle. The
        stateful backend is then replaced so synthetic paths cannot be used as
        real planning fallbacks.
        """

        if not self._owns_backend:
            raise RuntimeError("warm_up is only available for the default backend")

        from .scenarios import synthetic_scenarios

        original_config = self.config
        self.config = replace(self.config, maximum_cycle_time_ms=1e9)
        try:
            for scenario_name, (cones, state) in synthetic_scenarios().items():
                outcome = self.plan(cones, state)
                if not outcome.valid:
                    raise RuntimeError(
                        f"planner warm-up failed for {scenario_name}: {outcome.reason}"
                    )
        finally:
            self.config = original_config
            self._backend = _make_vendor_backend(self.config.mission)
