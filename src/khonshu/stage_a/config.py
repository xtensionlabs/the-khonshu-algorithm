"""Configuration for the Stage A geometry, search, and safety stages."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StageAConfig:
    """Geometry and runtime limits for online Stage A centerline planning."""

    safe_speed_mps: float = 1.0
    maximum_cycle_time_ms: float = 30.0
    minimum_cone_clearance_m: float = 0.8
    vehicle_length_m: float = 2.4
    vehicle_width_m: float = 1.4
    cone_radius_m: float = 0.15
    collision_margin_m: float = 0.25
    orange_cones_are_obstacles: bool = True
    minimum_track_width_m: float = 1.5
    maximum_track_width_m: float = 5.0
    target_track_width_m: float = 3.5
    maximum_boundary_pair_distance_m: float = 5.0
    maximum_pair_longitudinal_m: float = 1.8
    maximum_candidate_gap_m: float = 5.0
    maximum_start_distance_m: float = 5.0
    maximum_observation_range_m: float = 40.0
    maximum_voronoi_support_distance_m: float = 4.0
    maximum_heading_change_rad: float = 1.3
    maximum_initial_heading_error_rad: float = 1.85
    turn_cost_weight: float = 1.25
    center_deviation_cost_weight: float = 2.5
    spline_sample_spacing_m: float = 0.25

    def __post_init__(self) -> None:
        positive = (
            "safe_speed_mps",
            "maximum_cycle_time_ms",
            "minimum_cone_clearance_m",
            "vehicle_length_m",
            "vehicle_width_m",
            "cone_radius_m",
            "minimum_track_width_m",
            "maximum_track_width_m",
            "target_track_width_m",
            "maximum_boundary_pair_distance_m",
            "maximum_pair_longitudinal_m",
            "maximum_candidate_gap_m",
            "maximum_start_distance_m",
            "maximum_observation_range_m",
            "maximum_voronoi_support_distance_m",
            "maximum_heading_change_rad",
            "maximum_initial_heading_error_rad",
            "center_deviation_cost_weight",
            "spline_sample_spacing_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_track_width_m >= self.maximum_track_width_m:
            raise ValueError("minimum_track_width_m must be below maximum_track_width_m")
        if not math.isfinite(self.turn_cost_weight) or self.turn_cost_weight < 0.0:
            raise ValueError("turn_cost_weight must be finite and non-negative")
        if not math.isfinite(self.collision_margin_m) or self.collision_margin_m < 0.0:
            raise ValueError("collision_margin_m must be finite and non-negative")
