import math

import numpy as np
import pytest

from khonshu import Cone, ConeColor, StageAConfig, StageAPlanner, VehicleState
from khonshu.scenarios import synthetic_scenarios


@pytest.mark.parametrize(
    "scenario_name",
    [
        "straight",
        "left_turn",
        "right_turn",
        "missing_inside_boundary",
        "unlabeled_cones",
        "false_positives",
    ],
)
def test_stage_a_generates_safe_centerline_from_delaunay_voronoi_graph(scenario_name):
    planner = StageAPlanner(StageAConfig(maximum_cycle_time_ms=1000.0))
    cones, state = synthetic_scenarios()[scenario_name]

    trajectory = planner.plan(cones, state)

    assert trajectory.valid, trajectory.reason
    assert not trajectory.emergency_stop
    assert len(trajectory.points) >= 2
    assert all(
        a.s_m < b.s_m for a, b in zip(trajectory.points, trajectory.points[1:])
    )
    assert planner.last_diagnostics["delaunay_edges"] > 0
    assert planner.last_diagnostics["cross_track_pairs"] > 0
    assert planner.last_diagnostics["voronoi_vertices"] > 0
    assert planner.last_diagnostics["graph_edges"] > 0
    assert planner.last_diagnostics["selected_nodes"] >= 2

    path_xy = np.asarray([(point.x, point.y) for point in trajectory.points])
    cone_xy = np.asarray([(cone.x, cone.y) for cone in cones])
    assert np.linalg.norm(path_xy[:, None, :] - cone_xy[None, :, :], axis=2).min() >= 0.8


def test_stage_a_transforms_vehicle_local_search_back_into_world_frame():
    local_cones, _ = synthetic_scenarios()["straight"]
    yaw = math.pi / 2.0
    offset_x, offset_y = 10.0, -4.0
    world_cones = tuple(
        Cone(
            offset_x - cone.y,
            offset_y + cone.x,
            cone.color,
        )
        for cone in local_cones
    )
    state = VehicleState(offset_x, offset_y, yaw, 0.0, 2.0, frame_id="map")
    planner = StageAPlanner(StageAConfig(maximum_cycle_time_ms=1000.0))

    trajectory = planner.plan(world_cones, state)

    assert trajectory.valid, trajectory.reason
    assert min(point.x for point in trajectory.points) >= offset_x - 1.0
    assert max(point.y for point in trajectory.points) > offset_y + 3.0


def test_stage_a_clears_displaced_inner_corner_cones_with_vehicle_footprint():
    cones, state = synthetic_scenarios()["left_turn"]
    cones = list(cones)
    radius_m = 20.0
    for distance_m in (9.0, 15.0, 21.0, 27.0):
        angle = distance_m / radius_m
        center_x = radius_m * math.sin(angle)
        center_y = radius_m * (1.0 - math.cos(angle))
        # Put an orange apex cone inside the curve, displaced from either
        # boundary row like the triangular corner markers in EUFS.
        cones.append(
            Cone(
                center_x - 0.9 * math.sin(angle),
                center_y + 0.9 * math.cos(angle),
                ConeColor.ORANGE_BIG,
            )
        )
    planner = StageAPlanner(StageAConfig(maximum_cycle_time_ms=1000.0))

    trajectory = planner.plan(cones, state)

    assert trajectory.valid, trajectory.reason
    assert planner.last_diagnostics["minimum_body_clearance_m"] >= 0.0


def test_orange_apex_cones_are_obstacles_not_centerline_boundaries():
    cones, state = synthetic_scenarios()["left_turn"]
    apex_cones = (
        Cone(7.0, 2.2, ConeColor.ORANGE_BIG),
        Cone(14.0, 4.8, ConeColor.ORANGE_SMALL),
    )
    planner = StageAPlanner(StageAConfig(maximum_cycle_time_ms=1000.0))

    trajectory = planner.plan((*cones, *apex_cones), state)

    assert trajectory.valid, trajectory.reason
    assert planner.last_diagnostics["orange_obstacles"] == len(apex_cones)
    sections = planner.last_debug_geometry["cross_sections_local"]
    boundary_xy = np.asarray([(cone.x, cone.y) for cone in cones])
    for first, second, _midpoint in sections:
        assert np.min(np.linalg.norm(boundary_xy - np.asarray(first), axis=1)) < 1e-6
        assert np.min(np.linalg.norm(boundary_xy - np.asarray(second), axis=1)) < 1e-6


def test_stage_a_deadline_overrun_requests_an_empty_stop_trajectory():
    cones, state = synthetic_scenarios()["straight"]
    planner = StageAPlanner(StageAConfig(maximum_cycle_time_ms=1e-12))

    trajectory = planner.plan(cones, state)

    assert trajectory.emergency_stop
    assert not trajectory.valid
    assert trajectory.points == ()
    assert trajectory.reason == "planning_deadline_exceeded"


def test_stage_a_rejects_insufficient_evidence_without_running_geometry():
    planner = StageAPlanner()
    cones = (Cone(2.0, 1.0, ConeColor.BLUE), Cone(2.0, -1.0, ConeColor.YELLOW))
    state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)

    trajectory = planner.plan(cones, state)

    assert trajectory.emergency_stop
    assert trajectory.reason == "insufficient_cones"
    assert planner.last_diagnostics["delaunay_edges"] == 0
