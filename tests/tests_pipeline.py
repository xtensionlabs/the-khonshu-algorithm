import numpy as np
from planner.cones import Cones
from planner.fake_tracks import (
    straight_then_turn,
    centre_pose,
    dense_centreline,
    dist_to_curve,
)
from planner.pipeline import plan, PlannerConfig, OK, NO_CONES, TOO_FEW_EDGES
from planner.transforms import rotation

DENSE = dense_centreline()


def test_clean_track_from_several_poses():
    cones, _ = straight_then_turn()
    for s in (0.0, 6.0, 12.0, 18.0, 24.0, 30.0):
        x, y, yaw = centre_pose(s)
        r = plan(cones, (x, y), yaw)
        assert r.status == OK, f"s={s}: {r.status}"
        assert len(r.path) >= 10
        assert dist_to_curve(r.path_map, DENSE).max() < 0.3, f"s={s}"


def test_path_starts_at_the_car_and_points_forward():
    cones, _ = straight_then_turn()
    x, y, yaw = centre_pose(24.0)
    r = plan(cones, (x, y), yaw)
    assert np.linalg.norm(r.path[0]) < 0.3
    assert r.path[-1, 0] > 5.0  # car frame: far end is ahead


def test_output_arrays_have_matching_lengths():
    cones, _ = straight_then_turn()
    r = plan(cones, (0.0, 0.0), 0.0)
    n = len(r.path)
    assert r.path_map.shape == (n, 2)
    assert len(r.heading) == len(r.curvature) == len(r.speed) == n


def test_curvature_matches_the_bend():
    cones, _ = straight_then_turn()
    x, y, yaw = centre_pose(24.0)  # next 15 m are all on the arc
    r = plan(cones, (x, y), yaw)
    assert abs(np.median(r.curvature) - 0.1) < 0.04


def test_speed_is_lower_in_the_bend_than_on_the_straight():
    cones, _ = straight_then_turn()
    straight = plan(cones, (0.0, 0.0), 0.0)
    x, y, yaw = centre_pose(24.0)
    bend = plan(cones, (x, y), yaw)
    assert bend.speed.min() < straight.speed.max()


def test_rigid_motion_of_the_world_gives_the_same_car_frame_path():
    cones, _ = straight_then_turn()
    x, y, yaw = centre_pose(24.0)
    base = plan(cones, (x, y), yaw)

    phi, shift = 0.8, np.array([50.0, -20.0])
    R = rotation(phi)
    moved = Cones(cones.blue @ R.T + shift, cones.yellow @ R.T + shift)
    r = plan(moved, R @ np.array([x, y]) + shift, yaw + phi)
    assert r.status == OK and r.path.shape == base.path.shape
    assert np.allclose(r.path, base.path, atol=1e-5)


def test_no_cones():
    r = plan(Cones([], []))
    assert r.status == NO_CONES and r.path.shape == (0, 2)


def test_one_colour_only():
    cones, _ = straight_then_turn()
    r = plan(Cones(cones.blue, []))
    assert r.status == TOO_FEW_EDGES and len(r.path) == 0


def test_cones_all_behind_the_car():
    cones, _ = straight_then_turn()
    x, y, yaw = centre_pose(30.0)
    r = plan(cones, (x, y), yaw + np.pi)  # facing the wrong way
    assert r.status != OK  # must fail cleanly, not crash


def test_speed_start_is_bounded_by_current_speed():
    cones, _ = straight_then_turn()
    r = plan(cones, (0.0, 0.0), 0.0, v_now=2.0)
    assert r.speed[0] <= 2.0 + 1e-9


def test_config_changes_take_effect():
    cones, _ = straight_then_turn()
    r = plan(cones, (0.0, 0.0), 0.0, cfg=PlannerConfig(spacing=1.0, horizon=10.0))
    assert np.allclose(np.linalg.norm(np.diff(r.path, axis=0), axis=1), 1.0, atol=0.02)
    assert r.path[-1, 0] <= 10.0 + 1e-6
