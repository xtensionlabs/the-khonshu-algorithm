import numpy as np
from planner.fake_tracks import straight_then_turn
from planner.triangulation import blue_yellow_edges, Edges
from planner.centerline import order_by_x, order_greedy, build_centerline


def test_greedy_recovers_full_track_including_u_turn():
    cones, centre = straight_then_turn()  # straight + 180 degree turn
    e = blue_yellow_edges(cones)
    path = build_centerline(e)
    assert path.shape == centre.shape
    assert np.allclose(path, centre, atol=1e-6)


def test_sort_by_x_breaks_on_u_turn():
    # This documents WHY the baseline is not enough.
    cones, centre = straight_then_turn()
    e = blue_yellow_edges(cones)
    assert not np.allclose(order_by_x(e.midpoints), centre)


def test_path_starts_at_the_car():
    cones, _ = straight_then_turn()
    path = build_centerline(blue_yellow_edges(cones), start=(0.0, 0.0))
    assert np.allclose(path[0], [0.0, 0.0])


def test_never_hops_backwards():
    mids = [[3.0, 0.0], [6.0, 0.0], [-3.0, 0.0]]  # last one is behind the car
    out = order_greedy(mids)
    assert np.allclose(out, [[3.0, 0.0], [6.0, 0.0]])


def test_stops_at_a_gap_larger_than_max_hop():
    out = order_greedy([[3.0, 0.0], [20.0, 0.0]], max_hop=6.0)
    assert np.allclose(out, [[3.0, 0.0]])


def test_duplicate_points_are_merged():
    out = order_greedy([[3.0, 0.0], [3.05, 0.02], [6.0, 0.0]])
    assert len(out) == 2


def test_heading_follows_a_turn():
    # points on a quarter circle of radius 10 starting at the origin, turning left
    a = np.linspace(0.3, np.pi / 2, 5)
    mids = np.column_stack((10 * np.sin(a), 10 * (1 - np.cos(a))))
    out = order_greedy(mids, max_hop=6.0)
    assert np.allclose(out, mids)


def test_empty_input_returns_just_the_car():
    e = Edges(np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)), np.zeros(0))
    path = build_centerline(e)
    assert path.shape == (1, 2)


def test_noisy_track_still_gives_a_forward_path():
    cones, _ = straight_then_turn(noise=0.2)
    path = build_centerline(blue_yellow_edges(cones))
    assert len(path) > 8
    assert path[-1, 1] > 10  # made it into the turn
