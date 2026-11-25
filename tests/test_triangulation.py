import numpy as np
from planner.cones import Cones
from planner.fake_tracks import straight_then_turn
from planner.triangulation import blue_yellow_edges


def test_clean_track_gives_one_edge_per_cone_pair():
    cones, centre = straight_then_turn()
    e = blue_yellow_edges(cones)
    assert len(e.midpoints) == len(centre)


def test_widths_equal_track_width_on_clean_track():
    cones, _ = straight_then_turn(half_width=1.5)
    e = blue_yellow_edges(cones)
    assert np.allclose(e.widths, 3.0, atol=0.01)


def test_midpoints_sit_on_true_centreline():
    cones, centre = straight_then_turn()
    e = blue_yellow_edges(cones)
    for m in e.midpoints:
        assert np.min(np.linalg.norm(centre - m, axis=1)) < 1e-6


def test_indices_are_valid_and_pair_up():
    cones, _ = straight_then_turn()
    e = blue_yellow_edges(cones)
    assert e.blue_idx.max() < len(cones.blue)
    assert e.yellow_idx.max() < len(cones.yellow)
    mids = (cones.blue[e.blue_idx] + cones.yellow[e.yellow_idx]) / 2
    assert np.allclose(mids, e.midpoints)


def test_larger_max_edge_lets_diagonals_in():
    cones, _ = straight_then_turn()
    tight = blue_yellow_edges(cones, max_edge=4.0)
    loose = blue_yellow_edges(cones, max_edge=5.0)
    assert len(loose.midpoints) > len(tight.midpoints)


def test_noisy_track_still_produces_edges():
    cones, _ = straight_then_turn(noise=0.2)
    assert len(blue_yellow_edges(cones).midpoints) > 5


def test_no_cones_or_one_colour_does_not_crash():
    assert len(blue_yellow_edges(Cones([], [])).midpoints) == 0
    assert len(blue_yellow_edges(Cones([[1, 1], [2, 1], [3, 1]], [])).midpoints) == 0


def test_collinear_cones_do_not_crash():
    c = Cones([[0, 1], [3, 1], [6, 1]], [[0, -1], [3, -1]])  # not collinear overall
    blue_yellow_edges(c)
    c2 = Cones([[0, 0], [1, 0]], [[2, 0], [3, 0]])  # all on one line
    assert len(blue_yellow_edges(c2).midpoints) == 0
