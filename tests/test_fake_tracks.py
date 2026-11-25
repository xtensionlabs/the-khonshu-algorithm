import numpy as np
from planner.fake_tracks import straight_then_turn


def test_cone_pairs_are_track_width_apart():
    cones, _ = straight_then_turn(half_width=1.5)
    d = np.linalg.norm(cones.blue - cones.yellow, axis=1)
    assert np.allclose(d, 3.0)


def test_drop_removes_cones():
    clean, _ = straight_then_turn()

    dropped, _ = straight_then_turn(drop=3)
    total = lambda c: len(c.blue) + len(c.yellow)
    assert total(dropped) == total(clean) - 3
