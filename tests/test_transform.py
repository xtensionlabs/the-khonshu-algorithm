import numpy as np
from planner.cones import Cones
from planner.transforms import (
    map_to_car,
    car_to_map,
    yaw_from_quaternion,
    cones_to_car,
    filter_ahead,
)


def test_point_ahead_of_rotated_car():
    # Car at origin facing +y (90 deg). A cone at map (0, 5) is dead ahead.
    out = map_to_car([[0.0, 5.0]], (0.0, 0.0), np.pi / 2)
    assert np.allclose(out, [[5.0, 0.0]])


def test_point_to_the_left():
    # Car facing +x. Cone at map (0, 2) is 2 m to its left -> car y = +2.
    out = map_to_car([[0.0, 2.0]], (0.0, 0.0), 0.0)
    assert np.allclose(out, [[0.0, 2.0]])


def test_translation_only():
    out = map_to_car([[10.0, 5.0]], (7.0, 5.0), 0.0)
    assert np.allclose(out, [[3.0, 0.0]])


def test_round_trip():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-50, 50, (20, 2))
    back = car_to_map(map_to_car(pts, (3.0, -4.0), 1.2), (3.0, -4.0), 1.2)
    assert np.allclose(back, pts)


def test_car_position_maps_to_origin():
    out = map_to_car([[3.0, -4.0]], (3.0, -4.0), 2.0)
    assert np.allclose(out, [[0.0, 0.0]])


def test_distances_preserved():
    a, b = np.array([[1.0, 2.0]]), np.array([[4.0, 6.0]])
    ta = map_to_car(a, (5.0, 5.0), 0.7)
    tb = map_to_car(b, (5.0, 5.0), 0.7)
    assert np.isclose(np.linalg.norm(ta - tb), 5.0)


def test_yaw_from_quaternion():
    # 90 deg about z: z = sin(45 deg), w = cos(45 deg)
    s = np.sin(np.pi / 4)
    assert np.isclose(yaw_from_quaternion(0, 0, s, s), np.pi / 2)
    assert np.isclose(yaw_from_quaternion(0, 0, 0, 1), 0.0)


def test_filter_ahead():
    c = Cones([[5.0, 1.0], [-10.0, 0.0], [30.0, 0.0]], [[5.0, -1.0]])
    out = filter_ahead(c)
    assert len(out.blue) == 1 and len(out.yellow) == 1


def test_empty_cones_do_not_crash():
    out = filter_ahead(cones_to_car(Cones([], []), (0, 0), 0.0))
    assert out.blue.shape == (0, 2)
