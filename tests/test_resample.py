import numpy as np
from planner.resample import arc_length, resample, heading, curvature, speed_profile


def circle(radius, a0=0.0, a1=np.pi, n=600, clockwise=False):
    a = np.linspace(a0, a1, n)
    sign = -1.0 if clockwise else 1.0
    return radius * np.column_stack((np.sin(a), sign * (1 - np.cos(a))))


def test_arc_length_of_a_3_4_5_triangle_leg():
    assert np.allclose(arc_length([[0, 0], [3, 4]]), [0.0, 5.0])


def test_straight_line_resamples_to_exact_spacing():
    pts = np.column_stack((np.linspace(0, 10, 7), np.zeros(7)))  # uneven input
    out = resample(pts, spacing=0.5)
    assert np.allclose(np.diff(out[:, 0]), 0.5)
    assert len(out) == 21


def test_curve_spacing_is_even():
    out = resample(circle(10.0), spacing=0.5)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.all((d > 0.99 * 0.5) & (d < 0.5 + 1e-9))


def test_starts_at_first_point():
    pts = circle(10.0)
    assert np.allclose(resample(pts)[0], pts[0])


def test_total_length_is_preserved():
    pts = circle(10.0)
    out = resample(pts, spacing=0.25)
    assert abs(arc_length(out)[-1] - np.pi * 10) < 0.3


def test_max_length_cuts_the_horizon():
    pts = np.column_stack((np.linspace(0, 30, 50), np.zeros(50)))
    out = resample(pts, spacing=0.5, max_length=10.0)
    assert abs(out[-1, 0] - 10.0) < 1e-9


def test_duplicates_and_tiny_inputs_do_not_crash():
    out = resample([[0, 0], [0, 0], [3, 0]], spacing=1.0)
    assert np.all(np.isfinite(out))
    assert resample([[1.0, 2.0]]).shape == (1, 2)
    assert resample(np.zeros((0, 2))).shape == (0, 2)


def test_curvature_of_a_straight_is_zero():
    out = resample(np.column_stack((np.linspace(0, 20, 30), np.zeros(30))), 0.5)
    assert np.allclose(curvature(out, 0.5)[2:-2], 0.0, atol=1e-9)


def test_curvature_of_a_circle_is_one_over_radius():
    for r in (5.0, 10.0, 20.0):
        out = resample(circle(r, n=1500), spacing=0.25)
        k = curvature(out, 0.25)[3:-3]
        assert np.allclose(k, 1.0 / r, rtol=0.03)


def test_curvature_sign_left_positive_right_negative():
    left = curvature(resample(circle(10.0), 0.25), 0.25)[3:-3]
    right = curvature(resample(circle(10.0, clockwise=True), 0.25), 0.25)[3:-3]
    assert np.all(left > 0) and np.all(right < 0)


def test_heading_of_a_straight_and_of_a_turn():
    straight = resample(np.column_stack((np.linspace(0, 20, 30), np.zeros(30))), 0.5)
    assert np.allclose(heading(straight, 0.5), 0.0)
    h = heading(resample(circle(10.0), 0.25), 0.25)
    assert abs(h[-1] - np.pi) < 0.1 and np.all(np.diff(h) > -1e-6)


def test_speed_on_a_straight_is_vmax():
    v = speed_profile(np.zeros(20), 0.5, v_max=8.0)
    assert np.allclose(v, 8.0)


def test_speed_drops_in_a_tight_corner():
    k = np.zeros(40)
    k[20:] = 0.5  # radius 2 m corner
    v = speed_profile(k, 0.5, v_max=8.0, a_lat=6.0)
    assert v[0] > v[-1]
    assert v[-1] <= np.sqrt(6.0 / 0.5) + 1e-9


def test_speed_profile_respects_acceleration_limits():
    rng = np.random.default_rng(0)
    k = np.abs(rng.normal(0, 0.15, 60))
    d, a_acc, a_brake = 0.5, 3.0, 5.0
    v = speed_profile(k, d, a_acc=a_acc, a_brake=a_brake)
    assert np.all(v[1:] ** 2 - v[:-1] ** 2 <= 2 * a_acc * d + 1e-9)
    assert np.all(v[:-1] ** 2 - v[1:] ** 2 <= 2 * a_brake * d + 1e-9)
