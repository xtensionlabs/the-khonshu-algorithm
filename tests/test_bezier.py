import numpy as np
from planner.bezier import (
    bezier_point,
    bezier_derivative,
    segment_controls,
    smooth_path,
    spline_smooth,
)


def test_endpoints_of_one_segment():
    ctrl = np.array([[0, 0], [1, 2], [3, 2], [4, 0]], dtype=float)
    out = bezier_point(ctrl, [0.0, 1.0])
    assert np.allclose(out[0], ctrl[0]) and np.allclose(out[1], ctrl[3])


def test_midpoint_matches_hand_calculation():
    # B(0.5) = (P0 + 3 P1 + 3 P2 + P3) / 8
    ctrl = np.array([[0, 0], [1, 2], [3, 2], [4, 0]], dtype=float)
    expected = (ctrl[0] + 3 * ctrl[1] + 3 * ctrl[2] + ctrl[3]) / 8
    assert np.allclose(bezier_point(ctrl, [0.5])[0], expected)


def test_end_derivatives_point_at_inner_controls():
    ctrl = np.array([[0, 0], [1, 2], [3, 2], [4, 0]], dtype=float)
    assert np.allclose(bezier_derivative(ctrl, [0.0])[0], 3 * (ctrl[1] - ctrl[0]))
    assert np.allclose(bezier_derivative(ctrl, [1.0])[0], 3 * (ctrl[3] - ctrl[2]))


def test_curve_passes_through_every_input_point():
    pts = np.array([[0, 0], [3, 0.5], [6, 2], [8, 5], [8.5, 9]], dtype=float)
    out = smooth_path(pts, per_segment=10)
    assert np.allclose(out[::10], pts)


def test_output_length():
    pts = np.random.default_rng(0).uniform(0, 10, (7, 2))
    assert len(smooth_path(pts, per_segment=15)) == (7 - 1) * 15 + 1


def test_straight_line_stays_straight():
    pts = np.column_stack((np.arange(0, 15, 3.0), np.zeros(5)))
    out = smooth_path(pts)
    assert np.allclose(out[:, 1], 0.0)


def test_uneven_spacing_does_not_loop_back():
    pts = np.array([[0, 0], [0.5, 0], [6, 0], [6.5, 0], [12, 0]], dtype=float)
    out = smooth_path(pts)
    assert np.all(np.diff(out[:, 0]) >= -1e-9)  # x never decreases


def test_tangent_direction_continuous_at_joins():
    pts = np.array([[0, 0], [3, 1], [6, 3], [8, 6], [9, 10]], dtype=float)
    ctrl = segment_controls(pts)
    for i in range(len(ctrl) - 1):
        a = bezier_derivative(ctrl[i], [1.0])[0]
        b = bezier_derivative(ctrl[i + 1], [0.0])[0]
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cos > 0.999999


def test_approximates_a_circle():
    ang = np.arange(0.0, 3.0, 0.3)
    pts = 10 * np.column_stack((np.cos(ang), np.sin(ang)))
    out = smooth_path(pts, per_segment=20)
    inner = out[20:-20]  # skip end segments (one-sided tangent)
    radius = np.linalg.norm(inner, axis=1)
    assert np.max(np.abs(radius - 10)) < 0.02


def test_start_heading_is_respected():
    pts = np.array([[0, 0], [3, 3], [6, 3]], dtype=float)
    out = smooth_path(pts, per_segment=50, start_heading=(1.0, 0.0))
    first_step = out[1] - out[0]
    assert abs(first_step[1]) < 0.2 * abs(first_step[0])  # leaves along +x


def test_degenerate_inputs_do_not_crash_or_nan():
    assert smooth_path(np.zeros((0, 2))).shape == (0, 2)
    assert smooth_path([[1.0, 2.0]]).shape == (1, 2)
    dup = smooth_path([[0, 0], [0, 0], [3, 0]])
    assert np.all(np.isfinite(dup))


def test_spline_reduces_noise_but_bezier_does_not():
    rng = np.random.default_rng(3)
    ang = np.linspace(0.0, 3.0, 25)
    true = 10 * np.column_stack((np.cos(ang), np.sin(ang)))
    noisy = true + rng.normal(0, 0.15, true.shape)

    def rms(p):
        return np.sqrt(np.mean((np.linalg.norm(p, axis=1) - 10) ** 2))

    spl = spline_smooth(noisy, smoothing=len(noisy) * 0.15**2)
    assert rms(spl) < rms(noisy)  # spline averages the noise
    bez = smooth_path(noisy, per_segment=10)
    assert rms(bez) > 0.8 * rms(noisy)  # Bezier keeps it
