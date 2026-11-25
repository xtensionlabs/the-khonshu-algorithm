from __future__ import annotations
import numpy as np


def bezier_point(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate one cubic Bezier. ctrl is (4, 2), t is (n,). Returns (n, 2)."""
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    p0, p1, p2, p3 = ctrl
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def bezier_derivative(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """First derivative B'(t) of one cubic Bezier. Returns (n, 2)."""
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    p0, p1, p2, p3 = ctrl
    return 3 * (
        (1 - t) ** 2 * (p1 - p0) + 2 * (1 - t) * t * (p2 - p1) + t**2 * (p3 - p2)
    )


def segment_controls(points, start_heading=None) -> np.ndarray:

    p = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(p)

    # unit tangent at every point: previous -> next (one-sided at the ends)
    t = np.empty_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    if start_heading is not None:
        t[0] = np.asarray(start_heading, dtype=float)
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.where(norm > 1e-9, norm, 1.0)  # safe divide (duplicates)

    seg_len = np.linalg.norm(p[1:] - p[:-1], axis=1, keepdims=True)  # (N-1, 1)
    c1 = p[:-1] + t[:-1] * seg_len / 3.0
    c2 = p[1:] - t[1:] * seg_len / 3.0

    return np.stack((p[:-1], c1, c2, p[1:]), axis=1)  # (N-1, 4, 2)


def smooth_path(points, per_segment: int = 20, start_heading=None) -> np.ndarray:

    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return p.copy()

    ctrl = segment_controls(p, start_heading)  # (S, 4, 2)
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)
    out = [bezier_point(c, t) for c in ctrl]  # S arrays of (per_segment, 2)
    out.append(p[-1:].copy())  # close with the last point
    return np.vstack(out)


def spline_smooth(points, smoothing=None, n: int = 100) -> np.ndarray:

    from scipy.interpolate import splprep, splev

    p = np.asarray(points, dtype=float).reshape(-1, 2)
    # splprep fails on consecutive duplicates, so drop them
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
    p = p[keep]
    if len(p) < 4:  # cubic spline needs at least 4 points
        return p
    if smoothing is None:
        smoothing = len(p) * 0.1**2
    tck, _ = splprep([p[:, 0], p[:, 1]], s=smoothing, k=3)
    x, y = splev(np.linspace(0.0, 1.0, n), tck)
    return np.column_stack((x, y))
