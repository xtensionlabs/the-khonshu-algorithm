from __future__ import annotations
import numpy as np


def arc_length(points) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros(0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def resample(points, spacing: float = 0.5, max_length=None) -> np.ndarray:

    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return p.copy()

    # drop consecutive duplicates (zero-length steps give flat spots in s)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
    p = p[keep]
    if len(p) < 2:
        return p

    s = arc_length(p)
    total = s[-1] if max_length is None else min(s[-1], max_length)
    s_new = np.arange(int(np.floor(total / spacing)) + 1) * spacing
    return np.column_stack((np.interp(s_new, s, p[:, 0]), np.interp(s_new, s, p[:, 1])))


def heading(points, spacing: float) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros(len(p))
    dx = np.gradient(p[:, 0], spacing)
    dy = np.gradient(p[:, 1], spacing)
    return np.unwrap(np.arctan2(dy, dx))


def curvature(points, spacing: float) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return np.zeros(len(p))
    dx = np.gradient(p[:, 0], spacing)
    dy = np.gradient(p[:, 1], spacing)
    ddx = np.gradient(dx, spacing)
    ddy = np.gradient(dy, spacing)
    denom = (dx**2 + dy**2) ** 1.5
    return (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9)


def speed_profile(
    kappa,
    spacing: float,
    v_max: float = 8.0,
    a_lat: float = 6.0,
    a_acc: float = 3.0,
    a_brake: float = 5.0,
    v_start=None,
    v_end=None,
) -> np.ndarray:
    k = np.abs(np.asarray(kappa, dtype=float))
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(k, 1e-6)))
    if len(v) == 0:
        return v
    if v_end is not None:  # e.g. be able to stop at the horizon
        v[-1] = min(v[-1], v_end)
    for i in range(len(v) - 2, -1, -1):  # backward: can we brake in time?
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_brake * spacing))
    if v_start is not None:  # the speed we are doing right now
        v[0] = min(v[0], v_start)
    for i in range(len(v) - 1):  # forward: can we accelerate to it?
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * a_acc * spacing))
    return v
