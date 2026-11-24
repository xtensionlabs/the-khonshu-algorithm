from __future__ import annotations
import numpy as np
from .cones import Cones


def straight_then_turn(
    straight=18.0,
    radius=10.0,
    turn_deg=180.0,
    spacing=3.0,
    half_width=1.5,
    noise=0.0,
    drop=0,
    seed=0,
):
    """Straight, then a left turn. Returns (Cones, true_centreline)."""
    rng = np.random.default_rng(seed)

    samples = [(x, 0.0, 0.0) for x in np.arange(0.0, straight, spacing)]
    arc_len = radius * np.radians(turn_deg)
    for s in np.arange(0.0, arc_len, spacing):
        theta = s / radius
        samples.append(
            (radius * np.sin(theta) + straight, radius * (1 - np.cos(theta)), theta)
        )

    samples = np.array(samples)
    centre, heading = samples[:, :2], samples[:, 2]

    normal = np.column_stack((-np.sin(heading), np.cos(heading)))
    blue = centre + normal * half_width
    yellow = centre - normal * half_width

    blue += rng.normal(0.0, noise, blue.shape)
    yellow += rng.normal(0.0, noise, yellow.shape)

    if drop:
        n = len(centre)
        keep = np.ones(2 * n, dtype=bool)
        keep[rng.choice(2 * n, drop, replace=False)] = False
        allpts = np.vstack((blue, yellow))
        blue = allpts[:n][keep[:n]]
        yellow = allpts[n:][keep[n:]]

    return Cones(blue, yellow), centre
