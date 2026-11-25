from __future__ import annotations
import numpy as np
from .triangulation import Edges


def order_by_x(mids) -> np.ndarray:
    """Baseline: sort by x. Fine on straights, fails on big turns."""
    mids = np.asarray(mids, dtype=float).reshape(-1, 2)
    return mids[np.argsort(mids[:, 0])]


def order_greedy(
    mids,
    start=(0.0, 0.0),
    heading=(1.0, 0.0),
    max_hop: float = 6.0,
    max_turn_deg: float = 90.0,
    min_hop: float = 0.3,
) -> np.ndarray:
    """Hop to the nearest midpoint that is ahead of the current heading.

    Returns the ordered points WITHOUT the start point, shape (K, 2).
    """
    remaining = np.asarray(mids, dtype=float).reshape(-1, 2).copy()
    cur = np.asarray(start, dtype=float)
    d = np.asarray(heading, dtype=float)
    d = d / np.linalg.norm(d)
    cos_limit = np.cos(np.radians(max_turn_deg))

    path = []
    while len(remaining):
        step = remaining - cur  # (R, 2)
        dist = np.linalg.norm(step, axis=1)  # (R,)

        # points sitting on the current position are duplicates: drop them
        dup = dist < min_hop
        if dup.any():
            remaining = remaining[~dup]
            continue

        cos = (step @ d) / dist  # angle to current heading
        ok = (dist <= max_hop) & (cos > cos_limit)
        if not ok.any():
            break  # gap or dead end: stop

        k = int(np.argmin(np.where(ok, dist, np.inf)))
        d = step[k] / dist[k]  # heading follows the path
        cur = remaining[k]
        path.append(cur)
        remaining = np.delete(remaining, k, axis=0)

    return np.array(path).reshape(-1, 2)


def build_centerline(
    edges: Edges, start=(0.0, 0.0), heading=(1.0, 0.0), **kwargs
) -> np.ndarray:
    """Ordered centreline in the car frame, first row = the car position."""
    ordered = order_greedy(edges.midpoints, start=start, heading=heading, **kwargs)
    return np.vstack((np.asarray(start, dtype=float).reshape(1, 2), ordered))
