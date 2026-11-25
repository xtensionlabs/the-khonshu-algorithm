from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay

try:
    from scipy.spatial import QhullError
except ImportError:  # older scipy (e.g. Python 3.8 on WSL)
    from scipy.spatial.qhull import QhullError

from .cones import Cones


@dataclass
class Edges:
    blue_idx: np.ndarray  # (K,) indices into cones.blue
    yellow_idx: np.ndarray  # (K,) indices into cones.yellow
    midpoints: np.ndarray  # (K, 2)
    widths: np.ndarray  # (K,) edge lengths in metres


def _empty_edges() -> Edges:
    return Edges(
        np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros(0)
    )


def triangulate(cones: Cones):
    """Delaunay on all cones. Returns (points, tri) or (points, None).

    points = blue rows first, then yellow rows.
    tri is None if a triangulation is impossible (too few or collinear points).
    """
    pts = np.vstack((cones.blue, cones.yellow))
    if len(cones.blue) < 1 or len(cones.yellow) < 1 or len(pts) < 3:
        return pts, None
    try:
        return pts, Delaunay(pts)
    except QhullError:  # e.g. all points on one line
        return pts, None


def blue_yellow_edges(
    cones: Cones, min_edge: float = 2.0, max_edge: float = 4.0
) -> Edges:
    """Edges of the Delaunay triangulation joining a blue and a yellow cone."""
    pts, tri = triangulate(cones)
    if tri is None:
        return _empty_edges()

    n_blue = len(cones.blue)
    s = tri.simplices
    # every triangle contributes 3 edges; sort each so (a,b) == (b,a); dedupe
    e = np.vstack((s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]))
    e = np.unique(np.sort(e, axis=1), axis=0)
    a, b = e[:, 0], e[:, 1]

    a_is_blue = a < n_blue
    b_is_blue = b < n_blue
    mixed = a_is_blue != b_is_blue
    length = np.linalg.norm(pts[a] - pts[b], axis=1)
    ok = mixed & (length >= min_edge) & (length <= max_edge)

    a, b, length = a[ok], b[ok], length[ok]
    a_is_blue = a < n_blue
    blue_i = np.where(a_is_blue, a, b)
    yellow_i = np.where(a_is_blue, b, a) - n_blue
    mids = (pts[a] + pts[b]) / 2.0
    return Edges(blue_i, yellow_i, mids, length)
