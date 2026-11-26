"""Internal value objects shared by the Stage A planning stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CrossTrackPair = tuple[int, int, np.ndarray, float, float]
Point2D = tuple[float, float]
Segment2D = tuple[Point2D, Point2D]


@dataclass(frozen=True)
class Candidate:
    xy: np.ndarray
    quality_cost: float
    source: str
    pair_index: int = -1
    clearance_m: float = 0.0


@dataclass(frozen=True)
class TopologyResult:
    """Candidates, evidence, and diagnostics produced from one cone batch."""

    cone_xy: np.ndarray
    cross_pairs: tuple[CrossTrackPair, ...]
    candidates: tuple[Candidate, ...]
    diagnostics: dict[str, object]
    debug_geometry: dict[str, list]
    failure_reason: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """Selected Dijkstra route and graph evidence for visualization."""

    route: tuple[int, ...]
    route_cost: float
    edge_count: int
    graph_edges_local: tuple[Segment2D, ...]
