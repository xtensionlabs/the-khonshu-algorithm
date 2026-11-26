"""Cone filtering, Delaunay topology, cross-sections, and Voronoi candidates."""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np
from scipy.spatial import Delaunay, QhullError, Voronoi

from ..types import Cone, ConeColor, VehicleState
from .config import StageAConfig
from .models import Candidate, CrossTrackPair, TopologyResult


def empty_debug_geometry() -> dict[str, list]:
    """Create the stable geometry payload consumed by ROS visualization."""

    return {
        "delaunay_edges_local": [],
        "voronoi_edges_local": [],
        "cross_sections_local": [],
        "center_candidates_local": [],
        "candidate_graph_edges_local": [],
        "selected_route_local": [],
    }


def local_unique_points(
    cones: tuple[Cone, ...], state: VehicleState
) -> tuple[np.ndarray, tuple[Cone, ...]]:
    """Transform map-frame cones into the vehicle frame and remove duplicates."""

    cosine = math.cos(state.yaw_rad)
    sine = math.sin(state.yaw_rad)
    rows: list[tuple[float, float]] = []
    unique: list[Cone] = []
    for cone in cones:
        dx, dy = cone.x - state.x, cone.y - state.y
        point = (cosine * dx + sine * dy, -sine * dx + cosine * dy)
        if any(math.hypot(point[0] - x, point[1] - y) < 0.03 for x, y in rows):
            continue
        rows.append(point)
        unique.append(cone)
    return np.asarray(rows, dtype=np.float64).reshape((-1, 2)), tuple(unique)


def opposite_boundary_colors(first: Cone, second: Cone) -> bool:
    """Return whether a cone pair spans the blue and yellow boundaries."""

    return {first.color, second.color} == {ConeColor.BLUE, ConeColor.YELLOW}


def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Merge nearby candidates deterministically, preferring Voronoi support."""

    kept: list[Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.quality_cost, item.xy[0], item.xy[1], item.source),
    ):
        match = next(
            (
                index
                for index, old in enumerate(kept)
                if math.hypot(
                    float(old.xy[0] - candidate.xy[0]),
                    float(old.xy[1] - candidate.xy[1]),
                )
                < 0.25
            ),
            None,
        )
        if match is None:
            kept.append(candidate)
        elif candidate.source == "voronoi_vertex" and kept[match].source != "voronoi_vertex":
            kept[match] = candidate
    return kept


def build_topology(
    cones: Iterable[Cone],
    state: VehicleState,
    config: StageAConfig,
) -> TopologyResult:
    """Build cross-track sections and supported center candidates from cones."""

    cone_list = tuple(cones)
    diagnostics: dict[str, object] = {
        "input_cones": len(cone_list),
        "boundary_cones": 0,
        "orange_obstacles": 0,
        "delaunay_edges": 0,
        "cross_track_pairs": 0,
        "voronoi_vertices": 0,
        "center_candidates": 0,
    }
    geometry = empty_debug_geometry()
    empty_xy = np.empty((0, 2), dtype=np.float64)

    def result(
        *,
        cone_xy: np.ndarray = empty_xy,
        cross_pairs: list[CrossTrackPair] | tuple[CrossTrackPair, ...] = (),
        candidates: list[Candidate] | tuple[Candidate, ...] = (),
        failure_reason: str | None = None,
    ) -> TopologyResult:
        return TopologyResult(
            cone_xy=cone_xy,
            cross_pairs=tuple(cross_pairs),
            candidates=tuple(candidates),
            diagnostics=diagnostics,
            debug_geometry=geometry,
            failure_reason=failure_reason,
        )

    if len(cone_list) < 3:
        return result(failure_reason="insufficient_cones")

    local_xy, unique_cones = local_unique_points(cone_list, state)
    if len(local_xy) < 3:
        return result(failure_reason="insufficient_unique_cones")

    orange_colors = {ConeColor.ORANGE_SMALL, ConeColor.ORANGE_BIG}
    original_boundary_indices = [
        index
        for index, cone in enumerate(unique_cones)
        if not (config.orange_cones_are_obstacles and cone.color in orange_colors)
    ]
    diagnostics["boundary_cones"] = len(original_boundary_indices)
    diagnostics["orange_obstacles"] = len(unique_cones) - len(original_boundary_indices)

    # Keep the rolling geometry bounded while allowing nearby behind-car cones
    # to support a cross-section immediately ahead of the vehicle.
    visible = (
        (local_xy[:, 0] >= -config.maximum_start_distance_m)
        & (np.linalg.norm(local_xy, axis=1) <= config.maximum_observation_range_m)
    )
    old_to_new = {
        int(old_index): new_index
        for new_index, old_index in enumerate(np.flatnonzero(visible))
    }
    local_xy = local_xy[visible]
    unique_cones = tuple(cone for cone, keep in zip(unique_cones, visible) if keep)
    boundary_indices = [
        old_to_new[index]
        for index in original_boundary_indices
        if index in old_to_new
    ]
    if len(local_xy) < 3:
        return result(
            cone_xy=local_xy,
            failure_reason="insufficient_forward_cones",
        )

    delaunay_edges: set[tuple[int, int]] = set()
    boundary_xy = local_xy[boundary_indices]
    if len(boundary_xy) >= 3:
        try:
            triangulation = Delaunay(boundary_xy)
            for simplex in triangulation.simplices:
                for i, j in (
                    (simplex[0], simplex[1]),
                    (simplex[1], simplex[2]),
                    (simplex[0], simplex[2]),
                ):
                    global_i, global_j = boundary_indices[int(i)], boundary_indices[int(j)]
                    delaunay_edges.add((min(global_i, global_j), max(global_i, global_j)))
        except QhullError:
            # The bounded-radius pair graph remains available for degenerate
            # Delaunay inputs; no random Qhull joggle is used.
            diagnostics["delaunay_status"] = "degenerate_input"
    else:
        diagnostics["delaunay_status"] = "insufficient_boundary_cones"
    diagnostics["delaunay_edges"] = len(delaunay_edges)
    geometry["delaunay_edges_local"] = [
        (tuple(map(float, local_xy[i])), tuple(map(float, local_xy[j])))
        for i, j in sorted(delaunay_edges)
    ]

    cross_pairs: list[CrossTrackPair] = []
    midpoint_clearances: list[float] = []
    for i, j in itertools.combinations(boundary_indices, 2):
        delta = local_xy[j] - local_xy[i]
        width = math.hypot(float(delta[0]), float(delta[1]))
        if not config.minimum_track_width_m <= width <= config.maximum_track_width_m:
            continue
        if width > config.maximum_boundary_pair_distance_m:
            continue

        color_supported = opposite_boundary_colors(unique_cones[i], unique_cones[j])
        geometric_cross = abs(float(delta[0])) <= min(
            config.maximum_pair_longitudinal_m,
            max(0.8, 0.65 * width),
        )
        if not color_supported and not geometric_cross:
            continue
        if not color_supported and abs(float(delta[1])) < 0.45 * width:
            continue

        midpoint = 0.5 * (local_xy[i] + local_xy[j])
        if midpoint[0] < -0.5:
            continue
        midpoint_clearance = float(np.linalg.norm(local_xy - midpoint, axis=1).min())
        if midpoint_clearance < config.minimum_cone_clearance_m:
            continue
        is_delaunay = (i, j) in delaunay_edges
        width_error = abs(width - config.target_track_width_m) / config.target_track_width_m
        skew_error = abs(float(delta[0])) / width
        color_cost = 0.0 if color_supported else 0.12
        triangulation_cost = 0.0 if is_delaunay else 0.08
        center_clearance_cost = 0.15 * max(
            0.0,
            0.5 * config.target_track_width_m - midpoint_clearance,
        )
        quality = (
            0.18 * width_error
            + 0.2 * skew_error
            + color_cost
            + triangulation_cost
            + center_clearance_cost
        )
        cross_pairs.append((i, j, midpoint, quality, width))
        midpoint_clearances.append(midpoint_clearance)

    diagnostics["cross_track_pairs"] = len(cross_pairs)
    if not cross_pairs:
        return result(
            cone_xy=local_xy,
            failure_reason="no_cross_track_pairs",
        )

    geometry["cross_sections_local"] = [
        (
            tuple(map(float, local_xy[i])),
            tuple(map(float, local_xy[j])),
            tuple(map(float, midpoint)),
        )
        for i, j, midpoint, _quality, _width in cross_pairs
    ]
    candidates = [
        Candidate(
            midpoint,
            quality,
            "delaunay_midpoint" if (i, j) in delaunay_edges else "radius_midpoint",
            pair_idx,
            midpoint_clearances[pair_idx],
        )
        for pair_idx, (i, j, midpoint, quality, _width) in enumerate(cross_pairs)
    ]

    try:
        voronoi = Voronoi(local_xy)
        vor_vertices = voronoi.vertices
        diagnostics["voronoi_vertices"] = len(vor_vertices)
        for ridge in voronoi.ridge_vertices:
            if len(ridge) != 2 or min(ridge) < 0:
                continue
            first, second = vor_vertices[ridge[0]], vor_vertices[ridge[1]]
            if max(np.linalg.norm(first), np.linalg.norm(second)) > config.maximum_observation_range_m:
                continue
            geometry["voronoi_edges_local"].append(
                (tuple(map(float, first)), tuple(map(float, second)))
            )
        pair_midpoints = np.asarray([pair[2] for pair in cross_pairs])
        for vertex in vor_vertices:
            if vertex[0] < -0.5 or np.linalg.norm(vertex) > config.maximum_observation_range_m:
                continue
            distances = np.linalg.norm(pair_midpoints - vertex, axis=1)
            nearest_pair = int(np.argmin(distances))
            support_distance = float(distances[nearest_pair])
            if support_distance > config.maximum_voronoi_support_distance_m:
                continue
            cone_clearance = float(np.linalg.norm(local_xy - vertex, axis=1).min())
            if cone_clearance < config.minimum_cone_clearance_m:
                continue
            _i, _j, _supported_midpoint, pair_quality, _width = cross_pairs[nearest_pair]
            normalized_deviation = support_distance / max(
                0.5 * config.target_track_width_m, 0.5
            )
            center_deviation_cost = (
                config.center_deviation_cost_weight
                * normalized_deviation
                * normalized_deviation
            )
            candidates.append(
                Candidate(
                    np.asarray(vertex, dtype=np.float64),
                    pair_quality + 0.06 + 0.025 * support_distance + center_deviation_cost,
                    "voronoi_vertex",
                    nearest_pair,
                    cone_clearance,
                )
            )
    except QhullError:
        diagnostics["voronoi_status"] = "degenerate_input"

    candidates = deduplicate_candidates(candidates)
    diagnostics["center_candidates"] = len(candidates)
    geometry["center_candidates_local"] = [
        tuple(map(float, candidate.xy)) for candidate in candidates
    ]
    reason = "insufficient_center_candidates" if len(candidates) < 2 else None
    return result(
        cone_xy=local_xy,
        cross_pairs=cross_pairs,
        candidates=candidates,
        failure_reason=reason,
    )
