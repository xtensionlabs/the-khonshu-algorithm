"""Delaunay/Voronoi candidate generation and cost-based Stage A planning.

The search graph is built from all geometrically plausible cross-track cone
pairs no farther apart than the Formula Student 5 m boundary limit. Delaunay
edges mark well-supported pairs; they do not exclude other pairs because false
positives can remove a correct edge from a triangulation. Finite Voronoi
vertices near supported cross-track pairs add medial-axis samples between
sections. A deterministic Dijkstra search chooses a connected, forward route,
which is then smoothed and checked against observed cones and cross-sections.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay, QhullError, Voronoi, cKDTree

from .types import Cone, ConeColor, Trajectory, TrajectoryPoint, VehicleState


@dataclass(frozen=True)
class StageAConfig:
    """Geometry and runtime limits for online Stage A centerline planning."""

    safe_speed_mps: float = 1.0
    maximum_cycle_time_ms: float = 30.0
    minimum_cone_clearance_m: float = 0.8
    vehicle_length_m: float = 2.4
    vehicle_width_m: float = 1.4
    cone_radius_m: float = 0.15
    collision_margin_m: float = 0.25
    orange_cones_are_obstacles: bool = True
    minimum_track_width_m: float = 1.5
    maximum_track_width_m: float = 5.0
    target_track_width_m: float = 3.5
    maximum_boundary_pair_distance_m: float = 5.0
    maximum_pair_longitudinal_m: float = 1.8
    maximum_candidate_gap_m: float = 5.0
    maximum_start_distance_m: float = 5.0
    maximum_observation_range_m: float = 40.0
    maximum_voronoi_support_distance_m: float = 4.0
    maximum_heading_change_rad: float = 1.3
    maximum_initial_heading_error_rad: float = 1.85
    turn_cost_weight: float = 1.25
    center_deviation_cost_weight: float = 2.5
    spline_sample_spacing_m: float = 0.25

    def __post_init__(self) -> None:
        positive = (
            "safe_speed_mps",
            "maximum_cycle_time_ms",
            "minimum_cone_clearance_m",
            "vehicle_length_m",
            "vehicle_width_m",
            "cone_radius_m",
            "minimum_track_width_m",
            "maximum_track_width_m",
            "target_track_width_m",
            "maximum_boundary_pair_distance_m",
            "maximum_pair_longitudinal_m",
            "maximum_candidate_gap_m",
            "maximum_start_distance_m",
            "maximum_observation_range_m",
            "maximum_voronoi_support_distance_m",
            "maximum_heading_change_rad",
            "maximum_initial_heading_error_rad",
            "center_deviation_cost_weight",
            "spline_sample_spacing_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_track_width_m >= self.maximum_track_width_m:
            raise ValueError("minimum_track_width_m must be below maximum_track_width_m")
        if not math.isfinite(self.turn_cost_weight) or self.turn_cost_weight < 0.0:
            raise ValueError("turn_cost_weight must be finite and non-negative")
        if not math.isfinite(self.collision_margin_m) or self.collision_margin_m < 0.0:
            raise ValueError("collision_margin_m must be finite and non-negative")


@dataclass(frozen=True)
class _Candidate:
    xy: np.ndarray
    quality_cost: float
    source: str
    pair_index: int = -1
    clearance_m: float = 0.0


class StageAPlanner:
    """Plan a local centerline using triangulation, Voronoi support, and Dijkstra.

    Inputs and outputs use the portable ``Cone``/``VehicleState`` types. Cone
    colors are treated as a useful hint, not a hard requirement: uncolored and
    mislabeled cones remain eligible when their geometry supports a cross-track
    pair. Orange cones are treated as interior obstacles by default, so start,
    finish, and displaced apex markers cannot pull the track centerline inward.
    """

    def __init__(self, config: StageAConfig | None = None):
        self.config = config or StageAConfig()
        self.last_diagnostics: dict[str, int | float | str] = {}
        self.last_debug_geometry: dict[str, list] = {}

    def plan(self, cones: Iterable[Cone], state: VehicleState) -> Trajectory:
        started = perf_counter()
        cone_list = tuple(cones)
        diagnostics: dict[str, object] = {
            "algorithm": "delaunay_voronoi_dijkstra",
            "input_cones": len(cone_list),
            "boundary_cones": 0,
            "orange_obstacles": 0,
            "delaunay_edges": 0,
            "cross_track_pairs": 0,
            "voronoi_vertices": 0,
            "center_candidates": 0,
            "graph_edges": 0,
            "selected_nodes": 0,
        }
        self.last_diagnostics = diagnostics
        self.last_debug_geometry = {
            "delaunay_edges_local": [],
            "voronoi_edges_local": [],
            "cross_sections_local": [],
            "center_candidates_local": [],
            "candidate_graph_edges_local": [],
            "selected_route_local": [],
        }

        def stop(reason: str) -> Trajectory:
            diagnostics["result"] = reason
            return Trajectory(
                frame_id=state.frame_id,
                timestamp_s=state.timestamp_s,
                points=(),
                valid=False,
                emergency_stop=True,
                reason=reason,
                compute_time_ms=(perf_counter() - started) * 1000.0,
            )

        if len(cone_list) < 3:
            return stop("insufficient_cones")

        local_xy, unique_cones = self._local_unique_points(cone_list, state)
        if len(local_xy) < 3:
            return stop("insufficient_unique_cones")

        orange_colors = {ConeColor.ORANGE_SMALL, ConeColor.ORANGE_BIG}
        boundary_indices = [
            index
            for index, cone in enumerate(unique_cones)
            if not (self.config.orange_cones_are_obstacles and cone.color in orange_colors)
        ]
        diagnostics["boundary_cones"] = len(boundary_indices)
        diagnostics["orange_obstacles"] = len(unique_cones) - len(boundary_indices)

        # Restrict the online geometry to a local forward window. Behind-car
        # points can still support a first cross-section just ahead of x=0.
        visible = (
            (local_xy[:, 0] >= -self.config.maximum_start_distance_m)
            & (np.linalg.norm(local_xy, axis=1) <= self.config.maximum_observation_range_m)
        )
        local_xy = local_xy[visible]
        unique_cones = tuple(cone for cone, keep in zip(unique_cones, visible) if keep)
        if len(local_xy) < 3:
            return stop("insufficient_forward_cones")

        delaunay_edges: set[tuple[int, int]] = set()
        boundary_xy = local_xy[boundary_indices]
        if len(boundary_xy) >= 3:
            try:
                triangulation = Delaunay(boundary_xy)
                for simplex in triangulation.simplices:
                    for i, j in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2])):
                        global_i, global_j = boundary_indices[int(i)], boundary_indices[int(j)]
                        delaunay_edges.add((min(global_i, global_j), max(global_i, global_j)))
            except QhullError:
                # Collinear/small observations still get the bounded-distance graph
                # below. No random Qhull joggle is used, keeping runs reproducible.
                diagnostics["delaunay_status"] = "degenerate_input"
        else:
            diagnostics["delaunay_status"] = "insufficient_boundary_cones"
        diagnostics["delaunay_edges"] = len(delaunay_edges)
        self.last_debug_geometry["delaunay_edges_local"] = [
            (tuple(map(float, local_xy[i])), tuple(map(float, local_xy[j])))
            for i, j in sorted(delaunay_edges)
        ]

        cross_pairs: list[tuple[int, int, np.ndarray, float, float]] = []
        midpoint_clearances: list[float] = []
        for i, j in itertools.combinations(boundary_indices, 2):
            delta = local_xy[j] - local_xy[i]
            width = math.hypot(float(delta[0]), float(delta[1]))
            if not (
                self.config.minimum_track_width_m
                <= width
                <= self.config.maximum_track_width_m
            ):
                continue
            if width > self.config.maximum_boundary_pair_distance_m:
                continue

            left_right_color = self._opposite_boundary_colors(unique_cones[i], unique_cones[j])
            # For unknown/mislabeled colors, a pair that is mostly lateral in
            # the vehicle frame remains a candidate. The 5 m distance limit
            # forms the complete radius graph recommended by CLC; triangulation
            # is a confidence feature, not a hard topology gate.
            geometric_cross = abs(float(delta[0])) <= min(
                self.config.maximum_pair_longitudinal_m,
                max(0.8, 0.65 * width),
            )
            if not left_right_color and not geometric_cross:
                continue

            # Reject same-side longitudinal neighbors even if their colors are
            # wrong. Opposite blue/yellow colors can support curved cross
            # sections whose axis is no longer aligned with the car heading.
            if not left_right_color and abs(float(delta[1])) < 0.45 * width:
                continue

            midpoint = 0.5 * (local_xy[i] + local_xy[j])
            if midpoint[0] < -0.5:
                continue
            midpoint_clearance = min(
                math.hypot(
                    float(cone_xy[0] - midpoint[0]),
                    float(cone_xy[1] - midpoint[1]),
                )
                for cone_xy in local_xy
            )
            if midpoint_clearance < self.config.minimum_cone_clearance_m:
                continue
            is_delaunay = (i, j) in delaunay_edges
            width_error = abs(width - self.config.target_track_width_m) / self.config.target_track_width_m
            skew_error = abs(float(delta[0])) / width
            color_cost = 0.0 if left_right_color else 0.12
            triangulation_cost = 0.0 if is_delaunay else 0.08
            center_clearance_cost = 0.15 * max(
                0.0,
                0.5 * self.config.target_track_width_m - midpoint_clearance,
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
            return stop("no_cross_track_pairs")

        self.last_debug_geometry["cross_sections_local"] = [
            (
                tuple(map(float, local_xy[i])),
                tuple(map(float, local_xy[j])),
                tuple(map(float, midpoint)),
            )
            for i, j, midpoint, _quality, _width in cross_pairs
        ]

        candidates: list[_Candidate] = [
            _Candidate(
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
                if max(np.linalg.norm(first), np.linalg.norm(second)) > self.config.maximum_observation_range_m:
                    continue
                self.last_debug_geometry["voronoi_edges_local"].append(
                    (tuple(map(float, first)), tuple(map(float, second)))
                )
            pair_midpoints = np.asarray([pair[2] for pair in cross_pairs])
            for vertex in vor_vertices:
                if vertex[0] < -0.5 or np.linalg.norm(vertex) > self.config.maximum_observation_range_m:
                    continue
                distances = np.linalg.norm(pair_midpoints - vertex, axis=1)
                nearest_pair = int(np.argmin(distances))
                support_distance = float(distances[nearest_pair])
                if support_distance > self.config.maximum_voronoi_support_distance_m:
                    continue
                cone_clearance = float(np.linalg.norm(local_xy - vertex, axis=1).min())
                if cone_clearance < self.config.minimum_cone_clearance_m:
                    continue
                _i, _j, supported_midpoint, pair_quality, _width = cross_pairs[nearest_pair]
                normalized_deviation = support_distance / max(
                    0.5 * self.config.target_track_width_m, 0.5
                )
                center_deviation_cost = (
                    self.config.center_deviation_cost_weight
                    * normalized_deviation
                    * normalized_deviation
                )
                candidates.append(
                    _Candidate(
                        np.asarray(vertex, dtype=np.float64),
                        pair_quality
                        + 0.06
                        + 0.025 * support_distance
                        + center_deviation_cost,
                        "voronoi_vertex",
                        nearest_pair,
                        cone_clearance,
                    )
                )
        except QhullError:
            diagnostics["voronoi_status"] = "degenerate_input"

        candidates = self._deduplicate_candidates(candidates)
        diagnostics["center_candidates"] = len(candidates)
        self.last_debug_geometry["center_candidates_local"] = [
            tuple(map(float, candidate.xy)) for candidate in candidates
        ]
        if len(candidates) < 2:
            return stop("insufficient_center_candidates")

        route, route_cost, graph_edges = self._dijkstra_route(
            candidates, local_xy, cross_pairs
        )
        diagnostics["graph_edges"] = graph_edges
        diagnostics["selected_nodes"] = len(route)
        diagnostics["route_cost"] = round(route_cost, 4)
        if len(route) < 2:
            return stop("no_connected_forward_route")

        route_points = np.asarray([candidates[index].xy for index in route], dtype=np.float64)
        self.last_debug_geometry["selected_route_local"] = [
            tuple(map(float, point)) for point in route_points
        ]
        route_pair_indices = tuple(
            dict.fromkeys(
                candidates[index].pair_index
                for index in route
                if candidates[index].pair_index >= 0
            )
        )
        if float(np.linalg.norm(route_points[0])) > self.config.maximum_start_distance_m:
            return stop("path_too_far_from_vehicle")

        sampled_s, sampled_xy, curvature = self._smooth(route_points)
        if len(sampled_xy) < 2 or not np.isfinite(sampled_xy).all() or not np.isfinite(curvature).all():
            return stop("spline_failure")

        cone_distances = np.linalg.norm(sampled_xy[:, None, :] - local_xy[None, :, :], axis=2)
        minimum_clearance = float(cone_distances.min())
        minimum_body_clearance = self._path_footprint_clearance(sampled_xy, local_xy)
        within_sections = self._within_cross_sections(
            sampled_xy, cross_pairs, local_xy, route_pair_indices
        )
        if (
            minimum_clearance < self.config.minimum_cone_clearance_m
            or minimum_body_clearance < 0.0
            or not within_sections
        ):
            # Keep the smooth cubic when it is safe. If it cuts across a cone
            # or section boundary, fall back to the collision-checked graph
            # polyline for this cycle rather than discarding a safe route.
            sampled_s, sampled_xy, curvature = self._smooth(route_points, cubic=False)
            cone_distances = np.linalg.norm(
                sampled_xy[:, None, :] - local_xy[None, :, :], axis=2
            )
            minimum_clearance = float(cone_distances.min())
            minimum_body_clearance = self._path_footprint_clearance(sampled_xy, local_xy)
            # Every polyline segment is the same candidate edge that passed
            # the local section-support check while building the Dijkstra
            # graph. Rechecking against every selected route section would
            # compare it to unrelated cross-sections and reject valid turns.
            within_sections = True
            diagnostics["smoothing_fallback"] = "collision_checked_polyline"
            diagnostics["section_validation"] = "dijkstra_edge_support"
        diagnostics["minimum_clearance_m"] = round(minimum_clearance, 3)
        diagnostics["minimum_body_clearance_m"] = round(minimum_body_clearance, 3)
        closest_path_idx, closest_cone_idx = np.unravel_index(
            int(np.argmin(cone_distances)), cone_distances.shape
        )
        diagnostics["closest_path_point_local"] = tuple(
            round(float(value), 3) for value in sampled_xy[closest_path_idx]
        )
        diagnostics["closest_cone_local"] = tuple(
            round(float(value), 3) for value in local_xy[closest_cone_idx]
        )
        if minimum_clearance < self.config.minimum_cone_clearance_m:
            return stop("cone_clearance_violation")

        if minimum_body_clearance < 0.0:
            return stop("vehicle_footprint_collision")

        if not within_sections:
            return stop("cross_section_boundary_violation")

        elapsed_ms = (perf_counter() - started) * 1000.0
        diagnostics["compute_ms"] = round(elapsed_ms, 3)
        if elapsed_ms > self.config.maximum_cycle_time_ms:
            return stop("planning_deadline_exceeded")

        cosine = math.cos(state.yaw_rad)
        sine = math.sin(state.yaw_rad)
        world_xy = np.column_stack(
            (
                state.x + cosine * sampled_xy[:, 0] - sine * sampled_xy[:, 1],
                state.y + sine * sampled_xy[:, 0] + cosine * sampled_xy[:, 1],
            )
        )
        points = tuple(
            TrajectoryPoint(
                s_m=float(s),
                x=float(xy[0]),
                y=float(xy[1]),
                curvature_per_m=float(kappa),
                target_speed_mps=self.config.safe_speed_mps,
            )
            for s, xy, kappa in zip(sampled_s, world_xy, curvature)
        )
        diagnostics["result"] = "ok"
        return Trajectory(
            frame_id=state.frame_id,
            timestamp_s=state.timestamp_s,
            points=points,
            valid=True,
            emergency_stop=False,
            compute_time_ms=elapsed_ms,
        )

    @staticmethod
    def _opposite_boundary_colors(first: Cone, second: Cone) -> bool:
        return {first.color, second.color} == {ConeColor.BLUE, ConeColor.YELLOW}

    @staticmethod
    def _local_unique_points(cones: tuple[Cone, ...], state: VehicleState) -> tuple[np.ndarray, tuple[Cone, ...]]:
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

    @staticmethod
    def _deduplicate_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
        # Keep the better cost/source deterministically while retaining the
        # preferred Voronoi support label when positions coincide.
        kept: list[_Candidate] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (item.quality_cost, item.xy[0], item.xy[1], item.source),
        ):
            match = next(
                (
                    i
                    for i, old in enumerate(kept)
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

    def _dijkstra_route(
        self,
        candidates: list[_Candidate],
        cone_xy: np.ndarray,
        cross_pairs: list[tuple[int, int, np.ndarray, float, float]],
    ) -> tuple[list[int], float, int]:
        """Search a collision-free, section-supported cone-center graph."""

        n = len(candidates)
        max_gap = self.config.maximum_candidate_gap_m
        footprint_outer_radius = math.hypot(
            0.5 * self.config.vehicle_length_m,
            0.5 * self.config.vehicle_width_m,
        ) + self.config.cone_radius_m + self.config.collision_margin_m
        adjacency: list[list[tuple[int, float, float]]] = [[] for _ in candidates]
        segment_clearance_cache: dict[tuple[int, int], float] = {}
        footprint_clearance_cache: dict[tuple[int, int], float] = {}
        section_support_cache: dict[tuple[int, int], bool] = {}
        section_midpoints = np.asarray([pair[2] for pair in cross_pairs])
        section_starts = np.asarray([cone_xy[pair[0]] for pair in cross_pairs])
        section_vectors = np.asarray(
            [cone_xy[pair[1]] - cone_xy[pair[0]] for pair in cross_pairs]
        )
        section_lengths_squared = np.maximum(
            np.sum(section_vectors * section_vectors, axis=1), 1e-9
        )
        section_geometry = tuple(
            (
                float(section_midpoints[index, 0]),
                float(section_midpoints[index, 1]),
                float(section_starts[index, 0]),
                float(section_starts[index, 1]),
                float(section_vectors[index, 0]),
                float(section_vectors[index, 1]),
                float(section_lengths_squared[index]),
            )
            for index in range(len(cross_pairs))
        )
        max_section_support = max(4.0, self.config.maximum_boundary_pair_distance_m)
        candidate_xy = np.asarray([candidate.xy for candidate in candidates])
        nearby_candidates = cKDTree(candidate_xy).query_ball_point(
            candidate_xy, r=max_gap
        )
        graph_edges_local: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []

        def edge_has_section_support(
            first: np.ndarray,
            second: np.ndarray,
            pair_indices: tuple[int, ...],
        ) -> bool:
            first_x, first_y = float(first[0]), float(first[1])
            delta_x = float(second[0]) - first_x
            delta_y = float(second[1]) - first_y
            distance = math.hypot(delta_x, delta_y)
            sample_count = max(2, int(math.ceil(distance / 0.25)) + 1)
            sections = tuple(section_geometry[index] for index in pair_indices)
            maximum_support_squared = max_section_support * max_section_support
            for sample_index in range(sample_count):
                fraction = sample_index / (sample_count - 1)
                x = first_x + fraction * delta_x
                y = first_y + fraction * delta_y
                best_section = None
                best_distance_squared = math.inf
                for section in sections:
                    midpoint_x, midpoint_y = section[0], section[1]
                    offset_x, offset_y = x - midpoint_x, y - midpoint_y
                    distance_squared = offset_x * offset_x + offset_y * offset_y
                    if distance_squared < best_distance_squared:
                        best_distance_squared = distance_squared
                        best_section = section
                if best_distance_squared > maximum_support_squared:
                    return False
                _mx, _my, start_x, start_y, vector_x, vector_y, length_squared = best_section
                projection = (
                    (x - start_x) * vector_x + (y - start_y) * vector_y
                ) / length_squared
                if projection < -0.12 or projection > 1.12:
                    return False
            return True

        edge_count = 0
        for i, neighbors in enumerate(nearby_candidates):
            for j in sorted(neighbors):
                if i == j:
                    continue
                delta = candidates[j].xy - candidates[i].xy
                distance = math.hypot(float(delta[0]), float(delta[1]))
                if distance < 0.35 or distance > max_gap or delta[0] < -0.45:
                    continue
                edge_key = (min(i, j), max(i, j))
                segment_clearance = segment_clearance_cache.get(edge_key)
                if segment_clearance is None:
                    first = candidates[edge_key[0]].xy
                    second = candidates[edge_key[1]].xy
                    segment_x = float(second[0] - first[0])
                    segment_y = float(second[1] - first[1])
                    length_squared = segment_x * segment_x + segment_y * segment_y
                    offsets_x = cone_xy[:, 0] - first[0]
                    offsets_y = cone_xy[:, 1] - first[1]
                    projections = np.clip(
                        (offsets_x * segment_x + offsets_y * segment_y)
                        / max(length_squared, 1e-9),
                        0.0,
                        1.0,
                    )
                    nearest_x = first[0] + projections * segment_x
                    nearest_y = first[1] + projections * segment_y
                    dx = cone_xy[:, 0] - nearest_x
                    dy = cone_xy[:, 1] - nearest_y
                    segment_clearance = float(np.sqrt(dx * dx + dy * dy).min())
                    segment_clearance_cache[edge_key] = segment_clearance
                if segment_clearance < self.config.minimum_cone_clearance_m:
                    continue
                footprint_clearance = footprint_clearance_cache.get(edge_key)
                if footprint_clearance is None:
                    if segment_clearance >= footprint_outer_radius:
                        footprint_clearance = segment_clearance - footprint_outer_radius
                    else:
                        footprint_clearance = self._segment_footprint_clearance(
                            candidates[edge_key[0]].xy,
                            candidates[edge_key[1]].xy,
                            cone_xy,
                        )
                    footprint_clearance_cache[edge_key] = footprint_clearance
                if footprint_clearance < 0.0:
                    continue
                edge_supported = section_support_cache.get(edge_key)
                if edge_supported is None:
                    pair_indices = tuple(
                        dict.fromkeys(
                            (
                                candidates[edge_key[0]].pair_index,
                                candidates[edge_key[1]].pair_index,
                            )
                        )
                    )
                    edge_supported = edge_has_section_support(
                        candidates[edge_key[0]].xy,
                        candidates[edge_key[1]].xy,
                        pair_indices,
                    )
                    section_support_cache[edge_key] = edge_supported
                if not edge_supported:
                    continue
                heading = math.atan2(float(delta[1]), float(delta[0]))
                adjacency[i].append((j, distance, heading))
                edge_count += 1
                if i < j:
                    graph_edges_local.append(
                        (
                            tuple(map(float, candidates[i].xy)),
                            tuple(map(float, candidates[j].xy)),
                        )
                    )

        self.last_debug_geometry["candidate_graph_edges_local"] = graph_edges_local

        starts: list[tuple[int, float, float]] = []
        vehicle_origin = np.zeros(2, dtype=np.float64)
        for i, candidate in enumerate(candidates):
            distance = math.hypot(float(candidate.xy[0]), float(candidate.xy[1]))
            if distance < 0.35 or distance > self.config.maximum_start_distance_m:
                continue
            heading = math.atan2(float(candidate.xy[1]), float(candidate.xy[0]))
            if abs(heading) > self.config.maximum_initial_heading_error_rad:
                continue
            if self._segment_footprint_clearance(
                vehicle_origin, candidate.xy, cone_xy
            ) < 0.0:
                continue
            starts.append((i, distance, heading))
        if not starts:
            return [], math.inf, edge_count

        # State is (previous candidate, current candidate), allowing Dijkstra's
        # edge cost to include change in heading without using a greedy local
        # choice. Positive weights make the selected path deterministic.
        State = tuple[int, int]
        best_cost: dict[State, float] = {}
        best_length: dict[State, float] = {}
        parent: dict[State, State | None] = {}
        queue: list[tuple[float, int, int]] = []
        for i, distance, heading in starts:
            state = (-1, i)
            cost = distance + self.config.turn_cost_weight * heading * heading + candidates[i].quality_cost
            if cost < best_cost.get(state, math.inf):
                best_cost[state] = cost
                best_length[state] = distance
                parent[state] = None
                heapq.heappush(queue, (cost, -1, i))

        settled: set[State] = set()
        while queue:
            cost, previous, current = heapq.heappop(queue)
            state = (previous, current)
            if state in settled or cost > best_cost.get(state, math.inf) + 1e-12:
                continue
            settled.add(state)
            previous_heading = 0.0 if previous < 0 else math.atan2(
                float(candidates[current].xy[1] - candidates[previous].xy[1]),
                float(candidates[current].xy[0] - candidates[previous].xy[0]),
            )
            for nxt, distance, heading in adjacency[current]:
                if nxt == previous:
                    continue
                turn = self._angle_difference(heading, previous_heading)
                if previous >= 0 and abs(turn) > self.config.maximum_heading_change_rad:
                    continue
                next_state = (current, nxt)
                edge_cost = (
                    distance
                    + self.config.turn_cost_weight * turn * turn
                    + candidates[nxt].quality_cost
                )
                next_cost = cost + edge_cost
                if next_cost + 1e-12 >= best_cost.get(next_state, math.inf):
                    continue
                best_cost[next_state] = next_cost
                best_length[next_state] = best_length[state] + distance
                parent[next_state] = state
                heapq.heappush(queue, (next_cost, current, nxt))

        possible_goals = [
            state
            for state in settled
            if best_length[state] >= 1.5
            and math.hypot(
                float(candidates[state[1]].xy[0]),
                float(candidates[state[1]].xy[1]),
            )
            >= 2.0
        ]
        if not possible_goals:
            return [], math.inf, edge_count

        # Prefer the longest observed horizon while keeping cost as a tie
        # breaker. This avoids stopping at a cheap but very short branch.
        goal = max(
            possible_goals,
            key=lambda state: (
                math.hypot(
                    float(candidates[state[1]].xy[0]),
                    float(candidates[state[1]].xy[1]),
                )
                - 0.12 * best_cost[state],
                -best_cost[state],
                -state[1],
            ),
        )
        route_states: list[State] = []
        cursor: State | None = goal
        while cursor is not None:
            route_states.append(cursor)
            cursor = parent[cursor]
        route_states.reverse()
        route: list[int] = []
        for previous, current in route_states:
            if not route or route[-1] != current:
                route.append(current)
        return route, best_cost[goal], edge_count

    @staticmethod
    def _angle_difference(angle_a: float, angle_b: float) -> float:
        return (angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi

    def _segment_footprint_clearance(
        self,
        first: np.ndarray,
        second: np.ndarray,
        cone_xy: np.ndarray,
    ) -> float:
        """Return cone clearance from the car swept along one straight edge."""

        first_x, first_y = float(first[0]), float(first[1])
        delta_x = float(second[0]) - first_x
        delta_y = float(second[1]) - first_y
        length = math.hypot(delta_x, delta_y)
        if length > 1e-9:
            forward_x, forward_y = delta_x / length, delta_y / length
        else:
            forward_x, forward_y = 1.0, 0.0
        lateral_x, lateral_y = -forward_y, forward_x
        half_length = 0.5 * self.config.vehicle_length_m
        half_width = 0.5 * self.config.vehicle_width_m
        minimum_distance = math.inf
        for cone in cone_xy:
            relative_x = float(cone[0]) - first_x
            relative_y = float(cone[1]) - first_y
            along = relative_x * forward_x + relative_y * forward_y
            lateral = relative_x * lateral_x + relative_y * lateral_y
            excess_along = max(
                -half_length - along,
                along - (length + half_length),
                0.0,
            )
            excess_lateral = max(abs(lateral) - half_width, 0.0)
            distance = math.hypot(excess_along, excess_lateral)
            if distance < minimum_distance:
                minimum_distance = distance
        return (
            minimum_distance - self.config.cone_radius_m
            - self.config.collision_margin_m
        )

    def _path_footprint_clearance(
        self,
        path_xy: np.ndarray,
        cone_xy: np.ndarray,
    ) -> float:
        """Check the car body, not just its centerline, along the sampled path."""

        origin = np.zeros(2, dtype=np.float64)
        initial_clearance = self._segment_footprint_clearance(
            origin, path_xy[0], cone_xy
        )
        if initial_clearance < 0.0:
            return initial_clearance
        outer_radius = math.hypot(
            0.5 * self.config.vehicle_length_m,
            0.5 * self.config.vehicle_width_m,
        ) + self.config.cone_radius_m + self.config.collision_margin_m
        point_clearances = np.linalg.norm(
            path_xy[:, None, :] - cone_xy[None, :, :], axis=2
        )
        minimum_point_clearance = float(point_clearances.min())
        interpolation_allowance = 0.5 * self.config.spline_sample_spacing_m
        if minimum_point_clearance >= outer_radius + interpolation_allowance:
            return min(
                initial_clearance,
                minimum_point_clearance - outer_radius - interpolation_allowance,
            )
        return min(
            initial_clearance,
            min(
                self._segment_footprint_clearance(first, second, cone_xy)
                for first, second in zip(path_xy[:-1], path_xy[1:])
            ),
        )

    def _smooth(
        self,
        route_points: np.ndarray,
        cubic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = [0]
        for i in range(1, len(route_points)):
            if float(np.linalg.norm(route_points[i] - route_points[keep[-1]])) >= 0.25:
                keep.append(i)
        route_points = route_points[keep]
        if len(route_points) < 2:
            return np.empty(0), np.empty((0, 2)), np.empty(0)
        segment_lengths = np.linalg.norm(np.diff(route_points, axis=0), axis=1)
        control_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        if cubic and len(route_points) >= 3:
            spline = CubicSpline(control_s, route_points, axis=0, bc_type="natural")
            sample_s = np.arange(0.0, control_s[-1], self.config.spline_sample_spacing_m)
            if not len(sample_s) or control_s[-1] - sample_s[-1] > 1e-6:
                sample_s = np.append(sample_s, control_s[-1])
            xy = spline(sample_s)
            first = spline(sample_s, 1)
            second = spline(sample_s, 2)
            denominator = np.maximum(np.sum(first * first, axis=1) ** 1.5, 1e-9)
            curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator
            return sample_s, xy, curvature
        sample_s = np.arange(0.0, control_s[-1], self.config.spline_sample_spacing_m)
        if not len(sample_s) or control_s[-1] - sample_s[-1] > 1e-6:
            sample_s = np.append(sample_s, control_s[-1])
        xy = np.column_stack(
            [np.interp(sample_s, control_s, route_points[:, axis]) for axis in range(2)]
        )
        return sample_s, xy, np.zeros(len(sample_s), dtype=np.float64)

    def _within_cross_sections(
        self,
        path_xy: np.ndarray,
        cross_pairs: list[tuple[int, int, np.ndarray, float, float]],
        cone_xy: np.ndarray,
        pair_indices: tuple[int, ...] | None = None,
    ) -> bool:
        selected_pairs = (
            cross_pairs
            if pair_indices is None
            else [cross_pairs[index] for index in pair_indices]
        )
        if not selected_pairs:
            return False
        local_sections = []
        for first_idx, second_idx, midpoint, _quality, _width in selected_pairs:
            a = cone_xy[first_idx]
            b = cone_xy[second_idx]
            local_sections.append((a, b, midpoint))
        mids = np.asarray([section[2] for section in local_sections])
        endpoints_a = np.asarray([section[0] for section in local_sections])
        segments = np.asarray([section[1] - section[0] for section in local_sections])
        max_support = max(4.0, self.config.maximum_boundary_pair_distance_m)
        offsets = path_xy[:, None, :] - mids[None, :, :]
        nearest = np.argmin(np.sum(offsets * offsets, axis=2), axis=1)
        nearest_distance = np.sqrt(np.sum(offsets[np.arange(len(path_xy)), nearest] ** 2, axis=1))
        if np.any(nearest_distance > max_support):
            return False
        path_offsets = path_xy - endpoints_a[nearest]
        selected_segments = segments[nearest]
        numerator = np.sum(path_offsets * selected_segments, axis=1)
        denominator = np.maximum(np.sum(selected_segments * selected_segments, axis=1), 1e-9)
        projection = numerator / denominator
        return bool(np.all((projection >= -0.12) & (projection <= 1.12)))
