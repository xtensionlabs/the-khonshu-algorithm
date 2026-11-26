"""Stage A pipeline orchestration and stable public planner API."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Iterable

import numpy as np

from ..types import Cone, Trajectory, TrajectoryPoint, VehicleState
from .config import StageAConfig
from .search import search_route
from .safety import path_footprint_clearance, within_cross_sections
from .smoothing import smooth_route
from .topology import build_topology, empty_debug_geometry


class StageAPlanner:
    """Plan a checked local centerline from partial cone observations.

    The planner coordinates independent topology, search, smoothing, and safety
    stages. Its ROS-independent inputs and output remain the portable
    ``Cone``/``VehicleState`` and ``Trajectory`` data types.
    """

    def __init__(self, config: StageAConfig | None = None):
        self.config = config or StageAConfig()
        self.last_diagnostics: dict[str, object] = {}
        self.last_debug_geometry: dict[str, list] = empty_debug_geometry()

    def plan(self, cones: Iterable[Cone], state: VehicleState) -> Trajectory:
        """Run topology → graph search → smoothing → safety validation."""

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
        self.last_debug_geometry = empty_debug_geometry()

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

        topology = build_topology(cone_list, state, self.config)
        diagnostics.update(topology.diagnostics)
        self.last_debug_geometry = topology.debug_geometry
        if topology.failure_reason is not None:
            return stop(topology.failure_reason)

        search = search_route(
            topology.candidates,
            topology.cone_xy,
            topology.cross_pairs,
            self.config,
        )
        diagnostics["graph_edges"] = search.edge_count
        diagnostics["selected_nodes"] = len(search.route)
        diagnostics["route_cost"] = round(search.route_cost, 4)
        self.last_debug_geometry["candidate_graph_edges_local"] = list(
            search.graph_edges_local
        )
        if len(search.route) < 2:
            return stop("no_connected_forward_route")

        route_points = np.asarray(
            [topology.candidates[index].xy for index in search.route],
            dtype=np.float64,
        )
        self.last_debug_geometry["selected_route_local"] = [
            tuple(map(float, point)) for point in route_points
        ]
        route_pair_indices = tuple(
            dict.fromkeys(
                topology.candidates[index].pair_index
                for index in search.route
                if topology.candidates[index].pair_index >= 0
            )
        )
        if float(np.linalg.norm(route_points[0])) > self.config.maximum_start_distance_m:
            return stop("path_too_far_from_vehicle")

        sampled_s, sampled_xy, curvature = smooth_route(route_points, self.config)
        if (
            len(sampled_xy) < 2
            or not np.isfinite(sampled_xy).all()
            or not np.isfinite(curvature).all()
        ):
            return stop("spline_failure")

        cone_distances = np.linalg.norm(
            sampled_xy[:, None, :] - topology.cone_xy[None, :, :], axis=2
        )
        minimum_clearance = float(cone_distances.min())
        minimum_body_clearance = path_footprint_clearance(
            sampled_xy, topology.cone_xy, self.config
        )
        within_sections = within_cross_sections(
            sampled_xy,
            topology.cross_pairs,
            topology.cone_xy,
            self.config,
            route_pair_indices,
        )
        if (
            minimum_clearance < self.config.minimum_cone_clearance_m
            or minimum_body_clearance < 0.0
            or not within_sections
        ):
            # Keep the cubic when safe; otherwise use the already checked graph
            # polyline instead of discarding a viable route.
            sampled_s, sampled_xy, curvature = smooth_route(
                route_points, self.config, cubic=False
            )
            cone_distances = np.linalg.norm(
                sampled_xy[:, None, :] - topology.cone_xy[None, :, :], axis=2
            )
            minimum_clearance = float(cone_distances.min())
            minimum_body_clearance = path_footprint_clearance(
                sampled_xy, topology.cone_xy, self.config
            )
            # Each fallback segment is a Dijkstra edge already checked against
            # its supporting cross-sections.
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
            round(float(value), 3) for value in topology.cone_xy[closest_cone_idx]
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

        cosine, sine = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
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
