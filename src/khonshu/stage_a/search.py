"""Collision-checked candidate graph construction and Dijkstra route search."""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.spatial import cKDTree

from .config import StageAConfig
from .models import Candidate, CrossTrackPair, SearchResult
from .safety import segment_footprint_clearance


def angle_difference(angle_a: float, angle_b: float) -> float:
    """Return the signed shortest angular difference in radians."""

    return (angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi


def search_route(
    candidates: tuple[Candidate, ...] | list[Candidate],
    cone_xy: np.ndarray,
    cross_pairs: tuple[CrossTrackPair, ...] | list[CrossTrackPair],
    config: StageAConfig,
) -> SearchResult:
    """Build supported graph edges, then choose a connected route with Dijkstra."""

    n = len(candidates)
    max_gap = config.maximum_candidate_gap_m
    footprint_outer_radius = math.hypot(
        0.5 * config.vehicle_length_m,
        0.5 * config.vehicle_width_m,
    ) + config.cone_radius_m + config.collision_margin_m
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
    max_section_support = max(4.0, config.maximum_boundary_pair_distance_m)
    candidate_xy = np.asarray([candidate.xy for candidate in candidates])
    nearby_candidates = cKDTree(candidate_xy).query_ball_point(candidate_xy, r=max_gap)
    graph_edges_local = []

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
            if segment_clearance < config.minimum_cone_clearance_m:
                continue
            footprint_clearance = footprint_clearance_cache.get(edge_key)
            if footprint_clearance is None:
                if segment_clearance >= footprint_outer_radius:
                    footprint_clearance = segment_clearance - footprint_outer_radius
                else:
                    footprint_clearance = segment_footprint_clearance(
                        candidates[edge_key[0]].xy,
                        candidates[edge_key[1]].xy,
                        cone_xy,
                        config,
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

    starts: list[tuple[int, float, float]] = []
    vehicle_origin = np.zeros(2, dtype=np.float64)
    for index, candidate in enumerate(candidates):
        distance = math.hypot(float(candidate.xy[0]), float(candidate.xy[1]))
        if distance < 0.35 or distance > config.maximum_start_distance_m:
            continue
        heading = math.atan2(float(candidate.xy[1]), float(candidate.xy[0]))
        if abs(heading) > config.maximum_initial_heading_error_rad:
            continue
        if segment_footprint_clearance(vehicle_origin, candidate.xy, cone_xy, config) < 0.0:
            continue
        starts.append((index, distance, heading))
    if not starts:
        return SearchResult((), math.inf, edge_count, tuple(graph_edges_local))

    # Search state retains the previous node so turn costs are globally
    # considered instead of taking a greedy nearest-midpoint hop.
    State = tuple[int, int]
    best_cost: dict[State, float] = {}
    best_length: dict[State, float] = {}
    parent: dict[State, State | None] = {}
    queue: list[tuple[float, int, int]] = []
    for index, distance, heading in starts:
        state = (-1, index)
        cost = (
            distance
            + config.turn_cost_weight * heading * heading
            + candidates[index].quality_cost
        )
        if cost < best_cost.get(state, math.inf):
            best_cost[state] = cost
            best_length[state] = distance
            parent[state] = None
            heapq.heappush(queue, (cost, -1, index))

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
            turn = angle_difference(heading, previous_heading)
            if previous >= 0 and abs(turn) > config.maximum_heading_change_rad:
                continue
            next_state = (current, nxt)
            edge_cost = (
                distance
                + config.turn_cost_weight * turn * turn
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
        return SearchResult((), math.inf, edge_count, tuple(graph_edges_local))

    # Prefer the longest observed horizon while keeping path cost as a tie
    # breaker, avoiding a cheap but very short branch.
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
    return SearchResult(tuple(route), best_cost[goal], edge_count, tuple(graph_edges_local))
