# Khonshu v0.1 architecture

## Planning pipeline

```text
EUFS / FSDS / real-car adapter
              |
      Cone[] + VehicleState
              |
    StageAPlanner (core)
  local geometry in vehicle frame
  Delaunay + bounded-radius cone graph
  cross-track midpoint candidates
  Voronoi medial-axis support
  cost-based Dijkstra route search
  spline + cone, vehicle-footprint, and boundary checks
              |
 Trajectory (global-frame x/y, s,
             curvature, target speed)
              |
   simulator or vehicle controller
```

The core has no ROS dependency. It accepts cone locations and vehicle pose in a common frame, converts the observations into a vehicle-aligned planning frame, then converts the accepted trajectory back to the input frame. Adapters own timestamps, ROS messages, TF, and command topics. The EUFS adapter currently uses timestamp-matched ground-truth odometry for development; FSDS and real-car adapters remain future work.

## Stage A — track geometry and centerline

`StageAPlanner` is the default ROS preview planner and has an independent `LocalPlanner`/FaSTTUBe comparison path.

1. **Build cone topology.** SciPy Delaunay edges identify locally supported cone neighbors. The planner also considers geometrically plausible cone pairs inside a 5 m radius. It does not use Delaunay as a hard gate: the CLC lane-detection paper describes how false-positive cones can remove a correct boundary edge from a Delaunay-only graph.
2. **Propose cross-track sections.** Opposite blue/yellow colors are a soft hint. Unknown and mislabeled cones are eligible when their geometry supports a cross-track pair. Pairs are filtered by plausible track width, distance, and orientation; their midpoints are centerline candidates.
3. **Add Voronoi support.** Finite Voronoi vertices near supported cross-track midpoints add between-section medial-axis candidates. Unbounded regions and vertices unsupported by cone pairs are excluded.
4. **Choose a continuous route.** A directed candidate graph connects nearby points. Dijkstra minimizes distance, candidate-quality, and heading-change costs, then selects the best reachable observation horizon. This is a center-candidate search; it is not a claim to reproduce CLC's two-boundary backtracking search or its learned candidate ranker.
5. **Smooth and validate.** A chord-length parameterized cubic spline is sampled at fixed spacing. The planner checks centerline clearance and sweeps a configurable rectangular vehicle footprint along candidate edges and the final path. The default footprint is 2.4 m by 1.4 m, inflated by a 0.15 m cone radius and 0.25 m tracking margin. It also rejects paths that leave supported cross-sections, insufficient evidence, and over-budget cycles. An invalid result is an empty trajectory with `emergency_stop=True`.

The KIT winning-team paper supports Delaunay filtering, boundary inference, cross-track midpoint control points, and spline smoothing. Voronoi support and explicit cost-based centerline search are additions in our branch, not methods attributed to that paper. See the [reference index](REFERENCES.md) and the paper saved in `docs/`.

## v0.1 limits and EUFS validation

- Stage A uses a constant 1.0 m/s target speed. The EUFS pure-pursuit validation node is separately capped at 0.8 m/s and publishes `AckermannDriveStamped` to `/cmd` in velocity mode.
- The planner enforces a configurable 0.8 m centerline clearance and a rectangular swept-footprint check. The footprint dimensions are a conservative v0.1 model and should be matched to the EUFS collision mesh and measured car before hardware use; these checks are not a complete vehicle safety case.
- The default planning deadline is 30 ms. The check runs after computation, so a consumer must stop on an empty or stale path. The EUFS follower does this and also stops on stale odometry.
- `run_eufs_stage_a_lap.sh` starts the visible EUFS rectangle course, the Stage A planner, and the low-speed follower. Orange cones are collision obstacles, not track-boundary samples, so displaced apex markers cannot attract the centerline. RViz opens by default and shows the planner's Delaunay, Voronoi, midpoint, candidate graph, and selected Dijkstra route markers; set `KHONSHU_SHOW_RVIZ=0` to disable it. The follower uses a shorter 1.0 m lookahead to reduce corner cutting. The helper uses a 200 ms cycle deadline to tolerate WSL GUI scheduling, while the planner default remains 30 ms. It exits after the vehicle returns near its start pose after a minimum travel distance, or reports a timeout. The test is a simulator integration result only.
- `LocalPlanner` retains the MIT-licensed FaSTTUBe pipeline for branch comparison. It is not silently used as a fallback when Stage A rejects a plan.

## Later stages

1. **Robust topology:** add deterministic boundary-pair/backtracking search with explicit lane-width and turn constraints; replay real perception logs with false positives, missing boundaries, and color errors. Compare with the current radius/Delaunay graph and FaSTTUBe.
2. **Dynamics and speed:** add a friction/G-G constrained forward-backward velocity pass, braking feasibility, and vehicle-specific limits.
3. **Global mapping and racing line:** after localisation and loop closure are reliable, evaluate minimum-curvature/minimum-time optimization with track-boundary constraints and OSQP.
4. **Portability:** replay identical serialized cone/state inputs through EUFS, FSDS, and hardware adapters and compare path geometry, stop decisions, latency, and collision outcomes.

The winning-team report describes a larger stack that also includes mapping, a G-G-aware velocity profile, and minimum-curvature trajectory optimization. Those are later-stage capabilities; they are not claimed for this Stage A v0.1 planner.
