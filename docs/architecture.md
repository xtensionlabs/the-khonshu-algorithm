# Khonshu v0.1 architecture

## Pipeline

```text
EUFS / FSDS / real-car adapter
              |
   Cone[] + VehicleState
              |
 LocalPlanner (ROS-independent)
   FaSTTUBe sorting, matching,
   virtual-cone and spline pipeline
              |
  Trajectory (x, y, s, curvature,
              constant target speed)
              |
EUFS / FSDS / real-car adapter
```

The adapter layer is the next integration task and is not implemented on this branch yet. It will own ROS topics, message types, TF lookups, and visualization. The core owns geometry, planning behavior, units, frame identifiers, validity checks, and the stop decision. The core accepts global-frame positions and yaw in radians; adapters must transform cone observations to the vehicle/map frame at the observation timestamp before planning.

## v0.1 behavior

- Uses the MIT-licensed FaSTTUBe Formula Student planner as the centerline and spline backend. It can use unknown cone colors and can infer a missing boundary through virtual cones.
- Keeps one backend instance alive so its path continuity and previous-path recovery work as designed.
- Publishes only finite trajectories with increasing path distance, at least two waypoints, enough sorted boundary evidence, and a start point within 5 m of the vehicle.
- Rejects a path whose sampled points approach an observed cone closer than the configurable 0.75 m clearance. This is a first collision screen, not a replacement for swept-footprint or track-boundary validation.
- Uses a constant 2 m/s target speed by default.
- Requests an emergency stop with an empty path if cone evidence is insufficient, the backend errors, or the resulting path fails validation. Consumers must treat `emergency_stop` as a stop command and must not track an empty path.
- Requests an emergency stop when a cycle exceeds the default 30 ms planning budget; this detects an overrun after the calculation, so the ROS adapter still needs a control-side watchdog for missed deadlines.
- Calls `warm_up()` before a timed run or drive. It runs all shared synthetic cases to precompile branch-specific Numba paths, then resets the stateful backend. The upstream project reports a 30–60 second first-call Numba compilation delay, so the first plan is not a real-time timing sample.
- Uses [`khonshu.scenarios.synthetic_scenarios`](../src/khonshu/scenarios.py) and [`examples/benchmark_synthetic.py`](../examples/benchmark_synthetic.py) for deterministic no-ROS straight, left/right turn, missing-boundary, unknown-color, and false-positive cases; simulator benchmarks must still record lap time and cone contacts.

## Roadmap

1. **v0.1 integration:** wire the shared types to EUFS and reproduce a clean lap; add identical replay inputs for the planner branches.
2. **Topology comparisons:** benchmark the FaSTTUBe baseline against Delaunay midpoint extraction and the deterministic CLC graph-search method on noisy and missing-cone cases. Keep whichever gives the best safety and latency across the team's recorded inputs.
3. **Dynamics:** replace constant speed with a friction/G-G constrained forward-backward velocity pass; retain a hard vehicle speed and acceleration cap.
4. **Global planning:** once mapping and loop closure are reliable, evaluate a minimum-curvature or minimum-time racing line with track-boundary constraints and an OSQP formulation.
5. **Portability gates:** replay the same serialized cone/state data through EUFS, FSDS, and hardware adapters; compare geometric output, stop decisions, latency, and collision outcomes.

The FaSTTUBe backend is a strong starting point, not a claim that this version is already competitive or safe for a real vehicle. Sensor uncertainty, braking feasibility, vehicle dynamics, and hardware latency still require validation before a track run.
