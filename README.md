# Khonshu Algorithm

Portable path planning for Strathmore Racing Club's Formula Student autonomous car. The core maps partial cone observations and vehicle state to a checked trajectory; ROS adapters connect it to simulators and, later, the real vehicle.

## Stage A v0.1

The active Stage A planner combines Delaunay-supported cone topology, a 5 m bounded-radius cone-pair graph, cross-track midpoint and supported Voronoi candidates, cost-based Dijkstra route selection, cubic-spline smoothing, and cone/cross-section validation. Blue/yellow colors are soft hints, unknown cones may contribute geometrically, and orange cones are treated as interior obstacles by default. The retained FaSTTUBe planner is available as an independent comparison backend.

The ROS-independent Python core is in `src/khonshu`. The EUFS adapter uses timestamp-matched cone and odometry data; the EUFS helper opens the Gazebo GUI and runs a low-speed pure-pursuit follower to validate an actual lap. FSDS and real-car adapters are not implemented yet.

## Setup

Use Python 3.9 or newer in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,visualize]"
```

On Linux, activate with `source .venv/bin/activate` instead.

## Minimal use

```python
from khonshu import Cone, ConeColor, StageAPlanner, VehicleState

planner = StageAPlanner()
cones = [
    *[Cone(x, 2.0, ConeColor.BLUE) for x in (3.0, 6.0, 9.0, 12.0)],
    *[Cone(x, -2.0, ConeColor.YELLOW) for x in (3.0, 6.0, 9.0, 12.0)],
]
state = VehicleState(x=0.0, y=0.0, yaw_rad=0.0, speed_mps=0.0, timestamp_s=0.0)
trajectory = planner.plan(cones, state)

if trajectory.emergency_stop:
    print("Stop requested:", trajectory.reason)
else:
    print("Waypoints:", len(trajectory.points))
```

Positions are meters in a right-handed frame (`x` forward, `y` left); yaw is radians counter-clockwise from +x. The planner searches in a vehicle-aligned frame and returns the accepted path in the input frame. Stage A uses a constant 1.0 m/s target speed and a default 30 ms planning deadline. The EUFS follower is separately limited to 0.8 m/s.

## Run and compare

```powershell
python -m pytest tests/test_stage_a.py -q
python examples/visualize_synthetic.py --planner stage_a --scenario false_positives
python examples/visualize_full_circuit_3d.py
python examples/benchmark_synthetic.py --planner stage_a --iterations 50
```

The full-circuit visualization replays the local Stage A planner at poses around a synthetic closed course and saves `artifacts/full-circuit-3d.png` plus a self-contained, rotatable `artifacts/full-circuit-3d.html`. Cone height is shown for perspective; track geometry and Delaunay triangulation remain in the XY plane.

For the visible EUFS rectangle lap, run `bash ros2/khonshu_eufs/run_eufs_stage_a_lap.sh` from the repository root in WSL. The helper opens Gazebo and RViz as two windows by default. RViz shows the live Delaunay mesh, Voronoi ridges, cross-sections, center candidates, graph, selected Dijkstra route, cones, and centerline. Set `KHONSHU_SHOW_RVIZ=0` only when you want a Gazebo-only run. The follower uses a 1.0 m lookahead to reduce corner cutting. After a successful lap, the helper leaves both windows open for 10 seconds; adjust this with `KHONSHU_POST_LAP_HOLD_S`. See [testing instructions](docs/testing.md), [architecture and limits](docs/architecture.md), and the [research reference index](docs/REFERENCES.md).

The local FaSTTUBe source provenance and license are recorded in `third_party/ft-fsd-path-planning/UPSTREAM.md`. Its output is not a silent fallback: Stage A rejects an invalid plan by returning an empty trajectory and an explicit stop reason.

## Repository map

```text
src/khonshu/
  types.py                 shared ROS-independent inputs and trajectory types
  planner.py               retained FaSTTUBe comparison backend
  stage_a/
    config.py              Stage A tuning and vehicle geometry parameters
    topology.py            cone filtering, Delaunay, cross-sections, Voronoi
    search.py              collision-checked candidate graph and Dijkstra
    safety.py              vehicle footprint and track-section checks
    smoothing.py           cubic spline and checked polyline sampling
    planner.py             Stage A pipeline orchestration
ros2/khonshu_eufs/         EUFS adapters, follower, launch scripts, RViz config
tests/                     deterministic planner and geometry checks
examples/                  synthetic visualizations and benchmarks
docs/                      architecture, setup, validation, and references
```

Import stable core types and planners from `khonshu`. Keep ROS message handling in `ros2/`; keep algorithm stages ROS-independent under `src/khonshu/`. Future speed profiling belongs in a separate Stage B module that consumes a validated Stage A trajectory.
