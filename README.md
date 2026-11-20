# Khonshu Algorithm

Path planning for Strathmore Racing Club's Formula Student autonomous car. Khonshu turns cone observations and vehicle state into a trajectory for Control.

## v0.1 baseline

The first baseline wraps FaSTTUBe's open-source Formula Student path planner. Its cone sorting and matching pipeline can use known and unknown cone colors, and it estimates a spline path with curvature. Khonshu adds a stable input/output contract, frame and unit conventions, a conservative constant target speed, and an explicit stop request when the input or result is not trustworthy.

The planner core has no ROS dependency. The ROS 2 adapters are not in this branch yet; when added, EUFS, FSDS, and real-car nodes will convert their messages to `Cone` and `VehicleState`, call `LocalPlanner`, then convert `Trajectory` back to native message types. That keeps algorithm behavior separate from simulator topic names. Paths that pass within 0.75 m of an observed cone or cycles that exceed the default 30 ms budget return an explicit stop request instead of an untrusted or stale trajectory. Tune the clearance against the car width and perception uncertainty.

## Setup

Use Python 3.9 or newer in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest
```

On Linux, activate the environment with `source .venv/bin/activate` instead. The vendored upstream library's first call may compile Numba functions; call `LocalPlanner.warm_up()` while the car is stationary before timing or driving.

## Minimal use

```python
from khonshu import Cone, ConeColor, LocalPlanner, VehicleState

planner = LocalPlanner()
planner.warm_up()

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

Coordinates are meters in a right-handed map frame: `x` forward, `y` left, and yaw counter-clockwise from +x. Cones and vehicle state in a call must use the same frame. v0.1 uses a fixed target speed of 2 m/s; it does not yet model braking, tire limits, or a G-G diagram.

## Project notes

- [Architecture and v0.1 limits](docs/architecture.md)
- [Research and implementation references](docs/REFERENCES.md)
- [Local v0.1 benchmark results](benchmarks/baseline-v0.1.md)
- [Vendored upstream provenance and license](third_party/ft-fsd-path-planning/UPSTREAM.md)

Run the shared 2D scenarios with `python -m pytest tests/test_vendor_smoke.py -m integration`. Run the latency baseline with `python examples/benchmark_synthetic.py --iterations 50`. It reports warm-up separately from steady-state p50/p95/max latency and cycles over the 30 ms deadline across straight, turning, missing-boundary, unknown-color, and false-positive cases. Narrow a rerun with `--scenario missing_inside_boundary --iterations 100`.

The reported upstream timing is a reference, not a Khonshu performance claim. Benchmark on the target EUFS, FSDS, and real-car computers after warm-up. Track completion, cone contacts, planning latency, and stop requests under the same track and sensor inputs.
