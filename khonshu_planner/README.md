# Khonshu Planner (v0.1 — ivan-barasa)

ROS-independent Delaunay + cubic spline + forward-backward velocity path planner wrapped in a 20 Hz ROS 2 Humble node for EUFS.

## Architecture
- `khonshu_planner/core/` — Pure Python (`numpy`/`scipy` only, zero ROS imports):
  - `types.py`: immutable `Cone`, `ConeColor`, `VehicleState`, `Waypoint`, `TrackEstimate`, `Trajectory`
  - `delaunay.py`: width-filtered Delaunay triangulation, orange gate handling, heading-constrained cost ordering, duplicate midpoint suppression
  - `smoothing.py`: Laplacian pre-smoothing, arc-length `CubicSpline`, curvature, and lateral-grip + forward-accel + backward-brake speed profile
  - `planner.py`: pipeline orchestration + explicit zero-speed failsafe stop
- `khonshu_planner/nodes/` — Thin ROS 2 adapters:
  - `planner_node.py`: `/ground_truth/cones` + `/ground_truth/state` -> `/planning/trajectory` (`eufs_msgs/msg/WaypointArrayStamped`) + `/planning/viz` (`visualization_msgs/msg/MarkerArray`) at 20 Hz
  - `pure_pursuit_sim.py`: standalone sim-only Pure Pursuit follower (`/planning/trajectory` -> `/cmd`) kept decoupled from planning

## Offline Tests & Latency Benchmark

    PYTHONPATH=khonshu_planner python3 khonshu_planner/tests/test_core.py

Baseline verified: `mean=0.94 ms`, `p95=1.48 ms`, `max=1.81 ms` (target `< 30 ms`).

## Build & Run in EUFS (ROS 2 Humble)

    mkdir -p ~/khonshu_ws/src
    ln -sfn ~/the-khonshu-algorithm/khonshu_planner ~/khonshu_ws/src/khonshu_planner
    cd ~/khonshu_ws
    source /opt/ros/humble/setup.bash
    source ~/eufs_ws/install/setup.bash
    colcon build --symlink-install

Terminal 1: `~/eufs_ws/run_sim.sh gazebo_gui:=false`

Terminal 2:

    source /opt/ros/humble/setup.bash
    source ~/eufs_ws/install/setup.bash
    source ~/khonshu_ws/install/setup.bash
    ros2 service call /ros_can/set_mission eufs_msgs/srv/SetCanState "{ami_state: 14}"
    ros2 launch khonshu_planner khonshu_sim.launch.py

Pass `with_controller:=false` to run only `planner_node` with an external controller.

## Verified Simulation Result (small_track)
- Live planning latency: `1.42–2.86 ms` at 20 Hz
- Completed autonomous `97.6 m` lap on `small_track` (`failsafe=False`)
