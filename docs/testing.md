# Stage A validation

## Deterministic 2D checks

Run the repeatable geometry cases without ROS or Gazebo:

```powershell
python -m pytest tests/test_stage_a.py -q
python examples/visualize_synthetic.py --planner stage_a --scenario left_turn
python examples/visualize_synthetic.py --planner stage_a --scenario false_positives
python examples/benchmark_synthetic.py --planner stage_a --iterations 50
```

The six shared cases cover straight, left/right turns, a missing boundary, unknown cone colors, and false positives. `test_stage_a.py` also checks world-frame transforms, vehicle-footprint clearance around four displaced orange corner cones, graph evidence, and explicit stop behavior. To compare the retained FaSTTUBe baseline, use `--planner fasttube` for the renderer and benchmark.

## EUFS lap in WSL

The WSL environment needs ROS 2 with Gazebo/RViz GUI support (WSLg or an X server), the built EUFS workspace, and a Python environment with the project and ROS message dependencies. The launcher uses these defaults:

- ROS setup script: `~/fsds_ws/scripts/ros_env.sh`
- EUFS workspace: `~/eufs_ws`
- Python environment: `~/.venvs/khonshu`

To use different locations, set `KHONSHU_ROS_ENV_SCRIPT`, `KHONSHU_EUFS_WS`, and `KHONSHU_PYTHON_ENV` before launching. The EUFS workspace must contain `src/eufs_sim/eufs_launcher/launch/simulation.launch.py`. The script checks these paths and reports any missing setup before starting. From the root of the repository clone in WSL, run:

```bash
cd "/path/to/the-khonshu-algorithm"
bash ros2/khonshu_eufs/run_eufs_stage_a_lap.sh
```

The helper opens the rectangle course in Gazebo, starts the Stage A centerline node, and follows its path with a low-speed pure-pursuit node. It opens both Gazebo and RViz windows by default. RViz displays the live planner construction over the track. Its marker namespaces are `stage_a_delaunay` (blue triangulation edges), `stage_a_voronoi` (purple ridges), `stage_a_cross_sections` (orange boundary pairs), `stage_a_midpoints` (yellow section midpoints), `stage_a_candidates` (cyan candidate nodes), `stage_a_candidate_graph` (gray graph edges), and `stage_a_dijkstra_route` (pink selected route). The green line is the published centerline. The visualization is in the RViz window beside Gazebo, rather than drawn inside the Gazebo scene. Set `KHONSHU_SHOW_RVIZ=0` before the command to disable the second window.

The simulator is configured for velocity commands and the follower sends `ackermann_msgs/AckermannDriveStamped` to `/cmd`. The follower uses a 1.0 m lookahead to track the centerline more closely through corners, and stops on an empty or stale path, stale odometry, or if the vehicle reaches its lap timeout. Stage A uses non-orange cones for Delaunay and cross-section boundary geometry; orange cones remain in the collision checks as interior obstacles. On completing one return to the start region, the follower prints `LAP_COMPLETE`, leaves Gazebo visible for 10 seconds, then cleans up the launched processes. Set `KHONSHU_POST_LAP_HOLD_S=0` to disable the hold. The helper allows 200 ms for each EUFS planning cycle because the WSL GUI load is higher than the standalone 30 ms planning target.

Useful live topics:

- `/khonshu/debug/status` — plan result, point count, compute time, Delaunay/Voronoi/graph counts, and route length in nodes.
- `/khonshu/control/status` — follower state, stop reason, travelled distance, and lap result.
- `/khonshu/centerline` — map-frame `nav_msgs/Path` consumed by the follower.
- `/ground_truth/odom` and `/ground_truth/cones` — EUFS development inputs.

A clean lap is evidence that this Stage A implementation can close the loop in the configured EUFS environment. Record lap time, cone contacts, planner p50/p95/max latency, and stop events before comparing branches. This result does not establish FSDS or real-car compatibility; those adapters and the vehicle-level safety checks are separate work.

Validated on the EUFS rectangle at a commanded 0.8 m/s: `LAP_COMPLETE distance_m=88.54 elapsed_s=116.90`. The run made one brief `waiting_for_valid_path` hold near 8 m, then continued around the full course. Keep that hold in the benchmark record; the final lap result alone does not establish zero interruptions or zero cone contacts.

After excluding orange cones from boundary geometry and shortening the follower lookahead to 1.0 m, a visible EUFS run with RViz construction markers completed: `LAP_COMPLETE distance_m=93.42 elapsed_s=124.10`. The planner reported 20 boundary cones and 4 orange obstacles at startup. It continued through all four corners and returned to the start; occasional short path-freshness holds recovered when the next path arrived. Planner cycle times stayed within the helper's 200 ms limit in the sampled status logs. The run did not have an automated cone-contact counter, so verify contacts from the Gazebo view or add collision telemetry before recording a zero-contact benchmark.
