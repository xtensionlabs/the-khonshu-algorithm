# Path-planning references

This index keeps the team's source material in one place. Open a title to visit the original paper, documentation, or repository.

## System and engineering workflow

- [The Software Stack That Won the Formula Student Driverless Competition](https://arxiv.org/abs/2210.10933) — GraphSLAM, Delaunay lane-boundary detection, spline trajectories, and a G-G-aware minimum-curvature racing line.
- [Python `src/` layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) — package layout used for the Khonshu core.
- [pytest parametrized tests](https://docs.pytest.org/en/stable/how-to/parametrize.html) — repeatable scenarios for geometry, sensor loss, and planner failures.

## Track estimation and centerline

- [ROS REP-103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst) — coordinate conventions and SI units used by the shared types.
- [ROS 2 TF2 listener tutorial](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html) — transform cone detections at the correct timestamp in simulator and vehicle adapters.
- [SciPy Delaunay triangulation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html) — reference for the team's alternate triangulation implementation and degeneracy handling.
- [Lane Detection using Graph Search and Geometric Constraints for Formula Student Driverless](https://arxiv.org/abs/2405.16369) — deterministic CLC search for noisy cone boundaries; compare against the v0.1 baseline.
- [Formula Trinity path-planning theory](https://docs.formulatrinity.ie/departments/path_planning/theory/) — Arc Centre Cone Virtualisation for missing boundaries.
- [FaSTTUBe Formula Student path planner](https://github.com/papalotis/ft-fsd-path-planning) — MIT-licensed baseline vendored in `src/fsd_path_planning`; upstream identifies it as the 2021/22 and 2022/23 planner and describes unknown-color cone handling.

## Smoothing, speed, and racing line

- [SciPy `CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) — C2 spline interpolation and derivative API.
- [TUMFTM global race-trajectory optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) — minimum-curvature/minimum-time formulations, G-G diagrams, and velocity-profile references for later stages.
- [OSQP Python interface](https://osqp.org/docs/interfaces/python.html) — sparse quadratic-programming API for a future constrained racing-line optimizer.

## Mapping and path following

- [Graph-Based SLAM for Formula Student Autonomous Racing](https://leonce-m.github.io/projects/slam/) — cone data association, keyframes, map optimization, and loop closure context for global planning.
- [F1TENTH Lab 5 Pure Pursuit template](https://github.com/f1tenth/f1tenth_lab5_template) — path-following example for a lightweight secondary simulator workflow.

## Autonomous racing leagues

- [Indy Autonomous Challenge](https://www.indyautonomouschallenge.com/) — university-led high-speed autonomy competition; useful reference for real-time operation and validation across simulated and physical vehicles.
- [Abu Dhabi Autonomous Racing League (A2RL)](https://a2rl.io/) — high-speed autonomous car and drone competition; use its public competition material as a source of robustness and performance-testing ideas.
- [A2RL autonomous racing technology overview](https://a2rl.io/pdf/A2RL%20Race%20Tech.pdf) — organizer's overview of the race-car autonomy system.

## Simulator and ROS 2 integration

- [EUFS simulator](https://gitlab.com/eufs/public/eufs_sim) — primary ROS 2 and Gazebo integration target.
- [EUFS message definitions](https://gitlab.com/eufs/public/eufs_msgs) — simulator-native message types for the EUFS adapter.
- [Formula Trinity ROS path-planning tutorial](https://docs.formulatrinity.ie/tutorials/path_planning_tutorial/) — EUFS `/cones` and `ConeArrayWithCovariance` subscriber example plus RViz marker workflow; page is marked as partly deprecated, so verify against the installed EUFS release.
- [Formula Student Driverless Simulator (FSDS)](https://fs-driverless.github.io/Formula-Student-Driverless-Simulator/) — secondary simulator for cross-environment replay.
