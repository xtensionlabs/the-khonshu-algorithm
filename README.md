# Khonshu — Path Planning

Path-planning algorithms for the autonomous Formula Student car developed by **Strathmore Racing Club | FS-AI DDT**.

## Purpose

Khonshu is the planning layer between perception/localisation and vehicle control. It transforms detected cone positions and vehicle state into a safe, smooth, and drivable trajectory for the car to follow.

The planner is responsible for:

- Estimating the track and generating a usable centreline from cone observations.
- Producing smooth waypoints and a feasible target-speed profile.
- Supporting reliable autonomous driving when the track is only partially observed.
- Evolving from a robust exploratory centreline into an optimised racing line as more track information becomes available.

## Initial Focus

The first milestone is a dependable baseline planner that can complete autonomous laps in simulation. Development will initially focus on:

1. Track and centreline estimation.
2. Trajectory smoothing.
3. Basic velocity profiling.
4. ROS 2 integration and visualisation.

The implementation will be tested independently in simulation before being connected to the full perception, localisation, and control stack.

## Status

Early development. This repository will grow to include the planning algorithms, simulation interfaces, visualisation tools, benchmarks, and documentation required to validate the autonomous system.

## Team

Built by the Path Planning sub-team of **Strathmore Racing Club | FS-AI DDT** for Formula Student autonomous events.
