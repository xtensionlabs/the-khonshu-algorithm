# Khonshu Path Planner

Khonshu is the path-planning project for the Strathmore University autonomous Formula Student car. It turns perceived cone positions and vehicle state into a safe, smooth path for the car to follow.

The long-term goal is a ROS 2 planner that runs in the EUFS simulator and on the real car, with track estimation, smooth waypoint generation, and speed planning.

## Project setup

Requires Python 3.8 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

## Run the demo

The demo generates a synthetic track, plots the cones and centreline, and saves it as `demo_track.png`:

```bash
python scripts/demo_track.py
```

## Package

The `planner` package contains:

- `planner.cones`: cone and track observation data structures.
- `planner.fake_tracks`: generated test tracks for development.

Example:

```python
from planner.fake_tracks import straight_then_turn

cones, centreline = straight_then_turn()
```

## ROS 2 and EUFS simulation

This repository is currently being developed as a standalone Python planner. ROS 2 nodes and the EUFS Sim integration will connect cone detections and localisation inputs to planned trajectories and vehicle control.

The planning pipeline should be validated in EUFS Sim before deployment to the real car.
