"""Run a repeatable straight-track latency check without ROS or a simulator."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from khonshu import LocalPlanner, VehicleState
from khonshu.scenarios import synthetic_scenarios


def main() -> None:
    all_scenarios = synthetic_scenarios()
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--deadline-ms", type=float, default=30.0)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(all_scenarios),
        help="run only the selected scenario; repeat the option to select several",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.deadline_ms <= 0.0:
        parser.error("--deadline-ms must be greater than zero")

    scenario_reports = {}
    selected_names = args.scenario or list(all_scenarios)
    for scenario_name in selected_names:
        cones, initial_state = all_scenarios[scenario_name]
        planner = LocalPlanner()
        warmup_started = perf_counter()
        planner.warm_up()
        warmup_seconds = perf_counter() - warmup_started

        latencies_ms = []
        valid_count = 0
        stop_reasons: dict[str, int] = {}
        deadline_misses = []
        for index in range(args.iterations):
            state = VehicleState(
                x=initial_state.x,
                y=initial_state.y,
                yaw_rad=initial_state.yaw_rad,
                speed_mps=initial_state.speed_mps,
                timestamp_s=index / 20.0,
                frame_id=initial_state.frame_id,
            )
            trajectory = planner.plan(cones, state)
            latencies_ms.append(trajectory.compute_time_ms)
            if trajectory.compute_time_ms > args.deadline_ms:
                deadline_misses.append(
                    {"cycle": index, "latency_ms": trajectory.compute_time_ms}
                )
            valid_count += int(trajectory.valid)
            if trajectory.emergency_stop:
                stop_reasons[trajectory.reason] = stop_reasons.get(trajectory.reason, 0) + 1

        scenario_reports[scenario_name] = {
            "valid_cycles": valid_count,
            "stop_reasons": stop_reasons,
            "cycles_over_deadline": deadline_misses,
            "latency_ms": {
                "p50": float(np.percentile(latencies_ms, 50)),
                "p95": float(np.percentile(latencies_ms, 95)),
                "max": float(max(latencies_ms)),
            },
            "warmup_seconds": warmup_seconds,
        }

    report = {
        "iterations": args.iterations,
        "deadline_ms": args.deadline_ms,
        "scenarios": scenario_reports,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "target_speed_mps": 2.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
