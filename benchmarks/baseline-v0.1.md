# Khonshu v0.1 local baseline

- Run date: 2026-11-20
- Environment: Windows 11, Python 3.12.10, NumPy 2.4.6, SciPy 1.17.1
- Method: 50 cycles per deterministic scenario after `LocalPlanner.warm_up()`
- Target speed: 2.0 m/s
- Deadline: 30 ms
- Result: 298/300 valid trajectories; two left-turn cycles exceeded 30 ms and correctly returned stop requests. No cone-clearance stops occurred.

| Scenario | p50 (ms) | p95 (ms) | Max (ms) | Valid/50 | Over 30 ms |
|---|---:|---:|---:|---:|---:|
| Straight | 11.37 | 15.74 | 22.19 | 50 | 0 |
| Left turn | 12.75 | 18.17 | 74.49 | 48 | 2 |
| Right turn | 12.97 | 16.96 | 18.39 | 50 | 0 |
| Missing inside boundary | 10.25 | 13.48 | 17.95 | 50 | 0 |
| Unlabeled cones | 10.65 | 13.54 | 15.24 | 50 | 0 |
| False positives | 9.68 | 16.85 | 22.20 | 50 | 0 |

A 100-cycle isolated follow-up on the left-turn fixture measured p50 8.77 ms, p95 15.58 ms, max 18.89 ms, with no deadline misses. The two misses were not reproduced in that rerun; keep the earlier outliers in the baseline because they occurred in the full run.

These are warmed synthetic 2D results on one Windows development machine. The Numba cache was already primed during this run. The measurements include host scheduling pauses, and the sporadic left-turn spikes were not repeatable in the isolated run. The 30 ms guard converted the late calculations to stop requests, but this is not a hard real-time guarantee. The measurements do not include ROS 2 transport, TF transforms, EUFS/FSDS simulation, control, lap time, or cone-contact validation. Repeat on the team's Linux/ROS 2 target and record the branch revision before using these values for decisions.

Command used:

```powershell
python examples/benchmark_synthetic.py --iterations 50
python examples/benchmark_synthetic.py --scenario left_turn --iterations 100
```
