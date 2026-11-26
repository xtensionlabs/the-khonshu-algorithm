import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from planner.fake_tracks import (
    straight_then_turn,
    centre_pose,
    dense_centreline,
    dist_to_curve,
)
from planner.pipeline import plan, OK

DENSE = dense_centreline()
S = np.arange(0.0, 36.0, 1.0)
CASES = [
    ("clean", 0.0, 0),
    ("noise 0.2", 0.2, 0),
    ("noise 0.4", 0.4, 0),
    ("drop 3", 0.0, 3),
    ("drop 8", 0.0, 8),
]

fig, ax = plt.subplots(figsize=(10, 5))
for name, noise, drop in CASES:
    cones, _ = straight_then_turn(noise=noise, drop=drop, seed=1)
    worst, fails = [], 0
    for s in S:
        x, y, yaw = centre_pose(s)
        r = plan(cones, (x, y), yaw)
        if r.status == OK:
            worst.append(dist_to_curve(r.path_map, DENSE).max())
        else:
            worst.append(np.nan)
            fails += 1
    ok = np.array(worst)[~np.isnan(worst)]
    print(
        f"{name:10s} failures {fails:2d}/{len(S)}   "
        f"mean worst-error {ok.mean():.3f} m   max {ok.max():.3f} m"
    )
    ax.plot(S, worst, "o-", ms=3, label=name)

ax.set_xlabel("car position along track (m)")
ax.set_ylabel("worst path error in that frame (m)")
ax.grid(True)
ax.legend()
fig.savefig("drive_along.png", dpi=120)
print("saved drive_along.png")
