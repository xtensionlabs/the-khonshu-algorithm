import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from planner.fake_tracks import straight_then_turn
from planner.triangulation import blue_yellow_edges
from planner.centerline import build_centerline
from planner.bezier import smooth_path, spline_smooth

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, noise in zip(axes, (0.0, 0.2)):
    cones, centre = straight_then_turn(noise=noise)
    raw = build_centerline(blue_yellow_edges(cones))

    ax.scatter(*cones.blue.T, color="tab:blue", s=15)
    ax.scatter(*cones.yellow.T, color="gold", s=15)
    ax.plot(*centre.T, "k--", lw=1, label="true centreline")
    ax.plot(*raw.T, "o:", color="gray", ms=5, label="raw midpoints")
    ax.plot(*smooth_path(raw).T, "r-", lw=2, label="Bezier (through points)")
    ax.plot(
        *spline_smooth(raw, smoothing=len(raw) * noise**2 / 2 + 1e-3).T,
        "g-",
        lw=2,
        label="smoothing spline",
    )
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_title(f"cone noise = {noise} m")
fig.savefig("demo_bezier.png", dpi=120)
print("saved demo_bezier.png")
