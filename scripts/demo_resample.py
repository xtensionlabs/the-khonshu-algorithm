import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from planner.fake_tracks import straight_then_turn
from planner.triangulation import blue_yellow_edges
from planner.centerline import build_centerline
from planner.bezier import smooth_path, spline_smooth
from planner.resample import resample, curvature, speed_profile, arc_length

SP = 0.5
cones, centre = straight_then_turn(noise=0.1)
raw = build_centerline(blue_yellow_edges(cones))

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for ax, (name, dense) in zip(
    axes[:2],
    (
        ("Bezier", smooth_path(raw)),
        ("spline", spline_smooth(raw, smoothing=len(raw) * 0.1**2, n=300)),
    ),
):
    path = resample(dense, SP)
    k = curvature(path, SP)
    sc = ax.scatter(*path.T, c=k, cmap="coolwarm", s=18, vmin=-0.2, vmax=0.2)
    ax.plot(*centre.T, "k--", lw=0.8)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(f"{name}: coloured by curvature")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="curvature (1/m)")
    axes[2].plot(arc_length(path), k, label=f"{name} curvature")

axes[2].axhline(0.1, color="k", ls="--", lw=0.8, label="true 1/R = 0.1")
axes[2].set_xlabel("distance along path (m)")
axes[2].legend()
axes[2].grid(True)
fig.savefig("demo_resample.png", dpi=120)
print("saved demo_resample.png")
