import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from planner.fake_tracks import straight_then_turn
from planner.triangulation import triangulate, blue_yellow_edges

cones, centre = straight_then_turn(noise=0.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, max_edge in zip(axes, (4.0, 5.0)):
    pts, tri = triangulate(cones)
    ax.triplot(pts[:, 0], pts[:, 1], tri.simplices, color="lightgray", lw=0.6)
    ax.scatter(*cones.blue.T, color="tab:blue")
    ax.scatter(*cones.yellow.T, color="gold")

    e = blue_yellow_edges(cones, max_edge=max_edge)
    for bi, yi in zip(e.blue_idx, e.yellow_idx):
        p, q = cones.blue[bi], cones.yellow[yi]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="tab:orange", lw=2)
    ax.plot(*e.midpoints.T, "r.", markersize=8)
    ax.plot(*centre.T, "k--", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"max_edge = {max_edge}  ({len(e.midpoints)} edges)")
fig.savefig("demo_triangulation.png", dpi=120)
print("saved demo_triangulation.png")
