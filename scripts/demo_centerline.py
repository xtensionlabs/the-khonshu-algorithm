import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from planner.fake_tracks import straight_then_turn
from planner.transforms import cones_to_car, filter_ahead
from planner.triangulation import blue_yellow_edges
from planner.centerline import order_by_x, build_centerline


def draw(ax, cones, path, title):
    ax.scatter(*cones.blue.T, color="tab:blue", s=25)
    ax.scatter(*cones.yellow.T, color="gold", s=25)
    ax.plot(*path.T, "r.-")
    for i, (x, y) in enumerate(path):
        ax.annotate(str(i), (x, y), fontsize=7, color="darkred")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)


cones, centre = straight_then_turn(noise=0.0)
edges = blue_yellow_edges(cones)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
draw(axes[0], cones, order_by_x(edges.midpoints), "Sort by x (breaks in the turn)")
draw(axes[1], cones, build_centerline(edges), "Greedy hop + direction check")

# car partway round the bend, planning in its own frame
a = 0.9
car_xy = (18 + 10 * np.sin(a), 10 * (1 - np.cos(a)))
near = filter_ahead(cones_to_car(cones, car_xy, a))
draw(
    axes[2],
    near,
    build_centerline(blue_yellow_edges(near)),
    "Car in the bend (car frame)",
)
fig.savefig("demo_centerline.png", dpi=120)
print("saved demo_centerline.png")
