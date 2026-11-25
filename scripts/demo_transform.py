import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from planner.fake_tracks import straight_then_turn
from planner.transforms import cones_to_car, filter_ahead

cones, centre = straight_then_turn()

# Pretend the car is partway along the track, at some arbitrary pose
car_xy, car_yaw = (20.0, 1.0), 0.5

car_cones = cones_to_car(cones, car_xy, car_yaw)
near = filter_ahead(car_cones)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, c, title in zip(axes, (cones, near), ("Map frame", "Car frame, filtered")):
    ax.scatter(*c.blue.T, color="tab:blue")
    ax.scatter(*c.yellow.T, color="gold")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)
axes[0].plot(*car_xy, "r^", markersize=12, label="car")
axes[0].legend()
axes[1].plot(0, 0, "r^", markersize=12)
fig.savefig("demo_transform.png", dpi=120)
