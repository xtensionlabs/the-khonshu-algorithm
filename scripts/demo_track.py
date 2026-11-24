import matplotlib

matplotlib.use("Agg")  # save to file
import matplotlib.pyplot as plt
from planner.fake_tracks import straight_then_turn

cones, centre = straight_then_turn()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(*cones.blue.T, color="tab:blue", label="blue")
ax.scatter(*cones.yellow.T, color="gold", label="yellow")
ax.plot(*centre.T, "k--", label="true centreline")
ax.set_aspect("equal")
ax.legend()
fig.savefig("demo_track.png", dpi=120)
print("saved demo_track.png")
