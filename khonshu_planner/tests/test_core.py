"""Offline unit tests, latency benchmark, and visual plot for Khonshu core planner."""

from __future__ import annotations

import os
import numpy as np

from khonshu_planner.core.planner import KhonshuPlanner
from khonshu_planner.core.types import Cone, ConeColor


def make_test_cones(noise: float = 0.0, drop: int = 0, seed: int = 0) -> list[Cone]:
    rng = np.random.default_rng(seed)
    cones: list[Cone] = []
    # Straight section (0..15 m)
    for x in np.arange(0.0, 15.0, 3.0):
        cones.append(Cone(float(x + rng.normal(0, noise)), float(1.8 + rng.normal(0, noise)), ConeColor.BLUE))
        cones.append(Cone(float(x + rng.normal(0, noise)), float(-1.8 + rng.normal(0, noise)), ConeColor.YELLOW))
    # Tight 90-degree left turn (radius 7 m)
    for a in np.linspace(0.0, np.pi / 2.0, 8)[1:]:
        cx, cy = 15.0 + 7.0 * np.sin(a), 7.0 - 7.0 * np.cos(a)
        nx, ny = -np.sin(a), np.cos(a)
        cones.append(Cone(float(cx + 1.8 * nx + rng.normal(0, noise)), float(cy + 1.8 * ny + rng.normal(0, noise)), ConeColor.BLUE))
        cones.append(Cone(float(cx - 1.8 * nx + rng.normal(0, noise)), float(cy - 1.8 * ny + rng.normal(0, noise)), ConeColor.YELLOW))
    if drop > 0 and drop < len(cones):
        keep = sorted(rng.choice(len(cones), size=len(cones) - drop, replace=False))
        cones = [cones[i] for i in keep]
    return cones


def run_tests() -> None:
    planner = KhonshuPlanner()

    # 1. Clean track
    clean_res = planner.plan(make_test_cones(noise=0.0, drop=0), current_speed=4.5)
    assert not clean_res.trajectory.is_failsafe
    assert len(clean_res.trajectory.waypoints) >= 20

    # 2. Noisy + missing cones
    noisy_res = planner.plan(make_test_cones(noise=0.4, drop=3), current_speed=4.5)
    assert not noisy_res.trajectory.is_failsafe
    assert len(noisy_res.trajectory.waypoints) >= 10

    # 3. Failsafe on empty / sparse cones
    for sparse in ([], [Cone(3.0, 1.5, ConeColor.BLUE), Cone(3.0, -1.5, ConeColor.YELLOW)]):
        fs_res = planner.plan(sparse, current_speed=4.0)
        assert fs_res.trajectory.is_failsafe
        assert fs_res.trajectory.waypoints[0].speed == 0.0

    # 4. 50-cycle latency benchmark (< 30 ms requirement)
    cones = make_test_cones(noise=0.3, drop=2)
    times = [planner.plan(cones, current_speed=4.5).trajectory.compute_time_ms for _ in range(50)]
    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    max_ms = float(np.max(times))
    assert max_ms < 30.0, f"Exceeded 30 ms budget: max={max_ms:.2f} ms"

    print(f"ALL CORE TESTS PASSED | mean={mean_ms:.2f} ms, p95={p95_ms:.2f} ms, max={max_ms:.2f} ms")


def save_preview(out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planner = KhonshuPlanner()
    cases = [
        ("Clean track", make_test_cones(noise=0.0, drop=0)),
        ("Noisy (0.4 m) + 3 missing", make_test_cones(noise=0.4, drop=3)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (title, cones) in zip(axes, cases):
        res = planner.plan(cones, current_speed=4.5)
        for a, b, c in res.track_estimate.triangles:
            xs = [a[0], b[0], c[0], a[0]]
            ys = [a[1], b[1], c[1], a[1]]
            ax.plot(xs, ys, color="#d0d0d0", lw=0.6, zorder=1)
        for p1, p2 in res.track_estimate.cross_edges:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#aaaaaa", lw=0.9, ls="--", zorder=2)

        bx = [c.x for c in cones if c.color == ConeColor.BLUE]
        by = [c.y for c in cones if c.color == ConeColor.BLUE]
        yx = [c.x for c in cones if c.color == ConeColor.YELLOW]
        yy = [c.y for c in cones if c.color == ConeColor.YELLOW]
        ax.scatter(bx, by, c="tab:blue", s=45, label="Blue", zorder=4)
        ax.scatter(yx, yy, c="gold", edgecolors="black", s=45, label="Yellow", zorder=4)

        if res.track_estimate.midpoints:
            mx, my = zip(*res.track_estimate.midpoints)
            ax.scatter(mx, my, c="black", marker="x", s=30, label="Midpoints", zorder=5)

        wps = res.trajectory.waypoints
        wx = [w.x for w in wps]
        wy = [w.y for w in wps]
        wv = [w.speed for w in wps]
        sc = ax.scatter(wx, wy, c=wv, cmap="RdYlGn", vmin=1.5, vmax=5.0, s=28, zorder=6)
        fig.colorbar(sc, ax=ax, label="Target speed (m/s)")

        ax.set_title(f"{title} ({res.trajectory.compute_time_ms:.2f} ms)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"Saved preview plot to {out_path}")


if __name__ == "__main__":
    run_tests()
    out = "/mnt/c/Users/iobar/FSAI/khonshu_core_preview.png"
    if os.path.isdir(os.path.dirname(out)):
        save_preview(out)
