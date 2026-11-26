"""Render a full synthetic circuit with Stage A Delaunay and path output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from khonshu import StageAConfig, StageAPlanner
from khonshu.scenarios import synthetic_full_circuit


MAX_DISPLAY_EDGE_LENGTH_M = 8.0


def _local_to_world(
    xy: tuple[float, float], state_x: float, state_y: float, yaw_rad: float
) -> tuple[float, float]:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        state_x + cosine * xy[0] - sine * xy[1],
        state_y + sine * xy[0] + cosine * xy[1],
    )


def _write_interactive_html(
    output: Path,
    cones: tuple,
    edge_map: dict,
    centerline_segments: list[list[tuple[float, float]]],
    start_pose: object,
    successful_plans: int,
    pose_count: int,
) -> None:
    """Write a dependency-free canvas viewer with orbit and zoom controls."""

    x_values = [cone.x for cone in cones]
    y_values = [cone.y for cone in cones]
    payload = {
        "bounds": [min(x_values), max(x_values), min(y_values), max(y_values)],
        "cones": [
            {
                "x": cone.x,
                "y": cone.y,
                "z": 0.32,
                "color": "#2563eb" if cone.color.value == "blue" else "#eab308",
            }
            for cone in cones
        ],
        "edges": [
            [[first[0], first[1], 0.0], [second[0], second[1], 0.0]]
            for first, second in edge_map.values()
        ],
        "centerlineSegments": [
            [[x, y, 0.0] for x, y in segment]
            for segment in centerline_segments
        ],
        "start": [start_pose.x, start_pose.y, 0.36],
        "summary": {
            "cones": len(cones),
            "validPlans": successful_plans,
            "poseCount": pose_count,
            "edges": len(edge_map),
            "edgeLimit": MAX_DISPLAY_EDGE_LENGTH_M,
        },
    }
    template = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Khonshu Stage A — Full Circuit 3D</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f8fafc; color: #0f172a; }
    header { padding: 18px 24px 10px; }
    h1 { margin: 0; font-size: clamp(20px, 3vw, 30px); }
    p { margin: 7px 0 0; color: #475569; }
    #summary { font-weight: 600; color: #334155; }
    #view { display: block; width: 100%; height: min(78vh, 900px); min-height: 480px; touch-action: none; cursor: grab; }
    #view:active { cursor: grabbing; }
    footer { display: flex; flex-wrap: wrap; gap: 14px 24px; padding: 10px 24px 20px; color: #475569; font-size: 14px; }
    .key { display: inline-flex; align-items: center; gap: 8px; }
    .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
    .line { width: 24px; border-top: 3px solid; display: inline-block; }
    .blue { background: #2563eb; } .yellow { background: #eab308; }
    .mesh { border-color: #7c3aed; } .path { border-color: #16a34a; }
    .start { color: #111827; font-size: 18px; line-height: 10px; }
  </style>
</head>
<body>
  <header>
    <h1>Khonshu Stage A — full synthetic circuit</h1>
    <p id="summary"></p>
  </header>
  <canvas id="view" aria-label="Interactive 3D view of the synthetic cone circuit"></canvas>
  <footer>
    <span class="key"><i class="dot blue"></i>Blue cones</span>
    <span class="key"><i class="dot yellow"></i>Yellow cones</span>
    <span class="key"><i class="line mesh"></i>Local Delaunay edges</span>
    <span class="key"><i class="line path"></i>Validated Stage A path segments</span>
    <span class="key"><b class="start">★</b>Start pose</span>
    <span>Drag to rotate · Scroll to zoom · Track and triangulation use the XY plane</span>
  </footer>
  <script id="scene-data" type="application/json">__SCENE_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById('scene-data').textContent);
    const canvas = document.getElementById('view');
    const ctx = canvas.getContext('2d');
    const [xmin, xmax, ymin, ymax] = data.bounds;
    const center = [(xmin + xmax) / 2, (ymin + ymax) / 2];
    const span = Math.max(xmax - xmin, ymax - ymin);
    document.getElementById('summary').textContent =
      `${data.summary.cones} dummy cones · ${data.summary.validPlans}/${data.summary.poseCount} valid local plans · ` +
      `${data.summary.edges} local Delaunay edges (≤${data.summary.edgeLimit} m)`;
    let yaw = -0.9, pitch = 0.72, zoom = 1.0, drag = null;

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function project(point) {
      const dx = point[0] - center[0], dy = point[1] - center[1], dz = point[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const horizontal = cy * dx - sy * dy;
      const depthAxis = sy * dx + cy * dy;
      const vertical = cp * depthAxis - sp * dz;
      const depth = sp * depthAxis + cp * dz;
      const rect = canvas.getBoundingClientRect();
      const focal = Math.min(rect.width, rect.height) * 2.25 * zoom;
      const perspective = focal / Math.max(span * 2.8 - depth, span * 0.8);
      return { x: rect.width / 2 + horizontal * perspective,
               y: rect.height / 2 - vertical * perspective };
    }

    function drawStar(x, y, radius) {
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const angle = -Math.PI / 2 + i * Math.PI / 5;
        const r = i % 2 === 0 ? radius : radius * 0.45;
        const px = x + Math.cos(angle) * r, py = y + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath(); ctx.fillStyle = '#111827'; ctx.fill();
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.stroke();
    }

    function draw() {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, 0, rect.width, rect.height);

      ctx.strokeStyle = 'rgba(124, 58, 237, 0.40)'; ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (const [a, b] of data.edges) {
        const pa = project(a), pb = project(b);
        ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y);
      }
      ctx.stroke();

      ctx.strokeStyle = 'rgba(22, 163, 74, 0.58)'; ctx.lineWidth = 3.2;
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      for (const segment of data.centerlineSegments) {
        ctx.beginPath();
        segment.forEach((point, index) => {
          const p = project(point);
          if (index === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
      }

      for (const cone of data.cones) {
        const base = project([cone.x, cone.y, 0]);
        const tip = project([cone.x, cone.y, cone.z]);
        const height = Math.max(8, Math.hypot(tip.x - base.x, tip.y - base.y) * 2.0);
        ctx.beginPath();
        ctx.moveTo(tip.x, tip.y);
        ctx.lineTo(base.x + height * 0.42, base.y);
        ctx.lineTo(base.x - height * 0.42, base.y);
        ctx.closePath(); ctx.fillStyle = cone.color; ctx.fill();
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1; ctx.stroke();
      }
      const start = project(data.start); drawStar(start.x, start.y, 9);
    }

    canvas.addEventListener('pointerdown', event => {
      drag = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!drag) return;
      yaw += (event.clientX - drag.x) * 0.009;
      pitch = Math.max(0.08, Math.min(1.48, pitch + (event.clientY - drag.y) * 0.006));
      drag = { x: event.clientX, y: event.clientY }; draw();
    });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('pointercancel', () => { drag = null; });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      zoom = Math.max(0.55, Math.min(2.4, zoom * (event.deltaY > 0 ? 0.9 : 1.1)));
      draw();
    }, { passive: false });
    new ResizeObserver(resize).observe(canvas);
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        template.replace(
            "__SCENE_DATA__", json.dumps(payload, separators=(",", ":"))
        ),
        encoding="utf-8",
    )


def render_full_circuit(output: Path, html_output: Path) -> tuple[int, int, int]:
    """Run Stage A around the loop and save a 3D circuit overview."""

    cones, poses, reference_centerline = synthetic_full_circuit()
    planner = StageAPlanner(
        StageAConfig(
            maximum_cycle_time_ms=5000.0,
            maximum_observation_range_m=18.0,
        )
    )
    edge_map: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[float, float], tuple[float, float]],
    ] = {}
    planned_segments: list[list[tuple[float, float]]] = []
    successful_plans = 0
    maximum_reference_offset_m = 0.0

    for pose in poses:
        trajectory = planner.plan(cones, pose)
        if trajectory.valid:
            successful_plans += 1
            near_points = [
                (point.x, point.y)
                for point in trajectory.points
                if math.hypot(point.x - pose.x, point.y - pose.y) <= 9.0
            ]
            if len(near_points) >= 2:
                planned_segments.append(near_points)
            for point_x, point_y in near_points:
                reference_offset = min(
                    math.hypot(point_x - ref_x, point_y - ref_y)
                    for ref_x, ref_y in reference_centerline
                )
                maximum_reference_offset_m = max(
                    maximum_reference_offset_m, reference_offset
                )

        for first_local, second_local in planner.last_debug_geometry.get(
            "delaunay_edges_local", []
        ):
            first = _local_to_world(first_local, pose.x, pose.y, pose.yaw_rad)
            second = _local_to_world(second_local, pose.x, pose.y, pose.yaw_rad)
            if math.hypot(second[0] - first[0], second[1] - first[1]) > MAX_DISPLAY_EDGE_LENGTH_M:
                continue
            first_key = (round(first[0] * 100), round(first[1] * 100))
            second_key = (round(second[0] * 100), round(second[1] * 100))
            if second_key < first_key:
                first, second = second, first
                first_key, second_key = second_key, first_key
            edge_map[(first_key, second_key)] = (first, second)

    if not planned_segments:
        raise RuntimeError("Stage A produced no valid local paths on the synthetic circuit")

    figure = plt.figure(figsize=(14, 10), facecolor="#f8fafc")
    axes = figure.add_subplot(111, projection="3d", facecolor="#f8fafc")

    for first, second in edge_map.values():
        axes.plot(
            (first[0], second[0]),
            (first[1], second[1]),
            (0.0, 0.0),
            color="#7c3aed",
            linewidth=0.8,
            alpha=0.32,
            zorder=1,
        )

    for segment in planned_segments:
        axes.plot(
            [point[0] for point in segment],
            [point[1] for point in segment],
            [0.0] * len(segment),
            color="#16a34a",
            linewidth=2.0,
            alpha=0.4,
            zorder=4,
        )

    for color_name, color in (("blue", "#2563eb"), ("yellow", "#eab308")):
        color_cones = [cone for cone in cones if cone.color.value == color_name]
        cone_faces = []
        for cone in color_cones:
            base = [
                (
                    cone.x + 0.15 * math.cos(2.0 * math.pi * side / 8),
                    cone.y + 0.15 * math.sin(2.0 * math.pi * side / 8),
                    0.0,
                )
                for side in range(8)
            ]
            tip = (cone.x, cone.y, 0.32)
            cone_faces.extend(
                [base[side], base[(side + 1) % 8], tip]
                for side in range(8)
            )
        axes.add_collection3d(
            Poly3DCollection(
                cone_faces,
                facecolors=color,
                edgecolors="white",
                linewidths=0.15,
                alpha=0.92,
            )
        )

    start_pose = poses[0]
    axes.scatter(
        [start_pose.x],
        [start_pose.y],
        [0.06],
        color="#111827",
        marker="*",
        s=140,
        depthshade=False,
        zorder=6,
    )

    x_values = [cone.x for cone in cones]
    y_values = [cone.y for cone in cones]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    padding = 4.0
    axes.set_xlim(min(x_values) - padding, max(x_values) + padding)
    axes.set_ylim(min(y_values) - padding, max(y_values) + padding)
    axes.set_zlim(0.0, 0.6)
    axes.set_box_aspect((x_span + 2 * padding, y_span + 2 * padding, 8.0))
    axes.view_init(elev=48, azim=-62)
    axes.set_proj_type("persp")
    axes.set_xlabel("x (m)", labelpad=9)
    axes.set_ylabel("y (m)", labelpad=9)
    axes.set_zlabel("z (m; height exaggerated)", labelpad=6)
    axes.grid(False)
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#f8fafc")
    axes.set_title(
        "Khonshu Stage A — full synthetic circuit\n"
        f"{len(cones)} dummy cones · {successful_plans}/{len(poses)} valid local plans · "
        f"{len(edge_map)} local Delaunay edges (≤{MAX_DISPLAY_EDGE_LENGTH_M:g} m)",
        loc="left",
        pad=18,
        fontsize=15,
        fontweight="bold",
    )

    legend = (
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#2563eb", label="blue cones", markersize=8),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#eab308", label="yellow cones", markersize=8),
        Line2D(
            [0],
            [0],
            color="#7c3aed",
            linewidth=1.3,
            label=f"local Delaunay edges (≤{MAX_DISPLAY_EDGE_LENGTH_M:g} m)",
        ),
        Line2D([0], [0], color="#16a34a", linewidth=2.5, label="validated Stage A path segments"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111827", label="start pose", markersize=11),
    )
    axes.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=True)
    figure.text(
        0.02,
        0.015,
        "Track and triangulation are computed in the XY plane; cone bases lie at z=0 and tips at 0.32 m. "
        "Only locally sized Delaunay edges are drawn to keep the circuit legible. "
        f"Each green segment is an unmodified, safety-validated local planner output. "
        f"Maximum distance from sampled nominal centreline: {maximum_reference_offset_m:.2f} m "
        "(nominal track half-width: 2.00 m).",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    _write_interactive_html(
        html_output,
        cones,
        edge_map,
        planned_segments,
        start_pose,
        successful_plans,
        len(poses),
    )
    return len(cones), successful_plans, len(edge_map)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts") / "full-circuit-3d.png",
        help="output image path",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=Path("artifacts") / "full-circuit-3d.html",
        help="interactive HTML output path",
    )
    args = parser.parse_args()
    cone_count, valid_count, edge_count = render_full_circuit(args.out, args.html_out)
    _, poses, _ = synthetic_full_circuit()
    print(f"Wrote {args.out.resolve()}")
    print(f"Wrote {args.html_out.resolve()}")
    print(
        f"cones={cone_count} valid_local_plans={valid_count}/{len(poses)} "
        f"delaunay_edges={edge_count}"
    )


if __name__ == "__main__":
    main()
