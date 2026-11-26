"""Render one deterministic 2D planner result as a dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

from khonshu import (
    LocalPlanner,
    PlannerConfig,
    StageAConfig,
    StageAPlanner,
    VehicleState,
)
from khonshu.scenarios import synthetic_scenarios


WIDTH = 1000
HEIGHT = 700
MARGIN = 70


def _svg_xy(x: float, y: float, bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    x_min, x_max, y_min, y_max = bounds
    scale = min(
        (WIDTH - 2 * MARGIN) / max(x_max - x_min, 1e-6),
        (HEIGHT - 2 * MARGIN) / max(y_max - y_min, 1e-6),
    )
    return MARGIN + (x - x_min) * scale, HEIGHT - MARGIN - (y - y_min) * scale


def _local_to_world(
    xy: tuple[float, float], state: VehicleState
) -> tuple[float, float]:
    """Convert a planner debug point from vehicle coordinates to map coordinates."""

    cosine = math.cos(state.yaw_rad)
    sine = math.sin(state.yaw_rad)
    return (
        state.x + cosine * xy[0] - sine * xy[1],
        state.y + sine * xy[0] + cosine * xy[1],
    )


def make_svg(scenario_name: str, planner_name: str = "stage_a") -> tuple[str, str]:
    cones, state = synthetic_scenarios()[scenario_name]
    if planner_name == "fasttube":
        planner = LocalPlanner(PlannerConfig(maximum_cycle_time_ms=5000.0))
        planner.warm_up()
    else:
        planner = StageAPlanner(StageAConfig(maximum_cycle_time_ms=5000.0))
    trajectory = planner.plan(cones, state)
    debug_geometry = getattr(planner, "last_debug_geometry", {})
    delaunay_edges = [
        (
            _local_to_world(first, state),
            _local_to_world(second, state),
        )
        for first, second in debug_geometry.get("delaunay_edges_local", [])
    ]
    delaunay_enabled = planner_name == "stage_a"
    title_suffix = (
        " — Delaunay mesh and centreline"
        if delaunay_enabled
        else " — centreline comparison"
    )

    all_x = [cone.x for cone in cones] + [state.x]
    all_y = [cone.y for cone in cones] + [state.y]
    if trajectory.points:
        all_x.extend(point.x for point in trajectory.points)
        all_y.extend(point.y for point in trajectory.points)
    padding = 1.5
    bounds = (
        min(all_x) - padding,
        max(all_x) + padding,
        min(all_y) - padding,
        max(all_y) + padding,
    )

    colors = {
        "blue": "#2563eb",
        "yellow": "#eab308",
        "orange_small": "#f97316",
        "orange_big": "#dc2626",
        "unknown": "#6b7280",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{MARGIN}" y="34" font-family="sans-serif" font-size="21" font-weight="700">Khonshu 2D {html.escape(planner_name)}: {html.escape(scenario_name)}{title_suffix}</text>',
    ]

    for first, second in delaunay_edges:
        x1, y1 = _svg_xy(first[0], first[1], bounds)
        x2, y2 = _svg_xy(second[0], second[1], bounds)
        elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#7c3aed" stroke-width="1.6" stroke-opacity="0.72"/>'
        )

    for cone in cones:
        px, py = _svg_xy(cone.x, cone.y, bounds)
        color = colors[cone.color.value]
        elements.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{color}" stroke="white" stroke-width="1.5"/>')

    if trajectory.points:
        path_points = " ".join(
            f"{_svg_xy(point.x, point.y, bounds)[0]:.1f},{_svg_xy(point.x, point.y, bounds)[1]:.1f}"
            for point in trajectory.points
        )
        elements.append(f'<polyline points="{path_points}" fill="none" stroke="#16a34a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
        outcome = (
            f"valid centerline: {len(trajectory.points)} points"
            + (f"; Delaunay edges: {len(delaunay_edges)}" if delaunay_enabled else "")
        )
        outcome_color = "#15803d"
    else:
        outcome = (
            f"STOP: {html.escape(trajectory.reason)}"
            + (f"; Delaunay edges: {len(delaunay_edges)}" if delaunay_enabled else "")
        )
        outcome_color = "#b91c1c"

    sx, sy = _svg_xy(state.x, state.y, bounds)
    elements.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="9" fill="#111827" stroke="white" stroke-width="2"/>')
    elements.append(f'<text x="{MARGIN}" y="{HEIGHT - 24}" font-family="sans-serif" font-size="16" fill="{outcome_color}">{outcome}; {trajectory.compute_time_ms:.2f} ms</text>')

    legend_y = 66
    legend_items = [
        ("blue cone", "#2563eb", "circle"),
        ("yellow cone", "#eab308", "circle"),
        ("orange cone", "#f97316", "circle"),
        ("unknown cone", "#6b7280", "circle"),
    ]
    if delaunay_enabled:
        legend_items.append(("Delaunay edge", "#7c3aed", "line"))
    legend_items.append(("centreline", "#16a34a", "line"))
    for index, (label, color, shape) in enumerate(legend_items):
        lx = MARGIN + index * 150
        if shape == "circle":
            elements.append(f'<circle cx="{lx}" cy="{legend_y}" r="5" fill="{color}"/>')
        else:
            width = 4 if label == "centreline" else 2
            elements.append(
                f'<line x1="{lx - 8}" y1="{legend_y}" x2="{lx + 8}" y2="{legend_y}" stroke="{color}" stroke-width="{width}"/>'
            )
        elements.append(f'<text x="{lx + 12}" y="{legend_y + 5}" font-family="sans-serif" font-size="13">{label}</text>')
    elements.append("</svg>")
    return "\n".join(elements), trajectory.reason


def main() -> None:
    choices = tuple(synthetic_scenarios())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=choices, default="left_turn")
    parser.add_argument("--planner", choices=("stage_a", "fasttube"), default="stage_a")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    output = args.out or Path("artifacts") / f"centerline-{args.planner}-{args.scenario}.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    svg, reason = make_svg(args.scenario, args.planner)
    output.write_text(svg, encoding="utf-8")
    print(f"Wrote {output.resolve()}")
    if reason:
        print(f"Planner result: {reason}")


if __name__ == "__main__":
    main()
