from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .cones import Cones
from .transforms import cones_to_car, filter_ahead, car_to_map
from .triangulation import blue_yellow_edges
from .centerline import build_centerline
from .bezier import smooth_path, spline_smooth
from .resample import resample, heading, curvature, speed_profile

OK = "ok"
NO_CONES = "no_cones"
TOO_FEW_EDGES = "too_few_edges"
PATH_TOO_SHORT = "path_too_short"


@dataclass
class PlannerConfig:
    # which cones to consider (car frame, metres)
    x_max: float = 15.0
    y_max: float = 8.0
    x_min: float = -2.0
    # blue-yellow edge filter (metres)
    min_edge: float = 2.0
    max_edge: float = 4.0
    min_edges: int = 3
    # ordering
    max_hop: float = 6.0
    max_turn_deg: float = 90.0
    # smoothing and output
    smoother: str = "spline"  # "spline" or "bezier"
    noise_sigma: float = 0.1  # expected midpoint noise (m)
    spacing: float = 0.5
    horizon: float = 15.0
    # speed profile: PLACEHOLDERS, get real values from the vehicle team
    v_max: float = 8.0
    a_lat: float = 6.0
    a_acc: float = 3.0
    a_brake: float = 5.0


@dataclass
class PlanResult:
    status: str
    path: np.ndarray  # (K, 2) car frame, first point at the car
    path_map: np.ndarray  # (K, 2) same points in the map frame
    heading: np.ndarray  # (K,) radians, car frame
    curvature: np.ndarray  # (K,) 1/m, left turn positive
    speed: np.ndarray  # (K,) m/s speed limit
    n_edges: int  # blue-yellow edges found (useful for debugging)


def _empty(status: str, n_edges: int) -> PlanResult:
    z2, z1 = np.zeros((0, 2)), np.zeros(0)
    return PlanResult(status, z2, z2.copy(), z1, z1.copy(), z1.copy(), n_edges)


def _smooth(raw: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    if cfg.smoother == "spline" and len(raw) >= 5:
        dense = spline_smooth(raw, smoothing=len(raw) * cfg.noise_sigma**2, n=300)
        if len(dense) >= 2:
            return dense
    return smooth_path(raw, per_segment=20, start_heading=(1.0, 0.0))


def plan(
    cones: Cones,
    car_xy=(0.0, 0.0),
    car_yaw: float = 0.0,
    cfg: PlannerConfig | None = None,
    v_now: float | None = None,
) -> PlanResult:

    cfg = cfg or PlannerConfig()

    if len(cones.blue) + len(cones.yellow) == 0:
        return _empty(NO_CONES, 0)

    near = filter_ahead(
        cones_to_car(cones, car_xy, car_yaw),
        x_max=cfg.x_max,
        y_max=cfg.y_max,
        x_min=cfg.x_min,
    )

    edges = blue_yellow_edges(near, cfg.min_edge, cfg.max_edge)
    n_edges = len(edges.midpoints)
    if n_edges < cfg.min_edges:
        return _empty(TOO_FEW_EDGES, n_edges)

    raw = build_centerline(edges, max_hop=cfg.max_hop, max_turn_deg=cfg.max_turn_deg)
    if len(raw) < 3:  # the car row plus fewer than 2 points
        return _empty(PATH_TOO_SHORT, n_edges)

    path = resample(_smooth(raw, cfg), cfg.spacing, cfg.horizon)
    if len(path) < 3:
        return _empty(PATH_TOO_SHORT, n_edges)

    k = curvature(path, cfg.spacing)
    v = speed_profile(
        k,
        cfg.spacing,
        v_max=cfg.v_max,
        a_lat=cfg.a_lat,
        a_acc=cfg.a_acc,
        a_brake=cfg.a_brake,
        v_start=v_now,
    )
    return PlanResult(
        OK,
        path,
        car_to_map(path, car_xy, car_yaw),
        heading(path, cfg.spacing),
        k,
        v,
        n_edges,
    )
