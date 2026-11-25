from __future__ import annotations
import numpy as np
from .cones import Cones


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def map_to_car(points, car_xy, car_yaw) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return (pts - np.asarray(car_xy, dtype=float)) @ rotation(car_yaw)


def car_to_map(points, car_xy, car_yaw) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return pts @ rotation(car_yaw).T + np.asarray(car_xy, dtype=float)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def cones_to_car(cones: Cones, car_xy, car_yaw) -> Cones:
    return Cones(
        map_to_car(cones.blue, car_xy, car_yaw),
        map_to_car(cones.yellow, car_xy, car_yaw),
    )


def filter_ahead(
    cones: Cones, x_max: float = 15.0, y_max: float = 8.0, x_min: float = -2.0
) -> Cones:
    def keep(pts):
        m = (pts[:, 0] > x_min) & (pts[:, 0] < x_max) & (np.abs(pts[:, 1]) < y_max)
        return pts[m]

    return Cones(keep(cones.blue), keep(cones.yellow))
