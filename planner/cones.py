from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass 
class Cones:
    blue: np.ndarray
    yellow: np.ndarray

    def __post_init__(self):
        self.blue = np.array(self.blue, dtype=float).reshape(-1, 2)
        self.yellow = np.array(self.yellow, dtype=float).reshape(-1, 2)
        