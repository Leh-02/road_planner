
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PlannerState:
    prev_goal_col: int | None = None
    avoid_side: str | None = None
    prev_centerline_bev: list[tuple[int, int]] | None = None
    prev_centerline_img: list[tuple[int, int]] | None = None
    lane_change_hold: int = 0

    prev_road_prob: np.ndarray | None = None
    prev_obst_prob: np.ndarray | None = None
