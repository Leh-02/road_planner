from dataclasses import dataclass
import numpy as np


@dataclass
class PlannerState:
    prev_goal_col: int | None = None
    avoid_side: str | None = None
    prev_centerline_img: list[tuple[int, int]] | None = None
    prev_centerline_bev: list[tuple[int, int]] | None = None
    prev_lane_center_bev: list[tuple[int, int]] | None = None
    prev_lane_left_bev: list[tuple[int, int]] | None = None
    prev_lane_right_bev: list[tuple[int, int]] | None = None
    prev_road_prob: np.ndarray | None = None
    prev_obst_prob: np.ndarray | None = None
    lane_change_hold: int = 0
    corridor_miss_count: int = 0
    lane_miss_count: int = 0
