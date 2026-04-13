from dataclasses import dataclass


@dataclass
class PlannerState:
    prev_goal_col: int | None = None
    avoid_side: str | None = None
    prev_centerline: list[tuple[int, int]] | None = None
    last_boxes: list | None = None
    lane_change_hold: int = 0
