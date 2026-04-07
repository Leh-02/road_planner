from dataclasses import dataclass


@dataclass
class PlannerState:
    prev_goal_col: int | None = None
    prev_dir: tuple[float, float] = (0.0, -1.0)
    avoid_side: str | None = None
    prev_centerline: list[tuple[int, int]] | None = None
    last_boxes: list | None = None