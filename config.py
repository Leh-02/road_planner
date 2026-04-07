
from dataclasses import dataclass


@dataclass
class Config:
    # --- Video ---
    video_source: str | int = "video/input.mp4"
    out_fps_fallback: float = 25.0
    preview_max_width: int = 1280
    preview_max_height: int = 720

    # --- Segmentation ---
    seg_model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    seg_every: int = 2  # 2 = faster, 3 = even faster but less stable on turns
    seg_input_max_side: int = 960  # shrink larger side before segmentation; 0 = do not resize

    # --- YOLO ---
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    obstacle_names: tuple[str, ...] = ("person", "bicycle", "motorcycle", "car", "bus", "truck")

    # --- Central road selection ---
    center_band_ratio: float = 0.25
    seed_y_ratio: float = 0.92
    seed_x_span_ratio: float = 0.06

    # --- Grid / planning (image-space) ---
    cell: int = 8
    inflate_cells: int = 2
    goal_y_ratio: float = 0.25
    start_search_rows_up: int = 12

    # --- Intersection handling ---
    max_branches: int = 3
    branch_min_width_cells: int = 6
    branch_center_limit_ratio: float = 0.38
    branch_label_deadband_cells: int = 2

    # --- Obstacle avoidance behavior ---
    avoid_roi_y1_ratio: float = 0.35
    avoid_roi_y2_ratio: float = 0.98
    lookahead_row_ratio: float = 0.70
    keep_avoid_until_clear: bool = True

    # --- Lane-change trigger / Tesla-like corridor ---
    lane_change_trigger_bottom_px: int = 180
    lane_change_center_band_ratio: float = 0.55
    lane_change_probe_up_cells: int = 8
    lane_change_vehicle_names: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")
    lane_change_block_half_width_cells: int = 6
    lane_change_vehicle_center_penalty: float = 4.0

    corridor_half_width_px: int = 30
    candidate_corridor_half_width_px: int = 18
    corridor_alpha: float = 0.34
    candidate_corridor_alpha: float = 0.18
    corridor_edge_thickness_px: int = 2

    # --- Path smoothing / visualization ---
    arrow_len_px: int = 140
    arrow_smooth: float = 0.75
    arrow_lookahead_idx: int = 12
    path_thickness_px: int = 4
    path_alpha: float = 0.95
    vehicle_size_px: int = 22
