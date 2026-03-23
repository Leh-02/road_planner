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

    # --- Intersection handling ---
    max_branches: int = 3
    branch_min_width_cells: int = 6
    branch_center_limit_ratio: float = 0.38

    # --- Obstacle avoidance behavior ---
    avoid_roi_y1_ratio: float = 0.35
    avoid_roi_y2_ratio: float = 0.98
    lookahead_row_ratio: float = 0.70
    keep_avoid_until_clear: bool = True

    # --- Path smoothing / visualization ---
    arrow_len_px: int = 140
    arrow_smooth: float = 0.75
    arrow_lookahead_idx: int = 12
    path_thickness_px: int = 4
    path_alpha: float = 0.95
    vehicle_size_px: int = 22