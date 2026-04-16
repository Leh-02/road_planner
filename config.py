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
    seg_every: int = 2
    seg_input_max_side: int = 960

    # --- YOLO ---
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    yolo_imgsz: int = 960
    obstacle_names: tuple[str, ...] = ("person", "bicycle", "motorcycle", "car", "bus", "truck")

    # --- Central road selection ---
    center_band_ratio: float = 0.25
    seed_y_ratio: float = 0.92
    seed_x_span_ratio: float = 0.06

    # --- Grid / planning ---
    cell: int = 8
    inflate_cells: int = 2
    goal_y_ratio: float = 0.22
    start_search_rows_up: int = 16
    astar_w_clear: float = 1.35
    astar_w_turn: float = 0.04
    goal_continuity_penalty: float = 0.004
    path_clearance_penalty: float = 24.0

    # --- Goal sampling ---
    max_branches: int = 5
    branch_min_width_cells: int = 6
    branch_multi_path_min_width_cells: int = 14
    branch_side_sample_ratio: float = 0.22
    branch_label_deadband_cells: int = 2

    # --- Obstacle avoidance ---
    avoid_roi_y1_ratio: float = 0.30
    avoid_roi_y2_ratio: float = 0.98
    lookahead_row_ratio: float = 0.68
    keep_avoid_until_clear: bool = True
    blocking_obstacle_bottom_px: int = 420
    obstacle_goal_margin_cells: int = 8
    obstacle_goal_penalty: float = 9.0

    # --- Lane-change / focus vehicle ---
    lane_change_trigger_bottom_px: int = 320
    lane_change_center_band_ratio: float = 0.85
    lane_change_probe_up_cells: int = 12
    lane_change_vehicle_names: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")
    lane_change_hold_frames: int = 16
    center_goal_penalty: float = 16.0
    wrong_side_penalty: float = 42.0

    # --- Visualization ---
    road_mask_color: tuple[int, int, int] = (0, 255, 0)
    road_mask_alpha: float = 0.16
    obstacle_mask_color: tuple[int, int, int] = (0, 0, 255)
    obstacle_mask_alpha: float = 0.20

    best_path_resample_points: int = 32
    centerline_smooth_alpha: float = 0.35

    corridor_start_half_width_px: int = 88
    corridor_end_half_width_px: int = 18
    corridor_alpha: float = 0.34
    best_corridor_fill_color: tuple[int, int, int] = (255, 140, 0)
    best_corridor_edge_color: tuple[int, int, int] = (255, 255, 255)
    best_corridor_center_color: tuple[int, int, int] = (0, 255, 255)
    best_path_text_color: tuple[int, int, int] = (0, 255, 255)
    best_corridor_edge_thickness_px: int = 3
    best_corridor_center_thickness_px: int = 4
    arrow_thickness_px: int = 5

    # corridor visibility
    clip_best_corridor_to_road: bool = False
