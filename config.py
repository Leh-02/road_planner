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
    obstacle_names: tuple[str, ...] = ("person", "bicycle", "motorcycle", "car", "bus", "truck")

    # --- Central road selection ---
    center_band_ratio: float = 0.25
    seed_y_ratio: float = 0.92
    seed_x_span_ratio: float = 0.06

    # --- Grid / planning ---
    cell: int = 8
    inflate_cells: int = 2
    goal_y_ratio: float = 0.25
    start_search_rows_up: int = 12
    astar_w_clear: float = 1.05
    astar_w_turn: float = 0.06
    goal_continuity_penalty: float = 0.020

    # --- Branch / lateral path sampling ---
    max_branches: int = 5
    branch_min_width_cells: int = 6
    branch_center_limit_ratio: float = 0.42
    branch_label_deadband_cells: int = 2
    branch_edge_margin_cells: int = 3
    branch_multi_path_min_width_cells: int = 12
    branch_quartile_path_min_width_cells: int = 20
    branch_side_offset_ratio: float = 0.22
    branch_side_offset_min_cells: int = 4

    # --- Obstacle avoidance ---
    avoid_roi_y1_ratio: float = 0.35
    avoid_roi_y2_ratio: float = 0.98
    lookahead_row_ratio: float = 0.70
    keep_avoid_until_clear: bool = True
    blocking_obstacle_bottom_px: int = 300
    obstacle_goal_margin_cells: int = 6
    obstacle_goal_penalty: float = 4.5

    # --- Lane-change / focus vehicle ---
    lane_change_trigger_bottom_px: int = 220
    lane_change_center_band_ratio: float = 0.60
    lane_change_probe_up_cells: int = 9
    lane_change_vehicle_names: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")
    lane_change_block_half_width_cells: int = 7
    lane_change_vehicle_center_penalty: float = 6.5
    lane_change_hold_frames: int = 10

    # --- Visualization ---
    road_mask_color: tuple[int, int, int] = (0, 255, 0)
    road_mask_alpha: float = 0.18
    obstacle_mask_color: tuple[int, int, int] = (0, 0, 255)
    obstacle_mask_alpha: float = 0.18

    best_path_resample_points: int = 34
    candidate_resample_points: int = 24
    centerline_smooth_alpha: float = 0.76

    corridor_start_half_width_px: int = 96
    corridor_end_half_width_px: int = 20
    corridor_alpha: float = 0.28
    best_corridor_fill_color: tuple[int, int, int] = (255, 140, 0)
    best_corridor_edge_color: tuple[int, int, int] = (255, 255, 255)
    best_path_text_color: tuple[int, int, int] = (255, 220, 0)
    best_corridor_edge_thickness_px: int = 3

    candidate_corridor_start_half_width_px: int = 62
    candidate_corridor_end_half_width_px: int = 14
    candidate_corridor_alpha: float = 0.18
    candidate_corridor_fill_color: tuple[int, int, int] = (0, 220, 255)
    candidate_corridor_edge_color: tuple[int, int, int] = (80, 255, 255)
    candidate_corridor_edge_thickness_px: int = 2
