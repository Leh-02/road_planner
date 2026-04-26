import argparse
import os

import cv2
import numpy as np

from config import Config
from segmentation.model import RoadSegmenter
from detection.yolo import ObstacleDetector
from mapping.bev import BEVProjector
from mapping.road_selection import select_central_road
from mapping.grid_builder import make_occupancy_grid, grid_distance_to_obstacles
from mapping.temporal import ema_prob, prob_to_mask
from planning.astar_weighted import astar_weighted
from planning.centerline import (
    fit_poly_centerline,
    build_metric_corridor_edges,
    lane_guidance_penalty,
    blend_centerlines,
    shift_centerline,
    smooth_centerline_x,
    clamp_polyline_shift,
    limit_centerline_curvature,
    trim_polyline_to_mask,
)
from lane.lane_detector import LaneDetector
from utils.visualization import (
    overlay_mask,
    draw_boxes,
    path_to_points_px,
    resample_polyline,
    smooth_polyline,
    draw_guidance_corridor,
    draw_projected_corridor,
    draw_focus_vehicle,
    draw_direction_arrow,
)
from utils.state import PlannerState


def segments_from_row_free(row_free: np.ndarray):
    segs = []
    in_seg = False
    s = 0
    for i, v in enumerate(row_free):
        if v and not in_seg:
            in_seg = True
            s = i
        if in_seg and (not v or i == len(row_free) - 1):
            e = i if (v and i == len(row_free) - 1) else i - 1
            segs.append((s, e))
            in_seg = False
    return segs


def _nearest_free_col(row_free: np.ndarray, desired_c: int, preferred_side: str | None = None, center_c: int | None = None):
    free_cols = np.where(row_free)[0]
    if free_cols.size == 0:
        return None

    desired_c = int(np.clip(desired_c, 0, len(row_free) - 1))
    if preferred_side in ("left", "right") and center_c is not None:
        if preferred_side == "left":
            side_cols = free_cols[free_cols < center_c]
        else:
            side_cols = free_cols[free_cols > center_c]
        if side_cols.size > 0:
            free_cols = side_cols

    idx = int(np.argmin(np.abs(free_cols - desired_c)))
    return int(free_cols[idx])


def pick_start_cell(grid: np.ndarray, center_c: int, max_rows_up: int = 12):
    gh, gw = grid.shape
    center_c = int(np.clip(center_c, 0, gw - 1))
    bottom_r = max(1, gh - 2)

    for r in range(bottom_r, max(0, bottom_r - max_rows_up), -1):
        c = _nearest_free_col(grid[r] == 0, center_c)
        if c is not None:
            return int(r), int(c)

    # Last chance: any free cell in the lower half of the grid.
    for r in range(bottom_r, max(0, gh // 2), -1):
        c = _nearest_free_col(grid[r] == 0, center_c)
        if c is not None:
            return int(r), int(c)
    return None


def resolve_goal_cell(grid: np.ndarray, desired_r: int, desired_c: int, preferred_side: str | None, center_c: int, max_row_delta: int = 16):
    gh, gw = grid.shape
    desired_r = int(np.clip(desired_r, 1, gh - 2))
    desired_c = int(np.clip(desired_c, 0, gw - 1))
    offsets = [0]
    for d in range(1, max_row_delta + 1):
        offsets.extend([-d, d])

    for off in offsets:
        r = desired_r + off
        if not (1 <= r < gh - 1):
            continue
        c = _nearest_free_col(grid[r] == 0, desired_c, preferred_side=preferred_side, center_c=center_c)
        if c is not None:
            return int(r), int(c)
    return None


def greedy_fallback_centerline(grid: np.ndarray, start, goal_r: int, center_c: int, preferred_side: str | None, cell: int, samples: int = 32):
    """Build a safe-enough centerline from free cells if A* fails.

    This prevents the visual corridor from disappearing on frames where segmentation,
    BEV, or obstacle inflation temporarily closes the search graph.
    """
    if start is None:
        return None
    gh, gw = grid.shape
    sr, sc = start
    goal_r = int(np.clip(goal_r, 1, gh - 2))
    rows = np.linspace(sr, goal_r, int(max(6, samples))).astype(np.int32)
    c_prev = int(sc)
    pts = []
    side_offset = max(3, int(round(gw * 0.16)))

    for r in rows:
        if preferred_side == "left":
            desired = max(0, center_c - side_offset)
        elif preferred_side == "right":
            desired = min(gw - 1, center_c + side_offset)
        else:
            desired = c_prev
        c = _nearest_free_col(grid[r] == 0, desired, preferred_side=preferred_side, center_c=center_c)
        if c is None:
            c = _nearest_free_col(grid[r] == 0, c_prev)
        if c is None:
            continue
        c_prev = int(round(0.72 * c_prev + 0.28 * c))
        pts.append((int(c_prev * cell + cell * 0.5), int(r * cell + cell * 0.5)))

    return pts if len(pts) >= 2 else None


def path_side_name(col: int, center_c: int, deadband_cells: int = 2):
    if col < center_c - deadband_cells:
        return "LEFT"
    if col > center_c + deadband_cells:
        return "RIGHT"
    return "CENTER"


def obstacle_present_in_roi(obst_mask_u8: np.ndarray, cfg: Config):
    h, _ = obst_mask_u8.shape[:2]
    y1 = int(h * cfg.avoid_roi_y1_ratio)
    y2 = int(h * cfg.avoid_roi_y2_ratio)
    roi = obst_mask_u8[y1:y2, :]
    return roi.mean() > 1.0


def _pt_inside_mask(mask_u8, pt_xy):
    if mask_u8 is None or pt_xy is None:
        return False
    h, w = mask_u8.shape[:2]
    x = int(np.clip(round(pt_xy[0]), 0, w - 1))
    y = int(np.clip(round(pt_xy[1]), 0, h - 1))
    return bool(mask_u8[y, x] > 0)


def _box_sample_points(box):
    x1, y1, x2, y2, *_ = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xs = np.linspace(x1 + 0.15 * bw, x2 - 0.15 * bw, 5)
    pts = [(float(x), float(y2)) for x in xs]
    pts += [
        (float(0.5 * (x1 + x2)), float(y1 + 0.82 * bh)),
        (float(0.5 * (x1 + x2)), float(y1 + 0.92 * bh)),
    ]
    return pts


def _box_touches_mask(box, mask_u8, bev: BEVProjector | None = None):
    if mask_u8 is None:
        return True
    pts = _box_sample_points(box)
    if bev is not None:
        pts = bev.image_to_bev_points(pts)
    return any(_pt_inside_mask(mask_u8, p) for p in pts)


def filter_relevant_boxes(boxes, names_dict, frame_hw, cfg: Config, bev: BEVProjector | None, road_mask_for_planning, lane_relevance_mask_bev):
    out = []
    infos = []
    for box in boxes:
        x1, y1, x2, y2, cls_id, conf = box
        name = str(names_dict.get(cls_id, "")).lower()

        relevant = True
        bev_bc = None
        if bev is not None:
            proj = bev.image_to_bev_points([(0.5 * (x1 + x2), y2)])
            bev_bc = proj[0] if proj else None
            if cfg.obstacle_must_touch_road:
                relevant = relevant and _box_touches_mask(box, road_mask_for_planning, bev=bev)
            if cfg.use_lane_relevance_filter and lane_relevance_mask_bev is not None:
                # Apply this only for trusted lane masks. It removes parked/side cars but
                # no longer drops a lead car just because one sample point is outside.
                relevant = relevant and _box_touches_mask(box, lane_relevance_mask_bev, bev=bev)
        else:
            if cfg.obstacle_must_touch_road:
                relevant = relevant and _box_touches_mask(box, road_mask_for_planning, bev=None)

        if relevant:
            out.append(box)
            infos.append({"box": box, "name": name, "bev_bottom_center": bev_bc})
    return out, infos


def boxes_to_bev_footprint_mask(boxes, frame_hw, bev: BEVProjector, cfg: Config, road_mask_bev=None):
    """Project only the lower footprint of each detection box into BEV.

    Warping the whole YOLO rectangle makes a tall car box become a huge BEV polygon,
    often blocking all lanes. For navigation we need the road contact footprint.
    """
    out = np.zeros((cfg.bev_height, cfg.bev_width), dtype=np.uint8)
    H, W = frame_hw
    for box in boxes:
        x1, y1, x2, y2, *_ = box
        x1 = float(np.clip(x1, 0, W - 1))
        x2 = float(np.clip(x2, 0, W - 1))
        y1 = float(np.clip(y1, 0, H - 1))
        y2 = float(np.clip(y2, 0, H - 1))
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)
        if y2 <= y1 or x2 <= x1:
            continue

        side_margin = 0.08 * bw
        top_margin = max(6.0, 0.28 * bh)
        fy1 = max(y1, y2 - top_margin)
        pts_img = [
            (x1 - side_margin, y2),
            (x2 + side_margin, y2),
            (x2 + side_margin * 0.45, fy1),
            (x1 - side_margin * 0.45, fy1),
        ]
        pts_bev = bev.image_to_bev_points(pts_img)
        if len(pts_bev) < 3:
            continue
        pts = []
        for x, y in pts_bev:
            pts.append((int(np.clip(round(x), 0, cfg.bev_width - 1)), int(np.clip(round(y), 0, cfg.bev_height - 1))))
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        if abs(cv2.contourArea(poly)) < 2.0:
            continue
        cv2.fillPoly(out, [poly], 255, lineType=cv2.LINE_AA)

    # Small metric inflation in BEV before grid inflation helps far objects remain visible.
    extra_px = max(3, int(round(0.25 / max(1e-6, cfg.meters_per_pixel_x))))
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * extra_px + 1, 2 * extra_px + 1)), iterations=1)
    if road_mask_bev is not None:
        out = cv2.bitwise_and(out, cv2.dilate(road_mask_bev, np.ones((7, 7), np.uint8)))
    return out


def boxes_to_image_footprint_mask(boxes, shape_hw):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2, *_ in boxes:
        bw = max(2, int(x2 - x1))
        bh = max(2, int(y2 - y1))
        y_top = int(round(y2 - max(6, 0.28 * bh)))
        x1e = int(np.clip(x1 - 0.08 * bw, 0, w - 1))
        x2e = int(np.clip(x2 + 0.08 * bw, 0, w - 1))
        y_top = int(np.clip(y_top, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2e > x1e and y2 > y_top:
            cv2.rectangle(mask, (x1e, y_top), (x2e, y2), 255, thickness=-1)
    return mask


def pick_focus_vehicle(boxes, names_dict, frame_hw, cfg: Config):
    h, w = frame_hw
    cx0 = w // 2
    band_half = max(12, int(w * 0.5 * cfg.lane_change_center_band_ratio))
    vehicle_names = {str(name).lower() for name in cfg.lane_change_vehicle_names}

    cand = []
    for box in boxes:
        x1, y1, x2, y2, cls_id, conf = box
        name = str(names_dict.get(cls_id, "")).lower()
        if name not in vehicle_names:
            continue

        cx = int(round((x1 + x2) * 0.5))
        if abs(cx - cx0) > band_half:
            continue

        bottom_gap_px = max(0, h - int(y2))
        area = max(1, int(x2 - x1)) * max(1, int(y2 - y1))
        cand.append((bottom_gap_px, abs(cx - cx0), -area, -conf, box, name))

    if not cand:
        return None

    cand.sort()
    bottom_gap_px, _, _, _, box, name = cand[0]
    x1, y1, x2, y2, _, _ = box
    return {
        "name": name,
        "box": box,
        "bottom_gap_px": int(bottom_gap_px),
        "center_x": int(round((x1 + x2) * 0.5)),
        "top_y": int(y1),
        "bottom_y": int(y2),
    }


def collect_blocking_obstacles(boxes, names_dict, frame_hw, cfg: Config, grid_shape, bev: BEVProjector | None = None, cell: int = 1):
    h, w = frame_hw
    gh, gw = grid_shape
    valid_names = {str(n).lower() for n in cfg.lane_change_vehicle_names} | {"person"}
    out = []

    for box in boxes:
        x1, y1, x2, y2, cls_id, conf = box
        name = str(names_dict.get(cls_id, "")).lower()
        if name not in valid_names:
            continue

        bottom_gap_px = max(0, h - int(y2))
        if bottom_gap_px > cfg.blocking_obstacle_bottom_px:
            continue

        width_px = max(1, x2 - x1)
        strength = 1.0 - float(bottom_gap_px) / float(max(1, cfg.blocking_obstacle_bottom_px))
        strength = float(np.clip(strength, 0.15, 1.0))

        if bev is None:
            cx = int(round((x1 + x2) * 0.5))
            grid_c = int(np.clip(round((cx / max(1, w - 1)) * (gw - 1)), 0, gw - 1))
            half_w_cells = max(1, int(round((width_px / max(1, w)) * gw * 0.5)))
        else:
            bev_pts = bev.image_to_bev_points([(x1, y2), (x2, y2), (0.5 * (x1 + x2), y2)])
            if len(bev_pts) != 3:
                continue
            bx1 = bev_pts[0][0]
            bx2 = bev_pts[1][0]
            bxc = bev_pts[2][0]
            grid_c = int(np.clip(round(float(bxc) / max(1, cell)), 0, gw - 1))
            width_bev_px = abs(float(bx2) - float(bx1))
            half_w_cells = max(1, int(round((width_bev_px / max(1, cell)) * 0.5)))

        out.append({"grid_c": int(grid_c), "half_w_cells": int(half_w_cells), "strength": strength})
    return out


def choose_dynamic_goal_cols(grid: np.ndarray, scan_r: int, center_c: int, cfg: Config):
    row_free = grid[scan_r] == 0
    segs = segments_from_row_free(row_free)
    segs = [(c1, c2) for (c1, c2) in segs if (c2 - c1 + 1) >= cfg.branch_min_width_cells]
    if not segs:
        return [center_c]

    cols = []
    for c1, c2 in segs:
        width = c2 - c1 + 1
        if c1 <= center_c <= c2 and width >= cfg.branch_multi_path_min_width_cells:
            cols.extend([
                int(round(c1 + width * cfg.branch_side_sample_ratio)),
                int(round((c1 + c2) * 0.5)),
                int(round(c2 - width * cfg.branch_side_sample_ratio)),
            ])
        else:
            cols.append(int(round((c1 + c2) * 0.5)))

    for c1, c2 in segs:
        cols.append(int(round((c1 + c2) * 0.5)))

    out = []
    for c in cols:
        c = int(np.clip(c, 0, grid.shape[1] - 1))
        if c not in out:
            out.append(c)
    out.sort(key=lambda c: abs(c - center_c))
    return out[: cfg.max_branches] or [center_c]


def choose_preferred_side(focus_vehicle, avoid_side, frame_hw, center_c: int, grid_w: int, cfg: Config, bev: BEVProjector | None = None, cell: int = 1):
    if focus_vehicle is None:
        return avoid_side

    frame_h, frame_w = frame_hw
    if bev is None:
        focus_grid_c = int(np.clip(round((focus_vehicle["center_x"] / max(1, frame_w - 1)) * (grid_w - 1)), 0, grid_w - 1))
    else:
        bev_pts = bev.image_to_bev_points([(focus_vehicle["center_x"], focus_vehicle["bottom_y"])])
        if not bev_pts:
            return avoid_side
        focus_grid_c = int(np.clip(round(float(bev_pts[0][0]) / max(1, cell)), 0, grid_w - 1))

    if focus_grid_c < center_c - cfg.branch_label_deadband_cells:
        return "right"
    if focus_grid_c > center_c + cfg.branch_label_deadband_cells:
        return "left"
    return avoid_side


def reorder_goal_cols(goal_cols, center_c: int, preferred_side: str | None, deadband_cells: int):
    left_cols = [c for c in goal_cols if c < center_c - deadband_cells]
    center_cols = [c for c in goal_cols if abs(c - center_c) <= deadband_cells]
    right_cols = [c for c in goal_cols if c > center_c + deadband_cells]
    if preferred_side == "left" and left_cols:
        return left_cols + center_cols + right_cols
    if preferred_side == "right" and right_cols:
        return right_cols + center_cols + left_cols
    return goal_cols


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def create_video_writer(save_path: str, fps: float, size: tuple[int, int]):
    root, ext = os.path.splitext(save_path)
    ext = ext.lower()
    if ext == ".avi":
        candidates = [("XVID", save_path), ("MJPG", save_path)]
    else:
        mp4_path = save_path if ext == ".mp4" else root + ".mp4"
        avi_path = root + ".avi"
        candidates = [("mp4v", mp4_path), ("XVID", avi_path), ("MJPG", avi_path)]
    for codec, out_path in candidates:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return out_path, writer, codec
        writer.release()
    raise RuntimeError("Could not open VideoWriter with mp4v/XVID/MJPG on this OpenCV build.")


def fit_frame_to_window(frame_bgr, max_w, max_h):
    if max_w <= 0 or max_h <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 0.999:
        return frame_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_target_lane_centerline(lane_info, preferred_side: str | None, lane_change_active: bool, cfg: Config):
    if lane_info is None or lane_info.get("center_bev") is None:
        return None
    base = lane_info["center_bev"]
    lane_width_px = float(lane_info.get("lane_width_px", max(16.0, cfg.lane_width_m / max(1e-6, cfg.meters_per_pixel_x))))
    if lane_change_active and preferred_side in ("left", "right"):
        direction = 1.0 if preferred_side == "left" else -1.0
        offset_px = direction * lane_width_px * float(cfg.lane_change_target_offset_scale)
        return shift_centerline(base, offset_px)
    return base


def corridor_half_width_from_lane(lane_info, cfg: Config):
    if lane_info is None:
        return cfg.corridor_half_width_m
    lane_width_px = lane_info.get("lane_width_px")
    if lane_width_px is None:
        return cfg.corridor_half_width_m
    lane_width_m = float(lane_width_px) * float(cfg.meters_per_pixel_x)
    lane_half_m = 0.5 * lane_width_m * float(cfg.corridor_lane_fill_scale)
    return max(0.75, min(1.65, lane_half_m))


def draw_corridor_from_bev(vis, central_road, centerline_bev, bev: BEVProjector, cfg: Config, lane_info=None):
    if centerline_bev is None or len(centerline_bev) < 2:
        return vis

    half_width_m = corridor_half_width_from_lane(lane_info, cfg)
    left_bev, right_bev = build_metric_corridor_edges(
        centerline_bev,
        half_width_m=half_width_m,
        meters_per_pixel_x=cfg.meters_per_pixel_x,
        meters_per_pixel_y=cfg.meters_per_pixel_y,
        out_hw=(cfg.bev_height, cfg.bev_width),
    )
    if left_bev is None or right_bev is None:
        return vis

    center_img = bev.bev_to_image_points(centerline_bev)
    left_img = bev.bev_to_image_points(left_bev)
    right_img = bev.bev_to_image_points(right_bev)
    vis = draw_projected_corridor(
        vis,
        center_img,
        left_img,
        right_img,
        road_mask=central_road if cfg.clip_best_corridor_to_road else None,
        fill_color=cfg.best_corridor_fill_color,
        edge_color=cfg.best_corridor_edge_color,
        center_color=cfg.best_corridor_center_color,
        edge_thickness=cfg.best_corridor_edge_thickness_px,
        center_thickness=cfg.best_corridor_center_thickness_px,
        fill_alpha=cfg.corridor_alpha,
        max_step_px=cfg.projected_max_step_px,
        top_y_ratio=cfg.corridor_display_top_y_ratio,
        min_visible_points=cfg.projected_min_visible_points,
        fallback_centerline=cfg.render_fallback_centerline_when_polygon_fails,
    )
    vis = draw_direction_arrow(vis, center_img, color=cfg.best_path_text_color, thickness=cfg.arrow_thickness_px)
    return vis


def corridor_from_previous_bev(vis, central_road, st: PlannerState, bev: BEVProjector, cfg: Config, lane_info=None):
    return draw_corridor_from_bev(vis, central_road, st.prev_centerline_bev, bev, cfg, lane_info=lane_info)


def main():
    cfg = Config()
    st = PlannerState()

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(cfg.video_source), help="video path or webcam index (0,1,...)")
    ap.add_argument("--save", default="output/result.mp4", help="output video path")
    ap.add_argument("--show", action="store_true", help="show preview window")
    args = ap.parse_args()

    source = int(args.source) if isinstance(args.source, str) and args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps_in) or fps_in <= 1.0:
        fps_in = cfg.out_fps_fallback

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = max(2, make_even(src_w))
    h = max(2, make_even(src_h))

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    actual_save_path, writer, codec_used = create_video_writer(args.save, fps_in, (w, h))

    print(f"[INFO] Input size: {src_w}x{src_h}")
    print(f"[INFO] Output size: {w}x{h}")
    print(f"[INFO] Output fps: {fps_in:.2f}")
    print(f"[INFO] Output codec: {codec_used}")
    print(f"[INFO] Saving to: {actual_save_path}")

    segmenter = RoadSegmenter(cfg.seg_model_id, max_side=cfg.seg_input_max_side)
    detector = ObstacleDetector(cfg.yolo_model, cfg.yolo_conf, cfg.obstacle_names, imgsz=cfg.yolo_imgsz)
    bev = BEVProjector.from_config((h, w), cfg) if cfg.use_bev else None
    lane_detector = LaneDetector(cfg) if (cfg.use_bev and cfg.use_lane_detection) else None

    if args.show:
        cv2.namedWindow("Central Road Routing", cv2.WINDOW_NORMAL)

    frame_idx = 0
    last_road = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = frame[:h, :w]

        if frame_idx % max(1, cfg.seg_every) == 0 or last_road is None:
            road = segmenter.road_mask(frame)
            last_road = road
        else:
            road = last_road

        central_road = select_central_road(road, cfg.center_band_ratio, cfg.seed_y_ratio, cfg.seed_x_span_ratio)
        plan_road_raw = bev.warp_mask(central_road) if bev is not None else central_road
        st.prev_road_prob = ema_prob(st.prev_road_prob, plan_road_raw, alpha=cfg.road_ema_alpha)
        road_plan_s = prob_to_mask(st.prev_road_prob, thr=cfg.road_prob_threshold)

        lane_info = None
        if lane_detector is not None:
            lane_info = lane_detector.detect(frame, road_plan_s, bev, prev_center_bev=st.prev_lane_center_bev)
            if lane_info["confidence"] >= cfg.lane_min_confidence and lane_info.get("center_bev") is not None:
                st.prev_lane_center_bev = lane_info.get("center_bev")
                st.prev_lane_left_bev = lane_info.get("left_bev")
                st.prev_lane_right_bev = lane_info.get("right_bev")
                st.lane_miss_count = 0
            else:
                st.lane_miss_count += 1
                if st.prev_lane_center_bev is not None and st.lane_miss_count <= cfg.lane_hold_frames:
                    lane_info = {
                        "center_bev": st.prev_lane_center_bev,
                        "left_bev": st.prev_lane_left_bev,
                        "right_bev": st.prev_lane_right_bev,
                        "lane_mask_bev": None,
                        "relevance_mask_bev": None,
                        "confidence": max(0.0, cfg.lane_min_confidence - 0.04),
                        "lane_width_px": max(16.0, cfg.lane_width_m / max(1e-6, cfg.meters_per_pixel_x)),
                        "has_markings": False,
                    }
                else:
                    lane_info = None

        boxes_all = detector.detect(frame)
        lane_relevance_mask_bev = None
        if (
            lane_info is not None
            and lane_info.get("confidence", 0.0) >= cfg.lane_boundary_trust_threshold
            and lane_info.get("relevance_mask_bev") is not None
        ):
            lane_relevance_mask_bev = lane_info.get("relevance_mask_bev")

        relevant_boxes, _ = filter_relevant_boxes(
            boxes_all,
            detector.names,
            (h, w),
            cfg,
            bev,
            road_plan_s,
            lane_relevance_mask_bev,
        )

        obst_overlay_mask = detector.boxes_to_mask(relevant_boxes, (h, w))
        if bev is not None:
            plan_obst_raw = boxes_to_bev_footprint_mask(relevant_boxes, (h, w), bev, cfg, road_mask_bev=road_plan_s)
        else:
            plan_obst_raw = boxes_to_image_footprint_mask(relevant_boxes, (h, w))

        st.prev_obst_prob = ema_prob(st.prev_obst_prob, plan_obst_raw, alpha=cfg.obstacle_ema_alpha)
        obst_plan_s = prob_to_mask(st.prev_obst_prob, thr=cfg.obstacle_prob_threshold)

        grid, cell = make_occupancy_grid(road_plan_s, obst_plan_s, cfg.cell, cfg.inflate_cells)
        clearance = grid_distance_to_obstacles(grid)

        gh, gw = grid.shape
        center_c = gw // 2
        start = pick_start_cell(grid, center_c, max_rows_up=cfg.start_search_rows_up)
        goal_r = int(np.clip(cfg.goal_y_ratio * gh, 2, gh - 3))
        look_r = int(np.clip(cfg.lookahead_row_ratio * gh, 2, gh - 3))

        focus_vehicle = pick_focus_vehicle(relevant_boxes, detector.names, (h, w), cfg)
        blocking_obstacles = collect_blocking_obstacles(relevant_boxes, detector.names, (h, w), cfg, (gh, gw), bev=bev, cell=cell)
        near_obstacle = obstacle_present_in_roi(obst_overlay_mask, cfg) or bool(blocking_obstacles)

        lane_change_now = focus_vehicle is not None and focus_vehicle["bottom_gap_px"] <= cfg.lane_change_trigger_bottom_px
        if lane_change_now or near_obstacle:
            st.lane_change_hold = cfg.lane_change_hold_frames
        else:
            st.lane_change_hold = max(0, st.lane_change_hold - 1)
        lane_change_active = st.lane_change_hold > 0

        if near_obstacle and st.avoid_side is None:
            row_free = grid[look_r] == 0
            free_cols = np.where(row_free)[0]
            if free_cols.size > 0:
                left_free = int(np.sum(free_cols < center_c))
                right_free = int(np.sum(free_cols > center_c))
                st.avoid_side = "left" if left_free >= right_free else "right"
        elif (not near_obstacle) and cfg.keep_avoid_until_clear:
            st.avoid_side = None

        branch_scan_r = look_r
        if focus_vehicle is not None:
            if bev is None:
                lead_top_r = int(np.clip((focus_vehicle["top_y"] / max(1, h)) * gh, 0, gh - 1))
            else:
                proj = bev.image_to_bev_points([(focus_vehicle["center_x"], focus_vehicle["top_y"])])
                if proj:
                    lead_top_r = int(np.clip(float(proj[0][1]) / max(1, cell), 0, gh - 1))
                else:
                    lead_top_r = int(np.clip((focus_vehicle["top_y"] / max(1, h)) * gh, 0, gh - 1))
            branch_scan_r = int(np.clip(lead_top_r - cfg.lane_change_probe_up_cells, goal_r, gh - 3))

        goal_cols = choose_dynamic_goal_cols(grid, branch_scan_r, center_c, cfg)
        preferred_side = choose_preferred_side(focus_vehicle, st.avoid_side, (h, w), center_c, gw, cfg, bev=bev, cell=cell)
        goal_cols = reorder_goal_cols(goal_cols, center_c, preferred_side, cfg.branch_label_deadband_cells)

        target_lane_center = build_target_lane_centerline(lane_info, preferred_side, lane_change_active, cfg)
        st.prev_target_lane_bev = target_lane_center if target_lane_center is not None else st.prev_target_lane_bev
        lane_width_px = lane_info.get("lane_width_px") if lane_info is not None else max(16.0, cfg.lane_width_m / max(1e-6, cfg.meters_per_pixel_x))

        best_path = []
        best_goal_c = None
        best_score = float("inf")

        if start is not None:
            for gc in goal_cols:
                goal = resolve_goal_cell(grid, goal_r, int(np.clip(gc, 0, gw - 1)), preferred_side, center_c, max_row_delta=18)
                if goal is None:
                    continue
                path, cost = astar_weighted(
                    grid,
                    start,
                    goal,
                    clearance=clearance,
                    w_clear=cfg.astar_w_clear,
                    w_turn=cfg.astar_w_turn,
                    prev_dir=None,
                )
                if not path:
                    continue

                path_clear = np.array([clearance[r, c] for (r, c) in path], dtype=np.float32)
                mean_clear = float(path_clear.mean()) if path_clear.size else 0.0
                low_clear_penalty = cfg.path_clearance_penalty / (mean_clear + 1.0)

                continuity_pen = 0.0
                if st.prev_goal_col is not None:
                    continuity_pen = cfg.goal_continuity_penalty * abs(goal[1] - st.prev_goal_col)

                obstacle_pen = 0.0
                for obs in blocking_obstacles:
                    margin = obs["half_w_cells"] + cfg.obstacle_goal_margin_cells
                    rel = abs(goal[1] - obs["grid_c"]) / float(max(1, margin))
                    obstacle_pen += cfg.obstacle_goal_penalty * obs["strength"] * max(0.0, 1.0 - rel)

                side_pen = 0.0
                side_name = path_side_name(goal[1], center_c, cfg.branch_label_deadband_cells)
                if lane_change_active and preferred_side == "left":
                    if side_name == "CENTER":
                        side_pen += cfg.center_goal_penalty
                    elif side_name == "RIGHT":
                        side_pen += cfg.wrong_side_penalty
                elif lane_change_active and preferred_side == "right":
                    if side_name == "CENTER":
                        side_pen += cfg.center_goal_penalty
                    elif side_name == "LEFT":
                        side_pen += cfg.wrong_side_penalty

                lane_pen = 0.0
                if target_lane_center is not None:
                    path_pts = path_to_points_px(path, cell)
                    lane_pen = lane_guidance_penalty(path_pts, target_lane_center, lane_width_px, weight=cfg.lane_center_penalty_weight)

                score = cost + low_clear_penalty + continuity_pen + obstacle_pen + side_pen + lane_pen
                if score < best_score:
                    best_score = score
                    best_path = path
                    best_goal_c = goal[1]

        if best_goal_c is not None:
            st.prev_goal_col = best_goal_c

        vis = frame.copy()
        vis = overlay_mask(vis, central_road, cfg.road_mask_color, alpha=cfg.road_mask_alpha)
        vis = overlay_mask(vis, obst_overlay_mask, cfg.obstacle_mask_color, alpha=cfg.obstacle_mask_alpha)
        vis = draw_boxes(vis, boxes_all, detector.names)
        vis = draw_focus_vehicle(vis, focus_vehicle if lane_change_active else None)

        pts_best = path_to_points_px(best_path, cell)
        if pts_best is None or len(pts_best) < cfg.min_path_points:
            pts_best = greedy_fallback_centerline(
                grid,
                start,
                goal_r,
                center_c,
                preferred_side if lane_change_active else None,
                cell,
                samples=cfg.best_path_resample_points,
            )
        path_valid = pts_best is not None and len(pts_best) >= cfg.min_path_points

        if path_valid and len(pts_best) >= 2:
            pts_best = resample_polyline(pts_best, n=cfg.best_path_resample_points)

            if cfg.use_polyfit_centerline and len(pts_best) >= cfg.polyfit_min_points:
                out_hw = (cfg.bev_height, cfg.bev_width) if bev is not None else (h, w)
                pts_best = fit_poly_centerline(
                    pts_best,
                    degree=cfg.polyfit_degree,
                    samples=cfg.best_path_resample_points,
                    out_hw=out_hw,
                    smooth_window=cfg.centerline_smooth_window,
                )

            if target_lane_center is not None and bev is not None:
                lane_weight = cfg.lane_change_blend_weight if lane_change_active else cfg.lane_primary_blend_weight
                pts_best = blend_centerlines(pts_best, target_lane_center, weight=lane_weight)

            pts_best = smooth_centerline_x(pts_best, window=cfg.centerline_smooth_window)

            if bev is not None:
                pts_best = trim_polyline_to_mask(pts_best, road_plan_s, min_keep=cfg.min_path_points)
                pts_best = limit_centerline_curvature(pts_best, max_dx_per_step=cfg.max_curve_dx_per_step_px)
                pts_best = clamp_polyline_shift(pts_best, st.prev_centerline_bev, max_shift_px=cfg.max_polyline_shift_px)
                pts_best = smooth_polyline(st.prev_centerline_bev, pts_best, alpha=cfg.centerline_smooth_alpha)
                st.prev_centerline_bev = pts_best
                st.corridor_miss_count = 0
                vis = draw_corridor_from_bev(vis, central_road, pts_best, bev, cfg, lane_info=lane_info)
            else:
                pts_best = limit_centerline_curvature(pts_best, max_dx_per_step=cfg.max_curve_dx_per_step_px)
                pts_best = clamp_polyline_shift(pts_best, st.prev_centerline_img, max_shift_px=cfg.max_polyline_shift_px)
                pts_best = smooth_polyline(st.prev_centerline_img, pts_best, alpha=cfg.centerline_smooth_alpha)
                st.prev_centerline_img = pts_best
                st.corridor_miss_count = 0
                vis = draw_guidance_corridor(
                    vis,
                    pts_best,
                    road_mask=None if not cfg.clip_best_corridor_to_road else central_road,
                    fill_color=cfg.best_corridor_fill_color,
                    edge_color=cfg.best_corridor_edge_color,
                    center_color=cfg.best_corridor_center_color,
                    edge_thickness=cfg.best_corridor_edge_thickness_px,
                    center_thickness=cfg.best_corridor_center_thickness_px,
                    half_w_bottom=cfg.corridor_start_half_width_px,
                    half_w_top=cfg.corridor_end_half_width_px,
                    fill_alpha=cfg.corridor_alpha,
                )
                vis = draw_direction_arrow(vis, pts_best, color=cfg.best_path_text_color, thickness=cfg.arrow_thickness_px)
        else:
            st.corridor_miss_count += 1
            if bev is not None and st.corridor_miss_count <= cfg.corridor_hold_frames:
                vis = corridor_from_previous_bev(vis, central_road, st, bev, cfg, lane_info=lane_info)
            elif st.prev_centerline_img is not None and len(st.prev_centerline_img) >= 2 and st.corridor_miss_count <= cfg.corridor_hold_frames:
                vis = draw_guidance_corridor(
                    vis,
                    st.prev_centerline_img,
                    road_mask=None if not cfg.clip_best_corridor_to_road else central_road,
                    fill_color=cfg.best_corridor_fill_color,
                    edge_color=cfg.best_corridor_edge_color,
                    center_color=cfg.best_corridor_center_color,
                    edge_thickness=cfg.best_corridor_edge_thickness_px,
                    center_thickness=cfg.best_corridor_center_thickness_px,
                    half_w_bottom=cfg.corridor_start_half_width_px,
                    half_w_top=cfg.corridor_end_half_width_px,
                    fill_alpha=cfg.corridor_alpha,
                )
                vis = draw_direction_arrow(vis, st.prev_centerline_img, color=cfg.best_path_text_color, thickness=cfg.arrow_thickness_px)

        mode = "AVOID: " + (preferred_side.upper() if preferred_side else "NONE")
        cv2.putText(vis, mode, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        if focus_vehicle is not None:
            trig = "ON" if lane_change_active else "OFF"
            cv2.putText(
                vis,
                f"LC TRIGGER: {trig} ({focus_vehicle['bottom_gap_px']} px)",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 200, 255) if lane_change_active else (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

        if best_goal_c is not None:
            best_name = path_side_name(best_goal_c, center_c, deadband_cells=cfg.branch_label_deadband_cells)
        elif preferred_side in ("left", "right"):
            best_name = preferred_side.upper()
        else:
            best_name = "CENTER"
        cv2.putText(
            vis,
            "BEST PATH: " + best_name,
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            cfg.best_path_text_color,
            2,
            cv2.LINE_AA,
        )

        if lane_info is not None:
            cv2.putText(
                vis,
                f"LANE CONF: {lane_info.get('confidence', 0.0):.2f}",
                (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(vis)
        if args.show:
            preview = fit_frame_to_window(vis, cfg.preview_max_width, cfg.preview_max_height)
            cv2.imshow("Central Road Routing", preview)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                args.show = False
                cv2.destroyAllWindows()
        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
