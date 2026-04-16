import argparse
import os

import cv2
import numpy as np

from config import Config
from segmentation.model import RoadSegmenter
from detection.yolo import ObstacleDetector
from mapping.road_selection import select_central_road
from mapping.grid_builder import make_occupancy_grid, grid_distance_to_obstacles
from planning.astar_weighted import astar_weighted
from utils.visualization import (
    overlay_mask,
    draw_boxes,
    path_to_points_px,
    resample_polyline,
    smooth_polyline,
    draw_guidance_corridor,
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


def pick_start_cell(grid: np.ndarray, center_c: int, max_rows_up: int = 12):
    gh, gw = grid.shape
    center_c = int(np.clip(center_c, 0, gw - 1))
    bottom_r = max(1, gh - 2)

    if grid[bottom_r, center_c] == 0:
        return bottom_r, center_c

    for r in range(bottom_r, max(0, bottom_r - max_rows_up), -1):
        free_cols = np.where(grid[r] == 0)[0]
        if free_cols.size == 0:
            continue
        idx = int(np.argmin(np.abs(free_cols - center_c)))
        return int(r), int(free_cols[idx])

    return bottom_r, center_c


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


def collect_blocking_obstacles(boxes, names_dict, frame_hw, cfg: Config, grid_shape):
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

        cx = int(round((x1 + x2) * 0.5))
        width_px = max(1, x2 - x1)
        grid_c = int(np.clip(round((cx / max(1, w - 1)) * (gw - 1)), 0, gw - 1))
        half_w_cells = max(1, int(round((width_px / max(1, w)) * gw * 0.5)))
        strength = 1.0 - float(bottom_gap_px) / float(max(1, cfg.blocking_obstacle_bottom_px))
        out.append({
            "grid_c": grid_c,
            "half_w_cells": half_w_cells,
            "strength": float(np.clip(strength, 0.15, 1.0)),
        })

    return out


def choose_dynamic_goal_cols(grid: np.ndarray, scan_r: int, center_c: int, cfg: Config):
    row_free = (grid[scan_r] == 0)
    segs = segments_from_row_free(row_free)
    segs = [(c1, c2) for (c1, c2) in segs if (c2 - c1 + 1) >= cfg.branch_min_width_cells]
    if not segs:
        return [center_c]

    cols = []
    for c1, c2 in segs:
        width = c2 - c1 + 1
        if c1 <= center_c <= c2 and width >= cfg.branch_multi_path_min_width_cells:
            left_c = int(round(c1 + width * cfg.branch_side_sample_ratio))
            mid_c = int(round((c1 + c2) * 0.5))
            right_c = int(round(c2 - width * cfg.branch_side_sample_ratio))
            cols.extend([left_c, mid_c, right_c])
        else:
            cols.append(int(round((c1 + c2) * 0.5)))

    # add centers of other wide segments too
    for c1, c2 in segs:
        c = int(round((c1 + c2) * 0.5))
        cols.append(c)

    out = []
    for c in cols:
        c = int(np.clip(c, 0, grid.shape[1] - 1))
        if c not in out:
            out.append(c)

    out.sort(key=lambda c: abs(c - center_c))
    return out[: cfg.max_branches] or [center_c]


def choose_preferred_side(focus_vehicle, avoid_side, frame_w: int, center_c: int, grid_w: int, cfg: Config):
    if focus_vehicle is None:
        return avoid_side

    focus_grid_c = int(
        np.clip(round((focus_vehicle["center_x"] / max(1, frame_w - 1)) * (grid_w - 1)), 0, grid_w - 1)
    )

    # If the obstacle is left of center, go right. If right of center, go left.
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

        central_road = select_central_road(
            road,
            cfg.center_band_ratio,
            cfg.seed_y_ratio,
            cfg.seed_x_span_ratio,
        )

        boxes = detector.detect(frame)
        obst_mask = detector.boxes_to_mask(boxes, (h, w))
        grid, cell = make_occupancy_grid(central_road, obst_mask, cfg.cell, cfg.inflate_cells)
        clearance = grid_distance_to_obstacles(grid)

        gh, gw = grid.shape
        center_c = gw // 2
        start = pick_start_cell(grid, center_c, max_rows_up=cfg.start_search_rows_up)
        goal_r = int(np.clip(cfg.goal_y_ratio * gh, 2, gh - 3))
        look_r = int(np.clip(cfg.lookahead_row_ratio * gh, 2, gh - 3))

        focus_vehicle = pick_focus_vehicle(boxes, detector.names, (h, w), cfg)
        blocking_obstacles = collect_blocking_obstacles(boxes, detector.names, (h, w), cfg, (gh, gw))
        near_obstacle = obstacle_present_in_roi(obst_mask, cfg) or bool(blocking_obstacles)

        lane_change_now = (
            focus_vehicle is not None
            and focus_vehicle["bottom_gap_px"] <= cfg.lane_change_trigger_bottom_px
        )
        if lane_change_now or near_obstacle:
            st.lane_change_hold = cfg.lane_change_hold_frames
        else:
            st.lane_change_hold = max(0, st.lane_change_hold - 1)
        lane_change_active = st.lane_change_hold > 0

        if near_obstacle and st.avoid_side is None:
            # Wider free area at lookahead row wins.
            row_free = (grid[look_r] == 0)
            free_cols = np.where(row_free)[0]
            if free_cols.size > 0:
                left_free = int(np.sum(free_cols < center_c))
                right_free = int(np.sum(free_cols > center_c))
                st.avoid_side = "left" if left_free >= right_free else "right"
        elif (not near_obstacle) and cfg.keep_avoid_until_clear:
            st.avoid_side = None

        branch_scan_r = look_r
        if focus_vehicle is not None:
            lead_top_r = int(np.clip((focus_vehicle["top_y"] / max(1, h)) * gh, 0, gh - 1))
            branch_scan_r = int(np.clip(lead_top_r - cfg.lane_change_probe_up_cells, goal_r, gh - 3))

        goal_cols = choose_dynamic_goal_cols(grid, branch_scan_r, center_c, cfg)
        preferred_side = choose_preferred_side(focus_vehicle, st.avoid_side, w, center_c, gw, cfg)
        goal_cols = reorder_goal_cols(goal_cols, center_c, preferred_side, cfg.branch_label_deadband_cells)

        best_path = []
        best_goal_c = None
        best_score = float("inf")

        for gc in goal_cols:
            goal = (goal_r, int(np.clip(gc, 0, gw - 1)))
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
                continuity_pen = cfg.goal_continuity_penalty * abs(gc - st.prev_goal_col)

            obstacle_pen = 0.0
            for obs in blocking_obstacles:
                margin = obs["half_w_cells"] + cfg.obstacle_goal_margin_cells
                rel = abs(gc - obs["grid_c"]) / float(max(1, margin))
                obstacle_pen += cfg.obstacle_goal_penalty * obs["strength"] * max(0.0, 1.0 - rel)

            side_pen = 0.0
            side_name = path_side_name(gc, center_c, cfg.branch_label_deadband_cells)
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

            score = cost + low_clear_penalty + continuity_pen + obstacle_pen + side_pen
            if score < best_score:
                best_score = score
                best_path = path
                best_goal_c = gc

        if not best_path:
            best_goal_c = center_c
            best_path, _ = astar_weighted(
                grid,
                start,
                (goal_r, best_goal_c),
                clearance=clearance,
                w_clear=cfg.astar_w_clear,
                w_turn=cfg.astar_w_turn,
                prev_dir=None,
            )

        if best_goal_c is not None:
            st.prev_goal_col = best_goal_c

        vis = frame.copy()
        vis = overlay_mask(vis, central_road, cfg.road_mask_color, alpha=cfg.road_mask_alpha)
        vis = overlay_mask(vis, obst_mask, cfg.obstacle_mask_color, alpha=cfg.obstacle_mask_alpha)
        vis = draw_boxes(vis, boxes, detector.names)
        vis = draw_focus_vehicle(vis, focus_vehicle if lane_change_active else None)

        pts_best = path_to_points_px(best_path, cell)
        if pts_best and len(pts_best) >= 2:
            pts_best = resample_polyline(pts_best, n=cfg.best_path_resample_points)
            pts_best = smooth_polyline(st.prev_centerline, pts_best, alpha=cfg.centerline_smooth_alpha)
            st.prev_centerline = pts_best
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
        elif st.prev_centerline is not None and len(st.prev_centerline) >= 2:
            vis = draw_guidance_corridor(
                vis,
                st.prev_centerline,
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
            vis = draw_direction_arrow(vis, st.prev_centerline, color=cfg.best_path_text_color, thickness=cfg.arrow_thickness_px)

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
