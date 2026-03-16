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
    overlay_mask, draw_boxes, draw_path, draw_arrow_from_start,
    path_to_points_px, normalize
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

def choose_branches(grid, goal_r, center_c, cfg: Config, prefer_c=None):
    row_free = (grid[goal_r] == 0)
    segs = segments_from_row_free(row_free)
    segs = [(c1, c2) for (c1, c2) in segs if (c2 - c1 + 1) >= cfg.branch_min_width_cells]
    if not segs:
        return [center_c]

    cand = []
    for c1, c2 in segs:
        cc = (c1 + c2) // 2
        if abs(cc - center_c) <= int(cfg.branch_center_limit_ratio * grid.shape[1]):
            cand.append((cc, c2 - c1 + 1))
    if not cand:
        return [center_c]

    if prefer_c is None:
        prefer_c = center_c
    cand.sort(key=lambda t: (abs(t[0] - prefer_c), -t[1]))

    cols = [c for c, _ in cand[:cfg.max_branches]]
    out = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out

def obstacle_present_in_roi(obst_mask_u8, cfg: Config):
    H, W = obst_mask_u8.shape[:2]
    y1 = int(H * cfg.avoid_roi_y1_ratio)
    y2 = int(H * cfg.avoid_roi_y2_ratio)
    roi = obst_mask_u8[y1:y2, :]
    return roi.mean() > 1.0

def pick_avoid_side(grid, look_r, center_c):
    free = (grid[look_r] == 0)
    if not free.any():
        return None
    segs = segments_from_row_free(free)
    segs.sort(key=lambda s: abs(((s[0] + s[1]) // 2) - center_c))
    c1, c2 = segs[0]
    left_w = max(0, center_c - c1)
    right_w = max(0, c2 - center_c)
    if left_w == right_w == 0:
        return None
    return "left" if left_w >= right_w else "right"

def main():
    cfg = Config()
    st = PlannerState()

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(cfg.video_source), help="video path or webcam index (0,1,...)")
    ap.add_argument("--save", default="output/result.mp4", help="output video path (full length)")
    ap.add_argument("--show", action="store_true", help="show preview window")
    args = ap.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.save, fourcc, fps_in, (W, H))

    segmenter = RoadSegmenter(cfg.seg_model_id)
    detector = ObstacleDetector(cfg.yolo_model, cfg.yolo_conf, cfg.obstacle_names)

    frame_idx = 0
    last_road = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(1, cfg.seg_every) == 0 or last_road is None:
            road = segmenter.road_mask(frame)
            last_road = road
        else:
            road = last_road

        central_road = select_central_road(road, cfg.center_band_ratio, cfg.seed_y_ratio, cfg.seed_x_span_ratio)

        boxes = detector.detect(frame)
        obst_mask = detector.boxes_to_mask(boxes, (H, W))

        grid, cell = make_occupancy_grid(central_road, obst_mask, cfg.cell, cfg.inflate_cells)
        clearance = grid_distance_to_obstacles(grid)

        gh, gw = grid.shape
        center_c = gw // 2
        start = (gh - 2, center_c)
        goal_r = int(np.clip(cfg.goal_y_ratio * gh, 2, gh - 3))
        look_r = int(np.clip(cfg.lookahead_row_ratio * gh, 2, gh - 3))

        obst_present = obstacle_present_in_roi(obst_mask, cfg)
        if cfg.keep_avoid_until_clear:
            if obst_present and st.avoid_side is None:
                st.avoid_side = pick_avoid_side(grid, look_r, center_c)
            if (not obst_present) and st.avoid_side is not None:
                st.avoid_side = None
        else:
            st.avoid_side = pick_avoid_side(grid, look_r, center_c) if obst_present else None

        prefer = st.prev_goal_col if st.prev_goal_col is not None else center_c
        branch_cols = choose_branches(grid, goal_r, center_c, cfg, prefer_c=prefer)

        if st.avoid_side is not None and len(branch_cols) > 1:
            if st.avoid_side == "left":
                branch_cols.sort(key=lambda c: c)
            else:
                branch_cols.sort(key=lambda c: -c)

        prev_dir_grid = None
        if st.prev_dir is not None:
            dx, dy = st.prev_dir
            prev_dir_grid = (int(np.sign(dy)), int(np.sign(dx)))

        candidate_paths = []
        best_path, best_score, best_goal_c = [], float("inf"), None

        for gc in branch_cols:
            goal = (goal_r, int(np.clip(gc, 0, gw - 1)))
            path, cost = astar_weighted(grid, start, goal, clearance=clearance, w_clear=0.9, w_turn=0.08, prev_dir=prev_dir_grid)
            if path:
                candidate_paths.append((gc, path, cost))
                cont = 0.0 if st.prev_goal_col is None else 0.02 * abs(gc - st.prev_goal_col)
                score = cost + cont
                if score < best_score:
                    best_score = score
                    best_path = path
                    best_goal_c = gc

        if not best_path:
            best_goal_c = center_c
            best_path, _ = astar_weighted(grid, start, (goal_r, best_goal_c), clearance=clearance)

        if best_goal_c is not None:
            st.prev_goal_col = best_goal_c

        vis = frame.copy()
        vis = overlay_mask(vis, central_road, (0, 255, 0), alpha=0.28)
        vis = overlay_mask(vis, obst_mask, (0, 0, 255), alpha=0.20)
        vis = draw_boxes(vis, boxes, detector.names)

        start_xy = (W // 2, int(H * 0.92))

        # Show candidate arrows at intersections
        if len(candidate_paths) > 1:
            for (gc, path, cost) in candidate_paths[:cfg.max_branches]:
                pts = path_to_points_px(path, cell)
                if pts and len(pts) > 8:
                    p = pts[min(10, len(pts) - 1)]
                    dx, dy = p[0] - start_xy[0], p[1] - start_xy[1]
                    ndx, ndy = normalize(dx, dy)
                    vis = draw_arrow_from_start(vis, start_xy, (ndx, ndy), int(cfg.arrow_len_px * 0.65), (0, 255, 255), thickness=4)

        pts_best = path_to_points_px(best_path, cell)
        if pts_best and len(pts_best) >= 2:
            vis = draw_path(vis, pts_best, color=(255, 0, 0), thickness=3)

            p = pts_best[min(12, len(pts_best) - 1)]
            dx, dy = p[0] - start_xy[0], p[1] - start_xy[1]
            ndx, ndy = normalize(dx, dy)

            sx, sy = st.prev_dir
            a = float(np.clip(cfg.arrow_smooth, 0.0, 0.98))
            ndx = a * sx + (1 - a) * ndx
            ndy = a * sy + (1 - a) * ndy
            ndx, ndy = normalize(ndx, ndy)
            st.prev_dir = (ndx, ndy)

            vis = draw_arrow_from_start(vis, start_xy, st.prev_dir, cfg.arrow_len_px, (255, 0, 0), thickness=6)
        else:
            st.prev_dir = (0.0, -1.0)
            vis = draw_arrow_from_start(vis, start_xy, st.prev_dir, cfg.arrow_len_px, (255, 0, 0), thickness=6)

        mode = "AVOID:" + (st.avoid_side.upper() if st.avoid_side else "NONE")
        cv2.putText(vis, mode, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(vis)

        if args.show:
            cv2.imshow("Central Road Routing", vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                # stop showing, continue saving full video
                args.show = False
                cv2.destroyAllWindows()

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
