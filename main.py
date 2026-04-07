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
    normalize,
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
    h, _ = obst_mask_u8.shape[:2]
    y1 = int(h * cfg.avoid_roi_y1_ratio)
    y2 = int(h * cfg.avoid_roi_y2_ratio)
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


def class_name_from_box(box, names_dict):
    return str(names_dict.get(box[4], "")).lower()


def pick_lead_vehicle(boxes, names_dict, frame_hw, cfg: Config):
    h, w = frame_hw
    cx0 = w // 2
    band_half = max(10, int(w * 0.5 * cfg.lane_change_center_band_ratio))

    cand = []
    vehicle_names = {str(name).lower() for name in cfg.lane_change_vehicle_names}
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


def path_side_name(col: int, center_c: int, deadband_cells: int = 2):
    if col < center_c - deadband_cells:
        return "LEFT"
    if col > center_c + deadband_cells:
        return "RIGHT"
    return "CENTER"


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def create_video_writer(save_path: str, fps: float, size: tuple[int, int]):
    root, ext = os.path.splitext(save_path)
    ext = ext.lower()

    # Не використовуємо avc1/H264, щоб не було проблем з openh264 dll
    if ext == ".avi":
        candidates = [
            ("XVID", save_path),
            ("MJPG", save_path),
        ]
    else:
        mp4_path = save_path if ext == ".mp4" else root + ".mp4"
        avi_path = root + ".avi"
        candidates = [
            ("mp4v", mp4_path),
            ("XVID", avi_path),
            ("MJPG", avi_path),
        ]

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


def draw_arrow_safe(frame_bgr, start_xy, dir_vec, length_px, color, thickness=5):
    out = frame_bgr.copy()
    x0, y0 = map(int, start_xy)
    dx, dy = normalize(*dir_vec)
    x1 = int(x0 + dx * length_px)
    y1 = int(y0 + dy * length_px)

    # Важливо: параметри передаємо позиційно, бо у твоїй збірці OpenCV lineType= падає
    cv2.arrowedLine(out, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA, 0, 0.25)
    cv2.circle(out, (x0, y0), max(3, thickness), color, -1, cv2.LINE_AA)
    return out


def draw_vehicle_marker(frame_bgr, center_xy, dir_vec, size=22,
                        body_color=(255, 255, 255), nose_color=(0, 165, 255)):
    out = frame_bgr.copy()
    cx, cy = map(int, center_xy)
    dx, dy = normalize(*dir_vec)

    forward = np.array([dx, dy], dtype=np.float32)
    side = np.array([-dy, dx], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)

    tip = center + forward * (size * 1.20)
    rear = center - forward * (size * 0.90)
    left = rear + side * (size * 0.70)
    right = rear - side * (size * 0.70)

    pts = np.round(np.array([tip, left, right], dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(out, pts, body_color, lineType=cv2.LINE_AA)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (20, 20, 20), 2, lineType=cv2.LINE_AA)

    nose = np.round(center + forward * (size * 0.85)).astype(np.int32)
    cv2.circle(out, tuple(nose), max(2, size // 6), nose_color, -1, lineType=cv2.LINE_AA)
    return out


def draw_path_on_mask(frame_bgr, pts_xy, road_mask, color=(255, 0, 0), thickness=4, alpha=0.95):
    out = frame_bgr.copy()
    if pts_xy is None or len(pts_xy) < 2:
        return out

    layer = np.zeros_like(out, dtype=np.uint8)
    pts = np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(layer, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    step = max(1, len(pts_xy) // 12)
    for p in pts_xy[::step]:
        cv2.circle(layer, p, max(2, thickness), color, -1, lineType=cv2.LINE_AA)

    m = road_mask > 0
    clipped = np.zeros_like(layer, dtype=np.uint8)
    clipped[m] = layer[m]
    layer = clipped

    nz = np.any(layer > 0, axis=2)
    if not np.any(nz):
        return out

    out[nz] = cv2.addWeighted(out[nz], 1 - alpha, layer[nz], alpha, 0)
    return out


def corridor_edges_from_path(pts_xy, half_width_px: int):
    if pts_xy is None or len(pts_xy) < 2:
        return None, None, None

    pts = np.asarray(pts_xy, dtype=np.float32)
    perps = []
    n = len(pts)
    for i in range(n):
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == n - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]

        dx, dy = normalize(float(tangent[0]), float(tangent[1]))
        perps.append(np.array([-dy, dx], dtype=np.float32))

    perps = np.stack(perps, axis=0)
    left = np.round(pts + perps * float(half_width_px)).astype(np.int32)
    right = np.round(pts - perps * float(half_width_px)).astype(np.int32)
    poly = np.vstack([left, right[::-1]])
    return left, right, poly


def clip_layer_to_mask(layer, road_mask):
    clipped = np.zeros_like(layer, dtype=np.uint8)
    m = road_mask > 0
    clipped[m] = layer[m]
    return clipped


def blend_nonzero(dst, layer, alpha):
    nz = np.any(layer > 0, axis=2)
    if np.any(nz):
        dst[nz] = cv2.addWeighted(dst[nz], 1 - alpha, layer[nz], alpha, 0)
    return dst


def draw_corridor_on_mask(frame_bgr, pts_xy, road_mask, half_width_px,
                          fill_color=(255, 110, 0), edge_color=(255, 255, 255),
                          alpha=0.30, edge_thickness=2, center_color=None,
                          center_thickness=2):
    out = frame_bgr.copy()
    left, right, poly = corridor_edges_from_path(pts_xy, half_width_px)
    if poly is None:
        return out

    fill_layer = np.zeros_like(out, dtype=np.uint8)
    edge_layer = np.zeros_like(out, dtype=np.uint8)

    cv2.fillPoly(fill_layer, [poly.reshape(-1, 1, 2)], fill_color, lineType=cv2.LINE_AA)
    cv2.polylines(edge_layer, [left.reshape(-1, 1, 2)], False, edge_color, edge_thickness, cv2.LINE_AA)
    cv2.polylines(edge_layer, [right.reshape(-1, 1, 2)], False, edge_color, edge_thickness, cv2.LINE_AA)

    if center_color is not None:
        pts = np.asarray(pts_xy, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(edge_layer, [pts], False, center_color, center_thickness, cv2.LINE_AA)

    fill_layer = clip_layer_to_mask(fill_layer, road_mask)
    edge_layer = clip_layer_to_mask(edge_layer, road_mask)

    out = blend_nonzero(out, fill_layer, alpha)
    edge_nz = np.any(edge_layer > 0, axis=2)
    if np.any(edge_nz):
        out[edge_nz] = edge_layer[edge_nz]
    return out


def draw_focus_vehicle(frame_bgr, vehicle_info):
    if vehicle_info is None:
        return frame_bgr

    out = frame_bgr.copy()
    x1, y1, x2, y2, _, _ = vehicle_info["box"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 3)
    label = f'{vehicle_info["name"].upper()} GAP {vehicle_info["bottom_gap_px"]} px'
    cv2.putText(
        out,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    cfg = Config()
    st = PlannerState()

    out_fps_fallback = float(getattr(cfg, "out_fps_fallback", 25.0))
    preview_max_width = int(getattr(cfg, "preview_max_width", 1280))
    preview_max_height = int(getattr(cfg, "preview_max_height", 720))
    arrow_lookahead_idx = int(getattr(cfg, "arrow_lookahead_idx", 12))
    path_thickness_px = int(getattr(cfg, "path_thickness_px", 4))
    path_alpha = float(getattr(cfg, "path_alpha", 0.95))
    vehicle_size_px = int(getattr(cfg, "vehicle_size_px", 22))

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(cfg.video_source), help="video path or webcam index (0,1,...)")
    ap.add_argument("--save", default="output/result.mp4", help="output video path")
    ap.add_argument("--show", action="store_true", help="show preview window")
    args = ap.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps_in) or fps_in <= 1.0:
        fps_in = out_fps_fallback

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
    if actual_save_path != args.save:
        print(f"[INFO] Requested path {args.save} was replaced by {actual_save_path} for compatibility.")

    segmenter = RoadSegmenter(cfg.seg_model_id)
    detector = ObstacleDetector(cfg.yolo_model, cfg.yolo_conf, cfg.obstacle_names)

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

        lead_vehicle = pick_lead_vehicle(boxes, detector.names, (h, w), cfg)
        lane_change_active = (
            lead_vehicle is not None and
            lead_vehicle["bottom_gap_px"] <= cfg.lane_change_trigger_bottom_px
        )

        branch_scan_r = goal_r
        if lane_change_active:
            lead_top_r = int(np.clip((lead_vehicle["top_y"] / max(1, h)) * gh, 0, gh - 1))
            branch_scan_r = int(np.clip(lead_top_r - cfg.lane_change_probe_up_cells, goal_r, gh - 3))

        obst_present = obstacle_present_in_roi(obst_mask, cfg)
        if cfg.keep_avoid_until_clear:
            if obst_present and st.avoid_side is None:
                st.avoid_side = pick_avoid_side(grid, look_r, center_c)
            if (not obst_present) and st.avoid_side is not None:
                st.avoid_side = None
        else:
            st.avoid_side = pick_avoid_side(grid, look_r, center_c) if obst_present else None

        prefer = st.prev_goal_col if st.prev_goal_col is not None else center_c
        branch_cols = choose_branches(grid, branch_scan_r, center_c, cfg, prefer_c=prefer)

        if st.avoid_side is not None and len(branch_cols) > 1:
            if st.avoid_side == "left":
                branch_cols.sort(key=lambda c: c)
            else:
                branch_cols.sort(key=lambda c: -c)

        prev_dir_grid = None
        if st.prev_dir is not None:
            dx, dy = st.prev_dir
            prev_dir_grid = (int(np.sign(dy)), int(np.sign(dx)))

        lead_vehicle_grid_c = None
        if lead_vehicle is not None:
            lead_vehicle_grid_c = int(np.clip(round((lead_vehicle["center_x"] / max(1, w - 1)) * (gw - 1)), 0, gw - 1))

        candidate_paths = []
        best_path, best_score, best_goal_c = [], float("inf"), None

        for gc in branch_cols:
            goal = (goal_r, int(np.clip(gc, 0, gw - 1)))
            path, cost = astar_weighted(
                grid,
                start,
                goal,
                clearance=clearance,
                w_clear=0.9,
                w_turn=0.08,
                prev_dir=prev_dir_grid,
            )
            if path:
                candidate_paths.append((gc, path, cost))
                cont = 0.0 if st.prev_goal_col is None else 0.02 * abs(gc - st.prev_goal_col)

                vehicle_pen = 0.0
                if lane_change_active and lead_vehicle_grid_c is not None:
                    rel = abs(gc - lead_vehicle_grid_c) / float(max(1, cfg.lane_change_block_half_width_cells))
                    vehicle_pen = cfg.lane_change_vehicle_center_penalty * max(0.0, 1.0 - rel)

                score = cost + cont + vehicle_pen
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
        vis = overlay_mask(vis, central_road, (0, 255, 0), alpha=0.22)
        vis = overlay_mask(vis, obst_mask, (0, 0, 255), alpha=0.20)
        vis = draw_boxes(vis, boxes, detector.names)
        vis = draw_focus_vehicle(vis, lead_vehicle if lane_change_active else None)

        start_xy = (
            int(start[1] * cell + cell * 0.5),
            int(start[0] * cell + cell * 0.5),
        )

        if lane_change_active and len(candidate_paths) > 0:
            for (gc, path, _) in candidate_paths[:cfg.max_branches]:
                if best_goal_c is not None and gc == best_goal_c:
                    continue
                pts = path_to_points_px(path, cell)
                if pts and len(pts) >= 2:
                    vis = draw_corridor_on_mask(
                        vis,
                        pts,
                        central_road,
                        half_width_px=cfg.candidate_corridor_half_width_px,
                        fill_color=(0, 220, 255),
                        edge_color=(0, 255, 255),
                        alpha=cfg.candidate_corridor_alpha,
                        edge_thickness=max(1, cfg.corridor_edge_thickness_px - 1),
                    )

        if len(candidate_paths) > 1:
            cand_idx = max(4, arrow_lookahead_idx - 2)
            for (gc, path, _) in candidate_paths[:cfg.max_branches]:
                pts = path_to_points_px(path, cell)
                if pts and len(pts) > 4:
                    p = pts[min(cand_idx, len(pts) - 1)]
                    dx, dy = p[0] - start_xy[0], p[1] - start_xy[1]
                    ndx, ndy = normalize(dx, dy)
                    arrow_color = (0, 255, 255) if (best_goal_c is None or gc != best_goal_c) else (255, 200, 0)
                    vis = draw_arrow_safe(
                        vis,
                        start_xy,
                        (ndx, ndy),
                        int(cfg.arrow_len_px * 0.62),
                        arrow_color,
                        thickness=4,
                    )

        pts_best = path_to_points_px(best_path, cell)
        if pts_best and len(pts_best) >= 2:
            vis = draw_corridor_on_mask(
                vis,
                pts_best,
                central_road,
                half_width_px=cfg.corridor_half_width_px,
                fill_color=(255, 110, 0),
                edge_color=(255, 255, 255),
                alpha=cfg.corridor_alpha,
                edge_thickness=cfg.corridor_edge_thickness_px,
                center_color=(255, 0, 0),
                center_thickness=max(2, path_thickness_px - 1),
            )
            vis = draw_path_on_mask(
                vis,
                pts_best,
                central_road,
                color=(255, 0, 0),
                thickness=path_thickness_px,
                alpha=path_alpha,
            )

            arrow_idx = min(max(1, arrow_lookahead_idx), len(pts_best) - 1)
            p = pts_best[arrow_idx]
            dx, dy = p[0] - start_xy[0], p[1] - start_xy[1]
            ndx, ndy = normalize(dx, dy)

            sx, sy = st.prev_dir
            a = float(np.clip(cfg.arrow_smooth, 0.0, 0.98))
            ndx = a * sx + (1 - a) * ndx
            ndy = a * sy + (1 - a) * ndy
            ndx, ndy = normalize(ndx, ndy)
            st.prev_dir = (ndx, ndy)

            vis = draw_vehicle_marker(vis, start_xy, st.prev_dir, size=vehicle_size_px)
            vis = draw_arrow_safe(vis, start_xy, st.prev_dir, cfg.arrow_len_px, (255, 0, 0), thickness=6)
        else:
            st.prev_dir = (0.0, -1.0)
            vis = draw_vehicle_marker(vis, start_xy, st.prev_dir, size=vehicle_size_px)
            vis = draw_arrow_safe(vis, start_xy, st.prev_dir, cfg.arrow_len_px, (255, 0, 0), thickness=6)

        mode = "AVOID: " + (st.avoid_side.upper() if st.avoid_side else "NONE")
        cv2.putText(vis, mode, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        if lead_vehicle is not None:
            trig = "ON" if lane_change_active else "OFF"
            cv2.putText(
                vis,
                f"LC TRIGGER: {trig} ({lead_vehicle['bottom_gap_px']} px)",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 165, 255) if lane_change_active else (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

        if candidate_paths:
            option_names = []
            for gc, _, _ in candidate_paths[:cfg.max_branches]:
                name = path_side_name(gc, center_c, deadband_cells=cfg.branch_label_deadband_cells)
                if name not in option_names:
                    option_names.append(name)
            if option_names:
                cv2.putText(
                    vis,
                    "OPTIONS: " + " | ".join(option_names),
                    (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 255) if lane_change_active else (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if best_goal_c is not None:
            best_name = path_side_name(best_goal_c, center_c, deadband_cells=cfg.branch_label_deadband_cells)
            cv2.putText(
                vis,
                "BEST PATH: " + best_name,
                (15, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(vis)

        if args.show:
            preview = fit_frame_to_window(vis, preview_max_width, preview_max_height)
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
