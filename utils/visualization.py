import cv2
import numpy as np


def overlay_mask(frame_bgr, mask_u8, color, alpha=0.35):
    out = frame_bgr.copy()
    m = mask_u8 > 0
    if not np.any(m):
        return out

    col = np.zeros_like(out, dtype=np.uint8)
    col[:] = color
    out[m] = cv2.addWeighted(out[m], 1.0 - alpha, col[m], alpha, 0)
    return out


def draw_boxes(frame_bgr, boxes, names_dict):
    out = frame_bgr.copy()
    for (x1, y1, x2, y2, cls_id, conf) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{names_dict.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def path_to_points_px(path_rc, cell):
    if not path_rc:
        return None
    pts = []
    for r, c in path_rc:
        x = int(c * cell + cell * 0.5)
        y = int(r * cell + cell * 0.5)
        pts.append((x, y))
    return pts


def normalize(vx, vy, eps=1e-6):
    n = float((vx * vx + vy * vy) ** 0.5)
    if n < eps:
        return 0.0, -1.0
    return vx / n, vy / n


def resample_polyline(pts_xy, n=32):
    if pts_xy is None or len(pts_xy) < 2:
        return pts_xy

    pts = np.asarray(pts_xy, dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total < 1e-6:
        p = tuple(map(int, pts[0]))
        return [p for _ in range(n)]

    targets = np.linspace(0.0, total, n)
    out = []
    j = 0
    for t in targets:
        while j < len(seg) - 1 and s[j + 1] < t:
            j += 1
        denom = max(s[j + 1] - s[j], 1e-6)
        a = (t - s[j]) / denom
        p = (1.0 - a) * pts[j] + a * pts[j + 1]
        out.append((int(round(p[0])), int(round(p[1]))))
    return out


def smooth_polyline(prev_pts, cur_pts, alpha=0.35):
    if prev_pts is None or cur_pts is None:
        return cur_pts
    if len(prev_pts) != len(cur_pts):
        return cur_pts

    out = []
    for (px, py), (cx, cy) in zip(prev_pts, cur_pts):
        x = int(round(alpha * px + (1.0 - alpha) * cx))
        y = int(round(alpha * py + (1.0 - alpha) * cy))
        out.append((x, y))
    return out


def _build_parallel_sides(pts_xy, H, W, half_w_bottom, half_w_top):
    pts = np.asarray(pts_xy, dtype=np.float32)
    left, right = [], []
    y_top = float(np.min(pts[:, 1]))
    denom = max(H - y_top, 1.0)

    for i, p in enumerate(pts):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i + 1] if i < len(pts) - 1 else pts[i]
        tangent = p1 - p0
        dx, dy = normalize(float(tangent[0]), float(tangent[1]))
        nx, ny = -dy, dx

        rel = float(np.clip((p[1] - y_top) / denom, 0.0, 1.0))
        half_w = half_w_top + (half_w_bottom - half_w_top) * (rel ** 1.10)

        lp = p + np.array([nx, ny], dtype=np.float32) * half_w
        rp = p - np.array([nx, ny], dtype=np.float32) * half_w

        left.append((int(np.clip(round(lp[0]), 0, W - 1)), int(np.clip(round(lp[1]), 0, H - 1))))
        right.append((int(np.clip(round(rp[0]), 0, W - 1)), int(np.clip(round(rp[1]), 0, H - 1))))

    return left, right


def _clip_layer_to_mask(layer, road_mask):
    if road_mask is None:
        return layer
    clipped = np.zeros_like(layer, dtype=np.uint8)
    m = road_mask > 0
    clipped[m] = layer[m]
    return clipped


def _clip_points_to_frame(pts_xy, H, W):
    if pts_xy is None:
        return None
    out = []
    for x, y in pts_xy:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        out.append((xi, yi))
    return out


def _trim_large_steps(pts_xy, max_step_px: float = 150.0):
    if pts_xy is None or len(pts_xy) < 2:
        return pts_xy
    out = [pts_xy[0]]
    for pt in pts_xy[1:]:
        prev = out[-1]
        d = float(((pt[0] - prev[0]) ** 2 + (pt[1] - prev[1]) ** 2) ** 0.5)
        if d > float(max_step_px):
            break
        out.append(pt)
    return out


def _trim_to_top_y(pts_xy, top_y: int, min_keep: int = 2):
    if pts_xy is None or len(pts_xy) < 2:
        return pts_xy
    kept = []
    for pt in pts_xy:
        if pt[1] < top_y:
            if len(kept) >= int(min_keep):
                break
            continue
        kept.append(pt)
    return kept if len(kept) >= 2 else pts_xy


def _trim_before_lead_box(pts_xy, lead_box, stop_margin_px: int = 16, min_keep: int = 4):
    if pts_xy is None or len(pts_xy) < 2 or lead_box is None:
        return pts_xy
    x1, y1, x2, y2, *_ = lead_box
    x1 -= int(stop_margin_px)
    x2 += int(stop_margin_px)
    y_cut = int(y2 + stop_margin_px)

    kept = []
    for x, y in pts_xy:
        if (x1 <= x <= x2 and y <= y_cut) or y <= int(y1):
            if len(kept) >= int(min_keep):
                break
        kept.append((x, y))
    return kept if len(kept) >= 2 else pts_xy


def _polyline_pixels(layer):
    return int(np.count_nonzero(np.any(layer > 0, axis=2)))


def _draw_centerline_only(out, pts_xy, road_mask, center_color, center_thickness):
    if pts_xy is None or len(pts_xy) < 2:
        return out
    center_layer = np.zeros_like(out, dtype=np.uint8)
    cv2.polylines(center_layer, [np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)], False, center_color, center_thickness, cv2.LINE_AA)
    center_layer = _clip_layer_to_mask(center_layer, road_mask)
    center_nz = np.any(center_layer > 0, axis=2)
    if np.any(center_nz):
        out[center_nz] = center_layer[center_nz]
    return out


def draw_guidance_corridor(
    frame_bgr,
    pts_xy,
    road_mask=None,
    fill_color=(255, 140, 0),
    edge_color=(255, 255, 255),
    center_color=(0, 255, 255),
    edge_thickness=3,
    center_thickness=4,
    half_w_bottom=88,
    half_w_top=18,
    fill_alpha=0.34,
):
    out = frame_bgr.copy()
    if pts_xy is None or len(pts_xy) < 2:
        return out

    H, W = out.shape[:2]
    left, right = _build_parallel_sides(pts_xy, H, W, half_w_bottom, half_w_top)
    poly = np.array(left + right[::-1], dtype=np.int32).reshape(-1, 1, 2)
    fill_layer = np.zeros_like(out, dtype=np.uint8)
    edge_layer = np.zeros_like(out, dtype=np.uint8)
    center_layer = np.zeros_like(out, dtype=np.uint8)

    cv2.fillPoly(fill_layer, [poly], fill_color, lineType=cv2.LINE_AA)
    cv2.polylines(edge_layer, [np.array(left, dtype=np.int32).reshape(-1, 1, 2)], False, edge_color, edge_thickness, cv2.LINE_AA)
    cv2.polylines(edge_layer, [np.array(right, dtype=np.int32).reshape(-1, 1, 2)], False, edge_color, edge_thickness, cv2.LINE_AA)
    cv2.polylines(center_layer, [np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)], False, center_color, center_thickness, cv2.LINE_AA)

    fill_layer = _clip_layer_to_mask(fill_layer, road_mask)
    edge_layer = _clip_layer_to_mask(edge_layer, road_mask)
    center_layer = _clip_layer_to_mask(center_layer, road_mask)

    fill_nz = np.any(fill_layer > 0, axis=2)
    if np.any(fill_nz):
        out[fill_nz] = cv2.addWeighted(out[fill_nz], 1.0 - fill_alpha, fill_layer[fill_nz], fill_alpha, 0)

    edge_nz = np.any(edge_layer > 0, axis=2)
    if np.any(edge_nz):
        out[edge_nz] = edge_layer[edge_nz]

    center_nz = np.any(center_layer > 0, axis=2)
    if np.any(center_nz):
        out[center_nz] = center_layer[center_nz]

    return out


def draw_projected_corridor(
    frame_bgr,
    center_pts_xy,
    left_pts_xy,
    right_pts_xy,
    road_mask=None,
    fill_color=(255, 140, 0),
    edge_color=(255, 255, 255),
    center_color=(0, 255, 255),
    edge_thickness=3,
    center_thickness=4,
    fill_alpha=0.34,
    max_step_px: float = 150.0,
    top_y_ratio: float = 0.18,
    lead_box=None,
    stop_margin_px: int = 16,
    min_points: int = 6,
    fallback_centerline: bool = True,
):
    out = frame_bgr.copy()
    if center_pts_xy is None or left_pts_xy is None or right_pts_xy is None:
        return out
    if len(center_pts_xy) < 2 or len(left_pts_xy) < 2 or len(right_pts_xy) < 2:
        return out

    H, W = out.shape[:2]
    top_y = int(np.clip(round(float(top_y_ratio) * H), 0, H - 1))

    center_pts_xy = _clip_points_to_frame(center_pts_xy, H, W)
    left_pts_xy = _clip_points_to_frame(left_pts_xy, H, W)
    right_pts_xy = _clip_points_to_frame(right_pts_xy, H, W)

    center_pts_xy = _trim_large_steps(center_pts_xy, max_step_px=max_step_px)
    left_pts_xy = _trim_large_steps(left_pts_xy, max_step_px=max_step_px)
    right_pts_xy = _trim_large_steps(right_pts_xy, max_step_px=max_step_px)

    center_pts_xy = _trim_to_top_y(center_pts_xy, top_y=top_y, min_keep=min_points)
    left_pts_xy = _trim_to_top_y(left_pts_xy, top_y=top_y, min_keep=min_points)
    right_pts_xy = _trim_to_top_y(right_pts_xy, top_y=top_y, min_keep=min_points)

    center_pts_xy = _trim_before_lead_box(center_pts_xy, lead_box, stop_margin_px=stop_margin_px, min_keep=min_points)
    left_pts_xy = _trim_before_lead_box(left_pts_xy, lead_box, stop_margin_px=stop_margin_px, min_keep=min_points)
    right_pts_xy = _trim_before_lead_box(right_pts_xy, lead_box, stop_margin_px=stop_margin_px, min_keep=min_points)

    n = min(len(center_pts_xy), len(left_pts_xy), len(right_pts_xy))
    if n < 2:
        return out

    center_pts_xy = center_pts_xy[:n]
    left_pts_xy = left_pts_xy[:n]
    right_pts_xy = right_pts_xy[:n]

    if n < int(max(2, min_points)):
        if fallback_centerline:
            return _draw_centerline_only(out, center_pts_xy, road_mask, center_color, center_thickness)
        return out

    poly = np.array(left_pts_xy + right_pts_xy[::-1], dtype=np.int32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(poly)))
    if area < 32.0:
        if fallback_centerline:
            return _draw_centerline_only(out, center_pts_xy, road_mask, center_color, center_thickness)
        return out

    fill_layer = np.zeros_like(out, dtype=np.uint8)
    edge_layer = np.zeros_like(out, dtype=np.uint8)
    center_layer = np.zeros_like(out, dtype=np.uint8)

    cv2.fillPoly(fill_layer, [poly], fill_color, lineType=cv2.LINE_AA)
    cv2.polylines(edge_layer, [np.array(left_pts_xy, dtype=np.int32).reshape(-1, 1, 2)], False, edge_color, edge_thickness, cv2.LINE_AA)
    cv2.polylines(edge_layer, [np.array(right_pts_xy, dtype=np.int32).reshape(-1, 1, 2)], False, edge_color, edge_thickness, cv2.LINE_AA)
    cv2.polylines(center_layer, [np.array(center_pts_xy, dtype=np.int32).reshape(-1, 1, 2)], False, center_color, center_thickness, cv2.LINE_AA)

    fill_layer = _clip_layer_to_mask(fill_layer, road_mask)
    edge_layer = _clip_layer_to_mask(edge_layer, road_mask)
    center_layer = _clip_layer_to_mask(center_layer, road_mask)

    if _polyline_pixels(center_layer) == 0 and fallback_centerline:
        return _draw_centerline_only(out, center_pts_xy, road_mask, center_color, center_thickness)

    fill_nz = np.any(fill_layer > 0, axis=2)
    if np.any(fill_nz):
        out[fill_nz] = cv2.addWeighted(out[fill_nz], 1.0 - fill_alpha, fill_layer[fill_nz], fill_alpha, 0)

    edge_nz = np.any(edge_layer > 0, axis=2)
    if np.any(edge_nz):
        out[edge_nz] = edge_layer[edge_nz]

    center_nz = np.any(center_layer > 0, axis=2)
    if np.any(center_nz):
        out[center_nz] = center_layer[center_nz]

    return out


def draw_focus_vehicle(frame_bgr, vehicle_info):
    if vehicle_info is None:
        return frame_bgr
    out = frame_bgr.copy()
    x1, y1, x2, y2, _, _ = vehicle_info["box"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 3)
    label = f'{vehicle_info["name"].upper()} GAP {vehicle_info["bottom_gap_px"]} px'
    cv2.putText(out, label, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 200, 255), 2, cv2.LINE_AA)
    return out


def draw_direction_arrow(
    frame_bgr,
    pts_xy,
    color=(0, 255, 255),
    thickness=5,
    tip_length=0.30,
    top_y_ratio: float = 0.18,
    lead_box=None,
    stop_margin_px: int = 16,
):
    if pts_xy is None or len(pts_xy) < 2:
        return frame_bgr

    out = frame_bgr.copy()
    H, W = out.shape[:2]
    pts_xy = _clip_points_to_frame(pts_xy, H, W)
    pts_xy = _trim_large_steps(pts_xy, max_step_px=150.0)
    pts_xy = _trim_to_top_y(pts_xy, top_y=int(np.clip(round(float(top_y_ratio) * H), 0, H - 1)), min_keep=4)
    pts_xy = _trim_before_lead_box(pts_xy, lead_box, stop_margin_px=stop_margin_px, min_keep=4)
    if pts_xy is None or len(pts_xy) < 2:
        return out

    n = len(pts_xy)
    i0 = int(np.clip(round(n * 0.82), 1, n - 1))
    i1 = int(np.clip(round(n * 0.58), 0, n - 2))
    p0 = tuple(map(int, pts_xy[i0]))
    p1 = tuple(map(int, pts_xy[i1]))
    cv2.arrowedLine(out, p0, p1, color, thickness, cv2.LINE_AA, 0, float(np.clip(tip_length, 0.05, 0.8)))
    return out
