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
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
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


def smooth_polyline(prev_pts, cur_pts, alpha=0.80):
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
        half_w = half_w_top + (half_w_bottom - half_w_top) * (rel ** 1.12)

        lp = p + np.array([nx, ny], dtype=np.float32) * half_w
        rp = p - np.array([nx, ny], dtype=np.float32) * half_w

        lx = int(np.clip(round(lp[0]), 0, W - 1))
        ly = int(np.clip(round(lp[1]), 0, H - 1))
        rx = int(np.clip(round(rp[0]), 0, W - 1))
        ry = int(np.clip(round(rp[1]), 0, H - 1))

        left.append((lx, ly))
        right.append((rx, ry))

    return left, right


def _clip_layer_to_mask(layer, road_mask):
    if road_mask is None:
        return layer
    clipped = np.zeros_like(layer, dtype=np.uint8)
    m = road_mask > 0
    clipped[m] = layer[m]
    return clipped


def draw_guidance_corridor(
    frame_bgr,
    pts_xy,
    road_mask=None,
    fill_color=(255, 140, 0),
    edge_color=(255, 255, 255),
    edge_thickness=3,
    half_w_bottom=92,
    half_w_top=20,
    fill_alpha=0.25,
):
    out = frame_bgr.copy()
    if pts_xy is None or len(pts_xy) < 2:
        return out

    H, W = out.shape[:2]
    left, right = _build_parallel_sides(
        pts_xy,
        H=H,
        W=W,
        half_w_bottom=half_w_bottom,
        half_w_top=half_w_top,
    )

    poly = np.array(left + right[::-1], dtype=np.int32).reshape(-1, 1, 2)
    fill_layer = np.zeros_like(out, dtype=np.uint8)
    edge_layer = np.zeros_like(out, dtype=np.uint8)

    cv2.fillPoly(fill_layer, [poly], fill_color, lineType=cv2.LINE_AA)

    left_arr = np.array(left, dtype=np.int32).reshape(-1, 1, 2)
    right_arr = np.array(right, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(edge_layer, [left_arr], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)
    cv2.polylines(edge_layer, [right_arr], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

    fill_layer = _clip_layer_to_mask(fill_layer, road_mask)
    edge_layer = _clip_layer_to_mask(edge_layer, road_mask)

    fill_nz = np.any(fill_layer > 0, axis=2)
    if np.any(fill_nz):
        out[fill_nz] = cv2.addWeighted(out[fill_nz], 1.0 - fill_alpha, fill_layer[fill_nz], fill_alpha, 0)

    edge_nz = np.any(edge_layer > 0, axis=2)
    if np.any(edge_nz):
        out[edge_nz] = edge_layer[edge_nz]

    return out


def draw_focus_vehicle(frame_bgr, vehicle_info):
    if vehicle_info is None:
        return frame_bgr

    out = frame_bgr.copy()
    x1, y1, x2, y2, _, _ = vehicle_info["box"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 3)
    label = f'{vehicle_info["name"].upper()} GAP {vehicle_info["bottom_gap_px"]} px'
    cv2.putText(
        out,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return out
