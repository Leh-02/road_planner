import cv2
import numpy as np


def overlay_mask(frame_bgr, mask_u8, color, alpha=0.35):
    out = frame_bgr.copy()
    m = mask_u8 > 0
    if not np.any(m):
        return out

    col = np.zeros_like(out, dtype=np.uint8)
    col[:] = color
    out[m] = cv2.addWeighted(out[m], 1 - alpha, col[m], alpha, 0)
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


def draw_path(frame_bgr, pts_xy, color=(255, 0, 0), thickness=3, clip_mask=None, alpha=1.0, show_points=True):
    out = frame_bgr.copy()
    if pts_xy is None or len(pts_xy) < 2:
        return out

    layer = np.zeros_like(out, dtype=np.uint8)
    pts = np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(layer, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    if show_points:
        step = max(1, len(pts_xy) // 12)
        for p in pts_xy[::step]:
            cv2.circle(layer, p, max(2, thickness), color, -1, lineType=cv2.LINE_AA)

    if clip_mask is not None:
        m = clip_mask > 0
        clipped = np.zeros_like(layer, dtype=np.uint8)
        clipped[m] = layer[m]
        layer = clipped

    nz = np.any(layer > 0, axis=2)
    if not np.any(nz):
        return out

    if alpha >= 1.0:
        out[nz] = layer[nz]
    else:
        out[nz] = cv2.addWeighted(out[nz], 1 - alpha, layer[nz], alpha, 0)
    return out


def draw_arrow_from_start(frame_bgr, start_xy, dir_vec, length_px, color, thickness=5):
    out = frame_bgr.copy()
    x0, y0 = map(int, start_xy)
    dx, dy = normalize(*dir_vec)
    x1 = int(x0 + dx * length_px)
    y1 = int(y0 + dy * length_px)

    # ВАЖЛИВО: для твоєї збірки OpenCV lineType треба передавати позиційно, а не lineType=
    cv2.arrowedLine(out, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA, 0, 0.25)
    cv2.circle(out, (x0, y0), max(3, thickness), color, -1, lineType=cv2.LINE_AA)
    return out


def draw_vehicle_marker(frame_bgr, center_xy, dir_vec, size=22, body_color=(255, 255, 255), nose_color=(0, 165, 255)):
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


def normalize(vx, vy, eps=1e-6):
    n = (vx * vx + vy * vy) ** 0.5
    if n < eps:
        return 0.0, -1.0
    return vx / n, vy / n