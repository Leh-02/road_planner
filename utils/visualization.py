import numpy as np
import cv2


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


def draw_path(frame_bgr, pts_xy, color=(255, 0, 0), thickness=3):
    out = frame_bgr.copy()
    if pts_xy is None or len(pts_xy) < 2:
        return out
    pts = np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness)
    return out


def draw_arrow_from_start(frame_bgr, start_xy, dir_vec, length_px, color, thickness=5):
    out = frame_bgr.copy()
    x0, y0 = map(int, start_xy)
    dx, dy = dir_vec
    x1 = int(x0 + dx * length_px)
    y1 = int(y0 + dy * length_px)
    cv2.arrowedLine(out, (x0, y0), (x1, y1), color, thickness, tipLength=0.25)
    return out


def normalize(vx, vy, eps=1e-6):
    n = (vx * vx + vy * vy) ** 0.5
    if n < eps:
        return 0.0, -1.0
    return vx / n, vy / n


def resample_polyline(pts_xy, n=28):
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


def smooth_polyline(prev_pts, cur_pts, alpha=0.82):
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

        t = p1 - p0
        nrm = float(np.linalg.norm(t))
        if nrm < 1e-6:
            tx, ty = 0.0, -1.0
        else:
            tx, ty = t / nrm

        nx, ny = -ty, tx

        # Перспектива: внизу ширше, вгорі вужче
        rel = float(np.clip((p[1] - y_top) / denom, 0.0, 1.0))
        half_w = half_w_top + (half_w_bottom - half_w_top) * (rel ** 1.15)

        lp = p + np.array([nx, ny], dtype=np.float32) * half_w
        rp = p - np.array([nx, ny], dtype=np.float32) * half_w

        lx = int(np.clip(round(lp[0]), 0, W - 1))
        ly = int(np.clip(round(lp[1]), 0, H - 1))
        rx = int(np.clip(round(rp[0]), 0, W - 1))
        ry = int(np.clip(round(rp[1]), 0, H - 1))

        left.append((lx, ly))
        right.append((rx, ry))

    return left, right


def draw_guidance_corridor(
    frame_bgr,
    pts_xy,
    color=(255, 0, 0),
    thickness=4,
    half_w_bottom=90,
    half_w_top=18,
    fill_alpha=0.10,
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
    overlay = out.copy()
    cv2.fillPoly(overlay, [poly], color)
    out = cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0)

    left_arr = np.array(left, dtype=np.int32).reshape(-1, 1, 2)
    right_arr = np.array(right, dtype=np.int32).reshape(-1, 1, 2)

    cv2.polylines(out, [left_arr], isClosed=False, color=color, thickness=thickness)
    cv2.polylines(out, [right_arr], isClosed=False, color=color, thickness=thickness)

    return out