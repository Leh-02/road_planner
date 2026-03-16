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
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
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
