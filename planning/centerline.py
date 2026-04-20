import numpy as np


def fit_poly_centerline(pts_xy, degree: int = 2, samples: int = 32, out_hw: tuple[int, int] | None = None):
    if pts_xy is None or len(pts_xy) < 2:
        return pts_xy

    pts = np.asarray(pts_xy, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]

    order = np.argsort(ys)[::-1]
    ys = ys[order]
    xs = xs[order]

    y_round = np.round(ys).astype(np.int32)
    uniq_y = []
    uniq_x = []
    for yv in np.unique(y_round):
        mask = y_round == yv
        uniq_y.append(float(np.mean(ys[mask])))
        uniq_x.append(float(np.mean(xs[mask])))

    uniq_y = np.asarray(uniq_y, dtype=np.float32)
    uniq_x = np.asarray(uniq_x, dtype=np.float32)

    if uniq_y.size < 3:
        return [(int(round(x)), int(round(y))) for x, y in pts_xy]

    deg = int(max(1, min(int(degree), uniq_y.size - 1)))
    try:
        coef = np.polyfit(uniq_y, uniq_x, deg=deg)
    except np.linalg.LinAlgError:
        return [(int(round(x)), int(round(y))) for x, y in pts_xy]

    y_start = float(np.max(uniq_y))
    y_end = float(np.min(uniq_y))
    sample_ys = np.linspace(y_start, y_end, int(max(2, samples)))
    sample_xs = np.polyval(coef, sample_ys)

    out = []
    out_h = out_hw[0] if out_hw is not None else None
    out_w = out_hw[1] if out_hw is not None else None
    for x, y in zip(sample_xs, sample_ys):
        xi = float(x)
        yi = float(y)
        if out_w is not None:
            xi = float(np.clip(xi, 0, out_w - 1))
        if out_h is not None:
            yi = float(np.clip(yi, 0, out_h - 1))
        out.append((int(round(xi)), int(round(yi))))
    return out


def _interp_x_from_polyline(polyline, ys_query):
    pts = np.asarray(polyline, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]
    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]
    if ys.size < 2:
        return np.full(len(ys_query), xs[0] if xs.size else 0.0, dtype=np.float32)
    ys_unique, idx = np.unique(ys, return_index=True)
    xs_unique = xs[idx]
    return np.interp(ys_query, ys_unique, xs_unique).astype(np.float32)


def lane_guidance_penalty(path_pts_xy, lane_center_pts_xy, lane_width_px: float, weight: float = 1.0) -> float:
    if path_pts_xy is None or lane_center_pts_xy is None:
        return 0.0
    if len(path_pts_xy) < 2 or len(lane_center_pts_xy) < 2:
        return 0.0
    path = np.asarray(path_pts_xy, dtype=np.float32)
    ys = path[:, 1]
    pred_x = _interp_x_from_polyline(lane_center_pts_xy, ys)
    dx = np.abs(path[:, 0] - pred_x)
    lane_width_px = float(max(8.0, lane_width_px))
    return float(weight) * float(np.mean(dx / lane_width_px))


def blend_centerlines(primary_pts_xy, guide_pts_xy, weight: float = 0.35):
    if primary_pts_xy is None or guide_pts_xy is None:
        return primary_pts_xy
    if len(primary_pts_xy) < 2 or len(guide_pts_xy) < 2:
        return primary_pts_xy
    primary = np.asarray(primary_pts_xy, dtype=np.float32)
    ys = primary[:, 1]
    gx = _interp_x_from_polyline(guide_pts_xy, ys)
    w = float(np.clip(weight, 0.0, 1.0))
    out = []
    for (px, py), guide_x in zip(primary, gx):
        x = (1.0 - w) * px + w * guide_x
        out.append((int(round(x)), int(round(py))))
    return out


def clamp_polyline_shift(cur_pts_xy, prev_pts_xy, max_shift_px: float = 50.0):
    if cur_pts_xy is None or prev_pts_xy is None:
        return cur_pts_xy
    if len(cur_pts_xy) < 2 or len(prev_pts_xy) < 2:
        return cur_pts_xy
    cur = np.asarray(cur_pts_xy, dtype=np.float32)
    ys = cur[:, 1]
    prev_x = _interp_x_from_polyline(prev_pts_xy, ys)
    dx = np.clip(cur[:, 0] - prev_x, -float(max_shift_px), float(max_shift_px))
    out = []
    for x, y, dd in zip(prev_x + dx, ys, dx):
        out.append((int(round(x)), int(round(y))))
    return out


def limit_centerline_curvature(pts_xy, max_dx_per_step: float = 24.0):
    if pts_xy is None or len(pts_xy) < 3:
        return pts_xy
    pts = [(float(x), float(y)) for x, y in pts_xy]
    out = [pts[0]]
    prev_dx = 0.0
    for i in range(1, len(pts)):
        x_prev, y_prev = out[-1]
        x, y = pts[i]
        raw_dx = x - x_prev
        dx = float(np.clip(raw_dx, prev_dx - max_dx_per_step, prev_dx + max_dx_per_step))
        out.append((x_prev + dx, y))
        prev_dx = dx
    return [(int(round(x)), int(round(y))) for x, y in out]


def trim_polyline_to_mask(pts_xy, mask_u8, min_keep: int = 8):
    if pts_xy is None or len(pts_xy) < 2 or mask_u8 is None:
        return pts_xy
    h, w = mask_u8.shape[:2]
    kept = []
    misses = 0
    for x, y in pts_xy:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        if mask_u8[yi, xi] > 0:
            kept.append((xi, yi))
            misses = 0
        else:
            misses += 1
            if len(kept) >= int(min_keep) and misses >= 2:
                break
    return kept if len(kept) >= 2 else pts_xy


def build_metric_corridor_edges(center_pts_xy, half_width_m: float, meters_per_pixel_x: float, meters_per_pixel_y: float, out_hw: tuple[int, int] | None = None):
    if center_pts_xy is None or len(center_pts_xy) < 2:
        return None, None

    pts = np.asarray(center_pts_xy, dtype=np.float32)
    left = []
    right = []

    mpp_x = float(max(1e-6, meters_per_pixel_x))
    mpp_y = float(max(1e-6, meters_per_pixel_y))
    half_width_m = float(max(0.05, half_width_m))

    out_h = out_hw[0] if out_hw is not None else None
    out_w = out_hw[1] if out_hw is not None else None

    for i, p in enumerate(pts):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i + 1] if i < len(pts) - 1 else pts[i]

        dx_px = float(p1[0] - p0[0])
        dy_px = float(p1[1] - p0[1])
        tx_m = dx_px * mpp_x
        ty_m = dy_px * mpp_y
        norm = float((tx_m * tx_m + ty_m * ty_m) ** 0.5)
        if norm < 1e-6:
            tx_m, ty_m = 0.0, -1.0
            norm = 1.0
        tx_m /= norm
        ty_m /= norm

        nx_m = -ty_m
        ny_m = tx_m
        off_x_px = (nx_m * half_width_m) / mpp_x
        off_y_px = (ny_m * half_width_m) / mpp_y

        lx = float(p[0] + off_x_px)
        ly = float(p[1] + off_y_px)
        rx = float(p[0] - off_x_px)
        ry = float(p[1] - off_y_px)

        if out_w is not None:
            lx = float(np.clip(lx, 0, out_w - 1))
            rx = float(np.clip(rx, 0, out_w - 1))
        if out_h is not None:
            ly = float(np.clip(ly, 0, out_h - 1))
            ry = float(np.clip(ry, 0, out_h - 1))

        left.append((int(round(lx)), int(round(ly))))
        right.append((int(round(rx)), int(round(ry))))

    return left, right
