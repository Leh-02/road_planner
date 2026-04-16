
import numpy as np


def fit_poly_centerline(pts_xy, degree: int = 2, samples: int = 32, out_hw: tuple[int, int] | None = None):
    if pts_xy is None or len(pts_xy) < 2:
        return pts_xy

    pts = np.asarray(pts_xy, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]

    order = np.argsort(ys)[::-1]  # from bottom to top
    ys = ys[order]
    xs = xs[order]

    # Aggregate duplicate y values by average x for a more stable fit.
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
    if out_hw is not None:
        out_h, out_w = out_hw
    else:
        out_h = None
        out_w = None

    for x, y in zip(sample_xs, sample_ys):
        xi = float(x)
        yi = float(y)
        if out_w is not None:
            xi = float(np.clip(xi, 0, out_w - 1))
        if out_h is not None:
            yi = float(np.clip(yi, 0, out_h - 1))
        out.append((int(round(xi)), int(round(yi))))
    return out


def build_metric_corridor_edges(
    center_pts_xy,
    half_width_m: float,
    meters_per_pixel_x: float,
    meters_per_pixel_y: float,
    out_hw: tuple[int, int] | None = None,
):
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
