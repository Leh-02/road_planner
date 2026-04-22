import cv2
import numpy as np


class LaneDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def _threshold_lane_markings(self, bev_bgr: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hls, (0, 180, 0), (180, 255, 120))
        yellow_mask = cv2.inRange(hsv, (10, 40, 90), (45, 255, 255))

        gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad = cv2.convertScaleAbs(grad)
        grad_mask = cv2.threshold(grad, 35, 255, cv2.THRESH_BINARY)[1]

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        mask = cv2.bitwise_and(mask, grad_mask)

        k = max(3, int(self.cfg.lane_marking_morph_kernel) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    @staticmethod
    def _segment_centers(row_binary: np.ndarray):
        idx = np.where(row_binary > 0)[0]
        if idx.size == 0:
            return []
        centers = []
        start = idx[0]
        prev = idx[0]
        for c in idx[1:]:
            if c != prev + 1:
                centers.append(int(round((start + prev) * 0.5)))
                start = c
            prev = c
        centers.append(int(round((start + prev) * 0.5)))
        return centers

    @staticmethod
    def _poly_mask_from_edges(shape_hw, left_pts, right_pts):
        h, w = shape_hw
        if left_pts is None or right_pts is None or len(left_pts) < 2 or len(right_pts) < 2:
            return None
        poly = np.array(left_pts + right_pts[::-1], dtype=np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_AA)
        return mask

    @staticmethod
    def _smooth_pts(pts_xy, window: int = 5):
        if pts_xy is None or len(pts_xy) < 3:
            return pts_xy
        if window <= 1:
            return pts_xy
        if window % 2 == 0:
            window += 1
        arr = np.asarray(pts_xy, dtype=np.float32)
        xs = arr[:, 0]
        ys = arr[:, 1]
        pad = window // 2
        padded = np.pad(xs, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        xs_s = np.convolve(padded, kernel, mode="valid")
        return [(int(round(x)), int(round(y))) for x, y in zip(xs_s, ys)]

    @staticmethod
    def _interp_x_from_polyline(polyline, y_query: float):
        if polyline is None or len(polyline) < 2:
            return None
        pts = np.asarray(polyline, dtype=np.float32)
        ys = pts[:, 1]
        xs = pts[:, 0]
        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]
        ys_unique, idx = np.unique(ys, return_index=True)
        xs_unique = xs[idx]
        if ys_unique.size < 2:
            return float(xs_unique[0]) if xs_unique.size else None
        return float(np.interp(float(y_query), ys_unique, xs_unique))

    def detect(self, frame_bgr, road_bev_u8, bev, prev_center_bev=None):
        bev_bgr = bev.warp_image(frame_bgr)
        markings = self._threshold_lane_markings(bev_bgr)
        road = (road_bev_u8 > 0).astype(np.uint8) * 255
        road_dil = cv2.dilate(road, np.ones((5, 5), np.uint8))
        markings = cv2.bitwise_and(markings, road_dil)

        h, w = road.shape[:2]
        lane_width_px = float(max(16.0, self.cfg.lane_width_m / max(1e-6, self.cfg.meters_per_pixel_x)))
        search_margin_px = float(max(24.0, self.cfg.lane_search_margin_m / max(1e-6, self.cfg.meters_per_pixel_x)))
        top_y = int(np.clip(self.cfg.lane_top_y_ratio * h, 0, h - 1))
        ys = np.linspace(h - 4, top_y, int(max(8, self.cfg.lane_samples))).astype(np.int32)

        center_guess = float(w * 0.5)
        if prev_center_bev is not None and len(prev_center_bev) >= 2:
            center_guess = float(prev_center_bev[0][0])

        left_pts = []
        right_pts = []
        center_pts = []
        marking_hits = 0
        support_rows = 0
        history_guided_rows = 0

        for y in ys:
            road_cols = np.where(road[y] > 0)[0]
            if road_cols.size < 8:
                continue

            support_rows += 1
            road_left = float(road_cols[0])
            road_right = float(road_cols[-1])

            row_center_guess = center_guess
            prev_x = self._interp_x_from_polyline(prev_center_bev, float(y))
            if prev_x is not None:
                row_center_guess = float(np.clip(prev_x, road_left, road_right))
                history_guided_rows += 1

            row_centers = self._segment_centers(markings[y])
            left_mark = None
            right_mark = None
            if row_centers:
                cand_left = [c for c in row_centers if c < row_center_guess and row_center_guess - c <= search_margin_px]
                cand_right = [c for c in row_centers if c > row_center_guess and c - row_center_guess <= search_margin_px]
                if cand_left:
                    left_mark = float(max(cand_left))
                if cand_right:
                    right_mark = float(min(cand_right))

            if left_mark is not None and right_mark is not None:
                width = right_mark - left_mark
                if 0.55 * lane_width_px <= width <= 1.75 * lane_width_px:
                    left_x = left_mark
                    right_x = right_mark
                    marking_hits += 1
                else:
                    left_mark = None
                    right_mark = None

            if left_mark is None and right_mark is None:
                left_x = row_center_guess - lane_width_px * 0.5
                right_x = row_center_guess + lane_width_px * 0.5
            elif left_mark is not None and right_mark is None:
                left_x = left_mark
                right_x = left_mark + lane_width_px
                marking_hits += 1
            else:
                right_x = right_mark
                left_x = right_mark - lane_width_px
                marking_hits += 1

            left_x = max(road_left, left_x)
            right_x = min(road_right, right_x)
            if right_x - left_x < 0.45 * lane_width_px:
                c = float(np.clip(row_center_guess, road_left, road_right))
                left_x = max(road_left, c - lane_width_px * 0.5)
                right_x = min(road_right, c + lane_width_px * 0.5)
            if right_x <= left_x:
                continue

            center_x = 0.5 * (left_x + right_x)
            blend = 0.26 if prev_x is None else 0.18
            center_guess = (1.0 - blend) * center_guess + blend * center_x

            left_pts.append((int(round(left_x)), int(y)))
            right_pts.append((int(round(right_x)), int(y)))
            center_pts.append((int(round(center_x)), int(y)))

        if support_rows == 0:
            return {
                "markings_bev": markings,
                "lane_mask_bev": None,
                "relevance_mask_bev": None,
                "left_bev": None,
                "right_bev": None,
                "center_bev": None,
                "confidence": 0.0,
                "lane_width_px": lane_width_px,
            }

        smooth_window = int(max(1, self.cfg.lane_smooth_window))
        left_pts = self._smooth_pts(left_pts, smooth_window)
        right_pts = self._smooth_pts(right_pts, smooth_window)
        center_pts = self._smooth_pts(center_pts, smooth_window)

        lane_mask = self._poly_mask_from_edges((h, w), left_pts, right_pts)
        relevance_mask = None
        if lane_mask is not None:
            margin_px = int(round(self.cfg.obstacle_lane_margin_m / max(1e-6, self.cfg.meters_per_pixel_x)))
            margin_px = max(1, margin_px)
            relevance_mask = cv2.dilate(lane_mask, np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8))
            relevance_mask = cv2.bitwise_and(relevance_mask, road)

        mark_score = marking_hits / max(1, support_rows)
        coverage_score = len(center_pts) / max(1, len(ys))
        history_score = history_guided_rows / max(1, support_rows)
        confidence = 0.55 * mark_score + 0.30 * coverage_score + 0.15 * history_score
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "markings_bev": markings,
            "lane_mask_bev": lane_mask,
            "relevance_mask_bev": relevance_mask,
            "left_bev": left_pts if len(left_pts) >= 2 else None,
            "right_bev": right_pts if len(right_pts) >= 2 else None,
            "center_bev": center_pts if len(center_pts) >= 2 else None,
            "confidence": confidence,
            "lane_width_px": lane_width_px,
        }
