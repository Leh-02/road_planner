import cv2
import numpy as np


class LaneDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def _threshold_lane_markings(self, bev_bgr: np.ndarray) -> np.ndarray:
        """Detect white/yellow lane paint in BEV.

        The mask is intentionally based on color OR bright-edge evidence. The previous
        strict color AND gradient combination was too easy to lose on compressed video,
        shadows, and distant dashed lines.
        """
        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)

        # White lane paint: high lightness, low/medium saturation.
        white_mask = cv2.inRange(hls, (0, 155, 0), (180, 255, 150))
        # Yellow lane paint.
        yellow_mask = cv2.inRange(hsv, (12, 35, 80), (45, 255, 255))

        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        bright = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1]
        edge_mask = cv2.bitwise_and(cv2.threshold(grad_x, 28, 255, cv2.THRESH_BINARY)[1], bright)

        mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), edge_mask)

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
                if prev - start >= 1:
                    centers.append(int(round((start + prev) * 0.5)))
                start = c
            prev = c
        if prev - start >= 1:
            centers.append(int(round((start + prev) * 0.5)))
        return centers

    @staticmethod
    def _row_band_centers(markings: np.ndarray, y: int, band_half: int = 2):
        h, _ = markings.shape[:2]
        y1 = max(0, int(y) - band_half)
        y2 = min(h, int(y) + band_half + 1)
        band = markings[y1:y2]
        # Collapse several nearby rows into one robust scanline.
        row = (np.max(band, axis=0) > 0).astype(np.uint8) * 255
        return LaneDetector._segment_centers(row)

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

    def detect(self, frame_bgr, road_bev_u8, bev, prev_center_bev=None):
        bev_bgr = bev.warp_image(frame_bgr)
        markings = self._threshold_lane_markings(bev_bgr)
        road = (road_bev_u8 > 0).astype(np.uint8) * 255
        road_dil = cv2.dilate(road, np.ones((7, 7), np.uint8))
        markings = cv2.bitwise_and(markings, road_dil)

        h, w = road.shape[:2]
        lane_width_px = float(max(16.0, self.cfg.lane_width_m / max(1e-6, self.cfg.meters_per_pixel_x)))
        search_margin_px = float(max(24.0, self.cfg.lane_search_margin_m / max(1e-6, self.cfg.meters_per_pixel_x)))
        top_y = int(np.clip(self.cfg.lane_top_y_ratio * h, 0, h - 1))
        ys = np.linspace(h - 4, top_y, int(max(8, self.cfg.lane_samples))).astype(np.int32)

        # Start from the previous near-ego lane center; otherwise use the bottom road center.
        center_guess = float(w * 0.5)
        bottom_cols = np.where(road[min(h - 1, h - 4)] > 0)[0]
        if bottom_cols.size >= 8:
            center_guess = float(0.5 * (bottom_cols[0] + bottom_cols[-1]))
        if prev_center_bev is not None and len(prev_center_bev) >= 2:
            center_guess = float(prev_center_bev[0][0])

        left_pts = []
        right_pts = []
        center_pts = []
        marking_rows = 0
        two_mark_rows = 0
        support_rows = 0

        for y in ys:
            road_cols = np.where(road[y] > 0)[0]
            if road_cols.size < 8:
                continue

            support_rows += 1
            road_left = float(road_cols[0])
            road_right = float(road_cols[-1])
            road_center = 0.5 * (road_left + road_right)
            row_centers = self._row_band_centers(markings, int(y), band_half=2)

            left_mark = None
            right_mark = None
            if row_centers:
                cand_left = [c for c in row_centers if c < center_guess and center_guess - c <= search_margin_px]
                cand_right = [c for c in row_centers if c > center_guess and c - center_guess <= search_margin_px]
                if cand_left:
                    left_mark = float(max(cand_left))
                if cand_right:
                    right_mark = float(min(cand_right))

            used_marking = False
            if left_mark is not None and right_mark is not None:
                width = right_mark - left_mark
                if 0.50 * lane_width_px <= width <= 1.85 * lane_width_px:
                    left_x = left_mark
                    right_x = right_mark
                    used_marking = True
                    two_mark_rows += 1
                else:
                    left_mark = None
                    right_mark = None

            if left_mark is None and right_mark is None:
                # Weak fallback only for continuity; confidence below decides whether it is trusted.
                c = 0.65 * center_guess + 0.35 * road_center
                left_x = c - lane_width_px * 0.5
                right_x = c + lane_width_px * 0.5
            elif left_mark is not None and right_mark is None:
                left_x = left_mark
                right_x = left_mark + lane_width_px
                used_marking = True
            elif right_mark is not None and left_mark is None:
                right_x = right_mark
                left_x = right_mark - lane_width_px
                used_marking = True

            if used_marking:
                marking_rows += 1

            left_x = max(road_left, left_x)
            right_x = min(road_right, right_x)
            if right_x - left_x < 0.42 * lane_width_px:
                c = float(np.clip(center_guess, road_left, road_right))
                left_x = max(road_left, c - lane_width_px * 0.5)
                right_x = min(road_right, c + lane_width_px * 0.5)
            if right_x <= left_x:
                continue

            center_x = 0.5 * (left_x + right_x)
            # Trust visible markings more strongly than fallback road center.
            beta = 0.25 if used_marking else 0.10
            center_guess = (1.0 - beta) * center_guess + beta * center_x

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
                "has_markings": False,
            }

        smooth_window = int(max(1, self.cfg.lane_smooth_window))
        left_pts = self._smooth_pts(left_pts, smooth_window)
        right_pts = self._smooth_pts(right_pts, smooth_window)
        center_pts = self._smooth_pts(center_pts, smooth_window)

        lane_mask = self._poly_mask_from_edges((h, w), left_pts, right_pts)
        relevance_mask = None
        if lane_mask is not None and marking_rows >= 2:
            margin_px = int(round(self.cfg.obstacle_lane_margin_m / max(1e-6, self.cfg.meters_per_pixel_x)))
            margin_px = max(1, margin_px)
            relevance_mask = cv2.dilate(lane_mask, np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8))
            relevance_mask = cv2.bitwise_and(relevance_mask, road)

        marking_ratio = marking_rows / max(1, support_rows)
        two_mark_ratio = two_mark_rows / max(1, support_rows)
        point_ratio = len(center_pts) / max(1, len(ys))
        # No visible paint => low confidence. This prevents phantom lane masks from filtering real vehicles.
        confidence = 0.72 * marking_ratio + 0.18 * two_mark_ratio + 0.10 * point_ratio
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "markings_bev": markings,
            "lane_mask_bev": lane_mask if marking_rows >= 2 else None,
            "relevance_mask_bev": relevance_mask,
            "left_bev": left_pts if len(left_pts) >= 2 else None,
            "right_bev": right_pts if len(right_pts) >= 2 else None,
            "center_bev": center_pts if len(center_pts) >= 2 else None,
            "confidence": confidence,
            "lane_width_px": lane_width_px,
            "has_markings": marking_rows >= 2,
        }
