import cv2
import numpy as np


def _resolve_points(points, width: int, height: int) -> np.ndarray:
    out = []
    for x, y in points:
        if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
            px = float(x) * (width - 1)
            py = float(y) * (height - 1)
        else:
            px = float(x)
            py = float(y)
        out.append((px, py))
    return np.asarray(out, dtype=np.float32)


class BEVProjector:
    def __init__(self, src_points, dst_points, out_size: tuple[int, int]):
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.src_points = np.asarray(src_points, dtype=np.float32)
        self.dst_points = np.asarray(dst_points, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    @classmethod
    def from_config(cls, frame_hw: tuple[int, int], cfg):
        frame_h, frame_w = frame_hw
        bev_w = int(cfg.bev_width)
        bev_h = int(cfg.bev_height)
        src_points = _resolve_points(cfg.bev_src_points, frame_w, frame_h)
        dst_points = _resolve_points(cfg.bev_dst_points, bev_w, bev_h)
        return cls(src_points, dst_points, (bev_w, bev_h))

    def warp_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(mask_u8, self.H, self.out_size, flags=cv2.INTER_NEAREST)

    def warp_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame_bgr, self.H, self.out_size, flags=cv2.INTER_LINEAR)

    def image_to_bev_points(self, pts_xy):
        if pts_xy is None or len(pts_xy) == 0:
            return []
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in warped]

    def bev_to_image_points(self, pts_xy):
        if pts_xy is None or len(pts_xy) == 0:
            return []
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in warped]
