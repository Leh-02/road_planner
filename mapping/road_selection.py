import numpy as np
import cv2

def _clean_binary(mask_u8: np.ndarray) -> np.ndarray:
    binm = (mask_u8 > 0).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k2, iterations=2)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k1, iterations=1)
    return binm

def select_central_road(road_mask_u8: np.ndarray, center_band_ratio: float,
                        seed_y_ratio: float, seed_x_span_ratio: float) -> np.ndarray:
    H, W = road_mask_u8.shape[:2]
    binm = _clean_binary(road_mask_u8)

    cy = int(H * seed_y_ratio)
    cx = W // 2
    span = max(5, int(W * seed_x_span_ratio))

    # floodfill from multiple seeds around bottom-center
    step = max(3, span // 6 or 3)
    for dx in range(-span, span + 1, step):
        sx = int(np.clip(cx + dx, 0, W - 1))
        sy = int(np.clip(cy, 0, H - 1))
        if binm[sy, sx] == 0:
            continue
        ff = binm.copy()
        mask = np.zeros((H + 2, W + 2), np.uint8)
        cv2.floodFill(ff, mask, (sx, sy), 128)
        comp = (ff == 128).astype(np.uint8) * 255
        if comp.sum() > 0:
            return comp

    # fallback: connected components scoring
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binm > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return binm

    band_w = int(W * center_band_ratio)
    x1 = max(0, cx - band_w // 2)
    x2 = min(W, cx + band_w // 2)
    band = np.zeros((H, W), dtype=np.uint8)
    band[:, x1:x2] = 1

    best_i, best_score = 1, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 500:
            continue
        comp = (labels == i).astype(np.uint8)
        central_overlap = float((comp * band).sum())
        touches_bottom = float(comp[int(H * 0.92):, max(0, cx - 5):min(W, cx + 5)].sum() > 0)
        score = 2.5 * central_overlap + 0.2 * area + 1e6 * touches_bottom
        if score > best_score:
            best_score = score
            best_i = i

    return (labels == best_i).astype(np.uint8) * 255
