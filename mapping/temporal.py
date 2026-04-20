import numpy as np


def _to_prob(mask_or_prob) -> np.ndarray:
    arr = np.asarray(mask_or_prob)
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    return (arr > 0).astype(np.float32)


def ema_prob(prev_prob, cur_mask_or_prob, alpha: float = 0.8) -> np.ndarray:
    cur = _to_prob(cur_mask_or_prob)
    if prev_prob is None:
        return cur
    prev = _to_prob(prev_prob)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * prev + (1.0 - alpha) * cur


def prob_to_mask(prob, thr: float = 0.5) -> np.ndarray:
    prob = _to_prob(prob)
    return (prob >= float(thr)).astype(np.uint8) * 255
