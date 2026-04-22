from __future__ import annotations

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


class ObstacleDetector:
    def __init__(self, model_path: str, conf: float, obstacle_names, imgsz: int = 960):
        self.model_path = model_path
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.allowed_names = {str(name).lower() for name in obstacle_names}
        self.model = YOLO(model_path) if YOLO is not None else None
        self.names = {}
        if self.model is not None:
            raw_names = getattr(self.model.model, 'names', None) or getattr(self.model, 'names', {})
            self.names = {int(k): str(v) for k, v in dict(raw_names).items()}

    def detect(self, frame_bgr: np.ndarray):
        if self.model is None:
            return []
        result = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.asarray(boxes.cls)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.asarray(boxes.conf)
        for (x1, y1, x2, y2), cls_id, score in zip(xyxy, cls, conf):
            cls_id = int(cls_id)
            name = str(self.names.get(cls_id, '')).lower()
            if self.allowed_names and name not in self.allowed_names:
                continue
            out.append((
                int(round(float(x1))),
                int(round(float(y1))),
                int(round(float(x2))),
                int(round(float(y2))),
                cls_id,
                float(score),
            ))
        return out

    @staticmethod
    def boxes_to_mask(boxes, shape_hw):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2, *_ in boxes:
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        return mask
