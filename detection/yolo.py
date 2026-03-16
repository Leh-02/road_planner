import numpy as np
from ultralytics import YOLO

class ObstacleDetector:
    def __init__(self, model_path: str, conf: float, obstacle_names: tuple[str, ...]):
        self.model = YOLO(model_path)
        self.conf = conf
        self.obstacle_names = set(obstacle_names)

        self.names = self.model.names
        self.obstacle_ids = {i for i, n in self.names.items() if n in self.obstacle_names}

    def detect(self, frame_bgr: np.ndarray):
        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        r0 = res[0]
        boxes = []
        if r0.boxes is None:
            return boxes
        for b in r0.boxes:
            cls_id = int(b.cls.item())
            if cls_id not in self.obstacle_ids:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf.item())
            boxes.append((x1, y1, x2, y2, cls_id, conf))
        return boxes

    @staticmethod
    def boxes_to_mask(boxes, shape_hw):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x1, y1, x2, y2, _, _) in boxes:
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        return mask
