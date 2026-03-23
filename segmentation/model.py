import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class RoadSegmenter:
    def __init__(self, model_id: str, device: str | None = None, max_side: int | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_side = int(max_side) if max_side and max_side > 0 else None

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.road_id = None
        for k, v in self.model.config.id2label.items():
            if str(v).lower() == "road":
                self.road_id = int(k)
                break
        if self.road_id is None:
            raise RuntimeError("Class 'road' not found in model id2label.")

    def _resize_for_inference(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.max_side is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        largest = max(h, w)
        if largest <= self.max_side:
            return frame_bgr

        scale = self.max_side / float(largest)
        new_w = max(32, int(round((w * scale) / 32.0) * 32))
        new_h = max(32, int(round((h * scale) / 32.0) * 32))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @torch.inference_mode()
    def road_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame_bgr.shape[:2]
        frame_small = self._resize_for_inference(frame_bgr)

        img = Image.fromarray(frame_small[:, :, ::-1])
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        out = self.model(**inputs)
        logits = out.logits
        logits = torch.nn.functional.interpolate(
            logits, size=frame_small.shape[:2], mode="bilinear", align_corners=False
        )

        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        road_small = (pred == self.road_id).astype(np.uint8) * 255

        if road_small.shape[:2] != (orig_h, orig_w):
            road = cv2.resize(road_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            road = road_small

        return road
