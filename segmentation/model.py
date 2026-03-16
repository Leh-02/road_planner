import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class RoadSegmenter:
    def __init__(self, model_id: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

    @torch.inference_mode()
    def road_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        logits = out.logits
        logits = torch.nn.functional.interpolate(
            logits, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        road = (pred == self.road_id).astype(np.uint8) * 255
        return road
