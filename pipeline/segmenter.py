"""YOLOv8 person segmentation module."""

import os
import cv2
import numpy as np
from typing import List, Tuple

from core.config import (
    DEFAULT_SEG_MODEL,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
)


class PersonSegmenter:
    """YOLOv8-seg based person segmenter."""

    def __init__(
        self,
        model_path: str = DEFAULT_SEG_MODEL,
        device: str = "auto",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Resolve device — same pattern as PersonDetector in detector.py
        if device == "auto":
            try:
                import torch

                device = "mps" if torch.backends.mps.is_available() else "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                device = "cpu"

        self.device = device

        # Load model (download if needed — same behavior as yolov8n.pt)
        resolved_path = os.path.expanduser(model_path)
        if not os.path.exists(resolved_path):
            print(
                f"[PersonSegmenter] Downloading YOLO segmentation model: {model_path}"
            )

        self.model = YOLO(resolved_path)
        print(f"[PersonSegmenter] Loaded {model_path} on {device}")

    def detect_masks(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Detect person segmentation masks in a single frame.

        For performance, the frame is downscaled so its longest side is
        ``YOLO_INFER_SIZE`` (640 px) before inference.  Masks are then
        resized back to the original frame resolution so that callers
        receive full-resolution boolean masks.

        Args:
            frame: BGR image (numpy array)

        Returns:
            list of (track_id_placeholder, mask_2d) tuples for each person detected
            track_id_placeholder is just the detection index (0, 1, 2, ...)
            mask_2d is a boolean numpy array of shape (H, W) where True = person pixel
        """
        from core.config import YOLO_INFER_SIZE

        orig_h, orig_w = frame.shape[:2]

        # Downscale frame so longest side == YOLO_INFER_SIZE
        longest = max(orig_h, orig_w)
        if longest > YOLO_INFER_SIZE:
            scale = YOLO_INFER_SIZE / longest
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            infer_frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            infer_frame = frame

        results = self.model(
            infer_frame,
            classes=[0],  # person only
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        detections = []
        result = results[0]

        if result.masks is None:
            return detections

        masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)

        for idx, mask_raw in enumerate(masks):
            # Resize mask to original frame size
            mask_img = (mask_raw * 255).astype(np.uint8)
            mask_img = cv2.resize(
                mask_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            mask_resized = mask_img > 127
            detections.append((idx, mask_resized))

        return detections

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply person mask to frame — set non-person pixels to green.

        Args:
            frame: BGR image (numpy array)
            mask: boolean array (H, W), True = person pixel

        Returns:
            BGR image where non-person pixels are set to green (0, 255, 0)
        """
        result = frame.copy()
        result[~mask] = (0, 255, 0)
        return result
