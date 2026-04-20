"""YOLOv8 person detection module."""

import os
import numpy as np
from typing import List, Tuple, Optional

from core.config import (
    DEFAULT_YOLO_MODEL,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
)


class BoundingBox:
    """Simple bounding box container."""
    
    __slots__ = ['x1', 'y1', 'x2', 'y2', 'confidence']
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    def area(self) -> float:
        return self.width * self.height
    
    def to_int(self) -> 'BoundingBox':
        return BoundingBox(
            int(self.x1), int(self.y1), int(self.x2), int(self.y2), self.confidence
        )
    
    def __repr__(self):
        return f"BoundingBox({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f} conf={self.confidence:.2f})"


class PersonDetector:
    """YOLOv8-based person detector."""
    
    def __init__(
        self,
        model: str = DEFAULT_YOLO_MODEL,
        device: str = "auto",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        from ultralytics import YOLO
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Resolve device
        if device == "auto":
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                device = "cpu"
        
        self.device = device
        
        # Load model (download if needed)
        model_path = os.path.expanduser(model)
        if not os.path.exists(model_path):
            print(f"[PersonDetector] Downloading YOLO model: {model}")
        
        self.model = YOLO(model_path)
        print(f"[PersonDetector] Loaded {model} on {device}")
    
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detect all persons in a single frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of BoundingBox for detected persons
        """
        results = self.model(
            frame,
            classes=[0],  # Only detect 'person' class
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )
        
        bboxes = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                bboxes.append(BoundingBox(x1, y1, x2, y2, conf))
        
        return bboxes
    
    def detect_batch(self, frames: np.ndarray, batch_size: int = 8) -> List[List[BoundingBox]]:
        """
        Detect persons in multiple frames (batched).
        
        Args:
            frames: numpy array of shape (N, H, W, C)
            batch_size: Number of frames per batch
            
        Returns:
            List of BoundingBox lists, one per frame
        """
        all_bboxes = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            # Convert BGR to RGB for ultralytics
            batch_rgb = batch[:, :, :, ::-1]
            
            results = self.model(
                batch_rgb,
                classes=[0],
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
            )
            
            for result in results:
                bboxes = []
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        bboxes.append(BoundingBox(x1, y1, x2, y2, conf))
                all_bboxes.append(bboxes)
        
        return all_bboxes
