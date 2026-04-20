"""Actor cropping with green screen background."""

import numpy as np
from typing import List, Tuple
import cv2

from pipeline.detector import BoundingBox
from pipeline.tracker import FrameRecord
from core.config import GREEN_SCREEN_COLOR


class ActorCropper:
    """Crop actor from frame with green screen background."""
    
    def __init__(self, bg_color: Tuple[int, int, int] = GREEN_SCREEN_COLOR):
        self.bg_color = bg_color
    
    def crop_frame(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        output_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Crop actor from a single frame.
        The bbox region keeps the original image, outside is filled with green.
        
        Args:
            frame: Full BGR frame (H, W, 3)
            bbox: Bounding box of the actor
            output_size: Optional fixed output size (width, height). If None, uses bbox size.
            
        Returns:
            Cropped frame with green background
        """
        h, w = frame.shape[:2]
        
        # Clamp bbox to frame boundaries
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        
        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return green frame
            if output_size:
                ow, oh = output_size
            else:
                ow, oh = 224, 224
            result = np.full((oh, ow, 3), self.bg_color, dtype=np.uint8)
            return result
        
        # Extract actor region
        actor_crop = frame[y1:y2, x1:x2].copy()
        
        if output_size:
            ow, oh = output_size
            # Create green background of fixed size
            result = np.full((oh, ow, 3), self.bg_color, dtype=np.uint8)
            # Scale actor to fit (maintain aspect ratio, center)
            ah, aw = actor_crop.shape[:2]
            scale = min(ow / aw, oh / ah)
            new_w = int(aw * scale)
            new_h = int(ah * scale)
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(actor_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Center in output
                ox = (ow - new_w) // 2
                oy = (oh - new_h) // 2
                result[oy:oy + new_h, ox:ox + new_w] = resized
        else:
            # Output is the cropped actor region with green border
            result = actor_crop
        
        return result
    
    def crop_segment(
        self,
        frames: List[np.ndarray],
        records: List[FrameRecord],
        output_size: Tuple[int, int] = None,
    ) -> List[np.ndarray]:
        """
        Crop a sequence of frames for one actor segment.
        
        Args:
            frames: List of full frames (indexed by position in the segment)
            records: List of FrameRecord for this segment
            output_size: Fixed output size (width, height) or None for bbox size
            
        Returns:
            List of cropped frames
        """
        cropped = []
        for frame, rec in zip(frames, records):
            bbox = BoundingBox(rec.x1, rec.y1, rec.x2, rec.y2)
            cropped_frame = self.crop_frame(frame, bbox, output_size)
            cropped.append(cropped_frame)
        return cropped
    
    def crop_segment_from_dict(
        self,
        frames_dict: dict,
        records: List[FrameRecord],
        output_size: Tuple[int, int] = None,
    ) -> List[np.ndarray]:
        """
        Crop frames when frames are stored as {frame_idx: numpy_array}.
        
        Args:
            frames_dict: Dict mapping frame index to frame array
            records: List of FrameRecord
            output_size: Fixed output size (width, height)
        """
        cropped = []
        for rec in records:
            frame = frames_dict.get(rec.frame_idx)
            if frame is None:
                continue
            bbox = BoundingBox(rec.x1, rec.y1, rec.x2, rec.y2)
            cropped_frame = self.crop_frame(frame, bbox, output_size)
            cropped.append(cropped_frame)
        return cropped

    @staticmethod
    def compute_output_size(records_list: List[List[FrameRecord]], 
                            frame_height: int, 
                            frame_width: int) -> Tuple[int, int]:
        """
        Compute a uniform output size that fits all actor bounding boxes.
        Uses the maximum bbox dimensions across all actors and all segments.
        
        Returns:
            (width, height) tuple
        """
        max_w = 0
        max_h = 0
        
        for records in records_list:
            for rec in records:
                w = rec.x2 - rec.x1
                h = rec.y2 - rec.y1
                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h
        
        if max_w == 0 or max_h == 0:
            return (frame_width // 2, frame_height // 2)
        
        # Add 20% padding
        max_w = int(max_w * 1.2)
        max_h = int(max_h * 1.2)
        
        # Clamp to original frame size
        max_w = min(max_w, frame_width)
        max_h = min(max_h, frame_height)
        
        return (max_w, max_h)
