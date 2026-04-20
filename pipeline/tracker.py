"""ByteTrack multi-object tracking module."""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from pipeline.detector import BoundingBox
from core.config import (
    DEFAULT_TRACK_THRESH,
    DEFAULT_MAX_LOST_FRAMES,
)


@dataclass
class FrameRecord:
    """A record of a track at a specific frame."""
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float


class ByteTrackWrapper:
    """
    Wrapper around ByteTrack for person tracking.
    Uses the bytetrack package (pip install bytetrack).
    Falls back to a simple greedy tracker if bytetrack is unavailable.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        track_thresh: float = DEFAULT_TRACK_THRESH,
        max_lost: int = DEFAULT_MAX_LOST_FRAMES,
    ):
        self.fps = fps
        self.track_thresh = track_thresh
        self.max_lost = max_lost
        self.next_track_id = 1
        self.tracks: Dict[int, List[FrameRecord]] = {}
        self.active_tracks: Dict[int, FrameRecord] = {}  # track_id -> last record
        
        # Try to import ByteTrack
        self._use_bytetrack = False
        try:
            from yolov8.bytetrack import BYTETracker
            self.tracker = BYTETracker(
                track_thresh=self.track_thresh,
                track_buffer=max_lost,
                match_thresh=0.8,
                frame_rate=fps,
            )
            self._use_bytetrack = True
            print("[Tracker] Using ByteTrack backend")
        except ImportError:
            print("[Tracker] ByteTrack not available, using simple greedy tracker")
            self.tracker = None
    
    def update(self, bboxes: List[BoundingBox], frame_idx: int) -> List[Tuple[int, BoundingBox]]:
        """
        Update tracking with current frame detections.
        
        Args:
            bboxes: List of detected bounding boxes for current frame
            frame_idx: Current frame index
            
        Returns:
            List of (track_id, bbox) for this frame
        """
        if self._use_bytetrack:
            return self._update_bytetrack(bboxes, frame_idx)
        else:
            return self._update_greedy(bboxes, frame_idx)
    
    def _update_bytetrack(self, bboxes: List[BoundingBox], frame_idx: int) -> List[Tuple[int, BoundingBox]]:
        """Update using ByteTrack."""
        if not bboxes:
            # No detections, advance lost counters
            result = []
            # ByteTrack handles this internally
            return result
        
        # Convert to ByteTrack format: [[x1, y1, x2, y2, conf], ...]
        dets = np.array([[b.x1, b.y1, b.x2, b.y2, b.confidence] for b in bboxes])
        
        try:
            online_targets = self.tracker.update(dets)
        except Exception:
            # If ByteTrack fails, fallback to greedy
            return self._update_greedy(bboxes, frame_idx)
        
        result = []
        for t in online_targets:
            tid = t.track_id
            tlwh = t.tlwh
            x1, y1, x2, y2 = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
            bbox = BoundingBox(x1, y1, x2, y2)
            
            if tid not in self.tracks:
                self.tracks[tid] = []
            
            rec = FrameRecord(frame_idx, x1, y1, x2, y2)
            self.tracks[tid].append(rec)
            self.active_tracks[tid] = rec
            result.append((tid, bbox))
        
        return result
    
    def _update_greedy(self, bboxes: List[BoundingBox], frame_idx: int) -> List[Tuple[int, BoundingBox]]:
        """
        Simple greedy tracker fallback.
        Matches each bbox to the closest active track by IoU.
        """
        result = []
        matched_dets = set()
        matched_tracks = set()
        
        for tid, last_rec in list(self.active_tracks.items()):
            # Check if track is still alive (not lost for too long)
            if frame_idx - last_rec.frame_idx > self.max_lost:
                del self.active_tracks[tid]
                continue
            
            best_iou = 0
            best_det_idx = -1
            
            for i, bbox in enumerate(bboxes):
                if i in matched_dets:
                    continue
                iou = self._iou(last_rec, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_iou > 0.3:  # IoU threshold for matching
                bbox = bboxes[best_det_idx]
                matched_dets.add(best_det_idx)
                matched_tracks.add(tid)
                
                rec = FrameRecord(frame_idx, bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                self.tracks[tid].append(rec)
                self.active_tracks[tid] = rec
                result.append((tid, bbox))
        
        # Create new tracks for unmatched detections
        for i, bbox in enumerate(bboxes):
            if i not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = []
                rec = FrameRecord(frame_idx, bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                self.tracks[tid].append(rec)
                self.active_tracks[tid] = rec
                result.append((tid, bbox))
        
        return result
    
    def _iou(self, last_rec: FrameRecord, bbox: BoundingBox) -> float:
        """Compute IoU between last record and current bbox."""
        x1 = max(last_rec.x1, bbox.x1)
        y1 = max(last_rec.y1, bbox.y1)
        x2 = min(last_rec.x2, bbox.x2)
        y2 = min(last_rec.y2, bbox.y2)
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        
        area1 = (last_rec.x2 - last_rec.x1) * (last_rec.y2 - last_rec.y1)
        area2 = bbox.area()
        union = area1 + area2 - inter_area
        
        return inter_area / union if union > 0 else 0
    
    def finish(self) -> Dict[int, List[FrameRecord]]:
        """
        Return all completed tracks.
        
        Returns:
            {track_id: [FrameRecord, ...], ...}
        """
        return self.tracks


class PersonTracker:
    """High-level person tracker wrapper."""
    
    def __init__(
        self,
        fps: float = 30.0,
        track_thresh: float = DEFAULT_TRACK_THRESH,
        max_lost: int = DEFAULT_MAX_LOST_FRAMES,
    ):
        self.tracker = ByteTrackWrapper(
            fps=fps,
            track_thresh=track_thresh,
            max_lost=max_lost,
        )
    
    def update(self, bboxes: List[BoundingBox], frame_idx: int) -> List[Tuple[int, BoundingBox]]:
        return self.tracker.update(bboxes, frame_idx)
    
    def finish(self) -> Dict[int, List[FrameRecord]]:
        return self.tracker.finish()
