"""Centroid-distance based mask tracker.

Tracks person segmentation masks across frames using centroid proximity,
decoupled from ByteTrack. This ensures every detected person mask is
assigned to a tracked actor, even if ByteTrack misses them.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MaskActor:
    """Represents a tracked person via their segmentation mask across frames.

    Attributes:
        actor_id: Unique identifier for this actor.
        frames: List of (frame_idx, bool_mask, mask_area, bbox_area) entries.
        last_centroid: Most recent centroid (cx, cy) of the mask.
        last_frame_idx: Frame index of the most recent detection.
        closed: Whether this actor has been closed (no match for too long).
        frame_indices: Set of frame indices where this actor was detected.
    """

    actor_id: int
    frames: List[Tuple[int, np.ndarray, int, int]] = field(default_factory=list)
    # Each entry: (frame_idx, bool_mask, mask_area, bbox_area)
    # bool_mask is a boolean 2D array at original frame resolution
    last_centroid: Tuple[float, float] = (0.0, 0.0)
    last_frame_idx: int = -1
    closed: bool = False
    frame_indices: set = field(default_factory=set)


class MaskTracker:
    """Centroid-distance based mask tracker.

    Tracks masks from a person segmenter across frames by matching
    mask centroids to previously seen actors. Each mask in every
    frame gets assigned to either an existing actor (if centroid
    distance is below threshold) or a new actor.

    Args:
        max_lost_frames: Close an actor after this many frames without a match.
        match_threshold_px: Maximum centroid distance (pixels) to consider a match.
        min_mask_area: Minimum mask area (pixels) to consider for tracking.
            Smaller masks are filtered out as false positives or debris.
    """

    def __init__(
        self,
        max_lost_frames: int = 30,
        match_threshold_px: float = 150.0,
        min_mask_area: int = 20000,
    ):
        self.max_lost_frames = max_lost_frames
        self.match_threshold_px = match_threshold_px
        self.min_mask_area = min_mask_area
        self._next_actor_id = 0
        self._actors: Dict[int, MaskActor] = {}

    def _compute_centroid(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Compute centroid of a boolean mask.

        Args:
            mask: Boolean 2D array where True = person pixel.

        Returns:
            (cx, cy) centroid coordinates, or None if mask is empty.
        """
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        return (float(xs.mean()), float(ys.mean()))

    def _compute_mask_bbox(
        self, mask: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute bounding box from mask pixels.

        Returns:
            (x1, y1, x2, y2) or None if mask is empty.
        """
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def update(
        self,
        frame_idx: int,
        masks: List[Tuple[int, np.ndarray]],
    ) -> None:
        """Process all masks for a frame, assigning each to a MaskActor.

        Args:
            frame_idx: Current frame index.
            masks: List of (detection_idx, bool_mask) from PersonSegmenter.
        """
        if not masks:
            # No masks this frame — check for actors to close
            self._maybe_close_actors(frame_idx)
            return

        # Pre-compute centroids and match info for all masks this frame
        mask_infos: List[Tuple[np.ndarray, Tuple[float, float], int]] = []
        for _, mask in masks:
            centroid = self._compute_centroid(mask)
            if centroid is None:
                continue
            area = int(mask.sum())
            mask_infos.append((mask, centroid, area))

        # Filter out tiny masks (false positives / debris from scene transitions)
        original_count = len(mask_infos)
        mask_infos = [
            (mask, centroid, area)
            for mask, centroid, area in mask_infos
            if area >= self.min_mask_area
        ]
        skipped = original_count - len(mask_infos)
        if skipped > 0:
            print(
                f"[MaskTracker] Frame {frame_idx}: skipped {skipped} tiny masks "
                f"(< {self.min_mask_area} px)"
            )

        # Track which actors get matched this frame
        matched_actor_ids: set = set()

        # For each mask, find the closest active (non-closed) actor
        # that hasn't been matched to another mask in this frame
        for mask, centroid, area in mask_infos:
            best_actor_id: Optional[int] = None
            best_dist = float("inf")

            for aid, actor in self._actors.items():
                if actor.closed or aid in matched_actor_ids:
                    continue
                dist = (
                    (centroid[0] - actor.last_centroid[0]) ** 2
                    + (centroid[1] - actor.last_centroid[1]) ** 2
                ) ** 0.5

                # Area ratio penalty: if current mask area is very different
                # from actor's historical average, increase effective distance
                if actor.frames:
                    avg_area = np.mean([f[2] for f in actor.frames])
                    area_ratio = max(area, avg_area) / (min(area, avg_area) + 1)
                    if area_ratio > 3.0:
                        # Penalize: increase distance proportional to ratio mismatch
                        dist *= area_ratio / 3.0

                if dist < best_dist:
                    best_dist = dist
                    best_actor_id = aid

            if best_actor_id is not None and best_dist < self.match_threshold_px:
                # Assign to existing actor
                actor = self._actors[best_actor_id]
                mbbox = self._compute_mask_bbox(mask)
                bbox_area = (
                    int((mbbox[2] - mbbox[0]) * (mbbox[3] - mbbox[1]))
                    if mbbox
                    else area
                )
                actor.frames.append((frame_idx, mask, area, bbox_area))
                actor.last_centroid = centroid
                actor.last_frame_idx = frame_idx
                actor.frame_indices.add(frame_idx)
                matched_actor_ids.add(best_actor_id)
            else:
                # Create new actor
                actor = MaskActor(
                    actor_id=self._next_actor_id,
                    last_centroid=centroid,
                    last_frame_idx=frame_idx,
                )
                self._next_actor_id += 1
                mbbox = self._compute_mask_bbox(mask)
                bbox_area = (
                    int((mbbox[2] - mbbox[0]) * (mbbox[3] - mbbox[1]))
                    if mbbox
                    else area
                )
                actor.frames.append((frame_idx, mask, area, bbox_area))
                actor.frame_indices.add(frame_idx)
                self._actors[actor.actor_id] = actor
                matched_actor_ids.add(actor.actor_id)

        # Close actors that haven't been seen for too long
        self._maybe_close_actors(frame_idx)

    def _maybe_close_actors(self, frame_idx: int) -> None:
        """Close actors with no match for > max_lost_frames."""
        for aid, actor in self._actors.items():
            if actor.closed:
                continue
            if (
                actor.last_frame_idx >= 0
                and (frame_idx - actor.last_frame_idx) > self.max_lost_frames
            ):
                actor.closed = True
                print(
                    f"[MaskTracker] Closed actor_{aid}: no match for "
                    f"{frame_idx - actor.last_frame_idx} frames"
                )

    def get_active_actors(self) -> Dict[int, MaskActor]:
        """Return all actors (active and closed)."""
        return self._actors

    def finish(self) -> Dict[int, MaskActor]:
        """Finalize tracking, return all actors.

        Returns:
            Dict mapping actor_id to MaskActor with all frame data.
        """
        # Close any remaining active actors (end of video)
        for actor in self._actors.values():
            if not actor.closed:
                actor.closed = True
        return self._actors

    def get_mask_bboxes(
        self,
    ) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
        """Get approximate bounding boxes for each actor from mask data.

        For identity clustering, we need per-frame bboxes. This reconstructs
        approximate bboxes from the stored masks.

        Returns:
            {actor_id: [(frame_idx, x1, y1, x2, y2), ...]}
        """
        # We store bool masks now, so bboxes can be recomputed directly.
        # Return empty dict — identity module uses centroid-only mode.
        return {}
