"""Main ComfyUI node: VideoActorExtractor.

Detects, tracks, identifies, and extracts individual actors from video
with green screen background output.

Supports two input modes:
  1. IMAGE batch from VHS LoadVideo node (preferred)
  2. video_path string (fallback, for direct file input)
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_reader import get_video_info, extract_frames
from core.config import (
    DEFAULT_MAX_ACTORS,
    DEFAULT_FACE_THRESHOLD,
    DEFAULT_FPS_SAMPLE,
    DEFAULT_MIN_TRACK_LENGTH,
    DEFAULT_YOLO_MODEL,
    GREEN_SCREEN_COLOR,
    SEGMENT_GAP_SEC,
)

# DEFAULT_SEG_MODEL may not exist yet (parallel task adding it)
try:
    from core.config import DEFAULT_SEG_MODEL
except ImportError:
    DEFAULT_SEG_MODEL = "yolov8n-seg.pt"
from pipeline.detector import PersonDetector, BoundingBox
from pipeline.tracker import PersonTracker
from pipeline.identity import IdentityCluster
from pipeline.cropper import ActorCropper
from pipeline.merger import merge_segments, generate_actor_json

# PersonSegmenter may not exist yet (parallel task creating it)
try:
    from pipeline.segmenter import PersonSegmenter
except ImportError:
    PersonSegmenter = None

# Re-import FrameRecord for type hints in helper functions
from pipeline.tracker import FrameRecord


def find_best_mask(
    rec: FrameRecord, masks: List[Tuple[int, np.ndarray]]
) -> Optional[np.ndarray]:
    """Match a FrameRecord bbox to the best mask by centroid proximity.

    Args:
        rec: FrameRecord with bounding box coordinates.
        masks: List of (detection_idx, bool_mask_H_W) from PersonSegmenter.

    Returns:
        The mask with smallest Euclidean distance between bbox center and
        mask centroid, or None if masks is empty.
    """
    if not masks:
        return None

    cx = (rec.x1 + rec.x2) / 2.0
    cy = (rec.y1 + rec.y2) / 2.0

    best_mask = None
    best_dist = float("inf")

    for _, mask in masks:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        mask_cx = float(xs.mean())
        mask_cy = float(ys.mean())
        dist = ((cx - mask_cx) ** 2 + (cy - mask_cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_mask = mask

    return best_mask


def apply_bbox_green(frame: np.ndarray, rec: FrameRecord) -> np.ndarray:
    """Create a green-screened frame keeping only the bbox region original.

    Args:
        frame: BGR uint8 image.
        rec: FrameRecord with bounding box coordinates.

    Returns:
        BGR frame where everything outside the bbox is green (0, 255, 0).
    """
    h, w = frame.shape[:2]
    result = np.full_like(frame, (0, 255, 0), dtype=np.uint8)

    # Clamp bbox to frame boundaries
    x1 = max(0, int(rec.x1))
    y1 = max(0, int(rec.y1))
    x2 = min(w, int(rec.x2))
    y2 = min(h, int(rec.y2))

    if x2 > x1 and y2 > y1:
        result[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    return result


def _get_comfyui_output_dir() -> str:
    """Get ComfyUI output directory."""
    try:
        import folder_paths

        base = folder_paths.get_output_directory()
    except ImportError:
        # Fallback for testing outside ComfyUI
        base = os.path.join(os.path.expanduser("~"), "output")

    output_dir = os.path.join(base, "ComfyUI-VideoActorExtract")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


class VideoActorExtractor:
    """
    Main extraction pipeline node for ComfyUI.

    Input:
        - video_path: path to input video
        - max_actors: maximum number of actors to detect
        - face_threshold: face similarity threshold for identity merging
        - fps_sample: frames per second to sample for processing
        - min_track_length: minimum number of frames for a valid track

    Output:
        - actor_info_json: JSON string with actor information
        - output_dir: path to output directory containing per-actor MP4s
        - actor_preview_images: IMAGE tensor with top 5 preview frames per actor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "model_path": ("STRING", {"default": "yolov8n.pt", "multiline": False}),
                "seg_model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_SEG_MODEL,
                        "multiline": False,
                        "tooltip": "YOLOv8-seg model for person segmentation",
                    },
                ),
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional: original video path for metadata. If empty, metadata is estimated.",
                    },
                ),
                "max_actors": (
                    "INT",
                    {"default": DEFAULT_MAX_ACTORS, "min": 1, "max": 50},
                ),
                "face_threshold": (
                    "FLOAT",
                    {
                        "default": DEFAULT_FACE_THRESHOLD,
                        "min": 0.1,
                        "max": 0.99,
                        "step": 0.05,
                    },
                ),
                "min_track_length": (
                    "INT",
                    {"default": DEFAULT_MIN_TRACK_LENGTH, "min": 1, "max": 100},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("actor_info_json", "output_dir", "actor_preview_images")
    FUNCTION = "extract"
    CATEGORY = "video/actor"

    def extract(
        self,
        images: torch.Tensor,
        model_path: str = DEFAULT_YOLO_MODEL,
        seg_model_path: str = DEFAULT_SEG_MODEL,
        video_path: str = "",
        max_actors: int = DEFAULT_MAX_ACTORS,
        face_threshold: float = DEFAULT_FACE_THRESHOLD,
        min_track_length: int = DEFAULT_MIN_TRACK_LENGTH,
    ) -> Tuple[str, str, torch.Tensor]:
        """
        Run the full actor extraction pipeline.

        Args:
            images: IMAGE tensor from VHS LoadVideo, shape [B, H, W, C], RGB, 0-1
            model_path: YOLOv8 model path
            seg_model_path: YOLOv8-seg model path for person segmentation
            video_path: Optional original video path for metadata
            max_actors: Maximum number of actors to detect
            face_threshold: Face similarity threshold for identity merging
            min_track_length: Minimum number of frames for a valid track

        Returns:
            (actor_info_json, output_dir, actor_preview_images)
        """

        output_dir = _get_comfyui_output_dir()
        print(f"[VideoActorExtract] Output directory: {output_dir}")

        # Convert IMAGE tensor to numpy frames (RGB 0-1 -> BGR 0-255)
        # images shape: [B, H, W, C]
        print(f"[VideoActorExtract] Received IMAGE batch: {images.shape}")

        # Handle different tensor types
        if isinstance(images, torch.Tensor):
            # Convert to numpy, squeeze if needed
            imgs_np = images.cpu().numpy()
        else:
            imgs_np = np.array(images)

        # Ensure 4D: [B, H, W, C]
        # VHS LoadVideo IMAGE tensor is already [B, H, W, C] format
        if imgs_np.ndim == 3:
            imgs_np = imgs_np[np.newaxis, ...]
        # If already 4D, use as-is (no squeeze needed)

        num_frames = imgs_np.shape[0]
        img_h = imgs_np.shape[1]
        img_w = imgs_np.shape[2]

        print(f"  Total frames: {num_frames}, Resolution: {img_w}x{img_h}")

        # Convert RGB (0-1) to BGR (0-255)
        frames = np.clip(imgs_np * 255.0, 0, 255).astype(np.uint8)
        # RGB -> BGR
        frames = frames[:, :, :, ::-1].copy()

        # Build frame lookup
        frame_lookup = {i: frames[i] for i in range(num_frames)}

        # Estimate video metadata if no video_path provided
        if video_path and os.path.exists(video_path):
            from core.video_reader import get_video_info

            try:
                video_info = get_video_info(video_path)
                fps = video_info.fps
                total_frames = video_info.total_frames
                orig_width = video_info.width
                orig_height = video_info.height
            except Exception:
                fps = 30.0
                total_frames = num_frames
                orig_width = img_w
                orig_height = img_h
        else:
            fps = 30.0
            total_frames = num_frames
            orig_width = img_w
            orig_height = img_h

        print(f"  Estimated FPS: {fps}, Total frames: {total_frames}")

        # Step 3: Detect persons (process all frames since we already have them)
        print(f"[VideoActorExtract] Step 2: Running person detection...")
        detector = PersonDetector(model=model_path)
        all_bboxes = detector.detect_batch(frames)

        total_detections = sum(len(b) for b in all_bboxes)
        print(f"  Total detections: {total_detections} across {num_frames} frames")

        # Step 4: Track persons
        print("[VideoActorExtract] Step 3: Running multi-object tracking...")
        tracker = PersonTracker(fps=fps)

        for i, bboxes in enumerate(all_bboxes):
            tracker.update(bboxes, i)

        track_records = tracker.finish()

        # Filter short tracks
        long_tracks = {
            tid: recs
            for tid, recs in track_records.items()
            if len(recs) >= min_track_length
        }
        print(
            f"  Found {len(track_records)} tracks, {len(long_tracks)} with >= {min_track_length} frames"
        )

        if not long_tracks:
            # No actors found, return empty result with empty preview tensor
            empty_json = json.dumps(
                {
                    "actors": [],
                    "video_info": {
                        "total_frames": total_frames,
                        "fps": fps,
                        "width": orig_width,
                        "height": orig_height,
                        "duration_sec": total_frames / fps if fps > 0 else 0,
                    },
                },
                indent=2,
            )
            empty_preview = torch.zeros(0, img_h, img_w, 3, dtype=torch.float32)
            return (empty_json, output_dir, empty_preview)

        # Step 5: Identity clustering
        print("[VideoActorExtract] Step 4: Clustering actor identities...")
        identity = IdentityCluster(threshold=face_threshold)
        track_to_actor = identity.cluster_tracks(
            long_tracks, frame_lookup, min_track_length
        )

        # Group tracks by actor
        actor_tracks: Dict[str, List[int]] = {}
        for tid, actor_id in track_to_actor.items():
            if actor_id not in actor_tracks:
                actor_tracks[actor_id] = []
            actor_tracks[actor_id].append(tid)

        # Limit number of actors
        actor_ids_sorted = sorted(actor_tracks.keys())[:max_actors]
        print(f"  Identified {len(actor_ids_sorted)} actors (max: {max_actors})")

        # Step 5: Initialize PersonSegmenter
        print("[VideoActorExtract] Step 5: Initializing person segmenter...")
        segmenter = (
            PersonSegmenter(model_path=seg_model_path) if PersonSegmenter else None
        )
        if segmenter is None:
            print("  PersonSegmenter not available, using bbox green-screen fallback")

        # Step 5.5: Run segmentation on ALL frames, building per-track masked frame collections
        print(
            "[VideoActorExtract] Step 5.5: Segmenting frames and building masked tracks..."
        )
        track_masked_frames: Dict[int, List[Tuple[int, np.ndarray, int]]] = {
            tid: [] for tid in long_tracks
        }

        for frame_idx in range(num_frames):
            frame = frames[frame_idx]
            masks = segmenter.detect_masks(frame) if segmenter else []

            for tid in long_tracks:
                matching_recs = [
                    r for r in long_tracks[tid] if r.frame_idx == frame_idx
                ]
                for rec in matching_recs:
                    if masks:
                        best_mask = find_best_mask(rec, masks)
                        if best_mask is not None:
                            masked = segmenter.apply_mask(frame, best_mask)
                            area = int(best_mask.sum())
                        else:
                            masked = apply_bbox_green(frame, rec)
                            area = int((rec.x2 - rec.x1) * (rec.y2 - rec.y1))
                    else:
                        masked = apply_bbox_green(frame, rec)
                        area = int((rec.x2 - rec.x1) * (rec.y2 - rec.y1))
                    track_masked_frames[tid].append((frame_idx, masked, area))

            if (frame_idx + 1) % 50 == 0 or frame_idx == num_frames - 1:
                print(f"  Processed {frame_idx + 1}/{num_frames} frames")

        # Step 6: Identity clustering (already done above in Step 4)
        # Collect masked frames per actor from their tracks
        print("[VideoActorExtract] Step 6: Generating output videos and JSON...")
        video_info_dict = {
            "total_frames": total_frames,
            "fps": fps,
            "width": orig_width,
            "height": orig_height,
            "duration_sec": total_frames / fps if fps > 0 else 0,
        }
        actor_data_for_json = {}
        all_preview_frames: List[np.ndarray] = []

        for actor_id in actor_ids_sorted:
            tids = actor_tracks[actor_id]

            # Collect all (frame_idx, masked_frame, area) from all tracks of this actor
            actor_all_frames: List[Tuple[int, np.ndarray, int]] = []
            for tid in tids:
                actor_all_frames.extend(track_masked_frames.get(tid, []))

            if not actor_all_frames:
                continue

            # Sort by frame_idx
            actor_all_frames.sort(key=lambda x: x[0])

            # Select top 5 preview frames by mask area (descending)
            top5 = sorted(actor_all_frames, key=lambda x: x[2], reverse=True)[:5]
            for _, masked_bgr, _ in top5:
                # Convert BGR uint8 -> RGB float32 normalized 0-1
                rgb = masked_bgr[:, :, ::-1].copy()
                rgb_float = rgb.astype(np.float32) / 255.0
                all_preview_frames.append(rgb_float)

            # Split into segments (gap > 30 frames between consecutive frame_idxs)
            segments_frames: List[List[np.ndarray]] = []
            current_segment: List[np.ndarray] = [actor_all_frames[0][1]]
            segments_info = [
                {
                    "start_frame": actor_all_frames[0][0],
                    "end_frame": actor_all_frames[0][0],
                    "frame_count": 1,
                }
            ]

            for i in range(1, len(actor_all_frames)):
                gap = actor_all_frames[i][0] - actor_all_frames[i - 1][0]
                if gap > 30:  # ~1 second gap at 30fps means new segment
                    segments_frames.append(current_segment)
                    current_segment = []
                    segments_info.append(
                        {
                            "start_frame": actor_all_frames[i][0],
                            "end_frame": actor_all_frames[i][0],
                            "frame_count": 0,
                        }
                    )
                current_segment.append(actor_all_frames[i][1])
                segments_info[-1]["end_frame"] = actor_all_frames[i][0]
                segments_info[-1]["frame_count"] += 1

            segments_frames.append(current_segment)

            if not segments_frames or not any(segments_frames):
                continue

            # Filter empty segments
            segments_frames = [seg for seg in segments_frames if len(seg) > 0]
            segments_info = [s for s in segments_info if s["frame_count"] > 0]

            if not segments_frames:
                continue

            # Encode to video via merge_segments (no gaps, no green transitions)
            output_video_path = os.path.join(output_dir, f"{actor_id}.mp4")
            success = merge_segments(
                segments_frames,
                fps=fps,
                output_path=output_video_path,
                gap_sec=SEGMENT_GAP_SEC,
            )

            if success:
                print(
                    f"  {actor_id}: {len(segments_frames)} segments, "
                    f"{sum(s['frame_count'] for s in segments_info)} frames -> {output_video_path}"
                )

            actor_data_for_json[actor_id] = {
                "segments": segments_info,
                "total_frames": sum(s["frame_count"] for s in segments_info),
                "segment_count": len(segments_info),
            }

        # Step 8: Build preview tensor
        if all_preview_frames:
            preview_tensor = torch.from_numpy(np.stack(all_preview_frames, axis=0))
        else:
            preview_tensor = torch.zeros(0, img_h, img_w, 3, dtype=torch.float32)

        # Generate JSON
        json_path = os.path.join(output_dir, "actor_info.json")
        generate_actor_json(actor_data_for_json, video_info_dict, json_path)

        with open(json_path, "r") as f:
            json_content = f.read()

        print(f"[VideoActorExtract] Done! Output in {output_dir}")
        return (json_content, output_dir, preview_tensor)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "VideoActorExtractor": VideoActorExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoActorExtractor": "Video Actor Extract",
}
