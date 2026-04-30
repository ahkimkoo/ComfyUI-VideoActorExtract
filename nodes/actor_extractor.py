"""Main ComfyUI node: VideoActorExtractor.

Detects, tracks, and extracts individual actors from video
with green screen background output.

Uses a mask-centric pipeline:
  1. PersonSegmenter (YOLOv8-seg) detects all person masks per frame
  2. MaskTracker tracks masks across frames via centroid distance
  3. Each tracked mask becomes one output video (no face merging)
  4. Continuous segments are built with gap interpolation
  5. Per-actor videos and JSON are generated

Supports two input modes:
  1. IMAGE batch from VHS LoadVideo node (preferred)
  2. video_path string (fallback, for direct file input)
"""

import os
import sys
import json
import time
import uuid

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_reader import get_video_info
from core.config import (
    DEFAULT_MAX_ACTORS,
    DEFAULT_FACE_THRESHOLD,
    DEFAULT_MIN_TRACK_LENGTH,
    DEFAULT_YOLO_MODEL,
    DEFAULT_MAX_LOST_FRAMES,
    SEGMENT_GAP_SEC,
    BG_COLOR_MAP,
    BG_COLOR_GREEN,
)

# DEFAULT_SEG_MODEL may not exist yet (parallel task adding it)
try:
    from core.config import DEFAULT_SEG_MODEL
except ImportError:
    DEFAULT_SEG_MODEL = "yolov8n-seg.pt"
from pipeline.identity import IdentityCluster

# face_threshold is kept for backward compatibility but identity clustering
# now uses spatial-temporal constraints to prevent merging co-occurring tracks.
from pipeline.merger import merge_segments, generate_actor_json

# PersonSegmenter may not exist yet (parallel task creating it)
try:
    from pipeline.segmenter import PersonSegmenter
except ImportError:
    PersonSegmenter = None

# MaskTracker — our new centroid-distance mask tracker
from pipeline.mask_tracker import MaskTracker, MaskActor


def _get_comfyui_output_dir() -> str:
    """Get a unique ComfyUI output directory for this run."""
    try:
        import folder_paths

        base = folder_paths.get_output_directory()
    except ImportError:
        # Fallback for testing outside ComfyUI
        base = os.path.join(os.path.expanduser("~"), "output")

    run_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(base, "ComfyUI-VideoActorExtract", run_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _resolve_model_path(name: str) -> str:
    """Resolve a model filename to an absolute path.

    Resolution order:
    1. If already an absolute path and file exists, return as-is.
    2. Try ComfyUI folder_paths "video-actor-extract" folder.
    3. Try ~/App/ComfyUI/models/video-actor-extract/.
    4. Return the bare name (backward compat / let ultralytics download it).
    """
    # Already absolute and exists?
    if os.path.isabs(name) and os.path.isfile(name):
        return name

    # 1. Try ComfyUI folder_paths
    try:
        import folder_paths

        paths = folder_paths.get_folder_paths("video-actor-extract")
        for p in paths:
            candidate = os.path.join(p, name)
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass

    # 2. Try default ComfyUI models directory
    default_dir = os.path.expanduser("~/App/ComfyUI/models/video-actor-extract/")
    candidate = os.path.join(default_dir, name)
    if os.path.isfile(candidate):
        return candidate

    # 3. Return bare name (backward compat)
    return name


def _build_continuous_segments(
    actor_frames: List[Tuple[int, np.ndarray, int]],
    frame_shape: Tuple[int, int],
    max_gap: int = 30,
    interp_gap: int = 2,
) -> Tuple[List[List[Tuple[int, np.ndarray, int]]], List[dict]]:
    """Build continuous output segments from sparse frame detections.

    For each actor:
    1. Sort detections by frame_idx.
    2. Split into segments where gap > max_gap frames.
    3. Within each segment, interpolate small gaps (<= interp_gap frames)
       by duplicating the previous available bool mask.
    4. Output continuous segment list where frame_count == end_frame - start_frame + 1.

    Green screen compositing is NOT done here — it happens later in the
    encoding step so we avoid storing full BGR frames in memory.

    Args:
        actor_frames: Sorted list of (frame_idx, bool_mask, mask_area).
        frame_shape: (height, width) of original frames, used for green
            fallback during interpolation.
        max_gap: Gap in frames that triggers a new segment.
        interp_gap: Max gap within a segment to interpolate by duplication.

    Returns:
        (segments_frames, segments_info) where:
        - segments_frames: List of segment-lists (one per segment), each
          containing (frame_idx, bool_mask, mask_area) tuples — continuous
          from start_frame to end_frame.
        - segments_info: List of dicts with start_frame, end_frame, frame_count.
    """
    if not actor_frames:
        return [], []

    # --- Step 1: Group into segments by max_gap ---
    raw_segments: List[List[Tuple[int, np.ndarray, int]]] = []
    current_seg: List[Tuple[int, np.ndarray, int]] = [actor_frames[0]]

    for i in range(1, len(actor_frames)):
        gap = actor_frames[i][0] - actor_frames[i - 1][0]
        if gap > max_gap:
            raw_segments.append(current_seg)
            current_seg = []
        current_seg.append(actor_frames[i])
    raw_segments.append(current_seg)

    # --- Step 2: Interpolate each segment to be continuous ---
    h, w = frame_shape
    segments_frames: List[List[Tuple[int, np.ndarray, int]]] = []
    segments_info: List[dict] = []

    for seg in raw_segments:
        if not seg:
            continue

        start_frame = seg[0][0]
        end_frame = seg[-1][0]

        # Build a lookup: frame_idx -> (bool_mask, area)
        frame_map: Dict[int, Tuple[np.ndarray, int]] = {
            entry[0]: (entry[1], entry[2]) for entry in seg
        }

        # Walk from start_frame to end_frame, interpolating small gaps
        continuous: List[Tuple[int, np.ndarray, int]] = []
        prev_mask: Optional[np.ndarray] = None

        for fi in range(start_frame, end_frame + 1):
            if fi in frame_map:
                mask, area = frame_map[fi]
                continuous.append((fi, mask, area))
                prev_mask = mask
            elif prev_mask is not None:
                # Duplicate the previous bool mask (gap interpolation)
                continuous.append((fi, prev_mask, int(prev_mask.sum())))
            else:
                # Fallback: empty mask (all False)
                empty_mask = np.zeros((h, w), dtype=bool)
                continuous.append((fi, empty_mask, 0))
                prev_mask = empty_mask

        if continuous:
            segments_frames.append(continuous)
            segments_info.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_count": end_frame - start_frame + 1,
                }
            )

    return segments_frames, segments_info


def _actor_to_synthetic_records(
    actor: MaskActor,
) -> List:
    """Build approximate FrameRecord-like objects from a MaskActor's data.

    For identity clustering, we compute an approximate bounding box from
    the stored bool mask directly (True = person pixel).

    Args:
        actor: MaskActor with frames list of (frame_idx, bool_mask, area, bbox_area).

    Returns:
        List of FrameRecord-like objects for identity clustering.
    """
    from pipeline.tracker import FrameRecord

    records = []
    for frame_idx, mask, area, bbox_area in actor.frames:
        # Compute bbox from the bool mask directly
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        x1, y1, x2, y2 = (
            float(xs.min()),
            float(ys.min()),
            float(xs.max()),
            float(ys.max()),
        )
        records.append(FrameRecord(frame_idx=frame_idx, x1=x1, y1=y1, x2=x2, y2=y2))
    return records


class VideoActorExtractor:
    """
    Main extraction pipeline node for ComfyUI.

    Input:
        - images: IMAGE tensor from VHS LoadVideo
        - model_path: YOLOv8 detection model path
        - seg_model_path: YOLOv8-seg model path
        - video_path: optional original video path for metadata
        - max_actors: maximum number of actors to detect
        - face_threshold: face similarity threshold for identity merging
        - min_track_length: minimum number of frames for a valid track

    Output:
        - actor_info_json: JSON string with actor information
        - output_dir: path to output directory containing per-actor MP4s
        - actor_count: number of detected actors
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
                        "default": "yolov8n-seg.pt",
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
                "skip_every_n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 30,
                        "step": 1,
                        "tooltip": "Process every Nth frame. 1=all frames, 2=every other frame, etc. Higher values are faster but less precise.",
                    },
                ),
                "bg_color": (
                    ["green", "blue", "black", "white"],
                    {"default": "green"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("actor_info_json", "output_dir", "actor_count")
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
        skip_every_n: int = 1,
        bg_color: str = "green",
    ) -> Tuple[str, str, int]:
        """
        Run the full actor extraction pipeline.

        Pipeline:
        1. Convert IMAGE tensor to numpy BGR frames
        2. Initialize PersonSegmenter and MaskTracker
        3. For each frame: detect masks, pass to MaskTracker
        4. Finish tracking, get all MaskActors
        5. Filter short actors — each track becomes one output video
        6. Build continuous segments with interpolation
        7. Encode videos and generate JSON

        Args:
            images: IMAGE tensor from VHS LoadVideo, shape [B, H, W, C], RGB, 0-1
            model_path: YOLOv8 model path
            seg_model_path: YOLOv8-seg model path for person segmentation
            video_path: Optional original video path for metadata
            max_actors: Maximum number of actors to detect (deprecated, kept for compat)
            face_threshold: Face similarity threshold (deprecated, kept for compat)
            min_track_length: Minimum number of frames for a valid track
            skip_every_n: Process every Nth frame (1=all, 2=every other, etc.)

        Returns:
            (actor_info_json, output_dir, actor_count)
        """

        # Check and auto-download missing models
        from core.model_utils import ensure_models_exist

        ensure_models_exist()

        output_dir = _get_comfyui_output_dir()
        print(f"[VideoActorExtract] Output directory: {output_dir}")

        # Resolve model paths
        model_path = _resolve_model_path(model_path)
        seg_model_path = _resolve_model_path(seg_model_path)
        print(f"[VideoActorExtract] Resolved model_path: {model_path}")
        print(f"[VideoActorExtract] Resolved seg_model_path: {seg_model_path}")

        # ----------------------------------------------------------------
        # Determine frame dimensions from IMAGE tensor (no bulk conversion)
        # ----------------------------------------------------------------
        print(f"[VideoActorExtract] Received IMAGE batch: {images.shape}")

        # images shape: [B, H, W, C] or [H, W, C], RGB float32 0-1
        if isinstance(images, torch.Tensor):
            _sample = images[0] if images.ndim == 4 else images
            img_h = _sample.shape[0]
            img_w = _sample.shape[1]
        else:
            arr = np.array(images)
            if arr.ndim == 3:
                img_h, img_w = arr.shape[0], arr.shape[1]
            else:
                img_h, img_w = arr.shape[1], arr.shape[2]

        num_frames = images.shape[0] if images.ndim == 4 else 1

        print(f"  Total frames: {num_frames}, Resolution: {img_w}x{img_h}")

        # ----------------------------------------------------------------
        # Lazy frame getter: converts ONE frame at a time (~6MB each)
        # ----------------------------------------------------------------
        def _get_frame_bgr(idx: int) -> np.ndarray:
            """Convert a single frame from tensor to BGR uint8 numpy array."""
            frame_rgb = images[idx].cpu().numpy()  # (H, W, 3) float32, ~6MB
            return (frame_rgb * 255.0).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()

        # frame_lookup as a lazy dict — IdentityCluster.cluster_tracks uses .get()
        class _LazyFrameLookup(dict):
            """Dict-like that converts frames on-demand instead of bulk."""

            def __init__(self, n: int):
                super().__init__()
                self._n = n
                self._cache: Dict[int, np.ndarray] = {}

            def __getitem__(self, key: int) -> np.ndarray:
                if key not in self._cache:
                    self._cache[key] = _get_frame_bgr(key)
                return self._cache[key]

            def get(self, key: int, default=None):
                try:
                    return self[key]
                except (IndexError, KeyError):
                    return default

            def __contains__(self, key):
                return 0 <= key < self._n

            def __len__(self):
                return self._n

            def __iter__(self):
                return iter(range(self._n))

        frame_lookup = _LazyFrameLookup(num_frames)

        # ----------------------------------------------------------------
        # Estimate video metadata
        # ----------------------------------------------------------------
        if video_path and os.path.exists(video_path):
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

        # ----------------------------------------------------------------
        # Step 1: Initialize PersonSegmenter
        # ----------------------------------------------------------------
        print("[VideoActorExtract] Step 1: Initializing person segmenter...")
        if PersonSegmenter is None:
            print("  ERROR: PersonSegmenter not available. Cannot proceed.")
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
            return (empty_json, output_dir, 0)

        segmenter = PersonSegmenter(model_path=seg_model_path)

        # ----------------------------------------------------------------
        # Step 2: Initialize MaskTracker
        # ----------------------------------------------------------------
        print("[VideoActorExtract] Step 2: Initializing mask tracker...")
        mask_tracker = MaskTracker(
            max_lost_frames=DEFAULT_MAX_LOST_FRAMES,
            match_threshold_px=150.0,
            min_mask_area=20000,
        )

        # ----------------------------------------------------------------
        # Step 3: Process all frames — detect masks and track them
        # ----------------------------------------------------------------
        actual_frames = list(range(0, num_frames, skip_every_n))
        total_to_process = len(actual_frames)
        print(
            f"[VideoActorExtract] Step 3: Detecting masks and tracking actors..."
            f" ({total_to_process}/{num_frames} frames, skip_every_n={skip_every_n})"
        )
        t0 = time.time()
        t_detect = 0.0
        t_track = 0.0
        for frame_idx in actual_frames:
            frame_bgr = _get_frame_bgr(frame_idx)  # ~6MB per frame

            t1 = time.time()
            masks = segmenter.detect_masks(frame_bgr)
            t_detect += time.time() - t1

            t2 = time.time()
            mask_tracker.update(frame_idx, masks)
            t_track += time.time() - t2

            done = actual_frames.index(frame_idx) + 1
            if done % 50 == 0 or frame_idx == actual_frames[-1]:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  Processed {done}/{total_to_process} frames "
                    f"({elapsed:.1f}s, {rate:.1f} fps)"
                )

        elapsed_total = time.time() - t0
        print(
            f"  Frame processing complete in {elapsed_total:.1f}s "
            f"(Detection: {t_detect:.1f}s, Tracking: {t_track:.1f}s)"
        )

        # ----------------------------------------------------------------
        # Step 4: Finish tracking, get all MaskActors
        # ----------------------------------------------------------------
        all_actors = mask_tracker.finish()
        total_mask_detections = sum(len(a.frames) for a in all_actors.values())
        print(
            f"[VideoActorExtract] Step 4: Tracking complete. "
            f"{len(all_actors)} mask actors, {total_mask_detections} total detections"
        )

        if not all_actors:
            print("  No actors found with sufficient detections.")
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
            return (empty_json, output_dir, 0)

        # ----------------------------------------------------------------
        # Step 5: Filter short actors
        # ----------------------------------------------------------------
        long_actors: Dict[int, MaskActor] = {
            aid: actor
            for aid, actor in all_actors.items()
            if len(actor.frames) >= min_track_length
        }
        print(
            f"  {len(long_actors)} actors with >= {min_track_length} frames "
            f"(filtered out {len(all_actors) - len(long_actors)} short actors)"
        )

        if not long_actors:
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
            return (empty_json, output_dir, 0)

        # ----------------------------------------------------------------
        # Step 5a: Split mixed tracks (tracks containing different people)
        # ----------------------------------------------------------------
        # Some tracked masks may have switched identity mid-track (e.g.
        # tracker follows child, then adult appears in same mask region).
        # We detect and split these BEFORE identity clustering so each
        # subtrack gets an independent clustering decision.
        pre_split_records = {}
        for aid, actor in long_actors.items():
            pre_split_records[aid] = _actor_to_synthetic_records(actor)

        split_records, split_to_original = IdentityCluster._split_mixed_tracks(
            pre_split_records
        )

        if len(split_records) > len(long_actors):
            # Splitting happened — create new MaskActor objects for subtracks
            #
            # _split_mixed_tracks re-numbers IDs sequentially from 0, so
            # we need to build a reverse mapping: original_id -> [new_ids].
            # If multiple new IDs map to the same original, that track was split.
            orig_to_splits: Dict[int, List[int]] = {}
            for new_tid, orig_aid in split_to_original.items():
                orig_to_splits.setdefault(orig_aid, []).append(new_tid)

            new_long_actors: Dict[int, MaskActor] = {}

            for orig_aid, new_tids in orig_to_splits.items():
                if len(new_tids) == 1 and new_tids[0] == orig_aid:
                    # Not split and ID is unchanged — keep original MaskActor
                    new_long_actors[orig_aid] = long_actors[orig_aid]
                else:
                    # Split (or ID was renumbered) — reconstruct subtracks
                    original_actor = long_actors[orig_aid]

                    for new_tid in new_tids:
                        records = split_records[new_tid]
                        subtrack_frame_indices = set(r.frame_idx for r in records)

                        sub_frames = [
                            (fi, frame, area, bbox_area)
                            for fi, frame, area, bbox_area in original_actor.frames
                            if fi in subtrack_frame_indices
                        ]

                        new_actor = MaskActor(
                            actor_id=new_tid,
                            frames=sub_frames,
                            last_centroid=(0, 0),
                            last_frame_idx=sub_frames[-1][0] if sub_frames else -1,
                            closed=False,
                            frame_indices=subtrack_frame_indices,
                        )
                        new_long_actors[new_tid] = new_actor

            long_actors = new_long_actors
            print(
                f"[VideoActorExtract] Track splitting: "
                f"{len(pre_split_records)} -> {len(long_actors)} actors"
            )

        # ----------------------------------------------------------------
        # Step 5: Identity clustering — merge same-person tracks across time
        # ----------------------------------------------------------------
        # Key constraint: tracks that co-occur (overlap in time) can NEVER
        # be the same person. This prevents merging different people who
        # happen to have similar faces (e.g., twins, or face embedding limits).
        print("[VideoActorExtract] Step 5: Clustering actor identities...")

        # Resolve model directory for InsightFace (same location as YOLO models)
        insightface_model_dir = ""
        try:
            import folder_paths

            insightface_model_dir = folder_paths.get_folder_paths("video-actor-extract")
            insightface_model_dir = (
                insightface_model_dir[0] if insightface_model_dir else ""
            )
        except Exception:
            pass
        if not insightface_model_dir:
            # Fallback: derive from the resolved seg_model_path's parent directory
            insightface_model_dir = os.path.dirname(seg_model_path)

        identity = IdentityCluster(
            threshold=face_threshold, model_dir=insightface_model_dir
        )

        # Build synthetic FrameRecords from mask actors for identity clustering
        actor_synthetic_records: Dict[int, list] = {}
        for aid, actor in long_actors.items():
            actor_synthetic_records[aid] = _actor_to_synthetic_records(actor)

        # Use cluster_tracks — it now enforces spatial-temporal constraints
        actor_to_identity = identity.cluster_tracks(
            actor_synthetic_records,
            frame_lookup,
            min_track_length,
            max_lost_frames=DEFAULT_MAX_LOST_FRAMES,
        )

        # Retrieve face bbox areas for preview sorting
        face_bbox_areas = identity.get_face_bbox_areas()

        # Group mask actors by identity
        identity_to_actors: Dict[str, List[int]] = {}
        for aid, actor_id in actor_to_identity.items():
            if actor_id not in identity_to_actors:
                identity_to_actors[actor_id] = []
            identity_to_actors[actor_id].append(aid)

        # Limit number of actors
        actor_ids_sorted = sorted(identity_to_actors.keys())[:max_actors]
        print(f"  Identified {len(actor_ids_sorted)} unique actors (max: {max_actors})")

        # ----------------------------------------------------------------
        # Step 6: Build continuous segments with interpolation
        # ----------------------------------------------------------------
        print("[VideoActorExtract] Step 6: Building continuous segments...")
        video_info_dict = {
            "total_frames": total_frames,
            "fps": fps,
            "width": orig_width,
            "height": orig_height,
            "duration_sec": total_frames / fps if fps > 0 else 0,
        }
        actor_data_for_json = {}
        preview_dir = os.path.join(output_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        import cv2

        for i, actor_id in enumerate(actor_ids_sorted):
            # Merge all frames from all mask tracks belonging to this identity
            actor_all_frames: List[Tuple[int, np.ndarray, int, int]] = []
            for aid in identity_to_actors[actor_id]:
                actor_all_frames.extend(long_actors[aid].frames)

            if not actor_all_frames:
                continue

            # Sort by frame_idx and deduplicate (keep largest bbox_area for same frame)
            frame_best: Dict[int, Tuple[np.ndarray, int, int]] = {}
            for fi, masked, area, bbox_area in actor_all_frames:
                if fi not in frame_best or bbox_area > frame_best[fi][2]:
                    frame_best[fi] = (masked, area, bbox_area)

            actor_all_frames = sorted(
                [
                    (fi, masked, area, bbox_area)
                    for fi, (masked, area, bbox_area) in frame_best.items()
                ],
                key=lambda x: x[0],
            )

            # Save top 5 preview frames by face area (largest face first)
            # Only frames where InsightFace detected a face are candidates
            actor_face_areas: Dict[int, int] = {}
            for aid in identity_to_actors[actor_id]:
                if aid in face_bbox_areas:
                    actor_face_areas.update(face_bbox_areas[aid])

            face_detected_frames = [
                (fi, mask, area, bbox_area)
                for fi, mask, area, bbox_area in actor_all_frames
                if fi in actor_face_areas
            ]

            top5 = sorted(
                face_detected_frames, key=lambda x: actor_face_areas[x[0]], reverse=True
            )[:5]
            # Previews saved as JPG with selected bg color
            bg_bgr = BG_COLOR_MAP.get(bg_color, BG_COLOR_GREEN)
            preview_frame_indexes = []
            for k, (fi, bool_mask, _, _) in enumerate(top5):
                frame_bgr = frame_lookup.get(fi)
                if frame_bgr is None:
                    continue
                masked_bgr = frame_bgr.copy()
                masked_bgr[~bool_mask] = bg_bgr
                preview_path = os.path.join(preview_dir, f"actor_{i}_{k}.jpg")
                cv2.imwrite(preview_path, masked_bgr)
                preview_frame_indexes.append(fi)

            # Save frame indexes for SelectActorPreview to read
            indexes_path = os.path.join(preview_dir, f"actor_{i}_indexes.json")
            with open(indexes_path, "w") as f:
                json.dump(preview_frame_indexes, f)

            # Build continuous segments with interpolation
            # _build_continuous_segments expects (frame_idx, bool_mask, mask_area)
            segment_input = [(fi, mask, area) for fi, mask, area, _ in actor_all_frames]
            segments_data, segments_info = _build_continuous_segments(
                segment_input,
                frame_shape=(img_h, img_w),
                max_gap=30,
                interp_gap=2,
            )

            if not segments_data:
                continue

            # Convert bool-mask segments to BGR frames for encoding
            segments_bgr: List[List[np.ndarray]] = []
            for seg in segments_data:
                bgr_frames = []
                for fi, bool_mask, _ in seg:
                    frame_bgr = frame_lookup.get(fi)
                    if frame_bgr is None:
                        frame_bgr = np.full((img_h, img_w, 3), bg_bgr, dtype=np.uint8)
                    composited = frame_bgr.copy()
                    composited[~bool_mask] = bg_bgr
                    bgr_frames.append(composited)
                segments_bgr.append(bgr_frames)

            output_video_path = os.path.join(output_dir, f"{actor_id}.mp4")
            success = merge_segments(
                segments_bgr,
                fps=fps,
                output_path=output_video_path,
                gap_sec=SEGMENT_GAP_SEC,
            )

            if success:
                total_output_frames = sum(s["frame_count"] for s in segments_info)
                print(
                    f"  {actor_id}: {len(segments_info)} segments, "
                    f"{total_output_frames} frames -> {output_video_path}"
                )

            actor_data_for_json[actor_id] = {
                "segments": segments_info,
                "total_frames": sum(s["frame_count"] for s in segments_info),
                "segment_count": len(segments_info),
            }

        # ----------------------------------------------------------------
        # Step 8: Generate JSON
        # ----------------------------------------------------------------
        json_path = os.path.join(output_dir, "actor_info.json")
        generate_actor_json(actor_data_for_json, video_info_dict, json_path)

        with open(json_path, "r") as f:
            json_content = f.read()

        print(f"[VideoActorExtract] Done! Output in {output_dir}")
        return (json_content, output_dir, len(actor_ids_sorted))


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "VideoActorExtractor": VideoActorExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoActorExtractor": "Video Actor Extract",
}
