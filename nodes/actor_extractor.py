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
from pipeline.detector import PersonDetector, BoundingBox
from pipeline.tracker import PersonTracker
from pipeline.identity import IdentityCluster
from pipeline.cropper import ActorCropper
from pipeline.merger import merge_segments, generate_actor_json


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
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "model_path": ("STRING", {"default": "yolov8n.pt", "multiline": False}),
                "video_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional: original video path for metadata. If empty, metadata is estimated."}),
                "max_actors": ("INT", {"default": DEFAULT_MAX_ACTORS, "min": 1, "max": 50}),
                "face_threshold": ("FLOAT", {"default": DEFAULT_FACE_THRESHOLD, "min": 0.1, "max": 0.99, "step": 0.05}),
                "min_track_length": ("INT", {"default": DEFAULT_MIN_TRACK_LENGTH, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("actor_info_json", "output_dir")
    FUNCTION = "extract"
    CATEGORY = "video/actor"
    
    def extract(
        self,
        images: torch.Tensor,
        model_path: str = DEFAULT_YOLO_MODEL,
        video_path: str = "",
        max_actors: int = DEFAULT_MAX_ACTORS,
        face_threshold: float = DEFAULT_FACE_THRESHOLD,
        min_track_length: int = DEFAULT_MIN_TRACK_LENGTH,
    ) -> Tuple[str, str]:
        """
        Run the full actor extraction pipeline.
        
        Args:
            images: IMAGE tensor from VHS LoadVideo, shape [B, H, W, C], RGB, 0-1
            model_path: YOLOv8 model path
            video_path: Optional original video path for metadata
            max_actors: Maximum number of actors to detect
            face_threshold: Face similarity threshold for identity merging
            min_track_length: Minimum number of frames for a valid track
            
        Returns:
            (actor_info_json, output_dir)
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
        
        # Squeeze batch dimension if present
        if imgs_np.ndim == 4:
            imgs_np = imgs_np.squeeze(0)  # [B, H, W, C] -> [B, H, W, C] if B=1
            if imgs_np.ndim == 3:
                imgs_np = imgs_np[np.newaxis, ...]
        elif imgs_np.ndim == 3:
            imgs_np = imgs_np[np.newaxis, ...]
        
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
            tid: recs for tid, recs in track_records.items()
            if len(recs) >= min_track_length
        }
        print(f"  Found {len(track_records)} tracks, {len(long_tracks)} with >= {min_track_length} frames")
        
        if not long_tracks:
            # No actors found, return empty result
            empty_json = json.dumps({
                "actors": [],
                "video_info": {
                    "total_frames": total_frames,
                    "fps": fps,
                    "width": orig_width,
                    "height": orig_height,
                    "duration_sec": total_frames / fps if fps > 0 else 0,
                }
            }, indent=2)
            return (empty_json, output_dir)
        
        # Step 5: Identity clustering
        print("[VideoActorExtract] Step 4: Clustering actor identities...")
        identity = IdentityCluster(threshold=face_threshold)
        track_to_actor = identity.cluster_tracks(long_tracks, frame_lookup, min_track_length)
        
        # Group tracks by actor
        actor_tracks: Dict[str, List[int]] = {}
        for tid, actor_id in track_to_actor.items():
            if actor_id not in actor_tracks:
                actor_tracks[actor_id] = []
            actor_tracks[actor_id].append(tid)
        
        # Limit number of actors
        actor_ids_sorted = sorted(actor_tracks.keys())[:max_actors]
        print(f"  Identified {len(actor_ids_sorted)} actors (max: {max_actors})")
        
        # Step 6: Crop actors and create segments
        print("[VideoActorExtract] Step 5: Cropping actors with green screen...")
        cropper = ActorCropper()
        
        # Compute uniform output size
        all_actor_records = []
        for actor_id in actor_ids_sorted:
            for tid in actor_tracks[actor_id]:
                all_actor_records.append(long_tracks[tid])
        
        output_size = ActorCropper.compute_output_size(
            all_actor_records, orig_height, orig_width
        )
        print(f"  Output size: {output_size[0]}x{output_size[1]}")
        
        # Step 7: Generate output per actor
        print("[VideoActorExtract] Step 6: Generating output videos and JSON...")
        video_info_dict = {
            "total_frames": total_frames,
            "fps": fps,
            "width": orig_width,
            "height": orig_height,
            "duration_sec": total_frames / fps if fps > 0 else 0,
        }
        actor_data_for_json = {}
        
        for actor_id in actor_ids_sorted:
            tids = actor_tracks[actor_id]
            
            # Merge all track records for this actor
            all_recs = []
            for tid in tids:
                all_recs.extend(long_tracks[tid])
            all_recs.sort(key=lambda r: r.frame_idx)
            
            # Split into segments (continuous sequences)
            # Since we process every frame, a gap of > 30 frames means the person left the scene
            segments_recs = []
            current_segment = [all_recs[0]]
            
            for i in range(1, len(all_recs)):
                gap = all_recs[i].frame_idx - all_recs[i - 1].frame_idx
                if gap > 30:  # ~1 second gap at 30fps means new segment
                    segments_recs.append(current_segment)
                    current_segment = []
                current_segment.append(all_recs[i])
            segments_recs.append(current_segment)
            
            # Crop each segment
            actor_segments_frames = []
            actor_segments_info = []
            
            for seg_recs in segments_recs:
                seg_frames = cropper.crop_segment_from_dict(
                    frame_lookup, seg_recs, output_size=output_size
                )
                if seg_frames:
                    actor_segments_frames.append(seg_frames)
                    actor_segments_info.append({
                        "start_frame": seg_recs[0].frame_idx,
                        "end_frame": seg_recs[-1].frame_idx,
                        "frame_count": len(seg_recs),
                    })
            
            if not actor_segments_frames:
                continue
            
            # Encode to video
            output_video_path = os.path.join(output_dir, f"{actor_id}.mp4")
            success = merge_segments(
                actor_segments_frames,
                fps=fps,
                output_path=output_video_path,
                gap_sec=SEGMENT_GAP_SEC,
            )
            
            if success:
                print(f"  {actor_id}: {len(segments_recs)} segments, "
                      f"{sum(s['frame_count'] for s in actor_segments_info)} frames -> {output_video_path}")
            
            actor_data_for_json[actor_id] = {
                "segments": actor_segments_info,
                "total_frames": sum(s["frame_count"] for s in actor_segments_info),
                "segment_count": len(actor_segments_info),
            }
        
        # Generate JSON
        json_path = os.path.join(output_dir, "actor_info.json")
        generate_actor_json(actor_data_for_json, video_info_dict, json_path)
        
        with open(json_path, "r") as f:
            json_content = f.read()
        
        print(f"[VideoActorExtract] Done! Output in {output_dir}")
        return (json_content, output_dir)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "VideoActorExtractor": VideoActorExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoActorExtractor": "Video Actor Extract",
}
