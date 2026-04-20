"""Video reading and frame extraction utilities."""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class VideoInfo:
    total_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float


def get_video_info(video_path: str) -> VideoInfo:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps
    
    cap.release()
    return VideoInfo(
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def extract_frames(video_path: str, fps_sample: int = 3) -> tuple[np.ndarray, list[int]]:
    """
    Extract frames from video at the specified sample rate.
    
    Args:
        video_path: Path to video file
        fps_sample: Number of frames to extract per second
        
    Returns:
        Tuple of (frames_list, frame_indices_in_original_video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate which frames to extract
    if fps_sample >= original_fps:
        # Sample every frame
        frame_indices = list(range(total_frames))
    else:
        # Sample at fps_sample rate
        skip = max(1, int(original_fps / fps_sample))
        frame_indices = list(range(0, total_frames, skip))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # Frame read failed, use last valid frame or black frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames), frame_indices


def extract_frames_at_indices(video_path: str, indices: list[int]) -> list[np.ndarray]:
    """
    Extract specific frames from video by index.
    
    Args:
        video_path: Path to video file
        indices: List of frame indices to extract
        
    Returns:
        List of frames as numpy arrays (BGR)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # Fallback: return black frame
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    cap.release()
    return frames
