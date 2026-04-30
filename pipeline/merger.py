"""Video merger and JSON output module.

Merges multiple actor segments into a single video per actor using ffmpeg,
and generates the actor_info.json output.
"""

import os
import json
import subprocess
import tempfile
import numpy as np
from typing import List, Dict
import cv2

from core.config import (
    SEGMENT_GAP_SEC,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_CODEC_MAC,
    DEFAULT_PIXEL_FORMAT,
    DEFAULT_CRF,
)


def _get_ffmpeg_codec() -> str:
    """Select appropriate ffmpeg video codec for the current platform."""
    import platform

    system = platform.system()
    if system == "Darwin":
        return DEFAULT_VIDEO_CODEC_MAC
    return DEFAULT_VIDEO_CODEC


def _encode_frames_to_temp(
    frames: List[np.ndarray], temp_path: str, fps: float
) -> bool:
    """
    Encode a list of frames to a temporary video file.

    Returns True on success, False on failure.
    """
    if not frames:
        return False

    h, w = frames[0].shape[:2]
    codec = _get_ffmpeg_codec()

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
        if not out.isOpened():
            # Fallback to ffmpeg directly
            return _encode_with_ffmpeg(frames, temp_path, fps)

        for frame in frames:
            out.write(frame)
        out.release()
        return os.path.exists(temp_path) and os.path.getsize(temp_path) > 0
    except Exception:
        return _encode_with_ffmpeg(frames, temp_path, fps)


def _encode_with_ffmpeg(frames: List[np.ndarray], temp_path: str, fps: float) -> bool:
    """Encode frames using ffmpeg pipe."""
    if not frames:
        return False

    h, w = frames[0].shape[:2]
    codec = _get_ffmpeg_codec()

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        codec,
        "-pix_fmt",
        DEFAULT_PIXEL_FORMAT,
        "-crf",
        str(DEFAULT_CRF),
        "-preset",
        "fast",
        "-an",
        temp_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        raw_bytes = b"".join([cv2.imencode(".png", f)[1].tobytes() for f in frames])
        # Actually pipe raw BGR
        raw_data = np.stack(frames).tobytes()
        proc.communicate(input=raw_data, timeout=120)
        return proc.returncode == 0
    except Exception:
        # Last resort: write to images then use ffmpeg concat
        return False


def _create_green_frames(count: int, width: int, height: int) -> List[np.ndarray]:
    """Create a list of solid green frames."""
    green = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)
    return [green.copy() for _ in range(count)]


def merge_segments(
    segments: List[List[np.ndarray]],
    fps: float,
    output_path: str,
    gap_sec: float = SEGMENT_GAP_SEC,
) -> bool:
    """
    Merge multiple segments into one video.

    Args:
        segments: List of frame lists, each list is one continuous segment
        fps: Frame rate
        output_path: Output MP4 path
        gap_sec: (kept for API compatibility, no longer inserts green gaps)

    Returns:
        True on success
    """
    if not segments or all(len(s) == 0 for s in segments):
        return False

    # Filter out empty segments
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return False

    if len(segments) == 1:
        return _encode_frames_to_temp(segments[0], output_path, fps)

    # Concatenate all segments without gaps
    merged = []
    for seg in segments:
        merged.extend(seg)

    return _encode_frames_to_temp(merged, output_path, fps)


def merge_segments_rgba(
    segments: List[List[np.ndarray]],
    fps: float,
    output_path: str,
) -> bool:
    """
    Merge RGBA frame segments into a WebM video with VP9 alpha channel.

    Each frame must be a BGRA uint8 numpy array (OpenCV's default RGBA
    ordering). Person pixels have alpha=255, background pixels alpha=0.

    Args:
        segments: List of frame lists, each list is one continuous segment.
            Frames are BGRA uint8 arrays.
        fps: Frame rate.
        output_path: Output .webm file path.

    Returns:
        True on success, False on failure.
    """
    if not segments or all(len(s) == 0 for s in segments):
        return False

    # Filter out empty segments and flatten
    frames = [f for seg in segments if seg for f in seg]
    if not frames:
        return False

    h, w = frames[0].shape[:2]

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgra",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-an",
        output_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        raw_data = np.stack(frames).tobytes()
        proc.communicate(input=raw_data, timeout=300)
        return proc.returncode == 0
    except Exception:
        return False


def generate_actor_json(
    actor_data: Dict[str, dict],
    video_info: dict,
    output_path: str,
) -> str:
    """
    Generate actor_info.json file.

    Args:
        actor_data: {
            "actor_0": {
                "segments": [
                    {"start_frame": 24, "end_frame": 89, "frame_count": 66},
                    ...
                ],
                "total_frames": 156,
                "segment_count": 2,
            },
            ...
        }
        video_info: {"total_frames": ..., "fps": ..., "width": ..., "height": ..., "duration_sec": ...}
        output_path: Path to write JSON file

    Returns:
        Path to the generated JSON file
    """
    result = {
        "actors": [],
        "video_info": video_info,
    }

    for actor_id in sorted(actor_data.keys()):
        info = actor_data[actor_id]
        segments = []
        for i, seg in enumerate(info["segments"]):
            start_time = seg["start_frame"] / video_info["fps"]
            end_time = seg["end_frame"] / video_info["fps"]
            segments.append(
                {
                    "segment_id": i,
                    "start_frame": seg["start_frame"],
                    "end_frame": seg["end_frame"],
                    "start_time_sec": round(start_time, 2),
                    "end_time_sec": round(end_time, 2),
                    "frame_count": seg["frame_count"],
                }
            )

        result["actors"].append(
            {
                "actor_id": actor_id,
                "segment_count": info["segment_count"],
                "total_frames": info["total_frames"],
                "segments": segments,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path
