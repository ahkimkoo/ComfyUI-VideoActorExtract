#!/usr/bin/env python3
"""Standalone pipeline test — runs full VideoActorExtract without ComfyUI.

Reads all frames from a test video, converts to ComfyUI IMAGE tensor format,
instantiates VideoActorExtractor, and calls extract().

Usage:
    cd /Users/cherokee/Project/ComfyUI-VideoActorExtract && python tests/test_pipeline.py
"""

import os
import sys
import json
import time

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 1. Standard library — already imported above

# 2. Third-party
import cv2
import numpy as np
import torch

# 3. Local project
from nodes.actor_extractor import VideoActorExtractor


# ---------------------------------------------------------------------------
# Expected ground truth for sample1.mp4
# ---------------------------------------------------------------------------
EXPECTED = {
    "num_actors": 3,
    "description": (
        "1 man (appears at beginning and end with a gap), "
        "2 children (appear together in same shots)"
    ),
}


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------
def load_video_as_tensor(video_path: str) -> torch.Tensor:
    """Read all frames from a video file and convert to ComfyUI IMAGE format.

    ComfyUI IMAGE format: shape [B, H, W, C], RGB, float32, values 0-1.

    Args:
        video_path: Absolute or relative path to an MP4 video file.

    Returns:
        torch.Tensor of shape [B, H, W, C].

    Raises:
        ValueError: If the video cannot be opened or has zero frames.
    """
    print(f"[TestPipeline] Opening video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        f"[TestPipeline] Video info: {width}x{height}, "
        f"{fps:.1f} fps, {total_frames} frames"
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video reports zero frames: {video_path}")

    # Pre-allocate tensor — much faster than repeated torch.from_numpy
    frames = []
    idx = 0
    t0 = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # BGR uint8 -> RGB float32, normalize to 0-1
        frame_rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
        frame_float = frame_rgb.astype(np.float32) / 255.0
        frames.append(frame_float)

        idx += 1
        if idx % 50 == 0 or idx == total_frames:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            print(
                f"[TestPipeline] Loaded {idx}/{total_frames} frames "
                f"({elapsed:.1f}s, {rate:.0f} fps)"
            )

    cap.release()

    elapsed_total = time.time() - t0
    print(
        f"[TestPipeline] Frame loading complete: {len(frames)} frames in {elapsed_total:.1f}s"
    )

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")

    # Stack into [B, H, W, C] tensor
    tensor = torch.from_numpy(np.stack(frames, axis=0))
    print(f"[TestPipeline] Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    return tensor


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------
def print_results(
    actor_json_str: str,
    output_dir: str,
    preview_tensor: torch.Tensor,
) -> dict:
    """Parse and print detailed pipeline results.

    Args:
        actor_json_str: JSON string returned by extract().
        output_dir: Output directory path.
        preview_tensor: Preview image tensor.

    Returns:
        Parsed JSON dict for further inspection.
    """
    result = json.loads(actor_json_str)

    actors = result.get("actors", [])
    video_info = result.get("video_info", {})

    print("\n" + "=" * 70)
    print("[TestPipeline] PIPELINE RESULTS")
    print("=" * 70)

    # Video info
    print(f"\n[Video Info]")
    print(
        f"  Resolution:    {video_info.get('width', '?')}x{video_info.get('height', '?')}"
    )
    print(f"  FPS:           {video_info.get('fps', '?')}")
    print(f"  Total frames:  {video_info.get('total_frames', '?')}")
    print(f"  Duration:      {video_info.get('duration_sec', 0):.2f}s")

    # Actor summary
    num_actors = len(actors)
    print(f"\n[Actors] Found {num_actors} actor(s)")

    if not actors:
        print("  (no actors detected)")
    else:
        for actor in actors:
            actor_id = actor.get("actor_id", "?")
            segments = actor.get("segments", [])
            total_frames = actor.get("total_frames", 0)
            segment_count = actor.get("segment_count", len(segments))

            print(f"\n  Actor '{actor_id}':")
            print(f"    Total output frames: {total_frames}")
            print(f"    Segments: {segment_count}")

            for i, seg in enumerate(segments):
                start = seg.get("start_frame", "?")
                end = seg.get("end_frame", "?")
                count = seg.get("frame_count", "?")
                print(f"      Segment {i + 1}: frames {start}-{end} ({count} frames)")

            # Check if video file was created
            video_file = os.path.join(output_dir, f"{actor_id}.mp4")
            if os.path.exists(video_file):
                size_mb = os.path.getsize(video_file) / (1024 * 1024)
                print(f"    Video: {video_file} ({size_mb:.2f} MB)")
            else:
                print(f"    Video: NOT FOUND at {video_file}")

    # Preview tensor
    if preview_tensor is not None and preview_tensor.numel() > 0:
        print(f"\n[Preview] Tensor shape: {preview_tensor.shape}")
    else:
        print(f"\n[Preview] No preview frames generated")

    # Output directory
    print(f"\n[Output] Directory: {output_dir}")
    json_file = os.path.join(output_dir, "actor_info.json")
    if os.path.exists(json_file):
        print(f"  JSON: {json_file}")
    else:
        print(f"  JSON: NOT FOUND at {json_file}")

    print("=" * 70)

    return result


# ---------------------------------------------------------------------------
# Expected vs Actual comparison
# ---------------------------------------------------------------------------
def print_comparison(result: dict) -> None:
    """Compare actual pipeline results against expected ground truth.

    Args:
        result: Parsed JSON dict from the pipeline.
    """
    actors = result.get("actors", [])
    actual_num = len(actors)

    print("\n" + "=" * 70)
    print("[TestPipeline] EXPECTED vs ACTUAL")
    print("=" * 70)

    print(f"\n  Expected actors: {EXPECTED['num_actors']}")
    print(f"  Actual actors:   {actual_num}")
    print(
        f"  Match:           {'PASS' if actual_num == EXPECTED['num_actors'] else 'MISMATCH'}"
    )
    print(f"\n  Expected description: {EXPECTED['description']}")

    # Per-actor segment breakdown for diagnosis
    if actual_num != EXPECTED["num_actors"]:
        print(f"\n  [Diagnosis]")
        if actual_num > EXPECTED["num_actors"]:
            print(
                f"  More actors detected than expected. "
                f"Possible causes: split tracks, false positives, or identity clustering too strict."
            )
        else:
            print(
                f"  Fewer actors detected than expected. "
                f"Possible causes: identity clustering too aggressive (merging distinct people), "
                f"or short tracks filtered out by min_track_length."
            )

    # Print detailed per-actor frame ranges for manual inspection
    print(f"\n  [Per-actor frame ranges for manual inspection]")
    for actor in actors:
        actor_id = actor.get("actor_id", "?")
        segments = actor.get("segments", [])
        ranges = ", ".join(f"{s['start_frame']}-{s['end_frame']}" for s in segments)
        print(f"    Actor '{actor_id}': [{ranges}]")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------
def save_results(
    result: dict,
    output_dir: str,
    save_path: str,
) -> None:
    """Save pipeline results plus metadata to a JSON file for inspection.

    Args:
        result: Parsed JSON dict from the pipeline.
        output_dir: Pipeline output directory path.
        save_path: Where to write the combined results JSON.
    """
    report = {
        "expected": EXPECTED,
        "actual_num_actors": len(result.get("actors", {})),
        "match": len(result.get("actors", {})) == EXPECTED["num_actors"],
        "output_dir": output_dir,
        "pipeline_result": result,
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[TestPipeline] Results saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full pipeline test on sample1.mp4."""
    print("[TestPipeline] Starting standalone pipeline test")
    print(f"[TestPipeline] Project root: {PROJECT_ROOT}")

    # Resolve video path
    video_path = os.path.join(PROJECT_ROOT, "tests", "sample1.mp4")
    if not os.path.exists(video_path):
        print(f"[TestPipeline] ERROR: Test video not found: {video_path}")
        sys.exit(1)

    # Results save path
    results_save_path = os.path.join(
        PROJECT_ROOT, "tests", "test_pipeline_results.json"
    )

    # Step 1: Load video frames into ComfyUI IMAGE tensor
    print("\n[TestPipeline] Step 1: Loading video frames...")
    images = load_video_as_tensor(video_path)

    # Step 2: Instantiate the extractor
    print("\n[TestPipeline] Step 2: Instantiating VideoActorExtractor...")
    extractor = VideoActorExtractor()

    # Step 3: Run the full pipeline
    print("\n[TestPipeline] Step 3: Running extract() pipeline...")
    print("-" * 70)
    t0 = time.time()

    actor_json_str, output_dir, preview_tensor = extractor.extract(
        images=images,
        video_path=video_path,
    )

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"[TestPipeline] Pipeline completed in {elapsed:.1f}s")

    # Step 4: Print detailed results
    result = print_results(actor_json_str, output_dir, preview_tensor)

    # Step 5: Compare with expected ground truth
    print_comparison(result)

    # Step 6: Save results JSON
    save_results(result, output_dir, results_save_path)

    # Final verdict
    actual_num = len(result.get("actors", {}))
    if actual_num == EXPECTED["num_actors"]:
        print(f"\n[TestPipeline] RESULT: PASS ({actual_num} actors detected)")
    else:
        print(
            f"\n[TestPipeline] RESULT: MISMATCH "
            f"(expected {EXPECTED['num_actors']}, got {actual_num})"
        )


if __name__ == "__main__":
    main()
