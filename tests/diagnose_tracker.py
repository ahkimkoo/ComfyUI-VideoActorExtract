#!/usr/bin/env python3
"""Diagnose MaskTracker frame assignments for frames 85-115."""

import os, sys, cv2, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.segmenter import PersonSegmenter
from pipeline.mask_tracker import MaskTracker
from core.config import DEFAULT_MAX_LOST_FRAMES

# Load video
cap = cv2.VideoCapture(os.path.join(PROJECT_ROOT, "tests", "sample1.mp4"))
all_frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    all_frames.append(f)  # BGR
cap.release()
print(f"Loaded {len(all_frames)} frames")

# Init
seg = PersonSegmenter(
    model_path="/Users/cherokee/App/ComfyUI/models/video-actor-extract/yolov8n-seg.pt"
)
tracker = MaskTracker(max_lost_frames=DEFAULT_MAX_LOST_FRAMES, match_threshold_px=150.0)

# Process frames 190-289 (Track 3 creation zone)
for fi in range(190, 290):
    frame_bgr = all_frames[fi]
    masks = seg.detect_masks(frame_bgr)

    # Compute mask info
    mask_info = []
    for mid, mask in masks:
        ys, xs = np.where(mask)
        if len(ys) > 0:
            cx, cy = float(xs.mean()), float(ys.mean())
            area = int(mask.sum())
            mask_info.append((mid, cx, cy, area))

    # Snapshot before update
    actors_before = {}
    for aid, actor in tracker.get_active_actors().items():
        if not actor.closed:
            actors_before[aid] = (actor.last_centroid, actor.last_frame_idx)

    tracker.update(fi, masks, frame_bgr, seg)

    # Snapshot after update
    actors_after = {}
    for aid, actor in tracker.get_active_actors().items():
        actors_after[aid] = (
            actor.last_centroid,
            actor.last_frame_idx,
            actor.closed,
            len(actor.frames),
        )

    if fi >= 190:
        print(f"\nFrame {fi}: {len(masks)} masks")
        for mid, cx, cy, area in mask_info:
            assigned = "UNMATCHED"
            for aid, (cent, lfi, closed, cnt) in sorted(actors_after.items()):
                if lfi == fi and not closed:
                    dist = ((cx - cent[0]) ** 2 + (cy - cent[1]) ** 2) ** 0.5
                    assigned = f"-> actor_{aid} (dist={dist:.0f}px)"
                    break
            print(f"  mask_{mid}: ({cx:.0f},{cy:.0f}) area={area:>7d}  {assigned}")

        for aid, (cent, lfi) in sorted(actors_before.items()):
            still_active = aid in actors_after
            matched_this_frame = still_active and actors_after[aid][1] == fi
            if (
                not matched_this_frame
                and not actors_after.get(aid, (None, None, True, 0))[2]
            ):
                print(
                    f"  actor_{aid}: UNMATCHED (centroid=({cent[0]:.0f},{cent[1]:.0f}), last={lfi})"
                )

# Final summary
print("\n=== FINAL TRACK SUMMARY ===")
final = tracker.finish()
for aid, actor in sorted(final.items()):
    fidx = [f[0] for f in actor.frames]
    print(f"  actor_{aid}: frames {min(fidx)}-{max(fidx)}, count={len(fidx)}")
