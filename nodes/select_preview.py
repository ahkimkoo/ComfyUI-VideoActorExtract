"""SelectActorPreview node for ComfyUI.

Loads all preview images for a given actor from disk and returns
them as a single IMAGE batch tensor [N, H, W, 3], along with the
frame indexes of the selected previews.
"""

import glob
import json
import os

import numpy as np
import torch
from typing import Tuple


class SelectActorPreview:
    """Load all preview images for a specific actor as an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {"default": ""}),
                "actor_index": ("INT", {"default": 0, "min": 0, "max": 999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "indexes")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "select"
    CATEGORY = "video/actor"
    OUTPUT_NODE = False

    def select(self, output_dir: str, actor_index: int) -> Tuple[torch.Tensor, list]:
        """Load all preview images for a specific actor.

        Args:
            output_dir: Output directory from VideoActorExtractor.
            actor_index: Which actor (0-based).

        Returns:
            Tuple of:
            - 4D tensor [N, H, W, 4] RGBA float32 in 0-1 range.
            - List of frame indexes: [frame_idx_0, frame_idx_1, ...]
        """
        import cv2

        # Previews are now always PNG with alpha channel
        pattern = os.path.join(output_dir, "previews", f"actor_{actor_index}_*.png")
        paths = sorted(glob.glob(pattern))

        if not paths:
            # Fallback: check for legacy .jpg previews
            pattern_jpg = os.path.join(
                output_dir, "previews", f"actor_{actor_index}_*.jpg"
            )
            paths = sorted(glob.glob(pattern_jpg))

        if not paths:
            print(f"[SelectActorPreview] No previews found: {pattern}")
            return (torch.zeros(1, 512, 512, 4, dtype=torch.float32), [])

        frames = []
        for p in paths:
            img_bgra = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img_bgra is None:
                continue
            if img_bgra.ndim == 2:
                # Grayscale — convert to BGRA
                img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_GRAY2BGRA)
            elif img_bgra.shape[2] == 3:
                # BGR without alpha — add full opacity
                alpha = np.full((*img_bgra.shape[:2], 255), dtype=np.uint8)
                img_bgra = np.dstack([img_bgra, alpha])
            # BGRA -> RGBA
            img_rgba = img_bgra[:, :, [2, 1, 0, 3]].copy()
            frames.append(img_rgba.astype(np.float32) / 255.0)

        if not frames:
            return (torch.zeros(1, 512, 512, 4, dtype=torch.float32), [])

        tensor = torch.from_numpy(np.stack(frames))  # [N, H, W, 4]

        # Read frame indexes from the JSON file written by VideoActorExtractor
        indexes_path = os.path.join(
            output_dir, "previews", f"actor_{actor_index}_indexes.json"
        )
        if os.path.isfile(indexes_path):
            with open(indexes_path, "r") as f:
                indexes = json.load(f)
        else:
            indexes = []

        print(
            f"[SelectActorPreview] actor_{actor_index}: "
            f"loaded {len(frames)} previews, shape={tensor.shape}, "
            f"indexes={indexes}"
        )
        return (tensor, indexes)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SelectActorPreview": SelectActorPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectActorPreview": "Select Actor Preview",
}
