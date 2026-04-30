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
            - 4D tensor [N, H, W, 3] RGB float32 in 0-1 range.
            - List of frame indexes: [frame_idx_0, frame_idx_1, ...]
        """
        import cv2

        pattern = os.path.join(output_dir, "previews", f"actor_{actor_index}_*.jpg")
        paths = sorted(glob.glob(pattern))

        if not paths:
            print(f"[SelectActorPreview] No previews found: {pattern}")
            return (torch.zeros(1, 512, 512, 3, dtype=torch.float32), [])

        frames = []
        for p in paths:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue
            img_rgb = img_bgr[:, :, ::-1].copy()
            frames.append(img_rgb.astype(np.float32) / 255.0)

        if not frames:
            return (torch.zeros(1, 512, 512, 3, dtype=torch.float32), [])

        tensor = torch.from_numpy(np.stack(frames))  # [N, H, W, 3]

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
