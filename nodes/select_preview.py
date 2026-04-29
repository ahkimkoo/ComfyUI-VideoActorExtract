"""SelectActorPreview node for ComfyUI.

Selects a specific preview image for a specific actor from the
preview files saved by VideoActorExtractor.
"""

import os
import numpy as np
import torch
from typing import Tuple


class SelectActorPreview:
    """Select a specific preview image for a specific actor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {"default": ""}),
                "actor_index": ("INT", {"default": 0, "min": 0, "max": 999}),
                "photo_index": ("INT", {"default": 0, "min": 0, "max": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select"
    CATEGORY = "video/actor"
    OUTPUT_NODE = False

    def select(
        self, output_dir: str, actor_index: int, photo_index: int
    ) -> Tuple[torch.Tensor]:
        """Select a preview image from disk for a specific actor.

        Args:
            output_dir: Output directory from VideoActorExtractor.
            actor_index: Which actor (0-based).
            photo_index: Which photo (0-4).

        Returns:
            Tuple of one 4D tensor [1, H, W, 3] RGB float32 in 0-1 range.
            Must return a tuple (not bare tensor) so ComfyUI's merge_result_data
            correctly interprets it as a single IMAGE output.
        """
        preview_path = os.path.join(
            output_dir, "previews", f"actor_{actor_index}_{photo_index}.jpg"
        )
        if not os.path.isfile(preview_path):
            print(f"[SelectActorPreview] File not found: {preview_path}")
            return (torch.zeros(1, 512, 512, 3, dtype=torch.float32),)

        import cv2

        img_bgr = cv2.imread(preview_path)
        if img_bgr is None:
            return (torch.zeros(1, 512, 512, 3, dtype=torch.float32),)

        img_rgb = img_bgr[:, :, ::-1].copy()
        img_float = img_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).unsqueeze(0)  # [1, H, W, 3]
        return (tensor,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SelectActorPreview": SelectActorPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectActorPreview": "Select Actor Preview",
}
