"""
ComfyUI-VideoActorExtract

Extract individual actors from video with green screen background.
Detects, tracks, and identifies all people in a video, then outputs
per-actor cropped videos with green screen background.
"""

import os
import sys

# Add plugin directory to Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# Import nodes
from nodes.actor_extractor import VideoActorExtractor

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register nodes
if hasattr(VideoActorExtractor, "NODE_CLASS_MAPPINGS"):
    NODE_CLASS_MAPPINGS.update(VideoActorExtractor.NODE_CLASS_MAPPINGS)
if hasattr(VideoActorExtractor, "NODE_DISPLAY_NAME_MAPPINGS"):
    NODE_DISPLAY_NAME_MAPPINGS.update(VideoActorExtractor.NODE_DISPLAY_NAME_MAPPINGS)

# Also register directly
NODE_CLASS_MAPPINGS["VideoActorExtractor"] = VideoActorExtractor
NODE_DISPLAY_NAME_MAPPINGS["VideoActorExtractor"] = "Video Actor Extract"

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
