"""
ComfyUI-VideoActorExtract

Extract individual actors from video with green screen background.
"""

import os
import sys

# Add plugin directory to Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# Use importlib to avoid namespace conflicts with ComfyUI's own 'nodes' module
import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load core modules first
core_config = _load_module(
    "vae_core_config", os.path.join(plugin_dir, "core", "config.py")
)
vae_video_reader = _load_module(
    "vae_video_reader", os.path.join(plugin_dir, "core", "video_reader.py")
)

# Load pipeline modules
vae_detector = _load_module(
    "vae_detector", os.path.join(plugin_dir, "pipeline", "detector.py")
)
vae_tracker = _load_module(
    "vae_tracker", os.path.join(plugin_dir, "pipeline", "tracker.py")
)
vae_identity = _load_module(
    "vae_identity", os.path.join(plugin_dir, "pipeline", "identity.py")
)
vae_cropper = _load_module(
    "vae_cropper", os.path.join(plugin_dir, "pipeline", "cropper.py")
)
vae_merger = _load_module(
    "vae_merger", os.path.join(plugin_dir, "pipeline", "merger.py")
)
vae_segmenter = _load_module(
    "vae_segmenter", os.path.join(plugin_dir, "pipeline", "segmenter.py")
)

# Load main node
vae_actor_extractor = _load_module(
    "vae_actor_extractor", os.path.join(plugin_dir, "nodes", "actor_extractor.py")
)
vae_select_preview = _load_module(
    "vae_select_preview", os.path.join(plugin_dir, "nodes", "select_preview.py")
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register nodes from all node modules
for mod in [vae_actor_extractor, vae_select_preview]:
    if hasattr(mod, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
    if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("[VideoActorExtract] Plugin loaded successfully!")
