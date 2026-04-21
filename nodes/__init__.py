"""ComfyUI nodes for video actor extraction.

Exports the main VideoActorExtractor node for use in ComfyUI workflows.
"""

from __future__ import annotations

from .actor_extractor import VideoActorExtractor, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["VideoActorExtractor"]

__all__.extend(NODE_CLASS_MAPPINGS.keys())

WEB_DIRECTORY = "./js"
