# AGENTS.md — ComfyUI-VideoActorExtract

Guidelines for agentic coding agents working in this repository.

## Project Overview

ComfyUI custom node plugin that detects, tracks, and extracts individual actors from video,
outputting per-actor green-screen videos and structured JSON metadata.

- **Language**: Python 3.10+ (backend), JavaScript (ComfyUI widget)
- **Framework**: ComfyUI custom node plugin
- **Runtime**: Depends on ComfyUI host process; not a standalone application

## Repository Structure

```
__init__.py              # Plugin entry — dynamic importlib loading to avoid ComfyUI namespace conflicts
core/
  config.py              # Global constants (thresholds, defaults, codec settings)
  video_reader.py        # Video info extraction and frame sampling
nodes/
  actor_extractor.py     # Main ComfyUI node class (VideoActorExtractor) — pipeline orchestration
pipeline/
  detector.py            # YOLOv8 person detection → BoundingBox list
  segmenter.py           # YOLOv8-seg person segmentation → mask per person
  mask_tracker.py        # Centroid-distance mask tracking across frames
  tracker.py             # ByteTrack multi-object tracking (optional, fallback to greedy)
  identity.py            # InsightFace face embedding clustering for same-actor merging
  cropper.py             # Green-screen compositing per detected person
  merger.py              # FFmpeg encoding + JSON metadata generation
js/
  widget.js              # ComfyUI frontend extension (JSON display button)
```

## Build & Install

```bash
# Install dependencies
pip install -r requirements.txt

# Install in dev mode (optional)
pip install -e ".[dev]"
```

No build step required — this is a pure-Python ComfyUI plugin loaded at runtime.

## Linting

```bash
# Lint with ruff (uses default rules — no ruff.toml/config exists)
ruff check .

# Auto-fix where possible
ruff check --fix .
```

No formatter is configured. Keep existing style when making changes.

## Testing

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_foo.py

# Run a specific test by name
pytest tests/test_foo.py::test_specific_function -v
```

> **Note**: No test files exist yet. `pytest` is a dev dependency in `pyproject.toml`.
> Integration testing is done manually via ComfyUI's HTTP API (see README.md lines 283–367).
> When adding tests, create a `tests/` directory at the project root.

## Code Style Guidelines

### Naming Conventions (PEP 8)

| Element | Style | Example |
|---------|-------|---------|
| Functions / methods | `snake_case` | `_build_continuous_segments`, `detect_masks` |
| Variables | `snake_case` | `actor_frames`, `mask_resized` |
| Classes | `PascalCase` | `VideoActorExtractor`, `MaskTracker` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_MAX_ACTORS`, `GREEN_SCREEN_COLOR` |
| Private members | `_leading_underscore` | `_ensure_loaded`, `_compute_centroid` |
| ComfyUI node attributes | `UPPER_SNAKE_CASE` | `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION` |

### Import Order

Group imports in this order, separated by blank lines:

```python
# 1. Standard library
import os
import sys
import json

# 2. Third-party
import torch
import numpy as np
from typing import List, Tuple, Optional

# 3. Local project (absolute paths from project root)
from core.config import DEFAULT_MAX_ACTORS
from pipeline.detector import PersonDetector
```

**Lazy imports for heavy dependencies** — load YOLO/InsightFace inside methods on first use,
not at module top level. This keeps plugin import time fast:

```python
def _ensure_loaded(self):
    """Lazy-load InsightFace model."""
    if self._loaded:
        return
    import insightface
    ...
```

### Type Annotations

- Use `typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Optional` (Python 3.10 compat).
- Annotate public function signatures; private helpers may omit annotations for brevity.
- Use `@dataclass` for data containers (e.g., `FrameRecord`, `VideoInfo`).

```python
def detect(self, frame: np.ndarray) -> List[BoundingBox]:
    """Detect all persons in a single frame."""
    ...
```

### Docstrings (Google style)

```python
def _build_continuous_segments(
    actor_frames: List[Tuple[int, np.ndarray, int]],
    max_gap: int = 30,
) -> Tuple[List[List[np.ndarray]], List[dict]]:
    """
    Group frames into continuous segments with gap interpolation.

    Args:
        actor_frames: List of (frame_idx, frame_array, mask_id) tuples.
        max_gap: Maximum gap in frames before starting a new segment.

    Returns:
        Tuple of (segmented_frame_lists, segment_metadata).
    """
```

- Module-level docstrings: single-line summary.
- Class docstrings: one-line description of purpose.
- Public methods: `Args` / `Returns` sections.
- Skip docstrings for trivial private helpers.

### Logging

Use `print()` with a bracketed prefix — no `logging` module:

```python
print("[VideoActorExtract] Step 1: Initializing person segmenter...")
print("[MaskTracker] Lost track for actor_id=3 at frame 142")
```

Module prefix convention: `[VideoActorExtract]`, `[PersonSegmenter]`, `[MaskTracker]`, `[Identity]`.

### Error Handling

- **Graceful degradation**: wrap optional heavy imports in `try/except ImportError`,
  provide fallback behavior (e.g., greedy tracker when ByteTrack is unavailable).
- **No hard crashes in pipeline**: return empty/error JSON rather than raising from the
  main `extract()` method.
- **ValueError** for invalid user inputs (bad video path, missing frames).
- Silent `except Exception: pass` is acceptable only for truly optional features (e.g., proxy setup).

### Data Containers

Prefer `@dataclass` for structured data; use `__slots__` for performance-critical containers:

```python
@dataclass
class FrameRecord:
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float

class BoundingBox:
    __slots__ = ['x1', 'y1', 'x2', 'y2', 'confidence']
```

### ComfyUI Node Pattern

All nodes must expose these class attributes:

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}, "optional": {...}}

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("result_json", "output_dir", "preview")
    FUNCTION = "process"          # method name for ComfyUI to call
    CATEGORY = "video/actor"
    OUTPUT_NODE = True            # optional
```

### JavaScript (ComfyUI Widget)

Minimal — single file in `js/`. Register via:

```javascript
app.registerExtension({
    name: "VideoActorExtract.widget",
    // ...
});
```

## Commit Messages

Conventional commits, lowercase prefix:

```
feat: add person segmentation pipeline
fix: resolve namespace conflict with ComfyUI nodes module
docs: comprehensive README with architecture guide
improve: identity clustering with connected components
chore: add .gitignore
```

## Key Constraints

1. **No standalone execution**: This plugin only runs inside ComfyUI. There is no CLI entry point.
2. **`__init__.py` uses importlib**: All modules are loaded via `importlib.util` with prefixed names
   (`vae_*`) to avoid namespace collisions with ComfyUI's own `nodes` module. Do not switch to
   regular imports in `__init__.py`.
3. **Model weights** (`*.pt`) are gitignored. Models are loaded from `ComfyUI/models/video-actor-extract/`.
4. **Apple Silicon**: `onnxruntime-silicon` is used on ARM64 macOS (see requirements.txt conditional).
5. **Memory-sensitive**: Process frames one at a time; avoid accumulating full-video tensors in memory.
