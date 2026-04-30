"""Model existence check and auto-download utilities.

Ensures required models (YOLOv8, InsightFace) are available
in the ComfyUI model directory before pipeline execution.
"""

import os
import shutil
import urllib.request


# Model download URLs
_YOLO_MODEL_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
}

_INSIGHTFACE_REQUIRED_FILES = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
]


def _get_model_dir() -> str:
    """Resolve the model directory path.

    Priority:
    1. ComfyUI folder_paths 'video-actor-extract' folder
    2. Default: ~/App/ComfyUI/models/video-actor-extract/
    """
    try:
        import folder_paths

        paths = folder_paths.get_folder_paths("video-actor-extract")
        if paths:
            return paths[0]
    except Exception:
        pass
    return os.path.expanduser("~/App/ComfyUI/models/video-actor-extract/")


def _download_file(url: str, dest: str, desc: str = "") -> bool:
    """Download a file from URL to dest path.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        label = desc or url.split("/")[-1]
        print(f"[ModelCheck] Downloading {label} ...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[ModelCheck] Downloaded {label} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"[ModelCheck] Failed to download {label}: {e}")
        return False


def _download_yolo_model(model_name: str, model_dir: str) -> bool:
    """Download a YOLO model if it doesn't exist."""
    dest = os.path.join(model_dir, model_name)
    if os.path.isfile(dest):
        return True
    url = _YOLO_MODEL_URLS.get(model_name)
    if not url:
        print(f"[ModelCheck] Unknown YOLO model: {model_name}")
        return False
    return _download_file(url, dest, desc=model_name)


def _download_insightface_model(model_dir: str) -> bool:
    """Download buffalo_l InsightFace model if it doesn't exist.

    InsightFace's FaceAnalysis.prepare() auto-downloads, but it places
    files in <root>/models/buffalo_l/ instead of <root>/buffalo_l/.
    This function handles that quirk.
    """
    buffalo_dir = os.path.join(model_dir, "buffalo_l")

    # Check if already in the correct location
    if all(
        os.path.isfile(os.path.join(buffalo_dir, f))
        for f in _INSIGHTFACE_REQUIRED_FILES
    ):
        return True

    # Check if InsightFace downloaded to the wrong subdirectory
    wrong_dir = os.path.join(model_dir, "models", "buffalo_l")
    if all(
        os.path.isfile(os.path.join(wrong_dir, f)) for f in _INSIGHTFACE_REQUIRED_FILES
    ):
        print("[ModelCheck] Moving buffalo_l from models/buffalo_l/ to buffalo_l/ ...")
        if os.path.exists(buffalo_dir):
            shutil.rmtree(buffalo_dir)
        shutil.move(wrong_dir, buffalo_dir)
        # Clean up empty parent
        parent = os.path.dirname(wrong_dir)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)
        print("[ModelCheck] Moved buffalo_l to correct location.")
        return True

    # Download using InsightFace API
    print("[ModelCheck] Downloading InsightFace buffalo_l model (~350MB) ...")
    try:
        from insightface.app import FaceAnalysis

        # Need to create the target parent so prepare() finds it
        os.makedirs(model_dir, exist_ok=True)
        fa = FaceAnalysis(name="buffalo_l", root=model_dir, allowed_modules=[])
        fa.prepare(ctx_id=-1, det_size=(640, 640))
        del fa
    except Exception as e:
        print(f"[ModelCheck] Failed to download buffalo_l: {e}")
        return False

    # Fix path: InsightFace puts files in models/buffalo_l/
    if os.path.isdir(wrong_dir) and not os.path.isdir(buffalo_dir):
        print("[ModelCheck] Moving buffalo_l from models/buffalo_l/ to buffalo_l/ ...")
        shutil.move(wrong_dir, buffalo_dir)
        parent = os.path.dirname(wrong_dir)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)

    # Final verification
    if all(
        os.path.isfile(os.path.join(buffalo_dir, f))
        for f in _INSIGHTFACE_REQUIRED_FILES
    ):
        print("[ModelCheck] InsightFace buffalo_l ready.")
        return True

    print(
        "[ModelCheck] InsightFace buffalo_l download failed. Face clustering will be skipped."
    )
    return False


def ensure_models_exist() -> str:
    """Check and download missing models.

    Should be called at the start of the extraction pipeline.

    Returns:
        The resolved model directory path.
    """
    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    print(f"[ModelCheck] Model directory: {model_dir}")

    # Check and download YOLO models
    for name in _YOLO_MODEL_URLS:
        _download_yolo_model(name, model_dir)

    # Check and download InsightFace model
    _download_insightface_model(model_dir)

    return model_dir
