"""Global configuration constants for VideoActorExtract."""

# Default YOLOv8 model
DEFAULT_YOLO_MODEL = "yolov8n.pt"
# Default segmentation model
DEFAULT_SEG_MODEL = "yolov8n-seg.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# ByteTrack tracking
DEFAULT_TRACK_THRESH = 0.5
DEFAULT_MATCH_THRESHOLD = 0.8
DEFAULT_MAX_LOST_FRAMES = 30

# Face recognition
DEFAULT_FACE_THRESHOLD = 0.6
DEFAULT_MIN_FACE_CONFIDENCE = 0.3

# Video processing
DEFAULT_FPS_SAMPLE = 3
DEFAULT_MIN_TRACK_LENGTH = 5
DEFAULT_MAX_ACTORS = 10

# YOLO inference — longest-side target for downscaling before inference
YOLO_INFER_SIZE = 640

# Green screen
GREEN_SCREEN_COLOR = (0, 255, 0)  # BGR pure green

# Background colors (BGR format)
BG_COLOR_GREEN = (0, 255, 0)
BG_COLOR_BLUE = (255, 0, 0)
BG_COLOR_BLACK = (0, 0, 0)
BG_COLOR_WHITE = (255, 255, 255)

BG_COLOR_MAP = {
    "green": BG_COLOR_GREEN,
    "blue": BG_COLOR_BLUE,
    "black": BG_COLOR_BLACK,
    "white": BG_COLOR_WHITE,
}

# Segment gap (green screen frames between merged segments)
SEGMENT_GAP_SEC = 0.5
SEGMENT_GAP_FRAMES_DEFAULT = 15  # will be computed from fps

# FFmpeg
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_VIDEO_CODEC_MAC = "h264_videotoolbox"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_CRF = 18
