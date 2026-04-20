# ComfyUI-VideoActorExtract 架构设计

## 1. 项目结构

```
ComfyUI-VideoActorExtract/
├── __init__.py                  # ComfyUI 节点注册入口
├── requirements.txt             # Python 依赖
├── pyproject.toml               # 项目元数据
├── docs/
│   ├── requirements.md          # 需求文档
│   └── architecture.md          # 架构设计（本文件）
├── nodes/
│   ├── __init__.py              # 节点导出
│   ├── actor_extractor.py       # 主提取节点
│   └── actor_visualize.py       # 可视化节点（可选）
├── pipeline/
│   ├── __init__.py
│   ├── detector.py              # YOLOv8 人物检测
│   ├── tracker.py               # ByteTrack 多目标追踪
│   ├── identity.py              # InsightFace 身份识别 + 聚类
│   ├── cropper.py               # 人物抠图 + 绿幕合成
│   └── merger.py                # 多段视频合并 + 输出
├── core/
│   ├── __init__.py
│   ├── video_reader.py          # 视频读取 + 帧提取
│   ├── config.py                # 全局配置
│   └── logger.py                # 日志
└── js/
    └── widget.js                # 前端组件（JSON 展示）
```

## 2. 处理管线 (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VideoActorExtractor Node                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  video_path ──► VideoReader ──► 逐帧提取                              │
│                      │                                                │
│                      ▼                                                │
│               ┌──────────────┐                                       │
│               │   Detector    │  YOLOv8n, person class only           │
│               │  (每帧检测)    │  输出: [(x,y,w,h,conf), ...]          │
│               └──────┬───────┘                                       │
│                      │                                                │
│                      ▼                                                │
│               ┌──────────────┐                                       │
│               │   Tracker     │  ByteTrack                             │
│               │  (帧间关联)    │  输出: TrackID → [bbox_per_frame]     │
│               └──────┬───────┘                                       │
│                      │                                                │
│                      ▼                                                │
│               ┌──────────────┐                                       │
│               │   Identity    │  InsightFace 人脸 + DBSCAN 聚类        │
│               │  (身份合并)    │  输出: TrackID → ActorID             │
│               └──────┬───────┘                                       │
│                      │                                                │
│                      ▼                                                │
│               ┌──────────────┐                                       │
│               │    Cropper    │  抠图 + 绿幕合成                        │
│               │  (帧级裁剪)    │  输出: 按Actor分组的帧序列             │
│               └──────┬───────┘                                       │
│                      │                                                │
│                      ▼                                                │
│               ┌──────────────┐                                       │
│               │    Merger     │  ffmpeg 编码 + 多段合并                │
│               │  (视频输出)    │  输出: actor_X.mp4 + JSON             │
│               └──────────────┘                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. 模块详细设计

### 3.1 Detector (detector.py)

```python
class PersonDetector:
    """YOLOv8 人物检测器"""
    
    def __init__(self, model: str = "yolov8n.pt", device: str = "auto"):
        """
        - model: YOLOv8 权重路径，默认自动下载 yolov8n.pt
        - device: 'cuda', 'cpu', 或 'auto'
        """
    
    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """
        检测单帧中的所有人物
        返回: [(x1, y1, x2, y2, confidence), ...]
        """
    
    def detect_batch(self, frames: list[np.ndarray]) -> list[list[BoundingBox]]:
        """批量检测多帧"""
```

**关键参数：**
- 只检测 class 0 (person)
- conf threshold = 0.25
- iou threshold = 0.45
- 使用 YOLOv8n（nano）保证速度

### 3.2 Tracker (tracker.py)

```python
class PersonTracker:
    """ByteTrack 多目标追踪器"""
    
    def __init__(self, fps: float = 30.0, track_thresh: float = 0.5):
        """
        - fps: 视频帧率，用于速度估计
        - track_thresh: 追踪置信度阈值
        """
    
    def update(self, bboxes: list[BoundingBox]) -> list[Track]:
        """
        输入当前帧的所有检测框
        返回当前帧的追踪结果: [(track_id, x1, y1, x2, y2), ...]
        """
    
    def finish(self) -> dict[int, list[FrameRecord]]:
        """
        视频处理完毕后，返回所有 Track 的完整记录
        {track_id: [(frame_idx, x1, y1, x2, y2), ...], ...}
        """
```

**追踪策略：**
- 使用 ByteTrack（无需外观特征，仅靠 bbox + 卡尔曼滤波）
- 对消失的 track，如果 30 帧内未重现则标记为 finished
- 最终按 track_id 分组，每个 track 可能包含多个连续段

### 3.3 Identity (identity.py)

```python
class IdentityCluster:
    """基于人脸识别的身份聚类"""
    
    def __init__(self, threshold: float = 0.6):
        """
        - threshold: 人脸相似度阈值，低于此值视为不同人
        """
    
    def extract_faces(self, frame: np.ndarray, bboxes: list[BoundingBox]) -> list[np.ndarray]:
        """
        从每个 bbox 区域提取人脸特征向量
        """
    
    def cluster_tracks(self, track_records: dict[int, list[FrameRecord]], 
                       frames: list[np.ndarray]) -> dict[int, str]:
        """
        将多个 TrackID 聚类为 ActorID
        返回: {track_id: "actor_0", ...}
        
        策略:
        1. 对每个 track 的关键帧（均匀采样5帧）提取人脸 embedding
        2. 计算 track 间人脸平均相似度
        3. 使用 DBSCAN 或贪心合并，相似度 > threshold 的合并为同一人
        4. 如果某 track 所有帧都无人脸，fallback 到全身 ReID
        """
```

**身份合并策略：**
1. **优先人脸**：对每个 track 的中间帧运行 InsightFace 人脸检测
2. **特征聚合**：每个 track 取 3-5 个关键帧的人脸 embedding 做平均
3. **相似度矩阵**：计算 track 间的余弦相似度
4. **贪心合并**：从相似度最高的 pair 开始合并，直到低于阈值
5. **Fallback**：无脸的 track 保持独立（全身 ReID 可选启用）

### 3.4 Cropper (cropper.py)

```python
class ActorCropper:
    """人物抠图 + 绿幕合成"""
    
    def __init__(self, bg_color: tuple = (0, 255, 0)):
        """
        - bg_color: 背景颜色 BGR，默认纯绿
        """
    
    def crop_actor(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        从帧中抠出人物区域，背景替换为绿色
        返回: 裁剪后的图像 (bbox 区域内为原画面，区域外为绿色)
        """
    
    def crop_batch(self, frames: list[np.ndarray], 
                   bboxes: list[BoundingBox]) -> list[np.ndarray]:
        """批量处理"""
```

**抠图策略：**
- 保持 bbox 区域的原画面不变
- bbox 区域外填充纯绿色 (#00FF00)
- 如果 bbox 超出画面边界，裁剪到画面尺寸
- 所有帧统一尺寸（取所有 bbox 的最大外接矩形，确保视频尺寸一致）

### 3.5 Merger (merger.py)

```python
class VideoMerger:
    """多段视频合并 + 编码输出"""
    
    def __init__(self, output_dir: str):
        """
        - output_dir: 输出目录
        """
    
    def merge_actor_segments(self, actor_id: str, 
                             frames_list: list[list[np.ndarray]],
                             fps: float = 30.0) -> str:
        """
        将同一演员的多个出场段合并为一个视频
        段与段之间加入 0.5 秒绿屏过渡
        返回: 输出文件路径
        """
    
    def generate_json(self, actor_data: dict) -> str:
        """
        生成 actor_info_json
        返回: JSON 文件路径
        """
```

### 3.6 ComfyUI Node (actor_extractor.py)

```python
class VideoActorExtractor:
    """ComfyUI 主节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "max_actors": ("INT", {"default": 10, "min": 1, "max": 50}),
                "face_threshold": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.99, "step": 0.05}),
                "fps_sample": ("INT", {"default": 3, "min": 1, "max": 30}),
                "min_track_length": ("INT", {"default": 5, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")  # (json_path, output_dir)
    RETURN_NAMES = ("actor_info_json", "output_dir")
    FUNCTION = "extract"
    CATEGORY = "video/actor"
    
    def extract(self, video_path, max_actors, face_threshold, fps_sample, min_track_length):
        # 调用完整管线
        ...
```

## 4. 数据流

```
video.mp4 (1920x1080, 30fps, 5min)
  │
  ▼  [按 fps_sample=3 采样 → 900 帧]
  │
  ▼  Detector (YOLOv8n)
     └─ 每帧输出: [(x1,y1,x2,y2,conf), ...]
     └─ 900 帧 × 平均2人 = ~1800 个 bbox
  │
  ▼  Tracker (ByteTrack)
     └─ 输出: {track_0: [(0, bbox), (1, bbox), ..., (120, bbox)],
               track_1: [(0, bbox), ..., (120, bbox)],
               track_2: [(200, bbox), ..., (350, bbox)], ...}
     └─ 注意: track_2 从 200 帧才开始（人物中间出场）
  │
  ▼  Identity (InsightFace + DBSCAN)
     └─ 假设 track_0 和 track_2 是同一人（面部匹配）
     └─ 输出: {track_0: "actor_0", track_1: "actor_1", track_2: "actor_0"}
  │
  ▼  Cropper (按 actor 分组)
     └─ actor_0: [frame_0~120, frame_200~350]
     └─ actor_1: [frame_0~120]
  │
  ▼  Merger (ffmpeg 编码)
     └─ output/actor_0.mp4 (两段合并，中间 0.5s 绿屏)
     └─ output/actor_1.mp4 (单段)
     └─ output/actor_info.json
```

## 5. 显存/性能优化

| 优化项 | 方法 |
|--------|------|
| **帧采样** | 默认 3fps 采样，5分钟视频从 9000 帧降到 900 帧 |
| **YOLOv8n** | nano 模型仅 ~6MB 参数，显存 < 1GB |
| **InsightFace 按需** | 只对 track 的关键帧（3-5帧/track）运行，不是逐帧 |
| **CPU 回退** | macOS 无 GPU 时自动切 CPU，速度可接受（3fps 采样） |
| **批量处理** | YOLO 支持 batch inference，一次处理 8 帧 |
| **ffmpeg 硬件编码** | 如果可用，使用 h264_nvenc 或 h264_videotoolbox |

## 6. 错误处理

- 视频解码失败 → 抛出明确错误（格式不支持）
- 检测到 0 人 → 返回空 JSON，不报错
- InsightFace 模型下载失败 → 尝试本地路径，否则跳过人脸识别（仅用追踪）
- 显存不足 → 自动降级到 CPU，降低 batch size
