# ComfyUI-VideoActorExtract 需求文档

## 1. 项目概述

一个 ComfyUI 自定义节点插件，用于从短视频（5分钟以内）中自动检测、追踪、识别所有出场演员，并输出每个演员的出镜信息和绿幕抠图视频。

### 目标场景
- 剧情类短视频（多人对话、表演）
- 跳舞类短视频（单人或多人舞蹈）
- 人物最多同时出现 3-5 人
- 视频时长 ≤ 5 分钟

## 2. 功能需求

### 2.1 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| `video` | 文件路径 (str) | 输入视频文件路径 |
| `max_actors` | int (可选, 默认 10) | 最大检测演员数 |
| `face_threshold` | float (可选, 默认 0.6) | 人脸识别相似度阈值（低于此值视为不同人） |
| `fps_sample` | int (可选, 默认 3) | 视频采样帧率（用于追踪，降低计算量） |
| `min_track_length` | int (可选, 默认 5) | 最小追踪帧数（低于此值视为误检，过滤掉） |

### 2.2 输出
| 输出 | 类型 | 说明 |
|------|------|------|
| `actor_info_json` | JSON string | 每个演员的完整信息（见下方格式） |
| `actor_videos` | 文件路径列表 | 每个演员的绿幕抠图视频（多段合并） |

### 2.3 actor_info_json 格式
```json
{
  "actors": [
    {
      "actor_id": "actor_0",
      "segment_count": 2,
      "total_frames": 156,
      "segments": [
        {
          "segment_id": 0,
          "start_frame": 24,
          "end_frame": 89,
          "start_time_sec": 0.8,
          "end_time_sec": 2.97,
          "frame_count": 66
        },
        {
          "segment_id": 1,
          "start_frame": 200,
          "end_frame": 289,
          "start_time_sec": 6.67,
          "end_time_sec": 9.63,
          "frame_count": 90
        }
      ]
    }
  ],
  "video_info": {
    "total_frames": 300,
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "duration_sec": 10.0
  }
}
```

### 2.4 actor_videos 格式
- 每个演员一个独立 MP4 文件
- 文件名格式: `actor_{id}.mp4`
- 输出路径: ComfyUI `output/ComfyUI-VideoActorExtract/` 下按 session 分目录
- 画面内容：从原视频中抠出该演员的矩形区域，背景替换为纯绿色 (#00FF00)
- 多个出场段合并为同一个视频，段与段之间可加入 0.5 秒绿屏过渡

## 3. 非功能需求

### 3.1 性能
- 5分钟 1080p 视频，在 RTX 3060 上处理时间 < 3 分钟
- 显存占用 < 4 GB（YOLOv8n + ByteTrack + InsightFace）
- 支持 CPU 回退（Mac 用户）

### 3.2 兼容性
- Python ≥ 3.10
- ComfyUI 最新版
- macOS (Apple Silicon) / Linux (NVIDIA GPU)

### 3.3 依赖
- `ultralytics` (YOLOv8)
- `bytetrack` (或 `track` 模块)
- `insightface`
- `onnxruntime`
- `opencv-python`
- `ffmpeg` (系统级)
- `numpy`
- `scikit-learn` (DBSCAN 聚类)

## 4. ComfyUI 节点设计

### 4.1 主节点: VideoActorExtractor
```
输入:
  - video_path (STRING)
  - max_actors (INT, 默认 10)
  - face_threshold (FLOAT, 默认 0.6)
  - fps_sample (INT, 默认 3)
  - min_track_length (INT, 默认 5)

输出:
  - actor_info_json (STRING)
  - output_dir (STRING) - 输出目录路径
```

### 4.2 可视化节点: VideoActorVisualize (可选)
```
输入:
  - actor_info_json (STRING)
  - video_path (STRING)
  - show_tracking (BOOLEAN) - 是否显示追踪框

输出:
  - IMAGE (ComfyUI 图像格式) - 带标注的帧预览
```

## 5. 边界情况处理

- **人物出画/入画**: ByteTrack 自动处理，追踪消失即段结束
- **人物交叉遮挡**: ByteTrack 有遮挡预测能力，短遮挡不断ID
- **多人物相似度极高**（双胞胎/同款服装）: 依赖人脸识别，如果面部不可用则 fallback 到全身 ReID
- **极小人物**（远距离/画中画）: YOLOv8n 最小检测 20px，过小人物可能漏检
- **视频无音频/特殊编码**: 使用 OpenCV 解码，不支持的格式给出明确错误提示
