# ComfyUI-VideoActorExtract

从视频中自动检测、追踪、识别所有出场演员，输出每个演员的出镜信息和绿幕抠图视频。

## 功能

- 逐帧人物检测 (YOLOv8)
- 多目标追踪 (ByteTrack / 内置贪婪追踪器)
- 人脸识别 + 身份聚类 (InsightFace)
- 绿幕抠图输出 (纯绿背景 #00FF00)
- 多段出场合并 (段间 0.5s 绿屏过渡)
- JSON 输出 (帧范围、时间段)

## 安装

```bash
cd ~/App/ComfyUI/custom_nodes
ln -s ~/Project/ComfyUI-VideoActorExtract ComfyUI-VideoActorExtract

# 安装依赖 (使用 comfyui conda 环境)
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate comfyui
pip install ultralytics opencv-python insightface onnxruntime scikit-learn ffmpeg-python
```

### 下载 YOLOv8 模型

首次使用前需要下载 yolov8n.pt 模型文件:

```bash
# 方法1: 通过代理下载
https_proxy=http://127.0.0.1:8118 curl -L -o ~/Project/ComfyUI-VideoActorExtract/yolov8n.pt \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

# 方法2: 手动下载后放入项目目录
# 下载地址: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
# 放入: ~/Project/ComfyUI-VideoActorExtract/yolov8n.pt
```

### 下载 InsightFace 模型

首次运行会自动下载 buffalo_l 模型 (~350MB)，需联网。

## 节点参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| video_path | STRING | (必填) | 输入视频路径 |
| model_path | STRING | yolov8n.pt | YOLO 模型路径 (可填绝对路径) |
| max_actors | INT | 10 | 最大检测演员数 |
| face_threshold | FLOAT | 0.6 | 人脸相似度阈值 |
| fps_sample | INT | 3 | 采样帧率 (降低可提速) |
| min_track_length | INT | 5 | 最小追踪帧数 |

## 输出

- `actor_info_json`: JSON 字符串，包含每个演员的出场时间段、帧范围
- `output_dir`: 输出目录路径，包含:
  - `actor_0.mp4`, `actor_1.mp4` ... (每个演员的绿幕抠图视频)
  - `actor_info.json` (结构化 JSON 文件)

## JSON 输出格式

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

## 项目结构

```
ComfyUI-VideoActorExtract/
├── __init__.py                    # ComfyUI 入口
├── nodes/
│   └── actor_extractor.py         # 主节点
├── pipeline/
│   ├── detector.py                # YOLOv8 人物检测
│   ├── tracker.py                 # ByteTrack / 贪婪追踪
│   ├── identity.py                # InsightFace 身份聚类
│   ├── cropper.py                 # 绿幕抠图
│   └── merger.py                  # 视频合并 + JSON 输出
├── core/
│   ├── config.py                  # 全局配置
│   └── video_reader.py            # 视频读取
├── js/widget.js                   # 前端组件
├── docs/
│   ├── requirements.md            # 需求文档
│   └── architecture.md            # 架构设计
└── requirements.txt               # Python 依赖
```

## 限制

- 人物最小检测尺寸 ~20px，过小人物可能漏检
- 人脸识别依赖面部可见，全程无正面的人物会被分配独立ID
- 建议视频时长 ≤ 5 分钟，人物同时出现 ≤ 5 人
