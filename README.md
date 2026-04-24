# ComfyUI-VideoActorExtract

从视频中自动检测、追踪、识别所有出场人物，输出每个人物的绿幕抠图视频和出镜元数据。

**核心能力：**
- 逐帧人物分割（YOLOv8-seg）
- 质心距离多目标追踪（MaskTracker）
- 基于人脸识别的身份聚类（InsightFace），支持跨时段合并 + 共现互斥
- 纯绿幕背景输出（#00FF00）
- 多段出场自动合并为单个视频
- JSON 结构化元数据（帧范围、时间戳）

---

## 功能说明

### 它能做什么

1. **人物检测与分割** — 使用 YOLOv8-seg 逐帧检测画面中所有人物，生成像素级分割掩码
2. **多目标追踪** — 通过质心距离匹配，将同一人物的连续帧关联为 track
3. **身份聚类** — 使用 InsightFace 提取人脸特征，将同一人不同时间段的 track 合并为同一个 actor
   - **共现互斥约束**：同一帧出现的两个人物绝不会合并为同一人
   - **跨时段合并**：人物消失后重新出现，自动合并到同一 actor
4. **绿幕抠图** — 将人物轮廓外的区域替换为纯绿色背景
5. **视频编码** — 同一 actor 的所有出场段合并为一个 MP4 文件
6. **元数据输出** — 生成 JSON 文件，记录每个 actor 的出镜时间段

### 典型使用场景

- 影视剪辑：快速提取每个角色的所有镜头
- 视频分析：统计每个人物的出场时长
- 内容创作：绿幕抠图用于二次创作
- 人脸分析：按人物分组后进一步分析表情/动作

---

## 架构说明

### 处理管线

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VideoActorExtractor Node                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  IMAGE batch (VHS LoadVideo)                                         │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────┐                                                    │
│  │ PersonSegmenter │ YOLOv8-seg, 逐帧人物分割                         │
│  │  (每帧检测)     │ 输出: [(detection_idx, bool_mask), ...]          │
│  └──────┬───────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │  MaskTracker   │ 质心距离追踪, 帧间关联                              │
│  │  (帧间追踪)     │ 输出: {actor_id: [(frame_idx, masked_frame, area)]}│
│  └──────┬───────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │ IdentityCluster│ InsightFace 人脸特征 + 贪心聚类                    │
│  │  (身份合并)     │ 约束: 共现互斥 + 跨时段合并                        │
│  └──────┬───────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │ SegmentBuilder │ 连续段构建, 小间隙插值                              │
│  │  (段构建)       │ 输出: {actor_id: [segment_frames, ...]}            │
│  └──────┬───────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │    Merger      │ cv2.VideoWriter 编码 + JSON 生成                   │
│  │  (视频输出)     │ 输出: actor_X.mp4 + actor_info.json               │
│  └──────────────┘                                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **PersonSegmenter** | `pipeline/segmenter.py` | YOLOv8-seg 人物分割，输出像素级掩码 |
| **MaskTracker** | `pipeline/mask_tracker.py` | 质心距离追踪，将同一人的连续帧关联为 track |
| **IdentityCluster** | `pipeline/identity.py` | InsightFace 人脸识别 + 贪心聚类，合并同一个人 |
| **SegmentBuilder** | `nodes/actor_extractor.py` | 连续段构建，小间隙帧插值 |
| **Merger** | `pipeline/merger.py` | 视频编码 + JSON 元数据生成 |
| **VideoActorExtractor** | `nodes/actor_extractor.py` | ComfyUI 主节点，编排整个管线 |

### 身份聚类算法

```
输入: {track_id: [FrameRecord, ...]}
输出: {track_id: "actor_0", ...}

算法:
1. 对每个 track 采样最多 30 帧（优先面积最大的帧）
2. 每帧尝试多种 crop 尺寸 (0.6x, 1.0x, 1.5x, 2.0x) 检测人脸
3. 计算 track 间人脸 embedding 余弦相似度
4. 贪心聚类:
   - 遍历每个 track，找最佳匹配的现有 actor
   - 共现检查: 如果 track 与 actor 中任何成员在同一帧出现 → 禁止合并
   - 相似度 ≥ threshold → 合并
   - 否则 → 创建新 actor
5. 无脸 track → 时空重叠 fallback 合并
6. 重新编号为 actor_0, actor_1, ...
```

**关键约束：共现互斥**

同一帧出现的两个人物，即使人脸相似度很高，也绝不会合并为同一人。这解决了双胞胎、长相相似的人被错误合并的问题。

---

## 项目结构

```
ComfyUI-VideoActorExtract/
├── __init__.py                    # ComfyUI 入口 (importlib 动态加载)
├── requirements.txt               # Python 依赖
├── pyproject.toml                 # 项目元数据
├── nodes/
│   └── actor_extractor.py         # 主节点 + 连续段构建
├── pipeline/
│   ├── segmenter.py               # YOLOv8-seg 人物分割
│   ├── mask_tracker.py            # 质心距离追踪
│   ├── identity.py                # InsightFace 身份聚类
│   ├── detector.py                # YOLOv8 人物检测 (bbox)
│   ├── tracker.py                 # ByteTrack 追踪 (可选)
│   ├── cropper.py                 # 绿幕抠图 (历史遗留)
│   └── merger.py                  # 视频编码 + JSON 输出
├── core/
│   ├── config.py                  # 全局配置常量
│   └── video_reader.py            # 视频读取
├── docs/
│   ├── requirements.md            # 需求文档
│   └── architecture.md            # 架构设计
├── examples/
│   └── README.md                  # 工作流示例说明
└── js/
    └── widget.js                  # 前端组件
```

---

## 安装

### 1. 安装到 ComfyUI

```bash
cd ~/App/ComfyUI/custom_nodes
ln -s ~/Project/ComfyUI-VideoActorExtract ComfyUI-VideoActorExtract
```

### 2. 安装依赖

```bash
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate comfyui

cd ~/Project/ComfyUI-VideoActorExtract
pip install -r requirements.txt
```

**关键依赖：**
- `ultralytics` — YOLOv8 检测/分割
- `opencv-python` — 图像处理 + 视频编码
- `insightface` — 人脸识别
- `onnxruntime` — InsightFace 推理引擎
- `numpy` — 数值计算

### 3. 下载模型

```bash
# 创建模型目录
mkdir -p ~/App/ComfyUI/models/video-actor-extract/

# 下载 YOLOv8 检测模型
curl -L -o ~/App/ComfyUI/models/video-actor-extract/yolov8n.pt \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

# 下载 YOLOv8-seg 分割模型
curl -L -o ~/App/ComfyUI/models/video-actor-extract/yolov8n-seg.pt \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt"
```

InsightFace 模型 (`buffalo_l`, ~350MB) 会在首次运行时自动下载到 `~/.insightface/models/buffalo_l/`。

---

## 使用方法

### ComfyUI 工作流

1. 启动 ComfyUI: `python main.py --port 8188`
2. 打开 http://127.0.0.1:8188
3. 拖拽 `examples/video-actor-extract.json` 到画布（或手动创建）
4. 连接节点:
   ```
   VHS LoadVideo (IMAGE) → VideoActorExtractor → PreviewImage
   ```
5. 设置参数，点击 Queue Prompt

### 节点参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `images` | IMAGE | (必填) | 从 VHS LoadVideo 连接 |
| `model_path` | STRING | `yolov8n.pt` | YOLO 检测模型路径 |
| `seg_model_path` | STRING | `yolov8n-seg.pt` | YOLOv8-seg 分割模型路径 |
| `video_path` | STRING | `""` | 原始视频路径（用于精确元数据） |
| `max_actors` | INT | `10` | 最大检测人数 |
| `face_threshold` | FLOAT | `0.6` | 人脸相似度阈值 (0.1-0.99) |
| `min_track_length` | INT | `5` | 最小追踪帧数（过滤误检） |

### 参数调优指南

| 场景 | 建议参数 |
|------|---------|
| 单人视频 | `min_track_length=3` |
| 多人对话 | `face_threshold=0.65, min_track_length=5` |
| 过滤短暂误检 | `min_track_length=10` |
| 区分双胞胎/相似脸 | `face_threshold=0.75` |
| 合并远距离同一个人 | `face_threshold=0.5` |

---

## 输出

### 文件结构

每次运行在 `ComfyUI/output/ComfyUI-VideoActorExtract/{uuid}/` 下生成：

```
{uuid}/
├── actor_0.mp4          # 第一个人物的绿幕视频
├── actor_1.mp4          # 第二个人物的绿幕视频
├── ...
└── actor_info.json      # 结构化元数据
```

### JSON 格式

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
          "end_frame": 260,
          "start_time_sec": 6.67,
          "end_time_sec": 8.67,
          "frame_count": 61
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

---

## 测试

### 自动化测试脚本

```bash
# 使用 comfyui conda 环境
cd ~/Project/ComfyUI-VideoActorExtract
/opt/homebrew/anaconda3/envs/comfyui/bin/python -c "
import json, urllib.request, time

wf = {
    'prompt': {
        '1': {
            'class_type': 'VHS_LoadVideo',
            'inputs': {
                'video': 'your_video.mp4',
                'force_rate': 0,
                'custom_width': 0,
                'custom_height': 0,
                'frame_load_cap': 0,
                'skip_first_frames': 0,
                'select_every_nth': 1
            }
        },
        '2': {
            'class_type': 'VideoActorExtractor',
            'inputs': {
                'images': ['1', 0],
                'model_path': 'yolov8n.pt',
                'seg_model_path': 'yolov8n-seg.pt',
                'max_actors': 10,
                'face_threshold': 0.6,
                'min_track_length': 3
            }
        },
        '3': {
            'class_type': 'PreviewImage',
            'inputs': {'images': ['2', 2]}
        }
    }
}

data = json.dumps(wf).encode()
req = urllib.request.Request(
    'http://127.0.0.1:8188/prompt',
    data=data,
    headers={'Content-Type': 'application/json'}
)
r = urllib.request.urlopen(req, timeout=30).read()
pid = json.loads(r)['prompt_id']
print(f'Queued: {pid}')

for i in range(300):
    time.sleep(2)
    h = json.loads(urllib.request.urlopen(
        f'http://127.0.0.1:8188/history/{pid}', timeout=10
    ).read())
    if pid in h:
        s = h[pid].get('status', {})
        if s.get('status_str') == 'error':
            print('ERROR:', h[pid].get('messages', [])[-3:])
            break
        if h[pid].get('outputs'):
            print(f'Done! ({i*2}s)')
            for n, o in h[pid]['outputs'].items():
                print(f'  Node {n}: {list(o.keys())}')
            break
    if i % 15 == 0:
        print('.', end='', flush=True)
"
```

### 验证输出

```bash
# 查看最新输出
ls -lt ~/App/ComfyUI/output/ComfyUI-VideoActorExtract/ | head -5

# 检查 JSON 元数据
cat ~/App/ComfyUI/output/ComfyUI-VideoActorExtract/{uuid}/actor_info.json | python -m json.tool

# 检查视频文件
ffprobe ~/App/ComfyUI/output/ComfyUI-VideoActorExtract/{uuid}/actor_0.mp4
```

### 调试日志

ComfyUI 控制台会输出详细日志：

```
[VideoActorExtract] Step 1: Initializing person segmenter...
[PersonSegmenter] Loaded yolov8n-seg.pt on mps
[VideoActorExtract] Step 3: Detecting masks and tracking actors...
  Processed 50/290 frames (4.4s, 11.3 fps)
[MaskTracker] Closed actor_0: no match for 31 frames
[VideoActorExtract] Step 4: Tracking complete. 4 mask actors, 391 total detections
[VideoActorExtract] Step 5: Clustering actor identities...
[Identity] Face detection: 4/4 tracks have face embeddings
[Identity]   Track 0: 26 faces detected
[Identity] Building similarity matrix (4x4)...
[Identity] Merged track 3(197-288) -> actor_1 (sim=0.734)
[Identity] Final: 2 unique actors
```

---

## 限制

| 限制 | 说明 |
|------|------|
| 最小人物尺寸 | ~20px，过小人物可能漏检 |
| 人脸识别 | 依赖面部可见，全程无正面的人物会被分配独立 ID |
| 建议视频时长 | ≤ 5 分钟 |
| 建议同时人数 | ≤ 5 人 |
| 内存占用 | 全帧加载到内存，大视频需要足够 RAM |

---

## 许可证

MIT
