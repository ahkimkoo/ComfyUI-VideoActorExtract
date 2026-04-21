# ComfyUI 工作流示例

## video-actor-extract.json

从视频中提取每个演员的完整镜头，输出绿幕抠图视频。

### 使用方法

1. 打开 ComfyUI (http://127.0.0.1:8188)
2. 将 `video-actor-extract.json` 文件**拖拽**到 ComfyUI 画布中
3. 工作流包含三个节点:
   - **Load Video (Upload)** — 选择/上传视频
   - **Video Actor Extract** — 输入 IMAGE，输出 JSON 和路径
   - **Save Text File** — 保存 JSON 结果
4. 在 Load Video 节点中选择视频
5. 调整参数，点击 Queue Prompt

### VideoActorExtractor 节点参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| images | IMAGE | (连接) | 从 VHS LoadVideo 连接，必填 |
| model_path | STRING | yolov8n.pt | YOLO 模型路径 |
| video_path | STRING | (空) | 可选，原始视频路径（用于精确元数据） |
| max_actors | INT | 10 | 最多检测演员数 |
| face_threshold | FLOAT | 0.6 | 人脸相似度阈值 |
| min_track_length | INT | 5 | 最小追踪帧数 |

### 输出

- 绿幕视频: `ComfyUI/output/ComfyUI-VideoActorExtract/actor_X.mp4`
- JSON信息: `ComfyUI/output/ComfyUI-VideoActorExtract/actor_info.json`
- 文本文件: 通过 Save Text File 节点保存

### 参数调优

| 场景 | 建议参数 |
|------|---------|
| 单人舞蹈 | min_track_length=5 |
| 多人对话 | min_track_length=8, face_threshold=0.5 |
| 过滤误检 | min_track_length=10 |
