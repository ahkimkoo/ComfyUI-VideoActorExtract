# ComfyUI 工作流示例

## video-actor-extract.json

从视频中提取每个演员的完整镜头，输出绿幕抠图视频。

### 使用方法

1. 打开 ComfyUI (http://127.0.0.1:8188)
2. 将 `video-actor-extract.json` 文件**拖拽**到 ComfyUI 画布中
3. 在 "VideoActorExtractor" 节点中设置参数:
   - `video_path`: 输入视频文件路径 (必填)
   - `model_path`: YOLO 模型路径，默认 `yolov8n.pt`
   - `max_actors`: 最大演员数，默认 10
   - `face_threshold`: 人脸相似度阈值，默认 0.6
   - `fps_sample`: 采样帧率，默认 3
   - `min_track_length`: 最小追踪帧数，默认 5
4. 点击 "Queue Prompt" 运行

### 输出

- 绿幕视频保存在: `ComfyUI/output/ComfyUI-VideoActorExtract/actor_X.mp4`
- JSON 信息保存在: `ComfyUI/output/ComfyUI-VideoActorExtract/actor_info.json`
- JSON 文本也会通过 "Save Text File" 节点保存为 txt 文件

### 参数调优

| 场景 | 建议参数 |
|------|---------|
| 单人舞蹈视频 | min_track_length=5, fps_sample=3 |
| 多人对话视频 | min_track_length=8, face_threshold=0.5 |
| 快速动作视频 | fps_sample=5, min_track_length=3 |
| 过滤误检 | min_track_length=10 |
