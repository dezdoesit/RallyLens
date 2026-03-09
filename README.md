# TennisVision

Real-time tennis analysis app for iOS 26, running entirely on-device with CoreML + Apple Vision.

Tracks the tennis ball, detects player pose, and identifies court lines — all in real-time on iPhone.

## Features

- **Ball Tracking** — WASB (HRNet) model detects and tracks the tennis ball across frames with F1=94.0%
- **Pose Estimation** — Apple Vision framework detects 12 body joints (shoulders, elbows, wrists, hips, knees, ankles)
- **Court Line Detection** — CNN model identifies 14 court keypoints to draw court lines
- **Dual Input** — Live camera capture or import video from photo library
- **Real-time Overlay** — Canvas-based rendering of ball position, trajectory trail, skeleton, and court lines
- **Video Playback** — Original video plays alongside the analysis overlay

## Requirements

- iPhone with A17 Pro or later (tested on iPhone 15 Pro Max)
- iOS 26.0+
- Xcode 26
- Python 3.11+ (for model conversion only)

## Setup

### 1. Clone & Generate Project

```bash
git clone <repo-url>
cd Swing
brew install xcodegen   # if not installed
xcodegen generate
open TennisVision.xcodeproj
```

### 2. Model Conversion

The CoreML models (`BallTracker.mlpackage`, `CourtDetector.mlpackage`) are pre-built in `TennisVision/Models/`. To re-convert from PyTorch weights:

```bash
pip install torch coremltools numpy

# Ball tracker (WASB HRNet)
python Scripts/convert_wasb.py

# Court detector
python Scripts/convert_court_detector.py
```

### 3. Build & Run

Open in Xcode 26, select your iPhone, and run. The app requests camera permission on first launch.

## Usage

| Mode | How | Description |
|------|-----|-------------|
| Camera | Tap "Camera" button | Live rear camera feed with real-time analysis |
| Video | Tap "Video" button | Pick a video from photo library, plays with overlay |
| Stop | Tap "Stop" button | Stops the current analysis session |

The HUD in the top-right shows FPS, ball detection status, and pose detection status.

The overlay displays:
- **Yellow dot + white ring** — Current ball position
- **Yellow trail** — Ball trajectory (last 30 frames)
- **Green skeleton** — Player body pose (12 joints + 10 bone connections)
- **Cyan lines** — Tennis court lines (14 keypoints)

---

## Project Architecture

```
TennisVision/
├── App/
│   └── TennisVisionApp.swift           # @main entry point
├── Camera/
│   ├── CameraManager.swift             # AVCaptureSession (1280×720, 30fps)
│   ├── FrameProcessor.swift            # Resize → 512×288, 3-frame sliding window
│   └── VideoFileReader.swift           # AVAssetReader for video file input
├── Inference/
│   ├── BallTrackerInference.swift      # WASB CoreML ball detection
│   ├── PoseEstimationInference.swift   # Apple Vision body pose
│   └── CourtDetectorInference.swift    # Court keypoint CoreML detection
├── Models/
│   ├── BallTracker.mlpackage           # WASB HRNet (2.6 MB)
│   └── CourtDetector.mlpackage         # Court detector (20 MB)
├── Overlay/
│   └── OverlayView.swift              # Canvas rendering (ball, skeleton, court)
├── ViewModels/
│   └── AnalysisViewModel.swift        # InferenceEngine + AnalysisViewModel
├── Views/
│   ├── ContentView.swift              # Main UI (camera/video/overlay layers)
│   ├── CameraPreviewView.swift        # AVCaptureVideoPreviewLayer wrapper
│   ├── VideoPlayerView.swift          # AVPlayerLayer for video playback
│   └── StatsOverlayView.swift         # FPS / status HUD
└── Utils/
    └── CoordinateTransform.swift      # Normalized ↔ screen coordinate conversion
```

### Design Pattern: MVVM + Actor Concurrency

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   View Layer    │ ←── │  AnalysisVM      │ ←── │  Inference      │
│   (SwiftUI)     │     │  (@MainActor)    │     │  Engine         │
│                 │     │                  │     │  (serial queue) │
│  ContentView    │     │ @Published state │     │  backpressure   │
│  OverlayView    │     │ camera/video mgr │     │  frame proc     │
│  StatsOverlay   │     │ trajectory buf   │     │  3 models       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Key concurrency patterns (Swift 6 strict):**

- `AnalysisViewModel` — `@MainActor` isolated, all `@Published` state updates on main thread
- `InferenceEngine` — Dedicated serial `DispatchQueue` (`.userInitiated` QoS)
- **Backpressure** — `OSAllocatedUnfairLock` atomic flag: 1 frame in-flight, excess frames dropped
- All result types (`BallDetectionResult`, `PoseResult`, `CourtDetectionResult`) conform to `Sendable`
- Frame callbacks use `@Sendable` closures to cross actor boundaries safely

### Data Flow

```
Camera / VideoFileReader
    │
    ▼  CMSampleBuffer
InferenceEngine.processFrame()
    │  (drop if busy — backpressure)
    ▼
FrameProcessor
    │  resize to 512×288
    │  3-frame FIFO buffer
    │  BGRA → RGB float32, normalize [0,1]
    ▼  output: [9 × 288 × 512] float array
    │
    ├──→ BallTrackerInference     (every frame,  input: 9×288×512)
    ├──→ PoseEstimationInference  (every frame,  input: CVPixelBuffer)
    └──→ CourtDetectorInference   (every 10 frames, self-resizes to 360×640)
    │
    ▼  DispatchQueue.main.async
AnalysisViewModel (@Published properties)
    │
    ▼
SwiftUI Canvas overlay + HUD
```

---

## Models

### 1. Ball Tracker — WASB (HRNet)

| Property | Value |
|----------|-------|
| Architecture | HRNet (High-Resolution Network) with 4-stage multi-resolution branches |
| Variant | WASB-SBDT |
| Parameters | 1.48M |
| Input | `1 × 9 × 288 × 512` — 3 consecutive RGB frames, normalized [0,1] |
| Output | `1 × 1 × 288 × 512` — sigmoid heatmap, peak = ball position |
| F1 Score | 94.0% on tennis |
| CoreML size | 2.6 MB |
| Threshold | confidence > 0.3 |

**How it works:**

The model takes 3 consecutive video frames stacked channel-wise (9 channels). HRNet processes them through 4 stages of parallel multi-resolution branches — high-resolution branches preserve spatial detail while low-resolution branches capture context. Feature fusion layers exchange information across resolutions at each stage. The final output is a sigmoid heatmap where the peak indicates ball position. Peak detection uses `vDSP_maxvi` from the Accelerate framework.

**Why WASB over TrackNet:**

| Metric | TrackNet (original) | WASB HRNet |
|--------|-------------------|------------|
| Parameters | 10.7M | 1.48M (**7x smaller**) |
| F1 Score | ~89% | 94.0% (**+5**) |
| PyTorch weights | 42 MB | 6 MB |
| CoreML size | ~20 MB | 2.6 MB |

**Source:**
- Repository: [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (NTT Communications)
- Model Zoo: https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md
- License: MIT
- Weights file: `wasb_tennis_best.pth.tar`
- Conversion script: `Scripts/convert_wasb.py`

### 2. Court Detector — U-Net CNN

| Property | Value |
|----------|-------|
| Architecture | 18-ConvBlock U-Net (3× MaxPool → bottleneck → 3× Upsample) |
| Parameters | ~10.7M |
| Input | `1 × 3 × 360 × 640` — single RGB frame, normalized [0,1] |
| Output | `1 × 14 × 360 × 640` — 14 heatmaps (one per court keypoint) |
| Activation | Sigmoid |
| CoreML size | 20 MB |
| Threshold | >= 8/14 keypoints with confidence > 0.5 |

**How it works:**

Takes a single video frame and outputs 14 heatmaps, each corresponding to a court keypoint. Each heatmap's peak location gives the (x, y) position of one keypoint. The 14 keypoints define the full tennis court geometry:

```
 0 ──────────────── 1       Top baseline (doubles)
 │  4 ──────────  6 │       Singles sidelines (top)
 │  │   8 ── 9   │ │       Top service line
 │  │  12 ──13   │ │       Center service line
 │  │  10 ──11   │ │       Bottom service line
 │  5 ──────────  7 │       Singles sidelines (bottom)
 2 ──────────────── 3       Bottom baseline (doubles)
```

Court lines are drawn by connecting keypoint pairs. Detection runs every 10 frames since court position changes slowly. The court detector handles its own resize from the ball tracker's 512×288 pixelBuffer to 360×640 internally.

**Source:**
- Repository: [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
- License: MIT
- Weights file: `model_tennis_court_det.pt`
- Conversion script: `Scripts/convert_court_detector.py`

### 3. Pose Estimation — Apple Vision

| Property | Value |
|----------|-------|
| Framework | Apple Vision (`VNDetectHumanBodyPoseRequest`) |
| Engine | Neural Engine (ANE), built into iOS |
| Input | `CVPixelBuffer` (any resolution) |
| Output | 12 joint positions, normalized [0,1] |
| Latency | ~2-3ms on A17 Pro |
| Model size | 0 (built into iOS) |

**How it works:**

Uses Apple's built-in body pose detection running on the Neural Engine. No external weights needed. Tracks 12 joints relevant to tennis analysis:

| Group | Joints |
|-------|--------|
| Arms | Right/Left Shoulder, Elbow, Wrist |
| Legs | Right/Left Hip, Knee, Ankle |

Joints with confidence below 0.3 are filtered out. The skeleton overlay draws 10 bone connections (shoulder→elbow→wrist, hip→knee→ankle, shoulder↔shoulder, hip↔hip).

**Source:**
- Apple Documentation: [VNDetectHumanBodyPoseRequest](https://developer.apple.com/documentation/vision/vndetecthumanbodyposerequest)
- Built into iOS 14+, no external model or license required

---

## Rendering

The analysis overlay uses SwiftUI `Canvas` with 4 layers drawn in order:

1. **Court lines** — Cyan, 1.5pt — Connects 14 keypoints (baselines, sidelines, service lines)
2. **Skeleton** — Green, 2pt bones + 8px joint dots — 10 connections across 12 joints
3. **Trajectory trail** — Yellow, 2pt, 60% opacity — Last 30 ball positions as a path
4. **Ball position** — Yellow fill + white 2pt stroke — 20px circle at current detection

All coordinates are stored normalized [0,1] and scaled to screen dimensions at render time.

---

## Performance

Measured on iPhone 15 Pro Max (A17 Pro, ANE 35 TOPS):

| Component | Latency | Frequency |
|-----------|---------|-----------|
| WASB Ball Tracking | ~3-5ms | Every frame |
| Apple Vision Pose | ~2-3ms | Every frame |
| Court Detection | ~5-8ms | Every 10 frames |
| Frame Preprocessing | ~2ms | Every frame |
| **Total Pipeline** | **~10-15ms** | **~30 FPS** |

Memory optimizations:
- Reused `MLMultiArray` input buffers (no per-frame allocation)
- `autoreleasepool` wrapping each inference cycle
- Backpressure — single frame in-flight, excess dropped
- WASB model is 7x smaller than original TrackNet

---

## Conversion Scripts

| Script | Input Weights | Output | Description |
|--------|--------------|--------|-------------|
| `Scripts/convert_wasb.py` | `wasb_tennis_best.pth.tar` | `BallTracker.mlpackage` | WASB HRNet → CoreML (current ball tracker) |
| `Scripts/convert_tracknet.py` | `model_best.pt` | `BallTracker.mlpackage` | TrackNet U-Net → CoreML (legacy alternative) |
| `Scripts/convert_court_detector.py` | `model_tennis_court_det.pt` | `CourtDetector.mlpackage` | Court detector → CoreML |

All conversions use `coremltools` with:
- `convert_to="mlprogram"` — required for Neural Engine acceleration
- `minimum_deployment_target=iOS16`
- `compute_units=ALL` — CPU + GPU + Neural Engine

---

## Test Videos

Two test videos are included (git-ignored):
- `Videos_20260309092934.mp4` — 1280×720, 30fps, 30 seconds
- `Videos_20260309092950.mp4` — 960×544, 29fps, 135 seconds

Import to your iPhone's Photos library, then use the "Video" button in the app.

## License

Model weights are subject to their respective licenses:
- WASB-SBDT: [MIT License](https://github.com/nttcom/WASB-SBDT/blob/main/LICENSE)
- TennisCourtDetector: [MIT License](https://github.com/yastrebksv/TennisCourtDetector)
- Apple Vision: Built into iOS, no separate license

## Todoooooo
整个系统需要的模型，按模块梳理：

---

## 完整模型清单

```
TennisVision Pipeline
│
├── 1. 球追踪          → TrackNet / TrackNetV3
├── 2. 球场检测        → Court Keypoint Detector
├── 3. 姿态估计        → Apple Vision (内置，免费)
├── 4. 击球分类        → Stroke Classifier (轻量)
└── 5. 弹跳检测        → Bounce Detector (可选)
```

---

## 模型 1：球追踪 — TrackNet

| 项目 | 详情 |
|---|---|
| **任务** | 输入连续3帧，输出球的位置 heatmap |
| **输入** | `(1, 9, 360, 640)` Float32 |
| **输出** | `(1, 1, 360, 640)` Heatmap |
| **参数量** | ~4M |
| **推理耗时** | ~5ms (ANE) |
| **权重来源** | 需训练或下载预训练权重 |

**预训练权重获取：**
```
https://github.com/yastrebksv/TrackNet
→ 提供 balltrackerbest.pt

https://github.com/nttcom/WASB-SBDT  (TrackNetV3)
→ 更新的版本，精度更高
```

**CoreML 转换可行性：** ✅ 已验证（Apple Developer Forums 有人成功）

---

## 模型 2：球场检测 — Court Keypoint Detector

| 项目 | 详情 |
|---|---|
| **任务** | 输入一帧，回归 14 个关键点坐标 |
| **输入** | `(1, 3, 360, 640)` Float32 |
| **输出** | `(1, 28)` — 14点×(x,y) |
| **Backbone** | ResNet-18（轻量）或 ResNet-50（精准）|
| **参数量** | ~11M (ResNet-18) |
| **推理耗时** | ~8ms (ANE) |
| **权重来源** | 需下载或自训练 |

**预训练权重获取：**
```
https://github.com/yastrebksv/TennisCourtDetector
→ 提供 model_tennis_court_det.pt
→ PyTorch，ResNet-50 backbone
```

**CoreML 转换可行性：** ✅ ResNet 系列转换成熟

---

## 模型 3：姿态估计 — Apple Vision（内置）

这个**不需要任何模型文件**，Apple 已经内置在 Vision framework 里：

```swift
// 2D 姿态（iOS 14+）
VNDetectHumanBodyPoseRequest()   // 17个关节点

// 3D 姿态（iOS 17+，iPhone 15 Pro Max 支持）
VNDetectHumanBodyPose3DRequest() // 需要 LiDAR
```

| 项目 | 详情 |
|---|---|
| **关节点数量** | 17个（COCO格式）|
| **运行位置** | 完全在 ANE，系统级优化 |
| **推理耗时** | ~2-3ms |
| **需要下载** | ❌ 系统内置 |

---

## 模型 4：击球类型分类 — Stroke Classifier

这是可以**自己训练的最轻量模型**，输入是关节点序列：

| 项目 | 详情 |
|---|---|
| **任务** | 正手 / 反手 / 发球 / 截击 / 其他 |
| **输入** | 关节点时序序列 `(T, 34)` — T帧 × 17点×xy |
| **输出** | 5类 softmax 概率 |
| **架构** | LSTM 或小型 Transformer |
| **参数量** | <1M（极轻量）|
| **推理耗时** | <1ms |
| **训练数据** | THETIS 数据集（300段标注视频）|

```
THETIS 数据集：
https://thetis.image.ntua.gr/
→ 12种击球类型，多视角，免费学术使用
```

这个模型的输入**来自模型3的输出**（关节点坐标），所以不需要额外视频帧，极为轻量。

---

## 模型 5：弹跳检测 — Bounce Detector（可选）

| 项目 | 详情 |
|---|---|
| **任务** | 判断球是否刚刚弹跳 |
| **输入** | 球的轨迹序列 `(N, 3)` — N帧 × (x,y,confidence) |
| **输出** | 是否弹跳 (binary) |
| **架构** | 传统 ML：`TimeSeriesForestClassifier` 或小 LSTM |
| **参数量** | 极小 |
| **运行位置** | CPU，但几乎不耗时 |

这个不是深度学习模型，用**轨迹曲率突变**就能检测，甚至可以写规则：

```swift
// 简单规则：Y方向速度从正变负 = 弹跳
func detectBounce(trajectory: [CGPoint]) -> Bool {
    guard trajectory.count >= 3 else { return false }
    let n = trajectory.count
    let vy_prev = trajectory[n-2].y - trajectory[n-3].y
    let vy_curr = trajectory[n-1].y - trajectory[n-2].y
    return vy_prev > 0 && vy_curr < 0  // 速度方向反转
}
```

---

## 模型依赖关系图

```
相机帧
  │
  ├──→ [TrackNet] ──────────→ 球坐标 (x,y,conf)
  │                                │
  │                                ├──→ [Bounce Detector] → 是否弹跳
  │                                │
  ├──→ [Court Detector] ───→ 14关键点 + Homography
  │                                │
  │         球坐标 × H ────────────→ 落点实际坐标 → in/out 判断
  │
  └──→ [Vision Pose] ──────→ 17关节点
                                   │
                            [Stroke Classifier] → 击球类型
```

---

## 汇总对比

| 模型 | 需要下载 | 文件大小 | 转 CoreML | 频率 |
|---|---|---|---|---|
| TrackNet | ✅ | ~60MB | 需转换 | 每帧 |
| Court Detector | ✅ | ~90MB | 需转换 | 每10帧 |
| Vision Pose | ❌ 内置 | 0 | 无需 | 每帧 |
| Stroke Classifier | 自训练 | <5MB | 需转换 | 每次击球 |
| Bounce Detector | ❌ 规则 | 0 | 无需 | 每帧 |

**最小可行版本（MVP）：** 只用 TrackNet + Vision Pose，App 的核心功能（球追踪 + 姿态叠加）就能跑起来，Court Detector 和 Stroke Classifier 后续迭代加入。