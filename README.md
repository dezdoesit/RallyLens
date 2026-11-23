# RallyLens

Real-time tennis analysis app for iOS 26, running entirely on-device with CoreML + Apple Vision.

Tracks the tennis ball, detects player pose, and identifies court lines — all in real-time on iPhone.

The current Xcode target is named `TennisVision`.

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
cd RallyLens
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
