# TennisVision

Real-time tennis analysis app for iOS 26 / iPhone 15 Pro Max.

## Features

- **Ball Tracking** — TrackNet CoreML model detects tennis ball position every frame
- **Pose Estimation** — Apple Vision framework detects player skeleton (12 key joints)
- **Court Detection** — Placeholder for court line keypoint detection (model TBD)
- **Dual Input** — Live camera capture or video file playback from Photos library
- **Real-time Overlay** — Canvas-rendered ball position, trajectory trail, and skeleton

## Requirements

- Xcode 26+
- iOS 26.0+
- iPhone with A-series chip (optimized for A17 Pro / iPhone 15 Pro Max)

## Project Structure

```
TennisVision/
├── App/                    App entry point
├── Camera/
│   ├── CameraManager       AVCaptureSession, 1280×720 @ 30fps
│   ├── VideoFileReader      AVAssetReader-based video file playback
│   └── FrameProcessor       Resize to 640×360, 3-frame sliding window
├── Inference/
│   ├── BallTrackerInference    TrackNet CoreML (256-class per-pixel output)
│   ├── PoseEstimationInference Apple Vision VNDetectHumanBodyPoseRequest
│   └── CourtDetectorInference  Mock (no model weights yet)
├── Overlay/
│   └── OverlayView          Canvas: ball dot, trajectory, skeleton, court lines
├── ViewModels/
│   └── AnalysisViewModel     Coordinates input → inference → UI
├── Views/
│   ├── ContentView           Main view with camera/video mode toggle
│   ├── CameraPreviewView     UIViewRepresentable camera preview
│   └── StatsOverlayView      FPS, ball status, pose status HUD
├── Utils/
│   └── CoordinateTransform   Model-space ↔ screen-space mapping
├── Models/
│   └── BallTracker.mlpackage CoreML model (auto-generated from model_best.pt)
└── Resources/               Bundled test videos (optional)
```

## Setup

### 1. Install XcodeGen

```bash
brew install xcodegen
```

### 2. Generate Xcode Project

```bash
xcodegen generate
```

### 3. Model Conversion (already done)

The TrackNet model has already been converted to `BallTracker.mlpackage`. To reconvert:

```bash
# Activate Python environment
source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate

# Convert
python Scripts/convert_tracknet.py
```

**Model architecture:** 18 ConvBlocks (Conv2d + ReLU + BatchNorm2d), U-Net encoder-decoder.
- Input: `(1, 9, 360, 640)` — 3 consecutive RGB frames stacked
- Output: `(1, 256, 360, 640)` — 256-class per-pixel softmax classification

### 4. Build & Run

Open `TennisVision.xcodeproj` in Xcode, select your iPhone target, and run.

Or build from command line:

```bash
xcodebuild build \
  -project TennisVision.xcodeproj \
  -scheme TennisVision \
  -destination 'generic/platform=iOS'
```

## Usage

1. **Camera Mode** — Tap "Camera" to start live analysis from the rear camera
2. **Video Mode** — Tap "Video" to pick a tennis video from your Photos library
3. **Stop** — Tap "Stop" to halt processing

The overlay displays:
- **Yellow dot** — Detected ball position
- **Yellow trail** — Ball trajectory (last 30 frames)
- **Green skeleton** — Player body pose
- **HUD** — FPS counter, ball detection status, pose detection status

## Test Videos

Two test videos are included in the project root:
- `Videos_20260309092934.mp4` — 1280×720, 30fps, 30 seconds
- `Videos_20260309092950.mp4` — 960×544, 29fps, 135 seconds

Import these to your iPhone's Photos library, then use the "Video" button in the app to analyze them.

## Architecture Notes

- **Swift 6 Strict Concurrency** — All result types are `Sendable`. The `InferenceEngine` runs on a dedicated serial queue. UI updates dispatch to `@MainActor`.
- **Frame Pipeline** — Camera/video → `FrameProcessor` (resize + 3-frame buffer) → `InferenceEngine` (ball + pose + court) → `AnalysisViewModel` → SwiftUI Canvas overlay
- **CoreML Compute Units** — Set to `.all` (CPU + GPU + ANE). On A17 Pro, the Neural Engine handles most ops.
- **Court Detection** — Currently returns mock empty results. Add a `CourtDetector.mlpackage` and update `CourtDetectorInference.swift` to enable.

## Performance Targets (iPhone 15 Pro Max)

| Component | Expected Latency |
|---|---|
| TrackNet inference | ~4-6ms (ANE) |
| Vision Pose | ~2-3ms (ANE) |
| Total pipeline | ~15-30ms |
| Target framerate | 30fps |
