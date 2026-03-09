# TennisVision iOS App - Implementation Plan

## Overview
Build the complete TennisVision iOS 26 app with real TrackNet CoreML model, Vision pose estimation, video file playback, and live camera support.

---

## Phase 1: Model Conversion (Python)

### Task 1.1: Write TrackNet conversion script
- **File**: `Scripts/convert_tracknet.py`
- The actual model architecture (from `model_best.pt`) uses 18 conv blocks (conv1-conv18), NOT the encoder/decoder structure in CLAUDE.md
- Architecture: conv1-2 → pool → conv3-4 → pool → conv5-7 → pool → conv8-10 (bottleneck) → upsample → conv11-13 → upsample → conv14-15 → upsample → conv16-17 → conv18
- Output is 256-class per-pixel (NOT single-channel sigmoid)
- Convert to `.mlpackage` with `convert_to="mlprogram"`, `compute_units=ALL`
- Input shape: `(1, 9, 360, 640)`, Output shape: `(1, 256, 360, 640)`

### Task 1.2: Run conversion
```bash
source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate
python Scripts/convert_tracknet.py
```
- Output: `TennisVision/Models/BallTracker.mlpackage`

---

## Phase 2: Xcode Project Setup

### Task 2.1: Install XcodeGen
```bash
brew install xcodegen
```

### Task 2.2: Create `project.yml`
- Target: TennisVision (iOS 26.0)
- Swift 6, SwiftUI lifecycle
- Frameworks: AVFoundation, CoreML, Vision, Metal, MetalKit
- Source groups matching CLAUDE.md directory structure
- Include .mlpackage in Models/ as resources
- Include test videos as app resources (for on-device testing)
- Info.plist with camera + photo library permissions

### Task 2.3: Create directory structure
```
TennisVision/
├── App/
├── Models/          ← .mlpackage goes here
├── Camera/
├── Inference/
├── Overlay/
├── ViewModels/
├── Views/
├── Utils/
└── Resources/       ← test videos bundled here
```

### Task 2.4: Generate .xcodeproj
```bash
xcodegen generate
```

---

## Phase 3: Swift Source Files

### Task 3.1: App Entry Point
- **`App/TennisVisionApp.swift`** — SwiftUI @main entry point

### Task 3.2: Camera Layer
- **`Camera/CameraManager.swift`** — AVCaptureSession, 1280x720, 30fps, BGRA output
- **`Camera/VideoFileReader.swift`** — NEW: AVAssetReader-based video file playback, feeds same pipeline as camera
- **`Camera/FrameProcessor.swift`** — Resize to 640x360, 3-frame sliding window, BGRA→RGB float32 conversion

### Task 3.3: Inference Layer
- **`Inference/BallTrackerInference.swift`** — CoreML inference, 256-class softmax→argmax to find ball position from heatmap
- **`Inference/PoseEstimationInference.swift`** — Vision VNDetectHumanBodyPoseRequest, 12 key joints
- **`Inference/CourtDetectorInference.swift`** — Mock implementation (no model weights), returns empty result

### Task 3.4: UI Layer
- **`Views/ContentView.swift`** — Main view with camera/video toggle, mode selector
- **`Views/CameraPreviewView.swift`** — UIViewRepresentable wrapping AVCaptureVideoPreviewLayer
- **`Views/VideoPlayerView.swift`** — NEW: Video file playback view with play/pause/seek controls
- **`Views/StatsOverlayView.swift`** — FPS, ball status, pose status HUD
- **`Overlay/OverlayView.swift`** — Canvas rendering: ball position, trajectory, skeleton, court lines

### Task 3.5: ViewModel
- **`ViewModels/AnalysisViewModel.swift`** — Coordinates camera/video input → inference → UI updates
  - Supports two modes: `.camera` and `.videoFile(URL)`
  - Inference on dedicated serial queue
  - FPS tracking, trajectory buffer (30 frames)

### Task 3.6: Utilities
- **`Utils/CGImage+Resize.swift`** — Image resize helpers
- **`Utils/CoordinateTransform.swift`** — Coordinate mapping between model space and screen space

---

## Phase 4: Build & Verify

### Task 4.1: Generate Xcode project and verify build
```bash
xcodegen generate
xcodebuild build -project TennisVision.xcodeproj -scheme TennisVision -destination 'generic/platform=iOS' -allowProvisioningUpdates
```

### Task 4.2: Fix any Swift 6 concurrency issues
- Ensure all result types are `Sendable`
- Ensure `@MainActor` isolation is correct
- Fix any data race warnings

---

## Key Design Decisions

1. **256-class output handling**: The real TrackNet outputs 256 classes per pixel. We apply softmax along the channel dimension, take argmax to get a grayscale heatmap (0-255), then find the peak to locate the ball.

2. **Video file playback**: Uses `AVAssetReader` + `AVAssetReaderTrackOutput` to decode frames at controllable rate, converting to the same `CMSampleBuffer`/`CVPixelBuffer` format the camera provides.

3. **Court detection**: Mock for now — returns `isDetected = false`. The UI is wired up so when a real model is added later, it works automatically.

4. **Frame skipping strategy**: Ball tracking runs every frame. Pose runs every frame. Court detection would run every 10 frames (when real model added).

---

## File Count: ~16 Swift files + 1 Python script + 1 project.yml + 1 Info.plist
