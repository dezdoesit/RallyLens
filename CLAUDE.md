# TennisVision – iOS 26 Tennis Analysis App
## Claude Code 开发指南

---

## 项目概览

构建一个运行在 **iOS 26 / iPhone 15 Pro Max** 上的实时网球分析 App，实现：

1. **球追踪** — TrackNet CoreML 模型，逐帧定位网球坐标
2. **姿态估计** — Apple Vision `VNDetectHumanBodyPoseRequest`（原生，ANE 加速）
3. **球场线检测** — 轻量 CNN CoreML 模型，检测球场 14 个关键点
4. **实时视频推理** — AVFoundation 采集，640×360 降采样送模型，Metal overlay 渲染结果

**目标设备：** iPhone 15 Pro Max（A17 Pro，ANE 35 TOPS）  
**最低支持：** iOS 26.0  
**语言：** Swift 6，SwiftUI，Xcode 26  
**架构：** MVVM + Actor-based concurrency

---

## 目录结构

```
TennisVision/
├── CLAUDE.md                        ← 本文件
├── TennisVision.xcodeproj/
├── TennisVision/
│   ├── App/
│   │   ├── TennisVisionApp.swift
│   │   └── AppDelegate.swift
│   ├── Models/                      ← CoreML .mlpackage 文件放这里
│   │   ├── BallTracker.mlpackage    ← TrackNet 转换后的模型
│   │   └── CourtDetector.mlpackage ← 球场检测模型
│   ├── Camera/
│   │   ├── CameraManager.swift      ← AVFoundation 采集
│   │   └── FrameProcessor.swift     ← 帧预处理 + 分发
│   ├── Inference/
│   │   ├── BallTrackerInference.swift
│   │   ├── PoseEstimationInference.swift
│   │   └── CourtDetectorInference.swift
│   ├── Overlay/
│   │   ├── OverlayView.swift        ← Metal/Canvas 渲染层
│   │   └── TrajectoryRenderer.swift
│   ├── ViewModels/
│   │   └── AnalysisViewModel.swift
│   ├── Views/
│   │   ├── ContentView.swift
│   │   ├── CameraPreviewView.swift
│   │   └── StatsOverlayView.swift
│   └── Utils/
│       ├── CGImage+Resize.swift
│       └── CoordinateTransform.swift
├── Scripts/
│   ├── convert_tracknet.py          ← PyTorch → CoreML 转换脚本
│   └── convert_court_detector.py
└── ModelWeights/                    ← 原始 .pt 权重（不提交 git）
    ├── balltrackerbest.pt
    └── court_detector.pt
```

---

## Phase 0：环境准备

### 0.1 Xcode 项目创建

```bash
# 使用 Xcode 26 创建新项目
# File → New → Project → iOS → App
# Product Name: TennisVision
# Interface: SwiftUI
# Language: Swift
# Minimum Deployments: iOS 26.0
# Bundle ID: com.yourname.tennisvision
```

### 0.2 Info.plist 权限

在 `TennisVision/Info.plist` 添加：

```xml
<key>NSCameraUsageDescription</key>
<string>TennisVision 需要摄像头权限来实时分析网球比赛</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>TennisVision 需要访问相册来分析已录制的视频</string>
```

### 0.3 Capabilities

在 Xcode → Target → Signing & Capabilities 中添加：
- **CoreML** (自动包含，无需手动添加)
- 确认 Deployment Target = **iOS 26.0**

---

## Phase 1：CoreML 模型准备

### 1.1 TrackNet 转换（在 Mac 上运行）

**安装依赖：**
```bash
pip install coremltools==8.1 torch torchvision numpy
```

**`Scripts/convert_tracknet.py`：**

```python
import torch
import coremltools as ct
import numpy as np

# ── 模型结构定义（与训练时保持一致）──────────────────────────
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class BallTrackerNet(nn.Module):
    """TrackNetV2 架构：输入 3帧×3通道=9通道，输出 heatmap"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(ConvBlock(9, 64), ConvBlock(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512)
        )
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(ConvBlock(768, 256), ConvBlock(256, 256), ConvBlock(256, 256))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(ConvBlock(384, 128), ConvBlock(128, 128))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(ConvBlock(192, 64), ConvBlock(64, 64))
        self.out  = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.out(d1))

# ── 转换 ──────────────────────────────────────────────────────
def convert(weights_path: str, output_path: str):
    model = BallTrackerNet()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # TrackNet 输入：3帧 RGB 叠加 = (1, 9, 360, 640)
    example = torch.rand(1, 9, 360, 640)
    traced  = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input_frames",
            shape=(1, 9, 360, 640),
            dtype=np.float32
        )],
        outputs=[ct.TensorType(name="heatmap", dtype=np.float32)],
        convert_to="mlprogram",                     # 必须，ANE 加速
        minimum_deployment_target=ct.target.iOS16,  # iOS 26 向下兼容 iOS 16 API
        compute_units=ct.ComputeUnit.ALL,            # CPU+GPU+ANE
    )

    # 添加元数据
    mlmodel.short_description = "TrackNet: Tennis Ball Tracker"
    mlmodel.input_description["input_frames"] = "3 consecutive RGB frames stacked (9×360×640), normalized [0,1]"
    mlmodel.output_description["heatmap"] = "Gaussian heatmap of ball position (1×360×640)"

    mlmodel.save(output_path)
    print(f"✅ 模型已保存：{output_path}")
    print(f"   输入 shape: {example.shape}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    convert("ModelWeights/balltrackerbest.pt", "TennisVision/Models/BallTracker.mlpackage")
```

**运行：**
```bash
cd 项目根目录
python Scripts/convert_tracknet.py
```

### 1.2 模型权重下载

TrackNet 官方预训练权重：
```bash
# 从 Google Drive 下载（TrackNet 论文作者提供）
# https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut
# 下载后放到 ModelWeights/balltrackerbest.pt
```

### 1.3 球场检测模型

球场 14 关键点检测，使用轻量 ResNet-18 backbone：
- 权重来源：https://github.com/yastrebksv/TennisCourtDetector
- 同样用 `coremltools` 转换，输入 `(1, 3, 360, 640)`，输出 `(1, 28)` (14点×xy)
- 参考 `Scripts/convert_court_detector.py`（结构同上，自行实现）

### 1.4 验证模型

在 Xcode 中：
1. 将 `.mlpackage` 拖入 `TennisVision/Models/` 目录
2. 点击模型文件 → 查看 Xcode 自动生成的接口预览
3. 确认 "Compute Units: All (CPU, GPU, Neural Engine)" 显示正确

---

## Phase 2：相机采集层

### `Camera/CameraManager.swift`

```swift
import AVFoundation
import CoreVideo

/// 负责 AVCaptureSession 配置和帧输出
/// 运行在独立的 serial queue，不阻塞主线程
@MainActor
final class CameraManager: NSObject, ObservableObject {

    @Published var isRunning = false
    @Published var permissionGranted = false

    // 外部通过这个 closure 接收帧
    var onFrame: ((CMSampleBuffer) -> Void)?

    private let session       = AVCaptureSession()
    private let videoOutput   = AVCaptureVideoDataOutput()
    private let sessionQueue  = DispatchQueue(label: "com.tennisvision.camera", qos: .userInitiated)

    // 供 CameraPreviewView 使用
    let previewLayer: AVCaptureVideoPreviewLayer

    override init() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        super.init()
    }

    func requestPermission() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            permissionGranted = true
            await setupSession()
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            permissionGranted = granted
            if granted { await setupSession() }
        default:
            permissionGranted = false
        }
    }

    private func setupSession() async {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            // iPhone 15 Pro Max 用 1280×720，后续 resize 到 640×360
            self.session.sessionPreset = .hd1280x720

            // 选择后置广角摄像头
            guard let device = AVCaptureDevice.default(
                .builtInWideAngleCamera, for: .video, position: .back
            ),
            let input = try? AVCaptureDeviceInput(device: device),
            self.session.canAddInput(input) else { return }

            self.session.addInput(input)

            // 视频输出：BGRA 格式，方便 CoreML
            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.setSampleBufferDelegate(self, queue: self.sessionQueue)

            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
            }

            // 锁定帧率为 30fps
            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            if let device = self.session.inputs.first as? AVCaptureDeviceInput {
                try? device.device.lockForConfiguration()
                device.device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                device.device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
                device.device.unlockForConfiguration()
            }

            self.session.commitConfiguration()
        }
    }

    func start() {
        sessionQueue.async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async { self?.isRunning = true }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async { self?.isRunning = false }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        onFrame?(sampleBuffer)
    }
}
```

---

## Phase 3：帧预处理

### `Camera/FrameProcessor.swift`

```swift
import CoreVideo
import CoreImage
import Accelerate

/// 将相机帧 resize 到 640×360，归一化，并维护滑动窗口（3帧缓冲）
final class FrameProcessor {

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let targetSize = CGSize(width: 640, height: 360)

    // 3帧滑动窗口，TrackNet 需要连续帧
    private var frameBuffer: [CVPixelBuffer] = []
    private let bufferCapacity = 3

    /// 返回 (当前帧pixelBuffer, 3帧叠加的Float32数组[9×360×640])
    func process(_ sampleBuffer: CMSampleBuffer) -> (CVPixelBuffer, [Float32])? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }

        // Resize to 640×360
        guard let resized = resize(pixelBuffer, to: targetSize) else { return nil }

        // 维护帧缓冲
        frameBuffer.append(resized)
        if frameBuffer.count > bufferCapacity { frameBuffer.removeFirst() }
        guard frameBuffer.count == bufferCapacity else { return nil }

        // 3帧 → 展平为 Float32 数组，shape [9, 360, 640]，归一化 [0,1]
        let floatArray = stackFrames(frameBuffer)
        return (resized, floatArray)
    }

    private func resize(_ buffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let scaleX = size.width  / CGFloat(CVPixelBufferGetWidth(buffer))
        let scaleY = size.height / CGFloat(CVPixelBufferGetHeight(buffer))
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var output: CVPixelBuffer?
        CVPixelBufferCreate(
            nil,
            Int(size.width), Int(size.height),
            kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
            &output
        )
        guard let out = output else { return nil }
        ciContext.render(scaled, to: out)
        return out
    }

    private func stackFrames(_ buffers: [CVPixelBuffer]) -> [Float32] {
        let w = 640, h = 360
        var result = [Float32](repeating: 0, count: 9 * h * w)

        for (frameIdx, buffer) in buffers.enumerated() {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

            guard let base = CVPixelBufferGetBaseAddress(buffer) else { continue }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let ptr = base.assumingMemoryBound(to: UInt8.self)

            // BGRA → RGB，归一化 /255
            for y in 0..<h {
                for x in 0..<w {
                    let offset = y * bytesPerRow + x * 4
                    let b = Float32(ptr[offset])     / 255.0
                    let g = Float32(ptr[offset + 1]) / 255.0
                    let r = Float32(ptr[offset + 2]) / 255.0
                    // Channel-first layout: [frame*3+channel, h, w]
                    result[(frameIdx * 3 + 0) * h * w + y * w + x] = r
                    result[(frameIdx * 3 + 1) * h * w + y * w + x] = g
                    result[(frameIdx * 3 + 2) * h * w + y * w + x] = b
                }
            }
        }
        return result
    }
}
```

---

## Phase 4：推理层

### `Inference/BallTrackerInference.swift`

```swift
import CoreML
import Accelerate

struct BallDetectionResult {
    let x: Float          // 归一化坐标 [0,1]
    let y: Float
    let confidence: Float
    let isVisible: Bool
}

final class BallTrackerInference {

    private let model: BallTracker  // Xcode 自动生成的类
    private let heatmapW = 640, heatmapH = 360

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all   // CPU + GPU + ANE
        self.model = try BallTracker(configuration: config)
    }

    func predict(frames: [Float32]) throws -> BallDetectionResult {
        // 构造 MLMultiArray (shape: 1 × 9 × 360 × 640)
        let shape: [NSNumber] = [1, 9, 360, 640]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)

        frames.withUnsafeBufferPointer { ptr in
            mlArray.dataPointer.assumingMemoryBound(to: Float32.self)
                .initialize(from: ptr.baseAddress!, count: ptr.count)
        }

        let input  = BallTrackerInput(input_frames: mlArray)
        let output = try model.prediction(from: input)
        let heatmap = output.heatmap  // shape: 1 × 1 × 360 × 640

        return parseHeatmap(heatmap)
    }

    private func parseHeatmap(_ heatmap: MLMultiArray) -> BallDetectionResult {
        let count = heatmapW * heatmapH
        let ptr   = heatmap.dataPointer.assumingMemoryBound(to: Float32.self)

        // 找最大值坐标
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(count))

        let y = Int(maxIdx) / heatmapW
        let x = Int(maxIdx) % heatmapW

        return BallDetectionResult(
            x: Float(x) / Float(heatmapW),
            y: Float(y) / Float(heatmapH),
            confidence: maxVal,
            isVisible: maxVal > 0.5   // 置信度阈值
        )
    }
}
```

### `Inference/PoseEstimationInference.swift`

```swift
import Vision
import CoreGraphics

struct PoseResult {
    struct Joint {
        let name: VNHumanBodyPoseObservation.JointName
        let position: CGPoint   // 归一化坐标 [0,1]
        let confidence: Float
    }
    let joints: [Joint]
    let isDetected: Bool
}

final class PoseEstimationInference {

    // 关注的关键关节（用于击球分析）
    private let targetJoints: [VNHumanBodyPoseObservation.JointName] = [
        .rightWrist, .leftWrist,
        .rightElbow, .leftElbow,
        .rightShoulder, .leftShoulder,
        .rightHip, .leftHip,
        .rightKnee, .leftKnee,
        .rightAnkle, .leftAnkle
    ]

    private lazy var request = VNDetectHumanBodyPoseRequest()

    func predict(pixelBuffer: CVPixelBuffer) throws -> PoseResult {
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )
        try handler.perform([request])

        guard let observation = request.results?.first else {
            return PoseResult(joints: [], isDetected: false)
        }

        let joints: [PoseResult.Joint] = targetJoints.compactMap { jointName in
            guard let point = try? observation.recognizedPoint(jointName),
                  point.confidence > 0.3 else { return nil }
            return PoseResult.Joint(
                name: jointName,
                position: CGPoint(x: point.location.x, y: 1 - point.location.y), // flip Y
                confidence: Float(point.confidence)
            )
        }

        return PoseResult(joints: joints, isDetected: !joints.isEmpty)
    }
}
```

### `Inference/CourtDetectorInference.swift`

```swift
import CoreML

struct CourtDetectionResult {
    /// 14 个关键点，归一化坐标 [0,1]
    let keypoints: [CGPoint]   // count == 14
    let isDetected: Bool
}

final class CourtDetectorInference {

    private let model: CourtDetector
    private let numKeypoints = 14

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try CourtDetector(configuration: config)
    }

    func predict(frames: [Float32]) throws -> CourtDetectionResult {
        let shape: [NSNumber] = [1, 3, 360, 640]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)

        // 只用第一帧（3通道）
        let frameSize = 3 * 360 * 640
        frames[0..<frameSize].withUnsafeBufferPointer { ptr in
            mlArray.dataPointer.assumingMemoryBound(to: Float32.self)
                .initialize(from: ptr.baseAddress!, count: ptr.count)
        }

        let input  = CourtDetectorInput(input_frames: mlArray)
        let output = try model.prediction(from: input)
        // output shape: [1, 28]  — 14点 × (x, y)
        let rawPtr = output.keypoints.dataPointer.assumingMemoryBound(to: Float32.self)

        var keypoints: [CGPoint] = []
        for i in 0..<numKeypoints {
            let x = CGFloat(rawPtr[i * 2])
            let y = CGFloat(rawPtr[i * 2 + 1])
            keypoints.append(CGPoint(x: x, y: y))
        }

        return CourtDetectionResult(keypoints: keypoints, isDetected: true)
    }
}
```

---

## Phase 5：ViewModel（推理调度）

### `ViewModels/AnalysisViewModel.swift`

```swift
import SwiftUI
import Combine
import AVFoundation

@MainActor
final class AnalysisViewModel: ObservableObject {

    @Published var ballResult:   BallDetectionResult?
    @Published var poseResult:   PoseResult?
    @Published var courtResult:  CourtDetectionResult?
    @Published var ballTrajectory: [CGPoint] = []   // 最近 30 帧轨迹
    @Published var fps: Double = 0

    let cameraManager = CameraManager()

    private let frameProcessor    = FrameProcessor()
    private let ballTracker       = try? BallTrackerInference()
    private let poseEstimator     = PoseEstimationInference()
    private let courtDetector     = try? CourtDetectorInference()

    // 推理串行队列，避免帧积压
    private let inferenceQueue = DispatchQueue(label: "com.tennisvision.inference", qos: .userInitiated)

    // FPS 计算
    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount = 0

    private let maxTrajectoryLength = 30

    init() {
        cameraManager.onFrame = { [weak self] sampleBuffer in
            self?.processFrame(sampleBuffer)
        }
    }

    func startSession() async {
        await cameraManager.requestPermission()
        cameraManager.start()
    }

    func stopSession() {
        cameraManager.stop()
    }

    private func processFrame(_ sampleBuffer: CMSampleBuffer) {
        inferenceQueue.async { [weak self] in
            guard let self else { return }

            guard let (pixelBuffer, frames) = self.frameProcessor.process(sampleBuffer) else { return }

            let ball  = try? self.ballTracker?.predict(frames: frames)
            let pose  = try? self.poseEstimator.predict(pixelBuffer: pixelBuffer)
            let court = try? self.courtDetector?.predict(frames: frames)

            let now = CACurrentMediaTime()
            self.frameCount += 1
            let elapsed = now - self.lastFrameTime
            let fps = elapsed > 0 ? Double(self.frameCount) / elapsed : 0
            if elapsed > 1.0 { self.frameCount = 0; self.lastFrameTime = now }

            DispatchQueue.main.async {
                self.ballResult  = ball
                self.poseResult  = pose
                self.courtResult = court
                self.fps         = fps

                // 更新轨迹
                if let b = ball, b.isVisible {
                    let pt = CGPoint(x: CGFloat(b.x), y: CGFloat(b.y))
                    self.ballTrajectory.append(pt)
                    if self.ballTrajectory.count > self.maxTrajectoryLength {
                        self.ballTrajectory.removeFirst()
                    }
                }
            }
        }
    }
}
```

---

## Phase 6：UI 层

### `Views/ContentView.swift`

```swift
import SwiftUI

struct ContentView: View {

    @StateObject private var vm = AnalysisViewModel()

    var body: some View {
        ZStack {
            // 相机预览
            CameraPreviewView(previewLayer: vm.cameraManager.previewLayer)
                .ignoresSafeArea()

            // 推理结果 Overlay
            OverlayView(
                ballResult:   vm.ballResult,
                poseResult:   vm.poseResult,
                courtResult:  vm.courtResult,
                trajectory:   vm.ballTrajectory
            )
            .ignoresSafeArea()

            // HUD（右上角）
            VStack {
                HStack {
                    Spacer()
                    StatsOverlayView(
                        fps:          vm.fps,
                        ballVisible:  vm.ballResult?.isVisible ?? false,
                        poseDetected: vm.poseResult?.isDetected ?? false
                    )
                    .padding()
                }
                Spacer()
            }
        }
        .task { await vm.startSession() }
        .onDisappear { vm.stopSession() }
    }
}
```

### `Views/CameraPreviewView.swift`

```swift
import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        previewLayer.frame = uiView.bounds
    }
}
```

### `Overlay/OverlayView.swift`

```swift
import SwiftUI

struct OverlayView: View {

    let ballResult:  BallDetectionResult?
    let poseResult:  PoseResult?
    let courtResult: CourtDetectionResult?
    let trajectory:  [CGPoint]

    var body: some View {
        Canvas { context, size in
            // 1. 球场关键点
            if let court = courtResult {
                drawCourt(context: context, size: size, keypoints: court.keypoints)
            }

            // 2. 姿态骨骼
            if let pose = poseResult, pose.isDetected {
                drawSkeleton(context: context, size: size, joints: pose.joints)
            }

            // 3. 球轨迹
            drawTrajectory(context: context, size: size, points: trajectory)

            // 4. 当前球位置
            if let ball = ballResult, ball.isVisible {
                let pt = CGPoint(x: CGFloat(ball.x) * size.width,
                                 y: CGFloat(ball.y) * size.height)
                let circle = Path(ellipseIn: CGRect(x: pt.x - 8, y: pt.y - 8, width: 16, height: 16))
                context.fill(circle, with: .color(.yellow))
                context.stroke(circle, with: .color(.white), lineWidth: 2)
            }
        }
    }

    private func drawCourt(context: GraphicsContext, size: CGSize, keypoints: [CGPoint]) {
        guard keypoints.count == 14 else { return }
        // 连接关键点绘制球场线
        let courtLines: [(Int, Int)] = [
            (0,1),(1,2),(2,3),(3,0),   // 底线
            (4,5),(5,6),(6,7),(7,4),   // 发球线
            (0,4),(1,5),(2,6),(3,7),   // 边线
            (8,9),(10,11)              // 中线
        ]
        var path = Path()
        for (a, b) in courtLines {
            guard a < keypoints.count, b < keypoints.count else { continue }
            let ptA = CGPoint(x: keypoints[a].x * size.width, y: keypoints[a].y * size.height)
            let ptB = CGPoint(x: keypoints[b].x * size.width, y: keypoints[b].y * size.height)
            path.move(to: ptA)
            path.addLine(to: ptB)
        }
        context.stroke(path, with: .color(.cyan.opacity(0.7)), lineWidth: 1.5)
    }

    private func drawSkeleton(context: GraphicsContext, size: CGSize, joints: [PoseResult.Joint]) {
        let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
            (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
            (.leftShoulder,  .leftElbow),  (.leftElbow,  .leftWrist),
            (.rightHip,      .rightKnee),  (.rightKnee,  .rightAnkle),
            (.leftHip,       .leftKnee),   (.leftKnee,   .leftAnkle),
            (.rightShoulder, .leftShoulder),
            (.rightHip,      .leftHip),
        ]
        let jointMap = Dictionary(uniqueKeysWithValues: joints.map { ($0.name, $0.position) })

        var bonePath = Path()
        for (a, b) in connections {
            guard let pa = jointMap[a], let pb = jointMap[b] else { continue }
            bonePath.move(to: CGPoint(x: pa.x * size.width, y: pa.y * size.height))
            bonePath.addLine(to: CGPoint(x: pb.x * size.width, y: pb.y * size.height))
        }
        context.stroke(bonePath, with: .color(.green.opacity(0.8)), lineWidth: 2)

        for joint in joints {
            let pt = CGPoint(x: joint.position.x * size.width, y: joint.position.y * size.height)
            let dot = Path(ellipseIn: CGRect(x: pt.x - 4, y: pt.y - 4, width: 8, height: 8))
            context.fill(dot, with: .color(.green))
        }
    }

    private func drawTrajectory(context: GraphicsContext, size: CGSize, points: [CGPoint]) {
        guard points.count > 1 else { return }
        var path = Path()
        for (i, pt) in points.enumerated() {
            let screenPt = CGPoint(x: pt.x * size.width, y: pt.y * size.height)
            if i == 0 { path.move(to: screenPt) }
            else       { path.addLine(to: screenPt) }
        }
        // 轨迹渐变：旧 → 透明，新 → 亮黄
        context.stroke(path, with: .color(.yellow.opacity(0.6)), style: StrokeStyle(lineWidth: 2, lineCap: .round))
    }
}
```

### `Views/StatsOverlayView.swift`

```swift
import SwiftUI

struct StatsOverlayView: View {
    let fps: Double
    let ballVisible: Bool
    let poseDetected: Bool

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            Label(String(format: "%.0f FPS", fps), systemImage: "speedometer")
            Label(ballVisible  ? "Ball ✓" : "Ball –", systemImage: "circle.fill")
                .foregroundStyle(ballVisible ? .yellow : .gray)
            Label(poseDetected ? "Pose ✓" : "Pose –", systemImage: "figure.tennis")
                .foregroundStyle(poseDetected ? .green : .gray)
        }
        .font(.caption.monospaced())
        .foregroundStyle(.white)
        .padding(8)
        .background(.black.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
```

---

## Phase 7：iOS 26 特有注意事项

### 7.1 Swift 6 并发

iOS 26 默认开启 Swift 6 严格并发检查：

```swift
// ✅ 正确：@MainActor 隔离 UI 相关状态
@MainActor final class AnalysisViewModel: ObservableObject { ... }

// ✅ 正确：Sendable 类型跨并发边界传递
struct BallDetectionResult: Sendable { ... }
struct PoseResult: Sendable { ... }
struct CourtDetectionResult: Sendable { ... }

// ❌ 错误（会编译报错）：在非 @MainActor 上下文中修改 @Published 属性
// 必须用 DispatchQueue.main.async 或 await MainActor.run { }
```

### 7.2 iOS 26 Liquid Glass UI（可选）

```swift
// iOS 26 新增 .glassEffect() modifier（类似 visionOS）
StatsOverlayView(...)
    .glassEffect()                    // iOS 26 新 API
    // .background(.black.opacity(0.5))  // 旧 API，替换掉
```

### 7.3 CoreML on iOS 26

- iOS 26 的 ANE driver 支持更多算子，TrackNet 的 Upsample + BatchNorm 全部在 ANE 上运行
- 使用 `MLModelConfiguration().computeUnits = .all` 让系统自动调度

### 7.4 Vision Framework 更新

```swift
// iOS 26 中 VNDetectHumanBodyPoseRequest 已更新，支持 3D 姿态
// 如需 3D 关节点（iOS 26+）：
let request3D = VNDetectHumanBodyPose3DRequest()
// 但需要 TrueDepth / LiDAR，iPhone 15 Pro Max 支持
```

---

## Phase 8：性能优化

### 8.1 帧率策略

```swift
// 不要每帧都跑球场检测（变化慢），每 10 帧跑一次
private var courtFrameCounter = 0
// 在 processFrame 中：
courtFrameCounter += 1
if courtFrameCounter % 10 == 0 {
    court = try? self.courtDetector?.predict(frames: frames)
}
```

### 8.2 内存优化

```swift
// CVPixelBuffer 复用池，避免频繁分配
// FrameProcessor 中的 ciContext 设置为类属性（复用），而非每帧创建
private let ciContext = CIContext(options: [
    .useSoftwareRenderer: false,
    .priorityRequestLow: false
])
```

### 8.3 预期性能（iPhone 15 Pro Max）

| 任务 | 耗时 | 设备 |
|---|---|---|
| TrackNet 推理 | ~4-6ms | ANE |
| Vision Pose | ~2-3ms | ANE |
| Court Detect | ~5-8ms | ANE |
| 总 pipeline | ~15ms | → 60fps 理论可达 |
| 实际（含 UI） | ~30ms | → 稳定 30fps |

---

## Phase 9：模型不存在时的 Fallback

开发阶段 CoreML 模型可能还没准备好，用 Mock 数据先跑通 UI：

```swift
// BallTrackerInference.swift 中
func predictMock() -> BallDetectionResult {
    // 模拟球在屏幕上做圆周运动
    let t = Date().timeIntervalSinceReferenceDate
    return BallDetectionResult(
        x: Float(0.5 + 0.3 * cos(t)),
        y: Float(0.5 + 0.3 * sin(t)),
        confidence: 0.95,
        isVisible: true
    )
}
```

---

## 开发顺序建议

```
Step 1: 创建 Xcode 项目，配置 Info.plist 权限
Step 2: 实现 CameraManager，先在真机上看到相机预览
Step 3: 实现 FrameProcessor，打印帧尺寸确认 resize 正确
Step 4: 实现 OverlayView，用 Mock 数据验证绘制逻辑
Step 5: 在 Mac 上运行 convert_tracknet.py，生成 .mlpackage
Step 6: 集成 BallTrackerInference，验证球检测
Step 7: 集成 PoseEstimationInference（Vision，零配置即可用）
Step 8: 集成 CourtDetectorInference
Step 9: 性能调优（帧率策略 + 内存）
Step 10: iOS 26 Liquid Glass UI 收尾
```

---

## 常见问题

**Q: Xcode 显示模型 interface 为 "4 script tags" 而不是正常预览？**  
A: 运行 `xcrun coremlcompiler metadata path/to/BallTracker.mlpackage`，检查 specificationVersion 字段是否存在。通常是 coremltools 版本过旧导致，升级到 8.x 重新转换。

**Q: 模型 fallback 到 CPU，ANE 没有激活？**  
A: 必须用 `convert_to="mlprogram"`（不是旧的 `"neuralnetwork"`），且 `minimum_deployment_target` ≥ `ct.target.iOS16`。

**Q: AVCaptureSession 在 iOS 26 上崩溃？**  
A: iOS 26 起，AVCaptureSession 的某些配置需要在 `sessionQueue` 上调用，不能在 MainActor 上直接操作。本文档的 `CameraManager` 已正确处理。

**Q: Swift 6 编译报 "Sending non-Sendable" 错误？**  
A: 所有跨 Actor 边界传递的数据结构必须标记 `Sendable`。`BallDetectionResult`, `PoseResult`, `CourtDetectionResult` 都已标记。

---

*最后更新：2026-03*  
*目标：iOS 26.0+，iPhone 15 Pro Max，Xcode 26，Swift 6*
