import SwiftUI
import Combine
import AVFoundation
import os

enum InputMode: Sendable {
    case camera
    case videoFile(URL)
}

/// Handles inference on a dedicated serial queue, publishing results to the main actor.
/// Uses an atomic flag to enforce backpressure — frames are dropped when inference is busy.
final class InferenceEngine: @unchecked Sendable {

    private let frameProcessor = FrameProcessor()
    private var ballTracker: BallTrackerInference?
    private let poseEstimator = PoseEstimationInference()
    private let courtDetector = CourtDetectorInference()

    private let inferenceQueue = DispatchQueue(label: "com.tennisvision.inference", qos: .userInitiated)

    // Backpressure: only one frame in-flight at a time
    private let isBusy = OSAllocatedUnfairLock(initialState: false)

    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount = 0
    private var courtFrameCounter = 0

    var onResults: ((_ ball: BallDetectionResult?, _ pose: PoseResult?, _ court: CourtDetectionResult?, _ fps: Double) -> Void)?

    init() {
        do {
            ballTracker = try BallTrackerInference()
            print("BallTracker model loaded successfully")
        } catch {
            print("Failed to load BallTracker model: \(error)")
        }
    }

    func processFrame(_ sampleBuffer: CMSampleBuffer) {
        // Drop frame if inference is already running
        let canProcess = isBusy.withLock { busy -> Bool in
            if busy { return false }
            busy = true
            return true
        }
        guard canProcess else { return }

        inferenceQueue.async { [self] in
            defer {
                isBusy.withLock { $0 = false }
            }

            autoreleasepool {
                guard let (pixelBuffer, frames) = frameProcessor.process(sampleBuffer) else { return }

                let ball = try? ballTracker?.predict(frames: frames)
                let pose = try? poseEstimator.predict(pixelBuffer: pixelBuffer)

                courtFrameCounter += 1
                var court: CourtDetectionResult? = nil
                if courtFrameCounter % 10 == 0 {
                    court = courtDetector.predict(frames: frames)
                }

                // FPS calculation
                let now = CACurrentMediaTime()
                frameCount += 1
                let elapsed = now - lastFrameTime
                let currentFps = elapsed > 0 ? Double(frameCount) / elapsed : 0
                if elapsed > 1.0 {
                    frameCount = 0
                    lastFrameTime = now
                }

                onResults?(ball, pose, court, currentFps)
            }
        }
    }

    func reset() {
        inferenceQueue.async { [self] in
            frameProcessor.reset()
        }
    }
}

@MainActor
final class AnalysisViewModel: ObservableObject {

    @Published var ballResult: BallDetectionResult?
    @Published var poseResult: PoseResult?
    @Published var courtResult: CourtDetectionResult?
    @Published var ballTrajectory: [CGPoint] = []
    @Published var fps: Double = 0
    @Published var isProcessing = false
    @Published var inputMode: InputMode = .camera

    let cameraManager = CameraManager()
    private let engine = InferenceEngine()
    private var videoReader: VideoFileReader?
    private let maxTrajectoryLength = 30

    init() {
        engine.onResults = { [weak self] ball, pose, court, fps in
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.ballResult = ball
                self.poseResult = pose
                if let court { self.courtResult = court }
                self.fps = fps

                if let b = ball, b.isVisible {
                    let pt = CGPoint(x: CGFloat(b.x), y: CGFloat(b.y))
                    self.ballTrajectory.append(pt)
                    if self.ballTrajectory.count > self.maxTrajectoryLength {
                        self.ballTrajectory.removeFirst()
                    }
                }
            }
        }

        let frameHandler: @Sendable (CMSampleBuffer) -> Void = { [weak engine] sampleBuffer in
            engine?.processFrame(sampleBuffer)
        }
        cameraManager.onFrame = frameHandler
    }

    func startCamera() async {
        inputMode = .camera
        stopVideoPlayback()
        engine.reset()
        ballTrajectory = []
        await cameraManager.requestPermission()
        cameraManager.start()
        isProcessing = true
    }

    func startVideoPlayback(url: URL) async {
        inputMode = .videoFile(url)
        cameraManager.stop()
        engine.reset()
        ballTrajectory = []

        do {
            let reader = try await VideoFileReader(url: url)
            self.videoReader = reader
            reader.onFrame = { [weak engine] sampleBuffer in
                engine?.processFrame(sampleBuffer)
            }
            reader.onFinished = { [weak self] in
                DispatchQueue.main.async {
                    self?.isProcessing = false
                }
            }
            isProcessing = true
            reader.start()
        } catch {
            print("Failed to open video: \(error)")
        }
    }

    func stopVideoPlayback() {
        videoReader?.stop()
        videoReader = nil
    }

    func stopSession() {
        cameraManager.stop()
        stopVideoPlayback()
        isProcessing = false
    }
}
