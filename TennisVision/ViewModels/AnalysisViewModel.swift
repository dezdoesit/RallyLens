import SwiftUI
import Combine
import AVFoundation

enum InputMode: Sendable {
    case camera
    case videoFile(URL)
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

    private let frameProcessor = FrameProcessor()
    private var ballTracker: BallTrackerInference?
    private let poseEstimator = PoseEstimationInference()
    private let courtDetector = CourtDetectorInference()

    private let inferenceQueue = DispatchQueue(label: "com.tennisvision.inference", qos: .userInitiated)

    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount = 0
    private let maxTrajectoryLength = 30

    private var videoReader: VideoFileReader?
    private var courtFrameCounter = 0

    init() {
        do {
            ballTracker = try BallTrackerInference()
        } catch {
            print("Failed to load BallTracker model: \(error)")
        }

        cameraManager.onFrame = { [weak self] sampleBuffer in
            self?.processFrame(sampleBuffer)
        }
    }

    func startCamera() async {
        inputMode = .camera
        stopVideoPlayback()
        frameProcessor.reset()
        await cameraManager.requestPermission()
        cameraManager.start()
        isProcessing = true
    }

    func startVideoPlayback(url: URL) async {
        inputMode = .videoFile(url)
        cameraManager.stop()
        frameProcessor.reset()
        ballTrajectory = []

        do {
            let reader = try await VideoFileReader(url: url)
            self.videoReader = reader
            reader.onFrame = { [weak self] sampleBuffer in
                self?.processFrame(sampleBuffer)
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

    private nonisolated func processFrame(_ sampleBuffer: CMSampleBuffer) {
        let frameProcessor = self.frameProcessor
        let ballTracker = self.ballTracker
        let poseEstimator = self.poseEstimator
        let courtDetector = self.courtDetector

        self.inferenceQueue.async { [weak self] in
            guard let self else { return }
            guard let (pixelBuffer, frames) = frameProcessor.process(sampleBuffer) else { return }

            let ball = try? ballTracker?.predict(frames: frames)
            let pose = try? poseEstimator.predict(pixelBuffer: pixelBuffer)

            self.courtFrameCounter += 1
            var court: CourtDetectionResult? = nil
            if self.courtFrameCounter % 10 == 0 {
                court = courtDetector.predict(frames: frames)
            }

            let now = CACurrentMediaTime()
            self.frameCount += 1
            let elapsed = now - self.lastFrameTime
            let currentFps = elapsed > 0 ? Double(self.frameCount) / elapsed : 0
            if elapsed > 1.0 {
                self.frameCount = 0
                self.lastFrameTime = now
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.ballResult = ball
                self.poseResult = pose
                if let court { self.courtResult = court }
                self.fps = currentFps

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
