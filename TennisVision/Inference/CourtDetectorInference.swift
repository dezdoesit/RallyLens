import CoreGraphics

struct CourtDetectionResult: Sendable {
    let keypoints: [CGPoint]
    let isDetected: Bool
}

/// Mock court detector — returns empty results until a real model is integrated.
final class CourtDetectorInference: Sendable {

    func predict(frames: [Float32]) -> CourtDetectionResult {
        return CourtDetectionResult(keypoints: [], isDetected: false)
    }
}
