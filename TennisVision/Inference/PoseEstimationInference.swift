import Vision
import CoreGraphics

struct PoseResult: Sendable {
    struct Joint: Sendable {
        let name: VNHumanBodyPoseObservation.JointName
        let position: CGPoint
        let confidence: Float
    }
    let joints: [Joint]
    let isDetected: Bool
}

final class PoseEstimationInference: @unchecked Sendable {

    private let targetJoints: [VNHumanBodyPoseObservation.JointName] = [
        .rightWrist, .leftWrist,
        .rightElbow, .leftElbow,
        .rightShoulder, .leftShoulder,
        .rightHip, .leftHip,
        .rightKnee, .leftKnee,
        .rightAnkle, .leftAnkle
    ]

    func predict(pixelBuffer: CVPixelBuffer) throws -> PoseResult {
        let request = VNDetectHumanBodyPoseRequest()
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
                position: CGPoint(x: point.location.x, y: 1 - point.location.y),
                confidence: Float(point.confidence)
            )
        }

        return PoseResult(joints: joints, isDetected: !joints.isEmpty)
    }
}
