import CoreML
import Accelerate

struct CourtDetectionResult: Sendable {
    /// 14 keypoints in normalized coordinates [0,1]
    let keypoints: [CGPoint]
    let isDetected: Bool
}

/// Detects 14 tennis court keypoints from heatmap output.
/// Model output: (1, 14, 360, 640) — one heatmap per keypoint, sigmoid applied.
/// For each heatmap channel, find the peak to get the keypoint location.
final class CourtDetectorInference: @unchecked Sendable {

    private let model: CourtDetector
    private let heatmapW = 640
    private let heatmapH = 360
    private let numKeypoints = 14

    // Reuse input array
    private let inputArray: MLMultiArray

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try CourtDetector(configuration: config)
        self.inputArray = try MLMultiArray(shape: [1, 3, 360, 640], dataType: .float32)
    }

    /// Takes the first 3 channels (single frame RGB) from the 9-channel stacked frames
    func predict(singleFrameData: [Float32]) throws -> CourtDetectionResult {
        let count = min(singleFrameData.count, 3 * 360 * 640)
        let dest = inputArray.dataPointer.assumingMemoryBound(to: Float32.self)
        singleFrameData.withUnsafeBufferPointer { src in
            dest.update(from: src.baseAddress!, count: count)
        }

        let input = CourtDetectorInput(input_frame: inputArray)
        let output = try model.prediction(input: input)

        // Output shape: (1, 14, 360, 640)
        return parseHeatmaps(output.keypoint_heatmaps)
    }

    private func parseHeatmaps(_ heatmaps: MLMultiArray) -> CourtDetectionResult {
        let planeSize = heatmapW * heatmapH
        let ptr = heatmaps.dataPointer.assumingMemoryBound(to: Float32.self)

        var keypoints: [CGPoint] = []
        var detectedCount = 0
        let confidenceThreshold: Float = 0.5

        for kp in 0..<numKeypoints {
            let channelPtr = ptr.advanced(by: kp * planeSize)

            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(channelPtr, 1, &maxVal, &maxIdx, vDSP_Length(planeSize))

            let y = Int(maxIdx) / heatmapW
            let x = Int(maxIdx) % heatmapW

            if maxVal > confidenceThreshold {
                keypoints.append(CGPoint(
                    x: CGFloat(x) / CGFloat(heatmapW),
                    y: CGFloat(y) / CGFloat(heatmapH)
                ))
                detectedCount += 1
            } else {
                // Below threshold — still include position but mark unreliable
                keypoints.append(CGPoint(
                    x: CGFloat(x) / CGFloat(heatmapW),
                    y: CGFloat(y) / CGFloat(heatmapH)
                ))
            }
        }

        // Consider court detected if at least 8 out of 14 keypoints are confident
        return CourtDetectionResult(
            keypoints: keypoints,
            isDetected: detectedCount >= 8
        )
    }
}
