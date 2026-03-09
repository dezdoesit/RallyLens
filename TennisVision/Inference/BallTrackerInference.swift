import CoreML
import Accelerate

struct BallDetectionResult: Sendable {
    let x: Float          // normalized [0,1]
    let y: Float          // normalized [0,1]
    let confidence: Float // heatmap value 0-255
    let isVisible: Bool
}

final class BallTrackerInference: @unchecked Sendable {

    private let model: BallTracker
    private let heatmapW = 640
    private let heatmapH = 360

    // Reuse input MLMultiArray to avoid per-frame allocation
    private let inputArray: MLMultiArray

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try BallTracker(configuration: config)
        self.inputArray = try MLMultiArray(shape: [1, 9, 360, 640], dataType: .float32)
    }

    func predict(frames: [Float32]) throws -> BallDetectionResult {
        let count = min(frames.count, 9 * 360 * 640)
        let dest = inputArray.dataPointer.assumingMemoryBound(to: Float32.self)
        frames.withUnsafeBufferPointer { src in
            dest.update(from: src.baseAddress!, count: count)
        }

        let input = BallTrackerInput(input_frames: inputArray)
        let output = try model.prediction(input: input)

        // Output shape: (1, 1, 360, 640) — single-channel heatmap, values 0-255
        return parseHeatmap(output.heatmap)
    }

    private func parseHeatmap(_ heatmap: MLMultiArray) -> BallDetectionResult {
        let count = heatmapW * heatmapH
        let ptr = heatmap.dataPointer.assumingMemoryBound(to: Float32.self)

        // Find peak using vDSP
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(count))

        let y = Int(maxIdx) / heatmapW
        let x = Int(maxIdx) % heatmapW

        // Ball is visible if heatmap peak > ~50 (out of 255)
        return BallDetectionResult(
            x: Float(x) / Float(heatmapW),
            y: Float(y) / Float(heatmapH),
            confidence: maxVal,
            isVisible: maxVal > 50.0
        )
    }
}
