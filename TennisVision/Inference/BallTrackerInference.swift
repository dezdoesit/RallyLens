import CoreML
import Accelerate

struct BallDetectionResult: Sendable {
    let x: Float          // normalized [0,1]
    let y: Float          // normalized [0,1]
    let confidence: Float
    let isVisible: Bool
}

final class BallTrackerInference: @unchecked Sendable {

    private let model: BallTracker
    private let heatmapW = 640
    private let heatmapH = 360
    private let numClasses = 256

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

        // Output shape: (1, 256, 360, 640)
        return parseOutputFast(output.heatmap)
    }

    /// Efficient heatmap parsing: scan channels from 255 down, use vDSP to find
    /// the peak per channel. The first channel with a confident peak is the ball.
    private func parseOutputFast(_ output: MLMultiArray) -> BallDetectionResult {
        let h = heatmapH
        let w = heatmapW
        let planeSize = h * w  // 230400
        let ptr = output.dataPointer.assumingMemoryBound(to: Float32.self)

        var bestChannel = 0
        var bestVal: Float = 0
        var bestIdx: vDSP_Length = 0

        // Scan from highest channel (brightest) downward
        // The ball pixel will have high probability in a high channel
        for c in stride(from: numClasses - 1, through: 0, by: -1) {
            let channelPtr = ptr.advanced(by: c * planeSize)

            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(channelPtr, 1, &maxVal, &maxIdx, vDSP_Length(planeSize))

            if maxVal > bestVal {
                bestVal = maxVal
                bestIdx = maxIdx
                bestChannel = c
            }

            // Early exit: if we found a strong peak in a high channel, that's the ball
            if bestChannel > 100 && bestVal > 0.5 {
                break
            }
        }

        let y = Int(bestIdx) / w
        let x = Int(bestIdx) % w

        return BallDetectionResult(
            x: Float(x) / Float(w),
            y: Float(y) / Float(h),
            confidence: bestVal,
            isVisible: bestChannel > 50 && bestVal > 0.3
        )
    }
}
