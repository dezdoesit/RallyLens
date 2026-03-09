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

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try BallTracker(configuration: config)
    }

    func predict(frames: [Float32]) throws -> BallDetectionResult {
        let shape: [NSNumber] = [1, 9, 360, 640]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)

        let count = min(frames.count, 9 * 360 * 640)
        frames.withUnsafeBufferPointer { ptr in
            mlArray.dataPointer.assumingMemoryBound(to: Float32.self)
                .initialize(from: ptr.baseAddress!, count: count)
        }

        let input = BallTrackerInput(input_frames: mlArray)
        let output = try model.prediction(input: input)

        // Output shape: (1, 256, 360, 640)
        // Take argmax along channel dimension to get per-pixel heatmap value
        return parseOutput(output.heatmap)
    }

    private func parseOutput(_ output: MLMultiArray) -> BallDetectionResult {
        let h = heatmapH
        let w = heatmapW
        let c = numClasses
        let ptr = output.dataPointer.assumingMemoryBound(to: Float32.self)

        // For each pixel, find the argmax across 256 channels
        // Then find the pixel with the highest argmax value (= brightest heatmap)
        var maxHeatVal: Int = 0
        var bestX = 0
        var bestY = 0
        var bestConfidence: Float = 0

        for y in 0..<h {
            for x in 0..<w {
                var pixMax: Float = -Float.greatestFiniteMagnitude
                var pixArgmax: Int = 0

                for ch in 0..<c {
                    // Layout: [1, 256, 360, 640] → index = ch * h * w + y * w + x
                    let idx = ch * h * w + y * w + x
                    let val = ptr[idx]
                    if val > pixMax {
                        pixMax = val
                        pixArgmax = ch
                    }
                }

                if pixArgmax > maxHeatVal {
                    maxHeatVal = pixArgmax
                    bestX = x
                    bestY = y
                    bestConfidence = pixMax
                }
            }
        }

        let threshold = 50
        return BallDetectionResult(
            x: Float(bestX) / Float(w),
            y: Float(bestY) / Float(h),
            confidence: bestConfidence,
            isVisible: maxHeatVal > threshold
        )
    }
}
