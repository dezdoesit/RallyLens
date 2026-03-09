import CoreML
import CoreImage
import CoreVideo
import Accelerate

struct CourtDetectionResult: Sendable {
    /// 14 keypoints in normalized coordinates [0,1]
    let keypoints: [CGPoint]
    let isDetected: Bool
}

/// Detects 14 tennis court keypoints from heatmap output.
/// Model input: (1, 3, 360, 640). Output: (1, 14, 360, 640).
/// Handles its own resize from any input pixelBuffer to 360×640.
final class CourtDetectorInference: @unchecked Sendable {

    private let model: CourtDetector
    private let heatmapW = 640
    private let heatmapH = 360
    private let numKeypoints = 14

    // Reuse buffers
    private let inputArray: MLMultiArray
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try CourtDetector(configuration: config)
        self.inputArray = try MLMultiArray(shape: [1, 3, 360, 640], dataType: .float32)
    }

    /// Takes any pixelBuffer, resizes to 360×640, converts to float, and predicts.
    func predict(pixelBuffer: CVPixelBuffer) throws -> CourtDetectionResult {
        guard let resized = resize(pixelBuffer) else {
            return CourtDetectionResult(keypoints: [], isDetected: false)
        }

        // Convert pixelBuffer to float array [3, 360, 640]
        let dest = inputArray.dataPointer.assumingMemoryBound(to: Float32.self)
        let w = heatmapW
        let h = heatmapH

        CVPixelBufferLockBaseAddress(resized, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(resized, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(resized) else {
            return CourtDetectionResult(keypoints: [], isDetected: false)
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(resized)
        let ptr = base.assumingMemoryBound(to: UInt8.self)

        for y in 0..<h {
            for x in 0..<w {
                let offset = y * bytesPerRow + x * 4
                let r = Float32(ptr[offset + 2]) / 255.0
                let g = Float32(ptr[offset + 1]) / 255.0
                let b = Float32(ptr[offset])     / 255.0
                dest[0 * h * w + y * w + x] = r
                dest[1 * h * w + y * w + x] = g
                dest[2 * h * w + y * w + x] = b
            }
        }

        let input = CourtDetectorInput(input_frame: inputArray)
        let output = try model.prediction(input: input)

        return parseHeatmaps(output.keypoint_heatmaps)
    }

    private func resize(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let srcWidth = CGFloat(CVPixelBufferGetWidth(buffer))
        let srcHeight = CGFloat(CVPixelBufferGetHeight(buffer))
        let scaleX = CGFloat(heatmapW) / srcWidth
        let scaleY = CGFloat(heatmapH) / srcHeight
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var output: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        CVPixelBufferCreate(nil, heatmapW, heatmapH, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &output)
        guard let out = output else { return nil }
        ciContext.render(scaled, to: out)
        return out
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

            keypoints.append(CGPoint(
                x: CGFloat(x) / CGFloat(heatmapW),
                y: CGFloat(y) / CGFloat(heatmapH)
            ))

            if maxVal > confidenceThreshold {
                detectedCount += 1
            }
        }

        return CourtDetectionResult(
            keypoints: keypoints,
            isDetected: detectedCount >= 8
        )
    }
}
