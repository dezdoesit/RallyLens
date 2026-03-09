import CoreVideo
import CoreImage
import CoreMedia

final class FrameProcessor: @unchecked Sendable {

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let targetWidth = 640
    private let targetHeight = 360

    private var frameBuffer: [CVPixelBuffer] = []
    private let bufferCapacity = 3

    /// Process a sample buffer and return (resized pixelBuffer, 9×360×640 float array)
    /// Returns nil until 3 frames have been collected.
    func process(_ sampleBuffer: CMSampleBuffer) -> (CVPixelBuffer, [Float32])? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        guard let resized = resize(pixelBuffer) else { return nil }

        frameBuffer.append(resized)
        if frameBuffer.count > bufferCapacity {
            frameBuffer.removeFirst()
        }
        guard frameBuffer.count == bufferCapacity else { return nil }

        let floatArray = stackFrames(frameBuffer)
        return (resized, floatArray)
    }

    func reset() {
        frameBuffer.removeAll()
    }

    private func resize(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let srcWidth = CGFloat(CVPixelBufferGetWidth(buffer))
        let srcHeight = CGFloat(CVPixelBufferGetHeight(buffer))
        let scaleX = CGFloat(targetWidth) / srcWidth
        let scaleY = CGFloat(targetHeight) / srcHeight
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var output: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        CVPixelBufferCreate(
            nil,
            targetWidth, targetHeight,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &output
        )
        guard let out = output else { return nil }
        ciContext.render(scaled, to: out)
        return out
    }

    private func stackFrames(_ buffers: [CVPixelBuffer]) -> [Float32] {
        let w = targetWidth
        let h = targetHeight
        var result = [Float32](repeating: 0, count: 9 * h * w)

        for (frameIdx, buffer) in buffers.enumerated() {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

            guard let base = CVPixelBufferGetBaseAddress(buffer) else { continue }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let ptr = base.assumingMemoryBound(to: UInt8.self)

            // BGRA → RGB, normalized to [0,1]
            for y in 0..<h {
                for x in 0..<w {
                    let offset = y * bytesPerRow + x * 4
                    let r = Float32(ptr[offset + 2]) / 255.0
                    let g = Float32(ptr[offset + 1]) / 255.0
                    let b = Float32(ptr[offset])     / 255.0
                    result[(frameIdx * 3 + 0) * h * w + y * w + x] = r
                    result[(frameIdx * 3 + 1) * h * w + y * w + x] = g
                    result[(frameIdx * 3 + 2) * h * w + y * w + x] = b
                }
            }
        }
        return result
    }
}
