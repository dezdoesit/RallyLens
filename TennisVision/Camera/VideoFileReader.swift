import AVFoundation
import CoreVideo

final class VideoFileReader: @unchecked Sendable {

    private let asset: AVAsset
    private let videoTrack: AVAssetTrack
    private let readQueue = DispatchQueue(label: "com.tennisvision.videoreader", qos: .userInitiated)
    private var isReading = false

    var onFrame: (@Sendable (CMSampleBuffer) -> Void)?
    var onFinished: (@Sendable () -> Void)?

    let duration: CMTime
    let frameRate: Float
    let videoSize: CGSize

    init(url: URL) async throws {
        self.asset = AVAsset(url: url)
        self.duration = try await asset.load(.duration)

        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoReaderError.noVideoTrack
        }
        self.videoTrack = track

        let size = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let transformedSize = size.applying(transform)
        self.videoSize = CGSize(
            width: abs(transformedSize.width),
            height: abs(transformedSize.height)
        )
        self.frameRate = try await Float(track.load(.nominalFrameRate))
    }

    func start() {
        isReading = true
        readQueue.async { [weak self] in
            self?.readFrames()
        }
    }

    func stop() {
        isReading = false
    }

    private func readFrames() {
        guard let reader = try? AVAssetReader(asset: asset) else { return }

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        let trackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        trackOutput.alwaysCopiesSampleData = false

        guard reader.canAdd(trackOutput) else { return }
        reader.add(trackOutput)
        reader.startReading()

        let frameDuration = TimeInterval(1.0 / max(frameRate, 1.0))

        while isReading && reader.status == .reading {
            guard let sampleBuffer = trackOutput.copyNextSampleBuffer() else { break }
            onFrame?(sampleBuffer)
            Thread.sleep(forTimeInterval: frameDuration)
        }

        reader.cancelReading()
        isReading = false
        onFinished?()
    }

    enum VideoReaderError: Error {
        case noVideoTrack
    }
}
