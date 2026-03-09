import AVFoundation
import CoreVideo

final class VideoFileReader: @unchecked Sendable {

    private let asset: AVAsset
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

        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoReaderError.noVideoTrack
        }

        let size = try await videoTrack.load(.naturalSize)
        let transform = try await videoTrack.load(.preferredTransform)
        let transformedSize = size.applying(transform)
        self.videoSize = CGSize(
            width: abs(transformedSize.width),
            height: abs(transformedSize.height)
        )
        self.frameRate = try await Float(videoTrack.load(.nominalFrameRate))
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

        let semaphore = DispatchSemaphore(value: 0)
        var tracks: [AVAssetTrack] = []

        Task {
            tracks = (try? await asset.loadTracks(withMediaType: .video)) ?? []
            semaphore.signal()
        }
        semaphore.wait()

        guard let videoTrack = tracks.first else { return }

        let trackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        trackOutput.alwaysCopiesSampleData = false

        guard reader.canAdd(trackOutput) else { return }
        reader.add(trackOutput)
        reader.startReading()

        let frameDuration = TimeInterval(1.0 / max(frameRate, 1.0))

        while isReading && reader.status == .reading {
            guard let sampleBuffer = trackOutput.copyNextSampleBuffer() else { break }

            onFrame?(sampleBuffer)

            // Pace to approximate real-time playback
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
