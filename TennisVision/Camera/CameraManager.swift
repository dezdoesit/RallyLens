import AVFoundation
import CoreVideo

@MainActor
final class CameraManager: NSObject, ObservableObject {

    @Published var isRunning = false
    @Published var permissionGranted = false

    var onFrame: (@Sendable (CMSampleBuffer) -> Void)?

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.tennisvision.camera", qos: .userInitiated)

    let previewLayer: AVCaptureVideoPreviewLayer

    override init() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        super.init()
    }

    func requestPermission() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            permissionGranted = true
            setupSession()
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            permissionGranted = granted
            if granted { setupSession() }
        default:
            permissionGranted = false
        }
    }

    private func setupSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .hd1280x720

            guard let device = AVCaptureDevice.default(
                .builtInWideAngleCamera, for: .video, position: .back
            ),
            let input = try? AVCaptureDeviceInput(device: device),
            self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }
            self.session.addInput(input)

            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.setSampleBufferDelegate(self, queue: self.sessionQueue)

            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
            }

            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }

            if let deviceInput = self.session.inputs.first as? AVCaptureDeviceInput {
                try? deviceInput.device.lockForConfiguration()
                deviceInput.device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                deviceInput.device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
                deviceInput.device.unlockForConfiguration()
            }

            self.session.commitConfiguration()
        }
    }

    func start() {
        sessionQueue.async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async { self?.isRunning = true }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async { self?.isRunning = false }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        onFrame?(sampleBuffer)
    }
}
