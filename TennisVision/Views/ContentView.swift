import SwiftUI
import PhotosUI
import AVKit

struct ContentView: View {

    @StateObject private var vm = AnalysisViewModel()
    @State private var showVideoPicker = false
    @State private var selectedVideoItem: PhotosPickerItem?

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.ignoresSafeArea()

                // Camera preview (only in camera mode)
                if case .camera = vm.inputMode {
                    CameraPreviewView(previewLayer: vm.cameraManager.previewLayer)
                        .ignoresSafeArea()
                }

                // Video player (only in video mode)
                if case .videoFile = vm.inputMode, let player = vm.videoPlayer {
                    VideoPlayerView(player: player)
                        .ignoresSafeArea()
                }

                // Analysis overlay on top of video/camera
                OverlayView(
                    ballResult: vm.ballResult,
                    poseResult: vm.poseResult,
                    courtResult: vm.courtResult,
                    trajectory: vm.ballTrajectory
                )
                .ignoresSafeArea()

                // HUD + Controls
                VStack {
                    // Top: Stats
                    HStack {
                        Spacer()
                        StatsOverlayView(
                            fps: vm.fps,
                            ballVisible: vm.ballResult?.isVisible ?? false,
                            poseDetected: vm.poseResult?.isDetected ?? false
                        )
                        .padding()
                    }

                    Spacer()

                    // Bottom: Controls
                    HStack(spacing: 20) {
                        Button {
                            Task { await vm.startCamera() }
                        } label: {
                            Label("Camera", systemImage: "camera.fill")
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(.ultraThinMaterial)
                                .clipShape(Capsule())
                        }

                        Button {
                            showVideoPicker = true
                        } label: {
                            Label("Video", systemImage: "film.fill")
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(.ultraThinMaterial)
                                .clipShape(Capsule())
                        }

                        if vm.isProcessing {
                            Button {
                                vm.stopSession()
                            } label: {
                                Label("Stop", systemImage: "stop.fill")
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 10)
                                    .background(.red.opacity(0.8))
                                    .clipShape(Capsule())
                            }
                        }
                    }
                    .font(.callout.bold())
                    .foregroundStyle(.white)
                    .padding(.bottom, 30)
                }
            }
        }
        .photosPicker(isPresented: $showVideoPicker, selection: $selectedVideoItem, matching: .videos)
        .onChange(of: selectedVideoItem) { _, newValue in
            guard let item = newValue else { return }
            Task {
                await loadVideo(from: item)
            }
        }
        .task {
            await vm.startCamera()
        }
        .onDisappear {
            vm.stopSession()
        }
    }

    private func loadVideo(from item: PhotosPickerItem) async {
        guard let movie = try? await item.loadTransferable(type: VideoTransferable.self) else { return }
        await vm.startVideoPlayback(url: movie.url)
    }
}

struct VideoTransferable: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let tempDir = FileManager.default.temporaryDirectory
            let filename = "tennis_video_\(UUID().uuidString).mp4"
            let dest = tempDir.appendingPathComponent(filename)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return Self(url: dest)
        }
    }
}
