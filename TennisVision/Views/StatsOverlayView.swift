import SwiftUI

struct StatsOverlayView: View {
    let fps: Double
    let ballVisible: Bool
    let poseDetected: Bool

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            Label(String(format: "%.0f FPS", fps), systemImage: "speedometer")
            Label(ballVisible ? "Ball" : "Ball –", systemImage: "circle.fill")
                .foregroundStyle(ballVisible ? .yellow : .gray)
            Label(poseDetected ? "Pose" : "Pose –", systemImage: "figure.tennis")
                .foregroundStyle(poseDetected ? .green : .gray)
        }
        .font(.caption.monospaced())
        .foregroundStyle(.white)
        .padding(8)
        .background(.black.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
