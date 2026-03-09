import SwiftUI
import Vision

struct OverlayView: View {

    let ballResult: BallDetectionResult?
    let poseResult: PoseResult?
    let courtResult: CourtDetectionResult?
    let trajectory: [CGPoint]

    var body: some View {
        Canvas { context, size in
            // Court lines
            if let court = courtResult, court.isDetected {
                drawCourt(context: context, size: size, keypoints: court.keypoints)
            }

            // Skeleton
            if let pose = poseResult, pose.isDetected {
                drawSkeleton(context: context, size: size, joints: pose.joints)
            }

            // Trajectory
            drawTrajectory(context: context, size: size, points: trajectory)

            // Ball position
            if let ball = ballResult, ball.isVisible {
                let pt = CGPoint(
                    x: CGFloat(ball.x) * size.width,
                    y: CGFloat(ball.y) * size.height
                )
                let circle = Path(ellipseIn: CGRect(
                    x: pt.x - 10, y: pt.y - 10,
                    width: 20, height: 20
                ))
                context.fill(circle, with: .color(.yellow))
                context.stroke(circle, with: .color(.white), lineWidth: 2)
            }
        }
    }

    private func drawCourt(context: GraphicsContext, size: CGSize, keypoints: [CGPoint]) {
        guard keypoints.count == 14 else { return }
        let courtLines: [(Int, Int)] = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (8, 9), (10, 11),
        ]
        var path = Path()
        for (a, b) in courtLines {
            guard a < keypoints.count, b < keypoints.count else { continue }
            let ptA = CGPoint(x: keypoints[a].x * size.width, y: keypoints[a].y * size.height)
            let ptB = CGPoint(x: keypoints[b].x * size.width, y: keypoints[b].y * size.height)
            path.move(to: ptA)
            path.addLine(to: ptB)
        }
        context.stroke(path, with: .color(.cyan.opacity(0.7)), lineWidth: 1.5)
    }

    private func drawSkeleton(context: GraphicsContext, size: CGSize, joints: [PoseResult.Joint]) {
        let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
            (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
            (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
            (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
            (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
            (.rightShoulder, .leftShoulder),
            (.rightHip, .leftHip),
        ]
        let jointMap = Dictionary(uniqueKeysWithValues: joints.map { ($0.name, $0.position) })

        var bonePath = Path()
        for (a, b) in connections {
            guard let pa = jointMap[a], let pb = jointMap[b] else { continue }
            bonePath.move(to: CGPoint(x: pa.x * size.width, y: pa.y * size.height))
            bonePath.addLine(to: CGPoint(x: pb.x * size.width, y: pb.y * size.height))
        }
        context.stroke(bonePath, with: .color(.green.opacity(0.8)), lineWidth: 2)

        for joint in joints {
            let pt = CGPoint(x: joint.position.x * size.width, y: joint.position.y * size.height)
            let dot = Path(ellipseIn: CGRect(x: pt.x - 4, y: pt.y - 4, width: 8, height: 8))
            context.fill(dot, with: .color(.green))
        }
    }

    private func drawTrajectory(context: GraphicsContext, size: CGSize, points: [CGPoint]) {
        guard points.count > 1 else { return }
        var path = Path()
        for (i, pt) in points.enumerated() {
            let screenPt = CGPoint(x: pt.x * size.width, y: pt.y * size.height)
            if i == 0 { path.move(to: screenPt) }
            else { path.addLine(to: screenPt) }
        }
        context.stroke(
            path,
            with: .color(.yellow.opacity(0.6)),
            style: StrokeStyle(lineWidth: 2, lineCap: .round)
        )
    }
}
