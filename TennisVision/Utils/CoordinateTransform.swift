import CoreGraphics

enum CoordinateTransform {
    /// Convert model-space normalized coordinates to screen-space points
    static func modelToScreen(
        point: CGPoint,
        screenSize: CGSize
    ) -> CGPoint {
        CGPoint(
            x: point.x * screenSize.width,
            y: point.y * screenSize.height
        )
    }

    /// Convert screen-space point to model-space normalized coordinates
    static func screenToModel(
        point: CGPoint,
        screenSize: CGSize
    ) -> CGPoint {
        CGPoint(
            x: point.x / screenSize.width,
            y: point.y / screenSize.height
        )
    }
}
