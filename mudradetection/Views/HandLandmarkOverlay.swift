import SwiftUI

/// Draws hand skeleton lines and joint landmark dots directly over the live camera preview.
/// Transforms Vision coordinates (origin at bottom-left) to View coordinates (origin at top-left).
public struct HandLandmarkOverlay: View {
    public let hands: [DetectedHand]
    public let isMirrored: Bool

    public init(hands: [DetectedHand], isMirrored: Bool = false) {
        self.hands = hands
        self.isMirrored = isMirrored
    }

    public var body: some View {
        GeometryReader { geometry in
            let size = geometry.size

            Canvas { context, _ in
                for hand in hands {
                    // 1. Draw Finger Bone Connections
                    for (startKey, endKey) in DetectedHand.fingerConnections {
                        if let startLM = hand.landmarks[startKey],
                           let endLM = hand.landmarks[endKey] {

                            let startPt = convertPoint(startLM.point, in: size)
                            let endPt = convertPoint(endLM.point, in: size)

                            var path = Path()
                            path.move(to: startPt)
                            path.addLine(to: endPt)

                            // Muted Olive Green translucent skeleton connection
                            let boneColor = Color(red: 0.36, green: 0.45, blue: 0.28).opacity(0.85)
                            context.stroke(path, with: .color(boneColor), style: StrokeStyle(lineWidth: 3.5, lineCap: .round, lineJoin: .round))
                        }
                    }

                    // 2. Draw Landmark Joint Nodes
                    for (_, landmark) in hand.landmarks {
                        let pt = convertPoint(landmark.point, in: size)
                        let isFingertip = [HandLandmarkKey.thumbTip, .indexTip, .middleTip, .ringTip, .littleTip].contains(landmark.key)
                        let isWrist = landmark.key == .wrist

                        let radius: CGFloat = isFingertip ? 6.5 : (isWrist ? 7.0 : 4.5)

                        // Outer glow / halo
                        let haloRect = CGRect(x: pt.x - radius - 2, y: pt.y - radius - 2, width: (radius + 2) * 2, height: (radius + 2) * 2)
                        context.fill(Path(ellipseIn: haloRect), with: .color(Color.black.opacity(0.25)))

                        // Inner node fill
                        let nodeRect = CGRect(x: pt.x - radius, y: pt.y - radius, width: radius * 2, height: radius * 2)
                        let nodeColor: Color
                        if isFingertip {
                            // Gold / Ochre accent for fingertips
                            nodeColor = Color(red: 0.88, green: 0.65, blue: 0.25)
                        } else if isWrist {
                            // Terracotta for wrist base
                            nodeColor = Color(red: 0.72, green: 0.35, blue: 0.22)
                        } else {
                            // Ivory / Pearlescent for knuckles
                            nodeColor = Color(red: 0.98, green: 0.96, blue: 0.92)
                        }

                        context.fill(Path(ellipseIn: nodeRect), with: .color(nodeColor))

                        // Border stroke
                        context.stroke(Path(ellipseIn: nodeRect), with: .color(Color(red: 0.17, green: 0.12, blue: 0.09)), lineWidth: 1.5)
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }

    /// Vision normalized coordinates: X is [0, 1] left-to-right, Y is [0, 1] bottom-to-top.
    /// Screen coordinates: X is [0, width] left-to-right, Y is [0, height] top-to-bottom.
    private func convertPoint(_ visionPoint: CGPoint, in size: CGSize) -> CGPoint {
        let x = isMirrored ? (1.0 - visionPoint.x) * size.width : visionPoint.x * size.width
        let y = (1.0 - visionPoint.y) * size.height
        return CGPoint(x: x, y: y)
    }
}

