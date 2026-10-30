import Foundation
import CoreGraphics

/// Hand landmark geometric feature extractor.
/// Computes exactly 77 scale-invariant, translation-invariant features from 21 hand landmarks.
/// The calculations and ordering are mathematically identical to MLTraining/scripts/feature_extractor.py.
public final class HandFeatureExtractor: Sendable {
    public static let expectedFeatureCount = 77
    public static let landmarkCount = 21

    public init() {}

    /// Extracts the 77-element feature vector from 21 canonical landmark CGPoints
    /// Points must be ordered 0..20:
    /// 0: wrist, 1..4: thumb (CMC, MP, IP, Tip)
    /// 5..8: index (MCP, PIP, DIP, Tip), 9..12: middle (MCP, PIP, DIP, Tip)
    /// 13..16: ring (MCP, PIP, DIP, Tip), 17..20: little (MCP, PIP, DIP, Tip)
    public func extractFeatures(from points: [CGPoint]) -> [Float]? {
        guard points.count == Self.landmarkCount else {
            return nil
        }

        var features = [Float]()
        features.reserveCapacity(Self.expectedFeatureCount)

        let wrist = points[0]
        let middleMCP = points[9]

        // 1. Hand scale normalization factor: Distance between wrist and middle MCP
        var scale = computeDistance(wrist, middleMCP)
        if scale < 1e-6 {
            scale = 1.0 // Prevent division by zero
        }

        // 2. Features 1..40: Wrist-relative normalized coordinates (landmarks 1 to 20)
        for i in 1..<Self.landmarkCount {
            let normX = Float((points[i].x - wrist.x) / scale)
            let normY = Float((points[i].y - wrist.y) / scale)
            features.append(normX)
            features.append(normY)
        }

        // 3. Features 41..55: 15 Finger joint angles (normalized by pi to [0, 1])
        let angleJoints: [(Int, Int, Int)] = [
            (0, 1, 2),   // thumb CMC
            (1, 2, 3),   // thumb MP
            (2, 3, 4),   // thumb IP
            (0, 5, 6),   // index MCP
            (5, 6, 7),   // index PIP
            (6, 7, 8),   // index DIP
            (0, 9, 10),  // middle MCP
            (9, 10, 11), // middle PIP
            (10, 11, 12),// middle DIP
            (0, 13, 14), // ring MCP
            (13, 14, 15),// ring PIP
            (14, 15, 16),// ring DIP
            (0, 17, 18), // little MCP
            (17, 18, 19),// little PIP
            (18, 19, 20) // little DIP
        ]
        for (i1, i2, i3) in angleJoints {
            let angle = computeAngle(points[i1], points[i2], points[i3])
            features.append(angle)
        }

        // 4. Features 56..65: 10 Pairwise fingertip normalized distances
        let fingertipPairs: [(Int, Int)] = [
            (4, 8),   // thumb - index
            (4, 12),  // thumb - middle
            (4, 16),  // thumb - ring (Mayura critical)
            (4, 20),  // thumb - little
            (8, 12),  // index - middle (Kartarimukha critical)
            (8, 16),  // index - ring
            (8, 20),  // index - little
            (12, 16), // middle - ring
            (12, 20), // middle - little
            (16, 20)  // ring - little (Ardhapataka critical)
        ]
        for (i1, i2) in fingertipPairs {
            let dist = Float(computeDistance(points[i1], points[i2]) / scale)
            features.append(dist)
        }

        // 5. Features 66..75: 10 Extension ratios (fingertip to wrist & fingertip to MCP/CMC)
        let extensionPairs: [(Int, Int)] = [
            (4, 0),  // thumbTip to wrist
            (8, 0),  // indexTip to wrist
            (12, 0), // middleTip to wrist
            (16, 0), // ringTip to wrist
            (20, 0), // littleTip to wrist
            (4, 1),  // thumbTip to thumbCMC
            (8, 5),  // indexTip to indexMCP
            (12, 9), // middleTip to middleMCP
            (16, 13),// ringTip to ringMCP
            (20, 17) // littleTip to littleMCP
        ]
        for (i1, i2) in extensionPairs {
            let dist = Float(computeDistance(points[i1], points[i2]) / scale)
            features.append(dist)
        }

        // 6. Features 76..77: 2 Palm proportions
        let proportionPairs: [(Int, Int)] = [
            (5, 17), // indexMCP to littleMCP (palm width)
            (0, 12)  // wrist to middleTip (hand length)
        ]
        for (i1, i2) in proportionPairs {
            let dist = Float(computeDistance(points[i1], points[i2]) / scale)
            features.append(dist)
        }

        assert(features.count == Self.expectedFeatureCount, "Extracted feature count mismatch")
        return features
    }

    // MARK: - Mathematical Helpers

    private func computeDistance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        let dx = p1.x - p2.x
        let dy = p1.y - p2.y
        return sqrt(dx * dx + dy * dy)
    }

    /// Computes angle at vertex p2 in radians normalized by pi (range: [0, 1]).
    /// Vector u = p1 - p2, Vector v = p3 - p2.
    private func computeAngle(_ p1: CGPoint, _ p2: CGPoint, _ p3: CGPoint) -> Float {
        let ux = Double(p1.x - p2.x)
        let uy = Double(p1.y - p2.y)
        let vx = Double(p3.x - p2.x)
        let vy = Double(p3.y - p2.y)

        let dot = ux * vx + uy * vy
        let lenU = sqrt(ux * ux + uy * uy)
        let lenV = sqrt(vx * vx + vy * vy)

        let denom = lenU * lenV
        if denom < 1e-7 {
            return 0.0
        }

        var cosVal = dot / denom
        cosVal = max(-1.0, min(1.0, cosVal))
        let angleRad = acos(cosVal)
        return Float(angleRad / .pi) // Normalized to [0.0, 1.0]
    }
}

