import Foundation
import CoreGraphics
import Vision

/// Represents the 21 standard hand landmarks detected by Apple's Vision framework
public enum HandLandmarkKey: Int, CaseIterable, Sendable {
    case wrist = 0
    // Thumb
    case thumbCMC = 1
    case thumbMP = 2
    case thumbIP = 3
    case thumbTip = 4
    // Index
    case indexMCP = 5
    case indexPIP = 6
    case indexDIP = 7
    case indexTip = 8
    // Middle
    case middleMCP = 9
    case middlePIP = 10
    case middleDIP = 11
    case middleTip = 12
    // Ring
    case ringMCP = 13
    case ringPIP = 14
    case ringDIP = 15
    case ringTip = 16
    // Little
    case littleMCP = 17
    case littlePIP = 18
    case littleDIP = 19
    case littleTip = 20

    public var visionJointName: VNHumanHandPoseObservation.JointName {
        switch self {
        case .wrist: return .wrist
        case .thumbCMC: return .thumbCMC
        case .thumbMP: return .thumbMP
        case .thumbIP: return .thumbIP
        case .thumbTip: return .thumbTip
        case .indexMCP: return .indexMCP
        case .indexPIP: return .indexPIP
        case .indexDIP: return .indexDIP
        case .indexTip: return .indexTip
        case .middleMCP: return .middleMCP
        case .middlePIP: return .middlePIP
        case .middleDIP: return .middleDIP
        case .middleTip: return .middleTip
        case .ringMCP: return .ringMCP
        case .ringPIP: return .ringPIP
        case .ringDIP: return .ringDIP
        case .ringTip: return .ringTip
        case .littleMCP: return .littleMCP
        case .littlePIP: return .littlePIP
        case .littleDIP: return .littleDIP
        case .littleTip: return .littleTip
        }
    }
}

/// A single detected 2D landmark with confidence score
public struct LandmarkPoint: Identifiable, Sendable {
    public let id: Int
    public let key: HandLandmarkKey
    public let point: CGPoint
    public let confidence: Float

    public init(key: HandLandmarkKey, point: CGPoint, confidence: Float) {
        self.id = key.rawValue
        self.key = key
        self.point = point
        self.confidence = confidence
    }
}

/// Represents a detected hand containing all 21 landmark points and metadata
public struct DetectedHand: Identifiable, Sendable {
    public let id = UUID()
    public let landmarks: [HandLandmarkKey: LandmarkPoint]
    public let chirality: VNChirality
    public let averageConfidence: Float
    public let timestamp: TimeInterval

    public var isComplete: Bool {
        landmarks.count == HandLandmarkKey.allCases.count
    }

    /// Sorted array of 21 landmark points in exact canonical order (0 to 20)
    public var orderedPoints: [CGPoint]? {
        var points: [CGPoint] = []
        for key in HandLandmarkKey.allCases {
            guard let lm = landmarks[key], lm.confidence >= 0.2 else {
                return nil // Missing or low confidence landmark
            }
            points.append(lm.point)
        }
        return points
    }

    /// Standard skeletal bone connections for drawing hand overlay
    public static let fingerConnections: [(HandLandmarkKey, HandLandmarkKey)] = [
        // Palm base
        (.wrist, .thumbCMC),
        (.wrist, .indexMCP),
        (.wrist, .middleMCP),
        (.wrist, .ringMCP),
        (.wrist, .littleMCP),
        // Palm knuckle arch
        (.indexMCP, .middleMCP),
        (.middleMCP, .ringMCP),
        (.ringMCP, .littleMCP),
        // Thumb
        (.thumbCMC, .thumbMP),
        (.thumbMP, .thumbIP),
        (.thumbIP, .thumbTip),
        // Index
        (.indexMCP, .indexPIP),
        (.indexPIP, .indexDIP),
        (.indexDIP, .indexTip),
        // Middle
        (.middleMCP, .middlePIP),
        (.middlePIP, .middleDIP),
        (.middleDIP, .middleTip),
        // Ring
        (.ringMCP, .ringPIP),
        (.ringPIP, .ringDIP),
        (.ringDIP, .ringTip),
        // Little
        (.littleMCP, .littlePIP),
        (.littlePIP, .littleDIP),
        (.littleDIP, .littleTip)
    ]
}

