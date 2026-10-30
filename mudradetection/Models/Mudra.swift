import Foundation
import SwiftUI

/// Information and metadata for the 10 Asamyuta Hastas (Single Hand Gestures)
public enum MudraType: String, CaseIterable, Identifiable, Codable, Sendable {
    case pataka = "Pataka"
    case tripataka = "Tripataka"
    case ardhapataka = "Ardhapataka"
    case kartarimukha = "Kartarimukha"
    case mayura = "Mayura"
    case ardhachandra = "Ardhachandra"
    case mushti = "Mushti"
    case shikhara = "Shikhara"
    case suchi = "Suchi"
    case kapittha = "Kapittha"

    public var id: String { rawValue }

    public var sanskritTitle: String {
        switch self {
        case .pataka: return "पताक"
        case .tripataka: return "त्रिपताक"
        case .ardhapataka: return "अर्धपताक"
        case .kartarimukha: return "कर्तरीमुख"
        case .mayura: return "मयूर"
        case .ardhachandra: return "अर्धचन्द्र"
        case .mushti: return "मुष्टि"
        case .shikhara: return "शिखर"
        case .suchi: return "सूची"
        case .kapittha: return "कपित्थ"
        }
    }

    public var englishMeaning: String {
        switch self {
        case .pataka: return "Flag"
        case .tripataka: return "Three Parts of Flag"
        case .ardhapataka: return "Half Flag"
        case .kartarimukha: return "Scissors Face"
        case .mayura: return "Peacock"
        case .ardhachandra: return "Half Moon"
        case .mushti: return "Fist"
        case .shikhara: return "Peak / Spire"
        case .suchi: return "Needle"
        case .kapittha: return "Wood Apple"
        }
    }

    public var description: String {
        switch self {
        case .pataka:
            return "All four fingers held straight and close together, thumb slightly bent near the index finger base."
        case .tripataka:
            return "Ring finger bent downward at the joint while index, middle, little fingers and thumb remain extended."
        case .ardhapataka:
            return "Ring and little fingers bent down together while index and middle fingers extend straight."
        case .kartarimukha:
            return "Index and middle fingers extended apart in a V-scissor shape, while ring and little fingers bend to touch thumb."
        case .mayura:
            return "Tip of the ring finger touches the tip of the thumb, with other fingers extended gracefully."
        case .ardhachandra:
            return "Four fingers stretched straight together, while the thumb is opened and extended wide outward."
        case .mushti:
            return "Four fingers curled tightly into the palm into a fist, with the thumb placed firmly over them."
        case .shikhara:
            return "Fingers curled into a fist like Mushti, but the thumb is held erect pointing straight up."
        case .suchi:
            return "Index finger points straight up like a needle, while middle, ring, little fingers curl with thumb over them."
        case .kapittha:
            return "Index finger bent down over the thumb tip, with remaining fingers curled into the palm."
        }
    }

    public var iconSymbol: String {
        switch self {
        case .pataka: return "hand.raised.fill"
        case .tripataka: return "hand.point.up.fill"
        case .ardhapataka: return "hand.point.up.left.fill"
        case .kartarimukha: return "scissors"
        case .mayura: return "bird.fill"
        case .ardhachandra: return "moon.stars.fill"
        case .mushti: return "hand.fist.fill"
        case .shikhara: return "hand.thumbsup.fill"
        case .suchi: return "hand.point.up"
        case .kapittha: return "apple.logo"
        }
    }
}

/// The state of the real-time detector pipeline
public enum MudraDetectionState: Equatable, Sendable {
    case handNotDetected
    case uncertain(candidate: MudraType?, confidence: Float)
    case detected(mudra: MudraType, confidence: Float)

    public var displayText: String {
        switch self {
        case .handNotDetected:
            return "Hand Not Detected"
        case .uncertain(let candidate, let confidence):
            if let candidate = candidate {
                return "Uncertain (\(candidate.rawValue) \(Int(confidence * 100))%)"
            }
            return "Uncertain Mudra"
        case .detected(let mudra, _):
            return mudra.rawValue
        }
    }

    public var confidencePercentage: Int {
        switch self {
        case .handNotDetected:
            return 0
        case .uncertain(_, let conf):
            return Int(conf * 100)
        case .detected(_, let conf):
            return Int(conf * 100)
        }
    }
}

