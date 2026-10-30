import Foundation

/// Temporal smoothing buffer for real-time video inference.
/// Uses a sliding window with probability exponential smoothing to eliminate frame flicker
/// and prevent noisy transient predictions.
public final class PredictionSmoother: Sendable {
    private struct FramePrediction: Sendable {
        let mudra: MudraType?
        let confidence: Float
        let probabilities: [MudraType: Float]
        let timestamp: TimeInterval
    }

    private let windowSize: Int
    private let minConsecutiveFrames: Int
    private let certaintyThreshold: Float

    // Internal state protected by Lock
    private let lock = NSLock()
    private var history: [FramePrediction] = []
    private var lastStableState: MudraDetectionState = .handNotDetected
    private var noHandCount: Int = 0

    public init(windowSize: Int = 6, minConsecutiveFrames: Int = 3, certaintyThreshold: Float = 0.65) {
        self.windowSize = windowSize
        self.minConsecutiveFrames = minConsecutiveFrames
        self.certaintyThreshold = certaintyThreshold
    }

    /// Feeds a new frame result and returns the temporally smoothed detection state
    public func update(with result: MudraClassificationResult?, timestamp: TimeInterval) -> MudraDetectionState {
        lock.lock()
        defer { lock.unlock() }

        guard let result = result else {
            noHandCount += 1
            if noHandCount >= 3 {
                history.removeAll()
                lastStableState = .handNotDetected
            }
            return lastStableState
        }

        noHandCount = 0

        // Append to history
        let frame = FramePrediction(
            mudra: result.predictedMudra,
            confidence: result.confidence,
            probabilities: result.allProbabilities,
            timestamp: timestamp
        )
        history.append(frame)
        if history.count > windowSize {
            history.removeFirst()
        }

        // 1. Average class probabilities across the temporal sliding window
        var aggregatedProbs: [MudraType: Float] = [:]
        for h in history {
            for (mudra, prob) in h.probabilities {
                aggregatedProbs[mudra, default: 0.0] += prob
            }
        }

        let historyCount = Float(history.count)
        var bestMudra: MudraType? = nil
        var highestSmoothedProb: Float = 0.0

        for (mudra, sumProb) in aggregatedProbs {
            let avgProb = sumProb / historyCount
            if avgProb > highestSmoothedProb {
                highestSmoothedProb = avgProb
                bestMudra = mudra
            }
        }

        // 2. Count recent consistent votes
        let recentMatches = history.suffix(minConsecutiveFrames).filter { $0.mudra == bestMudra }.count

        if highestSmoothedProb >= certaintyThreshold && recentMatches >= minConsecutiveFrames, let mudra = bestMudra {
            lastStableState = .detected(mudra: mudra, confidence: highestSmoothedProb)
        } else {
            lastStableState = .uncertain(candidate: bestMudra, confidence: highestSmoothedProb)
        }

        return lastStableState
    }

    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        history.removeAll()
        noHandCount = 0
        lastStableState = .handNotDetected
    }
}

