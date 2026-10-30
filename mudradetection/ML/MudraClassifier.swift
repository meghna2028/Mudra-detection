import Foundation
import CoreML

public struct MudraClassificationResult: Sendable {
    public let predictedMudra: MudraType?
    public let confidence: Float
    public let allProbabilities: [MudraType: Float]
    public let isConfident: Bool

    public init(
        predictedMudra: MudraType?,
        confidence: Float,
        allProbabilities: [MudraType: Float],
        isConfident: Bool
    ) {
        self.predictedMudra = predictedMudra
        self.confidence = confidence
        self.allProbabilities = allProbabilities
        self.isConfident = isConfident
    }
}

/// MudraClassifier: Integrates Core ML with Swift for live Mudra inference
public final class MudraClassifier: @unchecked Sendable {
    private var modelInstance: MudraClassifierModel?
    public let confidenceThreshold: Float

    public init(confidenceThreshold: Float = 0.65) {
        self.confidenceThreshold = confidenceThreshold
        loadModel()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // CPU + GPU + Apple Neural Engine
            self.modelInstance = try MudraClassifierModel(configuration: config)
            print("[MudraClassifier] Core ML model loaded successfully.")
        } catch {
            print("[MudraClassifier] Error initializing MudraClassifierModel: \(error)")
        }
    }

    /// Predicts the Mudra gesture from a 77-element normalized landmark feature vector.
    /// Returns nil if features are invalid or model is not loaded.
    public func classify(features: [Float]) -> MudraClassificationResult? {
        guard features.count == HandFeatureExtractor.expectedFeatureCount else {
            print("[MudraClassifier] Invalid feature dimension: expected \(HandFeatureExtractor.expectedFeatureCount), got \(features.count)")
            return nil
        }

        guard let model = self.modelInstance else {
            print("[MudraClassifier] Warning: Model instance not initialized.")
            return nil
        }

        do {
            // Create MLMultiArray with shape [1, 77]
            let shape: [NSNumber] = [1, NSNumber(value: HandFeatureExtractor.expectedFeatureCount)]
            let multiArray = try MLMultiArray(shape: shape, dataType: .float32)

            for (index, value) in features.enumerated() {
                multiArray[index] = NSNumber(value: value)
            }

            let input = MudraClassifierModelInput(features: multiArray)
            let output = try model.prediction(input: input)

            // Extract class probabilities
            var probabilities: [MudraType: Float] = [:]
            var bestMudra: MudraType? = nil
            var maxConfidence: Float = 0.0

            for (label, probNum) in output.classLabel_probs {
                if let mudra = MudraType(rawValue: label) {
                    let prob = Float(probNum)
                    probabilities[mudra] = prob
                    if prob > maxConfidence {
                        maxConfidence = prob
                        bestMudra = mudra
                    }
                }
            }

            if bestMudra == nil, let directMudra = MudraType(rawValue: output.classLabel) {
                bestMudra = directMudra
                maxConfidence = 1.0
                probabilities[directMudra] = 1.0
            }

            let isConfident = maxConfidence >= confidenceThreshold

            return MudraClassificationResult(
                predictedMudra: bestMudra,
                confidence: maxConfidence,
                allProbabilities: probabilities,
                isConfident: isConfident
            )

        } catch {
            print("[MudraClassifier] Prediction failed: \(error)")
            return nil
        }
    }
}
