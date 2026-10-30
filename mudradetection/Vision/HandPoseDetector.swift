import Foundation
import CoreGraphics
import CoreMedia
import Vision
import Combine
import QuartzCore

/// Protocol for receiving hand detection results
public protocol HandPoseDetectorDelegate: AnyObject, Sendable {
    func handPoseDetector(_ detector: HandPoseDetector, didDetect hands: [DetectedHand])
    func handPoseDetector(_ detector: HandPoseDetector, didEncounterError error: Error)
}

/// Real-time Hand Pose Detector using Apple's Vision framework
public final class HandPoseDetector: @unchecked Sendable {
    public weak var delegate: HandPoseDetectorDelegate?

    private let handPoseRequest = VNDetectHumanHandPoseRequest()
    private let processingQueue = DispatchQueue(label: "com.mudradetection.vision.processing", qos: .userInteractive)

    private var isProcessingFrame = false
    private var lastProcessedTime: TimeInterval = 0
    private let minFrameInterval: TimeInterval // Minimum seconds between frames (e.g., 1/30s = ~0.033s)
    private let minLandmarkConfidence: Float

    public init(maxHands: Int = 2, targetFPS: Double = 30.0, minConfidence: Float = 0.25) {
        self.handPoseRequest.maximumHandCount = maxHands
        self.minFrameInterval = 1.0 / targetFPS
        self.minLandmarkConfidence = minConfidence
    }

    /// Processes a video frame sample buffer. Automatically drops frames if processing is active to prevent lag.
    public func processSampleBuffer(_ sampleBuffer: CMSampleBuffer, orientation: CGImagePropertyOrientation = .up) {
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastProcessedTime >= minFrameInterval else {
            return // Skip frame to maintain smooth target FPS
        }

        guard !isProcessingFrame else {
            return // Drop frame: previous inference still in progress
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        isProcessingFrame = true
        lastProcessedTime = currentTime

        processingQueue.async { [weak self] in
            guard let self = self else { return }
            defer { self.isProcessingFrame = false }

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
            do {
                try handler.perform([self.handPoseRequest])
                let hands = self.extractHands(from: self.handPoseRequest.results, timestamp: currentTime)
                self.delegate?.handPoseDetector(self, didDetect: hands)
            } catch {
                self.delegate?.handPoseDetector(self, didEncounterError: error)
            }
        }
    }

    /// Processes a static CGImage for testing or dataset extraction
    public func processImage(_ image: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> [DetectedHand] {
        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
        try handler.perform([handPoseRequest])
        return extractHands(from: handPoseRequest.results, timestamp: CACurrentMediaTime())
    }

    private func extractHands(from observations: [VNHumanHandPoseObservation]?, timestamp: TimeInterval) -> [DetectedHand] {
        guard let observations = observations, !observations.isEmpty else {
            return []
        }

        var detectedHands: [DetectedHand] = []

        for observation in observations {
            var landmarks: [HandLandmarkKey: LandmarkPoint] = [:]
            var totalConfidence: Float = 0
            var validPointsCount = 0

            for key in HandLandmarkKey.allCases {
                do {
                    let recognizedPoint = try observation.recognizedPoint(key.visionJointName)
                    // Only accept points that meet confidence threshold
                    if recognizedPoint.confidence >= minLandmarkConfidence {
                        let point = CGPoint(x: recognizedPoint.location.x, y: recognizedPoint.location.y)
                        landmarks[key] = LandmarkPoint(key: key, point: point, confidence: recognizedPoint.confidence)
                        totalConfidence += recognizedPoint.confidence
                        validPointsCount += 1
                    }
                } catch {
                    // Landmark not found in observation
                    continue
                }
            }

            let avgConfidence = validPointsCount > 0 ? (totalConfidence / Float(validPointsCount)) : 0.0
            let hand = DetectedHand(
                landmarks: landmarks,
                chirality: observation.chirality,
                averageConfidence: avgConfidence,
                timestamp: timestamp
            )
            detectedHands.append(hand)
        }

        return detectedHands
    }
}
