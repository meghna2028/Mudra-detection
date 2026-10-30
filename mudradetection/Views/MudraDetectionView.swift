import SwiftUI
import AVFoundation
import QuartzCore

/// Primary Bharatanatyam Mudra Detection Live View
public struct MudraDetectionView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var coordinator: DetectionCoordinator?

    @State private var detectionState: MudraDetectionState = .handNotDetected
    @State private var detectedHands: [DetectedHand] = []
    @State private var allProbabilities: [MudraType: Float] = [:]
    @State private var isShowingReference = false
    @State private var isShowingCollector = false
    @State private var showSkeleton = true
    @State private var simulatorSelectedMudra: MudraType = .pataka

    public init() {}

    public var body: some View {
        ZStack {
            // 1. Live Camera Preview Layer
            if cameraManager.permissionGranted {
                CameraPreview(session: cameraManager.captureSession)
                    .ignoresSafeArea()
            } else {
                // Fallback / Simulator Canvas
                simulatorCanvas
                    .ignoresSafeArea()
            }

            // 2. Hand Landmark & Skeleton Overlay
            if showSkeleton && !detectedHands.isEmpty {
                HandLandmarkOverlay(
                    hands: detectedHands,
                    isMirrored: cameraManager.currentPosition == .front
                )
                .ignoresSafeArea()
            }

            // 3. UI Controls and Overlay
            VStack(spacing: 0) {
                // Top Navigation & Action Bar
                topBar
                    .padding(.horizontal, 16)
                    .padding(.top, 8)

                Spacer()

                // Simulator Gesture Selector Bar (when camera unavailable)
                if !cameraManager.permissionGranted {
                    simulatorMudraPicker
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                }

                // Bottom Detection Result Card
                DetectionResultCard(
                    state: detectionState,
                    allProbabilities: allProbabilities,
                    onInfoTapped: { isShowingReference = true }
                )
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
            }
        }
        .onAppear {
            setupPipeline()
            cameraManager.startSession()
        }
        .onDisappear {
            cameraManager.stopSession()
        }
        .sheet(isPresented: $isShowingReference) {
            MudraReferenceView()
        }
        .sheet(isPresented: $isShowingCollector) {
            MudraDatasetCollectorView(currentHand: detectedHands.first)
        }
    }

    // MARK: - Pipeline Setup

    private func setupPipeline() {
        let coord = DetectionCoordinator(
            onResultsUpdated: { state, hands, probs in
                DispatchQueue.main.async {
                    self.detectionState = state
                    self.detectedHands = hands
                    self.allProbabilities = probs
                }
            }
        )
        self.coordinator = coord
        cameraManager.sampleBufferDelegate = coord
    }

    // MARK: - UI Subviews

    private var topBar: some View {
        HStack(alignment: .center, spacing: 12) {
            // App Title Pill
            HStack(spacing: 6) {
                Image(systemName: "hand.wave.fill")
                    .font(.system(size: 15))
                    .foregroundColor(Color(red: 0.72, green: 0.35, blue: 0.22)) // Terracotta

                Text("Mudra Vision")
                    .font(.system(size: 16, weight: .bold, design: .serif))
                    .foregroundColor(Color(red: 0.17, green: 0.12, blue: 0.09))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color(red: 0.984, green: 0.973, blue: 0.953).opacity(0.92))
            .cornerRadius(20)
            .shadow(color: Color.black.opacity(0.12), radius: 6, x: 0, y: 2)

            Spacer()

            // Toggle Skeleton
            Button(action: { showSkeleton.toggle() }) {
                Image(systemName: showSkeleton ? "point.3.filled.connected.trianglepath.dotted" : "point.3.connected.trianglepath.dotted")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(showSkeleton ? Color(red: 0.36, green: 0.45, blue: 0.28) : .gray)
                    .frame(width: 40, height: 40)
                    .background(Color(red: 0.984, green: 0.973, blue: 0.953).opacity(0.92))
                    .clipShape(Circle())
                    .shadow(color: Color.black.opacity(0.12), radius: 6, x: 0, y: 2)
            }

            // Flip Camera
            Button(action: { cameraManager.switchCamera() }) {
                Image(systemName: "camera.rotate.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(Color(red: 0.25, green: 0.20, blue: 0.16))
                    .frame(width: 40, height: 40)
                    .background(Color(red: 0.984, green: 0.973, blue: 0.953).opacity(0.92))
                    .clipShape(Circle())
                    .shadow(color: Color.black.opacity(0.12), radius: 6, x: 0, y: 2)
            }

            // In-App Data Collector Studio
            Button(action: { isShowingCollector = true }) {
                Image(systemName: "plus.rectangle.on.folder.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(Color(red: 0.72, green: 0.35, blue: 0.22))
                    .frame(width: 40, height: 40)
                    .background(Color(red: 0.984, green: 0.973, blue: 0.953).opacity(0.92))
                    .clipShape(Circle())
                    .shadow(color: Color.black.opacity(0.12), radius: 6, x: 0, y: 2)
            }
        }
    }

    private var simulatorCanvas: some View {
        ZStack {
            // Elegant studio background for simulator / no camera
            LinearGradient(
                colors: [Color(red: 0.18, green: 0.14, blue: 0.12), Color(red: 0.10, green: 0.08, blue: 0.06)],
                startPoint: .top,
                endPoint: .bottom
            )

            VStack(spacing: 12) {
                Image(systemName: "camera.viewfinder")
                    .font(.system(size: 48))
                    .foregroundColor(Color(red: 0.72, green: 0.35, blue: 0.22).opacity(0.8))

                Text("Simulation Mode Active")
                    .font(.system(size: 18, weight: .bold, design: .serif))
                    .foregroundColor(Color(red: 0.96, green: 0.94, blue: 0.90))

                Text("On real iPhone, camera preview streams live.\nSelect a mudra below to verify the Core ML engine.")
                    .font(.system(size: 13))
                    .multilineTextAlignment(.center)
                    .foregroundColor(Color.gray.opacity(0.8))
                    .padding(.horizontal, 32)
            }
        }
    }

    private var simulatorMudraPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(MudraType.allCases) { mudra in
                    Button(action: {
                        simulatorSelectedMudra = mudra
                        coordinator?.simulateMudra(mudra)
                    }) {
                        Text(mudra.rawValue)
                            .font(.system(size: 13, weight: .semibold, design: .serif))
                            .foregroundColor(simulatorSelectedMudra == mudra ? .white : Color(red: 0.25, green: 0.20, blue: 0.16))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(
                                simulatorSelectedMudra == mudra
                                    ? Color(red: 0.36, green: 0.45, blue: 0.28)
                                    : Color(red: 0.98, green: 0.96, blue: 0.92)
                            )
                            .cornerRadius(16)
                            .shadow(color: Color.black.opacity(0.15), radius: 4, x: 0, y: 2)
                    }
                }
            }
            .padding(.horizontal, 4)
        }
    }
}

// MARK: - Pipeline Coordinator
final class DetectionCoordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, HandPoseDetectorDelegate, @unchecked Sendable {
    private let detector = HandPoseDetector(maxHands: 1, targetFPS: 30.0, minConfidence: 0.25)
    private let extractor = HandFeatureExtractor()
    private let classifier = MudraClassifier(confidenceThreshold: 0.65)
    private let smoother = PredictionSmoother(windowSize: 5, minConsecutiveFrames: 3, certaintyThreshold: 0.65)

    private let onResultsUpdated: (MudraDetectionState, [DetectedHand], [MudraType: Float]) -> Void

    init(onResultsUpdated: @escaping (MudraDetectionState, [DetectedHand], [MudraType: Float]) -> Void) {
        self.onResultsUpdated = onResultsUpdated
        super.init()
        detector.delegate = self
    }

    // AVCaptureVideoDataOutput delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        detector.processSampleBuffer(sampleBuffer)
    }

    // HandPoseDetectorDelegate
    func handPoseDetector(_ detector: HandPoseDetector, didDetect hands: [DetectedHand]) {
        guard let primaryHand = hands.first,
              let points = primaryHand.orderedPoints,
              let features = extractor.extractFeatures(from: points) else {
            let state = smoother.update(with: nil, timestamp: CACurrentMediaTime())
            onResultsUpdated(state, [], [:])
            return
        }

        let result = classifier.classify(features: features)
        let state = smoother.update(with: result, timestamp: primaryHand.timestamp)
        let probs = result?.allProbabilities ?? [:]

        onResultsUpdated(state, hands, probs)
    }

    func handPoseDetector(_ detector: HandPoseDetector, didEncounterError error: Error) {
        print("[DetectionCoordinator] Vision error: \(error)")
    }

    /// Simulation trigger for testing without physical iPhone camera
    func simulateMudra(_ mudra: MudraType) {
        // Feed synthetic result through smoother and notify UI
        let syntheticProbs: [MudraType: Float] = [mudra: 0.96]
        let res = MudraClassificationResult(
            predictedMudra: mudra,
            confidence: 0.96,
            allProbabilities: syntheticProbs,
            isConfident: true
        )
        let state = smoother.update(with: res, timestamp: CACurrentMediaTime())
        onResultsUpdated(state, [], syntheticProbs)
    }
}
