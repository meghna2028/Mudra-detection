import AVFoundation
import SwiftUI
import Combine

/// Manages AVFoundation camera capture session for live video frames
public final class CameraManager: NSObject, ObservableObject, @unchecked Sendable {
    @Published public var isSessionRunning = false
    @Published public var permissionGranted = false
    @Published public var currentPosition: AVCaptureDevice.Position = .front

    public let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.mudradetection.camera.session", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()
    private var videoDeviceInput: AVCaptureDeviceInput?

    public var sampleBufferDelegate: AVCaptureVideoDataOutputSampleBufferDelegate? {
        didSet {
            videoOutput.setSampleBufferDelegate(sampleBufferDelegate, queue: sessionQueue)
        }
    }

    public override init() {
        super.init()
        checkPermissions()
    }

    public func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            self.permissionGranted = true
            self.sessionQueue.async { [weak self] in
                self?.setupSession()
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.permissionGranted = granted
                    if granted {
                        self?.sessionQueue.async {
                            self?.setupSession()
                        }
                    }
                }
            }
        default:
            DispatchQueue.main.async {
                self.permissionGranted = false
            }
        }
    }

    private func setupSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high

        // Add Video Input
        guard let device = selectDevice(for: currentPosition) else {
            captureSession.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
                self.videoDeviceInput = input
            }
        } catch {
            print("[CameraManager] Error creating device input: \(error)")
        }

        // Add Video Output
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        // Configure video orientation
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            if currentPosition == .front && connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }

        captureSession.commitConfiguration()
    }

    public func startSession() {
        sessionQueue.async { [weak self] in
            guard let self = self, !self.captureSession.isRunning else { return }
            self.captureSession.startRunning()
            DispatchQueue.main.async {
                self.isSessionRunning = self.captureSession.isRunning
            }
        }
    }

    public func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self = self, self.captureSession.isRunning else { return }
            self.captureSession.stopRunning()
            DispatchQueue.main.async {
                self.isSessionRunning = false
            }
        }
    }

    public func switchCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.captureSession.beginConfiguration()

            if let currentInput = self.videoDeviceInput {
                self.captureSession.removeInput(currentInput)
            }

            let newPosition: AVCaptureDevice.Position = (self.currentPosition == .front) ? .back : .front
            guard let newDevice = self.selectDevice(for: newPosition) else {
                self.captureSession.commitConfiguration()
                return
            }

            do {
                let newInput = try AVCaptureDeviceInput(device: newDevice)
                if self.captureSession.canAddInput(newInput) {
                    self.captureSession.addInput(newInput)
                    self.videoDeviceInput = newInput
                    self.currentPosition = newPosition
                }
            } catch {
                print("[CameraManager] Error switching camera: \(error)")
            }

            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
                if newPosition == .front && connection.isVideoMirroringSupported {
                    connection.isVideoMirrored = true
                }
            }

            self.captureSession.commitConfiguration()
        }
    }

    private func selectDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: position
        )
        return discoverySession.devices.first
    }
}

