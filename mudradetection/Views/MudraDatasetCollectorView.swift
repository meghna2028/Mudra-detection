import SwiftUI
import CoreGraphics

/// In-app tool for capturing live performer landmark datasets.
/// Records 21 hand landmarks and exports clean CSVs tagged with performer_id, session_id, and mudra class.
public struct MudraDatasetCollectorView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var selectedMudra: MudraType = .pataka
    @State private var performerID: String = "performer_live_01"
    @State private var sessionID: String = "session_01"
    @State private var isRecording = false
    @State private var capturedSamples: [RecordedSample] = []
    @State private var showShareSheet = false
    @State private var exportedCSVURL: URL?

    public let currentHand: DetectedHand?

    public struct RecordedSample: Identifiable {
        public let id = UUID()
        public let mudra: MudraType
        public let performerID: String
        public let sessionID: String
        public let landmarks: [CGPoint]
        public let timestamp: Date
    }

    public init(currentHand: DetectedHand?) {
        self.currentHand = currentHand
    }

    public var body: some View {
        NavigationStack {
            ZStack {
                Color(red: 0.968, green: 0.953, blue: 0.925)
                    .ignoresSafeArea()

                VStack(spacing: 16) {
                    // Header card
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Mudra Data Collection Studio")
                            .font(.system(size: 20, weight: .bold, design: .serif))
                            .foregroundColor(Color(red: 0.17, green: 0.12, blue: 0.09))

                        Text("Capture real performer landmark recordings for training and validation splits.")
                            .font(.system(size: 13, weight: .regular))
                            .foregroundColor(Color(red: 0.45, green: 0.38, blue: 0.32))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.white)
                    .cornerRadius(14)
                    .padding(.horizontal)

                    // Controls form
                    VStack(spacing: 12) {
                        // Mudra Picker
                        HStack {
                            Text("Target Mudra:")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundColor(Color(red: 0.25, green: 0.20, blue: 0.16))
                            Spacer()
                            Picker("Mudra", selection: $selectedMudra) {
                                ForEach(MudraType.allCases) { m in
                                    Text("\(m.rawValue) (\(m.sanskritTitle))").tag(m)
                                }
                            }
                            .pickerStyle(.menu)
                            .tint(Color(red: 0.36, green: 0.45, blue: 0.28))
                        }

                        Divider()

                        // Performer & Session fields
                        HStack(spacing: 12) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Performer ID")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(.gray)
                                TextField("e.g. performer_01", text: $performerID)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.system(size: 13))
                            }

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Session ID")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(.gray)
                                TextField("e.g. session_01", text: $sessionID)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.system(size: 13))
                            }
                        }
                    }
                    .padding()
                    .background(Color.white)
                    .cornerRadius(14)
                    .padding(.horizontal)

                    // Hand Status
                    HStack(spacing: 10) {
                        Circle()
                            .fill(currentHand != nil ? Color.green : Color.red)
                            .frame(width: 10, height: 10)

                        Text(currentHand != nil ? "Hand detected (ready to capture)" : "No hand in view")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(currentHand != nil ? Color.green.opacity(0.8) : Color.red.opacity(0.8))

                        Spacer()

                        Text("Total: \(capturedSamples.count) samples")
                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                            .foregroundColor(Color(red: 0.36, green: 0.45, blue: 0.28))
                    }
                    .padding(.horizontal, 24)

                    // Capture Buttons
                    HStack(spacing: 14) {
                        Button(action: captureSingleFrame) {
                            HStack {
                                Image(systemName: "camera.circle.fill")
                                Text("Capture Frame")
                            }
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(currentHand != nil ? Color(red: 0.36, green: 0.45, blue: 0.28) : Color.gray)
                            .cornerRadius(12)
                        }
                        .disabled(currentHand == nil)

                        Button(action: exportDataset) {
                            HStack {
                                Image(systemName: "square.and.arrow.up")
                                Text("Export CSV")
                            }
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundColor(Color(red: 0.36, green: 0.45, blue: 0.28))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(Color(red: 0.36, green: 0.45, blue: 0.28).opacity(0.12))
                            .cornerRadius(12)
                        }
                        .disabled(capturedSamples.isEmpty)
                    }
                    .padding(.horizontal)

                    // Captured breakdown list
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Session Log")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(Color(red: 0.35, green: 0.28, blue: 0.22))

                        ScrollView {
                            LazyVStack(spacing: 6) {
                                ForEach(capturedSamples.reversed()) { s in
                                    HStack {
                                        Text(s.mudra.rawValue)
                                            .font(.system(size: 13, weight: .semibold))
                                        Spacer()
                                        Text("\(s.performerID) • \(s.sessionID)")
                                            .font(.system(size: 11, design: .monospaced))
                                            .foregroundColor(.gray)
                                    }
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(Color.white)
                                    .cornerRadius(8)
                                }
                            }
                        }
                    }
                    .padding(.horizontal)

                    Spacer()
                }
                .padding(.top)
            }
            .navigationTitle("Dataset Collector")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Close") { dismiss() }
                        .foregroundColor(Color(red: 0.36, green: 0.45, blue: 0.28))
                }
            }
            .sheet(isPresented: $showShareSheet) {
                if let url = exportedCSVURL {
                    ShareSheet(activityItems: [url])
                }
            }
        }
    }

    private func captureSingleFrame() {
        guard let hand = currentHand, let points = hand.orderedPoints else { return }

        let sample = RecordedSample(
            mudra: selectedMudra,
            performerID: performerID.trimmingCharacters(in: .whitespacesAndNewlines),
            sessionID: sessionID.trimmingCharacters(in: .whitespacesAndNewlines),
            landmarks: points,
            timestamp: Date()
        )
        capturedSamples.append(sample)
    }

    private func exportDataset() {
        var csvText = "sample_id,mudra_class,performer_id,session_id"
        // 21 raw landmarks
        for i in 0..<21 {
            csvText += ",lm_\(i)_x,lm_\(i)_y"
        }
        csvText += "\n"

        for (index, sample) in capturedSamples.enumerated() {
            var row = "sample_\(index),\(sample.mudra.rawValue),\(sample.performerID),\(sample.sessionID)"
            for pt in sample.landmarks {
                row += ",\(pt.x),\(pt.y)"
            }
            csvText += row + "\n"
        }

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("mudra_dataset_\(sessionID).csv")
        do {
            try csvText.write(to: tempURL, atomically: true, encoding: .utf8)
            self.exportedCSVURL = tempURL
            self.showShareSheet = true
        } catch {
            print("Failed to write CSV: \(error)")
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

