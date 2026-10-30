import SwiftUI

/// Glassmorphic bottom card displaying live Mudra detection details
public struct DetectionResultCard: View {
    public let state: MudraDetectionState
    public let allProbabilities: [MudraType: Float]
    public let onInfoTapped: () -> Void

    public init(
        state: MudraDetectionState,
        allProbabilities: [MudraType: Float] = [:],
        onInfoTapped: @escaping () -> Void = {}
    ) {
        self.state = state
        self.allProbabilities = allProbabilities
        self.onInfoTapped = onInfoTapped
    }

    public var body: some View {
        VStack(spacing: 12) {
            // Header: Status Badge + Reference Button
            HStack {
                statusBadge

                Spacer()

                Button(action: onInfoTapped) {
                    HStack(spacing: 5) {
                        Image(systemName: "book.pages")
                            .font(.system(size: 13, weight: .semibold))
                        Text("Mudra Guide")
                            .font(.system(size: 13, weight: .semibold, design: .serif))
                    }
                    .foregroundColor(Color(red: 0.36, green: 0.45, blue: 0.28)) // Olive Green
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color(red: 0.36, green: 0.45, blue: 0.28).opacity(0.12))
                    .cornerRadius(12)
                }
            }

            // Main Content: Mudra Title, Sanskrit Name, and Confidence Meter
            HStack(alignment: .center, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    switch state {
                    case .detected(let mudra, _):
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(mudra.rawValue)
                                .font(.system(size: 26, weight: .bold, design: .serif))
                                .foregroundColor(Color(red: 0.17, green: 0.12, blue: 0.09)) // Deep Brown

                            Text(mudra.sanskritTitle)
                                .font(.system(size: 20, weight: .medium, design: .serif))
                                .foregroundColor(Color(red: 0.72, green: 0.35, blue: 0.22)) // Terracotta
                        }

                        Text(mudra.englishMeaning)
                            .font(.system(size: 15, weight: .medium, design: .serif))
                            .foregroundColor(Color(red: 0.42, green: 0.35, blue: 0.30))

                    case .uncertain(let candidate, _):
                        Text(candidate?.rawValue ?? "Analyzing Gesture...")
                            .font(.system(size: 24, weight: .bold, design: .serif))
                            .foregroundColor(Color(red: 0.35, green: 0.28, blue: 0.23))

                        Text("Adjust hand pose or lighting for clearer alignment")
                            .font(.system(size: 13, weight: .regular))
                            .foregroundColor(Color(red: 0.55, green: 0.48, blue: 0.43))

                    case .handNotDetected:
                        Text("Hold Hand in View")
                            .font(.system(size: 22, weight: .bold, design: .serif))
                            .foregroundColor(Color(red: 0.45, green: 0.38, blue: 0.33))

                        Text("Align palm inside camera frame to begin recognition")
                            .font(.system(size: 13, weight: .regular))
                            .foregroundColor(Color(red: 0.60, green: 0.52, blue: 0.47))
                    }
                }

                Spacer()

                // Circular Confidence Meter
                switch state {
                case .detected(_, let conf):
                    ConfidenceIndicator(confidence: conf, size: 58)
                case .uncertain(_, let conf):
                    ConfidenceIndicator(confidence: conf, size: 58)
                case .handNotDetected:
                    ZStack {
                        Circle()
                            .stroke(Color.gray.opacity(0.2), lineWidth: 3.5)
                        Image(systemName: "hand.raised")
                            .font(.system(size: 22))
                            .foregroundColor(Color.gray.opacity(0.5))
                    }
                    .frame(width: 58, height: 58)
                }
            }

            // Description / Posture hint
            if case .detected(let mudra, _) = state {
                Divider()
                    .background(Color(red: 0.88, green: 0.85, blue: 0.80).opacity(0.6))

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: mudra.iconSymbol)
                        .foregroundColor(Color(red: 0.36, green: 0.45, blue: 0.28))
                        .font(.system(size: 14))
                        .padding(.top, 2)

                    Text(mudra.description)
                        .font(.system(size: 12.5, weight: .regular))
                        .foregroundColor(Color(red: 0.30, green: 0.24, blue: 0.20))
                        .lineLimit(2)
                }
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Color(red: 0.984, green: 0.973, blue: 0.953)) // Ivory / Beige
                .shadow(color: Color.black.opacity(0.16), radius: 16, x: 0, y: 6)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color(red: 0.88, green: 0.85, blue: 0.79), lineWidth: 1.2)
        )
    }

    private var statusBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(badgeColor)
                .frame(width: 8, height: 8)

            Text(badgeTitle)
                .font(.system(size: 11.5, weight: .bold))
                .foregroundColor(badgeColor)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 4)
        .background(badgeColor.opacity(0.12))
        .cornerRadius(10)
    }

    private var badgeColor: Color {
        switch state {
        case .detected:
            return Color(red: 0.36, green: 0.45, blue: 0.28) // Olive Green
        case .uncertain:
            return Color(red: 0.78, green: 0.54, blue: 0.21) // Amber
        case .handNotDetected:
            return Color(red: 0.55, green: 0.48, blue: 0.43) // Muted Brown
        }
    }

    private var badgeTitle: String {
        switch state {
        case .detected:
            return "MUDRA IDENTIFIED"
        case .uncertain:
            return "LOW CONFIDENCE"
        case .handNotDetected:
            return "SCANNING"
        }
    }
}

