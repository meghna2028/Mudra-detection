import SwiftUI

/// An elegant circular or linear confidence meter tailored to the Bharatanatyam palette
public struct ConfidenceIndicator: View {
    public let confidence: Float
    public let showPercentage: Bool
    public let size: CGFloat

    public init(confidence: Float, showPercentage: Bool = true, size: CGFloat = 54) {
        self.confidence = max(0.0, min(1.0, confidence))
        self.showPercentage = showPercentage
        self.size = size
    }

    private var tintColor: Color {
        if confidence >= 0.85 {
            return Color(red: 0.36, green: 0.45, blue: 0.28) // Olive Green
        } else if confidence >= 0.65 {
            return Color(red: 0.78, green: 0.54, blue: 0.21) // Ochre / Warm Amber
        } else {
            return Color(red: 0.66, green: 0.30, blue: 0.20) // Muted Terracotta
        }
    }

    public var body: some View {
        ZStack {
            // Background track
            Circle()
                .stroke(Color(red: 0.88, green: 0.85, blue: 0.80).opacity(0.4), lineWidth: 4.5)

            // Progress fill
            Circle()
                .trim(from: 0.0, to: CGFloat(confidence))
                .stroke(
                    AngularGradient(
                        gradient: Gradient(colors: [tintColor.opacity(0.7), tintColor]),
                        center: .center,
                        startAngle: .degrees(-90),
                        endAngle: .degrees(270)
                    ),
                    style: StrokeStyle(lineWidth: 4.5, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .animation(.spring(response: 0.3, dampingFraction: 0.75), value: confidence)

            if showPercentage {
                Text("\(Int(confidence * 100))%")
                    .font(.system(size: size * 0.28, weight: .bold, design: .serif))
                    .foregroundColor(Color(red: 0.24, green: 0.15, blue: 0.10))
            }
        }
        .frame(width: size, height: size)
    }
}

