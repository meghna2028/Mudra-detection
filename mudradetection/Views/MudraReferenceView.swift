import SwiftUI

/// Comprehensive interactive reference catalog for the 10 Bharatanatyam Asamyuta Hastas
public struct MudraReferenceView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""
    @State private var selectedMudra: MudraType? = nil

    private var filteredMudras: [MudraType] {
        if searchText.isEmpty {
            return MudraType.allCases
        }
        return MudraType.allCases.filter {
            $0.rawValue.localizedCaseInsensitiveContains(searchText) ||
            $0.englishMeaning.localizedCaseInsensitiveContains(searchText) ||
            $0.sanskritTitle.contains(searchText)
        }
    }

    public init() {}

    public var body: some View {
        NavigationStack {
            ZStack {
                // Ivory / Warm Beige background
                Color(red: 0.968, green: 0.953, blue: 0.925)
                    .ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        // Introduction banner
                        VStack(alignment: .leading, spacing: 6) {
                            Text("असंयुत हस्ताः • Single Hand Gestures")
                                .font(.system(size: 13, weight: .bold, design: .serif))
                                .foregroundColor(Color(red: 0.72, green: 0.35, blue: 0.22)) // Terracotta

                            Text("Bharatanatyam Mudra Compendium")
                                .font(.system(size: 24, weight: .bold, design: .serif))
                                .foregroundColor(Color(red: 0.17, green: 0.12, blue: 0.09)) // Deep Brown

                            Text("Derived from the foundational treatise Abhinaya Darpana, each mudra conveys distinct expressive symbolism and geometric precision.")
                                .font(.system(size: 13.5, weight: .regular))
                                .foregroundColor(Color(red: 0.38, green: 0.32, blue: 0.28))
                        }
                        .padding(.horizontal)
                        .padding(.top, 8)

                        // Cards list
                        LazyVStack(spacing: 14) {
                            ForEach(filteredMudras) { mudra in
                                mudraCard(for: mudra)
                            }
                        }
                        .padding(.horizontal)
                        .padding(.bottom, 24)
                    }
                }
            }
            .searchable(text: $searchText, prompt: "Search mudras, meanings...")
            .navigationTitle("Mudra Dictionary")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(Color(red: 0.36, green: 0.45, blue: 0.28))
                }
            }
        }
    }

    private func mudraCard(for mudra: MudraType) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 12) {
                // Mudra icon badge
                ZStack {
                    Circle()
                        .fill(Color(red: 0.36, green: 0.45, blue: 0.28).opacity(0.12))
                        .frame(width: 44, height: 44)

                    Image(systemName: mudra.iconSymbol)
                        .font(.system(size: 20))
                        .foregroundColor(Color(red: 0.36, green: 0.45, blue: 0.28))
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(mudra.rawValue)
                            .font(.system(size: 19, weight: .bold, design: .serif))
                            .foregroundColor(Color(red: 0.17, green: 0.12, blue: 0.09))

                        Text("•")
                            .foregroundColor(Color.gray.opacity(0.5))

                        Text(mudra.sanskritTitle)
                            .font(.system(size: 17, weight: .semibold, design: .serif))
                            .foregroundColor(Color(red: 0.72, green: 0.35, blue: 0.22))
                    }

                    Text(mudra.englishMeaning)
                        .font(.system(size: 13.5, weight: .medium, design: .serif))
                        .foregroundColor(Color(red: 0.45, green: 0.38, blue: 0.32))
                }

                Spacer()
            }

            Text(mudra.description)
                .font(.system(size: 13.5, weight: .regular))
                .foregroundColor(Color(red: 0.28, green: 0.22, blue: 0.18))
                .lineSpacing(2)

            // Distinctive recognition feature tag
            HStack(spacing: 6) {
                Image(systemName: "sparkle")
                    .font(.system(size: 10))
                    .foregroundColor(Color(red: 0.78, green: 0.54, blue: 0.21))

                Text(keyVisualFeature(for: mudra))
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundColor(Color(red: 0.50, green: 0.42, blue: 0.30))
            }
            .padding(.top, 2)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(red: 0.99, green: 0.98, blue: 0.96))
                .shadow(color: Color.black.opacity(0.06), radius: 8, x: 0, y: 3)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color(red: 0.88, green: 0.85, blue: 0.80), lineWidth: 1.0)
        )
    }

    private func keyVisualFeature(for mudra: MudraType) -> String {
        switch mudra {
        case .pataka:
            return "All 4 fingers parallel and extended; thumb slightly bent."
        case .tripataka:
            return "Ring finger bent at PIP joint; other fingers fully straight."
        case .ardhapataka:
            return "Ring and little fingers bent down; index and middle straight."
        case .kartarimukha:
            return "Wide V-spread between index and middle; ring/little touch thumb."
        case .mayura:
            return "Thumb tip touches ring finger tip; index/middle/little extended."
        case .ardhachandra:
            return "Thumb abducted wide at ~90°; fingers straight together."
        case .mushti:
            return "All 4 fingers curled tightly into palm; thumb wrapped over."
        case .shikhara:
            return "Fist curled with thumb pointing vertically upward."
        case .suchi:
            return "Index finger straight up (needle); all other fingers curled."
        case .kapittha:
            return "Index finger curved over thumb tip; middle/ring/little curled."
        }
    }
}

