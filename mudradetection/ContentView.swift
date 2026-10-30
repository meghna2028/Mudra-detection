import SwiftUI

public struct ContentView: View {
    public init() {}

    public var body: some View {
        MudraDetectionView()
            .preferredColorScheme(.light)
    }
}

#Preview {
    ContentView()
}
