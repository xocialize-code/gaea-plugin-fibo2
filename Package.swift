// swift-tools-version: 6.2
import PackageDescription

// gaea-plugin-fibo2: Bria's Fibo 2.0 Dev (CC BY-NC 4.0 weights) as a Gaea research plugin. One worker executable built
// on GaeaPluginKit over fibo2-swift, packaged as a signed .gaeaplugin by scripts/build-plugin.sh.
//
// fibo2-inproc is not part of the plugin: it runs the same package in-process through MLXServeEngine, with the same
// dependency resolution, lane configuration and request, as the reference a run through Gaea is compared against.
//
// GAEA_PLUGIN_KIT_PATH points the kit at a local checkout while you work on both; FIBO2_SWIFT_PATH does the same for
// fibo2-swift.
let kitPath = Context.environment["GAEA_PLUGIN_KIT_PATH"]
let fibo2Path = Context.environment["FIBO2_SWIFT_PATH"]

let package = Package(
    name: "gaea-plugin-fibo2",
    platforms: [.macOS(.v26)], // mlx-engine-swift's MLXToolKit requires macOS 26
    dependencies: [
        kitPath.map { .package(path: $0) }
            ?? .package(url: "https://github.com/xocialize-code/gaea-plugin-kit.git", from: "0.15.0"),
        fibo2Path.map { .package(path: $0) }
            ?? .package(url: "https://github.com/xocialize/fibo2-swift.git", exact: "0.1.1"),
        // Exactly the engine GaeaPluginKit 0.15.0 and Gaea 1.9.0 are on (contract 1.65.0). A later 0.x could add fields
        // to the worker's memory report that GPP v1's closed schema doesn't carry, or a contract Gaea can't decode.
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", exact: "0.80.0"),
        // The fleet's mlx-swift fork, as mlx-engine-swift takes it: 0.32.4 is upstream 0.32.3 plus the
        // _updateInternal leak fix (ml-explore/mlx-swift#501; AB-D-0208). fibo2-swift names ml-explore's URL; this
        // root line wins.
        .package(url: "https://github.com/xocialize/mlx-swift", exact: "0.32.4"),
    ],
    targets: [
        .target(
            name: "Fibo2Plugin",
            dependencies: [
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
                .product(name: "MLXFibo2", package: "fibo2-swift"),
                .product(name: "Fibo2", package: "fibo2-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ]
        ),
        .executableTarget(
            name: "gaea-fibo2-worker",
            dependencies: [
                "Fibo2Plugin",
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
            ]
        ),
        .executableTarget(
            name: "fibo2-inproc",
            dependencies: [
                "Fibo2Plugin",
                .product(name: "MLXFibo2", package: "fibo2-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "Fibo2PluginTests",
            dependencies: ["Fibo2Plugin", .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
                           .product(name: "GaeaProtocol", package: "gaea-plugin-kit"),
                           .product(name: "MLXToolKit", package: "mlx-engine-swift")]
        ),
    ]
)
