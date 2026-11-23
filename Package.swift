// swift-tools-version: 6.2
import PackageDescription

// gaea-plugin-fibo2: Bria's Fibo 2.0 Dev, Edit Turbo and RMBG 3.0 (CC BY-NC 4.0 weights) as a Gaea research plugin. One worker executable built
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
            ?? .package(url: "https://github.com/xocialize/fibo2-swift.git", exact: "0.3.0"),
        // mlx-engine-swift 0.82.1, the first on the fleet's mlx-swift 0.33.1 (contract 1.66.0, which only adds imageLocate
        // to the 1.65.0 Gaea 1.9 decodes). Pinned exactly: a later 0.x could add fields to the worker's memory report
        // that GPP v1's closed schema doesn't carry, or a contract Gaea can't decode.
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", exact: "0.82.1"),
        // The fleet's mlx-swift fork, as mlx-engine-swift 0.82.1 takes it: 0.33.1 is upstream 0.33.0 with the vendored
        // MLX's head-dim-split SDPA routes off, which abort or give wrong attention on M5 (AB-D-0264, AB-R-1009). It
        // replaces the 0.32.4 leak-fix fork (AB-D-0208), whose fix upstream 0.33.0 carries. A root line naming
        // ml-explore's URL would win and drop the patch.
        .package(url: "https://github.com/xocialize/mlx-swift", exact: "0.33.1"),
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
