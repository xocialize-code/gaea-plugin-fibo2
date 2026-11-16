import Fibo2Plugin
import Foundation
import MLXFibo2
import MLXServeCore
import MLXToolKit

// The in-process reference for the plugin: one Fibo 2.0 Dev lane loaded by MLXServeEngine in this process, with the
// lane configuration the worker uses (Fibo2Plugin) and the request it would build (Fibo2Request, after the same size
// fitting). The first image is written to --out, and one JSON line of timings is printed. Not part of the plugin
// bundle.
//
//   fibo2-inproc [--lane bf16 | int8 | int4] [--prompt <text | @file>] [--negative <text | @file>] [--width 1024]
//                [--height 1024] [--steps 50] [--guidance 5] [--seed 42] [--no-text-images] [--image <png> ...]
//                [--models-root <dir>] [--out <png>] [--runs 1]
//
// --image (repeatable, 1–5) makes it an image.edit with the prompt as the instruction; without --width and --height
// the package draws at the size trained for the first image's aspect ratio, as in the worker.

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i...(i + 1)) }
    return args[i + 1]
}
func options(_ name: String) -> [String] {
    var values: [String] = []
    while let value = option(name) { values.append(value) }
    return values
}
func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}
func seconds(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}
func text(_ value: String) -> String {
    guard value.hasPrefix("@") else { return value }
    guard let s = try? String(contentsOfFile: String(value.dropFirst()), encoding: .utf8) else { fail("can't read \(value)") }
    return s.trimmingCharacters(in: .whitespacesAndNewlines)
}

let started = ContinuousClock.now
let quant = Quant(rawValue: option("--lane") ?? defaultQuant.rawValue) ?? defaultQuant
let prompt = text(option("--prompt") ?? "A photorealistic lighthouse on a rocky coast at dusk, warm light in the lantern room")
let negative = option("--negative").map(text)
let width = Int(option("--width") ?? ""), height = Int(option("--height") ?? "")
let steps = Int(option("--steps") ?? "") ?? defaultSteps
let guidance = Double(option("--guidance") ?? "") ?? defaultGuidance
let seed = UInt64(option("--seed") ?? "") ?? 42
let textImages = !flag("--no-text-images")
let imagePaths = options("--image")
let outPath = option("--out")
let runs = Int(option("--runs") ?? "") ?? 1
let modelsRoot = URL(fileURLWithPath: option("--models-root")
                     ?? NSHomeDirectory() + "/Library/Application Support/MLXEngine/Models", isDirectory: true)
let isEdit = !imagePaths.isEmpty
guard imagePaths.count <= maxImages else { fail("--image takes 1–\(maxImages) images") }

// The worker's size: the asked one fitted to the envelope; else 1024² (generate) or the package's choice (edit).
var size: (Int?, Int?) = (nil, nil)
if let width, let height {
    let fitted = try plan((width, height))
    size = (fitted.width, fitted.height)
} else if !isEdit {
    size = (1024, 1024)
}
let request = Fibo2Request(prompt: prompt, negative: negative, width: size.0, height: size.1, steps: steps,
                           guidance: guidance, seed: seed, textImages: textImages)

var result: [String: Any] = ["model": modelID, "lane": quant.rawValue, "steps": steps, "guidance": guidance, "seed": seed,
                             "images": imagePaths.count]

// The budget Gaea sends a worker on a large Mac (wiresWeights off, the automatic pool cap).
let engine = MLXServeEngine(workerBudget: WorkerBudget(ceilingBytes: 64_000_000_000, wiresWeights: false))
await engine.useModelStore(ModelStore(root: modelsRoot))
let id = PackageID(modelID)
_ = try await engine.register(Fibo2DevPackage.registration, configuration: configuration(quant: quant), id: id)

var clock = ContinuousClock.now
_ = try await engine.prepare(isEdit ? .imageEdit : .textToImage, package: id)
result["prepare_s"] = seconds(since: clock)

let images = try imagePaths.map { path in
    Image(format: path.lowercased().hasSuffix(".png") ? .png : .jpeg, data: try Data(contentsOf: URL(fileURLWithPath: path)))
}
// The worker's request, built by the same function (Fibo2Plugin's generate and edit).
let engineRequest: any CapabilityRequest = isEdit ? request.edit(images) : request.generate()

for run in 1...max(1, runs) {
    clock = ContinuousClock.now
    let response = try await engine.run(engineRequest, package: id)
    result["run\(run)_s"] = seconds(since: clock)
    let image = (response as? T2IResponse)?.image ?? (response as? IEditResponse)?.image
    if let image { result["width"] = image.width; result["height"] = image.height }
    if run == 1, let image, let outPath { try image.data.write(to: URL(fileURLWithPath: outPath)) }
}
result["total_s"] = seconds(since: started)

let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
