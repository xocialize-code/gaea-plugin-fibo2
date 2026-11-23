import Fibo2Plugin
import Foundation
import MLXFibo2
import MLXServeCore
import MLXToolKit

// The in-process reference for the plugin: one lane of a Fibo 2.0 model loaded by MLXServeEngine in this process, with the
// lane configuration the worker uses (Fibo2Plugin) and the request it would build (Fibo2Request, after the same size
// fitting). The first image is written to --out, and one JSON line of timings is printed. Not part of the plugin
// bundle.
//
//   fibo2-inproc [--model fibo-2-dev | fibo-2-edit-turbo | rmbg-3] [--lane bf16 | int8 | int4] [--prompt <text | @file>]
//                [--negative <text | @file>] [--width 1024] [--height 1024] [--steps <the model's>] [--guidance 5]
//                [--seed 42] [--no-text-images] [--image <png> ...] [--models-root <dir>] [--out <png>] [--runs 1]
//
// --image (repeatable, 1–5) makes it an image.edit with the prompt as the instruction; without --width and --height
// the package draws at the size trained for the first image's aspect ratio, as in the worker.
// With --model rmbg-3 one --image makes it an image.matte: --seed (default 0) and --binary are its only options, and
// --out gets the grey matte.

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
let modelName = option("--model") ?? Model.dev.id
guard let model = Model.allCases.first(where: { $0.id == modelName }) else { fail("unknown --model \(modelName)") }
let quant = Quant(rawValue: option("--lane") ?? defaultQuant.rawValue) ?? defaultQuant
let prompt = text(option("--prompt") ?? "A photorealistic lighthouse on a rocky coast at dusk, warm light in the lantern room")
let negative = option("--negative").map(text)
let width = Int(option("--width") ?? ""), height = Int(option("--height") ?? "")
let steps = Int(option("--steps") ?? "") ?? model.defaultSteps
let guidance = model.guided ? Double(option("--guidance") ?? "") ?? model.defaultGuidance : model.defaultGuidance
let isMatte = model == .rmbg
let seed = UInt64(option("--seed") ?? "") ?? (isMatte ? 0 : 42)
let binary = flag("--binary")
let textImages = !flag("--no-text-images")
let imagePaths = options("--image")
let outPath = option("--out")
let runs = Int(option("--runs") ?? "") ?? 1
let modelsRoot = URL(fileURLWithPath: option("--models-root")
                     ?? NSHomeDirectory() + "/Library/Application Support/MLXEngine/Models", isDirectory: true)
let isEdit = !imagePaths.isEmpty
guard imagePaths.count <= maxImages else { fail("--image takes 1–\(maxImages) images") }
guard isEdit || model.generates else { fail("\(model.title) needs --image") }
guard !isMatte || imagePaths.count == 1 else { fail("\(model.title) mattes one --image") }

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

var result: [String: Any] = ["model": model.id, "lane": quant.rawValue, "steps": steps, "guidance": guidance, "seed": seed,
                             "images": imagePaths.count]

// The budget Gaea sends a worker on a large Mac (wiresWeights off, the automatic pool cap).
let engine = MLXServeEngine(workerBudget: WorkerBudget(ceilingBytes: 64_000_000_000, wiresWeights: false))
await engine.useModelStore(ModelStore(root: modelsRoot))
let id = PackageID(model.id)
let registration: PackageRegistration
switch model {
case .dev: registration = Fibo2DevPackage.registration
case .editTurbo: registration = Fibo2EditTurboPackage.registration
case .rmbg: registration = Fibo2RMBGPackage.registration
}
_ = try await engine.register(registration, configuration: configuration(model, quant: quant), id: id)

var clock = ContinuousClock.now
_ = try await engine.prepare(isMatte ? .matting : isEdit ? .imageEdit : .textToImage, package: id)
result["prepare_s"] = seconds(since: clock)

let images = try imagePaths.map { path in
    Image(format: path.lowercased().hasSuffix(".png") ? .png : .jpeg, data: try Data(contentsOf: URL(fileURLWithPath: path)))
}
// The worker's request, built by the same function (Fibo2Plugin's generate and edit).
let engineRequest: any CapabilityRequest
if isMatte {
    engineRequest = MatteCall(kind: binary ? .binary : .softAlpha, seed: seed).request(images[0])
} else if isEdit {
    engineRequest = request.edit(images)
} else {
    engineRequest = request.generate()
}

for run in 1...max(1, runs) {
    clock = ContinuousClock.now
    let response = try await engine.run(engineRequest, package: id)
    result["run\(run)_s"] = seconds(since: clock)
    let image = (response as? T2IResponse)?.image ?? (response as? IEditResponse)?.image
    if let image { result["width"] = image.width; result["height"] = image.height }
    if run == 1, let image, let outPath { try image.data.write(to: URL(fileURLWithPath: outPath)) }
    if let matte = (response as? MattingResponse)?.matte {
        result["width"] = matte.width; result["height"] = matte.height; result["kind"] = matte.kind.rawValue
        if run == 1, let outPath { try matte.data.write(to: URL(fileURLWithPath: outPath)) }
    }
}
result["total_s"] = seconds(since: started)

let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
