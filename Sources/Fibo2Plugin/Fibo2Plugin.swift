import Fibo2
import Foundation
import GaeaPluginKit
import GaeaProtocol
import ImageIO
import MLXFibo2
import MLXToolKit

// gaea-plugin-fibo2: Bria's Fibo 2.0 Dev (CC BY-NC 4.0 weights, MIT port) as a Gaea research plugin, on fibo2-swift
// through MLXServeEngine (AB-T-0493).
//
// Two models on one 8.1B DiT architecture with Qwen3-VL-4B:
//   - fibo-2-dev: text-to-image and instruction editing of 1–5 images. The full base model: 50 steps with
//     classifier-free guidance 5, two DiT passes a step.
//   - fibo-2-edit-turbo (0.2.0): instruction editing only, distilled from Dev to 4 steps without guidance.
// Both read structured JSON captions (Bria's VGL); a short description or a plain instruction works too.
//
// The weights are non-commercial, so Gaea makes the models named-only and opt-in by itself; the offers say named-only
// too (AB-D-0226: a plugin may carry such a model, with its terms shown). Three lanes each, one per DiT quant, all made
// at load from Bria's own gated snapshots; bf16 is the default. Weights come from the models root Gaea sends in
// configure; the engine materializes a missing snapshot from Bria's repo with a Hugging Face token whose account has
// access. Nothing is mirrored. The two repos share their text encoder, VAE and fonts byte for byte, so Edit Turbo
// fetches only its transformer when Dev is installed.

/// The package's envelope (`Fibo2Envelope`): about one megapixel, the area the model was trained at.
public let maxPixels = Fibo2Envelope.maxOutputPixels, minPixels = Fibo2Envelope.minOutputPixels
public let minSide = Fibo2Envelope.minSide, grid = 16, maxImages = Fibo2Envelope.maxImages

/// The plugin's models: the package's two checkpoints.
public typealias Model = Fibo2Variant

extension Fibo2Variant {
    /// The MES model id, which is the package's name.
    public var id: String { packageName }
    /// image.generate is Dev's alone: Edit Turbo is distilled for editing.
    public var generates: Bool { self == .dev }
    /// Dev runs with guidance; Edit Turbo has none, so it takes no guidance and no negative prompt.
    public var guided: Bool { self == .dev }
    /// `x-fibo2.steps`: Dev 1–100 (the card's 50), Edit Turbo 1–16 (distilled for 4).
    public var stepRange: ClosedRange<Int> { self == .dev ? 1...100 : 1...16 }
}

public let modelID = Model.dev.id, modelTitle = Model.dev.title
/// The model card's settings for the full base model.
public let defaultSteps = Model.dev.defaultSteps
public let defaultGuidance = Model.dev.defaultGuidance

/// The lanes, default first: bf16, int8, and int4 (FFN up-projections at 4 bits, the rest of each block at 8).
public let quants: [Quant] = [.bf16, .int8, .int4]
public let defaultQuant = Quant.bf16

/// The package configuration for one lane of a model. fibo2-inproc builds the same one.
public func configuration(_ model: Model = .dev, quant: Quant) -> Fibo2Configuration {
    Fibo2Configuration(variant: model, quant: quant)
}

/// The MES size a job asks for (`size`, or `aspect` with `megapixels`, Image profile §2); nil when it asks for none.
func requestedSize(_ job: Job) throws -> (Int, Int)? {
    if let size = job.param("size") {
        guard let w = size["width"]?.intValue, let h = size["height"]?.intValue, w > 0, h > 0 else {
            throw GPPError(.invalidRequest, "size is {width, height} in pixels.")
        }
        return (w, h)
    }
    if let aspect = job.param("aspect")?.stringValue {
        let parts = aspect.split(separator: ":").compactMap { Double($0) }
        guard parts.count == 2, parts[0] > 0, parts[1] > 0 else { throw GPPError(.invalidRequest, "aspect is W:H, e.g. 16:9.") }
        let pixels = (job.param("megapixels")?.doubleValue ?? 1.0) * 1_000_000
        let height = (pixels * parts[1] / parts[0]).squareRoot()
        return (Int(height * parts[0] / parts[1]), Int(height))
    }
    return nil
}

/// A PNG or JPEG's pixel size as the model sees it, with the EXIF orientation applied; nil when ImageIO can't read it.
public func orientedPixelSize(of url: URL) -> (Int, Int)? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? Int,
          let height = properties[kCGImagePropertyPixelHeight] as? Int else { return nil }
    let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
    return orientation >= 5 ? (height, width) : (width, height)
}

public struct SizePlan: Sendable {
    public var width: Int, height: Int
    public var warning: Warning?
}

/// Fits an asked size to the envelope: scaled down to 1.1 × 1024² of area keeping the aspect, each side floored to a
/// multiple of 16. A change is reported as `size_snapped`. The package refuses rather than clamps, so the worker fits
/// it first.
public func plan(_ size: (Int, Int)) throws -> SizePlan {
    var (w, h) = (Double(size.0), Double(size.1))
    if w * h > Double(maxPixels) {
        let scale = (Double(maxPixels) / (w * h)).squareRoot()
        w *= scale
        h *= scale
    }
    let width = Int(w) / grid * grid, height = Int(h) / grid * grid
    guard min(width, height) >= minSide, width * height >= minPixels else {
        throw GPPError(.invalidRequest, "Fibo 2.0 Dev makes at least \(minSide) px a side and 256² of area, at most about "
            + "1.1 megapixels (it was trained at one); \(size.0)×\(size.1) doesn't fit.", detail: ["param": "size"])
    }
    let warning = (width, height) == size ? nil : Warning(
        code: "size_snapped", message: "Made \(width)×\(height) for \(size.0)×\(size.1): multiples of 16, at most about "
            + "1.1 megapixels, the area Fibo 2.0 was trained at.", param: "size")
    return SizePlan(width: width, height: height, warning: warning)
}

/// What one engine call carries, built the same way by the worker and fibo2-inproc so the two send byte-identical
/// requests.
public struct Fibo2Request: Sendable {
    public var prompt: String
    /// nil = the package's default: the model card's text-to-image negative, or the null caption for an edit.
    public var negative: String?
    /// nil (an edit without a size) = the size Fibo 2.0 was trained at for the first image's aspect ratio.
    public var width: Int?
    public var height: Int?
    public var steps: Int
    public var guidance: Double
    public var seed: UInt64
    /// `Fibo2MetaKeys.textImages`: false skips drawing the caption's texts for Qwen3-VL.
    public var textImages: Bool

    public init(prompt: String, negative: String? = nil, width: Int?, height: Int?, steps: Int = defaultSteps,
                guidance: Double = defaultGuidance, seed: UInt64, textImages: Bool = true) {
        self.prompt = prompt
        self.negative = negative
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.textImages = textImages
    }

    /// The package's per-request keys, only those the request sets.
    public var metaData: MetaData { textImages ? [:] : [Fibo2MetaKeys.textImages: .bool(false)] }

    /// Image `index` of an image.generate, with seed + index.
    public func generate(index: Int = 0) -> T2IRequest {
        T2IRequest(prompt: prompt, negativePrompt: negative, width: width, height: height, steps: steps,
                   guidanceScale: guidance, seed: seed + UInt64(index), metaData: metaData)
    }

    /// An image.edit of `images`, the prompt as the instruction.
    public func edit(_ images: [Image]) -> IEditRequest {
        IEditRequest(images: images, prompt: prompt, negativePrompt: negative, width: width, height: height, steps: steps,
                     guidanceScale: guidance, seed: seed, metaData: metaData)
    }
}

/// The run's settings shared by generate and edit, and the warnings for what the model ignores.
struct Shared {
    var steps: Int
    var guidance: Double
    var negative: String?
    var textImages = true
    var seed: UInt64
    var count: Int
    var warnings: [Warning] = []

    /// `counted`: image.generate takes MES `count`; image.edit has one output and no count.
    init(_ job: Job, model: Model = .dev, counted: Bool = true) throws {
        steps = job.param("x-fibo2.steps")?.intValue ?? model.defaultSteps
        guard model.stepRange.contains(steps) else {
            throw GPPError(.invalidRequest, "x-fibo2.steps is \(model.stepRange.lowerBound)–\(model.stepRange.upperBound) for \(model.title).")
        }
        // Edit Turbo's offer has no guidance param; Gaea passes a worker only the ones its offer names.
        guidance = model.guided ? job.param("x-fibo2.guidance")?.doubleValue ?? model.defaultGuidance : model.defaultGuidance
        guard (1...10).contains(guidance) else { throw GPPError(.invalidRequest, "x-fibo2.guidance is 1–10 (1 turns guidance off).") }
        if let value = job.param("x-fibo2.text_images") {
            guard case .bool(let b) = value else { throw GPPError(.invalidRequest, "x-fibo2.text_images is true or false.") }
            textImages = b
        }
        count = counted ? job.param("count")?.intValue ?? 1 : 1
        guard (1...4).contains(count) else { throw GPPError(.invalidRequest, "count is 1–4.") }
        if let text = job.param("negative_prompt")?.stringValue, !text.isEmpty {
            if guidance > 1 {
                negative = text
            } else {
                warnings.append(Warning(code: "negative_prompt_ignored",
                                        message: model.guided
                                            ? "With x-fibo2.guidance 1 there is no guidance pass, so a negative prompt does nothing."
                                            : "\(model.title) runs without guidance, so a negative prompt does nothing.",
                                        param: "negative_prompt"))
            }
        }
        if job.param("prompt_enhance")?.stringValue == "on" {
            warnings.append(Warning(code: "prompt_enhance_ignored",
                                    message: "No prompt-enhance profile writes Fibo 2.0's structured captions yet; the prompt goes as typed.",
                                    param: "prompt_enhance"))
        }
        // Image profile §2: a request without a seed still gets one, reported in effective.
        seed = job.seed ?? UInt64.random(in: 0..<(1 << 31))
    }

    func request(prompt: String, width: Int?, height: Int?) -> Fibo2Request {
        Fibo2Request(prompt: prompt, negative: negative, width: width, height: height, steps: steps, guidance: guidance,
                     seed: seed, textImages: textImages)
    }

    func effective(width: Int, height: Int, model: Model = .dev, counted: Bool = true) -> [String: JSONValue] {
        var values: [String: JSONValue] = [
            "size": ["width": .int(Int64(width)), "height": .int(Int64(height))],
            "seed": .int(Int64(seed)), "x-fibo2.steps": .int(Int64(steps)), "x-fibo2.text_images": .bool(textImages),
        ]
        if model.guided { values["x-fibo2.guidance"] = .double(guidance) }
        if counted { values["count"] = .int(Int64(count)) }
        return values
    }
}

/// MES image.generate: one engine call per image, each with seed + i.
func generate() -> EngineOperation {
    EngineOperation("image.generate", outputNames: ["image": "images"]) { job in
        guard let prompt = try job.text("prompt"), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GPPError(.invalidRequest, "image.generate needs a prompt.")
        }
        let size = try plan(try requestedSize(job) ?? (1024, 1024))
        var shared = try Shared(job)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        let request = shared.request(prompt: prompt, width: size.width, height: size.height)
        let effective = shared.effective(width: size.width, height: size.height)
        return (0..<shared.count).map { i in
            EngineCall(request.generate(index: i), effective: effective, warnings: shared.warnings)
        }
    }
}

/// MES image.edit: an instruction edit of 1–5 images, named in the instruction in order (<image_1>, <image_2>). One
/// output, drawn from noise. Without a size it is the size Fibo 2.0 was trained at for the first image's aspect ratio
/// (about one megapixel), so the first image is cropped to that ratio, not stretched.
func edit(_ model: Model = .dev) -> EngineOperation {
    EngineOperation("image.edit", outputNames: ["image": "image"]) { job in
        guard let instruction = try job.text("instruction") ?? job.text("prompt"), !instruction.isEmpty else {
            throw GPPError(.invalidRequest, "image.edit needs an instruction.")
        }
        let name = job.inputs["images"] == nil ? "image" : "images"
        let references = job.inputs[name] ?? []
        guard (1...maxImages).contains(references.count) else {
            throw GPPError(.invalidRequest, "image.edit takes 1–\(maxImages) images; got \(references.count).",
                           detail: ["param": "images"])
        }
        if let problem = Fibo2Prompt.editPromptProblem(instruction) {
            throw GPPError(.invalidRequest, "\(model.title): \(problem).", detail: ["param": "instruction"])
        }
        var shared = try Shared(job, model: model, counted: false)
        var width: Int?, height: Int?
        let made: (Int, Int)
        if let asked = try requestedSize(job) {
            let size = try plan(asked)
            if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
            (width, height, made) = (size.width, size.height, (size.width, size.height))
        } else {
            guard let input = orientedPixelSize(of: references[0]) else {
                throw GPPError(.invalidRequest, "image.edit needs readable PNG or JPEG images.")
            }
            let trained = Fibo2ImagePrep.imageSize(aspectRatio: Double(input.0) / Double(input.1))
            made = (trained.width, trained.height)
            if made != input {
                shared.warnings.insert(Warning(
                    code: "size_snapped",
                    message: "Made \(made.0)×\(made.1) for a \(input.0)×\(input.1) image: Fibo 2.0 draws at the size it was "
                        + "trained at for the image's aspect ratio, about one megapixel.", param: "size"), at: 0)
            }
        }
        if job.param("strength") != nil {
            shared.warnings.append(Warning(code: "param_ignored",
                                           message: "Fibo 2.0 edits by conditioning on the image from noise, so strength does nothing.",
                                           param: "strength"))
        }
        let images = try references.indices.map { try job.image(name, index: $0) }
        let request = shared.request(prompt: instruction, width: width, height: height)
        return [EngineCall(request.edit(images),
                           effective: shared.effective(width: made.0, height: made.1, model: model, counted: false),
                           warnings: shared.warnings)]
    }
}

public func lanes(_ model: Model = .dev) -> [LaneDefinition] {
    quants.map { quant in
        LaneDefinition(id: quant.rawValue, default: quant == defaultQuant, configuration: configuration(model, quant: quant))
    }
}

/// CC BY-NC 4.0 (the model card's `bria-fibo` licence): non-commercial. The notice names the terms and the gate, as
/// pixel-agency-normalizer's `ImageNormalizer.notice` does for the other restricted models.
/// The licence binds each user (non-commercial use, with attribution), so a client's user is shown the notice and
/// accepts it too (`user_accept_required`, as AuK's Qwen Research terms: G-27, AB-D-0209), and Gaea carries the
/// attribution to clients (MES License.attribution; AB-A-0297).
public let attribution = "Made with Fibo 2.0 by Bria AI (CC BY-NC 4.0)"
let fiboLicence: JSONValue = [
    "spdx": "CC-BY-NC-4.0", "commercial_use": false, "accepted_required": true, "user_accept_required": true,
    "terms_url": "https://creativecommons.org/licenses/by-nc/4.0/legalcode",
    "attribution": .string(attribution),
    "notice": .string("Fibo 2.0 weights © Bria AI, CC BY-NC 4.0: non-commercial use only, with attribution. Commercial "
        + "use needs a licence from Bria (platform.bria.ai/licensing/pricing?model=fibo-2). The weights are gated: the "
        + "Hugging Face account behind your token needs access to the model's repo (briaai/Fibo-2.0-dev, "
        + "briaai/Fibo-2.0-edit-turbo)."),
]
let limits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, sizeMultiple: grid)
let editLimits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, sizeMultiple: grid, maxImages: maxImages)
/// Fibo 2.0's edit redraws the picture at the size it was trained at for the first image's aspect ratio, which crops
/// that image to the ratio. So it can't paint a marked region of a picture in place: Gaea brings a fill's result back
/// to the picture's size and composites it through the mask, and a cropped result lands shifted (GPP-1 §4.3, as Gaea
/// widened it for this case; AB-A-0297). Gaea reads this and keeps both models off image.fill and image.extend.
/// image.edit itself is unaffected. Drop it if the package ever returns the whole frame at the size asked.
let imageToImage = ["image_to_image": true]

/// The MES extensions a model's offers name. Edit Turbo has no guidance.
func params(_ model: Model) -> JSONValue {
    var properties: [String: JSONValue] = [
        "x-fibo2.steps": [
            "type": "integer", "minimum": .int(Int64(model.stepRange.lowerBound)), "maximum": .int(Int64(model.stepRange.upperBound)),
            "default": .int(Int64(model.defaultSteps)),
            "description": .string(model == .dev
                ? "Denoising steps, two DiT passes each. The model card uses 50; 30 is faster and close."
                : "Denoising steps, one DiT pass each. The model is distilled for 4."),
            "x-mes-ui": ["widget": "slider", "label": "Steps", "advanced": true]],
        "x-fibo2.text_images": [
            "type": "boolean", "default": true,
            "description": .string("Draw each text the caption's text_render asks for and show it to the text encoder, as "
                + "the model's training did. It helps the model spell the text."),
            "x-mes-ui": ["widget": "toggle", "label": "Text images", "advanced": true],
        ],
    ]
    if model.guided {
        properties["x-fibo2.guidance"] = [
            "type": "number", "minimum": 1, "maximum": 10, "default": .double(model.defaultGuidance),
            "description": "Classifier-free guidance. 1 turns it off: one pass a step, half the time, weaker results.",
            "x-mes-ui": ["widget": "slider", "label": "Guidance", "advanced": true]]
    }
    return ["type": "object", "additionalProperties": false, "properties": .object(properties)]
}

let captionNote = "The model reads structured JSON captions (Bria's VGL, docs.bria.ai/vgl): short_description, objects, "
    + "background_setting, lighting, aesthetics, photographic_characteristics, style_medium, and text_render "
    + "[{text, location, …}] for text to draw. A short plain description works too, with less control. "

let editNote = "Instruction editing of 1–5 images, named in order as <image_1>, <image_2>. The instruction goes as typed, or "
    + "as JSON with edit_instruction and the result's caption under structured_caption. Drawn from noise at the size "
    + "trained for the first image's aspect ratio (about one megapixel, cropped to that ratio); strength does nothing. "

func offerModel(_ model: Model, edit: Bool) -> JSONValue {
    let use: String
    switch (model, edit) {
    case (.dev, true):
        use = "The full base model: two passes a step. " + editNote
            + "About 6 minutes with one image at 50 steps on an M5 Max, 11 with two, 32 with five; Fibo 2.0 Edit Turbo "
            + "does the same edit in 4 steps. "
    case (.dev, false):
        use = "The full base model: two passes a step. Text-to-image at about one megapixel (1024×1024, 1424×800, "
            + "1248×832): 50 steps with guidance 5, about 3 minutes on an M5 Max. A negative prompt is honoured; "
            + "without one the model card's is used. " + captionNote
    case (.editTurbo, _):
        use = "Distilled from Fibo 2.0 Dev for editing: 4 steps without guidance, one pass a step, so a negative prompt "
            + "does nothing. " + editNote
    }
    return [
        "id": .string(model.id), "title": .string(model.title), "license": fiboLicence,
        "memory_class": "large", "speed_class": model == .dev ? "slow" : "standard",
        "quality_tiers": model == .dev ? ["best"] : ["draft", "standard"], "named_only": true,
        "notes": .string("Research plugin, named-only and opt-in (CC BY-NC 4.0: non-commercial). " + use),
    ]
}

public let fibo2Plugin = GaeaPluginDefinition(
    id: "com.xocialize.gaea.fibo2",
    version: "0.2.2",
    minGaea: "1.9.0",
    team: "CFZS6RXUCP",
    repo: "https://github.com/xocialize-code/gaea-plugin-fibo2",
    title: "Fibo 2.0",
    summary: "Bria's Fibo 2.0 (CC BY-NC 4.0: non-commercial; gated weights): an 8.1B DiT on Qwen3-VL-4B that reads "
        + "structured JSON captions. Fibo 2.0 Dev, the full 50-step base model, for text-to-image and instruction "
        + "editing of up to 5 images; Fibo 2.0 Edit Turbo for the same edits in 4 steps. Strong prompt adherence and "
        + "text rendering. A research plugin: named-only and opt-in.",
    license: "Apache-2.0",
    executable: "gaea-fibo2-worker",
    models: [
        .engine(Fibo2DevPackage.self, id: Model.dev.id, title: Model.dev.title, lanes: lanes(.dev),
                operations: [generate(), edit(.dev)]),
        .engine(Fibo2EditTurboPackage.self, id: Model.editTurbo.id, title: Model.editTurbo.title, lanes: lanes(.editTurbo),
                operations: [edit(.editTurbo)]),
    ],
    offers: [
        PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(.dev, edit: false),
                             params: params(.dev), defaults: ["count": 1], limits: limits),
        PluginManifest.Offer(operation: "image.edit", profile: "image", model: offerModel(.dev, edit: true),
                             params: params(.dev), limits: editLimits, supports: imageToImage),
        PluginManifest.Offer(operation: "image.edit", profile: "image", model: offerModel(.editTurbo, edit: true),
                             params: params(.editTurbo), limits: editLimits, supports: imageToImage),
    ])
