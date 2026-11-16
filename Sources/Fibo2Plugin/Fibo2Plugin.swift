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
// One model, fibo-2-dev: text-to-image and instruction editing of 1–5 images in one 8.1B DiT on Qwen3-VL-4B. It is the
// full base model: 50 steps with classifier-free guidance 5, two DiT passes a step. It reads structured JSON captions
// (Bria's VGL); a short description works too.
//
// The weights are non-commercial, so Gaea makes the model named-only and opt-in by itself; the offers say named-only
// too (AB-D-0226: a plugin may carry such a model, with its terms shown). Three lanes, one per DiT quant, all made at
// load from Bria's own gated snapshot; bf16 is the default. Weights come from the models root Gaea sends in configure;
// the engine materializes a missing snapshot from briaai/Fibo-2.0-dev with a Hugging Face token whose account has
// access. Nothing is mirrored.

/// The package's envelope (`Fibo2Envelope`): about one megapixel, the area the model was trained at.
public let maxPixels = Fibo2Envelope.maxOutputPixels, minPixels = Fibo2Envelope.minOutputPixels
public let minSide = Fibo2Envelope.minSide, grid = 16, maxImages = Fibo2Envelope.maxImages

public let modelID = "fibo-2-dev", modelTitle = "Fibo 2.0 Dev"
/// The model card's settings for the full base model.
public let defaultSteps = 50
public let defaultGuidance = 5.0

/// The lanes, default first: bf16, int8, and int4 (FFN up-projections at 4 bits, the rest of each block at 8).
public let quants: [Quant] = [.bf16, .int8, .int4]
public let defaultQuant = Quant.bf16

/// The package configuration for one lane. fibo2-inproc builds the same one.
public func configuration(quant: Quant) -> Fibo2Configuration {
    Fibo2Configuration(quant: quant, snapshotPath: "", defaultSteps: defaultSteps, defaultGuidance: defaultGuidance)
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
    init(_ job: Job, counted: Bool = true) throws {
        steps = job.param("x-fibo2.steps")?.intValue ?? defaultSteps
        guard (1...100).contains(steps) else { throw GPPError(.invalidRequest, "x-fibo2.steps is 1–100.") }
        guidance = job.param("x-fibo2.guidance")?.doubleValue ?? defaultGuidance
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
                                        message: "With x-fibo2.guidance 1 there is no guidance pass, so a negative prompt does nothing.",
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

    func effective(width: Int, height: Int, counted: Bool = true) -> [String: JSONValue] {
        var values: [String: JSONValue] = [
            "size": ["width": .int(Int64(width)), "height": .int(Int64(height))],
            "seed": .int(Int64(seed)), "x-fibo2.steps": .int(Int64(steps)), "x-fibo2.guidance": .double(guidance),
            "x-fibo2.text_images": .bool(textImages),
        ]
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
func edit() -> EngineOperation {
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
            throw GPPError(.invalidRequest, "Fibo 2.0 Dev: \(problem).", detail: ["param": "instruction"])
        }
        var shared = try Shared(job, counted: false)
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
        return [EngineCall(request.edit(images), effective: shared.effective(width: made.0, height: made.1, counted: false),
                           warnings: shared.warnings)]
    }
}

public func lanes() -> [LaneDefinition] {
    quants.map { quant in
        LaneDefinition(id: quant.rawValue, default: quant == defaultQuant, configuration: configuration(quant: quant))
    }
}

/// CC BY-NC 4.0 (the model card's `bria-fibo` licence): non-commercial. The notice names the terms and the gate, as
/// pixel-agency-normalizer's `ImageNormalizer.notice` does for the other restricted models.
let fiboLicence: JSONValue = [
    "spdx": "CC-BY-NC-4.0", "commercial_use": false, "accepted_required": true,
    "terms_url": "https://creativecommons.org/licenses/by-nc/4.0/legalcode",
    "notice": .string("Fibo 2.0 Dev weights © Bria AI, CC BY-NC 4.0: non-commercial use only, with attribution. Commercial "
        + "use needs a licence from Bria (platform.bria.ai/licensing/pricing?model=fibo-2). The weights are gated: the "
        + "Hugging Face account behind your token needs access to briaai/Fibo-2.0-dev."),
]
let limits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, sizeMultiple: grid)
let editLimits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, sizeMultiple: grid, maxImages: maxImages)

let params: JSONValue = [
    "type": "object", "additionalProperties": false,
    "properties": [
        "x-fibo2.steps": ["type": "integer", "minimum": 1, "maximum": 100, "default": .int(Int64(defaultSteps)),
                          "description": "Denoising steps, two DiT passes each. The model card uses 50; 30 is faster and close.",
                          "x-mes-ui": ["widget": "slider", "label": "Steps", "advanced": true]],
        "x-fibo2.guidance": ["type": "number", "minimum": 1, "maximum": 10, "default": .double(defaultGuidance),
                             "description": "Classifier-free guidance. 1 turns it off: one pass a step, half the time, weaker results.",
                             "x-mes-ui": ["widget": "slider", "label": "Guidance", "advanced": true]],
        "x-fibo2.text_images": [
            "type": "boolean", "default": true,
            "description": .string("Draw each text the caption's text_render asks for and show it to the text encoder, as "
                + "the model's training did. It helps the model spell the text."),
            "x-mes-ui": ["widget": "toggle", "label": "Text images", "advanced": true],
        ],
    ],
]

let captionNote = "The model reads structured JSON captions (Bria's VGL, docs.bria.ai/vgl): short_description, objects, "
    + "background_setting, lighting, aesthetics, photographic_characteristics, style_medium, and text_render "
    + "[{text, location, …}] for text to draw. A short plain description works too, with less control. "

func offerModel(edit: Bool) -> JSONValue {
    let use = edit
        ? "Instruction editing of 1–5 images, named in order as <image_1>, <image_2>. The instruction goes as typed, or as "
            + "JSON with edit_instruction and the result's caption under structured_caption. Drawn from noise at the "
            + "size trained for the first image's aspect ratio (about one megapixel, cropped to that ratio); strength "
            + "does nothing. About 6 minutes with one image at 50 steps on an M5 Max, 11 with two, 32 with five. "
        : "Text-to-image at about one megapixel (1024×1024, 1424×800, 1248×832): 50 steps with guidance 5, about "
            + "3 minutes on an M5 Max. A negative prompt is honoured; without one the model card's is used. " + captionNote
    return [
        "id": .string(modelID), "title": .string(modelTitle), "license": fiboLicence,
        "memory_class": "large", "speed_class": "slow",
        "quality_tiers": ["best"], "named_only": true,
        "notes": .string("Research plugin, named-only and opt-in (CC BY-NC 4.0: non-commercial). The full base model: two "
            + "passes a step. " + use),
    ]
}

public let fibo2Plugin = GaeaPluginDefinition(
    id: "com.xocialize.gaea.fibo2",
    version: "0.1.0",
    minGaea: "1.9.0",
    team: "CFZS6RXUCP",
    repo: "https://github.com/xocialize-code/gaea-plugin-fibo2",
    title: "Fibo 2.0 Dev",
    summary: "Bria's Fibo 2.0 Dev (CC BY-NC 4.0: non-commercial; gated weights): an 8.1B DiT on Qwen3-VL-4B that reads "
        + "structured JSON captions. Text-to-image and instruction editing of up to 5 images, with strong prompt "
        + "adherence and text rendering. The full 50-step base model, so it is slow. A research plugin: named-only and "
        + "opt-in.",
    license: "Apache-2.0",
    executable: "gaea-fibo2-worker",
    models: [
        .engine(Fibo2DevPackage.self, id: modelID, title: modelTitle, lanes: lanes(), operations: [generate(), edit()]),
    ],
    offers: [
        PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(edit: false), params: params,
                             defaults: ["count": 1], limits: limits),
        PluginManifest.Offer(operation: "image.edit", profile: "image", model: offerModel(edit: true), params: params,
                             limits: editLimits),
    ])
