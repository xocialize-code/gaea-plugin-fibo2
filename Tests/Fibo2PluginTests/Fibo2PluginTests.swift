import Fibo2
import Foundation
@testable import Fibo2Plugin
@testable import GaeaPluginKit
import GaeaProtocol
import MLXFibo2
import MLXToolKit
import Testing

@Suite struct Fibo2PluginTests {
    static let fox = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "Fixtures/fox.png")

    func job(_ operation: String, _ params: [String: JSONValue] = [:], model: [String: JSONValue] = [:],
             text: (String, String)? = ("prompt", "a red fox in the snow"), images: [URL] = [], seed: UInt64? = 7) throws -> Job {
        var inputs: [String: [URL]] = [:]
        if let (name, value) = text {
            let url = FileManager.default.temporaryDirectory.appending(path: "fibo2-test-\(UUID().uuidString).txt")
            try Data(value.utf8).write(to: url)
            inputs[name] = [url]
        }
        if !images.isEmpty { inputs["images"] = images }
        return Job(requestID: "r", job: "job_1", operation: operation, model: modelID, lane: "bf16", inputs: inputs,
                   params: params, modelParams: model, adapters: [], seed: seed, outputDir: URL(fileURLWithPath: "/tmp"))
    }

    @Test func sizesAreFittedToTheTrainedArea() throws {
        #expect(try plan((1024, 1024)).warning == nil)
        #expect(try plan((1424, 800)).warning == nil)
        let big = try plan((2048, 2048))
        #expect(big.width == 1072 && big.height == 1072 && big.warning?.code == "size_snapped")
        #expect(big.width * big.height <= maxPixels)
        let odd = try plan((1000, 700))
        #expect(odd.width == 992 && odd.height == 688)
        #expect(throws: GPPError.self) { try plan((128, 128)) }
        // Every fitted size is inside the package's envelope: the package refuses, so the worker fits first.
        for size in [(4000, 3000), (1920, 1080), (500, 2000), (1024, 1024)] {
            let fitted = try plan(size)
            #expect(Fibo2Envelope.violation(width: fitted.width, height: fitted.height, imageCount: 0) == nil)
        }
    }

    @Test func sharedSettingsHaveTheCardsDefaultsAndTheirRanges() throws {
        let plain = try Shared(job("image.generate"))
        #expect(plain.steps == 50 && plain.guidance == 5 && plain.count == 1 && plain.seed == 7 && plain.negative == nil)
        let set = try Shared(job("image.generate", ["count": 3, "negative_prompt": "blurry"],
                                 model: ["x-fibo2.steps": 30, "x-fibo2.guidance": 3.5, "x-fibo2.text_images": false]))
        #expect(set.steps == 30 && set.guidance == 3.5 && set.count == 3 && set.negative == "blurry" && !set.textImages)
        #expect(throws: GPPError.self) { try Shared(job("image.generate", model: ["x-fibo2.steps": 0])) }
        #expect(throws: GPPError.self) { try Shared(job("image.generate", model: ["x-fibo2.guidance": 0.5])) }
        #expect(throws: GPPError.self) { try Shared(job("image.generate", ["count": 5])) }
        // Without guidance there is no negative pass.
        let unguided = try Shared(job("image.generate", ["negative_prompt": "blurry"], model: ["x-fibo2.guidance": 1]))
        #expect(unguided.negative == nil && unguided.warnings.map(\.code) == ["negative_prompt_ignored"])
        #expect(try Shared(job("image.generate", ["prompt_enhance": "on"])).warnings.map(\.code) == ["prompt_enhance_ignored"])
        #expect(try Shared(job("image.generate", seed: nil)).seed < (1 << 31))
    }

    @Test func theRequestsCarryWhatTheJobAsked() throws {
        let request = Fibo2Request(prompt: "a fox", negative: nil, width: 1024, height: 1024, steps: 30, guidance: 4, seed: 9,
                                   textImages: false)
        let t2i = request.generate(index: 2)
        #expect(t2i.prompt == "a fox" && t2i.negativePrompt == nil && t2i.steps == 30 && t2i.guidanceScale == 4 && t2i.seed == 11)
        #expect(t2i.metaData[Fibo2MetaKeys.textImages] == .bool(false))
        let edit = Fibo2Request(prompt: "make it snow", width: nil, height: nil, seed: 9).edit([Image(format: .png, data: Data())])
        #expect(edit.width == nil && edit.height == nil && edit.images.count == 1 && edit.metaData.isEmpty)
    }

    @Test func anEditWithoutASizeReportsTheTrainedSize() throws {
        let input = try #require(orientedPixelSize(of: Self.fox))
        let trained = Fibo2ImagePrep.imageSize(aspectRatio: Double(input.0) / Double(input.1))
        #expect(trained.width % 16 == 0 && trained.height % 16 == 0)
        #expect(Fibo2Envelope.violation(width: trained.width, height: trained.height, imageCount: 1) == nil)
    }

    @Test func editTurboHasNoGuidanceAndItsOwnSteps() throws {
        let turbo = try Shared(job("image.edit", ["negative_prompt": "blurry"], model: ["x-fibo2.guidance": 7]),
                               model: .editTurbo, counted: false)
        #expect(turbo.steps == 4 && turbo.guidance == 1 && turbo.negative == nil)
        #expect(turbo.warnings.map(\.code) == ["negative_prompt_ignored"])
        #expect(turbo.effective(width: 1024, height: 1024, model: .editTurbo, counted: false)["x-fibo2.guidance"] == nil)
        #expect(throws: GPPError.self) { try Shared(job("image.edit", model: ["x-fibo2.steps": 17]), model: .editTurbo) }
        #expect(try Shared(job("image.edit", model: ["x-fibo2.steps": 8]), model: .editTurbo).steps == 8)
        #expect(configuration(.editTurbo, quant: .int8).variant == .editTurbo)
        #expect(configuration(.editTurbo, quant: .int8).defaultSteps == 4)
        #expect(params(.editTurbo)["properties"]?["x-fibo2.guidance"] == nil)
        #expect(params(.dev)["properties"]?["x-fibo2.guidance"] != nil)
    }

    @Test func aMatteCarriesItsKindAndASeedOnlyWhenAsked() throws {
        let image = Image(format: .png, data: Data([1]))
        let plain = MatteCall().request(image)
        #expect(plain.preferredKind == .softAlpha && plain.metaData.isEmpty)
        #expect(MatteCall().effective == ["x-fibo2.kind": "softAlpha", "seed": 0])
        let asked = MatteCall(kind: .binary, seed: 9)
        #expect(asked.request(image).preferredKind == .binary)
        #expect(asked.request(image).metaData[Fibo2MetaKeys.seed] == .int(9))
        #expect(asked.effective == ["x-fibo2.kind": "binary", "seed": 9])
        #expect(configuration(.rmbg, quant: .int8).variant == .rmbg)
        #expect(Model.rmbg.id == "rmbg-3" && !Model.rmbg.generates)
    }

    @Test func theManifestOffersThreeNamedOnlyNonCommercialModels() throws {
        let manifest = try fibo2Plugin.manifest()
        #expect(manifest.version == "0.3.0" && manifest.minGaea == "1.9.0")
        #expect(manifest.models.map(\.id) == ["fibo-2-dev", "fibo-2-edit-turbo", "rmbg-3"])
        for model in manifest.models {
            #expect(model.lanes.map(\.id) == ["bf16", "int8", "int4"] && model.lanes[0].default)
        }
        // Bria's own gated repos, pinned to commits; nothing is mirrored.
        for lane in manifest.models[0].lanes {
            #expect(lane.resolved.weightSources.map(\.repo) == ["briaai/Fibo-2.0-dev"])
            #expect(lane.resolved.weightSources[0].revision?.count == 40)
        }
        for lane in manifest.models[1].lanes {
            #expect(lane.resolved.weightSources.map(\.role) == ["transformer", "shared"])
            #expect(lane.resolved.weightSources.allSatisfy { $0.repo == "briaai/Fibo-2.0-edit-turbo" && $0.revision?.count == 40 })
        }
        // RMBG 3.0: its own transformer, on Edit Turbo's text encoder and VAE.
        for lane in manifest.models[2].lanes {
            #expect(lane.resolved.weightSources.map(\.repo) == ["briaai/RMBG-3.0", "briaai/Fibo-2.0-edit-turbo"])
            #expect(lane.resolved.weightSources.allSatisfy { $0.revision?.count == 40 })
        }
        #expect(manifest.offers.map(\.operation) == ["image.generate", "image.edit", "image.edit", "image.matte"])
        #expect(manifest.offers.map { $0.modelID } == ["fibo-2-dev", "fibo-2-dev", "fibo-2-edit-turbo", "rmbg-3"])
        for offer in manifest.offers {
            let model = offer.model
            #expect(model["license"]?["spdx"] == "CC-BY-NC-4.0" && model["license"]?["commercial_use"] == false)
            #expect(model["license"]?["accepted_required"] == true && model["named_only"] == true)
            // The licence binds each user, and asks for attribution (AB-A-0297).
            #expect(model["license"]?["user_accept_required"] == true)
            #expect(model["license"]?["attribution"]?.stringValue == (offer.modelID == "rmbg-3"
                ? "Background removed with RMBG 3.0 by Bria AI (CC BY-NC 4.0)" : "Made with Fibo 2.0 by Bria AI (CC BY-NC 4.0)"))
            #expect(model["speed_class"] != nil && model["memory_class"] == "large")
            #expect(model["license"]?["notice"]?.stringValue?.contains("non-commercial") == true)
        }
        #expect(manifest.offers[1].limits?.maxImages == 5 && manifest.offers[2].limits?.maxImages == 5)
        #expect(manifest.offers[0].model["speed_class"] == "slow" && manifest.offers[2].model["speed_class"] == "standard")
        // The edit crops to a trained aspect ratio, so Gaea keeps both models off fill and extend (GPP-1 §4.3).
        #expect(manifest.offers[0].supports == nil)
        #expect(manifest.offers[1].supports == ["image_to_image": true] && manifest.offers[2].supports == ["image_to_image": true])
        // The matte takes a picture of any size and one option.
        let matte = manifest.offers[3]
        #expect(matte.supports == nil && matte.limits?.maxImages == 1 && matte.limits?.maxPixels == 8192 * 8192)
        #expect(matte.params?["properties"]?["x-fibo2.kind"]?["default"] == "softAlpha")
        #expect(matte.model["license"]?["notice"]?.stringValue?.contains("briaai/RMBG-3.0") == true)
        #expect(manifest.engine?.contract == MLXEngineRelease.contract)
    }
}
