# gaea-plugin-fibo2

Bria's **Fibo 2.0** (an 8.1B DiT on Qwen3-VL-4B that reads structured JSON captions; CC BY-NC 4.0 weights) as a
[Gaea](https://github.com/xocialize-code/gaea) **research plugin**: Fibo 2.0 Dev, the full base model, and Fibo 2.0
Edit Turbo, its 4-step editor. It is built on
[GaeaPluginKit](https://github.com/xocialize-code/gaea-plugin-kit) over
[fibo2-swift](https://github.com/xocialize/fibo2-swift), runs in its own process, and speaks the Gaea Plugin Protocol
v1. Add it to Gaea by its repo URL:

```bash
gaea plugins add https://github.com/xocialize-code/gaea-plugin-fibo2
```

## A research plugin: named-only and opt-in

The weights are non-commercial, so Gaea makes the models **opt-in and named-only**: nothing runs them by default, and
a request reaches one only by naming it (`"model": "fibo-2-edit-turbo"`). The offers carry the licence, `commercial_use: false`
and a notice, so a user sees the terms before using it. This is a plugin, not a shipping clearance.

## What it offers

| Model | MES operation | Notes |
|---|---|---|
| `fibo-2-dev` | `image.generate` | Text-to-image at about one megapixel. 50 steps with guidance 5. Negative prompts honoured. |
| `fibo-2-dev` | `image.edit` | Instruction editing of 1–5 images, named in order as `<image_1>`, `<image_2>`. One output. |
| `fibo-2-edit-turbo` | `image.edit` | The same edits in 4 steps without guidance: about a twentieth of the DiT work. Editing only. |

- **Prompts.** Fibo 2.0 was trained on structured JSON captions ([VGL](https://docs.bria.ai/vgl)): `short_description`,
  `objects`, `background_setting`, `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`, and
  `text_render` for text to draw. Send that JSON as the prompt for the control the model is known for. A short plain
  description works too, with less control. No prompt-enhance profile writes these captions yet: `prompt_enhance: on`
  is reported `prompt_enhance_ignored`.
- **Edit instructions** go as typed. Or send JSON with `edit_instruction` and the result's caption under
  `structured_caption`.
- **Sizes.** Multiples of 16, at most about 1.1 megapixels: the area the model was trained at (1024×1024, 1424×800,
  1248×832). A larger size is scaled down and reported `size_snapped`. An edit without a size is drawn at the size
  trained for the first image's aspect ratio, so that image is cropped to the ratio, not stretched.
- **`count`** on `image.generate`: 1–4 images, each with `seed + i`. A request without a seed gets one, reported in
  `effective`. `image.edit` has one output; `strength` does nothing (`param_ignored`).
- **Negative prompt.** Dev: without one, text-to-image uses the model card's negative and an edit the model's null
  caption. Edit Turbo runs without guidance, so a negative prompt is reported `negative_prompt_ignored`.
- **Options**, as MES extensions:
  - `x-fibo2.steps`: Dev 1–100, default 50 (30 is faster and close); Edit Turbo 1–16, default 4;
  - `x-fibo2.guidance` (Dev only; 1–10, default 5; 1 turns guidance off and halves the time);
  - `x-fibo2.text_images` (default true): draw each `text_render` text and show it to the text encoder, as the
    model's training did.
- **Not offered:** masked edits as an input (MES `image.edit` has no mask; the package takes one, see fibo2-swift),
  LoRA adapters, and the model card's style presets.

## Speed

Dev is the full base model: two DiT passes a step. On an M5 Max, bf16, with nothing else on the GPU:

| Request | Per step | 50 steps | 30 steps |
|---|---|---|---|
| Text-to-image, 1024² | 3.5 s | about 3 min | about 1.8 min |
| Edit, 1 image | 7.6 s | about 6 min | about 4 min |
| Edit, 2 images | 12.8 s | about 11 min | about 6.5 min |
| Edit, 5 images | 38.8 s | about 32 min | about 19 min |

**Edit Turbo** makes 4 passes where Dev makes 100. A one-image edit at 1024² took 21.5 s end to end over the plugin
protocol on the int8 lane (loading the text encoder, encoding, 4 steps, decoding), against about 6 minutes for Dev.

## Lanes and memory

Each lane is a DiT quant, made at load from the bf16 snapshot. Both models have the same three lanes and the same
memory. The DiT and VAE stay resident; the text encoder loads
for each request and is released before denoising.

| Lane | Resident | Peak while running | Gaea schedules |
|---|---|---|---|
| `bf16` (default) | 16.6 GB | 13.3 GB | 30.4 GB |
| `int8` | 9.1 GB | 13.4 GB | 22.8 GB |
| `int4` | 7.5 GB | 13.2 GB | 21.1 GB |

The peak is the encode phase of an edit, measured on the process: the text encoder (8.9 GB) loads, encodes and is
released. Denoising needs 2.4–4.9 GB over resident.

`int4` keeps the attention projections and the FFN down-projection at 8 bits; uniform 4-bit drifted too far. When the
default lane doesn't fit, Gaea loads a smaller one and warns (`lane_fallback`). A request naming a lane
(`x-gaea.lane`) never falls back.

## Weights

Gaea sends its models root in `configure` (MLXEngine's store by default; `gaea models root` changes it). The worker
reads these there and downloads what is missing on `prepare` or `materialize`:

- `briaai/Fibo-2.0-dev`, pinned to commit `b01e5a34a0a9540c65f585c2c281535deabac16d` (about 25 GB: the transformer,
  the text encoder, the VAE, the tokenizer and two fonts).
- `briaai/Fibo-2.0-edit-turbo`, pinned to commit `7b133d62d92e897f1468af0427a73cc41a03cb5d`. Its text encoder, VAE,
  tokenizer and fonts are Dev's, byte for byte, so with Dev installed Edit Turbo fetches only its 16 GB transformer.
  Without Dev it fetches all 25 GB from its own repo.

Both repos are **gated**, each with its own form. Request access on the model pages
([Dev](https://huggingface.co/briaai/Fibo-2.0-dev), [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo))
with your own Hugging Face account, for non-commercial use. The engine then reads `HF_TOKEN`, the Keychain, or the CLI's token file.
This plugin doesn't mirror or redistribute any weights.

## Build and check locally

```bash
scripts/build-plugin.sh
```

```bash
CHECK_MODELS_ROOT=<a models root holding briaai/Fibo-2.0-dev> scripts/check-plugin.sh
```

`check-plugin.sh` generates one 512² image with the int8 lane in 8 steps, with every `.build` folder hidden so MLX
can find its metallib only inside the bundle. A control then shows that the bare worker, outside the bundle, can't.
`scripts/release.sh` builds, checks, notarizes with the fleet's `mlx-notary` profile, and publishes the GitHub release.
See the [demo plugin](https://github.com/xocialize-code/gaea-plugin-demo) for signing, notarization and CI secrets.

`fibo2-inproc` (`swift run -c release fibo2-inproc`; its header lists the options) is not part of the plugin. It runs
the same lane in-process through MLXServeEngine, with the request the worker would build.

## Licence

- **The weights: CC BY-NC 4.0** (Bria AI, `license_name: bria-fibo`;
  [terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode)). Non-commercial use only, with attribution.
  Commercial use needs a licence from Bria: <https://platform.bria.ai/licensing/pricing?model=fibo-2>. Bria states the
  model was trained only on licensed data.
- **In Gaea.** The offers say `commercial_use: false` and `accepted_required: true` and carry a notice with these
  terms. What a user does with the outputs is theirs to decide.
- **The plugin code** is Apache-2.0 (Gaea G-21), and fibo2-swift is MIT. The text encoder (Qwen3-VL-4B) is
  Apache-2.0. fibo2-swift embeds Pillow's built-in font, a subset of Aileron (CC0), to draw text images the way the
  model's training did.
