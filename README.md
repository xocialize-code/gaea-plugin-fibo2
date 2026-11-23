# gaea-plugin-fibo2

Bria's **Fibo 2.0** (an 8.1B DiT on Qwen3-VL-4B that reads structured JSON captions; CC BY-NC 4.0 weights) as a
[Gaea](https://github.com/xocialize-code/gaea) **research plugin**: Fibo 2.0 Dev, the full base model; Fibo 2.0
Edit Turbo, its 4-step editor; and [RMBG 3.0](https://huggingface.co/briaai/RMBG-3.0), the same model fine-tuned
for background removal. It is built on
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
| `rmbg-3` | `image.matte` | RMBG 3.0: a soft alpha matte of the subject, in 4 steps. Matting only. See [below](#rmbg-30-background-removal). |

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
- **Not on fill or extend** (0.2.1). Gaea runs those with an edit model by marking the region and compositing the
  result back at the picture's size. Fibo 2.0's edit redraws the picture at a trained size and crops the first image
  to that aspect ratio, so its result wouldn't line up, and both `image.edit` offers say so
  (`supports.image_to_image`). `image.edit` itself is unaffected.
- **Not offered:** masked edits as an input (MES `image.edit` has no mask; the package takes one, see fibo2-swift,
  and Gaea is adding a route for it), LoRA adapters, and the model card's style presets.

## RMBG 3.0: background removal

Bria's third background remover is Fibo 2.0 Edit Turbo, fine-tuned and distilled to do one edit: render the picture's
alpha matte. Because it starts from a model that understands pictures, it knows what the subject is. It keeps every
person of a group, clears the holes in mesh and spokes, and leaves a cast shadow out.

```json
{"operation": "image.matte", "model": "rmbg-3", "inputs": {"image": "…"}}
```

- **The result** is MES `image.matte`'s: `mask`, a grey PNG the picture's size (white is the subject, grey is partial
  transparency at hair, fur and motion blur), and `extract`, the cut-out Gaea makes from it.
- **Nothing to tune.** The model was trained on one instruction and runs 4 steps without guidance. There is no
  prompt, and two options:
  - `x-fibo2.kind`: `softAlpha` (default) or `binary`, the matte thresholded at half.
  - `seed`: 0 by default, so the same picture gives the same matte. Bria reports rare broken mattes (an empty one, a
    dropped object); another seed redraws it.
- **Any size in, up to 8192×8192.** The model sees the picture at about one megapixel, and the matte is resized to
  the picture. On a large picture its edges are softer than the pixels.
- **The whole frame.** The picture is stretched to the size the model was trained at for its aspect ratio (a few
  pixels in 1024 at most), and the matte is stretched back. The model card's snippet crops the picture to that size
  instead, which leaves the cut-off strips without a matte.
- **Time:** about 20 s a picture on an M5 Max (16 s of denoising), on every lane. BiRefNet, Gaea's default matte,
  takes a fraction of a second: RMBG 3.0 is for the pictures that one gets wrong.
- **Lanes** agree: on a 5071×3294 photo, int8 moved 0.007 % of the pixels across the half-way line against bf16, and
  int4 0.06 %.
- **Where it loses** (Bria's own list): glass, veils, smoke and splashes tend to come out opaque; a busy or
  low-contrast scene can pull in grass or props beside the subject.

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

Each lane is a DiT quant, made at load from the bf16 snapshot. All three models have the same three lanes and the same
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

- `briaai/RMBG-3.0`, pinned to commit `87330e3f87d9d856a21453e5acd120219984a66b`: a 16 GB transformer and nothing
  else. It runs on Edit Turbo's text encoder, VAE and fonts, which are Dev's: with either installed it fetches only
  its transformer, and without them it fetches those 9 GB from Edit Turbo's repo.

All three repos are **gated**, each with its own form, and Gaea never stores a Hugging Face token. So fetch the weights
yourself before the first job; a download started by a job fails without access.

1. Request access on the model pages ([Dev](https://huggingface.co/briaai/Fibo-2.0-dev),
   [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo), [RMBG 3.0](https://huggingface.co/briaai/RMBG-3.0))
   with your own Hugging Face account, for non-commercial use.
2. Log in and download into Gaea's models root (Settings → General shows it; MLXEngine's store by default):

```bash
hf auth login
```

```bash
hf download briaai/Fibo-2.0-dev --revision b01e5a34a0a9540c65f585c2c281535deabac16d --exclude "assets/*" --cache-dir "<models root>"
```

```bash
hf download briaai/Fibo-2.0-edit-turbo --revision 7b133d62d92e897f1468af0427a73cc41a03cb5d --include "transformer/*" --include "model_index.json" --cache-dir "<models root>"
```

```bash
hf download briaai/RMBG-3.0 --revision 87330e3f87d9d856a21453e5acd120219984a66b --include "transformer/*" --include "model_index.json" --cache-dir "<models root>"
```

3. Load them once, so the next job starts at once:

```bash
gaea plugins warm com.xocialize.gaea.fibo2 fibo-2-edit-turbo
```

The Edit Turbo line fetches only its transformer; that is enough when Dev is under the same root. Without Dev, drop its
`--include` options to fetch the whole repo. RMBG 3.0's repo is its transformer, so it needs Dev or the whole Edit
Turbo repo beside it. The engine then reads `HF_TOKEN`, the Keychain, or the CLI's token file.
This plugin doesn't mirror or redistribute any weights.

## Build and check locally

```bash
scripts/build-plugin.sh
```

```bash
CHECK_MODELS_ROOT=<a models root holding briaai/Fibo-2.0-dev> scripts/check-plugin.sh
```

`check-plugin.sh` generates one 512² image with the int8 lane in 8 steps, with every `.build` folder hidden so MLX
can find its metallib only inside the bundle. When the root also holds `briaai/RMBG-3.0`, it then mattes the fox
fixture with `rmbg-3`. A control then shows that the bare worker, outside the bundle, can't.
`scripts/release.sh` builds, checks, notarizes with the fleet's `mlx-notary` profile, and publishes the GitHub release.
See the [demo plugin](https://github.com/xocialize-code/gaea-plugin-demo) for signing, notarization and CI secrets.

`fibo2-inproc` (`swift run -c release fibo2-inproc`; its header lists the options) is not part of the plugin. It runs
the same lane in-process through MLXServeEngine, with the request the worker would build.

## Licence

- **The weights: CC BY-NC 4.0** (Bria AI, `license_name: bria-fibo`;
  [terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode)). Non-commercial use only, with attribution.
  Commercial use needs a licence from Bria: <https://platform.bria.ai/licensing/pricing?model=fibo-2>. Bria states the
  model was trained only on licensed data.
- **RMBG 3.0's weights are under the same licence** ("the same terms as the Fibo 2.0 family", its model card), with
  its own page for commercial terms: <https://platform.bria.ai/licensing/pricing?model=rmbg-3.0>. Bria's hosted API
  and fal endpoint are the commercially licensed routes.
- **In Gaea.** The offers say `commercial_use: false` and carry a notice with these terms. The operator accepts
  them to enable a model (`accepted_required`), and each client's user is shown the notice and accepts it too
  (`user_accept_required`), because the licence binds every user. Gaea carries the attribution "Made with Fibo 2.0
  by Bria AI (CC BY-NC 4.0)" to clients, and "Background removed with RMBG 3.0 by Bria AI (CC BY-NC 4.0)" for a matte. What a user does with the outputs is theirs to decide.
- **The plugin code** is Apache-2.0 (Gaea G-21), and fibo2-swift is MIT. The text encoder (Qwen3-VL-4B) is
  Apache-2.0. fibo2-swift embeds Pillow's built-in font, a subset of Aileron (CC0), to draw text images the way the
  model's training did.
