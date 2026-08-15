# MiniMax Music 3 Integration Design

MiniMax Music 3 is a lyrics- and caption-conditioned music generation model. This
document is the implementation contract for adding it to SushiUI as a second
audio architecture alongside ACE-Step 1.5.

Revision 2 (2026-08-14) replaces revision 1 after an audit against the actual
snapshot, the actual upstream source code, and the installed environment. The
corrections are listed in [Revision history](#revision-history); several
revision-1 statements were wrong and had shaped the phase plan.

## Scope

Phase 1 is inference plus frontend:

1. text-to-music (`txt2aud`) end to end, with progress and cancellation;
2. extend (`outpaint/audio`) and regenerate-from-a-point (`aud2aud`,
   `mode="repaint"`) for SushiUI-generated songs;
3. an LLM-driven caption rewriter that runs on the user's own local LLM;
4. INT8 ConvRot as an opt-in memory path.

Training is out of scope for phase 1 but is explicitly designed for; see
[Training forward-compatibility](#training-forward-compatibility). The single
decision phase 1 must not get wrong is the
[per-generation state contract](#per-generation-state-contract): without it,
extend and repaint become impossible for anything already in the gallery.

The model revision this document is written against is
`bd348f9c49ea3c1b39f33ace3436f8fad435f24e`, recorded in
`M:/model/minimax-music3/manifest.json`.

## Sources of truth

- [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) — weights, configs, model card, license.
- [huggingface/diffusers#14456](https://github.com/huggingface/diffusers/pull/14456) at commit `dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d` — **the only source of Music3 model and pipeline code**. Apache-2.0.
- [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3) — the flat repack and the published INT8 ConvRot artifacts.
- [MiniMax-AI/MiniMax-Music3 `skills/music-caption-rewriter`](https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills/music-caption-rewriter) — the caption rewriter, distributed as markdown, not as code.

The weights keep the **MiniMax Music 3 Community License**
(`official/LICENSE`). The Comfy-Org repack's Apache label covers its own
repacking work, not the original weights. The upstream license must be
preserved and surfaced.

## Dependency gate (blocking)

**There is no Music3 code in the snapshot.** The entire `official/` tree
contains exactly one Python file, `scripts/end_to_end/minimax_ttm_test.py`,
which is a `urllib` client for an SGLang server. Every model class lives in the
unmerged diffusers PR.

The installed environment is `diffusers 0.38.0` / `transformers 5.1.0`; the model
declares `_diffusers_version: 0.40.0.dev0`. Upgrading diffusers repo-wide to an
unreleased commit to serve one architecture is not acceptable — nine other
architectures depend on the pinned version.

**Decision: vendor.** The PR's Music3 code is ~62 KB across 11 files, Apache-2.0,
and splits cleanly:

| Upstream file | Disposition |
|---|---|
| `models/transformers/transformer_minimax_music3.py` | vendor nearly verbatim |
| `models/transformers/minimax_music3_rvq_depth_decoder.py` | vendor nearly verbatim |
| `models/condition_embedders/condition_embedder_minimax_music3.py` | vendor nearly verbatim |
| `models/autoencoders/minimax_music3_vocoder.py` | vendor nearly verbatim |
| `modular_pipelines/minimax_music3/*.py` (5 blocks + pipeline) | **port**, not vendor — see below |

The four model modules import only APIs that exist in 0.38 — verified by import:
`ConfigMixin`, `register_to_config`, `ModelMixin`, `AttentionModuleMixin`,
`dispatch_attention_fn`, `TimestepEmbedding`, `RMSNorm`,
`Transformer2DModelOutput`, `lru_cache_unless_export`. Attention must be
re-pointed at SushiUI's own conduit (`backend/core/attention/`) rather than
`dispatch_attention_fn`, per the repo-wide rule that every architecture routes
through one dispatcher.

The five modular-pipeline blocks depend on the 0.40-only modular framework
(`ModularPipeline`, `LoopSequentialPipelineBlocks`, `InputParam.template`,
`guiders.ClassifierFreeGuidance`) and must be re-expressed as one plain pipeline
class. This is mechanical: the blocks are thin, and the only guider behaviour
used is ordinary CFG, `uncond + scale * (cond - uncond)`. Porting is also what
makes staged offload, cancellation and per-chunk progress reporting possible at
all — the upstream loop exposes no hooks for them.

Two load-time gates, both cheap and both required because the configs were
written by newer libraries:

- `language_model/config.json` was written by `transformers 5.13.0.dev0` and uses
  the `rope_parameters` form. Measured on the installed `transformers 5.1.0`: the
  config parses correctly and the rotary base comes out at 1e6 (recovered
  999997.4 from the fp32 `inv_freq`), so there is no silent fallback — but
  `config.rope_theta` is **`None`**, because the value now lives in
  `config.rope_parameters["rope_theta"]`. The load gate must read that field;
  a gate written against `rope_theta` would misfire on a healthy load.
- Assert the component class/shape census matches the tables below rather than
  trusting `from_pretrained` to have populated everything.

## Architecture, as verified

Three stages. The generation-time cost is dominated by the first.

**1. Autoregressive stage (25 Hz).** An 8B `Qwen3ForCausalLM` plus a 0.6B RVQ
depth decoder emit, per frame, one semantic code (vocab 16,384) and seven
residual codes (vocab 1,024 each). Both models must be device-resident together;
the loop calls their submodules directly. AR CFG scale is **1.5**, top-k **50**,
both hard-coded in the reference recipe. The unconditional branch is the same
prompt with every token except the first and last two replaced by
`<|audio_cfg|>` (151654). Output: `frame_hiddens`, shape
`[1, frames, 8 * 4096]`.

**2. Flow-matching stage (86.13 Hz).** The condition encoder softmax-mixes the 8
hidden states per frame, projects 4096→2048, and nearest-neighbour resamples
25 Hz→86.13 Hz. A 2.4B 1D DiT (36 layers, in_channels 128) denoises in
**200-frame windows with a 100-frame hop**; neighbouring windows overlap by ~344
latent frames, of which the first 172 are blended toward the previous window's
carry at every step. Flow CFG scale defaults to **1.7**; the unconditional branch
conditions on **zeros**, so there is no negative prompt anywhere in this model.
Scheduler: `FlowMatchEulerDiscreteScheduler` with `invert_sigmas: true` and
`sigmas = linspace(1, 1/steps, steps)`.

**3. Decode.** The 128-channel latent is **two folded 64-channel mono streams**
(`vocoder.forward` reshapes `[B, 128, L]` → `[2B, 64, L]`, decodes, and reshapes
back to `[B, 2, samples]`). Upsampling 8·8·4·2 = 512× to **44.1 kHz stereo**.
Windows are stitched by cropping 86 leading latent frames from every window after
the first and 258 trailing from every window before the last.

Sample-rate note: `vocoder/config.json` says `sampling_rate: 44100` and the
diffusers path emits 44.1 kHz. The model card's 32 kHz refers to the SGLang
serving path's WAV output. SushiUI writes what the vocoder produces and does not
resample.

### Component census

| Component | Class | Notes |
|---|---|---|
| `tokenizer` | `Qwen2Tokenizer` | music tokens `<|caption_start/end|>`, `<|lyrics_start/end|>`, `<|audio_start/end|>`, `<|audio_cfg|>` |
| `language_model` | `Qwen3ForCausalLM` | 8B, hidden 4096, 36 layers, vocab 200,000, max positions 10,240 |
| `rvq_depth_decoder` | `MiniMaxMusic3RVQDepthDecoder` | 4 layers, 8 codebooks, audio vocab 1,024 |
| `condition_encoder` | `MiniMaxMusic3ConditionEncoder` | 4 tensors only; 25 Hz → 86.13 Hz |
| `transformer` | `MiniMaxMusic3Transformer1DModel` | 2.4B, 36 layers, in_channels 128 |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | `invert_sigmas: true` |
| `vocoder` | `MiniMaxMusic3Vocoder` | decoder half of the DAV; 44.1 kHz stereo |

The 10,240-position budget cannot hold the documented 5,000-token prompt cap and
9,000-frame cap at once. Long prompts must be validated against the *remaining*
budget for the requested duration, not against 5,000 in isolation.

## Capability verdict

The question "can Music 3 accept references — voice, instrument, style?" is
answered here from the code and the weights, not from the model card.

| Capability | Verdict | Evidence |
|---|---|---|
| Text → music | **Yes** | the entire released pipeline |
| Reference audio conditioning the AR stage (voice/timbre/instrument) | **No** | the pipeline's complete input set is `prompt`, `lyrics`, `audio_duration`, `generator`, `num_inference_steps`, `output_type`. The RVQ tokenizer's *encoder* is not published, so no audio can be turned into semantic codes; and the DiT conditions on LM hidden states, not on audio |
| Negative prompt | **No** | the flow-stage unconditional branch is zeros; the AR unconditional branch is the token-masked prompt |
| Waveform → latent (DAV encode) | **Yes, but not exposed** | `official/dav.pth` (`62000_generator`) contains `encoder.*` (119 tensors), `mean_proj`/`logs_proj` → 64 ch, a VITS-style `flow` (304 tensors), and `decoder.*`. The published diffusers `vocoder` component is only the decoder half |
| Extend / regenerate-from-a-point for SushiUI-generated songs | **Yes** | AR resume from stored codes; see below |
| Mid-song infill with a preserved tail | **No** | the global LM is causal; there is no infilling contract |
| Cover / repaint of arbitrary user audio | **Not in phase 1; not proven** | DAV encode makes latents reachable, but the DiT's conditioning would come from a text-driven AR pass unrelated to that audio. Treat as an experiment, not a promise |

The first three rows are properties of the released model and must be encoded as
refusals in `arch_capabilities.py` with those reasons, not as unimplemented
features.

Revision 1 stated that reference audio, cover and repaint were "separate
capability gates". That framing implied they were merely deferred. Rows 2 and 7
are the accurate statement.

## Generation parameter contract

The upstream surface is small. Every default belongs in
`backend/api/param_defaults.py`; nothing below may be duplicated in the frontend.

| Parameter | Default | Bounds | Notes |
|---|---|---|---|
| `prompt` | — | non-empty | the **music description / caption**, not lyrics |
| `lyrics` | — | non-empty | structure tags such as `[verse]` must each own a line; text sharing a line with a leading tag is silently dropped by the checkpoint's input contract, so the UI must warn |
| `audio_duration` | 60.0 s | > 0, ≤ 360 s | an **upper bound**, not a target — the LM may emit the end-of-audio token earlier. UI wording must say so |
| `num_inference_steps` | 30 | ≥ 1 | per chunk, not per song |
| `seed` | — | — | maps to `torch.Generator` |
| `flow_guidance_scale` | 1.7 | > 0 | exposed; AR CFG 1.5 and top-k 50 stay fixed |

Instrumental tracks are expressed through caption and structure tags; `lyrics`
is required non-empty by the checkpoint contract.

`prompt` is markdown-stripped before tokenization (`_clean_caption`), so a
rewriter emitting markdown headings is safe but gains nothing from them.

## Modality surfaces

The user-facing goal is parity with the image side: txt2aud like txt2img, plus
inpaint and outpaint. What that maps to here is constrained by the causal LM.

**txt2aud** — `POST /generate/txt2aud`, the existing ACE-Step route surface.

**outpaint / extend** — `POST /generate/outpaint/audio`. Forward extension is
implemented by *resuming the AR loop*: replay the stored frame codes to rebuild
the LM's KV cache, then continue sampling. The original audio is preserved
exactly; only the appended span is new. Backward extension is not possible.

**inpaint / repaint** — `POST /generate/aud2aud` with `mode="repaint"`, matching
how ACE-Step exposes audio inpaint (`Img2ImgPanel`, not the Inpaint tab, which
has no audio branch at all). Two honest modes, and the UI must label which is
which:

- *regenerate from T onward* — AR-resume with the prefix codes as context and a
  new tail. Content changes; everything before T is preserved exactly.
- *re-render a range* — keep the codes, redraw the flow stage over that window
  with a new seed. Timbre and mix change; lyrics, melody and timing do not.

Mid-span infill with a preserved tail is refused with the causal-LM reason, in
the same style as H3's placement enumeration.

### Per-generation state contract

Extend and repaint both need the AR state of the original generation. Storing
`frame_hiddens` is not viable — `[1, 9000, 32768]` in bf16 is ~590 MB per song.
Storing the **frame codes** is: 8 int16 per frame is 144 KB for a full six-minute
song, and the hidden states are exactly recoverable by a teacher-forced replay of
the stored codes, which is one batched LM pass rather than 9,000 sequential
steps.

Phase 1 therefore writes a sidecar next to the audio file holding the frame
codes, the sample/frame rates, the prompt, the lyrics and the seed. **This must
ship with txt2aud, not after it** — a song generated without a sidecar can never
be extended or repainted.

## Caption rewriter (AI rewrite)

Upstream's "prompt refiner" is not runtime code. It is an agent skill —
`skills/music-caption-rewriter/`, a `SKILL.md` plus a genre router, 18 family
indexes and ~1,000 caption templates — installed with `npx skills add`. It calls
no model of its own; the npm step only copies markdown.

SushiUI will not install it. Instead the contract it encodes is re-implemented as
a music mode of the existing prompt-assist mechanism, which already routes to the
user's own LLM.

Reuse, from H3: `backend/core/extensions/minimax_h3_prompt_assistant.py`
(LM Studio and Ollama providers, loopback-only URL enforcement, SQLite result
cache, one self-repair retry, strict-JSON return), routes
`POST /prompt-assist/{models,template,transform,cache/clear}`,
`GET /schema/prompt-assist-defaults`, and the
`frontend/src/components/common/H3PromptAssist.tsx` UI.

Do not extend H3's mode enum. Its `SECTION_NAMES`, validators and system prompt
are video-specific; music gets a sibling module sharing the provider/cache layer.
The music contract: expand a short caption into a Structured Caption with exactly
the three headings **Global Metadata**, **Vocal Details**, **Arrangement**;
250–450 English words; never quote lyric lines; never invent BPM or key that the
user did not give; preserve explicit exclusions. Server-side validation checks
those properties before the result is accepted, the same way the H3 validator
does.

Whether to vendor upstream's genre router and templates is a separate decision
gated on that repository's license — the weights' Community License does not
cover it. Contract-only is sufficient to ship.

## Lyrics assistant

The motivating defect is in [Generation parameter contract](#generation-parameter-contract)'s
`lyrics` row: the checkpoint's own lyric normalizer, `_normalize_lyrics` in
`backend/core/models/minimax_music3/pipeline.py`, keeps only the leading
structure-tag run on a line and silently drops the rest of that line, with no
error anywhere:

```
"[Verse] The morning light"    ->  "[start]\n[verse]"
"[Verse]\nThe morning light"   ->  "[start]\n[verse]\nThe morning light"
```

The most natural way to type lyrics destroys them. `backend/core/extensions/
minimax_music3_lyrics_assistant.py` addresses this with three explicit,
opt-in, user-selected modes — never auto-applied, so a user who types lyrics
directly and never opens the assistant is unaffected:

- **`format`** — deterministic, no LLM call, no network. Fixes only the
  layout: moves any text sharing a line with a tag onto its own line, puts
  one tag per line, lowercases tag case, and drops blank-line noise. This
  mode is REQUIRED to preserve the user's words exactly: the ordered
  sequence of non-tag word tokens is identical before and after, enforced as
  a checked invariant inside `format_lyrics` itself (it raises rather than
  returning a result that fails the check, treating a violation as a bug in
  the function, never a user-input problem). Served by
  `POST /prompt-assist/music/lyrics/format`.
- **`structure`** — the LLM emits only tags, one per line, no prose. For a
  purely instrumental piece — `lyrics` is required non-empty by the
  checkpoint contract even with no words — this is the structural control
  surface: the user describes the arrangement, and the model returns the
  section map.
- **`complete`** — the user supplies a theme and/or partial lyrics; the LLM
  writes or finishes the words. Any lyric line the user actually supplied is
  validated to survive verbatim in the output (a contiguous word-token
  sublist match, not a raw substring match — see the module for why a raw
  substring check on joined text can false-accept a shuffled fragment).
  Both LLM-driven modes are served by
  `POST /prompt-assist/music/lyrics/transform`.

`format_lyrics` is run as the final pass over the LLM's raw output for both
`structure` and `complete`, so whatever `transform()` returns is
contract-clean by construction before validation even runs.

Tag vocabulary: the model card documents nine tags — `[Intro]` `[Verse]`
`[Pre-Chorus]` `[Chorus]` `[Post-Chorus]` `[Bridge]` `[Instrumental]`
`[Solo]` `[Outro]`. MiniMax's own example script also uses `[interlude]` and
a descriptive freeform tag, so a tag outside the documented nine is always a
warning, never a refusal — the assistant's validators and `format_lyrics`
itself follow this rule uniformly.

This is a sibling of the caption rewriter above, not an extension of it, and
like it a sibling of `minimax_h3_prompt_assistant.py`: it reuses that
module's provider/cache/transport layer (LM Studio and Ollama transport,
loopback-only URL enforcement, strict-JSON extraction, the SQLite result
cache, the one self-repair retry, the error type) unchanged, but has its own
domain logic, its own `GUIDE_VERSION`, and its own cache file — a
caption-rewrite cache entry and a lyrics-assist cache entry can never
collide, even for an identical-looking request. Model listing is shared
unchanged (`POST /prompt-assist/models`). Frontend:
`frontend/src/components/common/MusicLyricsAssist.tsx`, rendered beside
`MusicCaptionAssist.tsx` in `Txt2ImgPanel.tsx`.

Generation-time surfacing, independent of whether the assistant is used at
all: `POST /generate/txt2aud` scans the request's `lyrics` for any line
where text follows a leading tag (using the checkpoint's own leading-tag
regex, matched against the raw line exactly as `_normalize_lyrics` does, not
a whitespace-stripped copy of it) and adds one `warnings[]` entry per
offending line, naming the line and the text that will not reach the model.
This only applies to `POST /generate/txt2aud`: extend and repaint always
reuse the original song's sidecar-stored `prompt`/`lyrics` and refuse a
request that supplies a differing one (see
[Per-generation state contract](#per-generation-state-contract)), so a
caller's `lyrics` never reaches the model unmodified on those two routes.

## Quantization

The staged flat artifacts follow ComfyUI's `int8_convrot` layout, the same family
already running for H3 (`backend/core/models/common/convrot_int8_linear.py`).
Phase 1 loads BF16/FP16 only; INT8 lands after a BF16 A/B comparison.

Two facts must be respected by any local conversion:

- The flat DiT is **not** the two official `transformer/` shards. It holds 374
  tensors: 370 `diffusion_transformer.*` plus `latent_conditioners.{0,1}`,
  `cond_layer_logits`, `cond_layer_scale` — i.e. the condition encoder folded in.
  It also fuses QKV and uses `.gamma`/`.beta` norms. A census/SHA comparison
  against a transformer-only merge will always fail.
- The flat text encoder is the language model **and** the depth decoder merged,
  with plain HF-style keys (`model.layers.*`, `model.audio_decoder.*`). The
  pruned variant splits the vocabulary into `embed_tokens_prefill` [151675] and
  `embed_tokens_audio` [16384] with `lm_head_pruned` [16385]. There is no
  `Abab`-style key layout in the flat files.

`official/qwen_7B/` is a permanent exclusion, not an opt-in path: its
`auto_map` targets `modeling_abab.py` and `configuration_abab.py`, neither of
which is in the snapshot, and its 48-shard index is missing a shard.

## GGUF weights

A third distribution exists:
[molbal/Minimax-Music3-GGUF](https://huggingface.co/molbal/Minimax-Music3-GGUF)
(`d6ab7b87`, 2026-08-13), published for ComfyUI-GGUF. Two files are staged:
`diffusion_models/minimax_music3_dit_BF16.gguf` (4.98 GB) and
`text_encoders/minimax_music3_text_encoder_pruned_Q8_0.gguf` (9.59 GB). The repo
also carries Q8_0 / Q8_CR / Q4_0 DiTs and Q8_CR / Q4_0 text encoders.

Read directly from the GGUF headers, because it decides the work:

- **The tensor names are the flat ComfyUI layout, exactly.** The DiT is the same
  374 tensors as the flat safetensors — `diffusion_transformer.*`, fused
  `to_qkv`, `.gamma`/`.beta` norms, plus `latent_conditioners.*`,
  `cond_layer_logits`, `cond_layer_scale`. The text encoder is the same 328
  tensors as the flat pruned repack — `model.embed_tokens_prefill`,
  `model.embed_tokens_audio`, `model.lm_head_pruned`, `model.audio_decoder.*`.
  So **GGUF and the flat safetensors need one key remap, not two**, which is why
  they belong in the same phase.
- **The files carry no architecture configuration.** Only three metadata keys
  exist: `general.architecture = minimax_music3`, `general.quantization_version`,
  `general.file_type`. There are no dimensions, no layer counts, no sample rate.
  Every config must still come from `official/`, so the sibling probe is a
  requirement for the GGUF path rather than a convenience.
- **`general.architecture` is `minimax_music3`, not a llama.cpp architecture.**
  These are ComfyUI-convention GGUF containers, so no llama.cpp loader applies
  and none should be reached for.
- **The declared precision is not the whole story.** The "BF16" DiT is
  F32 + F16 on disk (226 F32 / 148 F16 tensors, `file_type=1`). The Q8_0 text
  encoder is 169 Q8_0 + 155 F32 + 4 BF16.

Two consequences worth deciding deliberately rather than drifting into:

**The pruned vocabulary is an architecture change, not a size change.** The
vendored `Qwen3ForCausalLM` path indexes `embed_tokens` at
`code + AUDIO_CODE_OFFSET` over a 200,000-entry table and masks the logits down
to the audio range. The pruned layout instead splits the table into
`embed_tokens_prefill` [151,675] for text and `embed_tokens_audio` [16,384] for
semantic codes, with `lm_head_pruned` [16,385] (the audio codes plus
end-of-audio). Adopting it means changing the offset arithmetic and the vocab
mask in the AR loop, which is checkpoint-contract code. That is the substance of
the work; the container format is the easy part.

**Dequantise-on-load would be a hollow feature.** Expanding Q8_0 to bf16 at load
yields a resident text encoder no smaller than the bf16 file already staged —
the only gain would be a smaller download. The reason to want Q8_0 is residency,
so the target is packed weights with dequantisation at use, in the shape the
existing `convrot_int8_linear.py` runtime already establishes for this repo.
Q8_0 is block-wise: 32 values per block with one fp16 scale.

No new pip dependency should be taken for this. The GGUF container is a short
header plus tensor records and is read here directly; the `gguf` package would
add a supply-chain dependency for a format this repo can parse in one file.

## Which tree the loader reads

The snapshot offers the same model twice: the seven-component `official/` tree,
and the flat ComfyUI-style repack in `diffusion_models/`, `text_encoders/` and
`vae/`. They are **not the same tensors**, so a loader has to commit to one.

Phase 2 reads `official/` and refuses the flat files with a message naming the
reason. That inverts what an H3-shaped design would do — H3 takes weights from
the flat files and completes the rest from `official/` — and the inversion is
deliberate: Music3's flat DiT fuses QKV, folds the condition encoder in, and
renames norms to `.gamma`/`.beta`, while its flat text encoder merges the
language model with the depth decoder. Every one of those needs a key remap that
`official/` does not, and `official/` was verified to load key-for-key with no
remap at all (441/441, 4/4, 47/47, 121/121).

The cost, which the refusal message stated before item 9 landed: until then, the
only available precision was the `official/` FP32 transformer cast at load
time.

**Item 9 has landed.** `core.models.minimax_music3.flat_remap` performs the DiT
and non-pruned text-encoder key remaps described above, and
`load_minimax_music3_from_path` now reads a flat, non-quantized DiT file
(FP32 or FP16) when pointed at one, with configs still coming from `official/`
and every other component (language model, depth decoder, vocoder, tokenizer,
scheduler) still read from `official/` unchanged. The flat text encoder's
builder exists and is tested but is not wired into that dispatch — there is
no existing detection hook that selects a text-encoder SOURCE the way the DiT
file already does, and inventing one is a decision left to a later phase.
`int8_convrot` (either file) and the pruned-vocabulary text encoder remain
refused, header-only (no multi-GB tensor read), with reasons naming
phase-plan items 13 and 10 respectively.

The flat "FP16" DiT is bit-exact under `official.bfloat16().half()`, not
`official.half()` directly (verified on sampled tensors): the repack went
through a bf16 cast, so it carries bf16 precision under an FP16 label. Loading
it at this loader's bf16 default is therefore bit-identical to casting
`official/`'s FP32 transformer to bf16 — a user picking the flat FP16 file
expecting extra precision over bf16 gets none.

## Memory

Reported by the model card: under 24 GB in bf16 with automatic CPU offload
(~22 GB), and about 8 GB with leaf-level group offloading of the language model.
The AR stage requires the language model and depth decoder resident together, so
offload policy must not split them. `_PEAK_VRAM_GB_BY_KIND` in `routes.py` needs
an entry. Loading the FP32 transformer or the 18.5 GB text encoder requires a
host-RAM budget, one component at a time.

## Progress and cancellation

Upstream's progress bar covers flow-matching chunks only. For a 300-second song
that is a small fraction of wall time: the AR stage runs 7,500 sequential LM
steps, each with 7 depth-decoder sub-steps, and would appear frozen. The port
must report AR progress in frames and accept cancellation between frames, then
report flow progress as `chunk * steps + step`. Weighting between the two stages
should come from measurement, not a guess.

## Phase plan

Each numbered item is one commit, independently verifiable.

1. **Vendor and port.** Music3 model modules under
   `backend/core/models/minimax_music3/vendor/`, attention re-pointed at the
   SushiUI conduit, plus the ported plain pipeline class. Standalone smoke script;
   host-RAM budget announced before it runs.
2. **Loader and registry.** `ModelType`, directory detection, load dispatch,
   component wiring. Loads the `official/` tree only — see
   [Which tree the loader reads](#which-tree-the-loader-reads). Verify detection
   on the real checkpoint before anything downstream.
3. **Pipeline backend: txt2aud.** `pipeline_backends/minimax_music3.py`, staged
   offload, AR + flow progress, cancellation, **frame-code sidecar**.
4. **API.** Routes accepting the new arch, `param_defaults.py` with the table
   above, `arch_capabilities.py` refusals with reasons, `openapi.yaml` in the
   same commit.
5. **Frontend txt2aud.** Caption and lyrics as distinct fields, schema-driven
   defaults, duration UI stating the upper-bound semantics.
6. **Caption rewriter.** Backend music mode plus validator, and its UI.
7. **Extend.** AR resume from the sidecar; `outpaint/audio`; frontend branch.
8. **Repaint.** Both modes above; `aud2aud`; frontend branch.
9. **The flat key remap.** One remap serving the flat safetensors *and* GGUF,
   since their tensor names are identical: split the fused QKV, rename the
   `.gamma`/`.beta` norms, unfold the condition encoder out of the DiT file, and
   split the merged language model and depth decoder. Configs continue to come
   from `official/`.
10. **The pruned vocabulary.** The offset arithmetic and vocab mask in the AR
    loop, against `embed_tokens_prefill` / `embed_tokens_audio` /
    `lm_head_pruned`. Checkpoint-contract code — verify against the full-vocab
    path by generating the same seed both ways and comparing codes.
11. **GGUF containers.** A reader in-repo (no new dependency), wired to item 9's
    remap, F32/F16/BF16 tensors first.
12. **Q8_0 residency.** Packed weights with dequantisation at use, following
    `convrot_int8_linear.py`. A dequantise-on-load shortcut is not worth
    shipping — see the GGUF section. Then a BF16 A/B on output.
13. **INT8 ConvRot**, which item 9 has by then unblocked, plus its own A/B.
14. **Docs.** `MODEL_FACTS.md` row with measured numbers, architecture counts in
    `AGENTS.md` / `ADD_A_MODEL_ARCHITECTURE.md` / `ARCHITECTURE_MAP.md`,
    `REQUEST_LIFECYCLE.md`, and a `DOC_MAP.md` row for this document.

A structural prerequisite inside item 4: audio defaults are currently flat
single-architecture dicts. Video already has `video_defaults_for_arch` and its
outpaint/inpaint twins; audio has no equivalent because ACE-Step was the only
audio architecture. The per-arch overlay mechanism must be introduced for audio
rather than branching on the architecture at each call site.

## Training forward-compatibility

Training is a later scope, but two findings decide whether it is reachable, and
both are recorded now so phase 1 does not foreclose them.

- **Flow-stage (DiT) training is reachable.** It needs ground-truth latents, which
  need the DAV encoder — and the encoder is published in `official/dav.pth`
  (`encoder.*`, `mean_proj`, `logs_proj`, `flow`), even though it is absent from
  the diffusers component set. Cost: reimplementing the encoder and remapping its
  keys. The 128-channel latent is two folded 64-channel mono streams, so stereo
  encoding is two mono passes stacked.
- **AR-stage (LM) training is blocked** on the RVQ tokenizer's encoder, which is
  not published anywhere in the release. Without it, audio cannot be turned into
  the semantic and residual codes the LM is trained to predict.

So a LoRA over the DiT is the tractable target and should be assumed by the
component wiring; LM-side training should not be promised. The frame-code sidecar
from phase 1 doubles as a training-data artifact for the flow stage.

## Verification gates

- Snapshot revision matches `manifest.json`; every LFS file's size and SHA-256
  match; no pointer files remain. *(Met.)*
- Vendored code carries its Apache-2.0 header and records the upstream commit.
- `config.rope_parameters["rope_theta"] == 1e6` after loading the language model
  (not `config.rope_theta`, which is `None` on transformers 5.1).
- Component class and tensor-shape census matches the table above.
- Loading Music3 does not select the ACE-Step or H3 path, and vice versa.
- A short deterministic smoke generation (10–15 s) produces finite, non-silent
  44.1 kHz stereo audio within the clamped range.
- Extend and repaint reproduce the preserved span **sample-exactly**; this is a
  gate, not an impression.
- Every backend edit passes `venv/Scripts/python.exe -m py_compile` *and* a real
  import.

## Current status

- Official snapshot and flat artifacts: staged and verified under
  `M:/model/minimax-music3/`, provenance in `manifest.json`.
- Upstream pipeline code: vendored and ported (phase plan item 1).
- Backend: phases 1-10 of the phase plan are implemented -- vendor/port,
  loader/registry, txt2aud, API/`param_defaults.py`/`arch_capabilities.py`
  wiring, extend (`/generate/outpaint/audio`), repaint
  (`/generate/aud2aud` with `mode="repaint"`, both the "regenerate" and
  "rerender" sub-modes), the flat key remap
  (`core.models.minimax_music3.flat_remap`), and the pruned-vocabulary remap
  (`core.models.minimax_music3.pruned_text_encoder_remap`). The flat remap
  covers the flat DiT (QKV split, `.gamma`/`.beta` norm rename, condition
  encoder unfolded out) and the flat NON-pruned text encoder (language model
  + RVQ depth decoder split apart, including the `model.audio_extra_embedding`
  -> `audio_embeddings` cross-component rename); both are proven total
  against the vendored classes' own `state_dict()` keys and numerically
  verified against the real snapshot. `load_minimax_music3_from_path` now
  loads a flat, non-quantized DiT file
  (`minimax_music3_dit_{fp32,fp16}.safetensors`) when pointed at one
  directly, with every other component still sourced from `official/`; the
  NON-pruned flat text-encoder builder is implemented and tested but not
  wired into that dispatch (no existing detection hook selects a
  text-encoder source the way the DiT file already does), same status as the
  PRUNED builder (item 10) below.

  **Item 10 (the pruned vocabulary) has landed.** The pruned flat text
  encoder (`minimax_music3_text_encoder_pruned_bf16.safetensors`) fuses each
  layer's `self_attn.qkv_proj` / `mlp.gate_up_proj` (GQA-uneven for the
  language model, equal thirds for the RVQ depth decoder) ON TOP OF the
  vocabulary split this section of the doc already described --
  `pruned_text_encoder_remap` unfuses both and splits the vocabulary into a
  real `Qwen3ForCausalLM`, PATCHED (its default `lm_head` removed,
  `lm_head_pruned` [16385, hidden] and `model.embed_tokens_audio` [16384,
  hidden] attached as new leaf modules; `config.vocab_size` set to the
  checkpoint's own 151,675 text rows). `core.models.minimax_music3.vocab_view`
  resolves which layout a loaded `language_model` is (by attribute presence)
  and routes the AR loop's three checkpoint-contract operations (embed text,
  embed a semantic code, compute audio logits) through it; the full-vocabulary
  path is untouched numerically (verified: 233 pre-existing MiniMax Music 3
  tests still pass, including every AR-loop/resume/frame-code test).
  `build_language_model_and_depth_decoder_from_pruned_flat_text_encoder` is
  implemented and tested (tiny real round-trip) but, like the non-pruned
  builder, not wired into `load_minimax_music3_from_path`'s directory
  detection.

  Which `lm_head_pruned` row is end-of-audio was DETERMINED, not assumed: row
  0 is bit-identical (bf16, 0.0 max abs diff) to `official/language_model`'s
  `lm_head.weight[AUDIO_END_TOKEN_ID]`, and rows 1..16384 are bit-identical to
  `lm_head.weight[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET+16384]` -- semantic code
  `c` lives at row `c + 1`. Every weight this phase touches (the fused
  per-layer projections' unfused pieces, both vocab tables, the pruned
  variant's shared body layers) was verified bit-identical, in bf16, to
  `official/`'s corresponding tensor, against the real snapshot -- not merely
  "close".

  **The verification gate found a real, non-fixable divergence, and the
  primary cause is the sampler, not the GEMM.** Generating the same seed
  through the full-vocab path and the pruned path produces DIFFERENT sampled
  frame codes from the first decode step onward, despite `text_ids`,
  `text_embeds`, `last_hidden` (all 36 layers), and (on CPU) the restricted
  logits themselves being proven BIT-IDENTICAL. The PRIMARY mechanism is
  `_sample_top_k`'s `torch.multinomial`: its RNG consumption depends on the
  category count, so a 200,000-wide call and a 16,385-wide call advance a
  identically-seeded generator differently and pick a different class even
  when fed bit-identical restricted logits (measured: 152/200 GPU trials and
  200/200 CPU trials mismatched). A SECONDARY, smaller effect is that the
  restricted logits are not always bit-identical between the two paths ON
  GPU -- `lm_head_pruned(last_hidden)` vs. `lm_head(last_hidden)` sliced to
  the same 16,385 rows differ by GEMM output-shape-dependent bf16 rounding
  (up to 0.03125 in bf16, ~3.8e-6 in fp32; CPU is exactly bit-identical, 0 of
  32,770 positions differ in both dtypes) -- but this is not the dominant
  cause: bit-identical restricted logits already fail to reproduce the same
  sample most of the time. The gate that IS meetable, and stronger than
  originally claimed: feeding ONE sampler the SAME restricted logit vector
  from both paths agrees -- this is what the argmax and top-50-by-
  conditional-logit-set check already established. Every alternative
  hypothesis for the divergence (EOA row position, code-offset arithmetic,
  the GQA-uneven LM split, the depth decoder's equal-thirds split, dropping
  the mask) was falsified against the real snapshot. Full account in
  `core.models.minimax_music3.vocab_view.PrunedVocabView`'s docstring.

  **Design consequence:** a seed does not reproduce the same song across the
  two text encoders. Making seeds portable would require sampling over the
  restricted 16,385-wide vector on BOTH paths -- deliberately not done here,
  because it would change the full-vocab path's output, and that path is the
  shipped reference existing songs were generated with. Songs remain
  reproducible by their stored frame codes (the sidecar) regardless of which
  text encoder generated them.

  **Item 11 (GGUF containers) has landed.** `core.models.common.gguf_container`
  is a native GGUF v3 reader (magic/version/metadata/tensor-info parsing, all
  13 metadata value types including nested arrays, memory-mapped lazy tensor
  access) with NO `gguf` pip dependency, per this section's own decision --
  deliberate even though `gguf` is already a dependency for MiniMax-H3's text
  encoder (`core.models.minimax_h3.te_gguf_native`), because this format is
  small enough to read in one file and the dependency buys nothing here.
  Materializes F32/F16/BF16 tensors only; any other GGML type (Q8_0 above
  all) is refused HEADER-ONLY (`gguf_container.refuse_unsupported_tensor_types`,
  no tensor byte read) with a reason naming item 12. The dim-order convention
  (GGUF's `ne[]` is `reversed(torch_shape)`) was verified against the real
  staged DiT's fused `to_qkv.weight` -- `[2048, 6144]` on disk, `[6144, 2048]`
  in torch, a deliberately non-square shape -- and cross-checked against the
  installed `gguf` package's own `GGUFReader` (dev-time verification only;
  production code never imports it). `core.models.minimax_music3.loader` now
  accepts a `.gguf` DiT file everywhere a flat `.safetensors` DiT file already
  was (`is_minimax_music3_gguf_dit`, extended `detect_minimax_music3_layout`,
  `build_transformer_and_condition_encoder_from_gguf_dit`), routing through
  item 9's `flat_remap.apply_flat_dit_state_dict` UNCHANGED -- proven against
  the real `M:/model/minimax-music3/diffusion_models/minimax_music3_dit_BF16.gguf`
  (374 tensors, F32×226 + F16×148, header parse ~5 ms, no out-of-range
  tensors) end to end into the vendored `MiniMaxMusic3Transformer1DModel` /
  `MiniMaxMusic3ConditionEncoder`, with per-tensor bit-exactness checks
  against `official/`.

  **What the "BF16" GGUF DiT actually equals was determined, not assumed.**
  Per tensor: a GGML-F32 tensor (226/374) is bit-identical to `official/`'s
  true FP32 weight; a GGML-F16 tensor (148/374, including the fused
  `to_qkv`) is bit-identical to `official.half()` taken DIRECTLY from the
  FP32 weight -- NOT via a bf16 detour, unlike the flat "fp16" safetensors
  DiT (this section's own earlier finding), which IS
  `official.bfloat16().half()`, a bf16-rounded value losslessly repacked
  into an fp16 container. Consequence: loading the GGUF file at this
  loader's bf16 default does NOT reproduce `official.bfloat16()`
  bit-exactly for the 148 GGML-F16 tensors -- casting an already
  fp16-rounded value to bf16 is a double rounding, measured exactly:
  0.00390625 (2⁻⁸) max abs diff on `proj_in.weight`, a GGML-F16 tensor,
  against 0.0 on `time_proj.weight`, a GGML-F32 tensor. The flat "fp16"
  safetensors DiT's own residual against `official.bfloat16()`, measured the
  same way on the same tensor, is 2.98e-08 -- four orders of magnitude
  closer, i.e. exact for any practical purpose. **At this loader's bf16
  default, the GGUF file is therefore the WORSE of the two sources, not a
  wash**: a user picking it for the smaller download is trading that for up
  to 2⁻⁸ of extra rounding on roughly 40% of the DiT's tensors. The loader's
  `(GGUF, remapped)` / `(flat, remapped)` log line states this fact at load
  time (`load_minimax_music3_from_path`), not only here.

  The pruned-vocabulary GGUF text encoder is readable by
  `build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder`
  (mirroring the safetensors pruned builder from item 10 exactly, down to
  its representation choice and gate ordering; NOT wired into directory
  detection, same status as every other text-encoder builder in this
  module) -- but the real staged
  `text_encoders/minimax_music3_text_encoder_pruned_Q8_0.gguf` (328 tensors:
  Q8_0×169 + F32×155 + BF16×4, matching this section's own census) is
  ALWAYS refused by this builder today, header-only, in ~2 ms, naming item
  12 -- proven against the real file, not a fixture. The 4 real BF16
  tensors it does carry were independently verified bit-identical (in bf16)
  to `official/rvq_depth_decoder`'s `audio_embeddings.weight` and
  `pos_embedding.weight`, confirming this reader's BF16 bit-reinterpret path
  against real (not only synthetic) data. A future all-F32/F16/BF16 pruned
  GGUF text encoder would load through this same builder unchanged, past
  the refusal gate -- proven with a tiny all-F32 fixture exercising the full
  `pruned_text_encoder_remap` path end to end.

  **Item 12 (Q8_0 residency) has landed for the pruned GGUF text encoder.**
  `core.models.common.gguf_container.GGUFStateDict.get_q8_0_packed`
  materializes one Q8_0 tensor's raw block layout (an `(out, in)` int8 codes
  tensor + an `(out, in // 32)` float16 per-block scale, `__getitem__`
  itself still refuses Q8_0 unchanged) -- verified against the real staged
  `minimax_music3_text_encoder_pruned_Q8_0.gguf`: every one of its 169 Q8_0
  tensors' declared byte length equals `n_elements // 32 * 34` exactly, and
  a dequantized sample matches the sibling bf16 safetensors file to a
  relative RMS of 0.537%-0.738% (25-tensor sample; mean 0.553%) and a max
  absolute error of 0.0004-0.0042 -- consistent with Q8_0's own per-block
  quantization noise floor, not a layout or split bug, and NOT bit-identical
  (Q8_0 is lossy by construction, unlike this reader's F32/F16/BF16 path).
  `core.models.common.gguf_q8_0_linear.GGUFQ8_0Linear` is the packed Linear
  this feeds: it dequantizes ONCE PER DEVICE MOVE (cached across every
  forward until the next `.to()`/`.cuda()`/`.cpu()` call, which drops the
  cache via an `_apply` override) rather than once per forward -- the AR
  stage this text encoder serves calls the language model up to ~9,000
  times per generation (design doc, "Autoregressive stage"), so re-expanding
  an 8B-parameter stack on every call was ESTIMATED FROM MEMORY BANDWIDTH
  (not benchmarked) to cost ~10-25 ms of pure memory traffic alone at a
  representative ~1-2 TB/s -- well over budget against a 40 ms/frame
  real-time target -- and was rejected outright on that estimate, not merely
  deprioritized. Stated as an estimate deliberately: this repo's rule against
  presenting an unmeasured number as measured applies to its own design
  justifications, not only to a checkpoint's claims.
  `core.models.minimax_music3.pruned_text_encoder_q8_0_remap` reuses item
  10's key plan and fused-projection splits UNCHANGED, splitting a fused
  tensor's packed `(codes, scale)` pair along the same output-row ranges as
  the dense split (Q8_0 blocks run along `in_features`, never across
  `out_features`, so this needs no dequantization first) -- cross-checked:
  every split piece, dequantized, matches item 10's independently
  bit-identical-to-`official/` dense remap to the same ~0.5% noise floor as
  an unsplit tensor, not a larger one. A new builder,
  `core.models.minimax_music3.loader.build_language_model_and_depth_decoder_
  from_pruned_gguf_q8_0_text_encoder`, constructs the REAL patched
  `Qwen3ForCausalLM` + `MiniMaxMusic3RVQDepthDecoder` from the real staged
  file end to end (not a fixture): 289 `GGUFQ8_0Linear` modules installed
  (169 source Q8_0 tensors, expanded by the qkv/gate_up splits -- 253 in the
  language model including `lm_head_pruned`, 36 in the depth decoder), zero
  stranded meta tensors, a real forward pass on an installed layer produces
  finite output, and it does NOT touch `load_minimax_music3_from_path`'s
  directory-detection dispatch, same "implemented and tested, not wired"
  status every other text-encoder builder in this module carries. The
  existing DENSE pruned-GGUF builder (`build_language_model_and_depth_
  decoder_from_pruned_gguf_text_encoder`) is UNCHANGED and still refuses
  Q8_0 header-only -- this is a wholly additive sibling, not a replacement.

  **A first version of this feature was reported, measured, and rejected
  before it shipped.** It let the packed `qweight`/`qscale` buffers move
  under `.to(device)` like any other buffer, alongside the cached dense
  mirror, on the SAME device. On a real device-move round trip that measured
  correctly (packed buffers freed on return to CPU, no leak) but it missed
  the actual failure mode: for the WHOLE AR stage, once the first forward per
  layer built the dense mirror, BOTH the packed source AND the dense mirror
  sat on the GPU together -- the module's GPU footprint was the full
  bf16-equivalent size PLUS the packed bytes riding along, i.e. WORSE than
  loading the plain bf16 file, not better, on exactly the card (24 GB,
  against the model card's own ~22 GB bf16-with-offload figure) this feature
  was meant to help. That is the hollow feature this section already warned
  against, arriving in a new shape rather than being avoided by naming it
  once. The fix is a placement rule, not a different algorithm: `qweight`/
  `qscale` are now PINNED host-resident for the module's whole life
  (`GGUFQ8_0Linear._apply` no longer forwards a device-changing call to
  them); the first forward on a given device copies them to that device as
  TRANSIENT temporaries for the one dequantize call, and only the resulting
  dense mirror is cached and kept resident there. The once-per-device-move
  (not once-per-forward) dequantization timing is UNCHANGED -- only WHERE the
  packed source lives during and after that dequantization changed.

  **The residency claim, measured on the real staged file, both arms.** Two
  independent audits agree the FIRST published host-RAM number (process RSS:
  10.581 GB Q8_0 vs 20.767 GB bf16, reported as 49.05% lower) was overstated
  by a load-path artifact, not a real difference in what the two arms hold:
  the bf16 arm's builder calls `read_state_dict`, which materializes the
  WHOLE 16.7 GB dense dict in memory, then remaps it (renaming keys) and
  clones every fused-projection split -- that intermediate heap (measured
  +4.06 GB of it) was still resident, un-garbage-collected, at the moment RSS
  was sampled, while the Q8_0 arm reads each tensor lazily from a memory-mapped
  file it closes before construction finishes (+1.0 GB of comparable
  overhead). Both arms' PROCESS RSS numbers are real measurements of what
  each arm's LOAD PATH costs, not of the two checkpoints' actual resident
  bytes, so they overstate the comparison in bf16's favor of being worse than
  it is.

  The number that survives scrutiny is header-only tensor-byte arithmetic on
  the two real files, which has no load-path artifact to inflate or deflate:
  the Q8_0 GGUF's 328 tensors total 9.589 GB on disk (matching its own file
  size) against the bf16 safetensors' 328 tensors at 16.707 GB (also matching
  its file size) -- a delta of **~7.12-7.14 GB, ~42.6-42.75%**, two
  independently-computed figures that agree with each other and, expectedly,
  with the 42.6% disk-footprint figure already stated above (both are
  dominated by the same tensor bytes). **Say ~42.7%, not 49%: the corrected,
  defensible host-RAM saving is ~42.7%, not the ~49% the process-RSS
  measurement first reported.**

  VRAM is unaffected by the host-RAM correction above (host RSS and VRAM were
  measured by different code paths). Its number ALSO CHANGED across this
  section's own history, for a different and legitimate reason: a real fix,
  not a methodology correction. First measured, `torch.cuda.
  max_memory_allocated` after every packed layer's dense mirror is forced
  resident was 16.924 GB (Q8_0) against 16.695 GB (bf16) -- 1.37% (0.229 GB)
  higher. The correct explanation for that transient was the fp32 working
  PAIR the first version of `dequantize_q8_0` allocated for one layer's
  dequantization: a widened-to-float32 codes tensor AND a separately-
  materialized expanded-scale tensor of the SAME size (~0.402 GB each for
  the largest such tensor, `mlp.gate_up_proj` at 24576 x 4096, ~0.805 GB
  together) -- NOT "each layer's packed copy briefly co-resident with the
  growing set of dense mirrors" (that copy is ~0.105 GB for the same layer,
  an order of magnitude too small to explain the number). `dequantize_q8_0`
  was rewritten to an IN-PLACE, BROADCAST form (multiply the widened codes
  tensor by `scale.unsqueeze(-1)` directly, relying on broadcasting rather
  than pre-expanding the scale to the full `(out, in)` shape), cutting the
  transient's SOURCE in roughly half by construction; RE-MEASURED after the
  rewrite, the real peak fell to **16.773 GB -- 0.47% (0.078 GB) higher than
  bf16's 16.695 GB**, not the theoretical ~0.402 GB a single remaining
  full-size buffer would predict (the caching allocator's own reuse behavior
  across the sequential per-layer loop accounts for the rest of the gap
  between the back-of-envelope estimate and the measured number; the
  measured number is what is reported). The two arms' STEADY-STATE allocated
  VRAM (`torch.cuda.memory_allocated` after every layer has been touched)
  are unaffected by any of this and remain 16.704 GB (Q8_0) vs 16.695 GB
  (bf16) -- a 9 MB difference, equal within rounding, which is the property
  the placement fix (not the in-place-dequant fix) was built to produce.
  **Say this plainly, in the terms it was asked for: Q8_0 residency saves
  host RAM and disk; it does not reduce VRAM during the autoregressive
  stage, because the language model and depth decoder must be co-resident
  and dense to compute.** A genuine VRAM-during-compute reduction would need
  a block-quantized GEMM kernel that reads the packed representation
  directly and never materializes a dense mirror at all (as llama.cpp's own
  CUDA Q8_0 kernels do) -- named here as the path forward so it is not
  re-derived later: no such kernel exists in this repo or its pinned
  dependencies, and per the no-new-dependency constraint this phase was
  built under, none is added here. What phase 12 actually delivers: a 42.6%
  smaller download/disk footprint, and a ~42.7% host RAM reduction (header-
  only tensor-byte arithmetic, not process RSS) for however long the
  component sits off-GPU under the pipeline backend's staged offload. It
  does NOT yet accrue to this repo's "keep models hot" cross-generation
  residency mode -- that mode is not wired for MiniMax Music 3 at all (no
  architecture-specific gap; the mechanism simply has not been extended to
  this arch), so the claim is scoped to staged offload only, not stated more
  broadly than what is actually wired.

  Placement is also PINNED, not merely documented: `qweight`/`qscale` never
  move under `_apply` (only `bias` does), and `_materialized_weight`
  self-heals if something outside that path relocates them. The one known
  gap: `diffusers/hooks/group_offloading.py` and `accelerate/hooks.py`'s
  `AlignDevicesHook` both bypass `_apply` entirely (direct `buffer.data`
  reassignment / `set_module_tensor_to_device`) and, if ever applied to this
  architecture's language model, would strand the packed buffers on GPU
  between forwards despite the self-heal (which only runs on the NEXT use).
  Neither is applied to this architecture today -- the shipped staged
  offload is a plain `component.to(device)`, no hooks -- so this is latent,
  not live, but `GGUFQ8_0Linear` is INCOMPATIBLE with group offloading or an
  `AlignDevicesHook` being wired onto a module holding it; see
  `gguf_q8_0_linear.py`'s own docstring for the same note, since a reader of
  either document should find it.

  Reproduction: `tmp/minimax_music3_q8_0_vs_bf16_arm_q8_0.py` and
  `tmp/minimax_music3_q8_0_vs_bf16_arm_bf16.py` -- one-shot measurement
  probes against the real staged files (not test-suite code; see the
  "Current status" note below on why they stay in `tmp/` rather than
  becoming tests). The fragile PROPERTIES (placement invariant, cache
  invalidation on `load_state_dict`, the dequant round trip, row-split
  exactness, bias refusal) are covered by
  `backend/tests/minimax_music3_gguf_q8_0_linear_test.py` instead, which
  needs no checkpoint.

  **Known debt (CLOSED, see below): tracked here rather than left implicit**:
  like every other GGUF-phase text-encoder builder in this module (items
  9-11), the new
  `build_language_model_and_depth_decoder_from_pruned_gguf_q8_0_text_encoder`
  is implemented and tested against the real file but is NOT wired into
  `load_minimax_music3_from_path`'s directory-detection dispatch -- there is
  still no hook in that dispatch that selects a text-encoder SOURCE the way
  the DiT file already does. Four builders (non-pruned flat, pruned flat,
  pruned GGUF dense, pruned GGUF Q8_0) are now implemented and unreachable
  from a real load; wiring that selection is accumulating across the GGUF
  phases and remains a decision for a later phase, not resolved here.
  Relatedly and deliberately: MiniMax Music 3 is NOT added to
  `core.models.common.int8_runtime_quantize.QUANTIZED_LINEAR_ARCHS` /
  `RUNTIME_INT8_ARCHS` by this phase. Adding it would advertise a
  `quantized_gemm_mode` capability (the `"w8a8"` per-generation toggle those
  tuples gate, served by `backend/api/quantized_gemm.py`) that does not
  actually exist for an architecture whose only quantized-Linear builder is
  unreachable from a real load -- the same "do not claim what is not wired"
  discipline as the paragraph above, recorded so a later phase does not add
  the entry reflexively when wiring the dispatch.

  Item 13 (INT8 ConvRot, plus its own A/B) was not done AT THE TIME this
  paragraph was written; `int8_convrot` (either flat safetensors file) was
  then still refused (header-only, no multi-GB read) with a reason naming
  phase 13. **Item 13 has since landed -- see its own status entry below.**
  Item 14 (docs) remains not done.

  **The wiring above was closed in a follow-up commit.** The mechanism
  chosen is a load-time file selection, not a post-load hot-swap: it mirrors
  MiniMax-H3's existing `te_override` (`POST /models/load`'s
  `text_encoder_file` field), the same field, generalised rather than
  duplicated -- `ModelLoader._refuse_load_time_te_choice` now accepts
  `minimax_h3` and `minimax_music3` for `text_encoder_file` (still refusing
  `clip_projection_file` for music3, which has no projection concept).
  `core.models.minimax_music3.loader.detect_minimax_music3_text_encoder_
  source` reads a named file's own content -- safetensors vs. GGUF, the
  pruned-vocabulary tells, and each GGUF tensor's own declared type -- and
  `build_language_model_and_depth_decoder_from_text_encoder_file` dispatches
  to the matching one of the four builders; none of the four needed a
  behavior change, only a caller.
  `load_minimax_music3_from_path(..., text_encoder_file=...)` then sources
  BOTH the language model and the RVQ depth decoder from that one file
  (mirroring the checkpoint's own merged layout), skipping the corresponding
  `official/` weight-presence checks while still reading every config from
  `official/` unchanged. The pruned vocabulary needed no separate flag:
  `vocab_view.resolve_vocab_view` already dispatches on the LOADED
  `language_model` object's own shape (`hasattr(..., "lm_head_pruned")`), so
  whichever builder ran decides the vocabulary view with no way for the two
  to drift apart. The component-switch catalog surface
  (`backend/core/models/components/component_catalog.py`,
  `POST /models/current/components/switch`) was deliberately NOT extended for
  this: it requires a per-architecture unload-first adapter music3 does not
  have (the generic "no adapter" disabled reason still applies to its
  `text_encoder`/`vae` slots), and building one was not needed to close this
  debt -- the catalog's existing generic `{slot}_origin` read already reports
  `selected_external` for an override load, with no catalog code change.
  Verified: the default (no override) path is BYTE-FOR-BYTE unchanged --
  proven with a real load of `M:/model/minimax-music3/official` (no
  `text_encoder_file`) reporting `text_encoder_origin: architecture_default`
  and every component from `official/`, exactly as before; all four builders
  are reachable and produce the correct vocabulary view, proven structurally
  against fixtures for all four AND with one real arm, the staged 9.59 GB
  `minimax_music3_text_encoder_pruned_Q8_0.gguf`, which built a real
  `Qwen3ForCausalLM` (289 packed `GGUFQ8_0Linear` modules, matching this
  section's own census) and `MiniMaxMusic3RVQDepthDecoder` end to end and
  resolved `PrunedVocabView`.

  **An independent audit of this wiring found and fixed six defects, two of
  them serious.** (F1) `_minimax_h3_te_selection_differs` had no
  `is_minimax_h3_model` guard, so it misfired while a Music 3 model was
  loaded -- a same-file re-request triggered a full teardown and rebuild for
  nothing, and a `clip_projection_file` sent against a loaded Music 3 model
  (newly refused by this phase) destroyed the live model BEFORE the refusal
  raised, leaving no model loaded at all. Fixed with the same guard on that
  side too, pinned by binding the REAL method (not a stub) in the fixtures
  that exercise `_load_model_locked`. (F2) The music3 branch's
  `_save_last_model` call carried no `text_encoder_file`, so a chosen
  encoder did not survive a backend restart -- silently rebuilt from
  `official/language_model` on the next auto-load, a different vocabulary
  view for a pruned source, with no warning. Fixed by passing it through,
  the same way the H3 branch already did. (F3) Every refusal ran only inside
  `load_minimax_music3_from_path`, reached after Step 1's teardown had
  already cleared the live model -- a typo'd path left the user unloaded
  with a 500. Fixed with a header-only preflight in `_load_model_locked`,
  before anything is torn down, mirroring the hybrid preflight already in
  that function; the raised `MiniMaxMusic3TextEncoderRefusal` (a `ValueError`
  subclass, mirroring `MiniMaxH3HybridRefusal`) is mapped to a 400 in
  `POST /models/load`. (F4) The safetensors arm of
  `detect_minimax_music3_text_encoder_source` had no architecture gate --
  anything that was not the DiT signature was assumed non-pruned, so a
  foreign safetensors file was only refused after a multi-GB
  `read_state_dict`. Fixed by running the same header-only key plan
  (`plan_flat_text_encoder_keys`) the real builder runs against the real
  tensors. (F5) The fixtures added alongside F1 stubbed the new gate with
  `lambda *_a: False`, which would have hidden both F1 itself and any
  regression of it from that suite; rebound via `types.MethodType` to the
  real implementation, and a dedicated
  `backend/tests/minimax_music3_text_encoder_choice_test.py` was added
  covering the detector matrix, the gate matrix (both directions, both
  architectures), and the F2/F3 fixes end to end. (F6) `text_encoder_origin`
  was emitted unconditionally, including `architecture_default` on a
  default load -- but `component_catalog._current_origin` reads that key
  BEFORE its own derivation, so the default path's origin changed from a
  derived `model_tree` to a hardcoded `architecture_default`. Fixed by
  emitting the key only when overridden, re-verified against the real
  `official/` tree (the key is now absent, not merely a different valid
  value, on the default path). Two further line-item fixes: `_refuse_load_
  time_te_choice` now names every invalid field in one raise instead of one
  per round trip, and this section's own `text_encoder_file` prose now
  states plainly that Music 3 has no discovery endpoint (unlike H3's
  `GET /models/minimax-h3/text-encoders`, which remains H3-only) and that a
  truncated file is refused by name rather than surfacing a bare parse
  error.

  **A second-pass audit against the four real staged files found the F4 fix
  had a regression, plus two judgement calls.** Running the detector against
  the real, phase-9-verified non-pruned flat bf16 encoder showed it was now
  REFUSED: `plan_flat_text_encoder_keys` -- unlike the membership-only checks
  elsewhere in this module -- reports ANY key it does not explicitly
  recognize, and F4's fix fed it the RAW safetensors header keys, which
  include safetensors' own `__metadata__` entry (a string-string map, not a
  tensor). The tiny synthetic test fixtures happened not to carry that entry,
  so a detector test suite covering only refusal shapes stayed green while
  the one accept path phase 9 verified bit-exact was silently broken.
  Fixed with a single shared helper, `_safetensors_tensor_keys` (`read_
  safetensors_header` minus `__metadata__`), and every tensor-key consumer in
  `loader.py` was audited and routed through it, not just the one that broke
  -- `flat_remap.py` itself was already safe (it only ever plans over a REAL
  state dict's keys, which never carries `__metadata__`). Judgement call 1:
  `minimax_music3_text_encoder_pruned_int8_convrot.safetensors` detected as
  `flat_pruned` rather than being refused -- truthful about the vocabulary
  layout, but the actual int8 refusal only fired later, inside the builder,
  which is reached AFTER the F3 preflight had already let the request
  through and torn the live model down -- reopening exactly the hole F3
  closed, for this one quantization. Fixed by adding the same header-only
  `_header_looks_quantized` check to the detector itself, before it commits
  to a kind string, so `_load_model_locked`'s F3 preflight (which calls only
  the detector) now refuses it pre-teardown too. Judgement call 2 (the DiT-
  as-text-encoder and truncated-file refusals) was verified unaffected by
  both fixes, against the real files, not only re-asserted from memory. All
  four real text-encoder files plus the real flat DiT file are now in
  `backend/tests/minimax_music3_text_encoder_choice_test.py`'s own matrix
  (skipped cleanly, matching `minimax_h3_te_int8_convrot_test.py`'s own
  convention, when `M:/model/minimax-music3` is not present) specifically so
  an accept-path regression like this one cannot pass unnoticed again.
- **Item 13, INT8 ConvRot, is DONE.** Both staged ConvRot artifacts load: the
  flat DiT (`minimax_music3_dit_int8_convrot.safetensors`, 2.50 GB) and the
  pruned text encoder (`minimax_music3_text_encoder_pruned_int8_convrot.
  safetensors`, 9.20 GB, which composes the pruned vocabulary AND ConvRot
  together). Almost entirely wiring: `core.models.common.convrot_int8_linear`
  (`ConvRotInt8Linear`, `swap_linears_to_convrot_int8`,
  `require_convrot_int8_runtime`) is reused UNCHANGED from MiniMax-H3's own
  ConvRot support, and the new `core.models.minimax_music3.convrot_remap`
  module places the ConvRot sidecars (`.weight_scale`, `.comfy_quant`) at the
  SAME destinations item 9's `flat_remap.plan_flat_dit_keys` /
  `apply_flat_dit_state_dict` and item 10's `pruned_text_encoder_remap.
  plan_pruned_text_encoder_keys` / `apply_pruned_text_encoder_state_dict`
  already compute for the corresponding dense `.weight` -- those two
  functions are called UNCHANGED for every dense tensor, including the
  quantized `.weight` tensors themselves (a row-wise split of int8 codes by
  a fused projection's OUTPUT rows is exact, because ConvRot's groups run
  along the K/`in_features` dimension, never across output rows -- the same
  argument item 12 makes for Q8_0). The one new per-arch artifact is
  `_int8_convrot_source_layers` (`core.models.minimax_music3.loader`), a
  header-only census that reads real bytes only for the tiny `.comfy_quant`
  marker tensors and validates them against the ONE ConvRot contract this
  loader implements (a deliberate duplicate of `minimax_h3.loader._supported_
  int8_convrot_marker`, not an import -- each architecture owns its own
  quantization CONTRACT validator in this repo, by existing convention).

  **Census, against the real files, header-only:** the flat DiT carries the
  SAME 374 dense (non-sidecar) tensor names item 9 already established, with
  144 of the `.weight` entries (36 layers x 4 quantized module kinds:
  `self_attn.to_qkv`, `self_attn.to_out`, `ff.ff.0.proj`, `ff.ff.2`) also
  carrying a validated ConvRot marker -- 662 total tensors (374 + 144
  `.weight_scale` + 144 `.comfy_quant`). The pruned text encoder carries the
  SAME 328 dense tensor names item 10 already established, with 160 `.weight`
  entries quantized (144 in the language model's 36 layers + 16 in the depth
  decoder's 4 layers, 4 quantized module kinds each: `self_attn.qkv_proj`,
  `self_attn.o_proj`, `mlp.gate_up_proj`, `mlp.down_proj`) -- 648 total
  tensors. Both `plan_flat_dit_keys` and `plan_pruned_text_encoder_keys`
  report ZERO unrecognized keys against these dense sets, unchanged from
  their item-9/10 behavior on the non-ConvRot siblings.

  **A real load, not only a structural assertion:** `build_transformer_and_
  condition_encoder_from_flat_dit` against the real ConvRot DiT produces 216
  `ConvRotInt8Linear` modules (144 source layers; the 36 fused `to_qkv` each
  expand to 3 destination Linears, so 108 non-fused + 108 fused-expanded =
  216), with no stranded meta tensor. `build_language_model_and_depth_
  decoder_from_pruned_flat_text_encoder` against the real ConvRot text
  encoder produces 252 `ConvRotInt8Linear` in the language model (36 layers x
  7: o_proj + 3-way qkv split + 2-way gate_up split + down_proj) and 28 in the
  RVQ depth decoder (4 layers x 7), resolves `PrunedVocabView` (not
  `FullVocabView`), and again leaves no stranded meta tensor. A full
  `load_minimax_music3_from_path` call with the ConvRot DiT as `model_path`
  AND the ConvRot text encoder as `text_encoder_file` composes both in one
  load, proving the DiT's ordinary model-path selection and the `text_
  encoder_file` override reach the SAME two files this status entry
  describes, together.

  **The BF16 A/B.** Chosen at the DEQUANTIZED-WEIGHT level, not a full
  generation: the AR stage the design doc's own "Progress and cancellation"
  section describes (7,500 sequential language-model steps for a 300-second
  song, each with 7 depth-decoder sub-steps) makes a full real generation
  prohibitively slow for a per-commit verification, and the two sampler
  widths (BF16 vs. a ConvRot-quantized run) are not directly comparable frame
  code as sequences for the SAME reason item 12's own A/B section states for
  Q8_0 -- codes diverge with the RNG path and the quantization noise floor
  the moment they diverge at all, so a meaningful A/B has to compare
  something that has ONE well-defined ground truth per layer: the
  dequantized weight against its BF16/FP16 sibling. `torch.ops.comfy_kitchen.
  dequantize_int8_convrot_weight_dtype` (the same op `ConvRotInt8Linear.
  _dequant_forward` calls) recovers each ConvRot layer's weight in its
  ORIGINAL (un-rotated) basis, directly comparable to the plain weight the
  flat FP16 DiT / BF16 text encoder store for the identical layer. Measured
  on sampled layers from both real files (relative RMS error,
  `((dequantized - reference)**2).mean().sqrt() / (reference**2).mean().sqrt()`):
  **~0.82-0.94% across the DiT's `to_qkv`/`to_out`/`ff.ff.0.proj`/`ff.ff.2`**
  and **~0.87-0.99% across the text encoder's language-model and depth-
  decoder `qkv_proj`/`o_proj`/`gate_up_proj`/`down_proj`** -- consistent with
  an 8-bit per-row-scaled quantization noise floor (in the same neighborhood
  as, though somewhat higher than, item 12's ~0.5% Q8_0 figure, which blocks
  4x narrower along K and so has a finer noise floor). `backend/tests/
  minimax_music3_int8_convrot_test.py` pins this at a generous 2% ceiling
  (not the measured figure, so the test does not silently drift) against
  BOTH real files, alongside the census and structural-build proofs above.
  Both the census figures and the error figures were reproduced
  INDEPENDENTLY (an audit pass, on layers this repo's own test suite does
  not sample), and the composition claim was attacked directly: dequantizing
  each split piece with ITS OWN split scale is bitwise equal to slicing the
  full-tensor dequant, while deliberately shifting one split's scale by a
  single row inflates the error from 0.87% to 26.5% -- proving a scale that
  failed to follow its rows through the split would be loud, not silently
  plausible, and that it does in fact follow them. The marker validator
  itself was also compared line-for-line against MiniMax-H3's
  `_supported_int8_convrot_marker` and found to be the same contract, not a
  looser copy.

  **The honest residency statement, stated plainly because item 12's own
  history shows this is the question that is easy to answer wrongly, and now
  MEASURED, not only argued from structure (independent audit).** ConvRot
  weights stay PACKED (int8 codes + a float32 per-row scale) from disk
  through GPU residency -- `ConvRotInt8Linear.forward` calls comfy-kitchen's
  CUDA path, which quantizes the ACTIVATION and runs an int8 GEMM directly on
  the packed codes (traced in source, not inferred: `dequantize_int8_linear`
  there is the int32-accumulator-to-output epilogue, not a weight dequant),
  when autograd is off -- the inference path, since both builders end in
  `requires_grad_(False)`. On a real 4096x4096 layer, two separate
  processes, bf16 activation, `torch.no_grad()`: resident weight 16.794 MB
  (ConvRot) vs. 33.554 MB (bf16), a 50.05% reduction, and the forward-call
  peak-memory DELTA 23.088 MB vs. 46.268 MB -- BELOW a single dense bf16
  weight's own size, so no dense mirror is materialized at any point during
  the call either. This is a genuine difference from item 12's Q8_0 finding,
  not the same result restated: Q8_0 residency (`GGUFQ8_0Linear`) keeps a
  DENSE MIRROR co-resident once a layer is touched (the AR loop's per-step
  cost otherwise being an unacceptable per-forward dequant of an
  8B-parameter stack), so its VRAM saving is zero during the AR stage by
  construction -- see item 12's own status entry. ConvRot has no such
  mirror. The phase-12 trap (a missing runtime silently falling back to
  dense residency) is structurally impossible here: `require_convrot_int8_
  runtime()` hard-fails at LOAD time if comfy-kitchen or its custom op is
  absent, before any packed weight is installed, rather than degrading.

  **Scope, honestly.** The 50.05%/measured-below-dense-weight numbers above
  are LAYER-LOCAL (one Linear, one forward call) -- they scale by
  construction to the whole DiT/text-encoder stack because every quantized
  Linear behaves identically and independently, but the AR stage's KV cache
  and activation memory were NOT measured and are NOT claimed to shrink by
  this factor (nothing about ConvRot touches either). Header-only tensor-byte
  arithmetic on the real files gives the disk/host-RAM figure directly (no
  process-RSS load-path artifact to correct for, per item 12's own
  methodology lesson): the ConvRot DiT is 2.50 GB against the flat FP16
  DiT's ~4.98 GB -- ~50% -- and the ConvRot pruned text encoder is 9.20 GB
  against the pruned BF16 text encoder's 16.71 GB, ~44.9% (slightly under
  half because the three vocabulary tables, the norms and `tokenizer_json`
  stay dense BF16, unquantized, in both files). **The one condition where the
  VRAM saving does NOT hold:** `ConvRotInt8Linear._dequant_forward` --
  reached when grad is enabled AND the input requires grad -- materializes a
  dense weight for that one call. The inference path never reaches it (both
  builders freeze every parameter), but a future training use of this
  checkpoint would need to account for it explicitly rather than assume the
  inference-path number.

  What is still refused: any quantization semantic other than this ONE
  validated ConvRot contract -- an unrecognized `.comfy_quant` marker (a
  different format, an unread field, an undecodable marker), a scaled file
  with NO marker at all, or a NON-pruned text encoder that is also quantized
  (no such distribution is staged; the pruned layout is the only one Comfy-
  Org published ConvRot for). `MiniMax Music 3` is still NOT added to `core.
  models.common.int8_runtime_quantize.QUANTIZED_LINEAR_ARCHS` -- that table
  governs the PER-GENERATION `unet_quantization` runtime converter, a
  different feature from loading a pre-quantized checkpoint, and this phase
  adds no such converter (see `backend/api/arch_capabilities.py`'s
  `unet_quantization` refusal message, updated in this phase to state that
  distinction rather than point at "a later phase" now that ConvRot LOADING
  has landed).

  **An independent audit of this phase found five fixes, all closed.** (F2,
  medium, a real regression) The DiT builder had LOST the header-only refusal
  for a quantized-but-unmarked file: `_int8_convrot_source_layers` returning
  `{}` for a header with a `.weight_scale` but no `.comfy_quant` marker fell
  through to a full multi-GB `read_state_dict` before being refused, while
  the pruned text encoder builder kept its own equivalent gate. Fixed by
  mirroring the TE builder's gate in the DiT builder exactly: a header that
  `_header_looks_quantized` flags is censused immediately, and an EMPTY
  census raises right there, before `read_state_dict`. The deleted fixture
  this regressed (`_write_quantized_flat_dit_header`, a scale-with-no-marker
  file) is restored as `_write_flat_dit_with_scale_and_no_marker` /
  `test_flat_dit_scale_with_no_marker_is_refused_header_only`, alongside the
  unrecognized-marker case that had replaced it. (F1, medium) A stale comment
  in `loader.py` (near `load_minimax_music3_from_path`'s flat-DiT branch)
  still said int8_convrot "is still refused, inside the builder itself...
  design doc phase 13" -- the exact claim this phase reverses; corrected.
  (F3, low) The same claim in `flat_remap.py`'s module docstring ("refused,
  not half-remapped" naming int8_convrot on either file); corrected to state
  what IS and is not handled by that module specifically (dense remap only;
  the sidecars are placed by `convrot_remap.py`, which calls this module's
  functions unchanged). (F4, low, latent) `_split_convrot_sidecar` reused the
  SAME marker tensor object across every split destination while cloning the
  scale chunks -- the sibling `pruned_text_encoder_remap._apply_splits`
  clones every split piece explicitly, for the stated reason that a future
  `safetensors.save_file` of a split-loaded module would otherwise refuse
  with "tensors share memory". No music3 export path exists today (latent,
  not live), but the sibling had already decided the question; the marker is
  now cloned too, pinned by `test_split_convrot_sidecar_equal_three_way`.
  (F7, style) The comment budget was trimmed: the "why a dim-0 row split is
  exact for ConvRot codes" argument, which appeared verbatim in this file's
  module docstring, in `_split_convrot_sidecar`'s docstring, AND in this
  design doc, now lives only here (a sentence plus a pointer remains at each
  code site).

  **Measured, not only argued from structure, by the same audit** (see the
  "honest residency statement" above, now updated with real numbers): a
  real 4096x4096 ConvRot layer resident at 16.794 MB against a dense bf16
  layer's 33.554 MB (50.05%), with the forward-call peak-memory delta
  (23.088 MB vs. 46.268 MB) measured BELOW a single dense weight's own size
  -- proving no dense mirror is ever materialized, not merely arguing it from
  the module tree. The mechanism was traced in comfy-kitchen's own source
  (an installed, already-pinned dependency -- nothing new was added): the
  CUDA path quantizes the activation and runs an int8 GEMM, and
  `dequantize_int8_linear` there is the int32-to-output epilogue, not a
  weight dequant. And the composition proof (dequantizing a split piece with
  its OWN split scale is bitwise equal to slicing the full-tensor dequant)
  was attacked directly by shifting a split's scale by one row, which
  inflated the error from 0.87% to 26.5% -- proving a misaligned scale would
  be loud, not silently plausible.

  **Two things recorded rather than fixed, per the audit's own
  instruction:** the VRAM numbers above are LAYER-LOCAL and scale by
  construction, but the AR stage's KV cache and activation memory were not
  measured, and `_dequant_forward` (reached only when grad is enabled and
  the input requires grad -- never on the inference path, since both
  builders freeze every parameter) DOES materialize a dense weight for that
  one call; a future training use of this checkpoint must account for that
  explicitly. Separately, `backend/tests/quantized_capability_parity_test.
  py::LoaderComponentNameAnchorTest` was found already failing on
  `['minimax_h3']` in the tree BEFORE this phase's diff -- unrelated to
  MiniMax Music 3 or to INT8 ConvRot, not touched or fixed here, flagged for
  separate attention.
- Frontend: txt2aud/extend UI shipped; repaint's UI branch is BLOCKED on a
  shared-worktree conflict (`frontend/src/components/generation/Img2ImgPanel.tsx`
  was dirty under another session's edits when this phase landed) -- not
  implemented in this phase. The text-encoder-file selection closed above has
  no frontend surface either, for the same shared-worktree reason -- the API
  and loader are usable today via `POST /models/load`'s `text_encoder_file`.

## Revision history

**Revision 2 (2026-08-14)** — audit against the snapshot, the upstream source at
`dafe3733`, and the installed venv. Corrections to revision 1:

- Revision 1 said the venv was not runnable because its base Python was missing.
  It runs; `diffusers 0.38.0` and `transformers 5.1.0` import fine. The real
  blocker is the version gap, which revision 1 did not mention at all.
- Revision 1 planned to load "the modular pipeline components" without noting
  that no Music3 code exists locally or in the installed diffusers. Vendoring is
  now the first phase.
- Revision 1's conversion gate said to merge "only the two official `transformer/`
  shards". The flat DiT also contains the condition encoder, so that comparison
  could never have matched.
- Revision 1 said the quantized text encoder must preserve a custom `Abab` key
  layout. The flat text encoders use plain HF-style keys.
- Revision 1 called `qwen_7B/` opt-in-until-verified. It is unloadable as
  snapshotted; it is now a permanent exclusion.
- Revision 1 left reference audio, cover and repaint as future gates. They are
  now decided on evidence — see [Capability verdict](#capability-verdict).
- Revision 1 dismissed 32 kHz as an SGLang artifact without citing the vocoder
  config; the citation is now `vocoder/config.json`.
- New material: latent/frame layout, generation-parameter table, the frame-code
  state contract, the caption rewriter, progress/cancellation, and the training
  reachability split.

**Revision 3 (2026-08-15)** — added the [Lyrics assistant](#lyrics-assistant)
section, documenting the `format`/`structure`/`complete` modes, the format
invariant, the documented-plus-freeform tag vocabulary, and the
generation-time `warnings[]` surfacing on `POST /generate/txt2aud`. No prior
material was corrected; this is additive, closing citations in
`minimax_music3_lyrics_assistant.py`, `routes.py`, its test file, and
`MusicLyricsAssist.tsx` that pointed at section titles this document never
had.
