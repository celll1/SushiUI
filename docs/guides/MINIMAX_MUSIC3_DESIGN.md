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
- Backend: phases 1-9 of the phase plan are implemented -- vendor/port,
  loader/registry, txt2aud, API/`param_defaults.py`/`arch_capabilities.py`
  wiring, extend (`/generate/outpaint/audio`), repaint
  (`/generate/aud2aud` with `mode="repaint"`, both the "regenerate" and
  "rerender" sub-modes), and the flat key remap
  (`core.models.minimax_music3.flat_remap`). The remap covers the flat DiT
  (QKV split, `.gamma`/`.beta` norm rename, condition encoder unfolded out)
  and the flat NON-pruned text encoder (language model + RVQ depth decoder
  split apart, including the `model.audio_extra_embedding` ->
  `audio_embeddings` cross-component rename); both are proven total against
  the vendored classes' own `state_dict()` keys and numerically verified
  against the real snapshot. `load_minimax_music3_from_path` now loads a
  flat, non-quantized DiT file (`minimax_music3_dit_{fp32,fp16}.safetensors`)
  when pointed at one directly, with every other component still sourced
  from `official/`; the flat text-encoder builder is implemented and tested
  but not wired into that dispatch (no existing detection hook selects a
  text-encoder source the way the DiT file already does). Items 10-14
  (pruned vocabulary, GGUF containers, Q8_0 residency, INT8 ConvRot, docs)
  are not yet done; `int8_convrot` and the pruned text-encoder variant are
  refused (header-only, no multi-GB read) with reasons naming those phases.
- Frontend: txt2aud/extend UI shipped; repaint's UI branch is BLOCKED on a
  shared-worktree conflict (`frontend/src/components/generation/Img2ImgPanel.tsx`
  was dirty under another session's edits when this phase landed) -- not
  implemented in this phase.

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
