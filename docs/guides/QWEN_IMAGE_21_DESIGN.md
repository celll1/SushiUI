# Qwen-Image 2.1 integration design

Status: **proposed implementation contract; implementation not started**

Architecture key: `qwen_image_21`

Upstream model: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)

Upstream checkpoint revision: pin the exact `Qwen/Qwen-Image-2.1` revision used
to build the local artifacts; do not follow `main` implicitly.

Upstream diffusers baseline: PR
[`huggingface/diffusers#14804`](https://github.com/huggingface/diffusers/pull/14804),
merged as `6256aa7` on 2026-09-18.

Training reference: PR
[`huggingface/diffusers#14808`](https://github.com/huggingface/diffusers/pull/14808),
which was still open when this design was written and is evidence, not a
dependency or authority.

This document is the implementation contract for adding Qwen-Image 2.1 to
SushiUI. It covers local artifact construction, generation, image editing,
training, quantized frozen bases, API/frontend wiring, verification, and the
boundaries that must remain explicit until measured.

## 1. Goal and interpretation of parity

The integration must provide the architecture-appropriate equivalent of the
SDXL and SenseNova surfaces:

* load locally without a Hub dependency after artifacts have been prepared;
* txt2img, native image-conditioned editing, multi-reference editing,
  img2img, inpaint, and outpaint;
* native RGBA input/output rather than silently discarding alpha;
* LoRA and full-parameter DiT training, resume, validation samples, and
  generation-loader round trips;
* a frozen INT8 ConvRot base for generation and additive-adapter training;
* attention selection, component offload, keep-hot residency, training block
  swap, activation offload, latent caching, and the shared diagnostics where
  the architecture can implement their semantics exactly;
* capability refusals for every shared feature that has no valid Qwen-Image
  2.1 implementation.

"Parity" does not mean pretending that a single-stream block-causal DiT is an
SDXL U-Net. In particular:

* Qwen-Image 2.1 has no released ControlNet architecture. Native editing is the
  supported conditioning path; `controlnet` training and ControlNet generation
  remain explicit refusals.
* The released text encoder is a Qwen3-VL model and stays frozen in the first
  release. The DiT receives pre-final-norm hidden states with a checkpoint-
  specific template; arbitrary TE replacement would not be equivalent to
  SDXL's CLIP override.
* ReLoRA is not admitted merely because ordinary LoRA works. It opens only
  after merge/reset, resume, and generation round-trip tests exist for this
  architecture.
* FBCache, Spectrum, TREAD, BlockSkip, NAG, regional prompting, and style-KV
  injection are measured or refused independently. A generic UI control is not
  evidence that the architecture consumes it.

## 2. Upstream facts that constrain the design

The released pipeline consists of:

| Component | Released contract | Consequence in SushiUI |
|---|---|---|
| Denoiser | 7.1B, 32 single-stream blocks, 32 heads x 128 dimensions, hidden size 4096, 64-channel unpatched latent tokens | New DiT model and block loop; no U-Net compatibility shim |
| Attention | Block causal: sequence-causal except each image block is internally bidirectional | Preserve image block boundaries; ordinary dense all-to-all attention is numerically wrong |
| Prefix cache | text and condition-image tokens use `t=0` modulation and are invariant across denoising steps | Cache per-layer prefix K/V after the first step; cache flag is part of reproducibility identity |
| Text/image encoder | `Qwen3VLForConditionalGeneration`, context width 4096 | Reuse installed Transformers Qwen3-VL classes; keep processor assets beside the weights |
| VAE | `AutoencoderKLQwenImage21`, RGBA, 64 latent channels, spatial compression 16 | Four-channel pixel path and a distinct VAE class; no reuse of the older Qwen-Image VAE |
| Sampler | flow matching, Euler, velocity target `noise - x0` | `t0` convention and `eps_minus_x0` training declaration |
| Recommended inference | 40 steps, CFG off (`true_cfg_scale=1`) | Per-architecture defaults; shared SD defaults must not leak in |
| Editing | one pipeline, up to ten reference/condition images | Native edit route owns reference ordering and VLM/VAE dual encoding |

The official model card recommends 2048-class aspect-ratio presets, while the
upstream pipeline's `output_resolution` default is 1024. SushiUI therefore uses
1024 as the safe initial API default and exposes the documented 2048 presets;
2048 is not treated as a minimum.

The upstream implementation has two exact prefill paths:

1. a multi-pass SDPA path, usable without compilation; and
2. a compiled FlexAttention path with a block mask.

The uncompiled FlexAttention fallback materializes an impractical dense fp32
score matrix at high resolution. SushiUI must never select FlexAttention unless
the required compiled path has passed its runtime probe.

## 3. Source and dependency policy

The current environment has Qwen3-VL model/processor classes but does not have
the Qwen-Image 2.1 diffusers classes. The integration therefore vendors the
minimum Apache-2.0 implementation from the pinned diffusers commit:

```text
backend/core/models/qwen_image_21/
  __init__.py
  loader.py
  pipeline_ops.py
  transformer.py
  autoencoder.py
  attention_processor.py
  lora.py
  artifact.py
  upstream.json
```

`upstream.json` records repository, commit, source paths, upstream license,
and local modifications. The same entries must be added to
`docs/legal/THIRD_PARTY_PROVENANCE.md` before vendored code is committed.

Do not vendor Qwen3-VL from Transformers. Load it through the installed
Transformers public classes. Do not upgrade all of diffusers to an unreleased
or newly released version solely for this architecture: that changes every
existing pipeline's runtime underneath the integration.

The SushiUI backend owns the generation loop in `pipeline_ops.py`. It may port
small, attributed preparation helpers from upstream, but must not instantiate
the upstream `QwenImage21Pipeline` as a black box. Owning the loop is required
for progress events, cancellation, offload, adapter sessions, cache identity,
diagnostics, and training/generation parity.

## 4. Artifact layout and why TE/DiT stay separate

The local layout is:

```text
<MODEL_ROOT>/qwen21/
  manifest.json
  diffusion_models/
    qwen_image_2.1_bf16.safetensors
    qwen_image_2.1_int8_convrot.safetensors
  text_encoders/
    qwen3vl_8b_bf16.safetensors
    qwen3vl_8b_int8_convrot.safetensors
  vae/
    qwen_image_2.1_vae_bf16.safetensors
  processor/
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    added_tokens.json
    merges.txt
    vocab.json
    preprocessor_config.json
    video_preprocessor_config.json
    chat_template.jinja
```

For the owner's checkout, `<MODEL_ROOT>/qwen21` maps to the requested local
Qwen directory. Tracked documentation intentionally does not record a
machine-specific drive letter.

TE and DiT are separate artifacts by design:

* the official checkpoint separates them;
* the DiT is about 14.2 GB BF16 and the Qwen3-VL encoder about 17.5 GB BF16;
* generation can release the TE after prompt/reference encoding;
* training can cache conditioning and omit the TE from steady-state residency;
* the two components have independent ConvRot eligibility and validation;
* a combined file would increase mmap pressure, make component selection
  ambiguous, and defeat load-time component offload;
* tokenizer/processor files cannot be embedded usefully as tensors and must
  remain a directory asset either way.

The VAE remains a third file. It is shared by both precision variants and is
not quantized.

### 4.1 `manifest.json`

The manifest is the selectable model entry point and contains no weights:

```json
{
  "format": "sushi-qwen-image-21-v1",
  "model_type": "qwen_image_21",
  "source_revision": "<pinned-hf-revision>",
  "variants": {
    "bf16": {
      "transformer": "diffusion_models/qwen_image_2.1_bf16.safetensors",
      "text_encoder": "text_encoders/qwen3vl_8b_bf16.safetensors"
    },
    "int8_convrot": {
      "transformer": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
      "text_encoder": "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
    }
  },
  "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",
  "processor": "processor"
}
```

The production manifest also stores SHA-256, byte size, tensor count, config
fingerprint, and license/provenance for every entry. Paths are relative and
must resolve inside the manifest directory. Mixed variants are rejected by
default; an explicit advanced component override may select a different TE,
but it becomes part of the model identity and cache key.

### 4.2 Artifact creation

Add `scripts/convert_qwen_image_21.py` with two modes:

* `import-original`: stream official diffusers shards into one BF16 DiT file,
  one BF16 TE file, and one BF16 VAE file; copy processor assets; and write the
  manifest.
* `quantize-convrot`: read the validated BF16 file component by component and
  emit the corresponding INT8 ConvRot file.

The converter must use `safe_open` and bounded layer-at-a-time residency. It
must not materialize the 30+ GB TE+DiT pair in a single Python state dict.

An already-published single-file repack may be imported only after tensor-key,
shape, dtype, config, and source-revision validation. File names and repository
labels alone are not a trust boundary.

### 4.3 Safetensors metadata

Every weight file includes at least:

```text
model_type=qwen_image_21
component=transformer | text_encoder | vae
variant=bf16 | int8_convrot
format=sushi-qwen-image-21-component-v1
source_repo=Qwen/Qwen-Image-2.1
source_revision=<pinned revision>
config_sha256=<canonical config digest>
```

The transformer metadata embeds its full construction config. The VAE metadata
embeds all 64 `latents_mean` and `latents_std` values, the 16x spatial factor,
RGBA channel count, and class name. The TE metadata embeds the Qwen3-VL config
digest, not a hand-maintained partial architecture guess.

Each quantized Linear carries a validated `.comfy_quant` marker, int8 `weight`,
and fp32 `weight_scale`. The accepted marker contract is exactly the shared
`convrot_marker.py` contract: tensorwise INT8, `convrot=true`, group size 256,
2-D weights, and valid scale shape. Any unknown marker aborts the header-only
preflight before the currently loaded model is torn down.

### 4.4 Quantization scope

Quantize eligible `nn.Linear` weights in both DiT and TE. Keep the following in
BF16/FP32 as appropriate:

* embeddings and token tables;
* normalization parameters;
* VAE weights;
* any Linear whose K dimension violates the ConvRot group contract;
* statistically sensitive or unsupported projections identified by the
  calibration gate.

The converter produces a census with eligible, quantized, deliberately kept,
and rejected layers. The loader compares the file against an architecture-
versioned expected census and refuses partial or mixed unknown layouts. The
census numbers are not frozen in this design before the real files are scanned.

## 5. Detection and preflight

Add `ModelType.QWEN_IMAGE_21 = "qwen_image_21"` and
`_keys_look_qwen_image_21(keys, metadata)`. Detection priority is:

1. manifest `model_type` and format;
2. safetensors metadata;
3. a conjunction of architecture-specific signature keys and shapes.

Never classify by filename. A component file by itself may be inspected and
identified, but the normal load endpoint selects the manifest so the loader can
prove that DiT, TE, VAE, and processor assets form one compatible set.

Header-only preflight validates:

* every relative path and digest;
* component/config identity;
* required signature tensors and exact released shapes;
* BF16 versus ConvRot marker consistency;
* complete ConvRot triples;
* processor files and chat-template digest;
* VAE RGBA/64-channel/16x geometry;
* license acknowledgement state if the application requires it.

Only after preflight succeeds may `PipelineManager` tear down the live model.

## 6. Loader ownership and component lifetime

`loader.load_qwen_image_21_components()` returns:

```text
transformer, text_encoder, processor, vae, scheduler,
variant, component_origins, quantization_census, source_revision
```

Construction order:

1. build modules on `meta` from embedded/pinned configs;
2. swap marker-validated Linear modules to `ConvRotInt8Linear` before loading;
3. stream weights directly into their final modules;
4. run missing/unexpected-key checks with an explicit allowlist only;
5. tie the TE weights declared tied by its config;
6. freeze TE and VAE; freeze DiT until a training adapter chooses otherwise;
7. record component origins and a stable model key.

No silent Hub fallback is allowed in normal generation or training. The
artifact creation command is the only network-aware stage.

Component lifetime for generation is encode TE -> release/offload TE -> denoise
DiT -> release/offload DiT -> decode VAE. `cpu_text_encoding`, keep-hot, and
explicit component offload alter placement, not computation.

## 7. Tensor and conditioning contract

Pixels are RGBA `[B,4,H,W]` in the VAE's declared range. RGB inputs are
converted to RGBA with opaque alpha. For Qwen3-VL only, RGBA is composited over
white exactly as upstream; the VAE still receives the original four channels.

The VAE encoder returns `[B,64,1,H/16,W/16]`. Normalize with the 64-element
vectors from the pinned VAE config. The DiT consumes unpatched spatial tokens:

```text
[B,64,1,h,w] -> [B,h*w,64]
```

`img_shapes` lists condition images first and target last. `img_mask` maps VLM
vision slots to 2x2 groups of latent tokens. Block boundaries come from
`img_shapes`, never from a run-length heuristic over the boolean mask; adjacent
condition images remain distinct bidirectional blocks.

Prompt encoding must preserve:

* separate upstream t2i and edit templates;
* left padding;
* up to ten condition images in stable user order;
* Qwen3-VL vision inputs and `mm_token_type_ids` when provided;
* the last decoder-layer state before the final RMSNorm;
* the exact number and location of image-pad tokens.

The current Transformers compatibility hook around the final RMSNorm is
isolated in one helper and tested. It can be removed only after the installed
Transformers API exposes and passes the equivalent `tie_last_hidden_states`
behavior.

## 8. Generation design

Add `backend/core/pipeline_backends/qwen_image_21.py` and mix it into
`PipelineManager`. The backend owns:

* staged component placement;
* prompt/reference encoding and cache keys;
* native edit preparation;
* flow timesteps and dynamic shift;
* block-causal metadata and prefix K/V cache;
* CFG, cancellation, progress, previews, and final decode;
* adapter apply/unload in a `finally` block.

### 8.1 Defaults

Add an `IMAGE_GEN_ARCH_OVERLAYS["qwen_image_21"]` entry in
`backend/api/param_defaults.py`:

```text
steps=40
cfg_scale=1.0
width=1024
height=1024
sampler=<the Qwen flow Euler choice exposed by the API>
```

Do not introduce route-local defaults. Negative prompt is ignored with an
actionable warning unless `cfg_scale > 1`. When enabled, true CFG runs positive
and negative conditions and doubles the DiT work, matching upstream semantics.

### 8.2 txt2img

Generate normalized Gaussian target latents, append their slots after the
prompt prefix, and run the flow Euler loop. The first step uses prefill mode and
extracts per-layer prefix K/V; later steps forward only the target with cached
prefix K/V. Progress and preview callbacks observe actual scheduler steps.

### 8.3 Native image editing and references

The native edit path is the primary `img2img` implementation. Every condition
image is processed twice: as Qwen3-VL vision context and as a clean VAE latent
prefix. The target begins as noise; `denoising_strength` does not redefine this
native path.

For compatibility with the shared img2img control, define two explicit modes:

* `native_edit` (default for this architecture): condition image plus prompt,
  target starts from noise;
* `sdedit`: target image latent is noised at the selected flow time and denoised,
  without pretending this is the upstream native edit behavior.

The mode must be persisted in request metadata. A reference list has a hard
limit of ten and stable order. Batch requests with different reference lists
are split into separate pipeline calls because upstream treats one flat list as
shared by the batch.

### 8.4 Inpaint and outpaint

Inpaint/outpaint use the native edit model, not a fabricated ControlNet:

1. build an RGBA annotated condition image and, where selected, a separate mask
   condition using the prompt format exercised by the model;
2. generate the full target through native editing;
3. composite protected source pixels back exactly outside the feathered mask;
4. store both the raw model result and composited output in diagnostics.

Exact outside-mask preservation is a SushiUI postcondition, not a claim about
the model. Outpaint expands the canvas and mask first, then uses the same path.
Acceptance requires seam and alpha tests; until those pass, the capability is
advertised as experimental rather than silently routed through SDXL logic.

### 8.5 RGBA output

PNG output preserves all four decoded channels. JPEG/WebP-without-alpha output
must require an explicit background/compositing choice rather than silently
dropping alpha. Gallery thumbnails may composite for display but retain the
original RGBA artifact.

### 8.6 CFG and advanced guidance

First release supports true CFG only. Dynamic CFG and CFG rescale may open
after tests prove the scheduler-space math. NAG, NegPip, regional prompting,
and reference-style KV injection remain refused: block-causal attention and
prefix caching make the SDXL injection sites non-equivalent.

### 8.7 Attention backends

Vendor both exact attention processors. The default is exact multi-pass SDPA.
The compiled Flex path is an explicit option and has a probe for Torch version,
compile state, block-mask construction, and peak-memory failure.

Decode steps may route their full attention through the shared attention
dispatcher where its semantics match. Prefill remains owned by the Qwen
processor: the conduit must not erase the block-causal mask. Add one registry
descriptor only if a new kernel is genuinely required.

### 8.8 Offload, block swap, and caches

Generation block swap wraps `transformer_blocks` with the existing frozen
module offloader. Adapter branches are installed before partitioning and move
with their owning block. Prefix K/V tensors are not block weights and have
their own placement/lifetime.

Keep-hot identity includes model manifest digest, DiT/TE variant, adapters,
attention path, compile state, and `use_kv_cache`. Enabling/disabling the K/V
cache is not bitwise reproducible in reduced precision and must be persisted.

FBCache and Spectrum ship only after their registered speed/quality gates.
They must not reuse a condition prefix across different prompts, references,
RGBA contents, sizes, or adapter stacks.

## 9. Training design

Add:

```text
backend/core/training/arch/qwen_image_21.py
backend/core/training/ops/qwen_image_21_ops.py
backend/core/training/adapters/qwen_image_21_adapter.py
```

Register `qwen_image_21` in `ARCH_REGISTRY`, `_EXPECTED_ARCH_KEYS`, arch
resolution, cache namespace selection, and all base-trainer dispatch points in
the same ordering.

Handler declarations:

```text
pixel_align = 32
timestep_convention = "t0"
velocity_sign = "eps_minus_x0"
depth_blocks = transformer.transformer_blocks
```

Pixel alignment is 32 because the VAE is 16x and one VLM image slot represents
a 2x2 latent group. The DiT itself has patch size 1.

### 9.1 Supported methods

| Method | First release | Contract |
|---|---|---|
| LoRA | yes | DiT only; BF16 or validated frozen ConvRot base |
| Full parameter | yes | DiT only; BF16 base required |
| ReLoRA | no | open only after merge/reset/resume and loader round trip |
| ControlNet | no | no released compatible architecture |
| TE training | no | Qwen3-VL frozen; no save/load contract yet |
| VAE training through model trainer | no | VAE frozen; generic VAE-decoder work requires explicit 4-channel/5-D support first |

"Full parameter" means the complete DiT, including input/output projections,
shared modulation, all blocks, and output norm/projection. It does not include
the Qwen3-VL encoder or VAE.

### 9.2 Dataset modes

Support two explicit objectives:

* `t2i`: RGBA target plus caption;
* `edit`: target RGBA image, one-to-ten condition images, and edit instruction.

The dataset schema records references by role and order. It does not overload
SDXL's single `use_reference_images` boolean. Buckets are multiples of 32 and a
batch has one target geometry. Reference images may have independent geometry;
their VLM slots and VAE latent shapes are cached with each example.

Transparent training data stays RGBA. Converting all examples to RGB would
erase a released capability and produce a VAE distribution mismatch.

### 9.3 Cached conditioning and latents

Cache keys include source bytes, target/reference role and order, crop/resize,
RGBA compositing rule for the VLM copy, processor/template digest, TE digest,
VAE digest, normalization vectors, bucket geometry, and augmentation seed.

Cache these independently:

* target VAE latents (posterior sample policy declared);
* clean condition VAE latents (posterior mode);
* Qwen3-VL hidden states, attention mask, and image-pad mask.

The final-RMSNorm compatibility path is part of the encoder fingerprint. A
cache made before that behavior changes must not be reused.

### 9.4 Flow objective

For normalized clean latent `x0`, noise `eps`, and sampled `sigma`:

```text
xt = (1 - sigma) * x0 + sigma * eps
target = eps - x0
```

Use the shared timestep-density and SD3 loss-weighting helpers. The scheduler's
1000-step grid and dynamic shifting parameters come from the pinned scheduler
config. Training diagnostics recover `x0` through the shared
`eps_minus_x0`/`t0` identity; do not add a second formula.

The model predicts over condition plus target tokens. Loss and reconstructed
latent use only the target tail. Any prefix token in the loss is a correctness
failure.

### 9.5 LoRA topology and adapter files

Initial LoRA target scope is attention projections:

```text
transformer_blocks.*.attn.to_q
transformer_blocks.*.attn.to_k
transformer_blocks.*.attn.to_v
transformer_blocks.*.attn.to_out.0
```

Expose optional MLP (`proj`, `gate_layer`, `out`) and model projection scopes
only after target census and round-trip tests. The default matches the upstream
training reference but the implementation uses SushiUI's adapter subsystem,
not PEFT mutation.

Add rows to both adapter capability tables, starting with ordinary LoRA only.
LoHa/LoKr/DoRA stay closed until the relevant generation and training
round-trip gates pass. Use `is_lora_wrappable_linear` so ConvRot layers are not
silently skipped. `BLOCK_SWAP_ADAPTER_ORDER` is `BEFORE_SPLIT`.

Checkpoint metadata records architecture, scope, rank, alpha, base manifest
digest, component variant, step, epoch, and adapter family. The generation
loader must apply a trainer-produced file and reproduce the same delta before
LoRA is advertised.

### 9.6 Quantized-base training

ConvRot is frozen-base only:

* LoRA/additive branches may train when the complete expected DiT ConvRot census
  is present;
* full parameter and ReLoRA refuse a ConvRot base before model load;
* TE ConvRot is allowed because TE is frozen and encoding runs under `no_grad`;
* training disables any W8A8 execution mode not covered by the backward gate;
* validation sampling uses the same dequant/fused policy as the training
  forward so previews do not measure a different model.

The acceptance test includes a real backward through at least one ConvRot base
layer, finite input gradients, finite adapter gradients, unchanged packed
buffers, save/reload, and a generation round trip.

### 9.7 Memory features

Training block swap partitions the 32 ordered blocks after adapters are
installed. Gradient checkpointing is required. The shared block-swap conductor,
pinned-memory settings, ring size, and activation dispatcher are used; no
architecture-local clone is introduced.

Conditioning and latent caches should let steady-state training release TE and
VAE. Without caches, stage TE, VAE, and DiT sequentially. Full-DiT training is
expected to need block swap or optimizer-state offload on common GPUs; this is
an expectation, not a measured number. Actual VRAM/host-RAM figures belong in
`MODEL_FACTS.md` only after measurement.

### 9.8 Full-checkpoint save and resume

Full-parameter saves use the shared SushiUI v2 format and `transformer.` keys,
with component metadata pointing at the immutable TE, VAE, and processor
digests. Because the BF16 DiT exceeds the shared 10 GB threshold, a trained
full checkpoint is expected to be a shard index plus safetensors shards. The
selectable item is the index file. This does not change the base artifact
contract, where each imported component is one single safetensors file.

Resume restores model, optimizer, scheduler, scaler, RNG, timestep-sampler
state, cache namespace, adapter scope, and component digests. A changed TE,
VAE, template, or normalization refuses resume rather than reusing stale
conditioning.

## 10. API and frontend surface

API changes are OpenAPI-first. Add the architecture to model-load schemas,
generation/training enums, examples, capability payloads, and frontend unions.

New parameters are limited to semantics not already represented:

* `qwen_edit_mode`: `native_edit | sdedit`;
* `qwen_use_kv_cache`: boolean;
* `qwen_attention_path`: `sdpa | flex_compiled`;
* `qwen_lora_scope`: scope CSV for training.

Defaults live only in `backend/api/param_defaults.py`. If existing generic
fields can carry the meaning without ambiguity, use them instead of adding a
Qwen-prefixed duplicate.

The frontend:

* shows 40 steps and CFG 1.0 from backend schema defaults;
* accepts up to ten ordered reference images;
* exposes native edit versus SDEdit clearly;
* keeps alpha in upload, preview, gallery, and send-to flows;
* labels FlexAttention as compiled-only and falls back before submission when
  capability data refuses it;
* derives all advanced-control visibility from `/schema/arch-capabilities`;
* adds Qwen fields to training config serialization, presets, resume display,
  and validation-sample controls.

Do not add a separate Qwen-only page. Extend the existing image generation and
training panels.

## 11. Capability boundary at first release

| Feature | State | Reason/gate |
|---|---|---|
| txt2img | supported | native path |
| multi-reference native edit | supported | released conditioning path, max 10 |
| img2img SDEdit | supported | explicit compatibility mode |
| inpaint/outpaint | experimental then supported | native edit plus exact protected-pixel composite; seam/alpha gate |
| RGBA | supported | native four-channel VAE |
| true CFG | supported | negative prompt + scale > 1 |
| NAG/NegPip/regional/style-KV | refused | no exact block-causal implementation |
| ControlNet generation/training | refused | no compatible released architecture |
| LoRA | supported after round trip | DiT; BF16/ConvRot base |
| full parameter | supported after real smoke | complete DiT; BF16 only |
| ReLoRA | refused | reset/resume contract absent |
| TE training/override | refused | fixed Qwen3-VL conditioning contract |
| VAE swap | refused | 64-channel RGBA latent contract has no validated replacement |
| generation block swap | supported after equivalence gate | ordered 32-block DiT |
| training block swap | supported after gradient gate | ordered 32-block DiT |
| compiled FlexAttention | opt-in | compile/runtime/memory probe |
| FBCache/Spectrum/TREAD/BlockSkip | refused initially | each requires measurement |

## 12. Implementation phases and gates

### P0 — source pin and artifact census

* pin upstream revisions and licenses;
* implement streaming header census and config fingerprints;
* build/import the five component weight files and processor directory;
* write manifest and checksums;
* record actual tensor/ConvRot census.

Gate: all files pass header-only validation; BF16 and ConvRot components report
the same logical layer topology; no model is loaded.

### P1 — vendored model and loader

* vendor Transformer, VAE, and attention processors;
* implement meta construction, streaming load, detection, and preflight;
* add model type and pipeline component registration.

Gate: tiny-model forwards, real BF16 load, real ConvRot load, no missing or
unexpected tensors, no CUDA initialization during header inspection.

### P2 — txt2img

* prompt encoding, VAE decode, flow loop, CFG, prefix K/V cache;
* progress, cancellation, previews, offload, keep-hot identity;
* 1024 and one documented 2048 aspect-ratio smoke.

Gate: seeded output matches the pinned upstream implementation within the
registered latent/image tolerances for cache on and off separately.

### P3 — native edit, references, and RGBA

* one and multiple reference images;
* native edit and SDEdit mode separation;
* RGBA upload/output/gallery preservation.

Gate: 1, 2, and 10 references; alpha round trip; reference order changes the
cache key; oversized reference list refused before encoding.

### P4 — inpaint and outpaint

* annotation/mask conditioning;
* exact protected-region composite;
* queue/gallery/send-to wiring.

Gate: protected pixels exact outside feather band, alpha exact where protected,
and no seam regression beyond the registered metric.

### P5 — LoRA training

* arch handler, ops, adapter targets, caches, block swap;
* BF16 and ConvRot-base paths;
* t2i and edit objectives;
* resume and generation round trip.

Gate: finite multi-step real runs for t2i and edit, non-zero target gradients,
unchanged frozen components, exact adapter resume, and visible generation
effect after reload.

### P6 — full-DiT training

* full adapter, sharded save, resume, validation generation;
* optimizer offload/block-swap combinations.

Gate: real multi-step smoke, all intended DiT groups update, TE/VAE remain
unchanged, saved index reloads through the production generation loader.

### P7 — acceleration and extended adapters

Open one feature at a time only after its correctness and measurement gate:
FlexAttention, LoHa/LoKr/DoRA, FBCache, Spectrum, TREAD, BlockSkip, and any
fused ConvRot training path.

## 13. Verification matrix

CPU/schema tests:

* detection priority and near-miss checkpoints;
* manifest path traversal, digest, and mixed-revision refusals;
* ConvRot marker validation, incomplete triples, and exact census;
* config construction and state-dict key remap;
* VAE shapes, normalization, RGBA behavior, pack/unpack identity;
* block metadata for adjacent condition images;
* prompt template, left padding, pre-final-norm hidden state, image-pad mask;
* t0 velocity/x0 recovery and target-tail-only loss;
* cache namespace invalidation;
* capability/default/OpenAPI/frontend enum parity;
* LoRA save/load/resume/rollback and quantized-base target discovery;
* full-save shard-index round trip.

CUDA tests:

* BF16 and ConvRot layer forward comparison;
* exact SDPA versus compiled Flex prefill/decode tolerance;
* K/V cache on/off each compared to its own reference;
* t2i/edit/RGBA real checkpoint smokes;
* component offload and keep-hot reuse without stale state;
* generation/training block-swap equivalence;
* BF16 LoRA, ConvRot-base LoRA, and BF16 full-DiT backward;
* validation generation after save/reload.

Every backend edit receives `py_compile` and a real import using the repository
virtualenv with the documented CUDA initialization stubs. Frontend build and
type-check remain owner-run per repository policy.

## 14. Acceptance measurements

Register the measurement protocol before running it:

* hardware, Torch/CUDA/comfy-kitchen versions, source revisions, dtype;
* resolution/aspect ratio, prompt/reference token counts, steps, seed;
* BF16 versus DiT-only ConvRot versus DiT+TE ConvRot;
* SDPA versus compiled Flex;
* K/V cache on/off;
* wall time, first-step time, later-step median, peak VRAM, host RSS/commit;
* image/latent error against the pinned upstream path;
* training step time and peak memory for BF16 LoRA, ConvRot LoRA, and full DiT.

No speed, memory-saving, or quality statement becomes user-visible until the
measurement is recorded in `docs/guides/MODEL_FACTS.md` with its conditions.

## 15. Known hazards

* The upstream model and diffusers support landed days before this design;
  checkpoint/config revisions are still moving. Always pin revisions.
* Upstream training PR #14808 is unmerged and has already changed around device
  placement, masks, validation, and caching. Port the mathematics, not the
  script wholesale.
* The model license is Qwen Research, while vendored diffusers code is
  Apache-2.0. Weight redistribution and code redistribution are separate legal
  questions.
* The final-RMSNorm hidden-state behavior differs across Transformers versions.
  A seemingly harmless library upgrade can materially alter rendered text.
* The VAE is RGBA and 64-channel with 16x compression. Reusing older
  Qwen-Image 16-channel helpers will produce plausible but wrong shapes.
* A VLM image slot represents four latent tokens. Failing to distinguish these
  grids corrupts masks and block boundaries.
* Prefix-cache slices must own storage. A contiguous view at batch size one can
  pin the whole prefill tensor per layer and add gigabytes of residency.
* FlexAttention is safe only when compiled. An automatic uncompiled fallback is
  an OOM path, not a performance preference.
* Cache-on and cache-off runs can diverge visibly in BF16 from early rounding
  differences even when both agree with fp32. Do not require bit identity
  between the two modes.
* Native edit and SDEdit are different algorithms. A single unlabeled strength
  slider must not blur the distinction.
* Training loss belongs only to target tokens. Prefix loss can appear to train
  while optimizing the wrong task.
* A full-DiT save exceeds the shared single-file writer's shard threshold. The
  index is the model entry point; selecting one shard must be refused.

## 16. Documentation updates during implementation

When implementation phases land, update in the same commits:

* `AGENTS.md` architecture and training-capable counts;
* `docs/guides/ADD_A_MODEL_ARCHITECTURE.md` counts;
* `docs/guides/ARCHITECTURE_MAP.md`;
* `docs/guides/MODEL_FACTS.md` with verified facts and measurements;
* `docs/reference/architectures/qwen_image_21.md` with implemented symbols;
* `backend/core/training/MODEL_ARCHITECTURES.md`;
* `docs/legal/THIRD_PARTY_PROVENANCE.md`;
* `openapi.yaml` and API examples.

This design remains the decision record. The architecture reference becomes
the concise current-code map once implementation exists.
