# Qwen-Image 2.1 integration design

Status: **implemented first release; measured gates are listed in section 12**

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
* exact block-causal attention, component CPU offload, training block swap,
  latent/conditioning caching, and the shared diagnostics where the
  architecture can implement their semantics exactly;
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
  injection are measured independently. A generic UI control is not evidence
  that the architecture consumes it.

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

### 3.1 Implementation provenance

The integration keeps model artifacts and SushiUI source code in separate
provenance domains:

* **Weights and model configuration.** External model artifacts retain their
  own license and are user-supplied data under `<MODEL_ROOT>`. They do not
  contribute source code to SushiUI or change SushiUI's source-code license.
* **Inference and training code.** Implementation work uses the Apache-2.0
  diffusers code at the pinned commits identified by this document, existing
  SushiUI code, and original local work.

The implementation input set is closed and auditable:

1. the Apache-2.0 diffusers PRs/commits pinned here;
2. the published model weight/config files and processor assets as data;
3. black-box behavior measured by executing the released weights through the
   pinned diffusers implementation; and
4. existing SushiUI code and independently written tests/specifications.

The vendored package records its exact Diffusers commit in `vendor/UPSTREAM.md`.

### 3.2 Vendoring and dependency choice

The installed Transformers package supplies Qwen3-VL, while the installed
Diffusers release predates these model classes. The integration therefore
vendors the minimum Apache-2.0 implementation from the pinned Diffusers commit:

```text
backend/core/models/qwen_image_21/
  __init__.py
  loader.py
  vendor/transformer.py
  vendor/autoencoder.py
  vendor/pipeline.py
  vendor/UPSTREAM.md
  lora.py
  artifact.py
  loader.py
```

`vendor/UPSTREAM.md` records the Apache-2.0 Diffusers commit and the local
integration change class. The package is also listed in
`docs/legal/THIRD_PARTY_PROVENANCE.md`.

Do not vendor Qwen3-VL from Transformers. Load it through the installed
Transformers public classes. Do not upgrade all of diffusers to an unreleased
or newly released version solely for this architecture: that changes every
existing pipeline's runtime underneath the integration.

The vendored pipeline owns the architecture math. The SushiUI backend owns its
lifecycle: progress, cancellation, component offload, adapter sessions, cache
selection, request defaults, and output compositing. This keeps the numerical
path pinned while still integrating with the shared runtime.

## 4. Artifact layout and why TE/DiT stay separate

The local layout is:

```text
<MODEL_ROOT>/qwen21/
  original/
    manifest.json
    qwen_image_2.1_original.safetensors
    qwen3vl_8b_original.safetensors
    qwen_image_2.1_vae_bf16.safetensors
    text_encoder_config.json
    processor/
    scheduler/
  int8_convrot/
    manifest.json
    qwen_image_2.1_int8_convrot.safetensors
    qwen3vl_8b_int8_convrot.safetensors
    qwen_image_2.1_vae_bf16.safetensors
    text_encoder_config.json
    processor/
    scheduler/
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    added_tokens.json
    merges.txt
    vocab.json
    preprocessor_config.json
    video_preprocessor_config.json
    chat_template.jinja
  prompt_enhancer/
    t2i/
      manifest.json
      qwen_image_2.1_pe_t2i_int8_convrot.safetensors
      config.json
      tokenizer.json
      system_prompt.txt
    i2i/
      manifest.json
      qwen_image_2.1_pe_i2i_int8_convrot.safetensors
      config.json
      tokenizer.json
      processor_config.json
      system_prompt.txt
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

### 4.1 Prompt-enhancer artifacts

The T2I and I2I prompt enhancers are separate Qwen3.5-VL 9B checkpoints and are
not part of the generation TE/DiT lifecycle. `subapps/qwen_image_21_pe_convert.py`
builds each from its source directory as one INT8 ConvRot safetensors plus the
small processor/config files. Eligible linears use the same validated runtime
as the main Qwen artifact; unsupported-width linears stay BF16. Source shards
are temporary conversion inputs and are not retained in the model directory.

`POST /prompt-assist/qwen-image-21/transform` selects the matching artifact by
mode. The official engine runs locally in-process; LM Studio and Ollama are
loopback-only alternatives using a selected local multimodal model for I2I.
Results are cached before model load. The API returns the rewritten prompt plus
`wh_ratio`/`ratio_follow`; these are advisory and do not mutate width or height.

The UI supports preview-and-apply and an opt-in automatic rewrite on Generate.
Txt2img switches to I2I mode when reference images exist; img2img, inpaint, and
outpaint always use I2I mode.

### 4.2 `manifest.json`

Each variant directory is independently selectable. Its manifest contains no
weights and points to one safetensors file per component:

```json
{
  "model_type": "qwen_image_21",
  "format_version": "1",
  "variant": "original",
  "components": {
    "transformer": "qwen_image_2.1_original.safetensors",
    "text_encoder": "qwen3vl_8b_original.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
    "processor": "processor",
    "scheduler": "scheduler",
    "text_encoder_config": "text_encoder_config.json"
  },
  "transformer_config": {},
  "vae_config": {}
}
```

Paths are relative to the manifest directory. DiT and TE variants are paired
by construction; there is no mixed-variant UI override.

### 4.3 Artifact creation

`subapps/qwen_image_21_convert.py` has two modes:

* `original`: stream source Diffusers shards into one BF16 DiT file,
  one BF16 TE file, and one BF16 VAE file; copy processor assets; and write the
  manifest.
* `int8_convrot`: read source components layer by layer and
  emit the corresponding INT8 ConvRot file.

The converter uses `safe_open`, writes one component at a time, and never holds
the TE and DiT together. The default 100 GiB safety ceiling keeps each released
component in a single safetensors file. `--max-shard-gb` exists only as an
operator escape hatch for filesystems that cannot accept such files; a split
result remains loadable through its index, but is not the standard artifact.

An already-published single-file repack may be imported only after tensor-key,
shape, dtype, config, and source-revision validation. File names and repository
labels alone are not a trust boundary.

### 4.4 Safetensors metadata

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

### 4.5 Quantization scope

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

Live preview uses the flow model's current clean estimate
`pred_x0 = x_t - sigma * velocity`, not the noisy post-step latent. The
unpatched `[B,N,64]` tensor is reshaped to `[B,64,H/16,W/16]`, projected by a
fixed 64-to-RGB matrix, and enlarged for the existing JPEG WebSocket payload.
This path does not run or move the 1.35 GB VAE during denoising. The preview is
RGB-only; native alpha remains available in the final RGBA decode.

### 8.3 Native image editing and references

The native edit path is the primary `img2img` implementation. Every condition
image is processed twice: as Qwen3-VL vision context and as a clean VAE latent
prefix. The target begins as noise; `denoising_strength` does not redefine this
native path.

The native edit route uses a condition image plus prompt, with the target
starting from noise. The shared `denoising_strength` field does not change this
algorithm. SDEdit is not advertised for this architecture because no separate,
tested flow path exists. A reference list has a hard limit of ten including the
primary image, and preserves user order.

Reference Guide is separate from native edit conditioning. Its image is encoded
into a target-sized normalized 64-channel latent, re-noised with one fixed noise
sample at each flow sigma, and blended into the post-step target over its
requested 0-1000 step range. Style Transfer is likewise separate: a target-sized
style latent is re-noised at each active step, a reference forward captures
post-RMSNorm/post-RoPE target-image Q/K/V, and the conditional forward injects
those keys and values into the target rows. Style transfer disables condition-
prefix KV caching for that generation because injected keys extend the full
joint attention layout; ordinary and Reference Guide runs retain the cache.
Multi-reference `stack` and `common_concept` use the shared style combiner.

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

The native VAE and conditioning path remain four-channel RGBA. The current
gallery response follows SushiUI's existing image-output conversion contract;
lossless alpha preservation beyond the model path is not claimed as a separate
gallery feature in this release.

### 8.6 CFG and advanced guidance

First release supports true CFG only. Dynamic CFG and CFG rescale may open
after tests prove the scheduler-space math. NAG, NegPip, and regional prompting
remain refused. Reference-style injection uses Qwen's own segmented block-
causal attention path described in section 8.3 rather than an SDXL injection
site.

### 8.7 Attention backends

The vendored transformer selects its exact block-causal SDPA processor. Shared
`attention_type` and `attention_impl` controls are explicitly refused. A
compiled Flex path is not exposed in the first release.

### 8.8 Offload, block swap, and caches

Generation uses Diffusers component CPU offload. Generation block swap and
keep-hot residency are not implemented and are declared unsupported. Adapter
branches are installed before generation and unloaded in `finally`.

`qwen_image_21_kv_cache` controls prefix K/V reuse and defaults to true.
Enabling/disabling the K/V cache is not bitwise reproducible in reduced
precision, so the request records the chosen flag.

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

### 9.2 Dataset mode

The first release trains text-to-image targets: RGBA target plus caption.
Existing RGB datasets gain an opaque alpha channel before VAE encoding.
Reference-conditioned edit training is explicitly unavailable; generation
support for references does not imply a cache or loss contract for paired edit
datasets. Buckets are multiples of 32 and a batch has one target geometry.

### 9.3 Cached conditioning and latents

Cache keys include source bytes, crop/resize, VAE identity and normalization,
processor/template identity, TE identity, bucket geometry, and augmentation
seed through the shared cache namespace.

Cache these independently:

* target VAE latents (posterior sample policy declared);
* Qwen3-VL hidden states and attention mask.

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

The trainer now reports `recon_loss = MSE(xt - sigma * predicted_velocity, x0)`
on both full-frame and complete-coverage partitioned steps. Its existing
`reconstruction_loss_weight` mixes this with the prediction objective; weight
zero still reports the reconstruction metric without altering gradients.
`qwen_debug_latent_view=latent` preserves tensor-channel previews. The opt-in
`pixel` view additionally VAE-decodes target, noisy, and predicted-clean
latents after backward, so debug previews do not retain a VAE next to step
activations. The raw `.pt` data remains available in both modes.

### 9.5 LoRA topology and adapter files

The default LoRA target scope is attention projections:

```text
transformer_blocks.*.attn.to_q
transformer_blocks.*.attn.to_k
transformer_blocks.*.attn.to_v
transformer_blocks.*.attn.to_out.0
```

`train_adapter: true` additionally targets `txt_in.in_layer` and
`txt_in.out_layer`, with `adapter_lr` selecting a separate optimizer rate.
The text encoder itself stays frozen. See
`docs/guides/CONDITIONING_ADAPTER_TRAINING.md` for the common-key contract.
Expose optional MLP (`proj`, `gate_layer`, `out`) and other model projection scopes
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
* the ConvRot base forward uses its INT8 kernel, while `grad_input` uses one
  cached BF16 dequantized weight per frozen Linear; LoRA and attention backward
  remain floating point;
* the 13.252 GiB cache is non-persistent, cannot be combined with block swap,
  and is never serialized into the LoRA;
* validation sampling and generation use the same ConvRot base function;
* adapter metadata binds this path to an `int8_convrot` generation base, and
  resume refuses a legacy dequant-forward checkpoint.

The acceptance test includes a real backward through at least one ConvRot base
layer, finite input gradients, finite adapter gradients, unchanged packed
buffers, save/reload, and a generation round trip.

### 9.7 Memory features

Training block swap partitions the 32 ordered blocks after adapters are
installed. Gradient checkpointing is required. The shared block-swap conductor,
pinned-memory settings, ring size, and activation dispatcher are used; no
architecture-local clone is introduced.

Qwen's cached latents are `[B, sequence, channels]`, so activation dispatch
keys the model by image-token sequence length in an architecture-specific
family rather than sharing ACE-Step's 3-D audio predictor. A measured
`3.5e-3 GiB/token` cold-start floor prevents the generic image seed from
underestimating the first long bucket. Reactive recovery attempts saved-tensor
offload before declaring a batch-1 bucket unfit; the resident BF16 ConvRot
backward cache is a leaf buffer and is not copied to CPU by that hook.

Activation offload is a capacity fallback, not the fast path. At 1536px,
rank 128 and checkpointing 24/32 blocks, an offloaded 8,892-token step used
26.57 GiB peak allocated and 8.24 s for forward+backward. The matching run's
non-offloaded steps used about 41.6 GiB and 5.96 s. Transfers are synchronous,
so offload stays opt-in.

Complete-coverage partitioned target training is specified separately in
[`QWEN_IMAGE_21_PARTITIONED_TRAINING_DESIGN.md`](QWEN_IMAGE_21_PARTITIONED_TRAINING_DESIGN.md).
Its fixed 2/4-region prototype is implemented; partition core losses are
area-weighted, including the reconstruction diagnostic and optional objective.

Conditioning and latent caches should let steady-state training release TE and
VAE. Without caches, stage TE, VAE, and DiT sequentially. Full-DiT training is
expected to need block swap or optimizer-state offload on common GPUs; this is
an expectation, not a measured number. Actual VRAM/host-RAM figures belong in
`MODEL_FACTS.md` only after measurement.

### 9.8 Full-checkpoint save and resume

Full-parameter saves contain the complete DiT state and embedded construction
config. Metadata records `companion_path`, which resolves the frozen TE, VAE,
processor, and scheduler from the run's base artifact. The shared 4 GiB writer
may produce a shard index for the trained DiT; the loader accepts either the
single file or the index and replaces only the companion artifact's DiT.

Resume restores model, optimizer, scheduler, scaler, RNG, timestep-sampler
state, cache namespace, adapter scope, and component digests. A changed TE,
VAE, template, or normalization refuses resume rather than reusing stale
conditioning.

## 10. API and frontend surface

API changes are OpenAPI-first. Add the architecture to model-load schemas,
generation/training enums, examples, capability payloads, and frontend unions.

New parameters are limited to semantics not already represented:

* `qwen_image_21_kv_cache`: boolean.

Defaults live only in `backend/api/param_defaults.py`. If existing generic
fields can carry the meaning without ambiguity, use them instead of adding a
Qwen-prefixed duplicate.

The frontend:

* shows 40 steps and CFG 1.0 from backend schema defaults;
* accepts up to ten ordered reference images;
* uses the existing ordered reference-image control for native edit;
* keeps alpha in upload, preview, gallery, and send-to flows;
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
| img2img SDEdit | refused | native edit is the only implemented image-conditioning algorithm |
| inpaint/outpaint | supported | native edit plus protected-pixel composite |
| RGBA | supported | native four-channel VAE |
| true CFG | supported | negative prompt + scale > 1 |
| Reference Guide | supported | normalized 64-channel flow-latent blend |
| style-KV | supported | exact segmented-SDPA block-causal path; condition KV cache disabled while active |
| NAG/NegPip/regional | refused | no exact block-causal implementation |
| ControlNet generation/training | refused | no compatible released architecture |
| LoRA | supported | DiT; BF16/ConvRot base; real ConvRot backward passed |
| full parameter | supported | complete DiT; BF16 only; companion-based resume |
| ReLoRA | refused | reset/resume contract absent |
| TE training/override | refused | fixed Qwen3-VL conditioning contract |
| VAE swap | refused | 64-channel RGBA latent contract has no validated replacement |
| generation block swap | refused | component CPU offload is used instead |
| training block swap | supported | ordered 32-block DiT; gradient checkpointing required |
| compiled FlexAttention | refused | exact vendored SDPA path only |
| FBCache/Spectrum/TREAD/BlockSkip | refused initially | each requires measurement |

## 12. Implementation phases and gates

Current first-release status (2026-09-21):

* P0/P1 complete: both variant manifests load; Original DiT/TE/VAE and ConvRot
  DiT/TE/VAE were loaded from the local artifacts with exact component checks.
* P2 complete at smoke level: a 256x256, two-step ConvRot txt2img run completed
  through processor, TE, DiT, scheduler, and VAE in 25.5 seconds.
* P3/P4 implemented: native edit consumes the primary image plus ordered
  `ref_images` (ten total maximum); inpaint/outpaint use protected-pixel
  compositing. Multi-reference quality and seam metrics remain unmeasured.
* Generation adapters and reference controls are wired independently: LoRA
  strength and 0-1000 step ranges drive the shared adapter session; Reference
  Guide blends the target latent; Style Transfer captures and injects
  post-RoPE Q/K/V. A real ControlNet selection is refused before denoising.
* P5 compute/save/resume gate passed on the real ConvRot artifact. A rank-4
  256x256 run completed steps 1-2 with losses 0.330088/0.293226, wrote a
  16,826,096-byte adapter plus optimizer/state at each step, resumed from step
  2, and completed step 3 at loss 0.316250. The earlier single-backward census
  also found 128 attention targets and 256 finite adapter-gradient tensors.
  Visible generation effect after reload remains a generation acceptance task.
* P6 compute/save/resume/load gate passed on the real 7.115B-parameter Original
  DiT. At 256x256, batch 1, BF16, 24 swapped blocks and host-resident
  AdamW8bit ring-buffer state, step 1 completed at loss 0.330214; its four-shard
  14,230,249,472-byte index loaded through the production component loader.
  Resume restored step 1 and completed step 2 at loss 0.373368. That resumed
  save exposed two generic checkpoint bugs: Windows retained the path-owned
  PyTorch writer after ENOSPC, and Qwen chained `companion_path` through a
  pruneable intermediate checkpoint. The writer is now explicitly scoped and
  the loader carries the terminal component source; both have regression tests.
* P7 is deferred. Shared capability data refuses every unimplemented
  acceleration instead of accepting an inert control.

The phase descriptions below are the gate definitions, not claims that every
measurement has already passed.

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
* t2i objective;
* resume and generation round trip.

Gate: finite multi-step real t2i runs, non-zero target gradients, unchanged
frozen components, exact adapter resume, and visible generation effect after
reload. Reference-conditioned edit training remains outside the first-release
contract in section 9.2.

### P6 — full-DiT training

* full adapter, sharded save, resume, validation generation;
* optimizer offload/block-swap combinations.

Gate: real multi-step smoke, all intended DiT groups update, TE/VAE remain
unchanged, saved index reloads through the production generation loader.

### P7 — acceleration and extended adapters

Open one feature at a time only after its correctness and measurement gate:
FlexAttention, LoHa/LoKr/DoRA, FBCache, Spectrum, TREAD, and BlockSkip. The
Qwen-only frozen-base ConvRot forward/cached-BF16-backward path has passed its
real-workload speed, memory, resume, and generation-base compatibility gates;
it does not generalize to other architectures or trainable INT8 weights.

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
* full-save single/index metadata and companion-path round trip.

CUDA tests:

* BF16 and ConvRot layer forward comparison;
* K/V cache on/off each compared to its own reference;
* t2i/edit/RGBA real checkpoint smokes;
* component offload without stale state;
* training block-swap equivalence;
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
* Training math must preserve device placement, masks, validation, and target
  tail slicing; these contracts are tested locally rather than inferred from
  a generic DiT trainer.
* Weight use/distribution remains subject to the weight license, while SushiUI
  implementation code comes only from Apache-2.0 diffusers or original local
  work. Never blur those provenance domains or treat a weight license as a
  source-code dependency.
* The final-RMSNorm hidden-state behavior differs across Transformers versions.
  A seemingly harmless library upgrade can materially alter rendered text.
* The VAE is RGBA and 64-channel with 16x compression. Reusing older
  Qwen-Image 16-channel helpers will produce plausible but wrong shapes.
* A VLM image slot represents four latent tokens. Failing to distinguish these
  grids corrupts masks and block boundaries.
* Prefix-cache slices must own storage. A contiguous view at batch size one can
  pin the whole prefill tensor per layer and add gigabytes of residency.
* Cache-on and cache-off runs can diverge visibly in BF16 from early rounding
  differences even when both agree with fp32. Do not require bit identity
  between the two modes.
* Native edit and SDEdit are different algorithms. This release implements
  native edit only; `denoising_strength` must not be described as changing it.
* Training loss belongs only to target tokens. Prefix loss can appear to train
  while optimizing the wrong task.
* A full-DiT training save may exceed the shared streaming writer's shard
  threshold. When it does, the index is the model entry point; an individual
  shard is never a complete checkpoint.

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
