# SenseNova SDXL Chimera design

Status: **proposed; not implemented**. This document defines the intended
architecture, artifact contract, training and generation wiring, frontend
surface, bootstrap-model build, and acceptance gates. Statements about speed,
memory, convergence, or output quality are targets until the measurements in
the final sections have been run.

The working architecture id is `sensenova_sdxl_chimera`; the user-facing name
is **SenseNova SDXL Chimera**. This is a new architecture. It does not change
the meaning or checkpoint format of `sensenova` or `sdxl`.

## 1. Goal and non-goals

SenseNova U1.5 performs a one-time understanding-prefix pass and then runs the
42-layer generation half of its Qwen backbone at every denoise step. The prefix
KV reuse is strong, but the generation iteration remains expensive. Migrating
SenseNova to an SDXL VAE and increasing `sensenova_gen_patch` reduces token
count, but `gen_patch=8` makes one generation token cover 64 output pixels on
an 8x VAE and removes fine spatial capacity before the denoiser begins.

This architecture keeps SenseNova's understanding branch and reference-image
input semantics, but replaces the complete generation branch with a
from-scratch, canonical-SDXL-shaped U-Net operating in the selected SDXL
model's four-channel latent space.

Goals:

1. Reference an existing SenseNova model for the understanding branch,
   tokenizer, chat template, and understanding vision tower. Do not duplicate
   those weights in the Chimera checkpoint by default.
2. Extract and bundle the VAE from a selected SDXL model through the existing
   VAE-source machinery. The generated checkpoint must not depend on the SDXL
   donor remaining present merely to decode images.
3. Make the U-Net tensor shapes and trainable parameter count exactly equal to
   the selected SDXL donor U-Net. The default initialization is the normal
   PyTorch/diffusers initialization from that config.
4. Offer an experimental initializer that copies all compatible U-Net weights
   from the selected SDXL donor instead of using fresh weights.
5. Replace SenseNova's layer-paired generation KV interface with a fixed-length
   SDXL-compatible conditioning bridge and immutable per-U-Net-site inference
   KV caches.
6. Train the main U-Net at the same latent feature resolutions and with the
   same trainable U-Net parameter/activation structure as SDXL. No
   `sensenova_gen_patch` exists on this architecture.
7. Support model construction, load, txt2img generation, training, sampling,
   checkpoint save/resume, and frontend operation. Add img2img, inpaint,
   outpaint, and reference-image conditioning behind explicit later gates.
8. Build a real full-size bootstrap artifact under
   `<MODEL_ROOT>/snu1.5_sdxl_chimera` and use it for API inference and training
   smoke tests. `<MODEL_ROOT>` is the machine's configured model root; tracked
   documentation must not contain a machine-specific drive letter.

Non-goals for the first shipped boundary:

- Training the SenseNova understanding weights.
- Reusing or converting the SenseNova `_mot_gen` weights, generation ViT,
  timestep embedder, pixel/latent head, or latent refiner.
- Claiming that a copied SDXL U-Net is immediately useful before its
  conditioning input has been aligned to the SDXL CLIP space.
- ReLoRA or ControlNet training.
- Runtime int8 conversion of the U-Net.
- Treating raw SenseNova KV tensors as a stable public interface to arbitrary
  denoisers.

## 2. Architectural decisions

| ID | Decision |
|---|---|
| C1 | New model id `sensenova_sdxl_chimera`; never infer this mode from a plain SenseNova or SDXL checkpoint. Detection is metadata/manifest-first. |
| C2 | The understanding component is an external, hash-pinned reference to a supported SenseNova checkpoint. Missing or changed source weights fail during header-only preflight, before the current model is unloaded. |
| C3 | The VAE is extracted from the selected SDXL donor and bundled into the Chimera artifact with `component.vae.*` metadata. |
| C4 | The U-Net config is copied from the selected SDXL donor. Construction asserts exact config compatibility and exact parameter-count equality with the donor U-Net. |
| C5 | `unet_init="scratch"` is the default. `unet_init="sdxl_transplant"` copies the donor U-Net tensors and is marked experimental. Both modes use the same U-Net config and parameter count. |
| C6 | The bridge emits SDXL-compatible `encoder_hidden_states [B,M,2048]`, pooled conditioning `[B,1280]`, and six size/crop ids. Initial `M=77`; the value is checkpoint structure, not a per-generation knob. |
| C7 | A transferred SDXL U-Net is not considered initialized for diffusion training until the bridge has passed the CLIP-space alignment gate. This is a hard gate for `sdxl_transplant`, advisory for `scratch`. |
| C8 | U-Net training predicts flow velocity directly. It does not reproduce SenseNova's x0-head followed by division by `(1-t)`. |
| C9 | Understanding and VAE are frozen. The default production U-Net stage uses a frozen bridge and cached conditioning, so its GPU-resident training graph matches SDXL as closely as possible. |
| C10 | Training-time U-Net cross-attention remains ordinary differentiable attention. Per-site K/V caching is inference-only because U-Net `to_k`/`to_v` weights change during training. |
| C11 | The Chimera artifact owns U-Net, bridge, config, and bundled VAE tensors. It records but does not bundle understanding weights by default. |
| C12 | A bootstrap builder is an explicit API/CLI operation. Loading or starting a training run never silently creates a multi-gigabyte model. |
| C13 | U-Net attention uses SenseNova-compatible three-axis RoPE rather than preserving the donor SDXL attention's position treatment. RoPE adds no trainable parameters, so U-Net parameter-count equality is preserved. |
| C14 | Original size, crop origin, and target size are wired both as SDXL-compatible six-value micro-conditioning and into the physical-coordinate construction used by spatial RoPE. |

## 3. Component and artifact contract

### 3.1 Components

| Role | Source | Trainable in first release | Persisted in Chimera checkpoint |
|---|---|---:|---:|
| Tokenizer/chat template | referenced SenseNova | no | reference only |
| Understanding token embeddings and decoder | referenced SenseNova, unsuffixed branch | no | reference only |
| Understanding vision tower | referenced SenseNova | no | reference only |
| Conditioning bridge/resampler | new | stage-dependent | yes |
| Pooled-conditioning head | new | stage-dependent | yes |
| U-Net | selected SDXL config; scratch or donor weights | yes in U-Net/joint stages | yes |
| VAE | extracted from selected SDXL model | no | yes |
| SDXL CLIP encoders | selected SDXL donor, bridge-alignment teacher only | no | no |
| Scheduler/time grid | Chimera config | no | config only |

The understanding-only wrapper must load only the unsuffixed SenseNova branch.
It must not instantiate or materialize `_mot_gen`, `vision_model_mot_gen`,
`timestep_embedder`, `fm_head`, or `fm_refiner`. The source key names remain the
existing SenseNova unsuffixed key names so the reference loader can read an
unchanged SenseNova checkpoint.

### 3.2 Bootstrap directory

The full-size bootstrap build writes:

```text
<MODEL_ROOT>/snu1.5_sdxl_chimera/
  chimera.json
  config.json
  model.safetensors.index.json
  model-00001-of-000NN.safetensors
  ...
```

The sharded tensor set contains:

```text
condition_bridge.*
unet.*
vae.*
```

`chimera.json` contains only relocatable declarations and verified
provenance:

```json
{
  "model_type": "sensenova_sdxl_chimera",
  "format_version": 1,
  "understanding": {
    "locator": "model:<SENSENOVA_SOURCE>",
    "content_hash": "<HASH>",
    "model_type": "sensenova",
    "branch": "understanding"
  },
  "sdxl_donor": {
    "provenance": "model:<SDXL_SOURCE>",
    "content_hash": "<HASH>"
  },
  "unet": {
    "initialization": "scratch",
    "config_hash": "<HASH>",
    "parameter_count": 0
  },
  "conditioning": {
    "context_tokens": 77,
    "context_dim": 2048,
    "pooled_dim": 1280,
    "bridge_state": "unaligned",
    "position_encoding": {
      "mode": "sensenova_3d_rope",
      "layout": "rotate_half",
      "axes_ratio": [2, 1, 1],
      "spatial_unit_pixels": 32,
      "version": 1
    }
  },
  "prediction": {
    "type": "flow_velocity",
    "time_direction": "zero_noise_to_one_clean"
  }
}
```

The real `parameter_count` is written by the builder. Placeholder values are
not accepted by the loader.

Absolute external locators may exist in a machine-local artifact, but saves
also record a content hash and a user-facing basename. A locator is never
silently rebound to a different file with the same name. A portable export may
later add `bundle_understanding=true`; that is outside the first boundary.

### 3.3 Load and preflight order

Before unloading the active model:

1. Read only `chimera.json`, the shard index, and safetensors headers.
2. Require `model_type == "sensenova_sdxl_chimera"` and supported
   `format_version`.
3. Resolve the understanding locator, detect it as `sensenova`, and verify its
   content/config hash and required unsuffixed keys.
4. Verify that no `_mot_gen` tensor is required by the Chimera tensor map.
5. Validate U-Net config, tensor census, shapes, dtype, and saved parameter
   count.
6. Parse bundled `component.vae.*`; verify four latent channels, 8x spatial
   compression, and the recorded identity hash.
7. Validate bridge shapes against the referenced SenseNova head/KV dimensions
   and the fixed SDXL conditioning dimensions.
8. If initialization is `sdxl_transplant`, require donor provenance and report
   whether `bridge_state` is `unaligned`, `aligned`, or `joint_finetuned`.

Any failure leaves the currently loaded model untouched.

## 4. U-Net equality with SDXL

The builder reads the donor's real U-Net config and constructs
`UNet2DConditionModel` from that config. It does not maintain a handwritten
second copy of the SDXL block layout.

The following must match the donor exactly:

- down/up block types and count;
- `block_out_channels`;
- `layers_per_block` and `transformer_layers_per_block`;
- attention head dimensions/counts;
- cross-attention dimension, required to be 2048 for the first format;
- input/output channels, required to be four;
- addition embedding layout, required to accept pooled 1280 plus SDXL time ids;
- every U-Net parameter name and shape;
- total parameter count.

The builder computes:

```text
P_donor = sum(numel(p) for donor_unet.parameters())
P_new   = sum(numel(p) for chimera_unet.parameters())
```

and refuses unless `P_new == P_donor`. A unit test additionally requires an
exact `(name, shape)` census match.

The conditioning bridge is deliberately outside the U-Net count. The model
reports all three numbers separately:

```text
unet_trainable_parameters
bridge_trainable_parameters
total_trainable_parameters_for_stage
```

### 4.1 Initialization modes

`scratch` (default):

- construct from the donor config;
- use the module's normal initialization under a recorded seed;
- zero-initialize the final output convolution after construction;
- do not channel-copy or partially inherit donor weights.

`sdxl_transplant` (experimental):

- construct the identical U-Net;
- strict-load every donor U-Net tensor;
- do not zero the donor output convolution;
- record donor content hash and config hash;
- require bridge alignment before the normal U-Net diffusion stage.

The transplanted U-Net expects CLIP-derived conditioning. A random bridge can
have the right `[B,77,2048]` shape while being in the wrong feature space. The
shape match alone is not evidence that transplant initialization is useful.

It also did not train with Chimera's SenseNova-style RoPE. Transplant mode is
therefore a **weight warm start**, not a claim that the copied U-Net initially
computes the same function as the donor.

### 4.2 SenseNova-style positional encoding

The U-Net keeps the donor's modules and trainable tensors, but its attention
processor applies rotary position encoding to Q/K immediately before attention
dispatch. It uses SenseNova's `rotate_half` layout and checkpoint-declared
`rope_theta` / `rope_theta_hw` conventions.

For an attention head of width `D_head`, format v1 requires divisibility by
four and uses the SenseNova split:

```text
t: D_head / 2
h: D_head / 4
w: D_head / 4
```

A donor whose head width cannot satisfy this contract is refused by the
bootstrap builder rather than silently using another positional scheme.

#### U-Net spatial query positions

Every attention site is mapped into one shared canonical grid independent of
its down/up stage. The canonical unit is 32 output pixels, the native
SenseNova generation-token width. For feature position `(i,j)` in a map of
shape `(H_s,W_s)`:

```text
h = crop_top  / 32 + (i + 0.5) * target_h / (32 * H_s) - 0.5
w = crop_left / 32 + (j + 0.5) * target_w / (32 * W_s) - 0.5
t = prefix_terminal_t
```

RoPE accepts floating-point positions. An H/8 map consequently has
quarter-step coordinates, H/16 has half-step coordinates, and H/32 has
integer-step coordinates for an aligned uncropped canvas. The same physical
location receives the same coordinate at every U-Net stage.

For spatial self-attention every token has the same `t`, so its temporal
rotation cancels in Q/K relative products. It is effectively a
SenseNova-layout 2-D RoPE over `h,w`, while sharing the same three-axis
implementation as cross-attention.

#### Prefix-memory positions

The bridge resampler returns memory content and a position triple:

```text
C       [B,77,2048]
P_ctx   [B,77,3]
```

Each memory row's position is the masked, normalized attention-weighted
barycenter of the source prefix positions that produced it. Text content keeps
sequence position on `t` with `h=w=0`; reference-image content retains its
spatial contribution. Token-type and reference-id embeddings remain separate.

Cross-attention performs:

```text
Q = rope_3d(to_q(unet_feature), P_query)
K = rope_3d(to_k(C),            P_ctx)
V =         to_v(C)
```

Inference caches post-RoPE K. Crop, target size, context, positional-layout
version, or reference-layout changes therefore invalidate the cache.

A barycenter is a hypothesis, especially when one memory row combines distant
reference regions. The bridge exposes resampler weights and position variance
for diagnostics. Before the first compatible checkpoint is published, compare:

1. full three-axis cross-attention RoPE (format-v1 proposal);
2. spatial RoPE on U-Net self-attention only;
3. no RoPE, reproducing donor SDXL attention treatment.

The selected mode is checkpoint structure, never a live generation toggle.

#### Resolution and crop conditioning

The six SDXL values remain wired through the donor U-Net's existing addition
embedding:

```text
[original_h, original_w, crop_top, crop_left, target_h, target_w]
```

This is intentionally redundant: RoPE tells attention where an individual
query lies, while micro-conditioning tells all relevant blocks the global
canvas, crop, and requested target geometry. Training uses real dataset crop
metadata. Missing values are derived from the actual preprocessed geometry,
not a fixed 1024x1024 constant. Inference defaults to original size equal to
target size and a zero crop origin unless explicitly supplied.

## 5. Conditioning bridge

### 5.1 Prefix inputs

The understanding prefix keeps SenseNova's existing query builder, tokenization,
reference-image embedding, block-causal mask, and `(t,h,w)` indexes. The wrapper
returns:

```text
last_hidden_state                 [B,L,D_llm]
selected post-RoPE K/V per layer [B,H_kv,L,D_head]
attention mask                    [B,L]
token type                        text | special | reference
position indexes                  [3,L]
reference segment ids             [L]
```

Selected layers are relative to the source checkpoint depth rather than fixed
to 42:

```text
round(0.25 * (layers - 1))
round(0.50 * (layers - 1))
round(0.75 * (layers - 1))
layers - 1
```

The prefix forward must return these tensors explicitly. It must not use a
mutable cache side effect as an autograd boundary. Existing SenseNova's
`return_kv` seam is the implementation precedent.

### 5.2 Canonical memory

Each selected layer owns a projection because different layers' K/V spaces are
not assumed aligned:

```text
E_l = Project_l(concat(flatten_heads(K_l), flatten_heads(V_l)))
E_h = Project_hidden(RMSNorm(last_hidden_state))
```

Layer, token-type, reference-id, and position embeddings are added before a
small learned-query resampler. The resampler emits:

```text
encoder_hidden_states C = [B,77,2048]
pooled_text_embeds    G = [B,1280]
context positions P_ctx = [B,77,3]
```

The initial gate on each KV residual path is zero. The final-hidden path is
active at initialization. This permits the bridge to begin from a stable
content representation while measuring whether selected KV layers add value.

### 5.3 CLIP-space alignment stage

The selected SDXL donor's two frozen text encoders produce the target tensors
through the existing SDXL prompt encoder:

```text
C_teacher = concat(CLIP-L penultimate, OpenCLIP-bigG penultimate) [B,77,2048]
G_teacher = OpenCLIP projection output                           [B,1280]
```

For the same caption, the frozen SenseNova understanding branch and trainable
bridge produce `C_student`, `G_student`. The bridge alignment loss is:

```text
L_bridge = mse(norm(C_student), norm(C_teacher))
         + lambda_scale * mse(rms(C_student), rms(C_teacher))
         + lambda_pool  * mse(G_student, G_teacher)
```

Padding rows are masked. An additional cosine diagnostic is reported but is
not silently substituted for the declared loss.

Alignment is a necessary compatibility stage for `sdxl_transplant`. For a
scratch U-Net it is recommended because it permits the same conditioning cache
and supplies a stable scale, but the scratch route may proceed from
`unaligned` only with an explicit warning.

The first implementation aligns text-only conditioning. Reference-image slots
are zero-gated and trained in the later joint/reference phase; SDXL CLIP has no
equivalent teacher target for them.

### 5.4 Conditioning cache

After bridge alignment, the frozen understanding+bridge pair may precompute:

```text
C [77,2048] bf16
G [1280] bf16
P_ctx [77,3] fp32
mask / size-independent metadata
```

per normalized caption. The cache key includes:

- understanding source content hash;
- tokenizer/chat-template hash;
- bridge checkpoint hash;
- caption after all configured caption transforms;
- negative/dropout variant;
- dtype and context length.
- positional-layout version and RoPE bases.

The cache stores canonical memory, not U-Net site K/V. U-Net `to_k` and `to_v`
weights are trainable, so caching their outputs during training would become
stale after the first optimizer step.

Caption mutations that occur per epoch either participate in the key or force
on-the-fly conditioning. A cache hit must not bypass configured caption dropout.

## 6. Diffusion/flow contract

The VAE yields normalized four-channel latents:

```text
x0 [B,4,H/8,W/8]
```

Training uses the SenseNova time direction but predicts velocity directly:

```text
x_t    = t*x0 + (1-t)*sigma*epsilon
target = x0 - sigma*epsilon
v_pred = unet(x_t, t, C, G, time_ids)
loss   = mse(v_pred, target)
```

`t=0` is noise and `t=1` is clean. Sampling uses:

```text
x_next = x + (t_next - t) * v_pred
```

The first checkpoint format fixes `sigma=1` after VAE normalization. A future
noise-scale calibration cannot reuse SenseNova's resolution-dependent pixel
formula without a separate measured decision.

The training timestep distribution and inference shift live in architecture
defaults. They must be added to `backend/api/param_defaults.py`; routes and UI
must not hardcode a second value.

## 7. Generation wiring

### 7.1 Load ownership

Add:

```text
backend/core/models/sensenova_sdxl_chimera/
  __init__.py
  config.py
  loader.py
  understanding.py
  conditioning_bridge.py
  prefix.py
  attention_processor.py
  pipeline_ops.py

backend/core/pipeline_backends/sensenova_sdxl_chimera.py
```

The loader returns:

```text
{
  type,
  understanding,
  tokenizer,
  condition_bridge,
  unet,
  vae,
  scheduler_config,
  metadata,
  source_provenance
}
```

Add `sensenova_sdxl_chimera` to `ModelType`, detection, model-info responses,
pipeline component slots, teardown, hot-model accounting, and current-model
reporting. Metadata detection must run before broad SD/SDXL key heuristics.

### 7.2 Prompt prefill

For each required CFG branch:

1. Build the SenseNova text/reference prefix.
2. Run the frozen understanding wrapper once.
3. Run the bridge to get `C`, `G`.
4. Free native selected K/V and last hidden after the bridge completes.
5. For each U-Net cross-attention site, run its existing `to_k(C)` and
   `to_v(C)` once, apply three-axis RoPE to K using `P_ctx`, and store
   immutable post-RoPE K/V.

The site cache is keyed by the exact attention module identity, branch, batch,
dtype, device, bridge-output hash, positional-layout version, RoPE bases,
original/target size, crop origin, and reference layout. It is
single-generation state and is cleared in `finally`.

The attention processor introduces no parameters. When no site cache is armed,
it executes the ordinary U-Net path. That fallback is used for training and is
part of the parity test.

### 7.3 CFG

The semantic branches match SenseNova:

- text-only: `cond`, optional `uncond`;
- with references: `cond`, optional `img_cond`, optional `uncond`.

The first implementation supports sequential branches and a batch-concatenated
mode. Batched CFG repeats `x_t` on the batch axis and concatenates branch K/V;
it must be numerically equivalent to sequential execution within the declared
dtype tolerance. Automatic selection uses a measured VRAM estimate, not only
batch count.

### 7.4 Image routes

Delivery order:

1. txt2img;
2. img2img using SDEdit start time in the same latent space;
3. inpaint with the standard latent mask/input-channel contract selected by
   the donor U-Net config;
4. outpaint delegated through the same inpaint implementation;
5. reference-image conditioning after its independent quality gate.

The loader exposes the U-Net's actual in/out channel contract. A donor with an
unsupported inpaint-specific input layout is refused during bootstrap rather
than partially copied.

### 7.5 Preview and VAE handling

Use the bundled VAE's declared family/identity through the shared VAE registry.
TAESD preview selection, tiling, decode normalization, and override
compatibility follow the existing component metadata. A generation-time VAE
override is unsupported in the first boundary because the U-Net was trained in
the bundled donor VAE's latent identity, even though another SDXL-shaped VAE
has the same four channels.

## 8. Training design

### 8.1 Training stages

One architecture-specific key selects the stage:

```text
chimera_training_stage:
  bridge_align | unet | joint
```

#### `bridge_align`

Trainable:

- conditioning bridge;
- pooled-conditioning head.

Frozen:

- SenseNova understanding;
- SDXL donor CLIP encoders;
- U-Net;
- VAE.

The run consumes captions and performs no image diffusion loss. It saves a
normal Chimera checkpoint with updated bridge tensors and
`bridge_state="aligned"` only after the registered validation gate passes.

#### `unet` (default diffusion stage)

Trainable:

- complete U-Net, with exactly the selected SDXL donor's U-Net parameter set.

Frozen:

- understanding;
- bridge;
- VAE.

Use cached `C/G` whenever the caption pipeline permits it. With a cache hit,
neither the understanding branch nor bridge is loaded onto the GPU for the
step. The live forward/backward graph is therefore the ordinary SDXL U-Net
graph plus the flow objective.

For `sdxl_transplant`, `bridge_state="aligned"` is required. For `scratch`, an
unaligned bridge is accepted only through an explicit experimental override
and emits a persistent run warning.

#### `joint` (experimental finish)

Trainable:

- bridge;
- U-Net.

Frozen:

- understanding;
- VAE.

Conditioning is computed on the fly. The understanding phase runs under
`no_grad`; its outputs are detached inputs to the bridge. Understanding weights
are then moved/offloaded before the U-Net backward according to a dedicated
phase conductor. This stage does not claim the same throughput or peak VRAM as
SDXL training.

### 8.2 Architecture handler and ops

Add:

```text
backend/core/training/arch/sensenova_sdxl_chimera.py
backend/core/training/ops/sensenova_sdxl_chimera_ops.py
backend/core/training/adapters/sensenova_sdxl_chimera_adapter.py
```

Register the handler in `ARCH_REGISTRY`, `_EXPECTED_ARCH_KEYS`,
`resolve_arch_name`, cache namespace resolution, and every architecture census.

The handler declares:

- latent channels: 4;
- latent ndim: 4;
- VAE scale factor: 8 from the bundled component declaration;
- VAE normalization: component-declared SDXL normalization;
- pixel alignment: donor U-Net structural requirement, at least 8 and reported
  from the loaded config;
- text conditioning: fixed 77x2048 plus pooled 1280;
- added conditioning: SDXL six-value time ids;
- position encoding: SenseNova-layout three-axis RoPE, with U-Net spatial
  queries mapped into the shared 32px canonical coordinate grid;
- prediction: flow velocity, `t=0` noise and `t=1` clean.

The ops module owns component load, bridge alignment, conditioning cache,
VAE encode/decode, noising, U-Net forward, loss, training-time sample, and
checkpoint restoration. Do not route it through `sd_sdxl_ops.py` by pretending
the architecture has CLIP text encoders at generation time.

### 8.3 Supported methods

First release:

| Method/feature | Boundary |
|---|---|
| `full_finetune` | supported for the stage-selected bridge/U-Net tensors |
| ordinary LoRA | unsupported until generation apply + training save/resume round-trip tests exist |
| LyCORIS | unsupported |
| ReLoRA | unsupported |
| ControlNet training | unsupported |
| train understanding | unsupported |
| VAE training | unsupported in a Chimera run |
| latent cache | supported for `unet` stage |
| conditioning cache | supported for `unet` stage |
| gradient checkpointing | supported through the donor U-Net implementation |
| block swap | unsupported initially; add only after a dedicated U-Net training conductor is measured |
| REPA | unsupported initially; it needs its own U-Net tap validation |
| EMA | follows the ordinary U-Net path only after save/resume coverage |

Both adapter capability tables need explicit conservative rows. Do not inherit
the existing SDXL or SenseNova rows by name.

### 8.4 Parameter groups and saves

Parameter groups are stage-exact:

```text
bridge_align -> condition_bridge
unet         -> unet
joint        -> unet, condition_bridge
```

The active-parameter census must prove that every declared tensor receives a
gradient and that no frozen understanding, donor CLIP, or VAE tensor enters the
optimizer.

Every save writes:

- U-Net config and tensors;
- bridge config and tensors;
- bundled VAE tensors and identity metadata, unless safely inherited from the
  same artifact without rewriting;
- understanding locator and required hash;
- initialization provenance;
- bridge state and alignment metrics;
- flow/time contract;
- training stage lineage.

Resume refuses a changed understanding hash, VAE identity, donor U-Net config,
context length, conditioning dimensions, or prediction contract. Relocating an
identical understanding file is allowed after hash verification.

### 8.5 Training API parameters

Proposed keys, all defaulted only in `backend/api/param_defaults.py` and fully
declared in `openapi.yaml`:

| Key | Values | Default | Meaning |
|---|---|---|---|
| `chimera_training_stage` | `bridge_align`, `unet`, `joint` | `unet` | Active training graph |
| `chimera_allow_unaligned_scratch` | bool | false | Explicitly permit scratch U-Net diffusion training before bridge alignment |
| `chimera_conditioning_cache` | bool | true | Cache frozen bridge outputs during `unet` stage |
| `chimera_bridge_lr` | float | shared LR unless set | Bridge group LR override |
| `chimera_context_dropout` | float | 0.1 | CFG/null-conditioning training probability |
| `chimera_clip_hidden_weight` | float | declared value | Bridge hidden alignment weight |
| `chimera_clip_pooled_weight` | float | declared value | Bridge pooled alignment weight |

The exact numerical defaults for the loss weights are selected during the P0
probe and then entered once in `param_defaults.py`; this proposal does not
invent unmeasured constants.

## 9. Bootstrap builder

### 9.1 API and CLI

Add a shared builder function and two callers:

```text
backend/core/models/sensenova_sdxl_chimera/builder.py
examples/api/build_sensenova_sdxl_chimera.py
POST /api/v1/models/sensenova-sdxl-chimera/initialize
```

Inputs:

```text
output_name                 relative name under configured model root
understanding_source        existing SenseNova model
sdxl_source                 existing SDXL model
unet_initialization         scratch | sdxl_transplant
initialization_seed         integer
context_tokens              fixed to 77 in format v1
```

The endpoint accepts only an output name, not an arbitrary output path. The
backend resolves and verifies that the target remains beneath its configured
model root. It refuses a non-empty target. It writes to a temporary sibling
directory and atomically renames only after validation, so an interrupted build
does not leave a selectable partial model.

`unet_initialization="scratch"` is the API default. The experimental option is
labelled in both OpenAPI and UI.

### 9.2 Build sequence

1. Header-preflight the SenseNova and SDXL sources without unloading the live
   model.
2. Verify the SenseNova source has the supported dense understanding branch,
   tokenizer siblings, vision tower, and declared geometry.
3. Load the SDXL donor U-Net config and compute its parameter census.
4. Build a same-config U-Net under the recorded seed.
5. For `sdxl_transplant`, strict-copy the donor U-Net; for `scratch`, retain
   default initialization and zero only the output convolution.
6. Construct the bridge from the observed SenseNova dimensions.
7. Extract the SDXL VAE through `resolve_vae_source("model:<source>")`, preserve
   its normalization/identity metadata, and bundle its tensors.
8. Write U-Net, bridge, and VAE through the shared sharded single-file writer.
9. Write the manifest last.
10. Reload the artifact through the production loader on CPU/meta where
    possible; compare all tensor/config censuses and hashes.

### 9.3 Full-size bootstrap artifacts

The first real artifact is built at:

```text
<MODEL_ROOT>/snu1.5_sdxl_chimera
```

with `scratch` initialization. The same sources may be used to build a sibling
experimental artifact named `snu1.5_sdxl_chimera_sdxl_init`.

These are machine-local test artifacts, not committed fixtures. CPU tests use
tiny synthetic configs in a temporary directory. No test assumes that a drive
letter or the real source models exist.

## 10. API and backend integration surface

### 10.1 OpenAPI-first changes

Update `openapi.yaml` before or with code for:

- the initializer request/response;
- the new model type in model-info schemas;
- architecture capability responses;
- training stage and Chimera training parameters;
- any architecture-specific status fields;
- examples for scratch and transplanted bootstrap builds.

All request defaults come from `backend/api/param_defaults.py`. The Pydantic and
`Form()` declarations in `backend/api/routes.py` reference those values.

### 10.2 Generation routing

Required backend touch points:

- `backend/core/model_loader.py`: type, detection, header preflight, loader;
- `backend/core/pipeline.py`: component slot, loaded flag, dispatch, teardown,
  current-model info;
- `backend/core/pipeline_backends/sensenova_sdxl_chimera.py`: route-specific
  entry points;
- `backend/api/routes.py`: image-route acceptance and initializer endpoint;
- `backend/api/arch_capabilities.py`: honest feature support/refusals;
- `backend/core/models/components/wiring.py`: Chimera wiring;
- generation override/component registry: bundled VAE reporting and refusal of
  identity-changing override;
- gallery/queue metadata: architecture id and reference-image payloads.

No new WebSocket message type is required. Prefill uses the existing phase
progress convention; denoise steps use the normal progress callback.

### 10.3 Training routing

Required touch points:

- architecture handler/ops/adapter files from section 8;
- `ARCH_REGISTRY`, expected-key assertion, and resolution order;
- `base_trainer.py` model flags, cache namespace, component loading, parameter
  census, sample path, and checkpoint lineage;
- `training_config.py` section vocabulary;
- `train_runner.py` preflight and stage contract;
- training route create/update Pydantic models;
- capability matrix and required values;
- metric registry for bridge alignment and Chimera-specific diagnostics.

Proposed metrics:

```text
chimera_clip_hidden_cosine
chimera_clip_hidden_rms_ratio
chimera_clip_pooled_cosine
chimera_velocity_loss
chimera_prefix_seconds
chimera_unet_step_seconds
chimera_condition_cache_hit_rate
```

## 11. Frontend plan

Frontend build/type-check remains the repository owner's step. Implementation
is verified by careful type reading and backend schema tests.

### 11.1 Model initializer

Add a **Create SenseNova SDXL Chimera** section to the model load UI:

- SenseNova understanding-source selector;
- SDXL donor selector;
- U-Net initialization selector:
  - `Scratch (default)`;
  - `Copy SDXL U-Net (experimental)`;
- initialization seed;
- output model name, constrained to a basename;
- estimated output size from donor headers;
- Build button and progress/error state.

The UI sends source identifiers returned by the model catalog; it does not
construct filesystem paths. After a successful build it refreshes the model
catalog and offers to load the new artifact. It does not auto-start training.

Likely files:

```text
frontend/src/utils/api.ts
frontend/src/components/common/ModelLoadSection.tsx
frontend/src/contexts/ModelComponentsContext.tsx
```

### 11.2 Generation panels

Add the model id to frontend model unions and capability resolution. Do not
fork the complete SDXL panel. Existing txt2img/img2img/inpaint/outpaint controls
render according to backend capabilities.

Chimera-specific status shown read-only:

- understanding source and verification state;
- VAE donor identity;
- U-Net initialization provenance;
- bridge state;
- context length;
- whether inference KV caching is active.

SenseNova-only controls such as MoT phase eviction, SenseNova KV-cache
streaming, `sensenova_gen_patch`, and latent refiner must not appear. The new
site-local U-Net KV cache is automatic and is not a generation checkbox in
format v1.

Reference-image controls remain hidden until the reference acceptance gate has
passed.

Primary files:

```text
frontend/src/components/generation/Txt2ImgPanel.tsx
frontend/src/components/generation/Img2ImgPanel.tsx
frontend/src/components/generation/InpaintPanel.tsx
frontend/src/components/generation/OutpaintPanel.tsx
frontend/src/components/generation/GenerationQueueProcessor.tsx
frontend/src/utils/api.ts
```

### 11.3 Training UI

The training form reads `training_required_values` and capability metadata.
When Chimera is selected it shows:

- training stage;
- bridge-alignment controls only for `bridge_align`;
- conditioning-cache toggle only for `unet`;
- the explicit unaligned-scratch override only when relevant;
- U-Net and bridge parameter counts;
- immutable provenance for understanding, VAE, and U-Net initialization;
- a hard explanation when transplanted U-Net training is blocked on bridge
  alignment.

Files:

```text
frontend/src/components/training/TrainingConfig.tsx
frontend/src/components/training/trainingConfigDefinitions.tsx
frontend/src/components/training/trainingParams.ts
frontend/src/utils/api.ts
```

The UI does not replicate stage legality. It disables controls from backend
capabilities and displays backend refusal text.

## 12. Implementation phases and gates

### P0 — configuration, census, and tiny-model contract

Implement config schemas, donor-config import, scratch/transplant construction,
bridge shapes, and tiny CPU models.

Gate:

- exact donor/new U-Net `(name,shape)` census and parameter-count equality;
- scratch and transplant initialization are distinguishable and reproducible;
- bridge emits 77x2048 and pooled 1280;
- flow noising and Euler identities pass;
- malformed/missing source declarations fail before construction.

### P1 — builder and production loader

Implement sharded artifacts, manifest, source hashes, VAE extraction/bundling,
understanding-only loader, detection, and preflight.

Gate:

- tiny artifact build -> production reload -> identical tensors/config;
- source relocation with same hash succeeds;
- source mutation/missing source refuses before active-model teardown;
- bundled VAE round-trip preserves identity and normalization;
- no `_mot_gen` parameter is materialized.

### P2 — conditioning and txt2img inference

Implement prefix capture, bridge, SDXL-compatible conditioning, flow sampler,
attention processor, site-local K/V cache, VAE decode, pipeline dispatch, and
API txt2img wiring.

Gate:

- cached vs uncached U-Net attention output parity per dtype;
- sequential vs batched CFG parity;
- cond/uncond branch separation;
- deterministic same-seed generation;
- finite random-weight full-size smoke output;
- cache cleanup after success and exception;
- no persistent native 42-layer KV after bridge construction.

### P3 — bridge alignment training

Implement `bridge_align`, donor CLIP teacher loading, loss/metrics, saves,
resume, and UI.

Pre-register alignment thresholds on a held-out caption set after measuring the
teacher's own dtype/re-encode floor. `bridge_state="aligned"` is written only
when every threshold passes; a low training loss alone is insufficient.

Gate:

- only bridge parameters change;
- resume reproduces LR/optimizer state and validation metrics;
- teacher encoders and understanding remain bit-identical;
- aligned checkpoint production-reloads for generation.

### P4 — U-Net full training

Implement `unet` stage, latent cache, conditioning cache, gradient
checkpointing, full save/resume, and training-time samples.

Gate:

- trainable parameter count equals donor SDXL U-Net exactly;
- every U-Net parameter receives a finite gradient in the coverage test;
- no bridge/understanding/VAE parameter changes;
- cached and on-the-fly frozen conditioning produce equal train inputs;
- three finite optimizer steps, save, reload, three more steps;
- training-time sample equals standalone generation for the same checkpoint,
  seed, prompt, steps, and settings;
- measured U-Net-stage peak VRAM is compared with ordinary SDXL under identical
  resolution, batch, dtype, checkpointing, optimizer, and cache settings.

### P5 — full-size bootstrap model and real smoke

With the repository owner choosing the two real source models, build:

```text
<MODEL_ROOT>/snu1.5_sdxl_chimera
```

Do not start or restart servers directly. Use the already running backend; if
it is absent, ask the owner to start it. A backend restart, if required, goes
through `POST /api/v1/system/restart-backend`.

Smoke sequence:

1. Initialize scratch artifact through the production API.
2. Load it through `/api/v1/models/load`.
3. Confirm `/api/v1/models/current` reports both source provenances, four
   latent channels, VAE identity, exact U-Net count, and `bridge_state`.
4. Generate a small txt2img sample with CFG off and assert a finite decoded
   image. Quality is not a gate for random weights.
5. Create a tiny `bridge_align` run, execute finite steps, save, and reload.
6. Create a tiny `unet` run, execute three finite steps, save, reload, and
   execute three more.
7. Generate from the resumed checkpoint.
8. Repeat construction as `sdxl_transplant`; complete/already provide bridge
   alignment, then run the same U-Net smoke.

The full-size artifact is machine-local and never committed.

### P6 — frontend completion

Wire initializer, model unions/status, generation dispatch, training stages,
capabilities, warnings, and queue persistence.

Gate:

- every initializer field round-trips through the API;
- no Chimera request is serialized as `sensenova` or `sdxl`;
- editing an unrelated training field does not reset stage/provenance controls;
- required-value refusals are visible before submission;
- queue reload retains the architecture id and reference payload metadata;
- repository owner runs frontend type-check/build.

### P7 — img2img, inpaint, outpaint, references, and joint finish

Ship each independently after its gate:

- img2img: zero/full denoising-strength endpoint tests and deterministic SDEdit;
- inpaint: mask polarity, latent geometry, and preserved-region comparison;
- outpaint: canvas placement and preserved-region comparison;
- reference images: fixed prompt/reference suite against no-reference controls,
  including identity/detail and prompt adherence;
- joint stage: measured peak VRAM/step time and proof that understanding stays
  frozen while bridge and U-Net update.

No route is advertised before its own backend and frontend gate passes.

## 13. Verification matrix

### CPU and schema

- exact U-Net parameter/name/shape parity with donor config;
- bridge tensor shape and mask behavior for lengths 1, 77, and long SenseNova
  prefixes;
- agreement with the existing SenseNova `rotate_half`/three-axis RoPE helper at
  integer positions;
- equal physical points mapping to equal `(h,w)` positions across U-Net stages;
- crop offsets shifting query coordinates by exactly
  `(crop_top/32,crop_left/32)`;
- self-attention's constant `t` axis having no effect on relative attention;
- context-position barycenters respecting padding masks and reference
  boundaries;
- selected-layer indices for non-42-layer tiny configs;
- artifact schema strictness and unknown-version refusal;
- source hash and relocation behavior;
- VAE extraction/bundle/load parity;
- OpenAPI/default/Pydantic/frontend-default parity;
- architecture registry and capability completeness;
- save/resume stage legality;
- condition-cache key sensitivity.

### CUDA smoke

- cached/uncached cross-attention parity in bf16;
- cached post-RoPE K parity and invalidation on crop/size/reference changes;
- full txt2img pass for scratch and transplant artifacts;
- gradient checkpointing on/off gradient comparison on a small config;
- three-step bridge and U-Net runs;
- save/reload and generation;
- peak allocated/reserved VRAM and seconds split into prefix, bridge, U-Net
  step, and VAE decode.

### Acceptance measurements

Compare against ordinary SDXL and latent SenseNova using the same SDXL VAE,
hardware, dtype, resolution, batch, optimizer, checkpointing, prompt set, and
step count.

Report separately:

- one-time prefix latency;
- per-denoise-step latency;
- total generation latency at 20 and 30 steps;
- training seconds/update;
- peak allocated/reserved VRAM;
- context-cache disk size and hit rate;
- 1024, 2048, and 4096 output resolutions;
- CFG 1 and CFG-enabled modes;
- scratch and transplanted initialization.

The U-Net-stage memory claim passes only if its measured peak is within a
pre-registered tolerance of ordinary SDXL. The test must exclude the
`bridge_align` and `joint` stages, which deliberately have different graphs.

The architectural speed premise passes only if the new denoise loop materially
beats latent SenseNova at 2048 while retaining the H/8 convolutional path. A
2x per-step target is a project gate, not an unmeasured claim.

Quality evaluation separates:

- prompt adherence;
- global composition;
- fine detail and repeated-grid artifacts;
- text rendering;
- reference identity/style fidelity when that path is enabled;
- VAE round-trip floor;
- scratch vs transplant convergence at matched updates and data order.

## 14. Known hazards

1. **SDXL transplant conditioning mismatch.** Correct tensor shape is not
   correct feature space. Bridge alignment is mandatory before judging donor
   weight reuse.
2. **Parameter parity is not process parity.** Keeping the frozen SenseNova
   understanding model GPU-resident during every U-Net step destroys the SDXL
   memory claim. The default U-Net stage uses cached conditioning.
3. **Context compression.** Mapping a long multimodal SenseNova prefix to 77
   rows may lose reference detail. Raising context length changes attention
   activation and is a new checkpoint format decision, not a generation knob.
4. **Layer-space mixing.** K/V from different understanding layers cannot be
   averaged directly. Each selected layer needs its own adapter before fusion.
5. **Stale training KV.** U-Net site K/V cannot be cached across optimizer
   steps because `to_k/to_v` are trainable. Only canonical conditioning is
   training-cacheable.
6. **External understanding dependency.** Hash-pinned references save disk but
   are not portable. Missing dependencies must fail early and clearly.
7. **Donor variation.** “SDXL” checkpoints may carry modified U-Nets or VAEs.
   Format v1 accepts only the declared four-channel, 2048-context,
   1280-pooled contract; equality is to the selected accepted donor, not an
   assumed parameter number.
8. **Random full-size smoke quality.** A scratch bootstrap can prove wiring and
   finite math but cannot establish useful generation quality.
9. **Reference conditioning lacks a CLIP teacher.** It needs diffusion/joint
   training and a separate acceptance suite.
10. **Frontend architecture aliases.** Reusing `isSDXL` to render Chimera may
    accidentally expose unsupported SDXL component switching. Use capability
    data and the explicit model id.
11. **RoPE changes donor behavior.** A transplanted SDXL U-Net no longer
    computes the donor function even with identical tensors. Compare all three
    positional ablations at matched initialization and data order before
    attributing a convergence change to transplantation.
12. **Soft context positions.** A barycenter can poorly describe a resampler
    row attending to distant reference regions. Report position variance;
    high-variance rows may require typed memory banks rather than pretending
    they have one location.

## 15. Documentation updates when implemented

After the corresponding behavior ships:

- add the architecture count and routing notes to `AGENTS.md` and
  `docs/guides/ADD_A_MODEL_ARCHITECTURE.md`;
- add implementation facts to `docs/guides/MODEL_FACTS.md`;
- add training internals to `backend/core/training/MODEL_ARCHITECTURES.md`;
- add a stable architecture reference under
  `docs/reference/architectures/sensenova_sdxl_chimera.md`;
- update `docs/guides/ARCHITECTURE_MAP.md` and request lifecycle where routing
  differs;
- record measured speed/memory/quality only with conditions;
- record any adapted external implementation and license in
  `docs/legal/THIRD_PARTY_PROVENANCE.md`.

This proposal remains the implementation plan until those current-behavior
documents supersede each completed section.
