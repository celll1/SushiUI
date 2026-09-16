# SenseNova Latent Refiner (optional fine-scale head branch)

Status: **implemented on `flux2` through `567fe62f`.** P0-P3 are complete.
The isolated GPU memory matrix and a real 2048-bucket `refiner_only` attach,
save and two resumes are complete; the matched quality/convergence experiment
at the end is still pre-registered and has not been run. Nothing here claims
that five smoke-test updates improve image quality.

### Implementation and runtime record (2026-09-16/17)

- Forward/module and strict declaration: `fe37f4bb`; training modes and
  persistence: `73541704`; API/UI/probes: `a792a624`.
- Follow-up correctness commits preserve bf16 bases (`2392124e`), bound attach
  calibration memory (`a620d484`, `d7aa561e`), separate base lineage from save
  layout (`901564dd`, `15b794bc`), and admit a self-contained refined
  checkpoint as a portable base-training input (`567fe62f`).
- Run 128,
  `sensenova_refiner_only_w128d3_run127_s93791`, starts from run127 step 93,791,
  uses dataset 38 only, one 2048-area bucket family, batch size 1, width 128,
  depth 3 and `refiner_only`. It completed steps 1-5 with finite flow losses
  `2.0197, 1.1605, 0.9929, 0.6876, 0.9306`. Steps 4 and 5 each reloaded the
  previous full checkpoint and restored all 32 named optimizer tensors.
- The attach guard measured 64 deterministic samples capped to 1024^2 pixels:
  pooled RMS `0.9785`, means
  `[0.2319, -0.1943, 0.0249, -0.3664]`, and per-channel RMS
  `[1.2435, 0.8825, 0.9190, 0.8124]`. During this measurement the base is on
  CPU; the VAE is returned to CPU and the CUDA cache is cleared before the base
  is restored. This fixes the original ~48 GiB base+VAE overlap.
- The final step-5 checkpoint has 1,149 tensors, including 33 refiner tensors,
  and records `sensenova_trained_branch=gen`,
  `sensenova_save_layout_branch=both`,
  `sensenova_refiner_training_mode=refiner_only`, effective format `bf16`, and
  the run127 source path. Decoder weights remained frozen.
- `sn_refiner_delta_rel` rose from exactly zero at steps 1-2 to
  `5.66e-6, 1.37e-5, 2.64e-5` at steps 3-5. This proves the zero-init branch
  begins moving and that resume continues it; it is far below the 0.05 capacity
  threshold and says nothing about convergence after five warmup updates.

## Problem

A VAE-swapped SenseNova run at `sensenova_gen_patch=8` on the SDXL VAE (run127)
puts all 42 transformer layers at a 64px token. Below that, the only learnable
computation is the pixel head `ConvDecoder`
(`backend/core/models/sensenova/vendor/modeling_fm_modules.py:583-603`):

| Grid | patch 8 on an 8x VAE |
|---|---|
| 64px | transformer (42 layers, width 4096) |
| 32px | head `conv1` (3x3, 1024 -> 1024) |
| 16px | head `conv2` (3x3, 256 -> 16) |
| 8px (one latent cell) | `PixelShuffle(2)` only |

The model predicts x0: `v = (x_pred - z) / (1 - t)`
(`vendor/modeling_neo_chat.py:704`; training `ops/sensenova_ops.py:2352-2353`).
The head receives the last hidden state only (`modeling_neo_chat.py:683-687`;
`sensenova_ops.py:2339-2341`). There is no path from the noisy input `z` to
the output at the latent-cell grid, and no timestep input to the head. Detail
already present in `z` has to be packed into a 64px token and re-drawn by two
convolutions. At low noise, where `x0 ≈ z`, that is all of the detail.

### Symptom measurement and its provenance

The numbers below come from a CPU spectrum/grid probe run once on 2026-09-16
against run127's and run126's sample PNGs and 12 real images, from a session
scratchpad (`hf_probe.py`, `hf_probe.json`, outside the repo). **They are not
reproducible from the repo as-is**: the probe is not committed, the sample
directories are on `M:\sushiUI\training\...`, and the run-to-run comparison has
no matched control. They motivate the design; nothing below depends on their
exact values. The committed replacement is
`backend/core/training/probes/sensenova_refiner_quality.py`.

Observed on run127's samples (same prompt and seed): 64px-periodic gradient
structure at 2.5x an incommensurate-period (61px) control (0.51 vs 0.20), and
8px-periodic structure at 4.5x the SDXL-VAE round-trip of real images (0.135
vs 0.030). 4-16px band energy about 0.4x of run126 (native 32px token). The
VAE round-trip kept 94-100% of every band in that probe. Single-sample grid64
values on run127 ranged 0.23-1.1. This is consistent with the missing path but
does not prove it: run127 and run126 also differ in the VAE swap itself, the
training length and the data.

The direction here keeps the transformer's token count and compute fixed. The
fine-grid capacity is added the way SDXL's UNet has it: convolutions at the
latent-cell grid plus a direct path from the input.

## Scope

In scope:
- SenseNova generation branch with a latent geometry (`gen_vae_scale_factor > 1`).
- Attach, continue and detach: full fine-tune only, same rule as the VAE swap
  rebuild (`backend/core/training/arch/sensenova.py:219-236`), and only on a
  run that trains the generation half (`train_unet`), because the branch is
  generation-side. Attach/continue on an understanding-only run is refused.
- Three full-fine-tune modes at a resume boundary: `joint` (base and refiner),
  `refiner_only` (frozen base) and `base_only` (frozen refiner, with gradients
  still flowing through it into the base). The state/mode matrix below is the
  complete contract; unsupported combinations are refused rather than coerced.
- Attach, continue and detach at a resume boundary.
- Generation: identical forward to training, driven by the checkpoint alone.
- Any non-full-fine-tune method on a base that already carries a refiner: the
  refiner runs frozen in the forward (`inherit`). This is what the
  value-dependent contract in [Configuration](#configuration-training) admits
  for LoRA / ReLoRA / ControlNet where the architecture otherwise allows the
  method.

Out of scope:
- Pixel-space SenseNova (`gen_vae_scale_factor == 1`). The refiner grid would
  be the pixel grid (2.36M positions at 1536px square). Refused, with the reason
  given.
- Adding or removing the branch inside a live training process. A running run's
  config cannot be edited (`PUT /training/runs/{id}` refuses while running or
  starting, `backend/api/routes.py:16067-16068`). "Mid-run" means between
  resume segments of the same run, which is where every other resume-time
  change in this repo happens.
- LoRA / ReLoRA / ControlNet training **of the refiner** (attach/detach under
  those methods).

## Mechanism

### Forward

One function, `apply_latent_refiner(refiner, x0_head, z_grid, t, noise_scale)`,
used by every call site (see [Call sites](#call-sites)):

```
x0_head  = head output, unpatchified to the latent grid   [B, C, H, W]
z_grid   = z unpatchified to the latent grid              [B, C, H, W]
c_in     = 1 / sqrt(t^2 + ((1 - t) * s)^2)                 [B, 1, 1, 1]   (s = noise_scale)
h        = stem(concat(x0_head, c_in * z_grid))           3x3, 2C -> width
h        = ResBlock_i(h, emb(t))   for i in 1..depth       ChannelRMSNorm2d + FiLM(t) + 3x3 x2
delta    = out(h)                                          3x3, width -> C, ZERO-INIT
x0       = x0_head + gate * delta
```

The v1 block is fixed, not an implementation choice left to P1. All 3x3
convolutions use stride 1 and zero padding 1. The timestep embedding is the
standard sinusoidal embedding of width `width` and max period 10,000, followed
by `Linear(width, 4*width) -> SiLU -> Linear(4*width, width)`. Each block owns
`Linear(width, 2*width)` and computes:

```
scale, shift = block_film(emb(t)).chunk(2, dim=channel)
u = norm1(h)
u = u * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
u = conv1(silu(u))
u = conv2(silu(norm2(u)))
h = h + u
```

Both norms are `ChannelRMSNorm2d(eps=1e-6)` with scale initialized to one;
all convolution and Linear biases use the PyTorch default initialization
except `out.weight` and `out.bias`, which are exactly zero. This definition,
including padding, epsilon, embedding and initialization, is part of
declaration version 1.

- `z = t * x0 + (1 - t) * eps * s` in this repo's convention
  (`sensenova_ops.py:2273`, t=1 clean). `c_in` gives the concatenated `z`
  unit-order scale at every `t` and every resolution. `s` comes from
  `compute_noise_scale`, which depends on token count. Raw `z` would have
  resolution-dependent statistics, and training uses one base resolution.
- **Input shapes.** Both callers hold `z` as tokens `[B, N, p*p*C]`
  (`sensenova_ops.py:2296`; `_t2i_predict_v`'s `z` argument). Both unpatchify
  it with `transformer.unpatchify` (the inverse of `patchify`, a permute and
  reshape with no arithmetic), so `z_grid` is the same tensor on both paths.
  `x0_head` is the head output `[B, C*p*p, token_h, token_w]` unpatchified the
  same way, before the token view the caller keeps for the `v` conversion.
- **Batch broadcasting of `t`.** Training holds `t` as `[B]` fp32
  (`sensenova_ops.py:2247-2249`) and hands `_build_step_context` `t` or `t[0]`
  (`:2297`); inference passes a 0-dim `ts[i]` (`sensenova_pipeline_ops.py:1052`,
  `:1059-1060`, `:734-739`). `apply_latent_refiner` accepts either: `t` is
  reshaped to `[-1]`, cast to fp32, expanded to `[B]` when it has one element,
  and refused otherwise (the same rule `_build_step_context` applies at
  `:684-692`). From it: `c_in` is `[B, 1, 1, 1]`, `emb(t)` is `[B, E]`, each
  FiLM scale/shift is `[B, width, 1, 1]`. `noise_scale` is one Python float
  per call on both paths (`sensenova_ops.py:2268`; `sensenova_pipeline_ops.py:1238`,
  `:1290`, `:1355`); a batch shares one resolution by construction
  (`batch_size > 1` requires bucketing, `arch_capabilities.py:1224-1227`), so
  one scale per batch is exact. In the packed train path only `image_embeds`
  is flattened to `[1, B*N, D]` (`sensenova_ops.py:2317`); `hidden` is viewed
  back to `[B, token_h, token_w, -1]` at `:2340` and `z` keeps its `[B, ...]`
  batch axis, so the refiner sees `[B, C, H, W]` there too. `train_step` passes
  its `[B]` `t`, never the `t[0]` it hands the embedder.
- `emb(t)` is the refiner's own sinusoidal embedding plus a 2-layer MLP. It
  does not reuse `fm_modules["timestep_embedder"]`, because that would couple
  the refiner's gradients into a tensor the transformer path already trains.
- **Normalization is spatially local.** `ChannelRMSNorm2d` computes, for each
  `[B, :, H, W]` position independently, `x / sqrt(mean_c(x^2) + eps)` in
  fp32, applies one learned per-channel scale (no bias), and casts back to the
  parameter dtype. GroupNorm is deliberately not used: PyTorch GroupNorm
  includes H and W in its reduction, which would make every output depend on
  the complete image, invalidate the finite receptive-field claim, and make
  halo tiling non-equivalent to a full-grid forward.
- Receptive field: the stem adds radius 1, each ResBlock adds 2, and `out`
  adds 1. The delta therefore has radius `2 * depth + 2`; at `depth=3` this
  is 8 latent cells, a 17-cell diameter (136px with the 8x SDXL VAE). It
  crosses the 8-cell (64px) transformer-token boundary, but is intentionally
  a local texture/detail branch rather than a second semantic backbone.
- `gate` is a persistent non-trainable buffer (`fm_refiner.gate`, 0-dim,
  float32), 1.0 while attached. It exists for annealed detach and is saved with
  the weights. Inference therefore reads exactly the value training last used.
- **Dtype policy is owned by the function.** `train_step` runs the head under
  `torch.autocast(cuda, bf16)` (`sensenova_ops.py:2327-2328`). Inference runs
  plain bf16 with no autocast (`sensenova_pipeline_ops.py`,
  `pipeline_backends/sensenova.py`). Today's head is convolutions only, so both
  paths compute in bf16. `apply_latent_refiner` disables autocast inside and
  casts explicitly: the RMS reduction and reciprocal square root in
  `ChannelRMSNorm2d` run in fp32; its result, FiLM, the timestep MLP and all
  convolutions run in the refiner's parameter dtype. Both callers therefore
  compute the same thing.
- **Only under `use_pixel_head`.** The refiner is built and called only on the
  ConvDecoder branch (`modeling_neo_chat.py:275-278, 674-692`). Training
  already refuses the deep/plain heads (`sensenova_ops.py:1065-1100`).
- **`noise_scale` is required.** With a declared refiner, `_t2i_predict_v`
  raises when `noise_scale` is not passed, instead of skipping the refiner. The
  vendor-internal callers (`modeling_neo_chat.py:987-1000, 1332-1345,
  1660-1714, 1884-1887`) have no SushiUI caller; this keeps them from silently
  running a different forward.
- **`c_in` assumes unit-RMS, mean-free latents with `x0` and `eps`
  uncorrelated.** It is exact only then: `Var(z) = t^2 Var(x0) + (1-t)^2 s^2`
  needs `Var(x0) = 1` per channel and zero cross-term. In this repo the SDXL
  run128 SDXL latents measured pooled RMS 0.9785 and per-channel RMS
  0.812-1.244. For another VAE it depends on that VAE's
  `norm`. This is not left to hold by assumption: `calibrate_before_training`
  (`arch/sensenova.py`) sees the run's latents and performs a
  per-channel mean/RMS measurement there whose result is logged and, when any
  channel's RMS is outside `[0.5, 2.0]` or `|mean| > 0.5`, refused for
  `attach` with the measured values. The declaration records
  `inputs: ["x0_head", "z_cin"]` so a later normalization change is a new
  version, not a silent reinterpretation. Attach samples 64 deterministic
  images capped to 1024^2 pixels while the base is CPU-resident; automatic
  noise-scale calibration remains uncapped at the assigned bucket geometry.
  The `[0.5, 2.0]` band is a guard
  against a mis-normalized VAE, not a tuned value.

Defaults: `width=128`, `depth=3`. This is a deliberately small residual
correction head, not an attempt to reproduce the base model at 4K. Parameter
count is independent of image size; compute and activation memory scale with
latent-grid area. Approximate forward arithmetic (not a benchmark):

| image | latent grid | transformer tokens | one width-128 bf16 map | refiner forward | transformer forward lower estimate |
|---|---:|---:|---:|---:|---:|
| 1536 | 192x192 | 576 | 9.4 MiB | 65 GFLOP | 9.4 TFLOP |
| 2048 | 256x256 | 1,024 | 16 MiB | 116 GFLOP | 16.7 TFLOP |
| 4096 | 512x512 | 4,096 | 64 MiB | 462 GFLOP | 66.8 TFLOP |

The transformer estimate is `2 * 8.17B * tokens` and excludes attention, so
the refiner remains below 1% of forward arithmetic. Width 256 is not a small
increment: it is approximately 4x the convolutional parameters/FLOPs and 2x
the feature activation. The pre-registered capacity ladder therefore changes
one axis at a time: width 64/depth 3 (~0.3M), 128/3 (~1.1M), 128/5 (~1.7M,
200px receptive-field diameter), then 256/3 (~4M). Dilation, down/up-sampling
or attention is not added until that ladder shows that local width/depth is
the limiting factor.

The v1 forward is full-grid. The local normalization makes an exact-halo
implementation possible later: split the latent grid into core tiles, include
`2 * depth + 2` cells of halo, preserve the full forward's zero padding only
at the outer image boundary, evaluate, and crop to the core. Such tiling must
pass full-grid equality tests before it becomes an execution optimization; it
is not a checkpoint semantic or a generation-quality toggle.

### Gradient checkpointing contract

Today only the decoder layer loop is checkpointed
(`sensenova_ops.py:2334`, `:2937-2938`, `checkpoint(..., use_reentrant=False)`);
`fm_head` is called plainly (`:2339`). The refiner does not inherit that flag by
position. The contract:

- `apply_latent_refiner` calls `torch.utils.checkpoint.checkpoint` itself,
  **per ResBlock**, when `checkpoint_blocks=True`; `train_step` passes
  `bool(trainer.gradient_checkpointing)`, inference passes nothing (no grad).
  The stem and `out` are not checkpointed (one activation each).
- `use_reentrant=False`, the policy every checkpoint call in this repo uses
  (`sensenova_ops.py:1502`, `:2938`; `anima_models.py:664`; `krea2/vendor/transformer.py:480`;
  `base_trainer.py:15062`).
- No stateful or RNG-consuming op inside a block: ChannelRMSNorm2d, FiLM,
  SiLU, conv.
  No dropout, no BatchNorm. The recompute is therefore deterministic, and the
  autocast-disabled region is inside the checkpointed function so the recompute
  runs under the same dtype policy as the first pass.
- Activation memory scales with area. The no-checkpoint retained-activation
  estimate for width 128/depth 3 is ~260 MiB at 1536, ~460 MiB at 2048 and
  ~1.85 GiB at 4096. With per-block checkpointing the retained block inputs
  are about 38 MiB, 64 MiB and 256 MiB respectively, plus one block's
  internals transiently during backward. The expected *measured peak delta*
  over the same base run is wider because allocator/workspace behavior is not
  captured: 0.1-0.3 GiB, 0.2-0.5 GiB and 0.8-1.5 GiB respectively. The isolated
  RTX 6000 Ada measurement is:

  | image | checkpointing | allocated delta | reserved delta |
  |---:|:---:|---:|---:|
  | 1536 | on | 0.204 GiB | 0.215 GiB |
  | 2048 | on | 0.362 GiB | 0.414 GiB |
  | 4096 | on | 1.447 GiB | 1.721 GiB |
  | 2048 | off | 0.722 GiB | 0.809 GiB |
  | 4096 | off | 2.888 GiB | 3.164 GiB |

  The 4096 checkpointed allocated delta passes the fixed 1.5 GiB gate by a
  narrow margin; the reserved delta does not fit that numerical bound and a
  complete 4096 base run was not attempted on the 48 GiB card. If a future
  complete 4096 run exceeds usable headroom, exact-halo execution becomes a
  release gate before increasing
  width; the architecture and checkpoint format do not change.

### Zero-init identity

With `out` zero-initialized, `x0 = x0_head` exactly at attach. The first
forward of an attached run is bit-identical to the same checkpoint without the
branch (verified in P1, see [Verification](#verification-plan)). The gradient
reaches `out` first and the stem/ResBlocks after `out` moves, which is the
ControlNet zero-conv pattern.

### Why the warmup matters here

run127 uses `lion8bit_ringbuffer`. Lion's update is `lr * sign(...)`, so a
fresh tensor moves at the full LR on its first step regardless of gradient
magnitude. Without a ramp, the zero-init `out` conv jumps to +-lr in every
element immediately. The existing fresh-parameter warmup covers this (next
section).

## Configuration (training)

All defaults go in `TRAINING_DEFAULTS` first (`backend/api/param_defaults.py`).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `sensenova_latent_refiner` | `"inherit" \| "attach" \| "detach"` | `"inherit"` | State request, see the state table |
| `sensenova_refiner_width` | int, `0` or a multiple of 16 in `[16,1024]` | `0` | `0` = inherit from checkpoint, else `128` on first attach |
| `sensenova_refiner_depth` | int, `0` or `[1,8]` | `0` | `0` = inherit from checkpoint, else `3` on first attach |
| `sensenova_refiner_training_mode` | `"joint" \| "refiner_only" \| "base_only"` | `"joint"` | Attach/continue trainability; detach has its own frozen-refiner contract |
| `sensenova_refiner_lr_factor` | float `> 0` | `1.0` | LR = `unet_lr * factor` |
| `sensenova_refiner_detach_mode` | `"anneal" \| "hard"` | `"anneal"` | How `detach` removes the branch |
| `sensenova_refiner_detach_steps` | int `>= 1` | `1000` | Anneal length, in optimizer updates |

`inherit` and `0` are sentinels for the same reason as `sensenova_gen_patch=0`
(`arch/sensenova.py:207-213`). `update_training_run` sends every Pydantic
default, so editing an unrelated field of a refined run must not arrive as
"off" and detach the branch silently.

### Capability contract (value-dependent)

The gate is on the **value**, not on the feature, because a refined base under
LoRA must still load and run its frozen refiner:

- `TRAINING_FEATURE_PARAMS["sensenova_latent_refiner"]` lists the seven keys
  (`arch_capabilities.py:271`), and `_add_training_feature_unsupported` (`:352`)
  declares the feature unsupported for every **non-SenseNova** arch, all
  methods. That hides the controls where the mechanism does not exist.
- On SenseNova the feature is **not** declared unsupported for any method.
  Instead `_add_training_required_value("sensenova", "sensenova_latent_refiner",
  "inherit", reason, methods=["lora", "relora", "controlnet"])` (`:471-489`)
  pins the state control to `inherit` under those methods. `training_required_values`
  narrows by method (`:1549-1551`), so a full fine-tune sees no pin. A client
  pins the control to the value; a PUT that sends the default `inherit` is
  accepted, so a normal LoRA create/update on a refined base is unchanged.
- `train_runner` enforces the same rule before the load: `attach` or `detach`
  with `training_method != full_finetune` is refused with the reason; `inherit`
  passes. Width/depth/training-mode/lr-factor/detach keys are inert on a base
  with no refiner under `inherit`, and are not refused on their own (a PUT
  carries their defaults). On a refined base, the mode is interpreted only by
  full fine-tuning; non-full methods always use the frozen-forward row.
- Latent geometry is a **runner** refusal, not a capability entry, for the same
  reason `sensenova_gen_patch` is (`train_runner.py:596-613`): whether the base
  is latent-space is read from its metadata, which the capability table cannot
  see. `attach` on a pixel base is refused with the reason given.
- No `unless` clause: the requirement holds for the listed methods regardless
  of the rest of the config.

### State and training-mode tables

Resolved in the arch handler at component load, before optimizer construction.
`ckpt` is the refiner state recorded in the checkpoint being loaded (none for a
fresh base).

| ckpt | request | Result |
|---|---|---|
| none | `inherit` | no refiner |
| none | `attach` | **attach**: build at width/depth (defaults when 0), zero-init `out`, `gate=1` |
| none | `detach` | no refiner, info notice |
| attached | `inherit` / `attach` | **continue**. A non-zero width/depth that differs from ckpt is refused (shape change is not a resume); the mode table below decides trainability |
| attached | `detach` + `hard` | **hard detach**: branch dropped before the first step. A warning gives the last logged `sn_refiner_delta_rel` |
| attached | `detach` + `anneal` | **anneal**: refiner frozen (`requires_grad=False`), `gate` ramps 1 -> 0 linearly over `detach_steps` updates from the resumed step. Rest of the model keeps training |
| annealing | `inherit` / `detach` | **continue anneal** from the recorded anchor. The gate is a function of (anchor, steps, current step), not of the saved gate |
| annealing | `attach` | refused in v1 (re-attach mid-anneal) |
| anneal complete (`gate == 0`) | any but `attach` | branch dropped at load. The next save omits it, which is lossless because `gate=0` already contributed nothing |
| any refiner state | non-full-FT method, `inherit` | **frozen forward**: module built and loaded under the strict check, `requires_grad=False`, in no optimizer group; the saved gate is used as is (no anneal clock runs, an annealing checkpoint keeps its saved gate for the whole run) |

For full fine-tuning, after the state row is resolved:

| resolved state | requested mode | Base | Refiner | Contract |
|---|---|---|---|---|
| newly attached | `joint` | train | train | default; zero-init identity and fresh-group warmup apply |
| newly attached | `refiner_only` | frozen | train | cheapest capacity test; base forward is under `no_grad` and `x0_head`/`z_grid` are detached |
| newly attached | `base_only` | - | - | refused: a frozen zero-init branch is exactly inert |
| continued attached | `joint` | train | train | normal joint continuation |
| continued attached | `refiner_only` | frozen | train | trains/retrains only the branch on the distributed base |
| continued attached | `base_only` | train | frozen | supported for downstream base adaptation; refiner forward stays in autograd so its input gradient reaches the base |
| detach (`anneal` or `hard`) | any | train | frozen/removed | mode is not consulted; detach is intrinsically base-only so the base can absorb the branch |

Here “base” means the generation-side scopes that the full-parameter adapter
would otherwise train, including the pixel head; it does not override the
existing `train_unet`/MoT-half scope contract. `refiner_only` does not
materialize or optimize decoder Linears merely because the API method is
`full_finetune`. `base_only` freezes refiner parameters but must not wrap its
forward in `no_grad`: doing so would cut the only gradient from the refined
output back into `x0_head`. The expected use is a small downstream base
adaptation followed, if necessary, by a short `joint` finish; it is supported
for model recipients but is not the recommended first training mode. A new run
pointed at a self-contained bf16 refiner distribution retains the frozen MoT
half in bf16 and requires bf16 saves; an in-place resume with the original int8
base available may restore that half to the lower-memory int8 layout.
`refiner_only` also refuses `repa_enable` and any explicit trainable scope
other than `generation_refiner`; otherwise its name would falsely imply that
only the refiner changes.

**Anneal clock.** The same axis the existing warmup uses. The anchor is the
resumed scheduler position (`resume_scheduler_position`, `base_trainer.py:7772`;
axis conversion `:380-390`) and the clock is `scheduler.last_epoch`
(`fresh_param_warmup.py:28`). `gate = clamp(1 - (last_epoch - anchor) / steps, 0, 1)`
is written to the buffer on update boundaries, before the forward of the next
micro-batch. The declaration persists `detach_anchor_step` (scheduler axis),
`detach_steps` and `detach_accum` (the accumulation interval the anchor was
taken on). On this route `gradient_accumulation_steps` is fixed at 1
(`train_runner.py:562-574`; served as a required value,
`arch_capabilities.py:1233-1236`), so `detach_accum` is 1 today; it is
persisted so a later relaxation of that rule cannot silently rescale the
anchor. A resume whose interval differs from `detach_accum` is refused.

**Anneal completion inside a segment.** At the update boundary where the gate
reaches 0, the module is deleted from `fm_modules` and the declaration is
deleted, in that order, before the next forward. This is safe mid-process
because under anneal the refiner is in no optimizer group and carries no fused
hook (`fused_backward_registration.py:98-104` skips frozen params), is not
enumerated by block swap or the phase evictor (see
[Enumerations](#enumerations-that-walk-the-model)), and the forward tests
`"fm_refiner" in fm_modules`. The forward before and after the drop is
value-equal: `x0_head + 0.0 * delta == x0_head` for finite `delta`, and
`delta` is finite whenever the load-time checks pass (non-finite tensors are
refused at load). The save that follows carries no refiner tensor and no
declaration, and no refiner optimizer state (it was never in this process's
optimizer).

**Declaration upkeep.** `trainer.sensenova_config_dict` is mutated at each
transition, the way `apply_noise_scale_gain` does (`arch/sensenova.py:270,
327-333`): `gen_refiner` is added on attach, gets the anchor on anneal start,
and is **deleted** at hard detach and at anneal completion. The save embeds
that dict verbatim (`sensenova_adapter.py:819-829` passes it as `raw_config`;
`loader.py:722-756` re-embeds it). Otherwise the strict check below would refuse
the next load.

Hard detach is not lossless. The rest of the model co-adapted with the branch.
The anneal exists so the model re-absorbs the branch's contribution before it
is removed.

Pixel-space geometry with `attach` is refused, with the reason given.

### Owning site and ordering

The trainability of the refiner is decided at one site and read by everything
after it. In load order:

1. `sensenova_ops.load_components` (`sensenova_ops.py:1191`) loads the tree
   (`:1210`), then `apply_latent_space` (`:1256`; `vae_swap.py:128-154`) runs
   the arch handler's `apply_vae_swap` for a declared VAE. **New:**
   `SenseNovaArchHandler.resolve_latent_refiner(trainer)` runs immediately
   after, on the CPU tree, before the freeze at `:1263`. It validates the
   declaration (see [Persistence](#persistence)), applies the state table,
   builds or drops `fm_modules["fm_refiner"]`, mutates the declaration, and
   records `trainer.sensenova_refiner_state ∈ {"none", "attached", "anneal", "frozen"}`
   and `trainer.sensenova_refiner_training_mode ∈ {"none", "joint",
   "refiner_only", "base_only"}`. `"frozen"` is the non-full-FT `inherit`
   row; detach records `"base_only"`. The freeze at `:1263` then applies to
   the refiner like everything else.
2. `SenseNovaFullParameterAdapter.prepare_models_for_training`
   (`sensenova_adapter.py:528`): `transformer.requires_grad_(False)` (`:554`),
   then `_scope_parameters` (`:555`). **New:** `_scope_parameters` adds a
   `generation_refiner` scope holding `fm_modules.fm_refiner.parameters()`
   **iff** the effective mode is `joint` or `refiner_only`, on both the
   default-scope path (`:405-428`) and the explicit-scope path (`:430-470`).
   Under `base_only`, `anneal` and `frozen` the scope is absent, so the freeze
   stands. In `refiner_only`, every other generation scope is omitted and the
   decoder stays in its loaded representation; this is a dedicated scope
   route, not full-parameter materialization followed by freezing. The
   unfreeze loop (`:556-558`) is what sets `requires_grad=True`; nothing else
   does.
   `_fm_parameters` (`:340-395`) and the explicit `generation_flow` scope
   (`:461-464`) both enumerate `fm_modules.parameters()` and must **exclude**
   `fm_refiner`, or the refiner would be double-collected into
   `generation_decoder` / `generation_flow`. `trainer._sensenova_scope_parameter_ids`
   (`:560-563`) then carries the new scope key.
3. `arch_param_groups` (`:602-662`): a new spec
   `("generation_refiner", unet_lr * sensenova_refiner_lr_factor, "generation_refiner")`
   **appended last in both `group_specs` tuples** (`:630-649`). The
   `if p.requires_grad` filter (`:652`) admits the scope only in `joint` or
   `refiner_only`. Every existing arch group keeps its index in `joint`; in
   `refiner_only`, `generation_refiner` is the sole arch group. The REPA projector group
   is appended after the arch groups (`base_adapter.py:630-631`), so on a REPA
   run its index moves by one when the refiner attaches; that is why the named
   optimizer load (item 6) is required rather than the positional one.
   `apply_layer_lr_decay` (`base_trainer.py:472-514`, called at `:8248`) may
   further split groups by depth; the refiner is in no depth block
   (`depth_blocks` returns the decoder layers, `arch/sensenova.py:441-445`)
   and keeps its group's rate. The `component` key is required: a group
   without one disables `lr_group_schedules` for the whole run
   (`base_trainer.py:561-574`).
4. `setup_optimizer` builds the optimizer from those groups and, on this route,
   registers the fused hooks (`base_trainer.py:8488-8497`) against
   `_fused_backward_target_module()` = `transformer` (`:8610`). Registration
   refuses a trainable transformer parameter in no group
   (`fused_backward_registration.py:84-94`) and skips frozen ones (`:98-104`),
   so `joint`, `refiner_only`, `base_only` and `anneal` all pass by
   construction. No `add_param_group`
   is needed because attach and continue both happen at process start.
5. `_record_configured_group_lrs` (`base_trainer.py:7026`) snapshots each
   group's base LR off the adapter's groups; that snapshot, not
   `_build_component_lr_list` (`:6936-6947`, which lists only the two MoT
   groups and is used only when its length matches), is what the resume
   re-asserts. The refiner group's LR therefore survives a resume.
6. `train()`: scheduler fast-forward (`:15818`), `load_optimizer_state`
   (`:15826`), then `_rearm_warmup_after_optimizer_reset` (`:15831-15833`).
   The named load (`:5990-6064`) marks refiner tensors fresh (`:6026-6029`),
   sets `partially_fresh` (`:6048-6050`), and passes only matched state
   (`:6056-6061`). Names come from `_build_ema_param_name_map`
   (`:9068-9091`, used for `_sushi_param_names` at `:5566-5579`), which walks
   every `nn.Module` attribute of the trainer with `named_parameters()`; the
   refiner is covered as `transformer.fm_modules.fm_refiner.*` without any
   change. Had it not been, a single unnamed parameter would drop
   `_sushi_param_names` for the whole optimizer (`:5578`) and force the
   positional path; the P2 test asserts the key is present after attach.

The P2 test for this ordering asserts, on a CPU trainer through the real
`prepare_models_for_training` -> `arch_param_groups` -> fused registration
sequence: every tensor under `fm_modules.fm_refiner.` with `requires_grad`
True (`joint`, `refiner_only`) or False (`base_only`, anneal, frozen); each
trainable refiner parameter id present in exactly one optimizer group, exactly
once, and that group named `generation_refiner`; no refiner id in any group in
the frozen modes; no base id in any group in `refiner_only`; `gate` in no
group; after hard detach or completion, no module under that name.

## Persistence

### Tensors

Under `fm_modules.fm_refiner.*`. The save iterates `transformer.state_dict()`
(`core/models/sensenova/loader.py:961`), which includes persistent buffers, so
`gate` is written. Every save format (`mixed` / `bf16` / `int8`) writes
non-decoder-Linear tensors as-is under the `other` census bucket
(`loader.py:1014-1022`), which the count check does not constrain
(`:1028-1037`), so no format-specific code is needed. `accept_resume_shaped_base`
(`sensenova_ops.py:735`) counts only decoder Linears. The bf16 single-half
resume restores only `iter_sensenova_lora_targets(branch=frozen_half)` from the
base (`loader.py:600-602`), so refiner tensors always come from the checkpoint.

`refiner_only` requires requested
`sensenova_full_finetune_save_format="mixed"`. The loaded decoder halves remain
in their original classes and are emitted unchanged; the refiner is added under
`other`. An int8/mixed source therefore stays mixed. A source such as run127
whose two halves are already floating point is preserved as bf16: the metadata
records requested `mixed`, effective `bf16`, the source base-training lineage
(`gen` for run127), save layout `both`, and mode `refiner_only`. This distinction
prevents a refiner-only save from falsely claiming that it trained both MoT
halves. `bf16` and `int8` are refused as *requested* refiner-only formats because
either would otherwise imply a base conversion that the run did not train.
Round-trip tests compare every non-refiner tensor to the input checkpoint and
require exact dtype, shape and value equality.

The result is a complete checkpoint, not a sidecar. It can always run inference
or continue `refiner_only` without the original base. It can also seed a new
`base_only` or `joint` full fine-tune without the original int8 base only when
all of the following fail-closed stamps agree: mode `refiner_only`, effective
format `bf16`, save layout `both`, requested branch equal to the inherited
base lineage, and a validated refiner declaration/tensor set. In that portable
route the frozen MoT half remains bf16, so resident VRAM is higher and every
subsequent full-parameter save must request `bf16`. Ordinary bf16 checkpoints
remain refused as new training bases. When the original int8 base is available,
the existing resume path may instead restore the frozen half to int8 for the
lower-memory layout.

### Declaration

The branch is **declared**, not inferred from key names. Inside `sensenova_config`
(the metadata block `latent_config_dict` writes,
`core/models/sensenova/latent_space.py:300-316`):

```json
"gen_refiner": {
  "version": 1,
  "width": 128,
  "depth": 3,
  "inputs": ["x0_head", "z_cin"],
  "norm": "channel_rms_v1",
  "detach_anchor_step": null,
  "detach_steps": null,
  "detach_accum": null
}
```

`NEOChatModel.__init__` builds `fm_modules["fm_refiner"]` only when
`config.gen_refiner` is present. When it is absent, the model is exactly
today's.

### Declaration schema, validated before model construction

`validate_gen_refiner_declaration(cfg_dict, state_dict_keys)` runs in
`load_sensenova_from_path` after `_load_sensenova_config` (`loader.py:1199`)
and before `NEOChatModel(config)` (`:1221-1222`), on the raw dict and the
still-plain state dict. It is fail-closed: any clause below raises, with the
clause named. The same function is the only validator, so the verdict is the
same at every entry point (next subsection).

| Clause | Rule |
|---|---|
| version | `version` present, integer, in the supported set `{1}`. Unknown -> refuse (no forward-compat guess) |
| inputs | exactly `["x0_head", "z_cin"]`, order included |
| norm | exactly `"channel_rms_v1"` |
| width | integer, `16 <= width <= 1024`, divisible by 16 |
| depth | integer, `1 <= depth <= 8` |
| detach fields | `detach_anchor_step`, `detach_steps`, `detach_accum` all null (attached) or all present (annealing): integers, `anchor >= 0`, `steps >= 1`, `accum >= 1`. Mixed presence -> refuse |
| tensor set | the set of `fm_modules.fm_refiner.*` keys equals the set the module class enumerates for (width, depth), no extra, no missing |
| tensor shapes | every tensor's shape equals the shape the class gives it at (width, depth, C, p); C and p come from the same config block (`gen_in_channels`, `gen_patch_size`) |
| tensor values | every `fm_modules.fm_refiner.*` tensor finite (a non-finite `delta` would make the gate-0 equivalence above false) |
| gate | key present, 0-dim, float32, finite, `0.0 <= gate <= 1.0` |
| gate vs clock | annealing declaration only: see precedence below |
| undeclared keys | any `fm_modules.fm_refiner.*` key with no declaration -> refuse |

**Gate precedence.** During training the buffer is derived from the clock
(state table), never read back. At a load the two must agree: the loader
recomputes `gate_expected = clamp(1 - (step_sched - anchor) / steps, 0, 1)`
with `step_sched` the checkpoint's own step converted with `detach_accum`
(the metadata `step`, `sensenova_adapter.py:781`). Because the buffer is
written at every update boundary before the forward and a save happens at a
step boundary, the two are equal up to fp32 rounding; the tolerance is
`1e-6` absolute, and a larger difference is refused (it means the step and
the declaration disagree, which no supported path produces). For generation
and for the non-full-FT frozen forward the saved gate is used after this
check; no clock runs there.

### Same verdict at every entry point

There is one loader (`load_sensenova_from_path`) and it is what all four
entries call: generation load, full-FT resume (`sensenova_ops.py:1210`), LoRA
training on a refined base (same `load_components`, `training_method` branch
at `:1206-1208` only decides materialization), and LoRA inference on a refined
base (generation loads through `core/model_loader.py:3144-3154`, a
pass-through to the same function; LoRA application then targets decoder
Linears only,
`sensenova_lora.py:278-308`, `pipeline_backends/sensenova.py:228-245`). A
refusal in the validator is therefore identical on all four.

### Strictness

`install_sensenova_state_dict` loads with `strict=False` and only prints
missing/unexpected keys (`loader.py:1110-1114`). A refiner whose tensors are
dropped would silently run as zero-init. `init_empty_weights()` resolves
`include_buffers=None` to the env flag `ACCELERATE_INIT_INCLUDE_BUFFERS`,
default False (accelerate 1.12.0, `big_modeling.py:61, 91-92`), so a missing
`gate` would not crash either. It would be materialized at 1.0 and cast to
the load dtype by `model.to(torch_dtype)` (`loader.py:1223`), which is
silently wrong for an annealing checkpoint. The schema check above runs
before construction and covers both directions for the
`fm_modules.fm_refiner.` prefix:

- declared and any refiner key missing -> refuse the load;
- refiner keys present and not declared -> refuse the load.

After the load, `gate` is re-asserted float32 (the `model.to(torch_dtype)`
above would otherwise leave a bf16 buffer to be overwritten by the fp32 tensor
under `assign=True`; the round-trip test checks the dtype).

This does not change the existing tolerance for other keys.

## Optimizer and warmup

- New param group `generation_refiner`, `lr = unet_lr * sensenova_refiner_lr_factor`,
  **appended last** in both group orders in
  `SenseNovaFullParameterAdapter.arch_param_groups`
  (`core/training/adapters/sensenova_adapter.py:630-649`). Every existing
  arch group keeps its index (see [Owning site and ordering](#owning-site-and-ordering)
  for the REPA projector group and `lr_layer_decay` splits).
- `_fm_parameters` must **exclude** `fm_refiner`. Today, with
  `sensenova_train_fm_modules`, every `fm_modules` param goes into
  `generation_decoder` (`sensenova_adapter.py:416-420`). The refiner's
  trainability is independent of `sensenova_train_fm_modules`: that flag is a
  required value only while `vae_swap_source` is set
  (`arch_capabilities.py:1252-1256`), and a resume on an already-swapped base
  can run with it off.
- `train_step`'s `fm_trainable` (`sensenova_ops.py:2290-2295`) asks
  `any(p.requires_grad for p in fm_modules.parameters())` to decide whether
  `_build_step_context` runs with grad. It must enumerate `fm_modules` minus
  `fm_refiner`; otherwise `refiner_only` would build a graph through the
  generation ViT for nothing. In that mode the complete base prediction is
  computed under `no_grad`, then `x0_head` and `z_grid` are detached before
  the refiner. In `base_only`, the refiner parameters are frozen but its
  operations remain in the graph so gradients reach `x0_head`. `joint` uses
  the normal graph for both.
- The accepted optimizer names remain the existing SenseNova full-FT set.
  `refiner_only` is exempt from the host-resident-state requirement for the
  ring-buffer optimizers because its sole optimizer group is ~1.1M parameters;
  all other full-FT preflight rules remain. This exception is keyed to the
  effective mode, not inferred from the current number of trainable tensors.
- Attach and continue both happen at process start, so any refiner group exists when
  the optimizer is constructed and when fused hooks are registered
  (`setup_optimizer`, `base_trainer.py:8497`). No `add_param_group` is needed.
  This is what makes the fused path work: registration refuses a trainable
  param with no group (`optimizers/fused_backward_registration.py:84-94`), and a
  later-added param would never get a hook.
- **Fresh detection works as is.** The named optimizer load marks tensors absent
  from the saved state as fresh and sets `partially_fresh`
  (`base_trainer.py:6026-6050`). run127's `step_093791_optimizer.pt` carries
  `_sushi_param_names` (audited from the pickle header), so the named path is
  taken. Without names, the legacy prefix remap also marks a trailing group
  fresh (`base_trainer.py:5975-5987`).
- **The re-arm is gated.** `_rearm_warmup_after_optimizer_reset` returns
  without arming when `rewarmup_on_optimizer_reset` is off
  (`base_trainer.py:7746-7753`; default True, `param_defaults.py:2325`) or when
  `lr_warmup_steps` converts to 0 scheduler updates (`:7755-7768`;
  `param_defaults.py:2281` default 0). An attach on a run with `lr_warmup_steps=0`
  therefore ramps nothing, and the zero-init `out` moves at full Lion LR on its
  first update. The runner refuses `attach` when the effective warmup is 0
  updates or the re-arm flag is off, with the reason, rather than warn.
- **Warmup on attach on the fused path: one defect, fixed in P0 (558ebf7d).**
  Fused-backward hooks held the param-group dict captured at registration,
  while a resume (`load_state_dict`, the trainer's direct or named load)
  replaces `param_groups` with new dicts. The scheduler then wrote the live
  dicts and the kernel read the orphaned one, so every fused resume applied the
  LR frozen at the resume position. A wholly fresh appended group such as
  `generation_refiner` gets its warmup from a composed `scheduler.lr_lambdas[i]`
  (`_rearm_warmup_after_optimizer_reset`), which acts through the live group
  LR; it failed to reach the kernel only as a consequence of the stale dict,
  not as a separate defect. Fix: hooks hold no group and resolve the live
  group on every call via `live_param_group(optimizer, param) -> (group,
  gindex, pindex)` (`optimizers/live_param_group.py`; an id->position cache
  accepted only if `optimizer.param_groups[gindex]["params"][pindex] is param`,
  rebuilt otherwise). Applied in `lion8bit_ringbuffer.py`,
  `adamw8bit_ringbuffer.py`, the generic hook in `base_trainer.py`
  `_setup_fused_backward_pass` (Adafactor / bitsandbytes AdamW8bit) and
  `adamw8bit_fused._param_index`. `fused_optimizer_groups.py`
  (`num_optimizer_groups > 0`) was not affected: its hook calls `step()` on the
  sub-optimizer, which reads live groups.
- Tests: `backend/tests/fused_hook_live_param_group_test.py` (19 tests, CPU,
  recording kernel stand-in; all 19 fail on the pre-fix code, pass after). They
  cover `optimizer.load_state_dict`, trainer direct load, named load, prefix
  remap, a `param_groups` split, and a wholly fresh appended group on resume:
  through `_rearm_warmup_after_optimizer_reset` + `reassert_config_lr` the
  fresh group's kernel LR is 0 at the anchor, 0.25x at +25 steps and the full
  live scheduled LR from +100 (warmup 100), while the restored group gets the
  live scheduled LR throughout. Verification level: CPU stand-in; real CUDA
  kernels not exercised.
- Arming the per-tensor cohort (`arm_fresh_param_warmup`) for whole-group fresh
  ids is not needed: the test shows the composed-lambda warmup reaches the
  kernel. The cohort stays limited to `mixed_ids`.
- Attach in `joint` or `refiner_only` therefore ramps the refiner 0 -> 1 over `lr_warmup_steps`
  (trainer attribute `optimizer_warmup_steps`, `base_trainer.py:2820`,
  `:7755`) from the resumed scheduler position. A new run that attaches at
  step 0 is covered by the global warmup.
- The refiner group's warmup is its own composed lambda, per group, with length
  `lr_warmup_steps`. The one-per-optimizer cohort (`fresh_param_warmup.py:14`)
  applies only to fresh tensors mixed into restored groups. A separate refiner
  warmup length would need a per-group length option; it is not proposed until
  a measurement shows the shared length is wrong.
- Annealed detach: the refiner is frozen, so it is in no group. The named load
  passes only matched state (`base_trainer.py:6056`), so the saved refiner
  state is dropped at that resume and is not in the next save. Fused
  registration skips frozen params (`fused_backward_registration.py:98-104`).
- Hard detach and anneal completion: tensors are gone. Same as above. The P2
  test asserts that the first optimizer save after each of hard detach,
  anneal start and anneal completion has no `fm_refiner` name in
  `_sushi_param_names` and no state entry for a refiner tensor.

The expected cost distinction is consequently simple. `joint` adds the
refiner arithmetic and the checkpointed activation delta in the table above,
but no second base forward; optimizer-state growth is only the refiner's
~1.1M parameters. `refiner_only` retains the same forward cost but removes
base backward and base optimizer state, and is the preferred first experiment
for deciding whether the branch has useful capacity. `base_only` removes only
the refiner's small optimizer state relative to `joint`; its backward must
still traverse the fixed refiner, so its activation cost is close to `joint`.
Joint training is the release-quality finish only after a refiner-only
capacity gate is positive, not the mandatory first run.

### Enumerations that walk the model

Placing the refiner under `fm_modules` puts it in reach of every walk of that
container. Each one, with its verdict:

| Walk | Site | Refiner | Why |
|---|---|---|---|
| Block swap | `sensenova_ops.setup_block_swap`, `layers = language_model.model.layers` (`sensenova_ops.py:1153`, `:1170`) | excluded | walks decoder layers only |
| MoT phase evictor | `mot_phase_eviction.py:76-79` via `select_mot_weight_modules`, `layers = language_model.model.layers` (`mot_weight_selector.py:121-133`) | excluded | decoder layers only; the refiner is always resident, like `fm_head` |
| Update census expectation | `trainable_params_of(optimizer)` (`update_census.py:187-201`) | included iff in a group | param_groups-driven; frozen (anneal) -> absent, correct |
| Update census active set | image-batch scope tuple `base_trainer.py:18693-18700` | **must be added** | the tuple names scopes; a `generation_refiner` scope not listed there is silently never checked on image batches. Added to that tuple, not to the text-batch tuple (`:18682-18687`) |
| Grad-norm components | `grad_norm_components` (`sensenova_adapter.py:664-718`): explicit map `:684-692`, default path `:697-712` (`_fm_parameters` -> `LORA_COMPONENT_UNET`) | **must be added** | with `fm_refiner` excluded from `_fm_parameters`, the default path would drop it to the module-derived bucket (`base_trainer.py:20046-20064`). Both paths map the scope to `LORA_COMPONENT_UNET` |
| Save census | `save_sensenova_full_finetune_checkpoint` (`loader.py:961-1022`) | included, bucket `other` | written as-is in every format; the count check covers decoder Linears only (`:1028-1037`) |
| Quantized export | same writer, `effective == "int8"` branch (`:971-979`) | excluded from quantization | only `stem in trained` (decoder Linears) is quantized; refiner convs stay in their dtype |
| LoRA inject / apply / restore | `iter_sensenova_lora_targets` (`sensenova_lora.py:278-308`) | excluded | the only target enumerator walks decoder layers |
| bf16 single-half resume | `loader.py:600-602` | excluded | restores frozen-half Linears from the base; refiner comes from the checkpoint |
| EMA / optimizer name map | `_build_ema_param_name_map` (`base_trainer.py:9068-9091`) | included | generic `named_parameters()` walk; required for `_sushi_param_names` (`:5574-5579`). `use_ema` itself is refused on this route (`train_runner.py:584-589`) |
| REPA tap | `forward_gen_decoder_layers` (`sensenova_ops.py:2944-2947`) | excluded | the tap is a decoder layer's hidden state, upstream of the head |
| `lr_layer_decay` | `apply_layer_lr_decay` (`base_trainer.py:472-514`), depth from `depth_blocks` (`arch/sensenova.py:441-445`) | excluded from decay | not in a decoder layer; keeps its group's rate |
| Four-phase / text-batch census | `:18681-18691` | excluded | understanding scopes only, correct |
| `_fm_parameters` dtype check | `sensenova_adapter.py:385-394` | included | floating-point convs, passes |

## Generation

The checkpoint is the only input, so there is no generation parameter:

1. `load_sensenova_from_path` builds the config from `sensenova_config`
   (`loader.py:161-186, 1199`), validates the declaration, and `NEOChatModel`
   builds the refiner from `gen_refiner` with its width/depth.
2. The schema check (above) guarantees trained weights and the saved `gate`.
3. `_t2i_predict_v` calls `apply_latent_refiner` after the head, before the
   `v` conversion. Every SushiUI denoise path reaches it through one
   function, `_predict_v_branch` (`sensenova_pipeline_ops.py:734-739`), which
   `_euler_run` calls per CFG branch (cond `:1155`, uncond `:1088`,
   img_cond `:1080`), so cond, img_cond (it2i reference editing) and uncond
   each get the refiner, exactly as training's single branch does. txt2img,
   img2img and inpaint all run `_euler_run` (`:1244`, `:1302`, `:1369`); spatial
   outpaint is not implemented for SenseNova (`pipeline_backends/sensenova.py:45`).
   The training-time sample drives the same loops
   (`sensenova_ops.generate_sample`, `:2597-2634`; `arch/sensenova.py:565-583`),
   so a sample made mid-anneal renders with the gate the step before it wrote.
4. Reference-style capture runs a cond forward whose output is discarded
   (`_style_capture`, `:903-927`; `_style_capture_multi`, `:930`). It goes
   through `_predict_v_branch` and therefore through the refiner; that is a
   cost, not a correctness question, since the refiner's output is not used
   by the capture. Not special-cased.
5. A LoRA loaded onto a refined base changes decoder Linears only
   (targets come from `iter_sensenova_lora_targets`, applied through the
   SenseNova backend's LoRA group, `pipeline_backends/sensenova.py:228-245`).
   The refiner runs as part of the base.

`noise_scale` must reach `_t2i_predict_v`, which does not take it today.
`_euler_run` already holds it (parameter at `:1244`, `:1302`, `:1369`) and
`_predict_v_branch` is its only caller of `_t2i_predict_v`; P1 adds a
`noise_scale` keyword to both and threads the existing value. The style
capture helpers hold it too (`:905`, `:932`). Training has it at
`sensenova_ops.py:2268`. No second computation is added.

A generation-side bypass toggle for A/B is deliberately not proposed. It would
be a way to run a checkpoint differently from how it was trained.

## Call sites

Every site that runs the head must go through `apply_latent_refiner`, or
refuse a refined model:

| Site | Action |
|---|---|
| `vendor/modeling_neo_chat.py:687` (`_t2i_predict_v`, pixel-head branch) | call |
| `core/training/ops/sensenova_ops.py:2339` (`train_step` re-implementation) | call |
| `sensenova_pipeline_ops.py:734-739` (`_predict_v_branch`, goes through `_t2i_predict_v`) | covered by the vendor call; add `noise_scale` |
| `probes/sensenova_cfg_null_parity.py:420-432` | call |
| `probes/sensenova_und_discrimination.py:273`, `probes/sensenova_real_checkpoint.py:270`, `probes/text_encode_vs_step.py:801,989`, `probes/sensenova_und_prefix.py` | refuse when `gen_refiner` is declared (several already assume pixel geometry) |

`train_step` and `_t2i_predict_v` share the function, not a copy. P1's parity
test runs both on one tiny model and requires identical output.

## Observability

- `sn_refiner_delta_rel = ||gate * delta|| / ||x0_head||`, per step, through
  `defer_extra_metric` (`base_trainer.py:20236`) plus an `EXTRA_METRIC_DEFS`
  entry (`metric_registry.py:74`; no DB column). It is what hard detach quotes,
  and it shows whether the branch does anything.
- `sn_refiner_gate`, logged only while annealing.
- A load-time log line and a `training_events` notice for each state
  transition (attach / continue / anneal start / anneal complete / hard detach
  / frozen forward).

## Plumbing checklist

Per the existing SenseNova option path (template: `sensenova_gen_patch`):

1. `backend/api/param_defaults.py` `TRAINING_DEFAULTS`: the 7 keys.
2. `openapi.yaml` `TrainingRunCreateRequest`: fields, descriptions, examples.
3. `backend/api/routes.py` `TrainingRunCreateRequest` plus request checks next to
   `_check_vae_swap_params` (`routes.py:15319`), on POST (`:15390`) and PUT
   (`:16101`). Use `model_fields_set` where "explicitly set" matters.
4. `backend/core/training/training_config.py` `_build_train_section`: keys in the
   dict literal. `train_section_key_vocabulary()` reads the literal's syntax tree
   (`training_config.py:730-751`).
5. `backend/core/training/train_runner.py` `_apply_sensenova_full_finetune_contract`
   (`:543`) and the method-independent path: normalization; refusals for
   `attach`/`detach` under non-full-FT, pixel geometry, width/depth change,
   illegal state/mode pairs, understanding-only run, REPA/extra trainable
   scopes or non-`mixed` save format in `refiner_only`, zero effective warmup
   or `rewarmup_on_optimizer_reset` off with a trainable fresh refiner.
6. `backend/core/training/arch/sensenova.py`: `resolve_latent_refiner` (state
   resolution, build, declaration mutation), anneal clock, per-channel latent
   statistics guard in `calibrate_before_training`.
7. `backend/core/training/adapters/sensenova_adapter.py`: `_fm_parameters` and
   `generation_flow` exclusion, mode-specific base/refiner scopes,
   `generation_refiner` group, `grad_norm_components` mapping.
8. `backend/core/training/base_trainer.py:18693-18700`: `generation_refiner` in
   the image-batch census scope tuple.
9. `backend/core/training/ops/sensenova_ops.py`: `train_step` call with `[B]` `t`
   and `checkpoint_blocks`; `fm_trainable` excludes `fm_refiner`.
10. `backend/core/models/sensenova/`: new `latent_refiner.py` (module,
    `apply_latent_refiner`, `validate_gen_refiner_declaration`),
    `latent_space.latent_config_dict` (declaration), `loader.py` (validator
    call before construction, gate dtype re-assert), `vendor/modeling_neo_chat.py`
    (build and call, `noise_scale` keyword), `sensenova_pipeline_ops.py`
    (`noise_scale` through `_predict_v_branch`).
11. `backend/api/arch_capabilities.py`: feature entry, unsupported for
    non-SenseNova archs; required value `inherit` for `lora`/`relora`/`controlnet`
    on SenseNova.
12. `frontend/src/utils/api.ts`, `trainingConfigDefinitions.tsx` `DEFAULT_PARAMS`,
    `trainingParams.ts` `PARAM_KEYS`, `TrainingConfig.tsx` controls (state select,
    width, depth, training mode, lr factor, detach mode/steps) with `unsupportedTrainingFeature`
    gating and the required-value pin.
13. `backend/core/training/metric_registry.py`: the two series.
14. Probes: call or refuse, per the table; commit the spectrum/grid probe.
15. `docs/guides/SENSENOVA_TRAINING_DESIGN.md` / `MODEL_FACTS.md` pointers once implemented.

## Phases

| Phase | Content | Gate |
|---|---|---|
| P0 (done, 558ebf7d) | Fused-hook stale group dict fix (all capturing optimizers; hooks resolve the live group per call) | CPU test (`fused_hook_live_param_group_test.py`): after `load_state_dict`, an LR written to the live group is the LR the kernel receives; a wholly fresh appended group gets LR 0 at the anchor via the composed lambda. Independent of the refiner, committed on its own |
| P1 (done, `fe37f4bb`) | `latent_refiner.py` with ChannelRMSNorm2d, vendor build+call, `train_step` call, `noise_scale` threading, declaration, schema validator | CPU identity, parity, strictness, locality and checkpoint-gradient tests pass |
| P2 (done, `73541704` plus persistence fixes) | State/mode tables, mode-specific scope and save behavior, optimizer group, warmup path, anneal, census/grad-norm wiring, latent statistics guard, metrics | Parameter-boundary, save/reload, lineage and portable-base tests pass |
| P3 (done, `a792a624`) | Params, API, YAML, runner refusals, capabilities, UI, quality and VRAM probes | `py_compile`, CUDA-stubbed real imports and API defaults verified |
| P4 (engineering gate complete for the selected route) | Isolated GPU memory matrix plus real run128 attach/continue/save | 1536/2048/4096 isolated checkpointed deltas measured; 2048 real `refiner_only` completed five finite steps and two resumes. Real joint/base-only/anneal/hard and full 4096 runs remain optional integration coverage, not evidence for quality |

The phase boundaries describe reviewable contracts; follow-up commits fix facts
found by the real run and are listed in the implementation record above.

## Verification plan

CPU, tiny `NEOChatModel` config, in the production dtype matrix (bf16, fp16,
fp32):

1. **Zero-init identity:** attach -> `x0` equals the refiner-less forward exactly.
2. **Train/infer parity:** `train_step`'s head+refiner output equals
   `_t2i_predict_v`'s on the same inputs, both with the caller inside
   `torch.autocast("cpu", dtype=bf16)` and without. Autocast is disabled
   off-CUDA at `sensenova_ops.py:2327`, so the test enables it explicitly.
   Cases: batch 1 with a 0-dim `t`; **batch 3 with three distinct `t` values**
   on the train side against three single-item inference calls; the packed
   train path (`prefix.packed` set) against the unpacked one; `c_in` and the
   FiLM tensors asserted `[B,1,1,1]` and `[B,width,1,1]`; the RMS reduction is
   asserted fp32 and its output in the parameter dtype.
3. **Round-trip:** save (each format) -> `load_sensenova_from_path` -> same
   output, `gate` preserved, 0-dim and float32 after load.
4. **Schema and strictness, each a separate refusal:** declared-but-missing
   tensor; present-but-undeclared tensor; tensor shape disagreeing with the
   declared width or depth; gate missing / non-scalar / NaN / `-0.1` / `1.5`;
   unknown `version`; `inputs` differing in content or order; missing/unknown
   `norm`; width not divisible by 16; annealing
   declaration with `detach_anchor_step` or `detach_steps` or `detach_accum`
   missing; saved gate differing from the recomputed one by more than `1e-6`.
   Every case asserted through the one loader entry with the four callers'
   arguments (generation, full-FT resume, LoRA training, LoRA inference) and
   the same exception text.
5. **State/mode tables:** every row and cross-product, including the
   default-edit case (`inherit` on a refined checkpoint keeps it), attach plus
   `base_only` refusal, detach's intrinsic base-only behavior, and the
   non-full-FT frozen forward row.
6. **Owning site / group membership:** the assertions listed under
   [Owning site and ordering](#owning-site-and-ordering); plus
   `_sushi_param_names` present after attach; plus `generation_refiner` in
   the census active set on an image batch and absent on a text batch.
7. **Mode gradients and persistence:** `joint` changes both parameter sets;
   `refiner_only` creates no base grads/group/state and an exact comparison
   shows every non-refiner tensor unchanged after a `mixed` save/reload;
   `base_only` creates no refiner parameter grad/state but a loss on refined
   output produces a non-zero base-head gradient through the fixed branch.
8. **Warmup:** resume of a refiner-less checkpoint with `attach` on the fused
   Lion RB path marks exactly the refiner tensors fresh and composes the warmup
   lambda for the refiner group (whole-group case). The LR the hook hands the kernel is 0 at the anchor
   and the live group LR after the warmup length. Restored tensors
   receive the live scheduled LR throughout (P0 regression). Refusal when the
   effective warmup is 0 or the re-arm flag is off.
9. **Anneal:** the gate follows (anchor, steps) across a simulated re-resume
   mid-anneal; at completion the module and the declaration are gone in-process,
   the next save has no refiner tensor, no declaration, and no refiner optimizer
   state; the output before and after the drop is equal. Same no-stale-state
   assertion for the first save after hard detach and after anneal start.
10. **Checkpointing:** gradients with `checkpoint_blocks=True` equal those
   without, per dtype.
11. **Locality:** perturbing one input latent cell cannot change delta values
    outside radius `2 * depth + 2`. A full-grid crop equals evaluation of the
    same crop with exactly that halo (including outer-boundary zero padding),
    within the dtype-specific convolution tolerance. This prevents a future
    spatial normalization from silently breaking the tiling premise.
12. **Latent statistics guard:** a synthetic latent stream with one channel at
    RMS 3.0 refuses `attach` with the measured value in the message.
13. **Refusals:** pixel geometry, `attach`/`detach` under LoRA, `inherit` under
    LoRA accepted, width/depth change, illegal mode/state pair, non-`mixed`
    refiner-only save, re-attach mid-anneal, understanding-only run, changed
    accumulation interval mid-anneal.

No convergence runs.

## Acceptance measurement (pre-registered quality gate, not run)

GPU runs require the user's explicit go-ahead; a training run is usually in
flight. The target range is 2048-4096px. A 1536-only improvement is not enough
to accept the design. Run128 is an engineering smoke at the 2048-area bucket
family, not this measurement: it proves execution, persistence and resume, not
quality or 4096 feasibility of the complete base workload.

### Stage 1: isolate branch capacity

Start from run127 `step_093791`, attach width 128/depth 3, and train
`refiner_only`. The base weights remain bit-identical, so any output change is
caused by the new branch rather than base co-adaptation. Use the production
mixture of 2048, 3072 and 4096 buckets; log the first 256
`(dataset_id, item_id, bucket)` triples so repeats are auditable. Evaluate at
2,000, 5,000 and 10,000 updates.

Samples at each point are 8 fixed prompts x 3 fixed seeds at both 2048 and
4096, with fixed steps and CFG. Metrics from the P3 probe are grid64, grid8,
4-16px band energy, and the 8-16px residual diagnostic. Also report
`sn_refiner_delta_rel`, peak allocated/reserved VRAM and seconds/update by
bucket. Blind review uses label-hidden zoomed crops; a scalar result alone is
not accepted.

Stage 1 advances when, relative to the unmodified source checkpoint at both
resolutions, grid64 and grid8 improve in at least 19/24 paired samples,
4-16px energy improves in at least 19/24, artifact-focused blind review
prefers the refined crop in at least 19/24, and large-scale composition is
judged worse in no more than 5/24. This is a capacity gate, not the final
causal comparison; only Stage 2 can establish benefit over continued base
training.

If width 128/depth 3 is active (`delta_rel >= 0.05`) but misses the quality
gate, change only one capacity axis at a time:

1. depth 5 / width 128, testing receptive-field limitation;
2. depth 3 / width 256, testing channel-capacity limitation;
3. only if both help independently, width 256/depth 5.

Width 64/depth 3 is an optional lower-cost ablation, not a rescue arm. A
dilated or multiscale architecture is considered only if depth helps and its
cost is unacceptable; attention is considered only if the residual diagnostic
shows a genuinely non-local error. This prevents one width-256/depth-6 run
from confounding receptive field, parameter count and compute.

### Stage 2: matched end-to-end control

Run only after Stage 1 shows useful capacity. Two arms branch from the **same**
checkpoint and optimizer state (run127 `step_093791` plus its optimizer), with
identical seed, dataset config, 2048/3072/4096 bucket sequence, warmup, LR
schedule and update budget:

- **Arm A:** `attach`, selected Stage-1 width/depth, mode `joint`.
- **Arm B:** `inherit` on the refiner-less checkpoint; normal base training.

The comparison is refused if the first 256 logged data/bucket triples differ.
Evaluate both at 2,000, 5,000 and 10,000 updates with the same 24 prompt/seed
pairs at 2048 and 4096. Report median/IQR and paired signs separately at each
resolution; do not pool the two resolutions.

### Pre-registered outcomes at 10,000 updates

- **Accept:** at both 2048 and 4096, A < B on grid64 in at least 19/24 pairs,
  A < B on grid8 in at least 19/24, A > B on 4-16px band energy in at least
  19/24 (two-sided binomial p < 0.01 for each), and blind inspection selects A
  as less blocky/hatched in at least 19/24. A may be judged worse in prompt
  adherence or large-scale composition in no more than 5/24 pairs. P4 must
  also keep the 4096
  checkpointed peak delta at or below 1.5 GiB and within the owner's usable
  headroom.
- **Architecture/capacity failure:** `delta_rel >= 0.05`, the one-axis ladder
  has been run, and none reaches the paired quality gate. If depth helps, test
  dilation/multiscale; if width helps, choose the smallest passing width. If
  neither helps and the high-frequency residual is not flat, the missing
  information is upstream of this head and the refiner should be detached.
- **Objective failure:** the 8-16px residual stays flat while total loss falls.
  Adding capacity is not justified; first test frequency-aware loss weighting
  as a separate design.
- **Inconclusive (LR-bound):** `delta_rel < 0.05` by 5,000 updates. Lion moves
  each element by at most `lr` per update; at run127's `unet_lr=1e-6` and
  factor 1.0 a flat 10k result is not evidence against the branch. P4 reports
  delta growth per 1k updates, then one pre-declared LR-factor retry is run.
- **Inconclusive (memory-bound):** the 4096 peak delta exceeds 1.5 GiB or usable
  headroom. Implement and verify exact-halo execution before any width increase,
  then repeat the same arm; reducing evaluation resolution is not acceptance.

The 0.05 activity threshold, 19/24 paired counts and 1.5 GiB 4096 delta are
fixed before the runs so the outcome cannot be chosen afterward. After a
positive result, the recommended production sequence is refiner-only warm-up
followed by a short joint finish. Base-only fine-tuning remains a supported
distribution/downstream operation, not the default recipe. A self-contained
bf16 refined distribution can do it without the original int8 base, but pays
for both MoT halves in floating-point VRAM and must continue saving as bf16.
