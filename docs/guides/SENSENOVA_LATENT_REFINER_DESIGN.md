# SenseNova Latent Refiner (optional fine-scale head branch)

Status: **design, not implemented.** Nothing here claims the branch improves a
run. The acceptance measurement at the end is pre-registered and has not been
run.

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
(`vendor/modeling_neo_chat.py:704`; training `ops/sensenova_ops.py:2349-2353`).
The head receives the last hidden state only (`modeling_neo_chat.py:683-687`;
`sensenova_ops.py:2339-2341`). There is no path from the noisy input `z` to
the output at the latent-cell grid, and no timestep input to the head. Detail
already present in `z` has to be packed into a 64px token and re-drawn by two
convolutions. At low noise, where `x0 ≈ z`, that is all of the detail.

The measured symptom on run127's samples (CPU spectrum/grid probe, same prompt
and seed, 2026-09-16): 64px-periodic gradient structure at 2.5x an
incommensurate-period control (0.51 vs 0.20), and 8px-periodic structure at
4.5x the SDXL-VAE round-trip of real images (0.135 vs 0.030). 4-16px band
energy is about 0.4x of run126 (native 32px token). The VAE round-trip keeps
94-100% of every band, so the VAE is not the limit. This is consistent with
the missing path, but does not prove it: run127 and run126 also differ in the
VAE swap itself.

The direction here keeps the transformer's token count and compute fixed. The
fine-grid capacity is added the way SDXL's UNet has it: convolutions at the
latent-cell grid plus a direct path from the input.

## Scope

In scope:
- SenseNova generation branch with a latent geometry (`gen_vae_scale_factor > 1`).
- Full fine-tune only, same rule as the VAE swap
  (`backend/core/training/arch/sensenova.py:219-236`).
- Attach, continue and detach at a resume boundary.
- Generation: identical forward to training, driven by the checkpoint alone.

Out of scope:
- Pixel-space SenseNova (`gen_vae_scale_factor == 1`). The refiner grid would
  be the pixel grid (2.36M positions at 1536px square). Refused, with the reason
  given.
- Adding or removing the branch inside a live training process. A running run's
  config cannot be edited (`PUT /training/runs/{id}` refuses while running).
  "Mid-run" means between resume segments of the same run, which is where every
  other resume-time change in this repo happens.
- LoRA / ReLoRA / ControlNet training of the refiner. A LoRA run on a base
  that already has a refiner runs the frozen refiner in its forward and is
  allowed.

## Mechanism

### Forward

One function, `apply_latent_refiner`, used by every call site (see
[Call sites](#call-sites)):

```
x0_head  = head output, unpatchified to the latent grid   [B, C, H, W]
z_grid   = z unpatchified to the latent grid              [B, C, H, W]
c_in     = 1 / sqrt(t^2 + ((1 - t) * s)^2)                 (s = noise_scale)
h        = stem(concat(x0_head, c_in * z_grid))           3x3, 2C -> width
h        = ResBlock_i(h, emb(t))   for i in 1..depth       GroupNorm + FiLM(t) + 3x3 x2
delta    = out(h)                                          3x3, width -> C, ZERO-INIT
x0       = x0_head + gate * delta
```

- `z = t * x0 + (1 - t) * eps * s` in this repo's convention
  (`sensenova_ops.py:2273`, t=1 clean). `c_in` gives the concatenated `z`
  unit-order scale at every `t` and every resolution. `s` comes from
  `compute_noise_scale`, which depends on token count. Raw `z` would have
  resolution-dependent statistics, and training uses one base resolution.
- `emb(t)` is the refiner's own sinusoidal embedding plus a 2-layer MLP. It
  does not reuse `fm_modules["timestep_embedder"]`, because that would couple
  the refiner's gradients into a tensor the transformer path already trains.
- Receptive field: each ResBlock adds a radius of 2 cells. The default
  `depth=3` gives a radius of 7 cells (stem included), which crosses the
  8-cell (64px) token boundary.
- `gate` is a persistent non-trainable buffer (`fm_refiner.gate`, scalar,
  float32), 1.0 while attached. It exists for annealed detach and is saved with
  the weights. Inference therefore reads exactly the value training last used.

- **Dtype policy is owned by the function.** `train_step` runs the head under
  `torch.autocast(cuda, bf16)` (`sensenova_ops.py:2327-2328`). Inference runs
  plain bf16 with no autocast (`sensenova_pipeline_ops.py`,
  `pipeline_backends/sensenova.py`). Today's head is convolutions only, so both
  paths compute in bf16. GroupNorm, FiLM and the timestep MLP would not: autocast
  runs GroupNorm in fp32. `apply_latent_refiner` therefore disables autocast
  inside and casts explicitly: GroupNorm in fp32, everything else in the
  refiner's parameter dtype. Both callers then compute the same thing.
- **Only under `use_pixel_head`.** The refiner is built and called only on the
  ConvDecoder branch (`modeling_neo_chat.py:275-278, 674-692`). Training
  already refuses the deep/plain heads (`sensenova_ops.py:1065-1100`).
- **`noise_scale` is required.** With a declared refiner, `_t2i_predict_v`
  raises when `noise_scale` is not passed, instead of skipping the refiner. The
  vendor-internal callers (`modeling_neo_chat.py:987-1000, 1332-1345,
  1660-1714, 1884-1887`) have no SushiUI caller; this keeps them from silently
  running a different forward.
- **`c_in` assumes unit-RMS latents.** True for the SDXL latents as normalized by
  this repo's VAE wiring (measured RMS ~1.01). For another VAE it depends on
  that VAE's `norm`. The declaration records `inputs: ["x0_head", "z_cin"]` so
  a later normalization change is a new version, not a silent reinterpretation.

Defaults: `width=128`, `depth=3`. These are sizes to measure, not claims.
Estimated cost at 1536px square: 192x192 = 36,864 cells. At width 128 a 3x3
conv is 147K params and ~5.4 GMAC (~10.9 GFLOP) forward. With three ResBlocks
(six such convs) the branch is ~1.1M params and ~65 GFLOP per image, against
~9,400 GFLOP (2 x 8.17B x 576 tokens) for the transformer forward. Activation memory is the real cost. Each width-128 bf16
activation at 1536px square is 9.4 MiB per image. The branch runs under
`gradient_checkpointing` when the trainer has it on. These numbers are
estimates; implementation phase P3 measures them.

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
| `sensenova_refiner_width` | int, `0` or `>= 16` | `0` | `0` = inherit from checkpoint, else `128` on first attach |
| `sensenova_refiner_depth` | int, `0` or `>= 1` | `0` | `0` = inherit from checkpoint, else `3` on first attach |
| `sensenova_refiner_lr_factor` | float `> 0` | `1.0` | LR = `unet_lr * factor` |
| `sensenova_refiner_detach_mode` | `"anneal" \| "hard"` | `"anneal"` | How `detach` removes the branch |
| `sensenova_refiner_detach_steps` | int `>= 1` | `1000` | Anneal length, in optimizer updates |

`inherit` and `0` are sentinels for the same reason as `sensenova_gen_patch=0`
(`arch/sensenova.py:206-211`). `update_training_run` sends every Pydantic
default, so editing an unrelated field of a refined run must not arrive as
"off" and detach the branch silently.

### State table

Resolved in the arch handler at component load, before optimizer construction.
`ckpt` is the refiner state recorded in the checkpoint being loaded (none for a
fresh base).

| ckpt | request | Result |
|---|---|---|
| none | `inherit` | no refiner |
| none | `attach` | **attach**: build at width/depth (defaults when 0), zero-init `out`, `gate=1` |
| none | `detach` | no refiner, info notice |
| attached | `inherit` / `attach` | **continue**. A non-zero width/depth that differs from ckpt is refused (shape change is not a resume) |
| attached | `detach` + `hard` | **hard detach**: branch dropped before the first step. A warning gives the last logged `sn_refiner_delta_rel` |
| attached | `detach` + `anneal` | **anneal**: refiner frozen (`requires_grad=False`), `gate` ramps 1 -> 0 linearly over `detach_steps` updates from the resumed step. Rest of the model keeps training |
| annealing | `inherit` / `detach` | **continue anneal** from the recorded anchor. The gate is a function of (anchor, steps, current step), not of the saved gate |
| annealing | `attach` | refused in v1 (re-attach mid-anneal) |
| anneal complete (`gate == 0`) | any but `attach` | branch dropped at load. The next save omits it, which is lossless because `gate=0` already contributed nothing |

**Anneal clock.** The same axis the existing warmup uses. The anchor is the
resumed scheduler position (`global_step // gradient_accumulation_steps`,
`base_trainer.py:380-390`) and the clock is `scheduler.last_epoch`
(`fresh_param_warmup.py:28`). `gate = clamp(1 - (last_epoch - anchor) / steps, 0, 1)`
is written to the buffer on update boundaries, before the forward of the next
micro-batch. The declaration persists `detach_anchor_step` (scheduler axis),
`detach_steps` and the accumulation interval. A resume with a different
`gradient_accumulation_steps` mid-anneal is refused.

**Declaration upkeep.** `trainer.sensenova_config_dict` is mutated at each
transition, the way `apply_noise_scale_gain` does (`arch/sensenova.py:270,
327-333`): `gen_refiner` is added on attach, gets the anchor on anneal start,
and is **deleted** before the first save after hard detach or anneal
completion. Otherwise the strict check below would refuse the next load.

Hard detach is not lossless. The rest of the model co-adapted with the branch.
The anneal exists so the model re-absorbs the branch's contribution before it
is removed.

Pixel-space geometry with `attach` is refused, with the reason given.

## Persistence

### Tensors

Under `fm_modules.fm_refiner.*`. Every save format (`mixed` / `bf16` / `int8`)
writes non-decoder-Linear tensors as-is (`core/models/sensenova/loader.py:1014-1022`),
so no format-specific code is needed. `accept_resume_shaped_base`
(`sensenova_ops.py:735-871`) counts only decoder Linears. The bf16 single-half
resume restores only `iter_sensenova_lora_targets(branch=frozen_half)` from the
base (`loader.py:600-602`), so refiner tensors always come from the checkpoint.

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
  "detach_anchor_step": null,
  "detach_steps": null
}
```

`NEOChatModel.__init__` builds `fm_modules["fm_refiner"]` only when
`config.gen_refiner` is present. When it is absent, the model is exactly
today's.

### Strictness

`install_sensenova_state_dict` loads with `strict=False` and only prints
missing/unexpected keys (`loader.py:1110-1114`). A refiner whose tensors are
dropped would silently run as zero-init. `init_empty_weights()` defaults to
`include_buffers=False` (accelerate 1.12.0), so a missing `gate` would not crash
either. It would be materialized at 1.0 and cast to the load dtype by
`model.to(torch_dtype)` (`loader.py:1223`), which is silently wrong for an
annealing checkpoint. So for the `fm_modules.fm_refiner.` prefix only:

- declared and any refiner key missing -> refuse the load;
- refiner keys present and not declared -> refuse the load.

This does not change the existing tolerance for other keys.

## Optimizer and warmup

- New param group `generation_refiner`, `lr = unet_lr * sensenova_refiner_lr_factor`,
  **appended last** in both group orders in
  `SenseNovaFullParameterAdapter.arch_param_groups`
  (`core/training/adapters/sensenova_adapter.py:632-648`). Every existing
  group keeps its index.
- `_fm_parameters` must **exclude** `fm_refiner`. Today, with
  `sensenova_train_fm_modules`, every `fm_modules` param goes into
  `generation_decoder` (`sensenova_adapter.py:416-420`).
- Attach and continue both happen at process start, so the group exists when
  the optimizer is constructed and when fused hooks are registered
  (`setup_optimizer`, `base_trainer.py:8497`). No `add_param_group` is needed.
  This is what makes the fused path work: registration refuses a trainable
  param with no group (`optimizers/fused_backward_registration.py:84-94`), and a
  later-added param would never get a hook.
- **Fresh detection works as is.** The named optimizer load marks tensors absent
  from the saved state as fresh and sets `partially_fresh`
  (`base_trainer.py:6026-6050`). run127's `step_090000_optimizer.pt` carries
  `_sushi_param_names` (audited from the pickle header), so the named path is
  taken. Without names, the legacy prefix remap also marks a trailing group
  fresh (`base_trainer.py:5975-5987`).
- **Warmup on attach does NOT work as is on the fused path. Two defects, both
  fixed in P0 before any refiner code:**
  1. *Whole-group fresh never reaches the per-tensor cohort.*
     `_rearm_warmup_after_optimizer_reset` arms `arm_fresh_param_warmup` only for
     `mixed_ids`, i.e. fresh tensors in groups that also hold restored tensors
     (`base_trainer.py:7806-7833`). A wholly fresh appended group such as
     `generation_refiner` instead gets a composed `scheduler.lr_lambdas[i]`
     (`base_trainer.py:7835-7839`), which the fused hook never reads (defect 2).
     Fix: on the fused path without fused groups, arm the cohort for **all** fresh
     ids, whole-group included. `fresh_param_warmup_factor` is read off the
     optimizer object (`fresh_param_warmup.py:25`) and does reach the hook
     (`lion8bit_ringbuffer.py:697`).
  2. *Fused hooks read a stale group dict after any resume.* The hooks capture
     the group dict at registration (`lion8bit_ringbuffer.py:642-647`, via
     `fused_backward_registration.py:57-62`). Registration runs in
     `setup_optimizer`, before the scheduler fast-forward
     (`base_trainer.py:15818`) and the state load (`:15826`).
     `_load_state_dict_uint8`'s `update_group` returns the *saved* dict, and
     `__setstate__` replaces `param_groups` with it
     (`lion8bit_ringbuffer.py:374-383`; the named load calls it at
     `base_trainer.py:6061`). From then on the scheduler writes the live dicts
     and the kernel reads the orphaned one. **This is an existing defect
     independent of the refiner.** Every fused Lion RB resume applies the LR
     frozen at the resume position, while the logged LR (read from the live
     dicts) keeps following the schedule. Fix: make `_load_state_dict_uint8`
     update the live group dicts in place (keep identity), or make hooks resolve
     their group by index at call time. Apply the same audit to the other
     ring-buffer/fused optimizers that capture `group`
     (`adamw8bit_ringbuffer.py`, `adafactor_fused.py`, `adamw8bit_fused.py`).
- With both fixed, attach ramps the refiner 0 -> 1 over `optimizer_warmup_steps`
  from the resumed scheduler position. A new run that attaches at step 0 is
  covered by the global warmup.
- Limitation, not changed here: the cohort is one per optimizer (the `setattr`
  at `fresh_param_warmup.py:14`) with one warmup length. The refiner ramps over
  `optimizer_warmup_steps` together with any other tensor fresh in the same
  resume. A separate refiner warmup length would need a multi-cohort extension.
  It is not proposed until a measurement shows the shared length is wrong.
- Annealed detach: the refiner is frozen, so it is in no group. The named load
  drops its saved state. Fused registration skips frozen params
  (`fused_backward_registration.py:100-105`).
- Hard detach and anneal completion: tensors are gone. Same as above.

## Generation

The checkpoint is the only input, so there is no generation parameter:

1. `load_sensenova_from_path` builds the config from `sensenova_config`
   (`loader.py:161-186, 1199`), and `NEOChatModel` builds the refiner from
   `gen_refiner` with its width/depth.
2. The strict prefix check (above) guarantees trained weights and the saved `gate`.
3. `_t2i_predict_v` calls `apply_latent_refiner` after the head, before the
   `v` conversion. CFG runs it per branch, so cond, img_cond and uncond each
   get the refiner, exactly as training's single branch does.
4. A LoRA loaded onto a refined base changes decoder Linears only
   (targets come from `iter_sensenova_lora_targets`, applied through the
   SenseNova backend's LoRA group, `pipeline_backends/sensenova.py:228-245`). The refiner runs as part
   of the base.

`noise_scale` must reach `_t2i_predict_v`, which does not take it today. The
pipeline computes it in `_build_step_context`'s callers. Training has it at
`sensenova_ops.py:2273`. P1 threads it explicitly and adds no second
computation.

A generation-side bypass toggle for A/B is deliberately not proposed. It would
be a way to run a checkpoint differently from how it was trained.

## Call sites

Every site that runs the head must go through `apply_latent_refiner`, or
refuse a refined model:

| Site | Action |
|---|---|
| `vendor/modeling_neo_chat.py:687` (`_t2i_predict_v`, pixel-head branch) | call |
| `core/training/ops/sensenova_ops.py:2339` (`train_step` re-implementation) | call |
| `sensenova_pipeline_ops.py:704-739` (`_predict_v_branch`, goes through `_t2i_predict_v`) | covered by the vendor call; add `noise_scale` |
| `probes/sensenova_cfg_null_parity.py:432` | call |
| `probes/sensenova_und_discrimination.py:289`, `probes/sensenova_real_checkpoint.py:303`, `probes/text_encode_vs_step.py:827,1003`, `probes/sensenova_und_prefix.py:386` | refuse when `gen_refiner` is declared (several already assume pixel geometry) |

`train_step` and `_t2i_predict_v` share the function, not a copy. P1's parity
test runs both on one tiny model and requires identical output.

## Observability

- `sn_refiner_delta_rel = ||gate * delta|| / ||x0_head||`, per step, through
  `defer_extra_metric` plus a `metric_registry` entry (no DB column). It is
  what hard detach quotes, and it shows whether the branch does anything.
- `sn_refiner_gate`, logged only while annealing.
- A load-time log line and a `training_events` notice for each state
  transition (attach / continue / anneal start / anneal complete / hard detach).

## Plumbing checklist

Per the existing SenseNova option path (template: `sensenova_gen_patch`):

1. `backend/api/param_defaults.py` `TRAINING_DEFAULTS`: the 6 keys.
2. `openapi.yaml` `TrainingRunCreateRequest`: fields, descriptions, examples.
3. `backend/api/routes.py` `TrainingRunCreateRequest` plus request checks next to
   `_check_vae_swap_params`, on POST and PUT. Use `model_fields_set` where
   "explicitly set" matters.
4. `backend/core/training/training_config.py` `_build_train_section`: keys in the
   dict literal. `train_section_key_vocabulary()` reads the literal's syntax tree.
5. `backend/core/training/train_runner.py` `_apply_sensenova_full_finetune_contract`:
   normalization and refusals (method, pixel geometry, width/depth change).
6. `backend/core/training/arch/sensenova.py`: state resolution, build, detach anneal.
7. `backend/core/training/adapters/sensenova_adapter.py`: `_fm_parameters` exclusion,
   `generation_refiner` group.
8. `backend/core/models/sensenova/`: new `latent_refiner.py` (module plus
   `apply_latent_refiner`), `latent_space.latent_config_dict` (declaration),
   `loader.py` (strict prefix check), `vendor/modeling_neo_chat.py` (build and call).
9. `backend/api/arch_capabilities.py`: unsupported for non-SenseNova archs and
   non-full-FT methods; `vae_swap`/latent geometry required for `attach`.
10. `frontend/src/utils/api.ts`, `trainingConfigDefinitions.tsx` `DEFAULT_PARAMS`,
    `trainingParams.ts` `PARAM_KEYS`, `TrainingConfig.tsx` controls (state select,
    width, depth, lr factor, detach mode/steps) with `unsupportedTrainingFeature` gating.
11. `backend/core/training/metric_registry.py`: the two series.
12. Probes: call or refuse, per the table.
13. `docs/guides/SENSENOVA_TRAINING_DESIGN.md` / `MODEL_FACTS.md` pointers once implemented.

## Phases

| Phase | Content | Gate |
|---|---|---|
| P0 | Fused-hook stale group dict fix (all capturing optimizers); cohort armed for whole-group fresh ids | CPU test: after `load_state_dict`, an LR written to the live group is the LR the kernel receives; a wholly fresh appended group gets factor 0 at the anchor. Independent of the refiner, committed on its own |
| P1 | `latent_refiner.py`, vendor build+call, `train_step` call, `noise_scale` threading, declaration, strict check | CPU tests below pass |
| P2 | State table, optimizer group, warmup path, anneal, metrics | CPU tests below pass |
| P3 | Params, API, YAML, runner refusals, capabilities, UI, probes | grep parity against `sensenova_gen_patch`, `py_compile` plus real import |
| P4 | GPU smoke, only with the user's go-ahead (a run is usually in flight) | ~3 steps, finite loss, attach/continue/anneal/hard each once; measured activation memory |

One commit per phase.

## Verification plan

CPU, tiny `NEOChatModel` config, in the production dtype matrix (bf16, fp16,
fp32):

1. **Zero-init identity:** attach -> `x0` equals the refiner-less forward exactly.
2. **Train/infer parity:** `train_step`'s head+refiner output equals
   `_t2i_predict_v`'s on the same inputs, both with the caller inside
   `torch.autocast("cpu", dtype=bf16)` and without. Autocast is disabled
   off-CUDA at `sensenova_ops.py:2327`, so the test enables it explicitly.
3. **Round-trip:** save (each format) -> `load_sensenova_from_path` -> same
   output, `gate` preserved and float32 after load.
4. **Strictness:** declared-but-missing and present-but-undeclared both refuse.
5. **State table:** every row, including the default-edit case (`inherit` on a
   refined checkpoint keeps it).
6. **Warmup:** resume of a refiner-less checkpoint with `attach` on the fused
   Lion RB path marks exactly the refiner tensors fresh and arms the cohort for
   them (whole-group case). The LR the hook hands the kernel is 0 at the anchor
   and the live group LR after `optimizer_warmup_steps`. Restored tensors
   receive the live scheduled LR throughout (P0 regression).
7. **Anneal:** the gate follows (anchor, steps) across a simulated re-resume
   mid-anneal; at completion the saved checkpoint has no refiner and the
   declaration is gone; the output before and after the drop is identical.
8. **Refusals:** pixel geometry, LoRA training, width/depth change, re-attach mid-anneal.

No convergence runs.

## Acceptance measurement (pre-registered, not run)

Attach to run127 at a resume. Compare samples from before and after with the
same prompt and seed, using the CPU probe from the Problem section
(64px/8px grid scores against the period-61 control, 4-16px band energy),
**plus visual inspection of zoomed crops**. A scalar metric alone is not
accepted.

- Supports the hypothesis: grid64 and grid8 fall toward the real/VAE-round-trip
  reference, 4-16px band energy rises, and crops show fewer 64px blocks and less
  8px hatching, over several sample steps (per-sample variance is large:
  grid64 ranged 0.23-1.1 on run127).
- Refutes it: `sn_refiner_delta_rel` grows but grid scores and crops do not
  change. The artifact is then upstream of the head, and the branch should be
  detached.
- Inconclusive: `sn_refiner_delta_rel` does not grow. Lion moves each element by
  at most `lr` per update. At run127's `unet_lr=1e-6` and factor 1.0, the
  zero-init `out` conv is bounded by 1e-6 x updates (<= 0.01 after 10k). A flat
  result can therefore be LR-bound, and says nothing about the hypothesis. P4
  reports the `delta_rel` growth rate per 1k updates, so
  `sensenova_refiner_lr_factor` can be chosen from a measurement before the
  acceptance run. The default stays 1.0 because no measurement supports another
  value yet.
