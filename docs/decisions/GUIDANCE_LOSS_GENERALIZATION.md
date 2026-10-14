# Architecture-independent guidance-target loss

Status: design decision, 2026-10-14. Implementation in progress (phases below).
Mechanism and validation gate of the loss itself:
[QWEN_IMAGE_21_GUIDANCE_LOSS.md](../guides/QWEN_IMAGE_21_GUIDANCE_LOSS.md).

## Problem

The guidance-target loss trains the conditional prediction toward a CFG-like target:

```
u   = stopgrad(model(x_t, t, null condition))
v_g = u + g(t) * (v - u)          v = the architecture's ordinary regression target
L   = mix(t) * MSE(c, v_g) + (1 - mix(t)) * MSE(c, v)
```

It addresses a symptom seen on Anima run 177: samples blur at CFG 1 and break at
high CFG while single-step predictions keep improving. The loss is only available
for Qwen-Image 2.1 and Ming-Image 0.1:

- The objective, selection and cap live inside `training/ops/qwen_image_21_ops.py`.
  Ming imports private helpers from it.
- Every option is a `qwen_guidance_loss_*` key.
- `base_trainer` gates the feature, the empty-prompt cache and the stochastic draw
  on `is_qwen_image_21 or is_ming_image_01`. Any other architecture is refused.

## Principle

The guided target is linear in the prediction. Inference CFG combines branches
linearly in the same space the model is trained to predict: epsilon, v, flow
velocity or x0. The objective is therefore space-agnostic, as long as `u` and `v`
are in the architecture's own training-target space.

Only two things are architecture-specific:

1. How to run the model once more on the same noisy input under the inference
   null condition.
2. How to map the training timestep to a noise level in [0, 1] (1 = pure noise).
   This drives the `sigma` scale schedule, the weight ramp, and the extrapolation
   cap's clean-space distance.

## Design

### Core module `core/training/guidance_loss.py`

Owns everything that is not architecture-specific. It is moved out of
`qwen_image_21_ops`, which keeps re-exports so existing imports stay valid.

- `GuidanceSettings.from_config(config)`:
  - Reads the canonical keys, falling back to the legacy `qwen_*` keys.
  - Validates them: weight in [0, 1]; scale in [1, 10]; schedule `constant` or
    `sigma`; mix mode `stochastic` or `blend`; weight ramp; finite cap >= 0.
  - Reports `active`: a positive weight, or a positive high-noise ramp weight.
- `plan(trainer, noise_level, cfg_drop_mask)`:
  - Selects eligible rows. Rows drawn for the CFG null are not eligible, as today.
  - Computes per-row mix and selection, reusing the batch-level stochastic draw so
    OOM retries and micro-batch slices see the same choice.
- `objective(plan, prediction, target, uncond, noise_level, *, per_sample_error, clean_scale)`:
  - Returns `(loss, per_sample_loss)`.
  - `per_sample_error` lets Ming keep frame weights.
  - `clean_scale(noise_level)` converts a target-space difference into clean-space
    for the optional RMS cap. Default: flow `sigma`.
  - Logs `guidance_loss_{normal,guided,mix_weight,selected_fraction,...}` through
    the extra-metrics channel. Legacy `qwen_guidance_*` registry entries stay so
    old runs keep their charts.
- `null_condition(trainer)`:
  - Caches the null encoding once per run on the CPU via
    `trainer.encode_caption("", cfg_null=True)`. This is the same call used by
    `cfg_uncond_drop_rate`, so the teacher sees exactly the inference uncond
    condition.
  - Requires a frozen text encoder, as today.

### Architecture contract (`ArchHandler`)

- Support is declared as the `guidance_loss` feature of
  `api/arch_capabilities.py` (`TRAINING_FEATURE_UNSUPPORTED`), not as a new
  handler attribute. The training form already hides features from that table,
  and the trainer refuses an active setting with its reason. An architecture is
  enabled by adding it to `GUIDANCE_LOSS_ARCHS`.
- `guidance_noise_level(trainer, timesteps)`. Default: flow sigma. DDPM
  architectures override it with sigma_t mapped from the scheduler's
  `alphas_cumprod`, normalised to [0, 1].
- Inside the arch's own `train_step` (ops body), after the forward closure is
  built:

  ```
  plan = guidance_loss.plan(...)
  if plan.selected.any():
      with torch.no_grad():
          u_sel = forward(null rows)
  loss = guidance_loss.objective(...)
  ```

  Building the null rows is stage-specific:
  - caption and encode stages: replace the selected rows' conditioning with the
    cached null encoding, using the arch's existing batching of encodings;
  - collated stage: `apply_cfg_null_collated` with an all-selected mask.

  The forward reuses the exact noisy latents, timestep, noise-alignment kwargs,
  position ids and partition plan of the conditional pass. When no row is
  selected there is no extra forward.

### Parameters (OpenAPI first, defaults in `param_defaults.py`)

- Canonical keys: `guidance_loss_weight`, `guidance_loss_mix_mode`,
  `guidance_loss_scale`, `guidance_loss_schedule`,
  `guidance_loss_extrapolation_max_rms`, `guidance_loss_weight_schedule`,
  `guidance_loss_high_noise_weight`, `guidance_loss_ramp_start`,
  `guidance_loss_ramp_end`.
- Defaults are unchanged. The weight is 0, so every run is unaffected until
  enabled.
- Legacy `qwen_guidance_loss_*` keys in saved YAML and API requests map to the
  canonical keys. New YAML writes only canonical keys. Edit Config reads either.
- The per-arch new-run default for Qwen (weight 1.0) is kept.
- `qwen_cfg_null_sigma_schedule` stays Qwen/Ming-only. It rewrites encodings
  inside the caption batch, which needs a per-arch replacement hook. Out of scope.
- The frontend `QwenTrainingObjective` becomes a generic objective control, shown
  for any architecture whose capability says supported.

## Architecture matrix

| Architecture | Target | Null condition | Plan |
|---|---|---|---|
| Qwen-Image 2.1, Ming-Image 0.1 | flow | existing | Phase 1: move onto the core, no behaviour change |
| Anima | flow (eps − x0) | empty caption (caption stage) | Phase 3 |
| Z-Image, Krea 2 base, FLUX.2 base, SenseNova SDXL Chimera | flow | caption stage | Phase 4 |
| Lens, MiniT2I | flow / x0 | collated rewrite | Phase 4 |
| LTX-2.3 | flow (video loss only) | caption stage | Phase 4; audio is not supervised, as today |
| SD1.5, SDXL | eps / v (DDPM) | caption stage | Phase 5; noise level from `alphas_cumprod` |
| Ideogram 4 | flow | separate uncond transformer | Phase 6; the teacher is the uncond transformer |
| SenseNova U1.5 | flow | encode-stage prefix | Phase 6 |
| FLUX.2 distilled, Krea 2 distilled/turbo, MiniMax-H3, ACE-Step | — | — | Refused: guidance-distilled, no CFG branch at inference |
| YuE2, MiniMax Music 3 | — | — | Refused: no diffusion CFG training path |

## Phases and verification

Each phase is committed separately. The CLAUDE.md comparison review applies to
phases 1 and 2.

1. **Core extraction.** Move the objective, selection, cap and settings into the
   core module, and put Qwen and Ming on it. Gate: the existing Qwen and Ming
   guidance tests pass unchanged, plus a test that the canonical and legacy keys
   resolve identically.
2. **Generic parameters and capability gate.** Canonical keys through
   `param_defaults`, OpenAPI, routes, the YAML builder and the frontend; the
   arch support attribute; the trainer gate and draw generalised. Gate: refusal
   tests for unsupported architectures, and Edit-Config round trip with legacy
   keys.
3. **Anima.** Gate:
   - CPU unit tests: the guided target is exact; the teacher sees the null
     encoding and the same noisy input with alignment noise; CFG-null rows get
     ordinary MSE; weight 0 makes no extra forward.
   - A ~3-step GPU smoke in the production dtype (FP32 weights, BF16 compute):
     finite loss, VRAM and iteration time recorded.

   Implementation notes:
   - The teacher forward runs before the training forward, so it is the full
     network: TREAD, BlockSkip and stochastic depth are not attached yet, and
     the REPA tap that the training forward stashes is not overwritten.
   - The null encoding is `encode_caption("", cfg_null=True)`, cached once per
     run; a text encoder parked on CPU is staged to the GPU for that one encode.
   - Anima partition training refuses the guidance loss for now.
4. **Remaining flow architectures**, one commit each, same unit gate.

   Implementation notes (CPU unit gate passed; GPU smoke pending, like Phase 3):
   - Z-Image, Krea 2, FLUX.2: caption-stage null. FLUX.2 keeps reference
     latents in the teacher, as its inference uncond branch does.
   - Guidance-distilled checkpoints are refused after load through
     `ArchHandler.guidance_loss_refusal` (Krea 2 `krea2_is_distilled`, FLUX.2
     `is_distilled`, Z-Image a "turbo" model path or `config.is_distilled`,
     the same detection as the `cfg_uncond_drop_rate` refusal).
   - SenseNova SDXL Chimera: only the plain `flow_velocity` contract. The v2
     and polar contracts and the `bridge_align` stage are refused, because
     inference CFG has not been verified to be linear in those
     parameterizations. Noise level is `1 - t` (t1 convention).
   - LTX-2.3: video loss only, as before. The teacher reuses the step's dummy
     audio.
   - Lens, MiniT2I: collated stage. The handler supplies
     `guidance_null_condition`, which is the arch's own
     `apply_cfg_null_collated` on the selected rows, so ops bodies still never
     rewrite the batch by label. MiniT2I uses noise level `1 - t`.
5. **SD1.5 / SDXL.** Noise-level mapping test against the scheduler.
6. **Ideogram 4 and SenseNova.**

### Phase 2 comparison review

| Behaviour | Before | After |
|---|---|---|
| Keys | `qwen_guidance_loss_*` only | canonical `guidance_loss_*`; legacy keys alias them (request, YAML, Edit Config, presets); a conflicting pair is rejected |
| Saved YAML without a mix-mode key | blend | blend (unchanged) |
| New YAML | writes `qwen_guidance_loss_*` | writes `guidance_loss_*` only |
| Gate | `is_qwen_image_21 or is_ming_image_01` | capability table; refusal names the reason |
| Activation | Qwen: weight > 0; Ming: weight > 0 or a positive high-noise ramp | both: weight > 0 or a positive high-noise ramp |
| Metrics | `qwen_guidance_loss_*`, `ming_guidance_*` | unchanged for Qwen/Ming; generic `guidance_loss_*` registered for new archs |

Breaking edge: a Qwen run with weight 0 and an active high-noise ramp now counts
as active, so it requires a frozen text encoder and caches the blank encoding.
Its loss is unchanged, because the Qwen condition still returns None at weight 0.

The quality acceptance stays the owner's: paired fixed-seed CFG 1 / 3 / 7 grids
at weights 0, 0.1 and 0.25, as in the Qwen guide. A lower training loss is not
acceptance.

## Cost

Each selected row adds one no-gradient forward (no activations kept), so iteration
time rises roughly in proportion to the selected fraction. Peak VRAM is
essentially that of an inference forward. Both need measuring per architecture.
