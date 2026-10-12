# Sampling after an image VAE swap

Status: design decision, 2026-10-12. Production implementation is pending.
The numerical and bounded inference validation is complete; long-training
quality acceptance is explicitly deferred to owner runs.

## Scope and confirmed facts

The primary problem is persistent jagged detail in **Anima with the Qwen-Image
2.1 VAE**, represented by run 176: native Anima DiT, target-native 64-channel,
16x codec, paired-spatial bridges and frozen Gaussian noise alignment. It is
not a finding that the Original Anima model itself is defective. A separate
SushiUI native-codec schedule discrepancy is useful for establishing a valid
baseline, not a demonstrated explanation of the swapped model's artifacts.

| Fact | Evidence and limit |
|---|---|
| Original Qwen 2.1 sampling matches upstream | AST-equal packing, shift calculation and timesteps calls at Diffusers revision `6256aa7`; released scheduler uses dynamic shifting and final positive evaluation sigma 0.02. |
| SushiUI Anima uses an unaudited resolution heuristic | Upstream sd-scripts and ComfyUI use shift 3; SushiUI computes an exponential shift from latent area, which is four times its patch-token count. This is an implementation difference, not a quality measurement of Original Anima. |
| Swap does not force constant 16/32px output blocks | The measured 256x2048 output head has rank 256; its four phases emit separate 64-channel predictions. Synthetic detail and one general image reconstruct sharply through the declared codec. This does not prove that the learned DiT produces correct detail. |
| Finer integration alone did not remove the artifacts | Earlier step-122519 known-target loops with low-sigma spacing <=0.0025 and FP32 state still had angular detail. Near-clean input retention is not evidence of learned detail synthesis. |
| Attention precision is a separate concern | Representative real Q/K/V gradients did not reproduce huge-logit kernel failures. Global-summary arithmetic was independently improved to FP32. |
| Current training semantics are known | Run 176 uses uniform sigma in [0,1], no morph/adaptive controller, and global summaries only on actual partition execution. Long-term quality after this change is unverified. |

The new validation is recorded in
[VAE_SWAP_SAMPLING_METHOD_VALIDATION_2026-10.md](../audits/VAE_SWAP_SAMPLING_METHOD_VALIDATION_2026-10.md).
No globally optimal sigma grid, solver or final training quality is claimed.

## Decision

Use a **model/codec/noise-contract sampling profile**, selected against an
explicit model-evaluation budget, with FP32 integration state. Retain audited
native behavior as the baseline. A swap must not automatically copy the target
VAE's original model scheduler: the denoiser and its learned velocity field
remain those of the source architecture.

The first implementation should provide audited native Euler and a conservative
swap candidate with a bounded final clean interval. Fixed clean-heavy grids and
second-order solvers remain alternatives to validate, not presumed winners.
Offline error-based grid calibration is the next stage; per-step online adaptive
sampling and architecture redesign are not required for the initial delivery.

## Mathematical separation

For a normalized target codec, the flow contract is

\[
z_0=\operatorname{Norm}(E_{\rm target}(I)),\quad
z_\sigma=(1-\sigma)z_0+\sigma\epsilon,\quad
v^*=\epsilon-z_0,\quad dz/d\sigma=v_\theta(z,\sigma).
\]

Ming uses the corresponding declared time/sign conversion; SDXL DDPM/epsilon
must use its own alpha/sigma and prediction conversion. Do not feed its noise
coordinate directly to a flow-only solver. Training samplers and inference grids
serve different purposes: uniform training exposure does not require uniform
inference steps. Changing the inference grid must not change the trained noise
path, latent normalization, target sign or timestep embedding units.

A scalar latent-space SNR summary is

\[
\operatorname{SNR}(\sigma)=\frac{(1-\sigma)^2\,
\operatorname{tr}\operatorname{Cov}(z_0)}{\sigma^2\,
\operatorname{tr}\operatorname{Cov}(\epsilon)}.
\]

The covariance, frequency correlations and decoder sensitivity can change after
a swap. Channel count or spatial compression alone cannot determine its best
grid. A single SNR scalar also hides frequency-dependent detail sensitivity.
Codec normalization and Gaussian alignment must be correct before tuning grids;
an inference-only rescaling of sigma is not a substitute for a correct path.

Separate three errors: learned-field bias, integration discretization, and
floating-point rounding. For a Lipschitz field, the terminal error is bounded
by a stability factor multiplying integrated field error plus numerical error.
More evaluations can reduce the latter while leaving the former unchanged.
For a differentiable decoder, locally `delta_image ~= J_decoder delta_latent`;
small latent error may produce visible contour error in sensitive directions.
The Jacobian approximation is local and is not a guarantee near saturation or
large out-of-distribution latents.

For shift `s` and a linear base grid,

\[
\sigma(u)=\frac{s u}{1+(s-1)u},\qquad
\sigma_{\rm last}=\frac{s}{N+s-1}.
\]

Large unchecked shifts can omit the entire low-noise evaluation band. Auditing
that omission is independent of proving what causes run 176's jaggies.

## Profiles and initial candidates

1. **Audited native**: preserve the architecture's actual source policy and
   output conventions. For Anima the comparison baseline is fixed shift 3;
   Qwen 2.1 retains its released dynamic-shift configuration and terminal 0.02.
   Do not reuse Qwen's function defaults instead of the model's configured
   `max_image_seq_len=8192`, `max_shift=0.9`.
2. **Swap-safe candidate**: native base grid, FP32 state, and final positive
   evaluation sigma no greater than 0.02 for a flow profile with at least two
   evaluations. If necessary stretch only the positive grid monotonically,
   preserving start 1 and final 0. This reduces a large final Euler jump; 0.02
   is a bounded starting candidate, not a measured universal optimum. An explicit
   one-evaluation request cannot meet this guard and must be rejected for this
   profile rather than silently exceeding its evaluation budget.
3. **Validation alternatives**: sigma-uniform, a split at 0.2 with 35% of
   intervals below it, a power grid, and a calibrated grid. The 0.2/35% choices
   are experimental fixtures, not automatic defaults for every VAE.
4. **Second-order candidate**: explicit midpoint with FP32 state, compared at
   equal NFE. Its second call is at the interval midpoint, so the last stage
   does not require a model prediction at exactly sigma 0. Native Euler remains
   the initial default until real-model acceptance supports another solver.

Anima full-frame generation bypasses the global adapter. Partial-region execution
uses it, with FP32 summary attention, and retains the full current-stage canvas.
Neither partition count nor deepest U-Net token count should silently alter the
sigma law. They change the executed field and cost and therefore belong in a
profile's compatibility/validation identity.

## Offline grid calibration

Calibrate a frozen checkpoint, never a simultaneously updating training model.
Use a small permitted reference/prompt set and keep independent validation cases.
Start with 24 sigma bins. For Euler, compare one full step and two half steps,
sharing the initial velocity call: 48 pilot evaluations per trajectory. For a
second-order method, compare one full midpoint step and two half midpoint steps:
120 pilot evaluations in the prototype. Report pilot cost separately and include
it when comparing total work; amortization across future samples is a distinct
claim.

Let a weighted local residual estimate the coefficient `C(sigma)` of a method
with local error `C h^(p+1)`. Under a separable, non-cancelling global-error
approximation, minimize

\[
\int C(\sigma)h(\sigma)^p\,d\sigma
\quad\text{subject to}\quad\int h(\sigma)^{-1}\,d\sigma=N.
\]

This gives grid density proportional to `C^(1/(p+1))`: square root for Euler,
cube root for midpoint. This is an approximate numerical optimum, not an image
quality theorem. It ignores nonlinear propagation and residual cancellation;
coarse pilot trajectories can themselves be biased.

Use latent residuals normalized by target-codec statistics first. Optional
decoder-sensitive weighting can use a few small decoded predicted-clean crops
and finite perturbations; do not load the decoder for every solver step. Actual
decoder weighting was not measured by the scalar oracle experiment. Clamp pilot
density extremes, preserve endpoint/positive-step guards and reserve coverage
across the whole sigma interval. Reject profiles that improve only calibration
cases, introduce visible contour artifacts, or exceed the work/memory budget.

The current real-model comparison does **not** justify automatically selecting
the clean-heavy or midpoint candidate. Therefore calibration only proposes a
grid; held-out inference remains the promotion gate. If fields are inaccurate
or multiple candidates are indistinguishable, retain the audited baseline and
log that no improvement was established.

## Training and architecture contract

This design assumes the swapped architecture is trained in the target codec's
coordinates. Before training, verify all of the following:

- Channel count, latent rank, compression, packing/unpacking and output phases
  match the target VAE. Every intended target channel participates in the loss.
- Calibration and normalization are saved; any frozen Gaussian complement is
  restored exactly. Training and generation use the same noise definition and
  bridge parameterization. Real calibration data may follow the owner's runtime
  policy; images inspected by an agent here must be rating:general.
- U-Net operators preserve target-grid geometry and output channels. Calibration
  profiles include the topology and skip policy; validate non-U-Net and U-Net
  variants separately. Token reduction does not itself imply a constant-pixel
  output bottleneck, nor does full rank prove learned fine-detail quality.
- Shared partition/global, crop-decode and checkpoint paths preserve exposure
  noise, concept order, dropout decisions and optimizer accumulation. A solver
  profile must not change dataset bookkeeping or optimizer resume.

For current run 176 retain the authorized uniform training distribution and
velocity objective as the controlled baseline. Evaluate loss by sigma band and
decoded contour behavior, not just aggregate decreasing loss. Pixel/crop detail
loss is a separate optional training intervention, not an automatic part of a
schedule change. `x0_hat=z_sigma-sigma*v_hat` means x0 reconstruction error carries
a sigma-squared factor; increasing the existing reconstruction mix does not
automatically strengthen low-noise velocity learning. Any inverse-sigma weighting
requires a bounded floor and separate gradient/memory acceptance, not an
unbounded compensation at zero.

Long adaptation may be required. Capacity, initialization and loss design can
still be responsible when an accurate integrator produces poor samples. No
schedule result here certifies that the current field will learn correct detail,
or that every source/target VAE pairing is equally trainable.

## Runtime integration and persistence

Suggested responsibilities, not implemented APIs:

| Component | Contract |
|---|---|
| Shared sampling contract | Architecture-owned sigma/time units, prediction target/sign, normalization and native profile; generation and training samples consume the same contract. |
| Grid builder | Finite, strictly descending positive evaluations followed by exact zero; no double shift; audited length definition. |
| Velocity callback | FP32 state to declared model dtype at the boundary; return FP32 velocity; retain FP32 sigma where the architecture's embedding supports it. |
| Solver | Count every stage and retry, honour cancellation and budget, keep persistent stages/residuals FP32. |
| Profile/calibration store | Versioned grids, solver, dtype policy, compatibility hashes, calibration cost and acceptance status. |

FP32 integration must survive the entire loop; upcasting inside `step()` and
casting its result back to BF16 each iteration does not meet the contract.
The prototype preserves BF16 model/timestep inputs to isolate solver/grid effects;
FP32 timestep delivery is proposed and still requires a real boundary test.

In a multi-stage evaluation, reuse the initial exposure noise and frozen alignment
coefficients. Evaluate CFG and its sigma-dependent rules at each stage's actual
sigma. Each partitioned stage needs the **complete current-stage noisy canvas**,
including for global summaries; do not reuse summaries from another stage.
Retain exact condition-prefix KV caching only where it is proven independent of
image state. Disable FBCache/Spectrum/extrapolated image-state caching during
calibration and initial higher-order acceptance. Step-range adapters and style
switches are discontinuities: keep stages within a stable policy interval and
align grids to switches, rather than varying the active model during one step.

Distinguish logical full-canvas velocity NFE from raw model forwards: CFG branches
and partitions can multiply the latter. Display both, and report loading,
encoding, decoding and calibration latency separately. Never compare 40 Euler
evaluations with 40 two-stage intervals as if they had equal cost.

Persist a versioned profile in checkpoint metadata/sidecar with architecture,
codec fingerprint and normalization, patch geometry, alignment/noise hash,
U-Net/global policy, sigma units/sign, solver/grid, state/model/timestep dtype,
resolution range, CFG assumptions, calibration checkpoint and status. Shape or
path-contract changes invalidate it. Continued training makes calibration stale
for quality without making a valid grid mathematically invalid; expose freshness
and retune at selected milestones rather than every step.

Resolution beyond a calibrated range falls back to an audited native profile,
not unconstrained exponential extrapolation. A new codec does not silently inherit
a profile keyed only by a display name. Explicit request settings override a
compatible saved profile; otherwise use that profile, then the audited native
fallback. Default materialization must not masquerade as an explicit override.

Thread the profile through txt2img, img2img, inpaint, queued/loop generation and
periodic training samples. Denoising strength must be interpreted against the
selected grid; it is not generally equal to start sigma. Diagnostics need an
explicit start sigma. Inpaint protection and reference noise also use each
stage's actual sigma. Reject unsupported architecture/solver combinations before
model load instead of ignoring a selected UI control.

API implementation must be OpenAPI-first with defaults in `param_defaults.py`.
Reuse one structured sampling-profile schema for generation and training-sample
settings; preserve config editing and checkpoint restoration. The frontend should
show sigma versus evaluations, clean-band counts, NFE/raw forwards, profile origin,
compatibility and stale/unvalidated state. Cancellation and budget exhaustion
must have explicit API/WS behavior. Owner build/type-check remains the UI gate.

## Delivery and acceptance

1. Audit/native-parity corrections and persistent FP32 state; golden sigma grids,
   packing, sign/time-unit, CFG/inpaint, noise-context and cancellation checks.
2. Versioned profile persistence and all generation/training-sample/UI paths;
   backward-compatible loading and exact config/resume preservation.
3. Optional midpoint and bounded offline calibration; verify fixed-NFE, held-out
   real-model comparisons before promotion. Preserve conservative fallback.
4. **Deferred owner-run gate**: after meaningful adaptation, compare frozen
   checkpoints under identical seed/prompt/NFE. Assess concept retention and
   contours at CFG 1 and the operating CFG, more than one aspect ratio, and
   separate U-Net/non-U-Net variants. Log model/codec/profile versions and phase
   changes. A falling training loss alone does not pass this gate.

The current task finishes at design and bounded validation. It does not implement
these sampling profiles or change run 176's production sampler. The run resumes
with its authorized uniform training and existing configuration.

## Primary references

- [Qwen 2.1 pipeline, pinned Diffusers source](https://github.com/huggingface/diffusers/blob/6256aa7/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- [Released Qwen 2.1 scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json)
- [ComfyUI Anima sampling settings](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/supported_models.py#L1045)
- [Anima upstream sample implementation](https://github.com/kohya-ss/sd-scripts/blob/main/library/anima_train_utils.py#L394)
- [Rectified-flow training densities and resolution-shift derivation](https://arxiv.org/html/2403.03206v1)
- [Diffusers flow Heun implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_heun_discrete.py)

Source inspections establish contracts; their model-quality benchmarks are not
treated as measured performance of this swapped Anima fine-tune.
