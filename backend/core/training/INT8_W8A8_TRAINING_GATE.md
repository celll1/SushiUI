# INT8 W8A8 in training — measurement gate G3 (pre-registered)

**Status: pre-registered. Written before the measurement it decides, and before
any `autograd.Function`, kernel, config key, or trainer change exists.**

This file is the decision rule for whether SushiUI should build a
gradient-capable INT8 W8A8 forward path for training (the thing
musubi-tuner PR #1008 does). It is *not* a design document and it does not
describe an implementation, because none exists and none may be written until
this gate is evaluated.

## Why this file lives here and not in `examples/api/README.md`

The two existing pre-registered gates, **G1** (FP8 `torch._scaled_mm`) and
**G2** (int8 W8A8 at inference), live in `examples/api/README.md` because both
are *inference* gates: their vehicle is a real generation driven over
`/api/v1`, their harness is `examples/api/bench_fp8_scaled_mm.py`, and their
result is a decision about a generation default.

G3 is a *training* gate. Its subject is `Int8Linear` as owned by a trainer —
the modules that `training/ops/{krea2,anima,ideogram4}_ops.load_components`
explicitly hand to `disable_int8_mm()` today — its arithmetic is over training
token counts and aspect-ratio buckets from `training/bucketing.py`, and its
verdict changes `backend/core/training/`, not a generation default. It needs no
backend and no HTTP call. Putting it next to the training code it governs (and
next to `TRAINING_PARAMS_GUIDE.md` / `MODEL_ARCHITECTURES.md`, which
`AGENTS.md` already routes training questions to) keeps the gate discoverable
from the code it constrains. `examples/api/README.md` carries a one-line
pointer so nobody mistakes G2 for covering the training case; the full rule
text exists only here, so there is exactly one copy to edit.

## What is already true, and therefore not the prize

Recorded so the gate is not later argued against a strawman:

- **LoRA over a quantized base already trains correctly today.** The base
  `Int8Linear` runs `_dequant_forward` (`vendor/int8_linear.py:662`), which is
  a plain `F.linear` on a dequantized weight; gradients flow through it to the
  LoRA parameters with nothing detached.
- **The memory win is already ours.** The int8 weight is stored as an int8
  buffer either way; the fast path does not reduce resident weight bytes.
- **What is refused is only the fast forward.** GATE 1/GATE 2 in
  `_int_mm_forward` refuse W8A8 whenever grad mode is on or the owner opted
  out.

So the entire remaining prize is **forward compute**, doubled by gradient
checkpointing's recompute. The gate is sized accordingly.

## The rule

### Build proceeds only if ALL of:

1. **Projected end-to-end step-time reduction >= 10%** versus the current
   dequant-path LoRA training, on **at least two architectures (Krea 2 and
   Anima)**, computed from measured per-layer timings and the measured forward
   fraction of a real step. The projection must include:
   - gradient-checkpoint recompute (the forward runs twice per step),
   - amortized Triton autotune cost across our real aspect-ratio buckets.
2. **No tested workload regresses by more than 3%.**
3. The projection **must not rely on an autotune-warm state that a real run
   would not have.**

### Alternative sufficient condition

The fused path **removes an otherwise unavoidable OOM** — i.e. it enables a
configuration that cannot run today at all. This is sufficient on its own,
independent of the 10% bar.

### Rationale for the 10% number, recorded now

The reference's own honest gain is ~5% against a clean bf16 baseline. Its
headline "faster than fp8" was measured against a dequantize-then-bf16 path
that SushiUI has already replaced with `_scaled_mm` plus fused epilogue/quant
kernels, so that comparison does not transfer. A bar at 10% is therefore
"clearly better than the reference's own honest number", which is the minimum
that justifies a new autograd path, a new artifact-compatibility invariant, and
the permanent maintenance of both.

Two designers were consulted independently and both expect this measurement to
fail. The measurement is taken anyway, because the decision is made by
measurement and not by prediction. **A "stop" verdict is a successful outcome
of this gate, not a failure of it.**

## If the build proceeds, shipping additionally requires

Recorded now so they cannot be negotiated later. These are **not** evaluated by
the pre-build measurement; they are the release conditions for code that does
not yet exist.

1. **Gradient correctness in the production dtype.** `grad_x` from any new
   autograd path matches the dequant-path autograd within bf16/fp16 tolerance
   **in the production dtype**, not in fp32 only. (Repo precedent: fp32-verified
   code has already shipped a crash onto the fp16 production path after a probe,
   a self-check and an audit all passed.)
2. **Quality, measured through the deployment path.** Held-out denoising loss
   through the **required W8A8 deployment path** within **1% relative** of the
   matched baseline, across **3 fixed seeds**, **plus** a blinded fixed-prompt
   visual check. Loss curves alone are explicitly insufficient — that is the
   gap in the reference, and G2's own history (an FP8 arm that passed on
   aggregate numbers and failed on flat-region mottle) is the local precedent.
3. **The base-function invariant, enforced automatically and in both
   directions.** An automated test must prove that a LoRA artifact records the
   forward mode it was trained under — *including the per-layer int8/e4m3
   selector decisions* — and that the inference loader enforces it both ways
   (a W8A8-trained artifact is refused on a dequant-path base, and vice versa).
   **If that coupling cannot be enforced, the feature does not ship regardless
   of speed. The invariant outranks the speedup.**

## Out of scope under this gate

- **Rotation** (Hadamard or otherwise) — out of scope.
- **QAT and full fine-tuning of a quantized base** — out of scope. (Full FT of
  a quantized checkpoint is separately *refused* today, because `Int8Linear`
  holds `weight` as a buffer; see `AnimaFullParameterAdapter`.)
- **Optimizer work** — no optimizer change is required or permitted under this
  gate.

## Measurement protocol this gate is to be evaluated with

Fixed here so the protocol cannot be chosen after seeing a number.

- Time the **existing** `Int8Linear` modules both ways — `_int_mm_forward`
  versus `_dequant_forward` — under `torch.no_grad()`, at the **real** layer
  shapes and token counts of Krea 2 and Anima, across the aspect-ratio buckets
  a real LoRA run uses (`training/bucketing.get_bucket_sizes`, divisibility 8,
  from real `base_resolutions`). Report per-shape **and** layer-count-weighted.
- Establish the **forward fraction of a real training step**. Prefer measuring
  it. If it must be derived from component timings instead, the projection is
  labelled **derived**, not measured, and the label travels with the number.
- Account explicitly for: the doubled forward under gradient checkpointing; the
  unchanged bf16 dgrad; layers the selector routes to e4m3 or leaves
  unquantized (they gain nothing); the `m > 16`, `k % 8`, `n % 8` and
  minimum-work gates, which exclude layers at *training* token counts too; and
  Triton autotune cost on first sight of each bucket.
- **GPU exclusivity** is a precondition, checked with the same logic
  `examples/api/bench_fp8_scaled_mm.py` uses (`C` versus `C+G` process types on
  WDDM, backend PID resolved from `backend/.port_info`). A foreign compute
  process means **stop and report**, not "measure anyway".
- **Power/clock state is recorded.** This card sits at a 240 W cap and idles at
  210 MHz; a previous measurement on it was invalidated by short loops that
  never left the idle clock. Warmup is on **wall time** and batch sizes are
  **calibrated**, per the method already used by `tmp/anima_int8_rollup_probe.py`.

## Result: G3 FAILED (gate closed, not reopened for a rerun)

This section records the measurement and the verdict. The rule above is
unedited from its pre-registered form; nothing below changes it.

### Synthetic projection (first pass)

Measured from safetensors headers plus synthetic tensors — no model was
loaded — under GPU exclusivity, peak 0.71 GB VRAM.

- Projected end-to-end step-time reduction: Krea 2 **32.4%**, Anima **19.5%**.
- Zero regressions across 103 gate-passing shapes.
- Autotune confirmed a non-issue: the kernels in this repo contain no
  `@triton.autotune` sweep; measured cost was 235 ms once per process and
  0.118 ms per newly-seen bucket thereafter.
- One input to the Anima number was **derived**, not measured: `f`, the
  forward-Linear share of a training step, derived at **43.9%** for Anima.
  Both criteria 1 and 3 appeared to pass on this pass.

### Real Anima training step (second pass, the one that decides the gate)

Measured against a real Anima training step, GPU exclusive, batch 1, at and
below 512px. Peak 3.18 GiB GPU / 0.98 GiB host.

- `f` measured **23.7%** at 512px against the real step, versus 43-48%
  against a "bare" step (the synthetic pass's proxy). The derivation method
  itself was not wrong; its denominator omitted the LoRA adapter GEMMs and
  the optimizer step. That omitted overhead is **+90-112%** at low
  resolution, but it does not scale with token count — the 341 rank-16
  adapter GEMM pairs are launch-bound and the optimizer step is
  token-independent — so the same omission falls to **+18.7%** at the
  1024px shipping default.
- Corrected Anima projection: **~15.2%** (down from the synthetic pass's
  19.5%). Criterion 1 (>=10% on both architectures) still passes on this
  corrected number; Krea 2's number is unaffected by this correction.
- **Criterion 2 was violated.** The W8A8 forward is slower than the dequant
  forward at low token counts. Measured forward-only A/B speedup (W8A8 vs
  dequant): **0.877x at 256px, 0.908x at 384px, 0.949x at 512px, 1.582x at
  1024px.** Projected effect on full step time: **-4.53% at 256px** (the
  criterion-2 floor is -3%) and **-1.75% at 512px**. At 512px, 223 of 231
  layers already pass the existing min-work admission gate, and the fused
  path is *still* slower there — at that token count the activation-quantize
  kernels cost more than the saved GEMM time, because the DiT forward is
  launch-bound rather than compute-bound.
- A hypothesised cheap partial win was tested and disproven. Anima runs
  gradient checkpointing with `torch.utils.checkpoint(..., use_reentrant=False)`
  (`backend/core/training/arch_handlers/anima_models.py:660-666`). Verified
  directly against `torch`: with `use_reentrant=False`, **both** the
  recompute pass and the original forward pass run with grad enabled, so a
  "skip W8A8 on the no-grad recompute pass" fast path does not exist for
  Anima under its current checkpoint mode — it was refused on all 12712
  measured calls, and its measured effect was **+0.73%**, i.e. noise, not a
  win. Switching Anima to `use_reentrant=True` to unlock that fast path is
  not a free flip: Anima's `x_embedder` is not a LoRA target module, so
  reentrant checkpointing would silently drop gradients through it, and even
  if that were fixed, the ceiling of the idea is 7.1%, below the criterion-2
  floor being violated at 256/512px regardless.

### The ruling

Criterion 1 (>=10% on Krea 2 and Anima) passes after correction. Criterion 2
(no tested workload regresses more than 3%) **fails**: 256px and 512px
regress the full step by more than the 3% floor. Per the rule as written,
failing any one of the three ALL-of criteria fails the gate; there is no
"proceeds ex-Anima-at-low-res" branch in the pre-registered rule.

Two designers were asked independently whether adding a resolution/token-
count admission condition — so the gate would apply only above the observed
crossover — is a legitimate extension of a rule that already has a
token-count term (`m*k*n`, where `m` is the token count) in its existing
min-work gates, or whether it is moving the goalposts after seeing the
result. **They disagreed**, and both readings are recorded here because the
disagreement exposes a genuine ambiguity in how the rule was written, not
because one side is obviously right:

- **Legitimate, conditionally.** The gate's own measurement protocol already
  required accounting for "the `m > 16`, `k % 8`, `n % 8` and minimum-work
  gates, which exclude layers at *training* token counts too" — so a
  token-count-aware admission rule is already inside the spirit of what was
  pre-registered. But this reading was conditioned on a rigorous derivation,
  not a fit to the failure: simply re-deriving the existing
  `_MIN_WORK_MKN` constant from the 256/512px failure would be curve-fitting
  the wrong model, because the cost that actually bites here is launch-bound
  activation-quantization overhead, which scales with `m*k`, not the
  `m*k*n` compute-volume term the current gate and its constant model.
- **Post hoc.** The gate as written already explicitly covered training
  token counts, the real aspect-ratio buckets, and the existing min-work
  selector. Recalibrating the admission threshold only after observing which
  resolutions failed changes the tested intervention from the one that was
  pre-registered; that is a different, unregistered gate wearing this one's
  name.

**The stricter reading was adopted.** G3 fails as written. A
resolution-floored (or otherwise token-count-admission-gated) variant of
this idea is a **new feature requiring a new pre-registered gate**, evaluated
against its own criteria and its own numbers before it is measured — not
this gate re-passing on a rerun with a moved goalpost.

### Prerequisites for any future proposal (not a plan — these must exist before a new gate is written)

- **Measure a real 1024px Anima training step.** The 1024px column above was
  spliced from a bare-step number plus the measured overhead correction, not
  measured end-to-end at 1024px. The same class of derivation was already
  wrong once, by 4.3 points (23.7% measured vs 43.9% derived `f`), and the
  margin between the corrected Anima projection (~15.2%) and the 10% bar is
  5.2 points — smaller than the error already observed once at a different
  resolution.
- Derive a **unified (m, k, n) admission selector from real training-time
  shapes, measured against the real training dequant path** — with
  calibration and holdout workloads separated *before* the holdout results
  are observed. Do not retrofit a nominal-resolution floor onto the existing
  inference-time constants.
- Any such selector must, without hand adjustment after the fact, refuse the
  256px shapes measured here and admit the 1024px shapes measured here. If
  the selector's natural crossover does not separate those two cases, there
  is no crossover to exploit and the low-resolution case dies honestly
  rather than being carved out by hand.
- Scope any new constants strictly to the training-time admission path. Do
  not retune the already-shipped inference gate (G2) to accommodate this.
