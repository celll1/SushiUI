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

---

# INT8/FP8 dequant-path autograd retention — measurement gate G4 (memory)

**Status: pre-registered, then measured; CLOSED, FAILED.** The rule below was
written before the `autograd.Function` it decides existed and before any number
under it was measured. It lives in this file because it is the same subject as
G3: G3 asked whether the quantized *forward* could be made faster in training
(closed, failed); G4 asks whether the quantized *dequant path* could be made to
retain less memory in training. G3's bullet above — "the memory win is already
ours" — is true only while gradient checkpointing is on, and G4 is what measured
the rest of that sentence. **G3's rule text is unchanged by G4.**

### The defect this gate decides a fix for

`_dequant_forward` (`models/ideogram4/vendor/int8_linear.py:662`,
`fp8_linear.py:631`) is

```python
w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
return F.linear(x, w, bias)
```

`F.linear` saves `w` for backward because `grad_input = grad_output @ w`. For a
bf16 `nn.Linear` the saved tensor is an **alias of the resident parameter** —
zero extra bytes. For a quantized Linear it is a **fresh (out, in) allocation in
the compute dtype**, on top of the resident 1-byte codes.

Per weight, per checkpoint unit: **int8 = 1 B resident + 2 B transient**, bf16 =
2 B + 0. Inside one live unit int8 is 1.5x *worse*; it wins today only because
per-block gradient checkpointing keeps one unit live at a time.

`gradient_checkpointing` defaults to `True` (`api/param_defaults.py`) but is
user-toggleable. With it **off**, every block's dequant temporary is live
simultaneously and the whole model materialises in the compute dtype on top of
the codes. Derived from safetensors headers (no model loaded), Krea 2's
transformer: **11.94 GiB of codes + 23.88 GiB of dequant temporaries = 35.81
GiB**, against a bf16 base of **23.88 GiB**. The quantized base is then strictly
and badly worse, silently.

### The intervention this gate is about

A `torch.autograd.Function` around the dequant path that **does not save the
dequantized weight**. It saves the int8/e4m3 codes and the scale — both live
module buffers, so saving them costs no new allocation — and rebuilds `w` in
backward.

### The rule — ship only if ALL of

#### B1. Forward stays bitwise identical. Non-negotiable.

Same standard `int8_fused.py` is held to: equality **on integer bit views**, not
`allclose`, over the dtype matrix {bf16, fp16, fp32} and over hostile inputs
(all-zero rows, denormals, a single huge outlier, NaN, +-Inf). Any single
differing bit fails the gate outright, independently of every number below.
This is not expected to cost anything — the forward *maths* is unchanged, only
what autograd retains — but it is proven, not assumed.

#### B2. Backward is correct in the production dtype, and emits no weight grad.

`grad_input` matches the current path within the production dtype's tolerance,
measured **in bf16 and fp16**, not fp32 only (repo precedent: fp32-verified code
has already shipped a crash onto the fp16 path after a probe, a self-check and
an audit all passed). The base is frozen and `weight` is a buffer, so a weight
gradient must **not** appear; a path that would need one must refuse rather than
silently drop it.

#### M1. Without gradient checkpointing, the retention must be O(1) in N.

On the synthetic probe (N Linears of 2048x2048 = 8 MiB/weight in one unit, LoRA
rank 16 on top, no checkpointing):

* `peak(int8, N=28) - peak(int8, N=4)` **<= 3 x weight_bytes (24 MiB)** beyond
  the bf16 arm's own N-growth. This is the actual claim — that the dequant
  temporary stops scaling with the number of layers.
* `peak(int8, N=28) < peak(bf16, N=28)`. The quantized arm must stop being a
  memory regression in this configuration; that is the trap being removed.

#### M2. With gradient checkpointing, the win must be at least the scaling law.

Same probe with `checkpoint(..., use_reentrant=False)`:

* peak reduction versus the current path **>= (N - 3) x weight_bytes** at N=28
  (**>= 200 MiB**), and
* `peak(int8) < peak(bf16)` still holds.

#### S1. Step-time ceiling — declared here, before measuring.

On the same synthetic (which is 100% Linear and launch-bound, i.e. the worst
case for an extra dequantize in backward):

* **<= +12%** with gradient checkpointing,
* **<= +30%** without.

A prototype measured +9.0% / +24.6%; the ceilings sit above those with margin
because the implementation being gated is not that prototype. **No real-step
cost may be extrapolated from this synthetic** — a real step contains attention,
norms, the LoRA adapter GEMMs and the optimizer step, which this probe does not.
If a real number is wanted later it must be measured, and this file will not be
edited to make one fit.

### Derived, not measured: what M1/M2 imply at real shapes

Labelled derived because no model was loaded; shapes come from safetensors
headers, quantized-byte totals from the runtime rollup already recorded in
`docs/guides/MODEL_FACTS.md`.

| | codes resident | dequant transient today | after | bf16 base |
|---|---|---|---|---|
| Krea 2, no checkpointing | 11.94 GiB | +23.88 GiB (**35.81 total**) | +<=0.38 GiB (**12.31**) | 23.88 |
| Krea 2, per-block checkpointing | 11.94 | +828 MiB / block | +<=384 MiB | 23.88 |
| Anima, no checkpointing | 2.33 GiB | +3.14 GiB (**5.46 total**) | +<=0.06 GiB (**2.39**) | 3.90 |
| Anima, per-block checkpointing | 2.33 | +120 MiB / DiT block | +<=64 MiB | 3.90 |

Anima's per-block win is small in absolute terms (10 quantized Linears per DiT
block, largest 32 MiB) and that is recorded now so it is not later presented as
more than it is. The load-bearing row is the no-checkpointing one.

### What is out of scope

* Changing the `gradient_checkpointing` default. Not this gate's subject.
* The W8A8 fast path in either module. G3 closed it for training; nothing here
  reopens it.
* Any accuracy claim about quantization itself. The forward is bitwise
  unchanged, so there is nothing new to claim.

### If a criterion fails

The fix does not ship, and `gradient_checkpointing: false` over a quantized base
gets a factual warning through the existing `add_warning` channel instead. If
the fix ships and removes the trap, the warning is unnecessary and must not be
added — a warning about a condition that no longer exists is noise.

---

### Result: G4 FAILED (S1). The autograd fix does not ship.

The rule above is unedited from its pre-registered form; nothing below changes
it. Five of the six criteria pass, one fails, and the rule is ALL-of.

Host: RTX 6000 Ada (sm_89), torch 2.10.0+cu130, bf16, no model loaded, peak
GPU allocation across every measurement below 0.5 GiB. Candidate implementation
kept at `tmp/dequant_linear_g4_candidate.py`; probes at
`tmp/dequant_{retention_probe,bitwise_verify,ab_repeat,overhead_attrib,cost_isolate,kernel_probe}.py`.

#### B1 — forward bitwise identity: **PASS**

576 forward comparisons on integer bit views (`.view(int16/int32)`, with NaN
placement compared separately so a NaN payload cannot hide a difference), over
{CPU, CUDA} x {int8 codes, e4m3 codes} x {bf16, fp16, fp32} x {bias, no bias} x
6 hostile input families (normal, all-zero rows, denormals + smallest-normal,
one huge outlier, NaN, +-Inf) x {grad enabled, `no_grad`}, with a zero scale and
a `float32.tiny` scale in every weight. **0 differing bits.**

#### B2 — backward correctness in the production dtype: **PASS**

`grad_input` came out **bitwise equal** to the shipped path — max |delta|
exactly 0.0 — in bf16, fp16 and fp32, on CPU and CUDA, for both code formats.
Stronger than the tolerance the gate asked for, because backward rebuilds `w`
from the identical expression the forward used and then runs the same `mm`.
No `.grad` appears on `weight`, `weight_scale` or `bias`; and when one of them
*does* require a gradient the dispatcher falls back to the eager path, which was
verified to produce that gradient rather than drop it.

#### M1 — no checkpointing, O(1) retention: **PASS**

28 Linears of 2048x2048 (8 MiB/weight bf16), 512 tokens, LoRA rank 16, peak by
`torch.cuda.max_memory_allocated`.

| arm | weights | peak N=4 | peak N=28 | growth |
|---|---|---|---|---|
| bf16 base + LoRA | 224.0 MiB | 78.8 MiB | 322.2 MiB | 243.4 MiB |
| int8, current | 112.2 | 86.8 | **426.4** | 339.6 |
| int8, candidate | 112.2 | 62.8 | **210.4** | 147.6 |

Excess growth over the bf16 arm: current **+96.2 MiB**, candidate **-95.8 MiB**
(below bf16's own growth, since int8 residency grows at 1 B/element). Ceiling
+24.0 MiB. `peak(candidate) 210.4 < peak(bf16) 322.2`, where the current path is
**426.4 > 322.2** — the trap, reproduced. **PASS.**

#### M2 — with checkpointing: **PASS**

Same probe under `checkpoint(..., use_reentrant=False)`.

| arm | peak |
|---|---|
| bf16 base + LoRA | 308.2 MiB |
| int8, current | 416.4 |
| int8, candidate | 202.5 |

Reduction **213.9 MiB** against a `(N-3) x 8 MiB = 200 MiB` floor; candidate
below the bf16 arm. **PASS.**

#### S1 — step time: **FAIL**

Interleaved A/B (`tmp/dequant_ab_repeat.py`), 5 rounds per arm, alternating
order, 1.5 s wall-time warmup and 40 timed steps per round.

**GPU exclusivity could not be established** for these runs, and it shows: the
`current` arm's per-round medians ranged 9.94-12.85 ms across four repeats of an
identical measurement, and one repeat returned -1.9% for a configuration that
returned +19.6% in another. The memory numbers above are unaffected (allocation
accounting is process-local and reproduced to the byte across every run); the
timing numbers are reported as a RANGE and the verdict is taken on the
least-contaminated estimator available, the minimum sample per arm per round.

| run | ckpt: current / candidate (median) | delta | ckpt: min / min | delta |
|---|---|---|---|---|
| A | 9.94 / 11.89 | +19.6% | 9.90 / 11.59 | **+17.1%** |
| B | 12.85 / 14.46 | +12.5% | 10.47 / 12.43 | **+18.7%** |
| C | 12.78 / 12.54 | -1.9% | 10.76 / 12.14 | **+12.8%** |
| D | 11.01 / 12.55 | +14.1% | 10.40 / 12.53 | **+20.5%** |

By minima, **every** repeat of the checkpointing configuration exceeds the +12%
ceiling (+12.8% to +20.5%); two earlier non-interleaved orderings gave +14.7%
and +14.8%. The no-checkpointing configuration measured +20.5%, +32.4%, +23.7%,
+25.4% by the same estimator — under its +30% ceiling in three of four.

The verdict rests on the checkpointing arm, where no estimator on any run puts
the candidate under the ceiling. **FAIL.** If the repo owner wants this decided
on an exclusive GPU rather than on this evidence, the protocol to re-run is the
minima column above; the pre-registered ceilings do not move either way.

Attributed (`tmp/dequant_overhead_attrib.py`): the Python `autograd.Function`
itself costs **-0.4%** with checkpointing (i.e. nothing) — the entire delta is
the one extra dequantize in backward, **1.58 ms** for 28 layers. Note this does
not reconcile with the isolated kernel probe below: 28 x 19.6 us is 0.55 ms, a
2.9x gap that was NOT explained. Candidates not separated here are the fresh
8 MiB allocation per call and a cold-cache backward `mm`. The gap matters
because 0.55 ms against the 10.4 ms baseline would be +5.3% and would PASS; the
wall-clock minima (+12.8% to +20.5%) are consistent with 1.58 ms and not with
0.55 ms, which is why the FAIL stands, but a re-run should close this. Two
implementation improvements were made before accepting the number, not after:
the dequantize was reduced from two kernels to one promoted multiply
(`codes * scale`, bitwise equal on every probed (device, code dtype, out dtype),
measured 19.6 us versus 32.3 us per 2048x2048 weight), which moved the
checkpointing delta from **+24.4%** to +14.7-19.6%. What was NOT done, and why:

* **Factorising the gradient** as `(grad_out * scale) @ codes.to(dtype)` avoids
  the (out, in) multiply entirely and measured ~40% cheaper, but it multiplies
  the scale (~1e-5) into the GRADIENT instead of into the weight. In fp16 that
  underflows: a 1e-4 gradient times a 1e-5 scale is 1e-9, below fp16's smallest
  denormal. Rejected on the production-dtype rule, not on speed.
* **Retaining `w` within a checkpoint unit** is exactly the retention M1/M2
  measure the removal of.
* **Inductor prologue fusion** would remove the 8 MiB round trip, but nothing in
  this repo compiles these layers and doing so is a different change.

#### The ruling

Criterion S1 fails on the gradient-checkpointing configuration, which is the
`gradient_checkpointing` default. The rule is ALL-of, so **G4 fails and the
candidate does not ship**, exactly as G3 failed on one criterion of three.

The temptation here has the same shape as G3's: every criterion passes in the
configuration the defect actually bites (checkpointing OFF — M1 pass, S1 pass at
+20.2% against a +30% ceiling), and only the *other* configuration's step time
fails. Scoping the intervention to "rebuild in backward only when the owner has
disabled gradient checkpointing" would pass every number in this document. That
is precisely the move G3's ruling refused: *"a resolution-floored (or otherwise
admission-gated) variant of this idea is a new feature requiring a new
pre-registered gate ... not this gate re-passing on a rerun with a moved
goalpost."* The scoped variant is a different intervention, its numbers are
already known here, and a rule written after seeing them is not pre-registered.
It is therefore **not** taken in this pass.

#### What shipped instead — the pre-registered failure branch

`adapters/base_adapter.warn_quantized_base_without_checkpointing`, called once
from `base_trainer.train()` where both facts are known, prints a factual line to
the training log (the channel a training run reaches the user through) and
offers the same text to `api.generation_status.add_warning` best-effort, the way
`int8_linear._report_int_mm_fallback` does. It states the layer count, the
measured bytes-per-element on both sides, the measured 426.4 / 322.2 MiB
synthetic peaks and the derived 35.81 / 23.88 GiB Krea 2 figures. No adjectives,
no unmeasured claim.

`backend/tests/quantized_base_checkpointing_warning_test.py` pins the warning's
trigger condition on both quantized classes AND the retention fact it asserts
(via `saved_tensors_hooks`: a quantized Linear saves one fresh compute-dtype
weight per forward and N of them for N layers in a live unit; an unquantized
frozen Linear saves an alias of its own parameter). If the retention ever stops
happening — a future gate landing the candidate, or a torch change — that test
fails and sends whoever changed it here to delete the warning.

#### Block swap, re-checked against `8244c509`

The candidate saves the CODE buffers for backward, so a swap that overwrote a
block's weight storage in place between forward and backward would corrupt its
gradients. Checked, not assumed: training swaps through
`memory_management/layer_offload_conductor.py`, which iterates
`named_parameters()` (the quantized weight/scale are BUFFERS, so it never
touches them) and moves a layer with `layer.to(...)`, which REPLACES a buffer
rather than writing into it — a saved reference therefore stays valid and simply
keeps that block's codes resident at 1 B/element until backward, where today's
saved `w` already keeps 2 B/element resident in the same situation. The
dtype-aware pairing fix in `8244c509` is confined to the two INFERENCE
offloaders (`block_offloading.py`, `flux_block_offloading.py`), where no
gradient exists and nothing is saved, so it neither helps nor threatens this.

#### Not measured, and deliberately so

* **No real Anima or Krea 2 step.** Real-machine verification was deferred for
  this pass; every number here is synthetic or header-derived and labelled.
  The real-step cost of the candidate is therefore **unknown**, and the S1
  percentages must not be read as real-step percentages — a real step contains
  attention, norms, adapter GEMMs and the optimizer step that this probe does
  not. The synthetic is the worst case for the candidate, so a real step would
  cost relatively less; how much less is not derivable from anything here.
* **The e4m3 arm's memory was not measured**, only its bitwise/gradient
  behaviour. The retention asymmetry is identical by construction (1-byte codes,
  compute-dtype `w`) and the regression test pins it on `Fp8Linear` too, but no
  peak-memory number is claimed for it.
* **The promoted-multiply dequantize did not ship with the candidate.** It is
  bitwise equal and measured 1.6x cheaper, but it arrived inside the candidate
  and has no gate of its own; landing it separately is a small, self-contained
  follow-up.

  **Follow-up landed (int8 only).** `Int8Linear._dequant_forward` now spells the
  definition as `codes * scale[:, None]`, letting integer/float promotion do the
  widening in one kernel. It needed a *test*, not a gate: bitwise identity is
  not a trade-off to be adjudicated, and it holds for a reason rather than by
  luck — every int8 code is exactly representable in bf16/fp16/fp32, so the
  widening rounds nothing in either spelling. Re-derived independently before
  shipping: 336 comparisons on integer bit views over {CPU, CUDA} x {bf16, fp16,
  fp32} x {bias, no bias} x hostile codes/scales/activations, **0 differing
  bits**, and `backend/tests/quantized_dequant_bitwise_test.py` pins it (it was
  mutation-checked against two plausible non-equal rewrites and failed on both).
  Re-measured interleaved A/B minima on sm_89, torch 2.10: the dequantize alone
  1.20-1.99x (20.4 -> 15.1 us at 2048x2048 bf16), the whole `_dequant_forward`
  1.22-1.23x at 1-64 tokens and 1.00-1.03x once the GEMM dominates at 4096
  tokens — free, never negative. **`Fp8Linear` deliberately did not take it:**
  `float8_e4m3fn` has no promoting multiply at all (RuntimeError on CPU and CUDA,
  all three compute dtypes), so folding its cast would raise, not accelerate.

---

# Fused ConvRot W8A8 forward for a frozen base in training — measurement gate G5 (pre-registered)

**Status: pre-registered, 2026-08-26. Written before the measurement it decides.**
No config key and no artifact-metadata field for this path exists. **Nothing
under this gate has shipped and no default has changed**; a candidate autograd
path may sit in the working tree behind the opt-in
`SUSHI_CONVROT_TRAIN_FUSED` environment flag, default off, whose existence
decides nothing here — this gate is decided by the measurement below and not by
the candidate's own numbers.

It lives in this file for the same reason G4 does: same subject. G3 asked
whether an unrotated INT8 W8A8 forward could be made faster in training (closed,
FAILED). G4 asked whether the dequant path could retain less memory in training
(closed, FAILED). G3 put **rotation explicitly out of scope**, so neither gate
decides the rotated case; G5 is that case, and it is a different intervention
because it changes the forward and the retention at the same time. **G3's and
G4's rule texts are unchanged by G5.** The design being decided is
`docs/guides/INT8_CONVROT_TRAINING_DESIGN.md`.

## What decides G5, and what does not

G5 is decided by a **real training step on a real ConvRot checkpoint**:

* **SenseNova is mandatory.** The subject is `ConvRotInt8Linear`
  (`backend/core/models/common/convrot_int8_linear.py`) as loaded by
  `training/ops/sensenova_ops.py` over the released
  `sensenova_int8_convrot.safetensors` base, under the LoRA route that already
  runs today.
* **MiniMax-H3 is mandatory.** Reachable means a ConvRot (or W4A8) base whose
  Linears sit on the **differentiable** path of a real MiniMax-H3 training step.
  **The determination is written here before any deciding number is taken, and
  it is: reachable.** `models/minimax_h3/loader.py::_load_transformer` swaps both
  `ConvRotInt8Linear` and `W4A8Linear` into the **DiT** from the `int8_convrot`
  and `w4a8_mixed` DiT files, both of which exist locally for both variants
  (`M:/model/minimax_h3/diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_{int8_convrot,w4a8_mixed}.safetensors`);
  `training/ops/minimax_h3_ops.load_components` freezes that DiT and trains LoRA
  over it, so those Linears are frozen and differentiable — exactly this gate's
  subject. An earlier reading that H3's only ConvRot file was the
  `qwen3vl_32b_minimax_h3_int8_convrot` text encoder was wrong; it holds only for
  the `fp8_scaled` DiT, which is a different file and is not a G5 workload. The
  text encoder itself is **not** a G5 workload for the opposite reason: its
  forward runs under `@torch.no_grad()`, so its Linears already take the fused
  inference kernel and never reach the path this gate decides.

**Nobody has measured a real ConvRot training step.** That is what makes G5
pre-registrable today.

### The 2026-08-26 numbers are inputs and priors, not the verdict

`tmp/CONVROT_MEASURED_EVIDENCE.md`; probes `tmp/convrot_probe{1,2,3}.py`. Every
step-level number there is **synthetic**: no model was loaded, weight shapes come
from the checkpoint header and four real weight tensors were read for the
accuracy arm only. The synthetic step has **no prefix pass, no MoT mask, no RoPE
and no data pipeline**, and its step time sits on a **CPU-dispatch-bound floor**
(arm B's step time is flat 231.69 -> 227.32 ms from 64 to 256 tokens while doing
29% less GPU work at 64 tokens). **That floor is exactly the quantity a real step
changes**, in either direction: a real step adds host work of its own and adds
GPU work of its own. G3's history is the reason this is written down — a derived
step-share was wrong by 4.3 points (43.9% derived vs 23.7% measured), and the
error was in the same class of extrapolation.

Carried forward as priors, labelled:

* **synthetic, per layer:** fused forward 1.06x-6.54x over the dequant forward at
  the real ConvRot shapes; backward 0.28x-0.93x (the added dequantization).
* **synthetic, whole step, checkpointing ON:** -15.4% at 64 image tokens,
  +18.9% at 256, +25.7% at 1024.
* **synthetic, whole step, checkpointing OFF:** -26.0% at 64 tokens, +17.1% at
  256; peak 23.22 -> 8.18 GiB at 64 tokens (bf16-equivalent for the same 294
  weights: 15.83 GiB).
* **measured on real weights (not a step):** `grad_x` bitwise equal to the
  current path at bf16, m in {64, 1024}; forward relative difference
  0.98e-2-1.07e-2 between the fused and dequant arms.

### The known crossover, recorded before the gate is evaluated

The synthetic **regresses at 64 image tokens (SenseNova 256 px)** and wins from
**256 tokens (512 px)** upward. The 64-token bucket is therefore a **required
tested workload** under criterion 2, and G5 as written can fail on it. A gate
that cannot fail at 256 px is not a gate.

## The rule

### Build proceeds only if ALL of:

1. **Measured end-to-end step-time reduction >= 10%** versus the current
   dequant-path training, on the mandatory architecture set defined above, at
   the shipping-default resolution of each. The number is taken from a **real
   training step**, not projected from per-layer timings; if any input to it is
   derived rather than measured, the projection is labelled **derived** and the
   label travels with the number. The measurement must include gradient-
   checkpoint recompute (the forward runs twice per step) and must be replicated
   on at least two configurations that both contain a frozen ConvRot half.
2. **No tested workload regresses by more than 3%** in end-to-end step time.
   The tested set must include the SenseNova **64-image-token (256 px)** bucket
   and the 256-token (512 px) bucket, gradient checkpointing **on and off** where
   the architecture permits, and cold as well as warm process state.
3. The measurement **must not rely on a warm state a real run would not have**,
   and must not be taken with a foreign compute process on the GPU.

### Alternative sufficient condition (OOM removal)

The fused path **enables a real configuration that cannot run today at all** —
demonstrated by a real training run of that configuration OOM-ing on the current
path and completing steps on the new path, same machine, same config, same seed,
same data. This is sufficient on its own, independent of the 10% bar.

**A synthetic peak does not satisfy this condition.** The 23.22 -> 8.18 GiB
figure above is a prior that says where to look — SenseNova with
`gradient_checkpointing: false`, where the dequantized weights of 294 live
Linears are retained simultaneously and the quantized base costs more than its
15.83 GiB bf16 equivalent — and nothing more. The condition is decided by
whether a configuration that cannot run does run.

### No admission rule is registered

G5 registers **no token-count, resolution or `(m, k, n)` admission condition**.
Criterion 2 applies to every tested workload uniformly. G3's ruling refused
rescuing a failing gate by scoping it to the configuration where it passes, and
the crossover here is known **before** the deciding numbers exist, so carving
the 64-token bucket out of G5 would be the same move made earlier.

If a later proposal wants an admission rule, it is a **new gate**, and it may be
written only under these conditions, registered here now:

* the selector is **derived from real training-time shapes measured against the
  real training dequant path**, not a nominal-resolution floor and not a retune
  of the shipped inference constants;
* its **calibration and holdout shapes are separated and named before any
  holdout number is observed**;
* it models the cost that actually binds. At 64 tokens that is **host dispatch**,
  not the `m*k*n` compute volume the existing inference gates model; a selector
  fitted to `m*k*n` would be curve-fitting the wrong model, the same error G3's
  ruling identified;
* without hand adjustment after the fact, it refuses the 64-token SenseNova
  shapes and admits the 1024-token shapes. If its natural crossover does not
  separate those two cases, there is no crossover to exploit.

### Rationale for the numbers, recorded now

The 10% bar and the 3% floor are **inherited unchanged from G3**, deliberately,
so that they are not numbers chosen to fit the 2026-08-26 synthetic. Their
justification is G3's and is unchanged: 10% is the minimum that justifies a new
autograd path, a new artifact-compatibility invariant, and the permanent
maintenance of both; 3% is the regression a supported workload may absorb
without the change being a trade rather than an improvement. The synthetic's own
numbers (-15.4% at 64 tokens, +25.7% at 1024) sit on both sides of both
thresholds, which is the intended property: the thresholds do not select an
outcome.

**A "stop" verdict is a successful outcome of this gate, not a failure of it.**

## If the build proceeds, shipping additionally requires

Recorded now so they cannot be negotiated later. These are **not** evaluated by
the pre-build measurement; they are release conditions for code that does not
yet exist. **They outrank speed:** a passing step-time number does not ship a
path that fails any of them.

1. **Gradient correctness in the production dtype — bf16 AND fp16, not fp32
   only.** `grad_x` matches the dequant-path autograd within the production
   dtype's tolerance in **bf16 and fp16**, with fp32 as an oracle. Repo
   precedent: fp32-verified code has already shipped a crash onto the fp16
   production path after a probe, a self-check and an audit all passed. The
   2026-08-26 bitwise `grad_x` result is **bf16 only, on a synthetic stack**, and
   does not discharge this condition. No weight, scale or bias gradient may
   appear; a configuration that would need one must refuse rather than silently
   drop it.
2. **Quality, measured through the deployment path.** Held-out denoising loss
   through the **required W8A8 ConvRot deployment path** within **1% relative**
   of the matched baseline, across **3 fixed seeds**, **plus** a blinded
   fixed-prompt visual check. Loss curves alone are insufficient (G2's local
   precedent: an FP8 arm passed on aggregate numbers and failed on flat-region
   mottle). The measured 0.98e-2-1.07e-2 forward difference between the arms is
   the size of the function change this condition is testing, not evidence about
   its effect.
3. **The base-function / artifact invariant of design doc §5, enforced
   automatically and in both directions.** An artifact trained under the fused
   mode records the mode, ConvRot group size, backward dtype and a canonical
   hash of the ordered ConvRot layer manifest; the inference loader refuses that
   artifact on a plain-INT8, FP8, bf16, dequant-forced, different-manifest or
   incompatible-ConvRot base, and refuses a dequant-trained artifact on a
   fused base. **This invariant is NOT implemented today** — no
   `convrot_w8a8_ste_v1` marker, no base-forward-mode field and no such loader
   check exists anywhere in the repository (verified 2026-08-26 by
   repository-wide search). If the coupling cannot be enforced automatically,
   the feature does not ship regardless of speed.
4. **No silent base-function switch mid-run.** A kernel exception must fail the
   run with the layer path and reason. Training one artifact against two
   different base functions is a correctness failure, not a fallback.

## Out of scope under this gate

* **Trainable ConvRot weights / QAT.** Refused by design doc §4.2 and
  corroborated by the earlier int8-resident full-FT investigation. Not reopened
  here.
* **G3 and G4.** Both are closed. Nothing in G5 reopens either, and nothing
  measured for ConvRot transfers to plain `Int8Linear`/`Fp8Linear`: their
  dequant path is a single promoted multiply, while ConvRot's includes an
  inverse Hadamard, which is a large part of why the ConvRot dequant arm is as
  expensive as it is.
* **Changing the `gradient_checkpointing` default.**
* **Activation fusion (`input_act`), `torch.compile` coverage, and optimizer
  changes.** None is required or permitted under this gate.

## Measurement protocol this gate is to be evaluated with

Fixed here so the protocol cannot be chosen after seeing a number.

* Time a **real training step** on the real checkpoint, both arms, with the same
  data order, batch size, optimizer and seed. Report **measured**, not derived,
  step time; if the forward share of the step is needed at all, measure it.
* Buckets: the SenseNova resolutions that map to 64, 256 and 1024 image tokens
  at minimum, from the resolution path a real run uses.
* Gradient checkpointing **on and off**, where the configuration can run at all
  in both states. If an arm cannot run in a state, that is an OOM observation
  under the alternative sufficient condition, not a missing cell.
* Report **peak allocated and reserved VRAM** per arm per bucket, and the bf16
  equivalent of the same weights.
* **Interleaved A/B repeats, verdict on the minimum sample per arm per round** —
  the estimator G4 adopted after non-exclusive runs produced a -1.9% and a
  +19.6% for the same configuration.
* **GPU exclusivity is a precondition**, checked with the logic
  `examples/api/bench_fp8_scaled_mm.py` uses (`C` vs `C+G` process types on WDDM,
  backend PID from `backend/.port_info`). A foreign compute process means **stop
  and report**, not "measure anyway".
* **Power/clock state recorded.** This card sits at a 240 W cap and idles at
  210 MHz; warmup is on wall time and batch sizes are calibrated.
* **Host RAM peak is budgeted and announced before the run**, since this gate
  loads a 17.58 GiB checkpoint — unlike every measurement recorded above it,
  which loaded no model.
