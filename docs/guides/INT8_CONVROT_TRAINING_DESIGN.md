# INT8 ConvRot mixed-precision training design

**Status: design, plus a synthetic measurement pass taken 2026-08-26. Nothing has
shipped and no default has changed.** No configuration key and no artifact field
exists; a candidate frozen-base autograd path sits behind the opt-in
`SUSHI_CONVROT_TRAIN_FUSED` environment flag, default off. Nothing it enables is
reachable without that flag: with the flag unset the enabling helper is never
called, every module keeps the class-default `_frozen_training_fused = False`,
and both quantized Linears take the same dequant forward they took before.
The 2026-08-26 measurement was taken on separate probe code, not on that
candidate and not on the repository's training path — see §3. The decision rule
is pre-registered as **gate G5** in
`backend/core/training/INT8_W8A8_TRAINING_GATE.md`; G5 is decided by a real
training step, which has not been taken.

This document evaluates using the inference-only INT8 ConvRot W8A8 kernel in
training while computing backward in bf16/fp16/fp32. The verdict is:

- **LoRA over a frozen ConvRot base: viable with measurement and quality gates.**
- **A frozen ConvRot subgraph beside a floating-point full fine-tune: viable.**
  SenseNova training of only one MoT half is the concrete first candidate.
- **Updating a weight while keeping that weight itself ConvRot INT8: out of
  scope.** That is QAT with a floating master weight and a refreshed quantized
  shadow, not a small mixed-precision extension.

The recommended implementation is therefore a gradient-capable wrapper for a
**frozen** `ConvRotInt8Linear`: fused W8A8 in forward, dequantized floating-point
`grad_input` in backward, and no weight or scale gradient.

## 1. Current implementation

### 1.1 Inference

`backend/core/models/common/convrot_int8_linear.py` stores each Linear as:

- an `int8` weight containing the rows of `W @ H.T`, grouped at K=256;
- an fp32 per-output-row scale;
- a validated `.comfy_quant` marker; and
- an optional floating bias.

With gradients disabled, `ConvRotInt8Linear.forward` calls
`comfy_kitchen.int8_linear(..., convrot=True)`. The kernel rotates and
dynamically quantizes the activation, performs INT8 GEMM, and returns an output
in the activation dtype. The installed `comfy-kitchen==0.2.28` also exposes
`dequantize_int8_convrot_weight_dtype`, which dequantizes the stored codes and
applies the inverse Hadamard rotation into bf16, fp16, or fp32.

This is post-training quantization. ConvRot's group-wise regular Hadamard
rotation suppresses weight and activation outliers before quantization; it does
not require retraining the base.

### 1.2 LoRA training works today, but does not consistently use the fused forward

`ConvRotInt8Linear.forward` currently selects `_dequant_forward` whenever grad
mode is enabled **and** the input requires a gradient. `_dequant_forward`
restores the weight to its original basis and calls `F.linear`. A layer before
the first trainable contribution can still receive `requires_grad=False` and
therefore use the fused kernel; after a LoRA contribution makes the activation
differentiable, downstream ConvRot Linears use the dequant path.

The shared `LoRALinearLayer` computes:

```text
y = base(x) + scale * up(down(x))
```

and freezes `base`. `is_lora_wrappable_linear` accepts `Int8Linear`, and
therefore its `ConvRotInt8Linear` subclass. SenseNova explicitly accepts a pure
588-layer ConvRot base for LoRA, and MiniMax-H3's target enumerator uses the
same shared predicate. Consequently:

- the INT8 codes and scales remain frozen buffers;
- only the floating LoRA matrices are optimizer parameters; and
- gradients propagate through the dequantized base matmul to earlier adapters.

So **LoRA on a ConvRot checkpoint is already structurally and numerically
valid**. Most differentiable downstream base Linears run as W8A16 rather than
W8A8: weights remain stored in INT8, but the Linear computation and its
activations are floating point. The small prefix of the graph before an input
requires gradients can already run W8A8.

There is also a known train/deploy skew after the graph becomes differentiable.
Training fits the adapter through dequantized downstream base functions, while
generation from the ConvRot checkpoint uses activation rotation and dynamic
INT8 quantization at every matching Linear.

That skew now has a size. **Measured on four real ConvRot weight tensors read
from `sensenova_int8_convrot.safetensors`** (probe 1), the relative forward
difference between the fused W8A8 output and the dequantized W8A16 output is
**‖B−A‖/‖A‖ = 0.98e-2 … 1.07e-2**, stable across m ∈ {64, 256, 1024} and all
four ConvRot shapes. That is the per-layer size of the function difference
between what training fits today and what generation runs. Its **effect** on a
trained adapter is still unmeasured end to end; the number bounds the input to
that question, not its answer.

### 1.3 Full fine-tuning is intentionally different

The ordinary quantized Linear stores `weight` as a buffer, so generic full FT
rejects it: `requires_grad_(True)` cannot make it trainable and the optimizer
would silently omit it. SenseNova is the exception with an explicit
materialization route. It replaces the selected plain-INT8 MoT half with real
bf16 `nn.Linear` parameters before training.

SenseNova currently refuses a ConvRot base for full FT. The stated reason is
that the Hadamard rotation must be inverted. That inversion is now technically
available through the same comfy-kitchen dequantization op already used by
`ConvRotInt8Linear._dequant_forward`; the refusal should therefore be read as a
still-valid **unimplemented and unmeasured format boundary**, not as a
mathematical impossibility.

## 2. Proposed frozen-base autograd function

### 2.1 Forward

For a frozen `ConvRotInt8Linear`, call the existing fused kernel even when
autograd is active:

```text
y = int8_linear(x, q(W H^T), scale, convrot=True)
```

The comfy-kitchen custom op has no registered backward in the pinned runtime,
so it cannot be exposed directly to autograd. A SushiUI-owned
`torch.autograd.Function` must define the boundary. Its `forward` calls the
existing inference op and saves references to the INT8 code and scale buffers,
plus dtype, shape, group size, and bias presence. It does **not** save a
dequantized weight or the input activation.

Saving a module buffer through `save_for_backward` does not copy its storage.
Mutation must remain forbidden between forward and backward; these buffers are
frozen today, which satisfies that contract.

### 2.2 Backward

For output gradient `g`, rebuild the effective floating weight in the chosen
backward dtype and compute:

```text
W_dq   = dequantize_int8_convrot_weight_dtype(qweight, scale, 256, dtype)
grad_x = g @ W_dq
```

Return no gradient for weight, scale, bias, group size, or mode arguments.
Bias is frozen, and `grad_x` does not depend on it.

The backward is a straight-through approximation to the fused W8A8 function:
rounding the rotated activation is piecewise constant, so its literal
derivative would be zero almost everywhere and is not useful for training.
Using `W_dq` treats activation quantization as the identity in backward. This
is not needed for a LoRA layer's own `down`/`up` parameter gradients, which come
from the parallel floating branch, but it is needed for gradients reaching
earlier adapters through later frozen base Linears.

**Measured, and stronger than "straight-through" implies for the gradient
itself**: `grad_x` produced by this backward is **bitwise equal** to what
autograd produces through the current `_dequant_forward` — max |Δ| exactly 0.0,
`torch.equal` True — at m ∈ {64, 1024}, bf16, on real ConvRot weights (probe 1).
This holds by construction: both paths compute `g @ W_dq` from the same
dequantization, so the change is gradient-identical to today's path rather than
merely within tolerance. The straight-through framing describes the relationship
to the *fused forward's* true derivative, which is a separate statement; against
the gradient the repository actually computes today, there is no approximation.
This was measured in **bf16 only, on probe code**; fp16 is a release condition
under G5, not a result.

The backward dtype should initially be **the incoming activation/gradient
dtype**, not a separate user setting. This preserves the current autocast and
MiniMax-H3 no-autocast policies and avoids an extra persistent master copy.
Production support is bf16 first; fp16 and fp32 are acceptance-test matrix
entries, not necessarily initial product modes.

### 2.3 Dispatch rule

The fused training path may run only when all of these are true:

1. the exact module type is `ConvRotInt8Linear`;
2. its weight, scale, and bias are frozen;
3. K is divisible by 256 and the comfy-kitchen runtime contract validates;
4. the activation is CUDA/ROCm-resident in a supported floating dtype; and
5. the owner explicitly selected the measured training mode.

Unsupported shapes or runtimes must use the current dequant path. A kernel
exception during an active training run must not silently switch the base
function for later steps; fail the run with the layer path and reason. A
silent fallback would train one artifact against two different base functions.

When `x.requires_grad` is false, the custom function need not build a backward
edge. The LoRA branch still owns its parameter gradients. Once an earlier LoRA
contribution makes `x` differentiable, the wrapper produces `grad_x` as above.

## 3. Measured effect (synthetic, 2026-08-26) and what is still unmeasured

Full record: `tmp/CONVROT_MEASURED_EVIDENCE.md`. Probes:
`tmp/convrot_probe1.py` (correctness + per-shape timing),
`tmp/convrot_probe2.py` (whole synthetic step, checkpointing on),
`tmp/convrot_probe3.py` (checkpointing off + CPU/GPU attribution).

Host: RTX 6000 Ada (sm_89, 240 W cap), torch 2.10.0+cu130, comfy-kitchen 0.2.28,
bf16, GPU exclusive, timings as minima over interleaved repeats.

**Label that governs every number in this section.** Two things were measured on
real data: the ConvRot layer census and the four weight tensors used for the
accuracy arm, read from `sensenova_int8_convrot.safetensors` (17.58 GiB, 588
`.comfy_quant` markers). **Everything else is synthetic — no model was loaded.**
The synthetic step has no prefix pass, no MoT mask, no RoPE and no data
pipeline. **No real SenseNova or MiniMax-H3 training step has been measured**,
and G5 is decided by one.

Real census (measured): 588 ConvRot Linears, 294 in the generation half —
168×(12288, 4096), 84×(4096, 12288), 168×(4096, 4096), 168×(1024, 4096) over
both halves.

### 3.1 Step time

Per frozen Linear under non-reentrant gradient checkpointing:

```text
current:   2 * (dequant + bf16_forward) + bf16_dgrad
proposed:  2 * convrot_int8_forward    + dequant + bf16_dgrad
```

**Per-shape, synthetic activations at real shapes, bf16, grad enabled** (ms,
minima; forward ratio A/B and backward ratio A/B):

| shape | m | fwd A | fwd B | fwd × | bwd A | bwd B | bwd × |
|---|---|---|---|---|---|---|---|
| (12288, 4096) | 64 | 0.3669 | 0.0642 | 5.72 | 0.0968 | 0.3440 | 0.28 |
| (12288, 4096) | 256 | 0.6042 | 0.0924 | 6.54 | 0.3938 | 0.6747 | 0.58 |
| (12288, 4096) | 1024 | 1.6362 | 0.3572 | 4.58 | 1.3693 | 1.7218 | 0.80 |
| (12288, 4096) | 4096 | 5.6268 | 1.7785 | 3.16 | 5.3178 | 5.6924 | 0.93 |
| (4096, 12288) | 64 | 0.3853 | 0.0632 | 6.10 | 0.1182 | 0.3814 | 0.31 |
| (4096, 12288) | 1024 | 1.7994 | 0.4133 | 4.35 | 1.4773 | 1.7826 | 0.83 |
| (4096, 4096) | 64 | 0.0752 | 0.0640 | 1.17 | 0.0565 | 0.1619 | 0.35 |
| (4096, 4096) | 1024 | 0.4939 | 0.1331 | 3.71 | 0.4184 | 0.5003 | 0.84 |
| (1024, 4096) | 64 | 0.0621 | 0.0586 | 1.06 | 0.0514 | 0.1356 | 0.38 |
| (1024, 4096) | 1024 | 0.0884 | 0.0641 | 1.38 | 0.0891 | 0.1647 | 0.54 |

**Linear-only rollup, synthetic**, 294-layer generation half, 2 forwards + 1
backward per step:

| m (image tokens) | A | B | ratio |
|---|---|---|---|
| 64 (≈256 px) | 139.23 ms | 106.58 ms | 1.306× |
| 256 (≈512 px) | 251.45 | 157.26 | 1.599× |
| 400 | 353.01 | 207.58 | 1.701× |
| 1024 (≈1024 px) | 743.56 | 403.23 | 1.844× |
| 4096 | 2671.38 | 1532.81 | 1.743× |

**Whole synthetic step** (42 blocks: RMSNorm, GQA SDPA, LoRA r16 on q/k/v/o,
AdamW), gradient checkpointing ON:

| m | A step | B step | Δ time |
|---|---|---|---|
| 64 | 200.75 ms | 231.69 ms | −15.4% (B slower) |
| 256 | 280.40 | 227.32 | +18.9% |
| 1024 | 1126.30 | 836.90 | +25.7% |

Gradient checkpointing OFF:

| m | A step | B step | Δ time |
|---|---|---|---|
| 64 | 121.05 ms | 152.52 ms | −26.0% (B slower) |
| 256 | 187.01 | 155.01 | +17.1% |

**There is a crossover**: the synthetic step regresses at 64 image tokens
(SenseNova 256 px) and improves from 256 tokens (512 px) upward.

#### The finding that changes this section's own model: a host-dispatch floor

The model above — "it wins only when twice the forward saving exceeds the
backward dequantization cost" — is a **GPU-time model, and it is incomplete**.
At low token counts the binding constraint is **host dispatch**, not GPU work.

`torch.profiler` over the same stack, checkpointing ON (the profiler inflates
wall time; the arms are read against each other):

| m | arm | wall | summed CUDA kernel time | GPU busy |
|---|---|---|---|---|
| 64 | A | 203.56 ms | 238.73 ms | 117% |
| 64 | B | 263.46 | 170.19 | 65% |
| 256 | A | 272.18 | 437.61 | 161% |
| 256 | B | 257.76 | 235.93 | 92% |
| 1024 | A | 1076.57 | 2126.57 | 198% |
| 1024 | B | 788.79 | 1539.80 | 195% |

**Arm B does 29% less GPU work at 64 tokens and is still 15% slower on the whole
step.** Its step time is flat from 64 to 256 tokens (231.69 → 227.32 ms), which
is the signature of a dispatch-bound floor: B issues strictly more launches per
Linear (fused rotate + quantize + GEMM + epilogue in both forward passes, plus a
dequantize node in backward) and one Python `autograd.Function` call per Linear
across 294 layers. A GPU-time accounting of forward saving versus backward
dequantization cost predicts a win at 64 tokens; the measurement is a loss.

This is G3's failure mode recurring for a different reason. G3 regressed at low
token counts because GPU-side activation-quantization kernels cost more than the
GEMM time they saved. Here the GPU side wins and the **host** side is the floor.
Any future admission rule must model that cost, which scales with launch count
per layer, not with `m*k*n` — this is pre-registered in G5.

Consequences for this design: the launch count per Linear is a design variable,
not an implementation detail. It is not known how much of the floor a real
SenseNova step would hide behind its own host work, and that is the single
largest uncertainty between these numbers and G5's verdict.

Nothing here transfers to plain `Int8Linear`/`Fp8Linear`. Their dequant path is
a single promoted multiply; ConvRot's includes an inverse Hadamard, which is a
large part of why arm A is as expensive as it is at these shapes. G3 and G4
remain closed.

### 3.2 VRAM

Resident weight memory is unchanged: the current and proposed LoRA paths both
keep one-byte INT8 codes plus scales.

The difference is autograd retention. Today `F.linear` saves the newly
dequantized floating weight for `grad_input`. The proposed function saves only
aliases to resident INT8 buffers and creates `W_dq` transiently in backward.
Therefore:

- without gradient checkpointing, retained dequantized weights stop scaling
  with the number of live Linears;
- with per-block checkpointing, the peak saving is bounded by the dequantized
  weights live in the checkpoint unit; and
- activations, LoRA parameters, gradients, and optimizer state are unchanged.

**Measured on the synthetic 42-block stack at real SenseNova shapes** (peak
allocated):

| checkpointing | m | A peak | B peak | Δ |
|---|---|---|---|---|
| ON | 64 | 8.08 GiB | 7.81 GiB | −3.3% (−0.27 GiB) |
| ON | 256 | 8.17 | 7.90 | −3.3% |
| ON | 1024 | 8.68 | 8.56 | −1.4% |
| OFF | 64 | 23.22 | 8.18 | −64.8% (−15.0 GiB) |
| OFF | 256 | 24.67 | 9.66 | −60.8% (−15.0 GiB) |

The same 294 weights in bf16 are **15.83 GiB**. With checkpointing off, arm A at
23.22 GiB is **above** the unquantized base while arm B at 8.18 GiB is **below**
it: the G4 trap reproduced at real ConvRot shapes, with a fused forward on top.
With per-block checkpointing the saving is ~3%, inside the bound stated above.

These are synthetic peaks. Under G5 they are a prior indicating where to look —
SenseNova with `gradient_checkpointing: false` — and the OOM-removal condition
is decided by whether a real configuration that cannot run today does run, not
by a peak on probe code.

The G4 prototype proved this retention mechanism can reduce memory, but failed
its step-time gate because it added backward dequantization without also gaining
a fused forward. This design gains both, and the measurements above show them
moving in opposite directions at 64 tokens; both effects are therefore measured
together under G5, at the same shapes, in the same run.

### 3.3 Training quality

The proposed forward matches ConvRot deployment more closely than current LoRA
training: both use the activation-quantized W8A8 base. It removes the current
W8A16-train/W8A8-deploy skew, whose per-layer size is the measured
0.98e-2 … 1.07e-2 relative forward difference recorded in §1.2. That is a reason
to test it, not proof that convergence improves: upstream adapter gradients pass
through an STE rather than the exact derivative of the dequantized base.
Convergence and quality through a W8A8 base remain **untested**; nothing
measured on 2026-08-26 addresses them.

Required comparisons are therefore:

- current ConvRot W8A16 training, deployed with fused ConvRot inference;
- proposed ConvRot W8A8-forward training, deployed identically; and
- where a matching base exists, an unquantized bf16 reference.

Compare held-out denoising loss, adapter-gradient statistics, short-run loss
curves, and fixed-seed generated samples. Weight reconstruction error alone
does not test the changed activation or gradient path.

## 4. Full fine-tuning boundary

### 4.1 Recommended: INT8 only on frozen modules

The same autograd function can be used in a partial full FT when some ConvRot
modules remain frozen. SenseNova already selects `gen`, `und`, or `both` MoT
halves. For a one-half run, a future loader may:

1. dequantize and inverse-rotate the selected half into bf16 `nn.Parameter`s;
2. leave the other half as frozen `ConvRotInt8Linear`; and
3. use fused ConvRot forward plus floating backward only on that frozen half.

This preserves normal full-precision optimization for every updated weight.
A `both` run materializes all 588 decoder Linears and therefore has no frozen
ConvRot half, so this feature provides no ConvRot benefit there; that part of
the scope note is unchanged by the measurement.

**The rest of the scope note is corrected.** It previously claimed the feature
"can reduce residency and forward time for the frozen half". Both halves of that
claim are narrower than measured:

- **Residency.** Resident weight bytes do not change at all (§3.2). What changes
  is autograd retention, and the measured retention saving is ~3% (0.27 GiB)
  with per-block gradient checkpointing on and −15.0 GiB with it off. The claim
  holds only in the checkpointing-off configuration.
- **Forward time.** Per-layer forward time falls at every measured shape, but
  the **whole synthetic step regresses** at 64 image tokens (−15.4% with
  checkpointing, −26.0% without), because the step is host-dispatch-bound there
  (§3.1). A per-layer forward saving does not imply a step-time saving.
- **Transfer.** Every number in §3 was taken at the **generation** half's shapes
  and layer count. The concrete one-half candidate here freezes the
  **understanding** half, which is the prefix pass the synthetic does not model.
  None of §3's step numbers transfers to that configuration; it would need its
  own measurement under G5.

Before widening SenseNova's existing refusal, save/resume must be designed.
The output needs one canonical choice:

- preserve/re-quantize the trained half as ConvRot so all 588 decoder Linears
  remain one flavour; or
- normalize both halves to plain INT8.

Writing a mixed plain-INT8/ConvRot base would violate the current loader and
training census and is not an acceptable shortcut. Re-quantizing a trained
half as ConvRot is the natural deployment-aligned choice, but requires an
offline/save-time quality gate and exact marker/scale validation.

### 4.2 Not recommended: trainable INT8 ConvRot weights

If a ConvRot weight itself is updated, backward also needs
`grad_W = grad_output.T @ x`. Since an INT8 tensor cannot receive useful small
updates, the implementation must keep a floating master parameter, build a
rotated/quantized shadow for forward, apply an STE, update the master, and
refresh the shadow after every optimizer update.

That design has unfavorable properties for this repository's single-GPU fine
tuning regime:

- at least 2 bytes/element for the bf16 master plus 1 byte/element for the
  INT8 shadow, before gradients and optimizer state;
- full activation retention for `grad_W`;
- a rotation and quantization pass after updates;
- optimizer-hook and stochastic-rounding interactions;
- a different checkpoint/resume contract; and
- QAT convergence and export validation requirements.

It may save some forward GEMM time, but backward still contains floating
`grad_x` and `grad_W` GEMMs. The likely VRAM and step-time trade is worse than
the existing SenseNova materialized-bf16 route. It is excluded until a separate
proposal demonstrates an OOM removal or a measured end-to-end gain.

**This verdict is unchanged by the 2026-08-26 measurement**, which covers only
frozen weights, and it is corroborated by the earlier int8-resident full
fine-tune investigation: that arm failed its own pre-registered criteria, and a
favourable probe result in it turned out to be broken three ways. G5 keeps
trainable ConvRot weights and QAT out of scope.

## 5. Artifact compatibility

**Not implemented.** No `convrot_w8a8_ste_v1` marker, no base-forward-mode
metadata field and no loader check for either exists anywhere in the repository
(verified 2026-08-26 by repository-wide search). This section is a requirement,
not a description. G5 makes it a release condition that outranks any step-time
result.

A LoRA trained with fused ConvRot forward is coupled to its base function. New
artifacts using this mode must record at least:

- training base forward mode (`convrot_w8a8_ste_v1`);
- ConvRot group size and compute/backward dtype;
- a canonical hash of the ordered ConvRot layer manifest (path, shape, scale
  shape, and quantization marker); and
- the base architecture and quantized flavour.

The inference loader must reject a `convrot_w8a8_ste_v1` artifact on a plain
INT8, FP8, bf16, dequant-forced, different-layer-manifest, or incompatible
ConvRot base. A warning is insufficient because those paths compute a different
function. Existing artifacts without this metadata remain legacy/unknown and
must not be relabelled as having used the new mode.

Backend implementation identity and comfy-kitchen version should be recorded
in training logs for reproducibility. Whether they belong in the hard artifact
contract depends on a cross-backend bitwise/quality comparison; do not assume
CUDA, Triton, eager, and HIP have identical rounding.

## 6. Interactions

### Gradient checkpointing

Supported in principle for `use_reentrant=False`. Both the original forward and
recompute should use the same fused path. Test call counts and outputs directly;
do not infer grad mode from checkpoint implementation details. Reentrant mode
remains architecture-specific and is not introduced by this feature.

### Block swap and offload

The saved INT8 code/scale references must remain valid until backward. Current
training offload moves module state with `.to(...)`; a saved reference can pin
the old storage instead of following the module. The feature must either pin
the checkpoint unit's quantized buffers deliberately or prove that its
offloader does not move them between forward and backward. Never re-read
`module.weight` from Python in backward after it may have moved.

### `torch.compile`

Not a phase-1 requirement. The comfy-kitchen op has a fake implementation, but
the SushiUI autograd wrapper and backend choice still need Dynamo/AOTAutograd
coverage before compile can be advertised. Eager execution is the initial
contract.

### Fused input activations

Comfy-kitchen can fold selected activations into ConvRot quantization. SushiUI's
current `ConvRotInt8Linear` does not expose `input_act`, and changing module
boundaries would also change what autograd must differentiate. Activation
fusion is explicitly out of the first implementation.

### Precision and scaling

bf16 is the primary dtype because it avoids fp16 underflow in the STE path and
matches SenseNova's enforced full-FT contract. fp16 requires finite-gradient
and underflow tests at real scales. fp32 is a correctness oracle and fallback,
not expected to be a speed-oriented mode.

## 7. Implementation seams

If the measurement gate passes, the smallest implementation is:

1. Add a frozen-base autograd helper beside
   `backend/core/models/common/convrot_int8_linear.py`.
2. Add an explicit per-module training-forward mode; do not overload
   `_force_dequant`, `_allow_int8_mm`, global grad mode, or an inference env
   variable.
3. Enable it only from training component loaders after the architecture has
   validated a pure ConvRot checkpoint and the selected method is compatible.
4. Teach LoRA save/load metadata about the base-function contract.
5. Add a separate SenseNova materialization helper for inverse-rotating the
   selected full-FT half only after the LoRA path is accepted.
6. If exposed as an API parameter, follow the OpenAPI-first checklist and put
   its default only in `backend/api/param_defaults.py`. The safe default remains
   the current dequant path until acceptance is complete.

Do not modify comfy-kitchen for phase 1. Its inference and dequantization ops
are sufficient; SushiUI owns the training-specific STE and artifact contract.

## 8. Acceptance plan

### Phase 0: CPU/meta structural tests

- LoRA target census is unchanged for SenseNova and MiniMax-H3 ConvRot bases.
- Codes, scales, and bias remain buffers with no gradient.
- Artifact metadata rejects mismatched base flavours and manifests.
- Full FT cannot accidentally select the frozen-base function for a trainable
  weight.

### Phase 1: CUDA/ROCm autograd correctness

- `grad_x` matches `F.linear(x, W_dq)` within dtype-appropriate tolerance for
  bf16 and fp16, with fp32 as an oracle.
- LoRA `down` and `up` gradients are finite and nonzero across early, middle,
  and late blocks.
- No weight/scale/bias gradient appears.
- Non-contiguous inputs, 3-D inputs, bias/no-bias, zero rows, outliers, NaN/Inf,
  checkpoint recompute, and repeated backward are covered.
- Unsupported dtype/device/shape behavior is deterministic and does not switch
  modes mid-run.

### Phase 2: pre-registered performance and memory gate — **G5, written 2026-08-26**

The thresholds are pre-registered in
`backend/core/training/INT8_W8A8_TRAINING_GATE.md` under **G5**, before the real
training step that decides them exists. They are not restated here, so there is
one copy to read and none to drift. In outline, G5 requires a real training step
on SenseNova (MiniMax-H3 additionally if a ConvRot/W4A8 base is reachable on its
differentiable path), inherits G3's >=10% bar and 3% regression floor unchanged,
requires the 64-token and 256-token SenseNova buckets and checkpointing on and
off among the tested workloads, and accepts OOM removal as an alternative
sufficient condition decided by a real configuration running.

**G5 registers no admission selector.** The synthetic crossover is known before
the deciding numbers exist, so an admission rule scoped to the token counts where
this path wins would be the move G3's ruling refused. The conditions under which
a successor gate may carry one — derivation from real training-time shapes,
calibration and holdout shapes separated before holdout numbers are observed, and
a cost model that matches the host-dispatch floor rather than `m*k*n` — are
recorded in G5 itself.

### Phase 3: quality gate

- three or more fixed seeds;
- matched data order, optimizer, learning rate, and initialization;
- held-out denoising loss and adapter-gradient statistics;
- fixed-prompt deployment through the fused ConvRot inference path; and
- blinded visual comparison against the current dequant-training baseline.

The feature remains opt-in until both performance and quality gates pass on the
architectures that expose it.

## 9. Final recommendation

Build and measure **fused ConvRot forward + floating `grad_input` for frozen
weights**, against G5. This addresses both opportunities in the question, and
the 2026-08-26 synthetic pass shows both moving:

- it replaces two dequantized forwards per checkpointed step with the fused
  inference kernel (measured per-layer forward ratio 1.06×-6.54×), at the cost
  of one dequantization in backward (ratio 0.28×-0.93×) and more launches per
  Linear, which is what makes the whole step regress at 64 image tokens; and
- it stops autograd retaining a full floating weight per live quantized Linear
  (measured −15.0 GiB with gradient checkpointing off, ~0.27 GiB with it on).

Neither result is a verdict. Both are synthetic, and G5 is decided by a real
training step that has not been taken.

Do not call this direct INT8 full fine-tuning. For LoRA, the base is frozen; for
partial full FT, INT8 is confined to the frozen half. Keep every updated weight
as a normal floating parameter. Treat trainable INT8 shadows as a separate QAT
project only if later measurements justify its additional memory and optimizer
complexity.

## References

- ConvRot paper: <https://arxiv.org/abs/2512.03673>
- Comfy-Kitchen implementation and capability matrix:
  <https://github.com/Comfy-Org/comfy-kitchen>
- Gates G3/G4 (non-rotated) and **G5** (this design's pre-registered gate):
  `backend/core/training/INT8_W8A8_TRAINING_GATE.md`
- Measurement record for §3, and the probe scripts that produced it (untracked
  working area): `tmp/CONVROT_MEASURED_EVIDENCE.md`,
  `tmp/convrot_probe1.py` (correctness + per-shape timing),
  `tmp/convrot_probe2.py` (whole synthetic step, checkpointing on),
  `tmp/convrot_probe3.py` (checkpointing off + CPU/GPU attribution)
- SenseNova quantized-base and full-FT record:
  `docs/guides/SENSENOVA_TRAINING_DESIGN.md`
