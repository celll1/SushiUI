# SenseNova U1.5 training contract

This document records the shipped behavior and refusal boundary. It intentionally
omits the chronological implementation log, abandoned alternatives, and future
roadmap.

## Supported methods

- LoRA training of the generation branch is supported.
- LoRA training of the understanding branch is opt-in through
  `train_text_encoder`.
- Full-parameter training supports the generation half, understanding half, or
  both MoT halves. `train_unet` and `train_text_encoder` select the halves.
- Plain and reference-conditioned datasets may be mixed. The reference path is
  part of the dataset contract, not an inference-time ControlNet path.

SenseNova remains unsupported for ReLoRA and ControlNet training. Unsupported
combinations are refused during preflight, before model weights are loaded.

## Full-parameter preflight contract

Every full-parameter run must satisfy all of the following:

- bf16 precision;
- gradient accumulation 1;
- `batch_size: 1`, *unless* `enable_bucketing` is on — the one conditional
  clause, and the only one that applies to LoRA as well (see "Packed batches"
  below). It is served on the capability axis as
  `training_required_values.sensenova.batch_size` with
  `unless: {enable_bucketing: true}`, so a client can pin the control while the
  condition stands and release it when bucketing lifts it;
- EMA disabled;
- `blocks_to_swap: 0`;
- optimizer `adafactor`, or a supported ring-buffer optimizer with
  `optimizer_state_host_resident` enabled.

Stochastic rounding is forced and announced for the accepted path.

`sensenova_full_finetune_save_format: int8` is the only export a NEW run may be
pointed at as its base. A run resuming its OWN checkpoint is a narrower
question and is answered separately by `sensenova_ops.accept_resume_shaped_base`,
which also accepts `mixed` (one trained half plus one `Int8Linear` half) and
`bf16` (both halves trained), losslessly. It decides on a class census of the
constructed tree; the checkpoint's metadata is required and required to agree,
but can only narrow acceptance. A resume has actually been run for the
`gen`/`mixed` pair; `both`/`bf16` reaching the same acceptance is an inference
from the same write/read path, not a measured run.

## Packed batches (batch_size > 1)

A physical batch is one pixel tensor `[B, 3, H, W]` at one resolution (one
noise scale), so `batch_size > 1` requires `enable_bucketing`; without it the
run is refused before the datasets are read (`train_runner`, and again in
`BaseTrainer.train` from its own argument). The rule is the same for LoRA and
full-parameter runs. Prompts of different token counts
are not padded: `sensenova_ops.encode_prompts` builds each item's prefix inputs
exactly as the single encode does (template, aligned null per item, reference
splice) and lays them end to end along the sequence axis with batch dim 1
(`pack_prefix_inputs`, vendor `PackedSegments`). RoPE reads each token's own
t/h/w index, so packing changes no position; the understanding stack keeps the
segments apart with `create_packed_block_causal_mask` on the eager path and the
varlen conduit entry (`core.attention.dispatch_attention_varlen`,
`flash_attn_varlen_func` or one SDPA call per segment) on the causal fast path.
The generation tokens are packed the same way; `PackedGenPlan` lays item i's
prefix K/V next to its own image K/V so the same varlen call confines each
image to its own prompt. Item i's image tokens sit at `t = text_lengths[i]`,
as in the single form. The frozen packed prefix goes through the training
prefix loop under `no_grad` (the vendor prefix forward has no packed-mask
entry); the differentiable and four-phase routes are unchanged, the cut
carrying the segment layout with it. Per-MNT null labels are one per item;
a frozen branch memoizes each label vector's prefix within the batch.

Measured on the int8 base at 512px (probe, 2026-09-02): packed vs single
K/V and generation hidden states differ at the bf16 kernel level only --
the same ~3% relative Frobenius the single training loop already shows
against the vendor eager prefix forward (layer 0 identical; the drift is 42
layers of SDPA/flash vs eager rounding), 0.0 for the native per-segment path
against a single SDPA, and 0.4% on the generation hidden states with a shared
prefix. The gen ViT and timestep embedder are batch-exact at B=2 and differ
by GEMM-shape rounding at B>=3. A B=3 step with the fm_modules trainable
backpropagated finite gradients at a 37.7 GiB peak (int8 base). The
`grad_timestep_cosine_probe` is disabled at batch_size > 1 (a summed backward
has no single t to bucket by).

## Model-specific training path

The model uses a two-pass prefix/denoise step. The generation and understanding
halves are distinct MoT branches, so memory estimates and trainable-parameter
censuses must name the selected half or `both`; a single aggregate claim is
misleading.

The int8 base remains frozen for LoRA. Full-parameter training follows the
accepted bf16 compute contract above and writes through the SenseNova-specific
checkpoint path. The training registry and architecture handler are the
authoritative implementation; this guide does not override their refusals.

### Trained scope

A full fine-tune's default scope is the 294 decoder Linears per selected half —
the set the int8 load dequantizes. That is a consequence of the quantization
layout, not a claim about what is worth training: `transformer.fm_modules` (the
generation ViT's patch and dense embeddings, the timestep and noise-scale
embedders, and the `fm_head` convolutions that emit the pixel prediction) is not
quantized, so it is not materialized and was never optimized. Measured on two
checkpoints of one run 4,960 steps apart, all 16 of its tensors are
byte-identical while the generation decoder moved 3.09e-3 relative.

`sensenova_train_fm_modules` (default off) adds those 16 tensors / 63,117,504
parameters (120.4 MiB bf16, counted from the checkpoint index) to the generation
parameter group at `unet_lr`. It is generation-side, so an understanding-only
branch warns (`sensenova_train_fm_modules_branch_mismatch`) and proceeds without
them. The decoder-Linear count is collected and checked exactly as before; the fm
parameters come from a separate path so the unmaterialized-int8 guard keeps its
exact expectation. Every save format already writes non-decoder tensors as they
stand, so an update survives the checkpoint. The `*_norm_mot_gen` norms stay
frozen either way.

Changing the setting on a resume changes the generation group's parameter count
(294 vs 310), which `optimizer.load_state_dict` rejects. The per-group
leading-prefix remap salvages the rest: on run 122's resume at step 38,768 it
kept 588 of 604 parameters' saved state and started the 16 added ones fresh, and
the re-warmup fired for them.

Until the `enable_grad` plumbing below, the option moved only the 4 `fm_head`
tensors: the other 12 sat inside a `@torch.no_grad()` and received no gradient at
all. Measured over 976 steps of run 122 with the option on -- `fm_head.conv2.bias`
moved 78.6% of its elements, the other 12 tensors stayed byte-identical. A run
from before that fix trained a smaller set than this section describes.

Cost is unmeasured. Three of the four fm modules (the generation ViT and both
embedders) are called from `_build_step_context`, the per-step embed builder the
training step shares with inference; it is no-grad by default and the training
step passes `enable_grad=True` only when the fm parameters are actually
trainable. So enabling the option builds an autograd graph there that the step
did not build before, and adds activation memory; with them frozen the step is
unchanged.

## Verified boundary

The production path has completed real short-run checks for generation-half and
understanding-path training, checkpoint save/load, resume, and generation from a
trained checkpoint. Both halves are supported but materially more expensive.
These checks demonstrate reachability and serialization correctness, not model
quality or convergence.

Not established by those checks:

- quality or convergence for a particular dataset;
- composition with block swap or activation-offload experiments;
- a universal maximum resolution or VRAM requirement;
- performance claims transferable between machines.

Report measurements with architecture half, resolution, token count, optimizer,
offload settings, checkpoint format, and hardware. Raw campaigns and future
optimization proposals belong in the ignored local working area.

## Measured footprint of the both-halves run (U-2-5)

These figures are quoted by the `text_encoder_training` advisory that
`GET /schema/arch-capabilities` serves, so they live here rather than only in
the working area; a test pins the two against each other.

Conditions: real plain-int8 checkpoint, `SenseNovaFullParameterAdapter`, 64px,
3 steps, adafactor, batch 1, accumulation 1, bf16, gradient checkpointing on,
`blocks_to_swap=0`, one process per arm, on a 48 GB card. The probe caps itself
with `set_per_process_memory_fraction(0.72)` = **34.551 GiB**, so percentages
below are against that cap unless the card is named.

| | understanding half (294 Linears) | both halves (588 Linears) |
|---|---|---|
| VRAM peak allocated | 26.2571 GiB (76.0% of the cap) | **32.6606 GiB** (94.5% of the cap, 68% of the card) |
| VRAM peak reserved | 26.5508 GiB | 33.9063 GiB |
| host RSS peak | 32.101 GiB | **51.965 GiB** |
| saved checkpoint | 25.129 GiB (`mixed`) | 32.682 GiB (`mixed` requested, written as `bf16`) |

A second both-halves run whose working set matched to three decimals peaked at
**61.67 GiB** host RSS, so the host requirement is quoted as a 51.97-61.67 GiB
range and should not be stated more precisely than tens of GiB.

These are footprint measurements. No quality or convergence claim attaches to
any of them.

## Measured resolution campaign

Every figure the `text_encoder_training` and `sensenova_mot_eviction` advisories
of `GET /schema/arch-capabilities` quote, with the conditions that make it
readable. They live here for the same reason as the U-2-5 footprint above: the
advisory is what a user starting a multi-day run actually reads, and
`backend/tests/sensenova_advisory_resolution_and_host_test.py` compares the two
so a figure cannot drift on one side alone.

Conditions: real int8 checkpoint, RTX 6000 Ada (device total 47.988 GiB),
`set_per_process_memory_fraction(0.72)` = **34.551 GiB** per-process gate,
adafactor lr 1e-6, batch 1, accumulation 1, `blocks_to_swap=0`, gradient
checkpointing on, bf16, native attention, stochastic rounding forced, 12 steps
(1 warmup + an 11-step steady window), one process per arm, host 93.585 GiB.
**The gate is not the card**: an arm that exceeds it OOMs inside its own process
while the card still holds free memory. Image tokens are `(res/32)^2` — 4 at
64px, 256 at 512px, 1024 at 1024px — so the 64px residency figures above carry
almost no activation term and must not be read as resolution-independent.

| arm | branch | res | four-phase | load peak | step peak | step - load | reserved peak |
|---|---|---:|---|---:|---:|---:|---:|
| C1 | gen | 64 | off | 25.1198 | 26.0821 | **+0.9623** | 26.168 |
| A1 | gen | 512 | off | 25.1198 | 26.2377 | **+1.1179** | 26.338 |
| A2 | gen | 1024 | off | 25.1198 | 26.7996 | **+1.6798** | 27.170 |
| B1 | both | 512 | off | 32.6606 | **33.9364** (98.2% of the gate) | +1.2758 | 34.109 |
| B2 | both | 512 | **on** | 32.6606 | 32.6606 (step 1) -> **18.7607 steady** | 0 | 33.906 |
| B3 | both | 1024 | off | 32.6606 | **OOM** at 34.0373 | — | 34.414 |
| B4 | both | 1024 | **on** | 32.6606 | 32.6606 (step 1) -> **19.2586 steady** | 0 | 34.221 |

GiB of `peak_allocated` throughout.

- The generation half's step cost splits into a resolution-independent part,
  **0.9623 GiB** (the 64px step − load, where activation is negligible), plus
  activation: **+0.156 GiB at 512px, +0.718 GiB at 1024px**. Four times the
  tokens cost 4.6x the activation, so it is **superlinear and must not be
  extrapolated**; above 1024px and off-square are unmeasured, as is the
  understanding half alone above 64px.
- **B3's OOM was the probe's own gate, not the card.** It failed on
  `Tried to allocate 192.00 MiB` with **9.95 GiB** still free on the card, so
  all that is known about both-halves-at-1024px without the split is that it
  **exceeds 34.55 GiB**.
- **`reserved` does not follow the split.** Every both-branch arm holds a peak
  reserved of **33.9-34.4 GiB** for the whole run: the caching allocator keeps
  the load-time high-water mark. The split lowers what a step needs, **not what
  the process holds**.
- Steady-state drift was 0.0 in every completed arm across the 11-step window,
  which rules out fast fragmentation and says nothing about slow.

### What the four-phase split costs and buys

Two costs, measured apart and not interchangeable.

**The split alone** (`probes/text_encode_vs_step.py --arm sensenova-four-phase`,
1024px, a 467-token prefix, understanding gradients supplied by a rank-4
both-branch LoRA over int8 halves, n=25, p50):

| | p50 (s) | mean (s) |
|---|---:|---:|
| single backward: prefix forward | 0.1728 | 0.1836 |
| single backward: gen forward + backward | 1.7584 | 1.7826 |
| four-phase: recomputed understanding forward | 0.1897 | 0.2076 |
| four-phase: understanding backward | 0.3291 | 0.3343 |
| **four-phase total / single-backward total** | **1.097** | **1.093** |

i.e. a **1.09-1.10x** step, adding no weight transfer over the three-phase form.
Those are LoRA-over-int8 conditions, not the bf16 both-branch full fine-tune the
ratio gets quoted for.

**The eviction transfers the split makes possible** are what a full fine-tune
adds on top: a **7.60 GiB** int8 half staged to pinned host memory and back
measured **0.666 s** per round trip and a step makes two; a bf16 half is
**15.09 GiB**, so the full-fine-tune route moves twice that volume. End to end
the train loop went **42.672 s -> 80.508 s = 1.89x** over 12 steps at 512px.

What it buys: the steady step peak falls from 33.9364 to 18.7607 GiB at 512px,
and at 1024px the both-branch step fits at 19.2586 GiB where without it the
probe OOMed.

### Host requirements

At 48 GB of VRAM the host is the binding constraint. **Measured** and
**recommended** are kept apart below; **the recommendations are the audit's
advice, not measurements.**

**Measured**

| quantity | value | source / conditions |
|---|---:|---|
| `both` run peak commit charge | **67.95 / 89.10 GiB** | two runs of the same B3 command; working sets matched at 49.108. It did not reproduce, so the larger is the bound |
| `gen` arm peak commit charge | ~65 GiB | same campaign |
| int8 re-entry (C3) peak commit | 51.2 GiB | same campaign |
| host the campaign ran on | 93.585 GiB | conditions above |
| `both` checkpoint (bf16) | **35,091,856,594 B = 32.68184 GiB** | the U-2-5 save table |
| checkpoint (int8, **gen**-branch save) | **18,885,547,920 B = 17.5885 GiB** | C1's save. That a `both` save is the same size is an **inference** from an int8 file quantizing all 588 Linears either way, **not a measurement** |

**Recommended** (derived from the above; not measured)

- a commit limit of at least **100 GiB**, preferably **110-120 GiB** — the
  89.10 GiB upper measurement plus the 21 GiB spread that did not reproduce;
- **96 GiB or more** of physical RAM: pinned staging demands physical pages and
  a pagefile does not necessarily substitute;
- **150-300 GiB** free for checkpoints, holding several 32.68 GiB bf16
  generations; int8-only runs can sit at the low end;
- **no competing GPU process at 1024px**: B3's OOM came from the 0.72
  per-process gate rather than the card, but a both-branch run holds
  33.9-34.4 GiB reserved for its whole duration.

Where a user sees this: the measured values and the advice above are carried by
the `text_encoder_training` advisory in `arch_capabilities.py`, served by
`GET /schema/arch-capabilities` and shown beside the training form's Train Text
Encoder control.
