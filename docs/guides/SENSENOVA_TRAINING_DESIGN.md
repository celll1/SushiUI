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
- gradient accumulation 1 (a physical batch above 1 needs `enable_bucketing`,
  see "Packed batches" below);
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
run is refused before the datasets are read. Prompts of different token counts
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
