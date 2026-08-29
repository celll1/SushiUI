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
- batch size 1;
- gradient accumulation 1;
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

## Model-specific training path

The model uses a two-pass prefix/denoise step. The generation and understanding
halves are distinct MoT branches, so memory estimates and trainable-parameter
censuses must name the selected half or `both`; a single aggregate claim is
misleading.

The int8 base remains frozen for LoRA. Full-parameter training follows the
accepted bf16 compute contract above and writes through the SenseNova-specific
checkpoint path. The training registry and architecture handler are the
authoritative implementation; this guide does not override their refusals.

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
