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

Stochastic rounding is forced and announced for the accepted path. Full
fine-tune checkpoints intended for resume must use
`sensenova_full_finetune_save_format: int8`; other export formats are not
resume contracts.

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
