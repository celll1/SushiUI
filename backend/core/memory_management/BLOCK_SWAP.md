# Block swap

## Current implementation

All weight streaming uses `offload_transfer_engine.py`.

| Path | Policy | Writeback |
|---|---|---|
| Generation | `FrozenSequentialTransferEngine` | none |
| Frozen-base/LoRA FLUX.2 | `FrozenLruTransferEngine` | none |
| Mutable training blocks | `MutableLruTransferEngine` through `LayerOffloadConductor` | after fused update |

The mutable path is wired for Z-Image, Anima, Lens, Ideogram 4, MiniT2I,
Krea 2, FLUX.2, LTX-2.3, MiniMax-H3, and ACE-Step 1.5. MiniMax-H3 supports LoRA
only; its quantized base and trainable adapters are packed in separate dtype
planes. SD1.5/SDXL use their U-Net memory path, and SenseNova uses MoT phase
eviction rather than block swap.

## Required settings

- `blocks_to_swap > 0`;
- `gradient_checkpointing = true` for mutable training;
- a fused-backward optimizer or fused optimizer groups;
- `block_swap_ring_size >= 1` (`2` is the default).

The architecture setup refuses mutable swap without checkpointing. Optimizer
setup separately refuses combinations that would update CPU parameters or retain
all gradients.

## Ordering invariants

- CPU masters are persistent and never alias another block.
- GPU slots are fixed and separated by dtype.
- H2D waits on prior slot work; compute waits only on the requested block event.
- Initial forward eviction is clean and has no D2H.
- Recompute eviction waits until the fused optimizer update is visible on the
  compute stream, then performs D2H.
- Checkpoint save calls `flush()` before reading model state.
- OOM recovery abandons active slots explicitly before retry.

Activation dispatch is independent. `LayerOffloadConductor` does not maintain a
second activation arena, so saved tensors are considered only once by
`ActivationDispatcher`.

Validation and exact commands are recorded in
`docs/audits/UNIFIED_OFFLOAD_TRANSFER_VALIDATION_2026-09.md`.
