# Block swap

## Current implementation

All weight streaming uses `offload_transfer_engine.py`.

| Path | Policy | Writeback |
|---|---|---|
| Generation explicit block loops | `FrozenSequentialTransferEngine` | none |
| Generation hook units | `FrozenModuleOffloadConductor` / `FrozenBranchedLayerOffloadConductor` | none |
| Frozen-base/LoRA FLUX.2 | `FrozenLruTransferEngine` | none |
| Mutable training blocks | `MutableLruTransferEngine` through `LayerOffloadConductor` | after fused update |
| SenseNova branch bundles | `MutableLruTransferEngine` through `BranchedLayerOffloadConductor` | method-aware |

The mutable path is wired for Z-Image, Anima, Lens, Ideogram 4, MiniT2I,
Krea 2, FLUX.2, LTX-2.3, MiniMax-H3, ACE-Step 1.5, and SenseNova. MiniMax-H3
supports LoRA only; its quantized base and trainable adapters are packed in
separate dtype planes. SenseNova keys each physical layer by MoT branch: LoRA
streams frozen base bundles while adapters stay resident, and full fine-tuning
writes updated bundles back. Its phase evictor may additionally manage the
layers left resident by block swap.

Generation covers every architecture. SD1.5/SDXL stream top-level U-Net stages;
Krea 2 and ACE-Step stream their native block/layer lists; SenseNova streams
branch-qualified MoT layers; MiniMax Music 3 independently streams its AR and
flow stages. The other generation backends retain their explicit-loop drivers,
which use the same immutable transfer engine underneath.

## Required settings

- `blocks_to_swap > 0`;
- `gradient_checkpointing = true` for mutable training;
- a fused-backward optimizer or fused optimizer groups;
- `block_swap_ring_size >= 1` (`2` is the default).

SenseNova requires ring size 2 or greater because a mixed-token decoder call
can use both branch bundles of one physical layer simultaneously.
Its generation conductor fixes the ring at exactly two slots; its mutable
training conductor accepts larger rings.

The architecture setup refuses mutable swap without checkpointing. Optimizer
setup separately refuses combinations that would update CPU parameters or retain
all gradients.

Dense Full training on LTX and the staged H3 path leaves swappable blocks on CPU
at initial load through `training/model_staging.py`; only resident blocks and
non-block weights move to the training device before adapter construction.
The conductor still captures ownership after adapters set trainability.

Shared encoder-refill main-model moves preserve mutable CPU masters through
`training/mutable_model_residency.py`. Parking releases idle GPU slots; restoring
moves only unmanaged weights and recreates the slots. Moving the whole module
with `.to(cuda)` would detach managed parameters from their CPU master views.

With `use_pinned_memory=false`, contiguous CPU block tensors are borrowed as
master segments instead of copied into packed planes. This avoids duplicating a
dense checkpoint in RAM, but submits a transfer per tensor rather than per dtype.
Pinned or noncontiguous sources retain the packed path. Memory-mapped source
files remain unchanged after dirty writeback (verified with safetensors).

The same layout helper backs frozen hook conductors and forward-only block
offload. Initial placement keeps streamed Linear weights/sidecars on CPU and
moves only the other block tensors to the device. The H2D-only path respects
`use_pinned_memory`; disabling it no longer creates pinned copies implicitly.

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

### Unused host cache at lifecycle boundaries

`host_cache.release_unused_host_cache()` releases unused pinned staging buffers
when a model is replaced, before a training process takes the GPU, and through
the debug release API. It does not run on every step or evict live buffers.
Uninitialized CUDA is left untouched. The installed Torch uses the private
CUDA fallback; newer releases use the public accelerator API.
Device `empty_cache()` alone leaves this host cache intact
([PyTorch allocator documentation](https://docs.pytorch.org/devlogs/eager/2026-08-09-pinned-memory-allocator/)).
Four CPU compatibility cases and a real CUDA test verified unused 128MiB
reclamation while preserving a live 1MiB buffer and its contents.
The debug inspection API reports deduplicated CPU storage bytes and pinned
allocator statistics without tensor contents. In an official FLUX generation
followed by Ming Layer restoration, live CPU storage returned to 25,272MiB
and unused pinned cache to zero, while process RSS stayed around 28–30GiB
instead of the cold-load 4.65GiB. These fixes do not explain that remaining
native/allocator retention; its investigation is still open.

A subsequent tiny-model replacement removed Ming's 24.91GiB mapped weights
and reduced visible CPU storage to 2,034MiB, but RSS remained 26.35GiB.
Windows CRT `_heapmin` did not improve the real model-switch measurement;
the prototype was removed. A separate 2GiB CPU tensor allocation returned
almost all RSS after deletion, so general CPU allocation alone does not
reproduce the retention. The underlying native owner is still unproven.
API restart recovered RAM. Subsequent official FLUX visual verification ran
in an isolated process; after its normal exit, backend RSS was 4.65GiB and
system available RAM was 59.52GiB. Process isolation avoided leaving these
verification allocations in the serving backend; it is not a fix for the
unresolved in-process model-switch retention.
