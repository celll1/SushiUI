# Memory management

This package owns weight transfer and saved-activation offload. The mechanisms
share CUDA stream/event ordering but have different lifetimes:

- `FrozenSequentialTransferEngine`: immutable generation weights in forward order;
- `FrozenLruTransferEngine`: immutable frozen-base weights under checkpoint recomputation;
- `MutableLruTransferEngine`: trainable weights, with D2H after fused optimizer update;
- `ActivationDispatcher`: saved tensors selected from the live bucket budget;
- ring-buffer optimizers: persistent pinned CPU optimizer state, implemented under
  `core/training/optimizers`, not by a recycled layer arena.

`TransformerBlockOffloader` and `FluxBlockOffloader` adapt generation and frozen
training paths to the frozen engines. `LayerOffloadConductor` adapts checkpointed
mutable training blocks to the mutable engine.

## Mutable training contract

Mutable block swap requires non-reentrant gradient checkpointing and a fused
optimizer path. The lifecycle is:

1. load a block and prefetch the next block;
2. release it without D2H after the initial checkpointed forward;
3. reload it for reverse-order recomputation;
4. retain its GPU slot until optimizer hooks clear its gradients;
5. write the updated dtype-separated bundle to its persistent pinned CPU master;
6. synchronize all masters before checkpoint serialization.

Active slots are never victims. Ring size 1 is the minimum-memory serialized
mode; ring size 2 overlaps transfer with adjacent block compute. Parameters and
registered buffers keep their dtype and object identity.

See `BLOCK_SWAP.md` for architecture coverage and
`docs/audits/UNIFIED_OFFLOAD_TRANSFER_VALIDATION_2026-09.md` for measurements.
