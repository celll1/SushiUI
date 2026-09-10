# SenseNova branch-aware block swap

## Goal

Add training block swap for SenseNova without treating one physical Qwen layer
as one indivisible weight bundle. Each layer contains an understanding half and
a generation half, so residency is keyed by `(branch, layer_index)`.

## Design

1. Add a branch-aware conductor backed by the shared mutable LRU transfer
   engine. Frozen LoRA bases use clean H2D streaming while trainable adapters
   remain resident; full-fine-tune weights use fused-update-aware D2H writeback.
2. Keep the model on CPU through adapter installation when block swap is
   requested. Stage only unswapped tensors and ring slots, avoiding the current
   whole-model load peak.
3. Compose with MoT phase eviction by excluding conductor-owned physical layers
   from the phase evictor. The remaining resident layers may still use the
   existing prefix/denoise/four-phase state machine.
4. Preserve the existing four-phase split as an independent opt-in. Block swap
   alone may follow the normal connected backward and reload understanding
   layers during checkpoint recomputation.
5. Admit `blocks_to_swap` in the SenseNova preflight, expose the existing ring
   and pinned-memory controls, and document the supported combinations.

## Safety boundaries

- Require gradient checkpointing and `0 <= blocks_to_swap < 42`.
- Never let the phase evictor and branch conductor own the same tensor.
- Flush dirty CPU masters before checkpoint reads and release active slots on
  abandoned steps through the existing trainer lifecycle.
- Keep activation dispatch independent and composable.
- Do not modify or rebuild the ring-buffer optimizer extensions.

## Verification status

Implementation is proceeding without test or GPU execution at the owner's
request while another training run is active. CPU/import and real-GPU gates are
therefore deferred and must be completed before treating this path as verified.
