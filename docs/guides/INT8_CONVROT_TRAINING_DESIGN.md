# INT8 ConvRot training status

This document records the shipped boundary and completed measurements. The
proposed fused autograd design and future acceptance work are retained only in
the ignored local research area.

## Current behavior

- Inference can use supported INT8 ConvRot checkpoints through the dedicated
  quantized modules.
- LoRA over a frozen quantized base is reachable, but does not imply that every
  layer uses a fused W8A8 training forward.
- Trainable INT8 ConvRot base weights are refused. Full-parameter training must
  not silently treat quantized weights as ordinary trainable tensors.
- No new fused gradient-capable ConvRot path is shipped by this document.

## Measurement result

The 2026-08-26 synthetic investigation found a real forward difference of about
`1e-2` relative magnitude between the studied ConvRot paths. Per-layer speedups
did not translate uniformly to whole-step speedups: host dispatch dominated a
short-token case, while longer-token behavior differed. A synthetic memory arm
reduced peak allocation substantially with gradient checkpointing disabled.

These are inputs to an engineering decision, not a release verdict. They were
not a real SenseNova or MiniMax-H3 training-quality run and must not be presented
as one.

## Authoritative gates

`backend/core/training/INT8_W8A8_TRAINING_GATE.md` owns the registered G3/G4/G5
criteria and results. G3 and G4 are closed failures under their preregistered
performance limits. G5 remains unresolved until its real-workload correctness,
performance, memory, precision, artifact-compatibility, and quality conditions
are actually met. An open gate does not advertise a supported capability.
