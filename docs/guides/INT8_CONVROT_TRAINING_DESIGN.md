# INT8 ConvRot training status

This document records the shipped boundary and completed measurements.

## Current behavior

- Inference can use supported INT8 ConvRot checkpoints through the dedicated
  quantized modules.
- Qwen-Image 2.1 LoRA on its ConvRot artifact runs the frozen base forward with
  the ConvRot INT8 kernel. Its custom autograd node computes only `grad_input`
  in BF16; adapter parameters and FlashAttention use their ordinary floating-
  point backward paths. The automatic policy reconstructs a bounded two-block
  BF16 backward ring on a CUDA prefetch stream. The dispatcher accounts for
  the resolved ring limit rather than the former 13.252 GiB full-cache floor.
- Other architectures do not inherit that policy. Their opt-in fused path
  remains experimental unless their own gate says otherwise.
- Trainable INT8 ConvRot base weights are refused. Full-parameter training must
  not silently treat quantized weights as ordinary trainable tensors.
- Any Qwen cache is non-persistent and never enters the LoRA artifact. A new
  adapter records `qwen_base_forward=convrot_int8_bf16_backward_v1`; generation
  refuses that adapter on a dense base, and resume refuses legacy/dequant and
  ConvRot-forward runs from being mixed.
- Cached Qwen training currently refuses block swap.

The `auto` policy selects the measured `prefetch_bf16` path with two resident
transformer blocks and one block of lookahead. Explicit `cached_bf16`,
`transient_bf16`, and legacy `dequant` modes remain diagnostic overrides.
Transient reconstruction reduced the measured partition peak by about 10 GiB,
but changed a 3.06 s transformer step to 6.61 s, so it is not an automatic
fallback. Cached, prefetch, and transient modes share the same artifact
base-function contract because their forward and floating grad-input equations
are the same.

## Measurement result

The 2026-08-26 synthetic investigation found a real forward difference of about
`1e-2` relative magnitude between the studied ConvRot paths. Per-layer speedups
did not translate uniformly to whole-step speedups: host dispatch dominated a
short-token case, while longer-token behavior differed. A synthetic memory arm
reduced peak allocation substantially with gradient checkpointing disabled.

These are inputs to an engineering decision, not a release verdict. They were
not a real SenseNova or MiniMax-H3 training-quality run and must not be presented
as one.

The Qwen-Image 2.1 gate was measured separately on 2026-09-21 with an RTX 6000
Ada, rank-128 LoRA, batch 1, BF16 training and packed FlashAttention. At 1024,
the previous dequant-forward path measured 3.460 s median forward+backward.
ConvRot forward plus cached BF16 backward measured 2.367 s with all 32 blocks
checkpointed (23.85/24.47 GiB allocated/reserved), and 2.087 s with 16 blocks
checkpointed (38.28/38.69 GiB). At 1536, 24 checkpointed blocks measured
5.907 s versus 7.864 s for dequant, with 40.78/41.53 GiB peak. Twenty blocks
overflowed the 48 GB device into shared memory (49.86 GiB reserved) and slowed
to 27 s, so it is rejected as an automatic setting. The automatic policy is
16 blocks through 1024, 24 through 1536, and all 32 above 1536.

Matched-seed losses stayed aligned across the old and new paths: step 1 was
0.294225 versus 0.294230 and step 8 was 0.205992 versus 0.205953. This is a
plumbing/numerical smoke gate, not a long-run quality equivalence claim.

A matched 60x104-latent partition probe (rank 128, two regions, 24/32
checkpointed blocks, global adapter enabled) measured INT8 ConvRot plus the
13.252 GiB BF16 cache at 3.334 s and 28.14 GiB peak. The dense BF16 base measured
4.561 s and 21.47 GiB: 6.67 GiB lighter but 37% slower. The cached path is thus
a speed policy, not a memory optimization. On the same model and workload, a
two-block ring with one-block asynchronous dequant lookahead measured 3.263 s
and 15.76 GiB for the partitioned step. That is 3.1% slower and 12.39 GiB lower
than the matched full-cache run; its 1.065 GiB reported resident limit consists
of two 416 MiB transformer blocks plus 0.252 GiB outside-block weights. Full and
prefetch losses and prediction metrics were identical in the probe, and the
small CUDA oracle matched output and input gradient bitwise. This passes the
bounded-cache gate and makes prefetch the automatic policy. Synchronous
per-layer rebuild remains an unacceptable substitute.

## Authoritative gates

`backend/core/training/INT8_W8A8_TRAINING_GATE.md` owns the registered G3/G4/G5
criteria and results. G3 and G4 are closed failures under their preregistered
performance limits. G5 remains unresolved as a generic cross-architecture
feature. Qwen-Image 2.1 ships only its narrower measured contract above; that
does not close G5 for SenseNova, MiniMax-H3, W4A8, or trainable quantized bases.
