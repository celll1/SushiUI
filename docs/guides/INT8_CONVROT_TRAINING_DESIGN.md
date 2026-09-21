# INT8 ConvRot training status

This document records the shipped boundary and completed measurements.

## Current behavior

- Inference can use supported INT8 ConvRot checkpoints through the dedicated
  quantized modules.
- Qwen-Image 2.1 LoRA on its ConvRot artifact runs the frozen base forward with
  the ConvRot INT8 kernel. Its custom autograd node computes only `grad_input`
  from a one-time BF16 weight cache; adapter parameters and FlashAttention use
  their ordinary floating-point backward paths.
- Other architectures do not inherit that policy. Their opt-in fused path
  remains experimental unless their own gate says otherwise.
- Trainable INT8 ConvRot base weights are refused. Full-parameter training must
  not silently treat quantized weights as ordinary trainable tensors.
- The Qwen cache is non-persistent and never enters the LoRA artifact. A new
  adapter records `qwen_base_forward=convrot_int8_bf16_backward_v1`; generation
  refuses that adapter on a dense base, and resume refuses legacy/dequant and
  ConvRot-forward runs from being mixed.
- Cached Qwen training currently refuses block swap.

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

## Authoritative gates

`backend/core/training/INT8_W8A8_TRAINING_GATE.md` owns the registered G3/G4/G5
criteria and results. G3 and G4 are closed failures under their preregistered
performance limits. G5 remains unresolved as a generic cross-architecture
feature. Qwen-Image 2.1 ships only its narrower measured contract above; that
does not close G5 for SenseNova, MiniMax-H3, W4A8, or trainable quantized bases.
