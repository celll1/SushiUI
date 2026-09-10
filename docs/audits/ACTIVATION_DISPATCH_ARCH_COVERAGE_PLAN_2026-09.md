# Activation dispatch architecture coverage review plan (2026-09)

## Goal

Determine whether the per-bucket activation dispatcher used by SDXL/Anima can
be used by every training architecture, especially MiniMax-H3, without changing
the training objective.

## Static review

1. Trace the API/config value through trainer construction and establish which
   training methods can actually enable the dispatcher.
2. Verify the saved-tensor hook lifetime, prediction key, calibration path,
   OOM recovery, and micro-batch semantics.
3. Classify every `ARCH_REGISTRY` architecture by latent shape, supported
   training methods, current reachability, and required architecture-specific
   work.
4. Review interactions with gradient checkpointing, block swap/fused backward,
   quantized LoRA bases, on-the-fly encoders, `torch.compile`, host memory, and
   SenseNova's non-flow objectives.

## Deliverable

Write a factual report under `docs/audits/` with separate verdicts for:

- already reachable;
- implementation present but configuration path disconnected;
- viable after small shared plumbing;
- requiring an architecture-specific predictor or execution seam;
- not applicable because no training path exists.

No model load, CUDA context, VRAM test, or server restart is part of this review.
