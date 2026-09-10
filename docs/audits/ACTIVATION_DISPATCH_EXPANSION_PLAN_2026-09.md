# Activation dispatch expansion plan (2026-09)

## Scope

Expand the existing `BaseTrainer` activation dispatcher to every supported
training method and every `ARCH_REGISTRY` model without changing defaults.
MiniMax Music 3 remains out of scope because it has no training path; standalone
VAE/tagger loops are recorded separately because they are not model-architecture
handlers and do not inherit `BaseTrainer`.

## Units

1. **Shared configuration**
   - resolve all five `activation_dispatch_*` settings from the run's
     `train_config` inside `BaseTrainer`;
   - retain explicit constructor arguments for direct callers;
   - use `api.param_defaults.TRAINING_DEFAULTS` for fallback values.
2. **Architecture workload keys**
   - preserve the existing 4-D image and 5-D video keys;
   - represent 3-D audio latents by sequence rows rather than channel width;
   - give SenseNova understanding/text batches a token-count key and an
     independent calibration family from its flow/image batches;
   - keep dispatcher instances separate per workload family so unrelated
     predictors cannot contaminate each other in a mixed-objective run.
3. **Contracts, documentation, and deferred GPU work**
   - add CPU tests for method-independent config resolution and every tensor
     rank/workload family;
   - synchronize architecture references;
   - document the real-checkpoint VRAM, host-memory, timing, and numerical tests
     that remain owner-run work.

All behavior remains opt-in. No model load, CUDA context, server restart, or
VRAM-consuming test is part of these commits.
