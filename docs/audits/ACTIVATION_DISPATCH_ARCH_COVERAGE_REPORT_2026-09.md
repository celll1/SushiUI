# Activation dispatch architecture coverage report (2026-09)

Status: static review and implementation complete; no model load or GPU
measurement was performed. Required execution work is tracked in
`ACTIVATION_DISPATCH_GPU_VALIDATION_BACKLOG_2026-09.md`.

Implementation follow-up: `BaseTrainer` now resolves the five settings from the
shared run configuration for every trainer method, uses workload-native keys for
4-D image, 5-D video, 3-D ACE-Step audio, and SenseNova instruction/text paths,
and keeps their learned predictors separate. The reachability findings below
describe the pre-expansion baseline that motivated those changes.

## Executive conclusion

The activation dispatcher is not an SDXL/Anima-specific implementation. Its
saved-tensor hook and its forward/backward lifetime live in `BaseTrainer`, so
the core mechanism is already architecture-neutral. MiniMax-H3's 5-D latent and
clip length are explicitly supported by the prediction key as well.

The actual gap found by the audit was configuration reachability:

- the five `activation_dispatch_*` values were passed to
  `FullParameterTrainer` only (`train_runner.py:4024-4028`);
- `LoRATrainer`, `ReLoRATrainer`, and `ControlNetTrainer` received
  `train_config`, but `BaseTrainer` did not read these five values back from
  that dict; their constructor defaults therefore kept dispatch disabled;
- MiniMax-H3 supports LoRA only, so its existing dispatcher implementation was
  unreachable in every supported H3 run.

Accordingly, MiniMax-H3 is **implemented with validation caveats** and does not
need a new H3 forward hook. The shared LoRA plumbing is now connected. Real H3
measurements remain required before calling it production ready, because the
cold-start predictor and PCIe/host-memory costs were tuned and timed on SDXL,
not H3.

## 1. What the feature actually does

`core/memory_management/activation_dispatcher.py:53-97` installs
`torch.autograd.graph.saved_tensors_hooks` around one training forward and its
backward. A CUDA floating-point, non-leaf tensor at least 4 MiB by default is
copied value-exactly to pageable CPU memory when autograd saves it, then copied
back synchronously when backward requests it. Small tensors, leaf tensors, and
non-floating tensors remain on the GPU.

`BaseTrainer._forward_backward_with_oom_recovery`
(`base_trainer.py:10324-10532`) enters that context before the architecture's
training step and exits it after backward. The dispatcher has three decisions:

1. `fast`: predicted activation fits, so the hook context is a no-op;
2. `offload`: saved activation is copied to CPU;
3. `escalate`: offload is enabled and a smaller physical micro-batch is planned
   when fused backward is not active.

The prediction is calibrated passively from real training steps. No extra
calibration forward is executed. Live headroom is recomputed with
`torch.cuda.mem_get_info()` every step, while exact observations are cached by
latent shape and physical batch size.

This is distinct from `LayerOffloadConductor.enable_activation_offload`.
Architecture-specific block-swap setup currently passes `False` for that older
conductor feature. If it is ever enabled, `_activation_dispatch_begin`
suppresses dispatcher offload to prevent two activation offload mechanisms from
stacking.

## 2. Reachability audit

The public configuration is complete: defaults are in
`api/param_defaults.py:2525-2533`, the API fields are in
`api/routes.py:16159-16164`, serialization is in
`training/training_config.py:435-440`, and the shared training UI is in
`TrainingConfig.tsx:5914-5951`. The break is later, at trainer construction.

Only one occurrence of
`activation_dispatch_enable=train_config.get(...)` exists in `train_runner.py`,
inside `FullParameterTrainer(...)`. Passing `train_config=train_config` to the
other trainer classes does not compensate: `BaseTrainer.__init__`
(`base_trainer.py:2697-2704`) assigns the explicit constructor arguments, whose
defaults are disabled.

Method abbreviations below are L=LoRA, R=ReLoRA, F=full parameter, and
C=ControlNet. Unsupported methods are taken from
`api/arch_capabilities.py:1020-1067`.

| Architecture | Training methods | Denoiser input at dispatch | Pre-expansion reachability | Implemented result |
|---|---:|---|---|---|
| SD1.5 | L/R/F/C | 4-D latent | F only | L/R/F/C viable |
| SDXL | L/R/F/C | 4-D latent | F only | L/R/F/C viable; existing measured baseline |
| Z-Image | L/R/F | 4-D latent | F only | L/R/F viable |
| Anima | L/R/F | 4-D latent | F only | L/R/F viable; existing full-FT path |
| Lens | L/R/F | 4-D latent | F only | L/R/F viable |
| Ideogram 4 | L | 4-D latent | none | L viable; quantized-base regression required |
| MiniT2I | L/R/F | 4-D pixel/latent tensor | F only | L/R/F viable |
| Krea 2 | L/R/F | 4-D latent | F only | L/R/F viable; quantized-base regression required |
| FLUX.2 | L/R/F | 4-D latent | F only | L/R/F viable, including reference-latent batches |
| LTX-2.3 | L/R/F | 5-D video latent | F only | L/R/F viable; temporal key already implemented |
| MiniMax-H3 | L | 5-D video latent | none | **L viable with caveats; highest-priority gap** |
| ACE-Step 1.5 | L/R/F | 3-D audio latent | F only | L/R/F connected with an audio-native key |
| SenseNova U1.5 | L/F | 4-D flow image or text batch | F flow objective only | Flow and text L/F connected with separate predictors |

ControlNet is implemented only for SD1.5 and SDXL, so connecting its constructor
does not expose the feature to unsupported architectures.

MiniMax Music 3 is not in `ARCH_REGISTRY` and has no training path; activation
dispatch is not applicable until a trainable audio encoder/objective exists.
The standalone tagger and VAE trainers do not subclass `BaseTrainer`, so they
are outside this report's “architecture plumbing” fix and require independent
integration.

## 3. MiniMax-H3 assessment

### 3.1 Already implemented correctly

H3 supplies `[B, 24, T_lat, H_lat, W_lat]`. The dispatcher key helper
(`base_trainer.py:10535-10556`) recognizes 5-D inputs and returns
`(H_lat, W_lat, T_lat, B)`. `ActivationDispatcher` includes `T_lat` in both its
exact cache key and regression volume. This prevents clips with the same canvas
but different duration from sharing a prediction.

The current source records an H3 activation fit at 384x640: approximately
2.36 GiB for a 22-frame clip and 8.90 GiB for a 124-frame clip. These values
establish that temporal extent is load-bearing; they are measurements of base
activation, **not measurements of dispatch savings or iteration overhead**.

The hook surrounds `minimax_h3_ops.train_step`, including its packed
`[text | audio | video]` transformer forward and backward. H3 therefore needs
no block-by-block instrumentation for the synchronous dispatcher.

### 3.2 Pre-expansion blocker (resolved)

H3's only supported method is LoRA. Before the expansion,
`self.activation_dispatch_enable` remained `False` even when the API/YAML/UI
value was `True`. `BaseTrainer` now resolves the value from `train_config`, so
the supported H3 LoRA route is opt-in reachable.

### 3.3 H3-specific caveats after connection

- H3 enforces physical batch size 1. The dispatcher's micro-batch escalation
  cannot reduce it further; only saved-activation offload and the reactive
  lower-threshold retry can rescue a tight step.
- The shared cold-start slope is `24e-6 GiB/(B*latent voxel)`, inherited from
  the image-model work. The two H3 observations in the source imply a much
  larger H3-specific slope plus a non-zero packed-sequence intercept. Until two
  distinct volumes calibrate the fit, a new H3 bucket can therefore be badly
  under-predicted and pay one failed/capped attempt before recovery.
- H3 commonly combines quantized frozen base weights, LoRA wrappers, gradient
  checkpointing, and optional block swap. The saved-tensor predicate should
  select non-leaf activations rather than quantized weight buffers, but this has
  not been verified end to end on the real H3 checkpoint.
- Dispatcher copies are synchronous and pageable. H3's 50-block packed stream
  may move much more data than SDXL's measured checkpoint boundaries. Optional
  weight block swap uses the same PCIe link, so “only a few percent slower”
  cannot be transferred from SDXL without measurement.
- CPU tensors are reclaimable (not pinned-cache resident), but H3 already has a
  very large memory-mapped text encoder and CPU-side model state. Host commit
  headroom must be measured alongside VRAM.

Verdict: **implement the shared LoRA plumbing, but retain opt-in status and do
not choose an H3 default until a real checkpoint campaign measures saved GiB,
host peak, and step-time distribution.**

## 4. Other architecture groups

### 4.1 Four-dimensional image paths

SD1.5, SDXL, Z-Image, Anima, Lens, Ideogram 4, MiniT2I, Krea 2, and FLUX.2 all
arrive at the generic recovery wrapper with a batch-first 4-D tensor. For these
architectures the key is unchanged from the original image implementation:
`(latent_h, latent_w, 1, batch)` and the regression variable is latent area.

No architecture-specific forward changes are needed. The missing work is to
pass the settings to L/R/C constructors and then run per-family tests. Quantized
LoRA paths (Ideogram 4, Krea 2, and some other checkpoint variants) deserve a
separate saved-tensor census because custom autograd functions may save a
different mix of tensors from dense modules.

### 4.2 LTX-2.3

LTX-2.3 has the same relevant 5-D geometry as H3. Temporal keying is already
covered by `activation_dispatcher_bucket_key_test.py`; no new dispatch seam is
needed. Its joint audio/video block loop and optional block swap change the
offloadable fraction, so H3 or SDXL coefficients must not be reused as measured
LTX results.

### 4.3 ACE-Step 1.5

The generic code was already present for ACE-Step full fine-tuning, despite the
architecture reference calling activation dispatch unsupported by referring
only to the disabled conductor flag. The reference is now corrected.

ACE-Step's 3-D `[B, T, C]` tensor previously became the pseudo-spatial key
`H=T, W=C, T_lat=1`. It now uses the audio-native representation
`(sequence_rows, 1, 1, batch)` and a predictor isolated from other workload
families.

### 4.4 SenseNova

The flow/image objective uses the image workload family and is reachable in both
LoRA and full-parameter runs through the shared configuration plumbing.

Instruction text objectives now use mean input-token rows as their workload and
an independent predictor. Their existing token-weighted micro-batch reduction
is retained. Real execution is still required to calibrate the cold-start prior
and confirm the approximation across different image grids and target lengths.

## 5. Composition and correctness boundaries

| Feature | Static conclusion |
|---|---|
| Gradient checkpointing | Composes mechanically. It reduces what autograd saves, so dispatch has fewer tensors left to move. The gain and slowdown must be measured with the architecture's shipped checkpointing mode. |
| Block swap | Weight swap and saved-activation offload are logically orthogonal, but contend for PCIe bandwidth. The dispatcher already avoids double activation offload if the conductor's separate activation flag is enabled. |
| Fused backward / fused optimizer groups | Offload remains usable. Proactive micro-splitting is disabled because a chunk can apply parameter updates before a later chunk fails; escalation lowers the saved-tensor threshold instead. |
| LoRA | Conceptually compatible: base weights remain leaves/buffers while adapter-dependent non-leaf activations are eligible. Quantized custom-autograd variants still require execution tests. |
| On-the-fly VAE/TE/VE | Their forwards occur before the dispatcher context, so tensors saved there are not offloaded. The two-stage splitter preserves their gradients, but dispatch savings mainly cover the denoiser step. |
| Cached latents/text embeddings | Fully compatible; the caches are inputs, not saved denoiser activations. |
| `torch.compile` | LoRA paths run eager by policy. Full-FT DiT compile plus external saved-tensor hooks is not covered by the current dispatcher tests and needs an ON/OFF matrix before support is claimed. |
| MNT | The dispatcher wraps each MNT forward/backward call and reuses the same shape key. No objective change is introduced. |
| Numerical result | CPU offload copies preserve tensor values. Micro-batch escalation preserves the intended full-batch mean, but can change floating-point reduction order and would need special care for stochastic/stateful layers. |
| Process-wide allocator cap | Enabling dispatch calls `set_per_process_memory_fraction` once. This prevents WDDM spill but also constrains other CUDA work in the same training process, including training-time samples. |

## 6. What is actually measured

The historical SDXL design retained in git (`c132fcc3`,
`docs/VRAM_OVERFLOW_PREVENTION_DESIGN.md`) reports measurements under the
then-current fp16, FlashAttention, gradient-checkpointed SDXL setup:

- 1024-1536px buckets: roughly 0.3-0.8 GiB saved for 5-10% synchronous
  iteration overhead;
- a 2048px batch-4 case: roughly 3.9 GiB / 21% saved for about 3% overhead;
- the offloadable remainder under checkpointing was about 20% of activation in
  that experiment.

These are historical, workload-specific SDXL figures, not a universal promise.
They support “small overhead” for a compute-heavy high-resolution bucket, but
not for every bucket. The current dispatch policy avoids paying that cost on
buckets predicted to fit.

There is no equivalent end-to-end H3, LTX-2.3, ACE-Step, or quantized-DiT result
in the tracked evidence. H3's two activation-footprint observations validate
the temporal predictor only.

## 7. Implementation outcome and remaining order

### P0 — shared connection (complete)

`BaseTrainer` resolves all five settings from the shared `train_config`, and a
static contract test proves every trainer constructed by `train_runner`
receives it. H3 LoRA and the other supported adapter paths are now reachable;
the default remains `False`.

### P1 — workload-family keys (complete; priors deferred)

The implementation preserves the 4-D image result, retains `(H,W,T,B)` for
H3/LTX, uses sequence rows for ACE-Step, and gives SenseNova text batches a
token-row key. Predictors are isolated by workload family. Family-specific
priors and an optional offload-first H3 policy remain measurement-dependent.

### P2 — add CPU and real-checkpoint gates

CPU/static tests:

- config propagation for all four trainer constructors;
- 4-D image and 5-D video key invariants;
- H3 clip lengths never share cache entries;
- ACE 3-D key behavior;
- SenseNova text batches use their token-row workload family;
- fused paths never micro-split;
- no double activation offload with a conductor.

Owner-run GPU matrix, one architecture at a time:

- same seed and batch, dispatch forced OFF versus forced ON;
- loss and gradient comparison, peak allocated/reserved VRAM, host
  working-set/commit peak, median/p95 iteration time, and bytes offloaded;
- gradient checkpointing on/off where supported;
- block swap off/on;
- dense versus quantized LoRA base;
- at least two spatial buckets and, for H3/LTX, at least two clip lengths.

### P3 — only after measurement

Choose per-architecture seed coefficients or an offload-first cold-start policy.
Do not enable dispatch by default merely because the shared plumbing works.
Async pinned-memory prefetch is a separate optimization: it may reduce transfer
stall, but it reintroduces pinned allocator lifetime and stream-ordering risks
that the current pageable synchronous implementation deliberately avoids.

## 8. Final verdict

- **MiniMax-H3:** opt-in implementation is connected to LoRA; highest-priority
  real-checkpoint validation target.
- **Other image architectures and LTX-2.3:** implemented through the shared
  plumbing; no per-block changes were required.
- **ACE-Step:** implemented with an audio-native sequence-row key.
- **SenseNova flow:** implemented for LoRA/full parameter. **SenseNova text
  tasks:** implemented with a separate token-row predictor; real-checkpoint
  validation remains.
- **VAE/tagger:** possible in principle through saved-tensor hooks, but not a
  small architecture extension because they own separate loops and predictors.
- **MiniMax Music 3:** not applicable while training itself is unavailable.

The shared implementation is complete. The next step is the real-checkpoint
matrix, starting with H3; a new H3-specific activation offload engine would
duplicate code that is already on its training spine.
