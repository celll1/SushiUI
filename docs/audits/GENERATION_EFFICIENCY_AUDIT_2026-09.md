# Generation efficiency audit plan (2026-09)

## Goal

Find removable generation code and equivalent implementations that reduce
latency, peak VRAM, allocation churn, or maintenance cost across every
generation architecture. Preserve generated results unless a proposed change
is explicitly classified as opt-in or measurement-dependent.

## Scope

1. Trace request dispatch, component staging, conditioning, denoising, decode,
   media conversion and cleanup for image, video and audio generation.
2. Inspect shared and architecture-specific paths for repeated transfers,
   synchronizations, tensor copies, allocator flushes, recomputation and
   duplicated orchestration.
3. Separate findings into:
   - statically equivalent cleanup suitable for direct implementation;
   - equivalent changes requiring timing or peak-VRAM confirmation;
   - numerical or quality trade-offs that must remain opt-in.
4. Check existing tests and invariants before recommending consolidation.

## Evaluation rules

- A deletion must identify the live behavior that could otherwise be lost.
- A performance proposal must name the allocation, transfer, synchronization,
  or repeated computation it removes.
- `torch.cuda.empty_cache()` is not treated as a VRAM reduction by itself;
  reserved memory, fragmentation risk and synchronization cost are considered
  separately.
- Results that depend on CUDA allocator behavior, kernel choice, compilation,
  model size or host bandwidth are marked for measurement rather than claimed
  from static inspection.
- Frontend request construction is included where it causes duplicated uploads
  or backend work, but UI rendering performance is outside this audit.

## Verification plan

- Compile and CUDA-free import any changed backend module.
- Run focused CPU/static tests for each implementation unit.
- Record bit-exact requirements and the smallest representative GPU benchmark
  needed for candidates that cannot be proven statically.
- Re-scan the generation tree after each cleanup unit.

## Result

The generation tree has useful static cleanup left. The highest-value finding
is not a kernel substitution: API generation enters its executor worker with
autograd enabled, while the shared SD1.5/SDXL sampling and prompt-encoding path
does not provide its own complete `no_grad` boundary. Several newer backends do
have local guards, but that does not protect the shared path or future entry
points. A central `torch.no_grad()` boundary is the first implementation unit.

The next tier is work whose result is immediately discarded: predicted-clean
latents and CFG diagnostics are built on every denoise step even when the
progress callback emits a preview only every few steps. Repeated device-scalar
reads and unconditional debug reductions also introduce avoidable CUDA
synchronization. These are more likely to improve latency consistency than a
large kernel rewrite, and do not require changing model arithmetic.

No static claim is made that removing every `empty_cache()` call is beneficial.
Many calls sit between mutually exclusive, multi-gigabyte components and may be
needed to make cached blocks reusable. Only demonstrably duplicate terminal
flushes are direct cleanup candidates; phase-boundary calls require a GPU
fragmentation benchmark.

## Implementation status

The statically equivalent units are implemented. Generation now enters a
common no-grad boundary; routine diagnostics and optional callback products are
demand-driven; safe SD preview clones and duplicate terminal cache flushes are
removed; routine VRAM logging avoids module scans; failed quantization returns
the original object; and denoise schedules are copied to CPU scalars once per
loop. Schedule snapshots cover SD1.5/SDXL, Z-Image, Flux2, Anima, Lens, Krea2,
Ideogram 4, MiniT2I and SenseNova.

The callback helper is also the tested pure-orchestration extraction used by
Anima. It preserves both progress reporting and Diffusers-format step callbacks
instead of dropping one or calling it with the sampler's incompatible
signature. Numerical denoise, mask and visit-schedule loops remain separate.

Runtime-FP8 failure paths no longer clone an unchanged full-precision model.
Successful Flux2 and Z-Image FP8 copies are now cached under model-, adapter-,
quantization- and source-object-aware identities. Each component retains at
most one quantized copy beside its source; switching identity or FP8 format
evicts that copy, block-swap offloads it, runtime INT8 discards it, and the
active component slot follows the object tracked by keep-hot. Real-model host
RAM and warm-generation timing remain in the verification backlog.

## Findings suitable for equivalent implementation

| Priority | Finding | Cost removed | Required proof |
|---|---|---|---|
| P0 | Put the blocking generation call in `backend/api/routes.py::_run_generation_in_executor` under `torch.no_grad()` | Autograd metadata and saved tensors throughout unguarded inference, especially shared SD prompt encoding and U-Net sampling | Executor test that grad is disabled inside the worker and request `contextvars` still propagate; representative seed comparison |
| P0 | Remove or developer-gate unconditional sampler and adapter diagnostics | GPU reductions, `.item()`/`.tolist()` synchronization and console I/O | Logging test; no tensor result changes |
| P1 | Give the progress callback a cheap `wants_preview(step, total)`/`wants_metrics(...)` query | Predicted-clean latent casts/copies and metrics that the callback discards between preview intervals | Existing callback behavior at initial, first, interval and final steps; unknown callbacks conservatively request data |
| P1 | Replace the scheduler `pred_original_sample.detach().clone()` with a non-copying detached view where ownership permits | One latent-sized device copy per SD denoise step | Supported-scheduler alias/ownership tests and seeded output comparison |
| P1 | Snapshot scheduler scalars once instead of calling `.item()`/`float(t)` in each iteration | One or more GPU-to-CPU synchronization points per denoise step | Exact scalar equality for each supported scheduler and seeded output comparison |
| P1 | Remove duplicate terminal `empty_cache()` calls | Redundant allocator flush/synchronization after a component has already been offloaded | Mocked call-count tests plus peak-reserved-memory measurement |
| P2 | Gate detailed device/module inspection behind developer logging | Repeated module scans, reductions and log contention at every staging phase | Normal/developer logging tests |
| P2 | Extract only pure orchestration shared by txt2img/img2img/inpaint | Three-way maintenance duplication without altering numerical loops | Existing mode-specific tests; no unification of mask or visit-schedule arithmetic |

### 1. Missing common autograd boundary

`_run_generation_in_executor` copies the request context into the worker and
calls the supplied function directly. PyTorch grad mode is thread-local, and a
new executor worker has grad enabled by default. A CPU probe using the project
environment confirmed both `torch.is_grad_enabled() is True` and
`Linear(...).requires_grad is True` inside such a worker.

The shared SD loops in `backend/core/inference/custom_sampling.py` and prompt
encoding in `backend/core/pipeline.py` are not wholly enclosed by `no_grad`.
There is also no reliable model-wide `requires_grad_(False)` fallback in that
path. Keeping graph history across denoise iterations is unnecessary for
generation and can retain large activations. The API executor is the narrowest
common boundary and protects nested upscale generation as well as future
backends.

Use `no_grad`, not `inference_mode`, for the first change. Some tensors leave
the worker for saving, preview, or callback handling; `inference_mode` imposes
additional tensor mutation rules and therefore needs a broader compatibility
audit. Local backend decorators can remain: nested `no_grad` is cheap and makes
the backend safe when called outside the API.

### 2. Debug work on the hot path

`backend/core/inference/custom_sampling.py` enables its first-iteration debug
blocks unconditionally in all three shared SD loops. These blocks perform
min/max/mean reductions and scalar extraction. Scheduler diagnostics also
materialize values with `.tolist()`. `backend/core/pipeline_backends/zimage.py`
computes a full adapter-weight norm during normal LoRA generation. These are
observability operations, not generation inputs, and should run only in
developer mode.

### 3. Preview and metric work that is discarded

`backend/api/generation_utils.py` decodes and transmits previews only for the
initial/first/final steps and `preview_interval` steps. Several denoise loops
nevertheless construct `pred_x0` every step before invoking that callback:

- Z-Image and Flux2 cast the latent and prediction to FP32 for the computation.
- Anima computes `pred_x0` in all three modes and an additional masked preview
  tensor for inpaint.
- Krea2, Lens and Ideogram 4 also construct predicted-clean latents on every
  callback step.
- The shared SD path calculates CFG norms/dot products and extracts several
  scalars in developer mode even on steps whose metrics are discarded.

The callback factory should expose a side-effect-free demand predicate. A
backend may omit preview-only values only when that predicate explicitly says
they are not wanted. A callback without the new attribute must retain the old
behavior. MiniMax-H3 and SenseNova already condition their analogous work on
whether a step callback exists; MiniT2I's predicted-clean value is part of the
sampler update and cannot be skipped.

### 4. Per-step scalar synchronization

Anima, Krea2, Lens, Ideogram 4, MiniT2I, SenseNova and Z-Image contain
per-iteration `.item()`, `float(tensor)`, or tensor truth-value conversions.
Some paths read the same invariant maximum sigma on every step. Each read can
force the CPU to wait for queued CUDA work.

Create the device schedule as before, then obtain its Python scalar view once
with one CPU transfer and enumerate both views together. Hoist invariant scalar
values outside the loop. Do not independently recompute schedules on the CPU:
using the scalar view of the actual device schedule avoids dtype/rounding drift.

### 5. Copies and allocator flushes

The three shared SD loops clone the scheduler's detached predicted-original
sample on every step. The downstream preview and reference-guide operations are
read-only or out-of-place, so the clone appears unnecessary. Because scheduler
implementations may return aliases, removal is contingent on an ownership test
covering every supported scheduler.

Krea2, Lens and Ideogram 4 each flush the CUDA cache immediately after final VAE
offload and then flush it again in unconditional cleanup. The second adjacent
flush is redundant. Calls between text encoding, denoising and VAE decode are
not classified as redundant: they may release cached blocks before the next
large component is staged and must be evaluated with allocated/reserved/peak
measurements and fragmentation-sensitive repeated runs.

### 6. Runtime quantization copies and cache identity

`backend/core/vram_optimization.py` deep-copies a full model even on several
runtime-FP8 failure/unsupported paths. That can leave the original CPU model
beside an unquantized copy while the copy is mistakenly treated as the
quantized result. Flux2 and Z-Image text-encoder FP8 paths also return a local
copy without consistently replacing or caching the component identity, so
later generations may repeat copy and quantization work. With keep-hot state,
component-name residency and the actual object can diverge.

This should be a separate correctness/performance unit: quantization helpers
must report success explicitly, never clone on failure, and cache a successful
object under a key including model identity and quantization mode. A persistent
cache trades repeated startup work for host RAM, so its eviction policy and
resident-object bookkeeping need dedicated tests before implementation.

## Architecture review matrix

| Architecture | Static result |
|---|---|
| SD1.5 / SDXL | Common no-grad, demand-driven previews/metrics, non-copying preview views and schedule snapshots implemented; numerical loops remain separate |
| Z-Image | Diagnostics, preview work and scalar synchronization reduced; bounded runtime-FP8 reuse implemented |
| Flux2 | Preview work and scalar synchronization reduced; bounded runtime-FP8 reuse implemented with attention implementation/backend in its identity |
| Anima | Preview/metric demand, scalar snapshots and dual-callback composition implemented |
| Lens | Preview/metric demand, scalar snapshots and duplicate terminal-flush removal implemented |
| Krea2 | Preview/metric demand, scalar snapshots and duplicate terminal-flush removal implemented |
| Ideogram 4 | Preview/metric demand, scalar snapshots and duplicate terminal-flush removal implemented |
| MiniT2I | Predicted-clean sampler state retained; schedule synchronization consolidated |
| SenseNova U1.5 | Existing callback-conditional preview retained; schedule synchronization consolidated |
| LTX-2.3 | Diffusers pipeline owns most denoising; common no-grad boundary is defensive; phase staging requires GPU measurement |
| MiniMax-H3 | Predicted-clean video latent is already callback-conditional; common boundary is defensive; large component staging requires GPU measurement |
| ACE-Step 1.5 | Common boundary is defensive; staged LM/DiT/VAE cleanup should be benchmarked rather than statically collapsed |
| MiniMax Music 3 | Autoregressive/depth-decoder residency dominates; common boundary is defensive; cache/offload trade-offs require long-form audio measurement |

## Adjacent correctness finding

Anima previously passed `progress_callback or step_callback` into all three
samplers. The explicit callback adapter now invokes both contracts and forwards
preview/metric demand predicates from the progress callback. Its calling
convention and step-only behavior have focused regression coverage.

## Measurement-dependent or non-equivalent ideas

The following are not approved as equivalent static cleanup:

- Cross-generation prompt-embedding caches. Keys must cover model/tokenizer,
  prompt parsing, maximum length, clip skip, adapter state and quantization;
  GPU caching also consumes persistent VRAM.
- Keeping the VAE resident from image encode through final decode. It avoids
  transfers but raises denoising peak VRAM.
- Changing attention backends, precision, quantization, `torch.compile`, tiled
  decode or interpolation implementations. These change numerical or kernel
  behavior and remain opt-in.
- Asynchronous preview decode. It changes CUDA stream contention and tensor
  lifetime and needs end-to-end measurement.
- Removing all phase-boundary `empty_cache()` calls. Measure repeated cold/hot
  generations and allocator fragmentation first.
- Replacing the SciPy latent-resize round trip with `torch.interpolate`. The
  interpolation result is not assumed numerically equivalent.
- Reusing a fixed Z-Image inpaint noise tensor. The current per-step random draw
  is suspicious for reproducibility and allocation cost, but changing it alters
  RNG/output semantics and belongs in a separate correctness review.
- Streaming video frames to both lossless and proxy encoders instead of calling
  `frames.tobytes()` for each output. This can reduce host RAM copies, not VRAM,
  and requires subprocess/codec failure tests.

## Implementation sequence

1. **Completed:** disable autograd for API generation.
2. **Completed:** remove hot-path diagnostics and detailed routine scans.
3. **Completed:** make preview and CFG metric work demand-driven and remove
   proven-safe SD preview copies.
4. **Completed:** snapshot denoise schedule scalars by architecture family.
5. **Completed:** remove only the three proven duplicate terminal allocator
   flushes; retain phase boundaries pending measurement.
6. **Implemented, measurement pending:** failure/unsupported quantization
   preserves original identity without cloning; successful runtime FP8 uses a
   one-entry-per-component cache with mode-switch, block-swap, runtime-INT8 and
   keep-hot lifecycle tests.
7. **Completed to the static-safe boundary:** extract callback demand,
   schedule-snapshot and callback-composition helpers; retain distinct
   numerical loops.

## GPU verification backlog

For each affected image family, run the same seed/configuration before and after
with preview disabled and with interval 1/4. Record output hash (bit-exact where
the arithmetic is unchanged), wall time after warm-up, peak allocated VRAM,
peak reserved VRAM, and synchronization-sensitive step timing. Repeat at least
three hot generations to reveal fragmentation. Video/audio families need one
short and one realistic-duration case; component staging and host RAM must be
recorded in addition to VRAM. Phase-boundary allocator changes and persistent
quantization caches are not complete until these measurements pass.

For runtime FP8 specifically, measure peak and steady-state host RAM with the
implemented source-plus-one-copy bound, and compare first versus repeated
generation startup time. The CPU tests already cover identity reuse, one-entry
eviction, source restoration, block-swap offload, runtime-INT8 discard and the
four Flux2/Z-Image text-encoder/transformer move paths; the real model sizes and
host allocator behavior still require observation.
