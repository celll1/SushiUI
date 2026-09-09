# Training iteration VRAM validation plan

## Purpose

The changes below can improve iteration latency, but static reasoning cannot
establish that they are regressions-free. They alter allocator behavior,
transfer overlap, reduction order, or compiled graph shape. Do not implement
them without an idle target GPU and a repeatable training configuration.

This plan deliberately excludes model-quality experiments: every candidate
must first preserve the same batches, sampled timesteps, loss definition, and
optimizer-step count.

## Shared measurement protocol

Use one representative LoRA run and one full-parameter run for the affected
architecture. Pin the seed, dataset snapshot, bucket list/order, batch size,
precision, gradient accumulation, MNT count, and all offload settings.

For each baseline/candidate pair:

1. Discard model load, cache fill, compile warm-up, and the first 20 iterations.
2. Measure at least 200 steady iterations and two swap-buffer refill boundaries.
3. Record median/p95/p99 iteration wall time, forward/backward/optimizer CUDA
   event time, host wait time, allocated/reserved/peak VRAM, driver free VRAM,
   shared GPU memory, refill time, and every allocator purge duration.
4. Run once from a cold process and once after all resolution buckets have been
   visited.
5. Compare the loss series, gradient norms, optimizer-step count, and final
   trainable tensors. Use the existing dtype-appropriate tolerances; require
   exact equality where the candidate does not change arithmetic order.

Reject a candidate on any new OOM, sustained shared-memory spill, unbounded
reserved-memory growth, skipped update, non-finite value, or checkpoint/resume
divergence. Keep raw measurements with the implementation commit.

## Candidate A: pressure-based allocator purge

### Current behavior

`BaseTrainer.train` calls `torch.cuda.empty_cache()` whenever the resolution
bucket changes and between MNT iterations. Globally shuffled buckets can make
the bucket-change call occur almost every batch. The policy protects Windows
from allocator reservation growth and catastrophic shared-memory spill.

### Candidate

- Preserve forced purge on OOM recovery and major model-residency transitions.
- Replace bucket-change and MNT unconditional purges with a policy using
  `reserved - allocated`, driver free memory, recent reservation growth, and a
  configurable safety margin.
- Cache device constants and sample driver memory no more often than required.
- Emit purge reason and duration so the policy remains observable.

### Acceptance

No OOM or shared-memory spill in the all-buckets pass; p95 iteration latency
must improve, and peak reserved VRAM must stay within the baseline safety
margin. Test Windows WDDM separately from Linux because expandable segments
are not enabled on Windows.

## Candidate B: pinned swap-buffer prefetch

### Current behavior

Text embeddings and latents are stored in ordinary pageable CPU tensors, while
their device copies request `non_blocking=True`. Pageable source memory cannot
provide the intended asynchronous H2D overlap.

### Candidate

- Keep the large swap buffers pageable.
- Pin only the next one or two batches in a bounded staging ring.
- Copy on a dedicated CUDA stream, record an event, and wait at the latest
  consumer boundary.
- Fall back to the existing synchronous copy when pinning fails or the host
  memory cap is reached.

### Acceptance

Show reduced H2D wait without increasing total pinned memory beyond the fixed
cap. Values copied to the compute stream must be exact. Test text-only,
latent-only, simultaneous refill, video/audio auxiliary payloads, and process
shutdown with an in-flight prefetch.

## Candidate C: single-pass gradient norm and clipping

### Current behavior

The run-invariant parameter/component census is cached. The non-fused optimizer
path still reduces those gradients for component norms, then `clip_grad_norm_`
traverses the gradients again to compute the total norm and apply scaling.

### Candidate

- Accumulate component and total squared norms in one device reduction.
- Apply the clip coefficient using the same epsilon, non-finite handling, and
  sparse-gradient policy as the installed PyTorch version.

### Acceptance

Benchmark LoRA and full-parameter runs separately. Require matching component
norms and clip decisions within dtype tolerance, matching skipped-step behavior
under GradScaler, and no difference in final parameters beyond that tolerance.
The full-parameter optimizer phase must improve materially; otherwise retain
the simpler library implementation.

## Candidate D: bucket-aware scheduling

### Current behavior

Batches are shuffled globally after being formed per resolution bucket. This
maximizes shape changes and can increase allocator churn and `torch.compile`
specialization work.

### Candidate

Randomize bucket order and within-bucket batches, then train short runs of the
same shape. Treat this as an opt-in scheduling feature, not an equivalence
refactor: it changes sample order and gradient trajectories.

### Acceptance

Only consider after allocator-policy measurements. Compare throughput and model
quality across multiple seeds; document the changed stochastic semantics in
the API and training guide before enabling it anywhere.

## Candidate E: `torch.compile` shape policy

### Current behavior

Compilation is opt-in and lazy per encountered shape. A large bucket set can
pay repeated compile costs or fall back after an unsupported shape.

### Candidate

- Measure eager, `dynamic=None`, and a supported dynamic-shape configuration.
- Record graph count, compile time, fallback reason, and steady kernel time.
- Consider compiling only the most frequent shapes and using eager execution
  elsewhere if the backend supports a stable dispatch boundary.

### Acceptance

Include compilation time in an epoch-level result, not only steady-state
iteration time. Require identical checkpoint keys and optimizer parameter
identity, plus no backward-only compilation failure.

## Completed static follow-up

The separate static pass deferred ControlNet/outpaint diagnostics, crop-decode
metrics, CFG-split monitoring, and convergence-latent transfer until the shared
post-backward synchronization point. It also cached the run-invariant gradient
parameter/component census. The remaining candidates in this document still
require the measurement protocol above.
