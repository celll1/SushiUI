# Training memory and iteration overhead audit — 2026-10-05

## Scope and evidence

This audit covers the shared LoRA/full-training loop, its 17 registered
architecture handlers and their dedicated step routes, encoder refill ownership,
activation/weight offload, optimizer state, diagnostics, CPU prefetch, snapshots
and checkpoint seams. Dedicated YuE2 token training and VAE decoder training
are separate owners; shared-loop changes do not imply they inherit every fix.
This is a source audit with CPU regressions, not a GPU benchmark of 17 models
or an exhaustive generation/UI performance audit.

Run 172 was stopped at the start of the investigation and became active again
during the audit. No running training was interrupted, no settings were edited,
and no new model or training subprocess was launched. CUDA initialization is
stubbed in this audit's import/test process. Native GPU measurements remain a
subsequent gate; an already-running child may have imported earlier code.

The immediately preceding change, `b4a46542`, coalesces ordinary non-paged
AdamW8bit waits, deduplicates identical saved activations and shares norms with
monitor-only probes. Its isolated GPU measurements are recorded in
[Ming design](../guides/MING_IMAGE_01_DESIGN.md). Those measurements are not a
native full-step acceleration claim.

## Implemented equivalent changes

| Finding and owner | Change | Applicable workloads |
|---|---|---|
| `base_trainer.py`: Ming Full trims allocator cache before every optimizer update | Trim only once before lazy moment initialization | Ming Full, including ordinary AdamW8bit |
| Shared loop releases allocator pools between same-bucket MNT iterations | Remove those per-MNT trims; retain bucket/encoder transitions, periodic trims for variable video/audio shapes, save/sample boundaries and OOM cleanup | Shared image/video/audio training loop |
| EMA initialization casts/transfers, then clones | One `.to(..., copy=True)` produces an independent FP32 shadow | Shared weight EMA when enabled |
| Ming gradient capture/restore transfers, then clones | One independent transfer/copy | Ming pending-gradient resume |
| VAE checkpoint snapshot transfers, then clones | One independent CPU copy, including EMA exports | Dedicated VAE training |
| Parameter tracker copies the reference twice at startup | Share read-only tensors in separate reference/previous lists initially | `param_tracking=true` |
| Parameter tracker builds a full third FP16 snapshot at each metric interval | Copy, reduce and replace one previous parameter at a time | `param_tracking=true` |
| CPU FP16 parameter tracking can alias live weights via `.cpu().to(fp16)` | Force independent snapshots | CPU FP16 diagnostics; also simplifies other dtype/device paths |
| Host optimizer allocator creates pageable zeros, then pins a second copy | Construct zeroed pinned CPU storage directly | Host-resident ring-buffer optimizer state |
| Ming latent windows retain normalized RGBA references after VAE encoding | Release those copies while retaining encoded references and native vision inputs | Ming edit/Layer inputs with references |
| Conditioning-only preparation generates unused VAE reference pixels | Skip their resize/normalization; native token/vision processor inputs remain identical | Ming reference-conditioned text encoding |

These changes do not reduce the number of optimizer updates, change MNT/GAS,
change clipping/precision/loss weights, or enable disk caches. Reserved allocator
VRAM may remain higher within a bucket: reuse is intentional, while allocated
tensor memory and shape-transition cleanup retain their existing semantics.

## Memory accounting

For `P` tracked elements, FP16 reference/previous snapshots previously held
`4P` bytes at initialization and a metric temporarily held `6P` bytes plus
FP32 reduction scratch. Initialization now holds `2P` bytes; after the first
metric, persistent storage remains `4P` bytes, with scratch bounded by the
largest parameter instead of an additional full model snapshot. For an
unchanged trainable parameter set, this bound holds at every metric. Dynamic
trainability changes retain the legacy full-snapshot path. For an
illustrative 8-billion-element tracked model this is about 14.9 GiB initially
instead of 29.8 GiB, and removes the additional 14.9 GiB metric snapshot.
Run 172 has parameter tracking disabled, so this saving is not attributed to it.

Each retained RGBA reference costs `16 * width * height` bytes at FP32. A
256-item window with one 1024-square reference per item could otherwise retain
4 GiB of these copies, separate from required vision inputs and latent references.
Reference-free Design training does not incur that cost.

Single-copy EMA/runtime/export changes remove one transient tensor allocation
at a time, not an entire persistent EMA, optimizer or gradient footprint.
Direct pinned allocation likewise removes the temporary pageable allocation;
it does not change the persistent host-state budget.

## Remaining opportunities, ordered by expected impact

1. **Activation restore overlap and pageable fallback.**
   `activation_dispatcher.py` restores on demand. The prior native run moved
   roughly 21–23 GiB of activations with only 2 GiB async capacity. This remains
   the strongest transfer candidate. A lookahead needs a bounded restored-GPU
   budget, correct repeated-unpack/checkpoint ordering, cancellation and event
   ownership. A larger pinned arena also needs host headroom during refill,
   optimizer parking and checkpoint save, not just idle RAM. Neither is enabled
   speculatively. Measure pack/unpack/compute separately on native Ming first.
2. **EMA checkpoint save duplicates live weights on their current devices.**
   `_save_ema_checkpoint` retains a full live-weight stash to guarantee rollback.
   Moving that stash to CPU could save VRAM but increase DRAM and transfer time;
   a parameter-by-parameter/sharded save needs an atomic rollback/export design.
   Preserve failure recovery before replacing this mechanism.
3. **Frozen conditioning transport.**
   Ming packs padded dual streams into a common tensor; CFG null conditioning
   materializes a zero tensor and output for each stream. These are candidates
   for a smaller transport and scalar/masked-fill zeroing. Trainable conditioning
   must retain both gradient routes, microbatch slicing and graph replay.
4. **Repeated source validation and preprocessing.**
   Ming verifies content hashes during preparation/refills and pixel loading.
   A bounded shared decode/validation result could avoid repeated I/O. Replacing
   hashes with size/mtime alone weakens source-drift rejection and is not applied.
   Frozen VAE batch assessment already showed no Ming FP32 speed advantage;
   see [VAE batching assessment](VAE_ENCODE_BATCHING_ASSESSMENT.md).
5. **Reduction and normalization kernels.**
   Global clipping still reads gradients separately from diagnostic norms.
   Native normalization and FP32 losses also create intermediates. Fusing these
   may help bandwidth/activation memory but needs overflow, dtype, reduction
   order and resumed-update comparisons. Lowering FP32 loss/normalization
   precision merely to save memory is not an equivalent refactor.
6. **Refactoring ownership and measurement.**
   `base_trainer.py` still owns many encoder/refill, MNT and metric transitions.
   Further extraction should preserve phase ownership and use one reusable
   timing/byte collector rather than synchronizing every layer or tile.
   Existing partition timers synchronize only when profiling is enabled;
   regular loss extraction already supplies the synchronization seam for
   deferred REPA/metric reads. Do not add another blanket device wait.

## Mechanisms checked and retained

- Weight streaming already shares `offload_transfer_engine.py`, with fixed
  slots, event waits and mutable writeback. A second weight prefetch system
  would duplicate ownership; see [block swap](../../backend/core/memory_management/BLOCK_SWAP.md).
- Activation async storage has a fixed pinned cap and falls back instead of
  growing with bucket shapes. Reusing a CPU slot before all saved consumers
  finish is unsafe; keep its step lifetime.
- Encoder refill parks the main model's optimizer state and preserves original
  devices. Frozen Ming vision is staged only when references require it.
- Pixel lookahead is single-worker and byte-bounded; CPU text queues are
  depth-bounded. Depth is not a byte cap for arbitrarily large captions/layers.
  General text-worker timed shutdown can still leave an in-flight CPU encode;
  a stronger shutdown contract requires cooperative encoder cancellation.
- YuE2 uses native token training, not the image VAE/refill loop. MiniMax Music 3
  is generation-only and is outside training optimizations. Tagger/style/VAE
  loops have distinct objectives and cannot inherit image conditioning changes.
- Integrity validation, checkpoint rollback, pending-gradient persistence and
  OOM recovery are kept even where they have measurable costs.

## Follow-up implementation: activation transfers and EMA export

Async activation offload now reserves two staging planes inside the existing
pinned budget, each capped at 256 MiB. Saved activations exceeding persistent
pinned storage use bounded GPU-to-pinned-to-pageable copies; oversized or
noncontiguous saves retain synchronous fallback. A single speculative reverse
save-order restore overlaps H2D with backward compute, observes a 2 GiB free
VRAM margin, and stops speculating after eight attempts with under 25% hits.
With the run's 2 GiB pinned budget, persistent storage is 1.5 GiB and each
staging plane is 256 MiB. Pending source tensors and the speculative restore
are individually bounded at 512 MiB; the pinned budget does not grow.

EMA export stores live weights on temporary disk one parameter at a time,
then restores the same parameter objects even after export failure. It avoids
a complete GPU live-weight clone and a complete additional RAM snapshot.
Temporary storage must hold the live trainable weights plus 1 GiB margin;
failed restoration retains rollback files and aborts instead of continuing
with EMA weights. This trades extra disk I/O for lower memory consumption.

Interrupted and emergency saves now prune prior generations only after model,
training state and optimizer/EMA state all succeed. Stopping run 172 exposed
the previous bug: ENOSPC during optimizer save was followed by deletion of the
old optimizer. Step 491 model/state survived, but its optimizer must start
fresh; the new guard prevents this deletion on subsequent save failures.

Focused CUDA/CPU transfer, exact-gradient/retained-graph, EMA rollback and
checkpoint retention tests pass: 67 cases. Final changed-module compilation
and real imports pass without CUDA initialization. A short 16-sine-chain
microbenchmark measured 22.22 ms before and 23.02 ms after (five-run medians,
64 MiB pinned budget): synchronous fallback fell from 72 MiB to zero, but
staging overhead made this short-compute case 3.6% slower. This is not evidence
of universal speedup; native model measurements are required.

Run 172 resumed from step 491 and completed steps 492–498 with finite losses
(0.263–0.900), unchanged run parameters, and the configured 200-step optimizer
rewarmup. For latent bucket 152x224, previous synchronous fallback was
19.706 GiB of 21.706 GiB transferred; after staging it was 3.743 GiB of
21.460 GiB (about 81% less fallback). Restore lookahead recorded 348 hits and
288 misses; peak allocation was 36.76 GiB versus the prior 36.45 GiB. Bucket
168x216 transferred 24.022 GiB with 3.758 GiB fallback and 37.13 GiB peak.
These are different training samples and fresh optimizer moments: they verify
native operation and bounded memory, not an identical-input throughput gain.
Post-initialization step intervals ranged from 4.2 to 12.5 seconds across
different buckets. The run remains active for normal training.

## Prior audit validation

Changed backend files pass `py_compile` and real imports with CUDA initialization
disabled. CPU gates cover repeated parameter metrics against the full-snapshot
oracle in FP32/BF16/FP16, snapshot independence, unchanged native prompt/vision
inputs, RGBA/frame order, swap windows, cache identities, pending-gradient/MNT
resume and host-state ownership. The VAE snapshot gate covers live and EMA
exports.

Validation results: 130 native/snapshot/prompt/batch/buffer/resume/cache cases
passed in the initial CUDA-disabled collection. The final focused snapshot
suite passed eight cases, including three added VAE/dynamic-trainability cases.
The independent host-state collection passed 51 cases and eight subtests;
five GPU-only cases were deselected. The initial combined collection had ten
host-pinning failures because CUDA availability was disabled by the native CPU
fixtures. Isolating the installed driver's CPU pinned allocation resolved them
without weakening residency checks. GPU update kernels were excluded from
the final host-only collection.

No API/default/schema or frontend surface changed. There is no frontend build
gate for this backend-only change. Per-architecture GPU timing and native
allocator-reservation/peak comparisons remain unmeasured in this audit.
