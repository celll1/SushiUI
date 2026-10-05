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

## Debug VAE decode residency follow-up

Deferred debug decode now shares `debug_decode_residency.py` across Ming,
Qwen 2.1, SenseNova, SDXL Chimera and latent MiniT2I. It parks non-VAE model
parameters, buffers, accumulated gradients, optimizer state, EMA shadows and
idle block-swap slots on CPU, preserving object identity and each original
device. CPU master bindings stay intact. The VAE is released before restoring
training residency; restoration failure is fatal rather than an optional
preview warning. Decode itself remains optional if it fails after successful
restoration. This adds CPU RAM and PCIe transfer work on debug steps only.

MiniT2I formerly decoded during its forward while its graph was live. Its
CPU preview snapshots now decode after backward, before the training step
returns. Differentiable crop-decode losses and normal sampling are outside
this non-training scope.

35 focused tests pass, including native Ming debug/backward invariance,
decode/save failure cleanup, CPU VAE dtype/mode restoration, mixed CPU/GPU
buffers, EMA/slot identity, gradient preservation and exact next-update
equivalence for ordinary AdamW and actual 8bit AdamW. Compilation and real
imports of all changed modules pass with CUDA initialization disabled.
Run 172 stopped and saved a complete step-1274 bundle before this update.

Native verification: run 172 restored all 521 named optimizer-state tensors
from step 1274, with no fresh state and no run-parameter changes. At debug
step 1300, allocation after component parking and VAE loading was 0.52 GiB
(this is not decode peak). All three decoded WebPs were saved and training
continued through step 1303 with finite loss. The debug iteration took about
29 seconds, including CPU parking, decode and restoration; ordinary adjacent
iterations were about 4–8 seconds. Historical missing WebPs are not regenerated.

## Caption encoding: static assessment

The initial static review made no caption-path edits or GPU benchmarks.
The current batch-1 swap window processes 256 batches sequentially in
`MingTrainingBatchBuffer._fill_conditions`. Residency is grouped for the whole
window, so it does not reload the conditioner for every caption. The frozen
encoder runs without gradients and stages vision only for reference inputs.

Priorities for an implementation pass:

1. **Skip unused vocabulary projection.** Ming conditioning calls the full
   `BailingMoeV2ForCausalLM.forward` with default `num_logits_to_keep=0`, which
   projects every token into the 157,184-word vocabulary even though only
   hidden states are consumed. A hidden-state-only decoder path, or initially
   limiting logits to one token, avoids this work. Preserve layers 5/12/20 and
   the native query/direct streams. Qwen 2.1 also invokes a conditional-LM
   wrapper with default full-token logits; its installed forward offers
   `logits_to_keep`, so the same opportunity deserves a shared review.
2. **Independent encoder microbatches.** The text-only Ming fast path accepts
   multiple prompts but receives the training batch size (currently one).
   Grouping compatible captions into batches of 2–4 may amortize the many
   small expert launches. Native padding and direct-stream lengths must remain
   equivalent after splitting results: the direct projection has bias and
   there is no direct-stream mask in `MingConditioning`. Group equal token
   lengths and compatible geometry first; test against single-caption output.
3. **MoE dispatch.** Each MoE layer copies token counts to CPU with
   `.cpu().numpy()` and loops over nonempty experts in Python. The shipped
   MLLM has 20 layers and 256 experts per layer. This introduces repeated
   synchronization and many small GEMMs; grouped expert execution could help,
   but must support the current INT8 conversion and preserve routing/reduction
   semantics. Static inspection alone cannot establish the dominant cost.
4. **Preparation/transport.** Native learnable-token preparation copies IDs
   and masks GPU→CPU→GPU to manipulate Python lists, and direct-stream bound
   selection uses `.item()`. Prepare indices on CPU before transfer or use
   equivalent tensor operations. Each condition is copied back to CPU before
   the next caption; bounded pinned output buffering could overlap transfer.
5. **Bounded reuse and smaller intermediates.** Reuse frozen conditioning in
   RAM only under an exact prompt/geometry/reference/weight identity, with a
   byte cap. Swap mode currently consumes its entries and recomputes subsequent
   windows/epochs. Connector output requests every hidden state but uses only
   the last, while main-LLM selected-layer capture could retain just the needed
   layers. Repeated image-byte hashing during caption preparation is another
   candidate for reuse under the existing validated dataset-snapshot contract.

Ming performs more than caption embedding: the released model appends 256
learnable query tokens plus boundary tokens, runs its MoE MLLM, combines three
hidden layers for direct conditioning and runs a separate query connector.
Removing these native operations would change conditioning, not optimize it.
No speedup multiplier is justified until isolated equivalence and timing tests.

### Caption implementation and native checks

Ming now invokes the MLLM and query connector decoders directly. Both LM
wrappers previously computed unused vocabulary logits. Selected hidden layers,
query/direct streams and live conditioning gradients are retained. Sparse MoE
counting uses a one-dimensional scatter-add instead of a tokens-by-experts
allocation; expert dispatch still synchronizes counts to CPU and executes each
nonempty expert separately. This change does not claim to remove that bottleneck.
Qwen 2.1 limits its unused logits to one token while preserving the scoped
final-RMSNorm bypass. CPU native-model oracles match outputs and gradients exactly;
the combined Ming conditioning/quantized tests passed 40 cases and the actual
small Qwen3VL caption oracle passed one case. Changed modules pass compilation
and real imports without initializing CUDA.

Run #172 was stopped once and saved a complete model/optimizer/state at step
1439 for isolated native GPU checks. The packed Ming Design conditioner used
BF16, eager attention as in training, actual dataset captions, square 1024
geometry and native prompt resolution 1536. DiT, VAE and vision stayed off GPU.
Timings include CPU output transport, two warmups and three measured repetitions.

| Native input tokens | Path | Median seconds/image | Peak allocated GiB | Single-caption output comparison |
| --- | --- | --- | --- | --- |
| 425 | Original, batch 1 | 0.970 | 17.975 | Reference |
| 425 | Decoder only, batch 1 | 1.000 | 17.806 | Exact |
| 425 | Decoder + sparse counts, batch 1 | 0.965 | 17.806 | Exact |
| 425 | Optimized, batch 2 | 0.497 | 18.001 | Both exact |
| 425 | Optimized, batch 4 | 0.274 | 18.401 | Failed: query max absolute difference 1368 |
| 209 | Optimized, batch 2 | 0.488 | 17.877 | Both exact |
| 602 | Optimized, batch 2 | 0.505 | 18.114 | Failed: query max absolute difference 1335.5 |
| 810 | Optimized, batch 2 | 0.510 | 18.236 | Failed: query max absolute difference 520 |

All tested optimized **single-caption** outputs match the old LM-wrapper path
exactly, including the longer captions. The batch-1 timing difference is too
small/noisy to claim a speedup; vocabulary projection removal reduces peak
allocated memory by about 0.17 GiB in the 425-token case. Batching clearly reduces
per-image time but is **not adopted**: even equal native token lengths do not
guarantee numerical equivalence for this conditioner. The large discrepancy is
not attributed to padding or a particular kernel without a layer-level oracle.
Training batch size, ordering, conditioning cache identity and run configuration
remain unchanged. A future independent encoder microbatch option needs to resolve
these numerical discrepancies before API/UI wiring, rather than selecting a
limit from one successful shape.

Local reproducibility: `local/deployment/benchmark_ming_captions.py --execute`
and `--execute --pairs-only`, using the repository venv with training stopped.
JSON measurements are retained under `local/deployment/ming_caption_benchmark*.json`;
these local artifacts are not checkpoint payloads or dataset disk caches.

After applying the commit, #172 resumed from step 1439 with all 521 named
optimizer tensors restored and none starting fresh. Its remaining 171-item
latent/caption refill completed with the existing progress phases; the conditioner
was released before training (CUDA allocated 0.01 GiB at that phase boundary).
The first native optimizer update reached step 1440 with finite loss 0.584264
and gradient norm 0.370451. The first iteration's 324-second elapsed time includes
the initial swap refill and is not a steady-state per-step timing.

## Live I/O follow-up: run 172

The run remained active throughout this investigation. Its epoch-5 bucket logs
report 18.492–22.276 GiB packed to CPU per measured step, with 3.651–3.744 GiB
on the synchronous fallback path. These are **pack-side** bytes, not a measured
D2H+H2D total. The 160x160 bucket reports 348 prefetch hits and 284 misses;
the latter restores are discarded before the requested tensor is restored.
Hit/miss counts do not establish the wasted byte volume.

Nonblocking py-spy sampling separates two intervals:

- A 20-second, 49-Hz sample caught the step-1650 debug decode. Most main-thread
  samples were optimizer/model parking or WebP output. The staging worker was
  idle; this interval is not representative of normal training.
- A 30-second, 29-Hz sample after debug decode (starting after step 1656)
  captured 859 main/autograd-thread samples each. Main-thread synchronous
  `cpu.copy_` took 103 samples (12.0%); staging queue backpressure took 104
  (12.1%). The autograd thread spent 176 samples (20.5%) waiting for restore
  futures, split between speculative prefetch (89) and demand restore (87).
  Background DB/TensorBoard/progress threads were predominantly idle.

These are stack-residence estimates, not CUDA-event durations or a measured
GPU-idle percentage; do not add percentages across threads. py-spy reported ten
sampling errors, and could not resolve some line tables in the large trainer
method. Transfer-hook and worker frames were available. The raw local traces
are `local/deployment/ming172_io_speedscope.json` and
`local/deployment/ming172_io_ordinary_speedscope.json`.

Source inspection explains the sampled waits: one worker serializes D2H staging
and H2D restoration; a full pending-GPU budget blocks forward in `future.result()`;
D2H waits for each chunk before the pinned-to-pageable CPU copy. H2D similarly
waits before reusing its single staging slice. Speculative restores also share
this worker, and the current reverse-save heuristic misses often enough to
justify a better policy. Noncontiguous or oversized activations are excluded
from staged packing, but the current statistics cannot say which accounts for
the synchronous fallback bytes.

Next changes should measure useful/wasted H2D bytes and wait times, identify
fallback layout/size/storage ownership, then compare separate D2H/H2D lanes and
a better bounded restore order against the current implementation. Preserve
the 2-GiB pinned cap, source-version checks, checkpoint recomputation ordering,
repeated backward and error cleanup. Increasing the pinned arena or retaining
extra GPU activations without host/VRAM headroom gates is not a demonstrated fix.
No backend behavior or run configuration was changed by this follow-up.

### Transfer follow-up implementation

D2H staging and H2D restoration now use independent single-worker queues and
separate existing pinned slices. A mismatched lookahead is retained for up to
four unpack requests, so a nearby future use can consume it without another
restore; a held lookahead prevents another speculative allocation. Expired
lookahead is cancelled when still queued, or drained with failures propagated.
The low-hit-rate gate now requires at least half the completed predictions to
be useful after eight attempts. Misses count actual discards, not every deferred
request, so old/new hit ratios are not directly comparable.

Noncontiguous sources can use staged packing when their retained storage plus
contiguous GPU scratch fits the existing staging budget. The reshape runs on
the D2H stream after its producer event. Oversized storage/layout reservations
retain the synchronous fallback. The pinned arena remains capped at the run's
configured 2 GiB and the staging GPU reservation at 512 MiB; there is still only
one bounded speculative restore. Both queues are drained on errors and shutdown.

Per-bucket logs now report restored H2D bytes, discarded prefetch bytes,
strided staged/fallback bytes and CPU pack/restore-future wait seconds. These
CPU timers do not represent complete DMA durations or GPU compute time.

Validation: 27 CPU cases plus 356 bucket subtests; ten CUDA cases cover loss/
gradient equality, nondefault producer streams, strided inputs, checkpoint
recomputation, repeated backward, mutation rejection and pack failures. The CPU
suite additionally checks that restore failures drain both workers. Compilation
and actual imports passed. This changes the shared offload mechanism, but does
not establish native performance for every training architecture.

An isolated 2-GiB-arena transfer workload alternated baseline/new/baseline/new,
with three warmed repetitions per arm and bitwise-equal gradients. The workload
packed 832 MiB per iteration: synchronous fallback decreased from 384 MiB to
zero. Baseline medians were 0.162/0.183 seconds; new medians 0.166/0.170 seconds.
Peak allocated memory was 0.438/0.500 GiB respectively. The timing difference is
small and does not establish a native step acceleration; the additional memory
stayed within the staging cap. Local reproduction is
`local/deployment/benchmark_activation_lanes.py --execute`, with #172 stopped;
measurements are in `local/deployment/activation_lanes_benchmark.json`.

Run #172 was stopped once at step 1920 with complete model, optimizer and state
saved for GPU validation. Its configuration is unchanged.

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
