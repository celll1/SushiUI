# DiT partition training and memory-execution plan

Status: **planned; implementation not started by this document**

This plan separates four controls that are currently easy to conflate:

1. activation-peak prediction;
2. saved-activation CPU offload;
3. execution-batch reduction by micro-batch splitting; and
4. gradient checkpointing depth.

It also defines a bounded asynchronous cache experiment for quantized frozen
base weights and a path for turning Qwen-Image 2.1 complete-coverage partition
training into an architecture-gated DiT facility. The shipped Qwen behavior is
the reference implementation, not a claim that one partition algorithm is
valid for every transformer architecture.

## 1. Current behavior and terminology

### 1.1 Activation dispatcher

`ActivationDispatcher` predicts the activation requirement for a bucket and
returns one of three decisions:

* `fast`: execute without activation offload;
* `offload`: execute with activation offload; or
* `escalate`: predicted to exceed the live budget even after offload.

The dispatcher is a predictor and policy selector. It is not itself the
offload implementation. The caller currently converts `escalate` into a
smaller execution micro-batch when possible while retaining the logical batch
through gradient accumulation. At batch 1 there is no smaller micro-batch, so
the caller can only use the strongest offload path or fail.

This distinction must become visible in configuration. Enabling prediction
must not implicitly authorize changing the execution batch contract.

### 1.2 Current activation offload is synchronous

The generic saved-tensor path in
`backend/core/memory_management/activation_dispatcher.py` currently performs:

```text
forward save: GPU tensor -> pageable CPU tensor, blocking copy
backward use: pageable CPU tensor -> GPU tensor, blocking copy
```

Both copies use `non_blocking=False`. No transfer stream, CUDA event, pinned
buffer ring, or prefetch window is involved. This is the **synchronous** mode.

This is separate from architecture-specific controls named
`cpu_offload_checkpointing` and `async_cpu_offload_checkpointing`. Those
controls are implemented for selected architectures and must not be described
as modes of the generic activation dispatcher.

Synchronous activation offload has useful properties:

* simple tensor lifetime and OOM recovery;
* value-exact transfer without stream-order ambiguity;
* no long-lived pinned-host allocator cache; and
* no extra GPU staging ring beyond the restored tensor.

Its main cost is that D2H and H2D latency is directly on the critical path.
Compute does not overlap the copies.

An asynchronous implementation would use a bounded pinned-host buffer ring,
one or more dedicated CUDA transfer streams, non-blocking copies, and CUDA
events before a restored tensor is consumed or a ring slot is reused. It can
overlap transfers with independent compute, but introduces:

* pinned host-memory residency and allocator-lifetime pressure;
* GPU staging/headroom requirements;
* event and slot-lifetime correctness requirements;
* possible PCIe or memory-bandwidth contention with kernels; and
* more difficult checkpoint-recompute and OOM-retry behavior.

It is not automatically faster. It should ship only when measured overlap
exceeds its staging and synchronization cost for the target workload.

### 1.3 Current Qwen checkpoint controls

Qwen partition training already exposes
`qwen_partition_gradient_checkpointing_blocks` in the UI. It accepts 0 through
32; an empty value inherits the base checkpoint count. The automatic partition
path now preserves that base count. It no longer scales 24 blocks down to
12--15 based on the region token count.

The base Qwen count is currently selected internally:

* 16 blocks for configured resolutions up to 1024;
* 24 blocks up to 1536; and
* all 32 blocks above 1536,

when cached ConvRot backward weights are used. An internal
`qwen_gradient_checkpointing_blocks` override exists, but it is not a versioned
API/UI field. Thus the answer to “can the user select the checkpoint layer
count?” is: **yes for the partition override, not yet through the supported UI
for the base count**.

The existing frontend text saying that an empty partition value “scales
automatically” is stale and must be corrected when implementation begins.

## 2. Decisions

### 2.1 Make batch reduction an independent option

Add a versioned training parameter:

```text
activation_dispatch_allow_batch_reduction: bool = true
```

The compatibility default is `true`, preserving current runs. For a known
dataset whose buckets have already been demonstrated to fit, the UI will allow
the user to turn it off.

When `false`:

* a predicted `escalate` decision reuses the existing fused-backward ladder:
  execute the original batch with activation offload and lower the saved-tensor
  threshold to `max(256 KiB, configured_threshold / 16)` rather than
  pre-emptively splitting it;
* a real CUDA OOM may advance the **same batch** by one further threshold rung,
  dividing the active threshold by 16 down to the existing 64 KiB floor;
* the retry must not switch to a smaller micro-batch; and
* if the most aggressive same-batch rung fails, the strict no-reduction mode
  fails loudly rather than silently changing execution batch size.

This single control covers both prediction-driven and OOM-recovery splitting.
Otherwise a user could disable false-positive proactive splitting yet still
receive a hidden batch change through the reactive path.

This is a generalization of the fused-backward threshold ladder, not a second
offload-retry implementation. There is one reactive retry in a forward/backward
call, but the initial proactive execution may already be the lowered 256 KiB
rung; the reactive retry can therefore reach 64 KiB. An identical retry at the
64 KiB floor is not attempted. The new strict mode differs from the existing
legacy fused terminal policy only at exhaustion: strict mode raises, while the
legacy path may mark the bucket unfittable and skip it.

The following decision table is authoritative:

| Dispatcher decision | Reduction allowed | Execution |
|---|---:|---|
| `fast` | either | Original batch, no offload |
| `offload` | either | Original batch, activation offload |
| `escalate`, non-fused | yes | Largest predicted-safe micro-batch, accumulated to one logical batch |
| `escalate`, non-fused | no | Original batch, lowered-threshold activation offload; no proactive split |
| `escalate`, fused backward | either | Original batch, lowered-threshold activation offload; splitting is always forbidden because hooks can apply updates during each backward |
| Actual OOM, non-fused | yes | Existing OOM retry may micro-split |
| Actual OOM, non-fused | no | Advance the same-batch threshold ladder once, then fail at its floor |
| Actual OOM, fused backward | either | Never micro-split; use the same offload ladder, then the configured terminal policy |

At batch 1 the option cannot reduce the batch in either mode. Diagnostics
still need to distinguish “no reduction permitted” from “no smaller batch
exists.”

The effective optimizer batch and logical progress remain unchanged when
micro-splitting is allowed. Nevertheless, floating-point reduction order and
performance can change, so the actual execution micro-batch must be recorded.

### 2.2 Keep checkpointing conservative by default

Partitioning must never automatically spend its activation-memory saving by
reducing the number of checkpointed blocks. The automatic rule is:

```text
partition checkpoint blocks = resolved base checkpoint blocks
```

A lower value is accepted only as an explicit user override. The UI must show
the resolved base value and warn when the override is lower, because the
result can be both larger and slower after WDDM spill.

Expose a supported base-depth parameter for architectures that declare partial
checkpoint support:

```text
dit_gradient_checkpointing_blocks: integer | null = null
```

`null` selects the architecture policy. `0` means no transformer blocks are
checkpointed, and the architecture's block count means all blocks are
checkpointed. Validation uses capability metadata rather than a hard-coded
maximum of 32.

The architecture-neutral partition override will be:

```text
dit_partition_gradient_checkpointing_blocks: integer | null = null
```

`null` inherits the resolved base count. During migration, the existing Qwen
field remains an accepted alias for old requests and saved runs; conflicting
old and new values are rejected. New run configurations write the canonical
DiT names only.

### 2.3 Keep activation prediction, offload, and checkpointing separately visible

The UI will group the controls but not collapse them into one checkbox:

* **Activation prediction/dispatch**: whether bucket-aware prediction is used;
* **Allow execution-batch reduction**: whether prediction/OOM recovery may
  micro-split;
* **Activation transfer mode**: synchronous initially, asynchronous only after
  its gate passes; and
* **Gradient checkpointing / checkpointed blocks**: recompute policy.

This permits, for example, prediction plus synchronous offload with strict
no-reduction, or checkpointing without activation offload.

## 3. Activation-offload transfer modes

The initial API addition for transfer behavior is:

```text
activation_offload_transfer_mode: sync | async
```

`sync` remains the default. `async` is experimental until its memory and
correctness gates pass. This field controls only generic saved-activation
offload; it does not rewrite architecture-specific checkpoint-offload flags.

### 3.1 Asynchronous implementation constraints

The implementation must not simply change `non_blocking=False` to `True`.
Correct asynchronous offload needs:

1. a size-bounded pinned-host arena with reusable slots;
2. a D2H stream and a restore/prefetch stream;
3. a recorded event after each D2H copy before the source may be released;
4. an event/wait before backward consumes a restored tensor;
5. a slot-reuse event so a host buffer is never overwritten early;
6. an explicit maximum pinned-host byte budget; and
7. teardown and OOM cleanup that synchronizes outstanding transfers.

Prefetch order is supplied by autograd demand and is not always known far
enough ahead to overlap H2D. The first prototype may overlap D2H saves and
restore only tensors whose backward order can be determined safely. If no
overlap window exists, it must fall back to the synchronous path rather than
retain additional unbounded buffers.

### 3.2 Transfer acceptance gates

For identical inputs and stochastic state:

* loss and trainable gradients must match the synchronous offload path within
  the existing BF16 tolerance;
* no tensor may be consumed before its restore event;
* pinned host residency and GPU staging residency must stay below configured
  bounds across changing bucket shapes;
* cancellation, OOM retry, exception, and run shutdown must release the arena;
* WDDM shared-memory growth must reach a steady bound rather than accumulate
  per shape; and
* median logical-image time must improve on at least one representative
  offload-bound workload without regressing the non-offloaded path.

## 4. Bounded ConvRot BF16 backward-cache prefetch

The Qwen measurements establish two useful endpoints for the same 60x104
latent, rank-128, fixed-2/global-adapter workload with 24 checkpointed blocks:

| Base/backward policy | Transformer step | Peak |
|---|---:|---:|
| INT8 ConvRot + full 13.252 GiB BF16 cache | 3.334 s | 28.14 GiB |
| Dense BF16 base | 4.561 s | 21.47 GiB |

The full cache is a speed policy and is heavier than the BF16 base. Synchronous
per-layer BF16 reconstruction is not acceptable: previous measurement reduced
memory but more than doubled transformer time. The remaining experiment is a
bounded cache whose next entries are reconstructed ahead of use.

### 4.1 Proposed controls

These remain Qwen/ConvRot-specific because they describe a particular
quantized kernel and backward contract:

```text
qwen_convrot_training_forward: auto | cached_bf16 | prefetch_bf16 | transient_bf16 | dequant
qwen_convrot_backward_cache_blocks: integer     # bounded resident block slots
qwen_convrot_backward_prefetch_depth: integer   # scheduled blocks ahead
```

The prototype is selected explicitly until the prefetch gate passes. It then
becomes `auto`; the full cache remains an explicit comparison/diagnostic mode.
A byte budget may replace the block count internally, but the resolved block
count and bytes must both be reported.

### 4.2 Execution design

The first implementation should reconstruct BF16 weights on GPU from the
resident ConvRot representation. Copying a full BF16 cache from CPU every
iteration is not the preferred first design because it replaces dequant work
with large PCIe traffic and pinned-host residency.

The cache is scheduled at transformer-block boundaries:

* allocate a fixed ring of BF16 block-weight slots;
* reconstruct block `n-1` on a low-priority prefetch stream while backward for
  block `n` performs work that does not need the same slot;
* record a ready event per slot and wait before the block consumes it;
* record a consumed event before reusing the slot;
* follow reverse block order during backward; and
* include checkpoint recomputation in the schedule rather than assuming one
  forward and one backward visit per block.

Linears outside the repeating transformer blocks must be inventoried. They may
use a small permanent cache only when its size is reported separately; they
must not create an unbounded exception to the ring budget.

The dequant kernel consumes SM and memory bandwidth, so overlap is a measured
question. Prefetch depth greater than one is not assumed beneficial.

### 4.3 Cache acceptance gates

Compare `full`, `prefetch`, `transient`, and dense BF16 using the same model,
bucket, adapter rank, partition plan, checkpoint count, and warmed iterations.
Record:

* persistent base/cache allocation;
* ring block count, bytes per populated block, outside-block bytes, and the
  theoretical full-cache bytes avoided by that ring size;
* activation peak above that persistent floor;
* absolute peak allocated and reserved;
* dequant, wait, forward, backward, and logical-step time;
* percentage of prefetch time hidden by compute; and
* loss, prediction delta, and every trainable gradient against `full`.

The initial success gate is:

* at least 4 GiB lower absolute peak than the full-cache path;
* median transformer step no more than 15% slower than full cache;
* faster than the measured dense-BF16 path on the matched workload; and
* no material numerical difference from full cache beyond normal BF16 kernel
  variation.

Failure of that gate leaves `full` and dense BF16 as the supported choices.

## 5. Architecture-cross-cutting DiT partition option

The user-facing option must not be named as a Qwen-only feature. Canonical new
run fields use a `dit_partition_` prefix:

```text
dit_partition_training_enabled
dit_partition_mode                         # off | fixed | adaptive
dit_partition_fixed_count
dit_partition_memory_fraction
dit_partition_token_fallback
dit_partition_elective_probability
dit_partition_min_core_side
dit_partition_max_regions
dit_partition_split_ratio_min
dit_partition_split_ratio_max
dit_partition_halo_tokens
dit_partition_sigma_adaptive
dit_partition_position_sidecar
dit_partition_seed
dit_partition_gradient_checkpointing_blocks
dit_partition_profile
dit_partition_global_adapter_enabled
dit_partition_global_rank
dit_partition_global_tokens
```

Existing `qwen_partition_*` fields remain read-compatible aliases for saved
runs during migration. The API, OpenAPI schema, frontend request type, training
configuration extraction, defaults, and run-resume logic must move together.

### 5.1 Shared and architecture-owned pieces

Reusable code moves under `core/training/partition/`:

* deterministic rectangle planner and coverage validation;
* halo expansion and non-overlapping loss-core accounting;
* logical-image/region execution bookkeeping;
* area-weighted objective assembly;
* common diagnostics and timing; and
* canonical configuration parsing and alias migration.

Each architecture supplies a `DiTPartitionAdapter` capability implementation:

```text
latent_grid_shape
token_alignment_contract
build_full_position_metadata
slice_target_and_conditioning
forward_region
select_prediction_core
resolved_checkpoint_depth
approximation_boundary
```

The adapter must state whether text/condition tokens form a prefix, whether
the prefix can attend to target tokens, how image positions are encoded, and
which attention edges partitioning removes. Unsupported architectures reject
the option during configuration validation before model loading.

### 5.2 Candidate architecture audit list

Only Qwen-Image 2.1 is the implemented reference. The following are audit
candidates, not compatibility claims:

| Architecture group | Initial assessment | Required audit |
|---|---|---|
| Flux2 | Requires a different approximation | Text and image use bidirectional joint attention, so each region changes the text-stream state; audit a joint-stream objective rather than reusing Qwen's shared-prefix oracle |
| Z-Image | Candidate | Stream layout, position contract, packed attention mask, training-op ownership |
| Anima and Lens | Candidate | Joint/dual stream boundary, position IDs, architecture-specific checkpoint offload interaction |
| Krea2 and Ideogram4 | Candidate | Proprietary tensor layout represented by local implementation, conditioning flow, output-token selection |
| MiniT2I | Candidate | Target grid mapping, attention topology, benefit at its typical sequence lengths |
| SenseNova U1.5 | High-risk candidate | MoT routing, branch ownership, mixed objectives, cross-region global semantics |
| SD1.5, SDXL, Chimera U-Net stage | Not this facility | Convolutional U-Net tiling requires a separate halo/receptive-field design |
| Video/audio transformers | Deferred | Temporal coverage, causal/audio coupling, and optimizer-step semantics need a separate design |

An architecture is enabled only after it passes a small block-diagonal oracle,
coverage/alignment tests, a real-GPU memory measurement, and a documented
quality-risk boundary. The existence of rectangular image tokens is not by
itself sufficient.

## 6. Diagnostics and persistence

Add the following run metrics and resolved configuration fields:

```text
activation_dispatch_decision
activation_dispatch_batch_reduction_allowed
activation_dispatch_requested_batch
activation_dispatch_execution_micro_batch
activation_dispatch_retry_kind
activation_offload_transfer_mode
activation_offload_bytes
activation_offload_wait_ms
activation_offload_overlap_ms
gradient_checkpoint_blocks_resolved
partition_gradient_checkpoint_blocks_resolved
convrot_cache_mode_resolved
convrot_cache_resident_bytes
convrot_prefetch_wait_ms
convrot_prefetch_hidden_ms
```

Dispatcher calibration state remains runtime state, but the policy and
resolved execution decisions are written to run diagnostics. Changing the
batch-reduction permission on resume is allowed and recorded because it is an
execution policy; it does not change adapter artifact structure. Changing
partition semantics or global-adapter structure continues to follow the
partition resume contract.

## 7. Implementation sequence

### Phase A: dispatcher policy separation

1. Add `activation_dispatch_allow_batch_reduction` OpenAPI-first and place its
   default only in `backend/api/param_defaults.py`.
2. Thread it through request parsing, config extraction, frontend types, and
   the memory UI.
3. Apply it to both predicted escalation and reactive OOM micro-splitting.
4. Generalize the existing fused-backward threshold ladder for strict
   no-reduction execution and add explicit terminal diagnostics; do not create
   a parallel retry mechanism.
5. Test `fast`, `offload`, and `escalate` with reduction on/off, including
   batch 1 and batch greater than 1.

### Phase B: checkpoint contract and UI correction

1. Add architecture capability metadata for partial checkpoint depth.
2. Expose `dit_gradient_checkpointing_blocks` and canonical partition override.
3. Preserve Qwen aliases for old runs.
4. Show the resolved base count in UI and warn on a lower partition override.
5. Remove the stale “scales automatically” description.
6. Assert that an automatic partition plan never lowers checkpoint depth.

### Phase C: bounded ConvRot prefetch prototype

Completed. The two-block/one-lookahead ring passed its gate on the matched
60x104 latent probe: 3.263 s and 15.76 GiB versus 3.165 s and 28.14 GiB for the
full cache in that measurement pair. The 3.1% time cost bought 12.39 GiB lower
absolute peak and remained faster than the 4.561 s dense-BF16 baseline. The
reported resident limit was 1.065 GiB (two 416 MiB blocks plus 0.252 GiB of
outside-block weights). Loss and prediction metrics were identical; the small
CUDA oracle was bitwise-equal for output and input gradient. `auto` therefore
selects the bounded prefetch path.

1. Inventory ConvRot linears by transformer block and outside-block ownership.
2. Add block-boundary scheduling hooks without changing full-cache behavior.
3. Implement a one-slot synchronous bounded ring as a lifetime oracle.
4. Add a dedicated asynchronous prefetch stream and events.
5. Benchmark depths 1 and 2 before admitting deeper prefetch.
6. Move `auto` from full cache to prefetch only after the acceptance gate.

### Phase D: asynchronous generic activation offload

Implemented behind the explicit `async` transfer mode; `sync` remains the
default. The implementation retains one fixed-size pinned byte arena across
steps, uses separate D2H and H2D streams with producer/ready events, records the
source on the D2H stream, and synchronizes both streams before arena reuse.
When a step exhausts the arena, remaining tensors use the synchronous pageable
path and report `sync_fallback_bytes`; pinned residency never grows with bucket
shape. CUDA value/gradient, arena-exhaustion, and exception-reuse tests pass.
The mode remains experimental until an architecture-level offload-bound timing
shows a benefit; the mechanism gate alone does not justify changing the default.
On the RTX 6000 Ada mechanism probe, async reduced median offloaded-step time
from 4.25 to 3.09 ms for the audio-shaped case, 17.76 to 8.81 ms for the
image-shaped case, and 19.58 to 8.68 ms for the video-shaped case. Loss matched
exactly and gradients stayed within the existing BF16 gate. These synthetic
results justify keeping the selectable implementation, but not making it the
default for real architectures.

1. Implement the bounded pinned-host arena and transfer engine.
2. Prove teardown/OOM/cancellation behavior.
3. Measure real overlap on offload-bound image and video workloads.
4. Expose `async` only after bounded-memory and correctness gates pass.

This phase is deliberately separate from ConvRot prefetch. One moves saved
activations across PCIe; the other reconstructs frozen BF16 weights ahead of
their backward use.

### Phase E: DiT partition extraction

Implemented for the audited boundary. Planner geometry, exact-once core
coverage, halo expansion, deterministic boundary variation, and row-major
region flattening now live under `core/training/partition/`. A
`DiTPartitionAdapter` declares prefix directionality, global-position support,
and input-token multiple; Qwen supplies the only implementation. Canonical
`dit_partition_*` API/config/UI fields replace Qwen spellings, while nullable
deprecated aliases keep old requests and saved runs readable. Capability
preflight rejects every other architecture before model loading; no candidate
was enabled by inference.

1. Introduce canonical `dit_partition_*` schema and Qwen aliases.
2. Extract only the planner, coverage/loss bookkeeping, and diagnostics first.
3. Implement Qwen's `DiTPartitionAdapter` without changing its math.
4. Re-run Qwen CPU oracle and matched GPU baselines.
5. Audit candidate architectures individually and enable none by inference.

## 8. Verification matrix

Every backend edit receives `py_compile` and a real import with CUDA
initialization stubbed as required by repository policy. API work also updates
`openapi.yaml` and schema/default tests.

The feature-level matrix is:

| Area | CPU/unit | GPU/integration | Failure test |
|---|---|---|---|
| No batch reduction | Decision table and config round-trip | Known bucket with false-positive prediction | Actual OOM fails without micro-split |
| Checkpoint depth | Bounds, alias conflict, inheritance | Same shape at auto/lower/all blocks | Unsupported architecture refuses |
| ConvRot prefetch | Ring lifetime/event state machine | Matched full/prefetch/BF16 benchmarks | Slot exhaustion and injected exception cleanup |
| Async activations | Pack/unpack value oracle | Multi-bucket steady host/GPU memory | Cancel/OOM during outstanding copy |
| DiT partition core | Coverage, halo, deterministic plan | Qwen memory/time/quality probes | Unsupported topology refuses before load |

## 9. Non-goals

This plan does not:

* claim that activation offload and activation dispatch are the same feature;
* make asynchronous transfer the default before measurement;
* treat fewer checkpointed blocks as an automatic partition optimization;
* promise that bounded cache prefetch beats either full cache or BF16 base;
* claim numerical equivalence between dense attention and hard partitioning;
  or
* enable partitioning on another architecture solely because it is called a
  DiT.
