# Unified offload transfer design proposal (2026-09)

## Decision summary

Unify the **mechanism**, not the ownership semantics.

Block swap, layer offload, activation offload, and host-resident optimizer state
all need pinned host storage, bounded device slots, asynchronous copies, CUDA
event dependencies, prefetch, and a flush boundary.  They should share one
transfer engine.  They must remain separate policies because their data has
different lifetimes and write-back rules.

The current names obscure this boundary:

| Current term | Object being moved | CPU copy lifetime | GPU copy | Dirty/write-back rule |
|---|---|---|---|---|
| Block swap / layer offload | model block weights | whole model/run | needed for forward and backward | immutable for inference/frozen base; dirty after a full-FT update |
| Activation offload | saved autograd tensors | one graph/iteration | needed by backward consumer | never written back; discard after consume |
| AdamW/Lion "ring-buffer" | optimizer moments/codes | whole run and checkpoint | needed only for one update window | dirty after every optimizer update |

Therefore `LayerOffloadConductor`, `TransformerBlockOffloader`,
`FluxBlockOffloader`, and per-parameter optimizer `.cuda()` staging should not
remain independent implementations of transfer ordering.  Conversely, model
weights and optimizer state must not alias the same storage or share one
residency state machine.

## Proposed common engine

Introduce an architecture-neutral `OffloadTransferEngine` with these narrow
responsibilities:

1. own persistent host masters, grouped by dtype and mutability;
2. own a bounded ring of reusable device slots;
3. pack a logical item into one flat tensor per dtype plane, including
   quantization sidecars;
4. enqueue H2D/D2H on one transfer stream;
5. order producer -> copy -> consumer with CUDA events, never a hot-path global
   synchronize;
6. prefetch a caller-supplied sequence into free slots;
7. expose `flush(dirty_only=True)` for checkpoint and teardown boundaries;
8. keep transfer counters and event timing so overlap is measurable.

A conceptual interface is:

```text
register(key, tensor bundle, mutable)
acquire(key, consumer_stream) -> resident bundle + ready event
prefetch(keys, after_event)
release(key, dirty, producer_event)
flush(keys | all)
stats() -> bytes, waits, copy time, overlap, slot occupancy
```

`acquire()` may make the consumer stream wait on a transfer event; it must not
block the host.  `release()` may make the transfer stream wait on a producer
event; it must not copy mutable data before the update kernel completes.

### Multi-dtype tensor bundles

One flat buffer for the whole block is insufficient for mixed BF16/FP8 weights
and FP32/other quantization sidecars.  Each logical item should have one packed
plane per dtype.  Layout metadata maps `(module, attribute)` to a plane offset,
shape, and original tensor identity.  Bare adapter parameters that are not part
of an immutable base plane remain device-resident unless a policy explicitly
registers them.

This removes duplicated per-tensor submissions without silently casting mixed
dtypes or stranding `weight_s_rel`, scales, and other sidecars.

## Model-weight policies on top

### Frozen/inference H2D-only

- CPU master is immutable.
- `release(dirty=False)` repoints the model to the master and retains no D2H.
- ring size 2 is the default minimum for copy/compute overlap; 3 is useful only
  when measurement shows two blocks of look-ahead are consumed before copy
  completion.
- prefetch wraps across denoise-step boundaries instead of clearing slots.

This should be the first common path and can replace the current generation
H2D-only implementations plus FLUX.2 LoRA training.

### Full-parameter bidirectional

- CPU master is mutable.
- a block is marked dirty only after its optimizer update has completed;
- D2H waits on the update event, not merely the module backward event;
- the next H2D reads that master only after its D2H completion event;
- checkpoint calls `flush()` before serialization;
- gradient checkpointing and fused-update ordering are explicit gates, not
  assumptions.

This must be implemented separately after the frozen path.  Reusing an
inference schedule and adding a backward hook is not sufficient.

### Architecture adapters

Each architecture supplies only:

- the ordered heavy-block sequence (FLUX.2 concatenates dual then single);
- tensors/sidecars in a block bundle;
- resident auxiliary modules;
- actual forward and checkpoint-recompute boundaries;
- unsupported feature combinations.

The engine owns no model-specific forward code.  Existing native block loops or
small wrappers call the same `acquire/prefetch/release` protocol.  SD1.5/SDXL
remain explicitly unsupported until a U-Net block ordering policy is designed;
SenseNova keeps its phase-half evictor until it can express that policy without
weakening its per-run contract.

## Optimizer ring-buffer redesign

The present AdamW/Lion host-state implementation is correctness-oriented and
its CPU buffers are deliberately persistent, not a recyclable ring.  The CPU
allocator should remain as-is in the first migration: its non-aliasing lifetime
and checkpoint placement tests are valuable.

The GPU staging should change:

1. group optimizer state by model block and dtype;
2. allocate two or three fixed GPU state slots through the common engine;
3. on a block backward-pre boundary, prefetch that block's state while later
   block backward compute is running;
4. let per-parameter fused update kernels use views into the resident slot;
5. after the block's last exactly-once update, record an update event and submit
   one coalesced D2H per dtype plane;
6. use the existing update census to refuse a missing/duplicate finalization;
7. keep parameters outside the ordered block list on the existing per-parameter
   fallback until an order is proven.

This is a performance redesign, not a correctness repair.  The current CPU
optimizer suite passes; replacing it wholesale together with weight offload
would couple two failure domains and make checkpoint regressions difficult to
localize.

The expected benefit is removal of temporary `.cuda()` allocation per
parameter, fewer DMA submissions, and overlap of state H2D with backward
compute.  It does not remove the unavoidable state H2D+D2H bytes.  A result is
accepted only if CUDA events show reduced exposed wait and iteration median/p95
without increasing peak VRAM beyond the configured slots.

## Migration sequence

1. Add the transfer engine and CPU/fake-stream state-machine tests without
   changing production routing.
2. Migrate one frozen generation architecture and prove bit identity,
   sidecar identity, wrap-prefetch, bounded slots, and cleanup.
3. Migrate the remaining compatible generation paths and FLUX.2 frozen-base
   training through architecture adapters.
4. Replace `LayerOffloadConductor` for frozen-base LoRA paths; refuse rather
   than fall back when its required checkpoint/recompute contract is absent.
5. Add mutable full-FT release-after-update and checkpoint flush, then migrate
   one image DiT and one video DiT before the other architectures.
6. Add optimizer GPU-state slots while retaining the persistent CPU allocator
   and per-parameter fallback.
7. Delete the three legacy scheduling implementations only after route and
   checkpoint compatibility tests pass.

Each migration is one architecture family per commit.  Default routing stays
unchanged until its replacement has bounded CUDA evidence.

## Required evidence

For each shared policy family record:

- numerical identity against no offload (or declared tolerance where kernels
  are not deterministic);
- exact parameter and optimizer-state identity after save/resume;
- transferred H2D/D2H bytes versus the theoretical bundle size;
- number of DMA submissions per block;
- copy, compute, and exposed-wait intervals from CUDA events;
- overlap ratio and slot occupancy;
- iteration median/p95 and peak allocated/reserved VRAM;
- ring sizes 1/2/3 and pinned/pageable fallback behavior.

No `non_blocking=True` call, thread-pool submission, or CUDA event by itself is
evidence that useful overlap occurred.
