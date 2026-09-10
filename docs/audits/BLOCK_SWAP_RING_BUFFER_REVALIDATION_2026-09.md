# Block swap / ring-buffer revalidation (2026-09)

## Scope and verdict

This is a static revalidation of the training paths for all 13 entries in
`ARCH_REGISTRY`.  It distinguishes three mechanisms which older documents call
"ring buffer" interchangeably:

1. `LayerOffloadConductor` and its CPU parameter arena;
2. `FluxBlockOffloader` and its fixed GPU H2D slots;
3. the persistent pinned host state used by the AdamW/Lion 8-bit optimizers.

The current implementation is **not ready for a blanket correctness or low
overhead claim**.  FLUX.2's frozen-base H2D-only path is fail-closed and has the
strongest correctness contract, but its training misses use a global CUDA
synchronization.  The nine `LayerOffloadConductor` architectures share common
correctness and scheduling defects.  SD1.5/SDXL silently accept a setting they
do not implement.  SenseNova correctly refuses it.

No running training job or application server was touched.  CUDA performance
numbers are deliberately not claimed by this static pass.

## Architecture matrix

| Architecture | Block list / implementation | Setup point | Static verdict |
|---|---|---|---|
| SD1.5 | none; handler is a no-op | never wired | **unsupported but not refused**; a positive value still selects central fused-optimizer behavior |
| SDXL | none; handler is a no-op | never wired | **unsupported but not refused**, same as SD1.5 |
| Z-Image | `transformer_original.layers` / `LayerOffloadConductor` | inside component load, before adapter injection | **unsafe**; has every common conductor defect and snapshots before LoRA changes the module tree |
| Anima | `transformer.blocks` / `LayerOffloadConductor` | after adapter setup | **unsafe common implementation**; block-list selection and setup order are otherwise correct |
| Lens | `transformer.transformer_blocks` / `LayerOffloadConductor` | after adapter setup | **unsafe common implementation** |
| Ideogram 4 | conditional and optionally trained unconditional `.layers` / two conductors | after adapter setup | **unsafe common implementation**; the two independent 8 GiB arenas can also double pinned-host pressure |
| MiniT2I | `transformer.model.net.double_blocks` / `LayerOffloadConductor` | after adapter setup | **unsafe common implementation**; only the double-block stack is managed |
| Krea 2 | `transformer.transformer_blocks` / `LayerOffloadConductor` | after adapter setup | **unsafe common implementation** |
| FLUX.2 | unified dual + single block lists / `FluxBlockOffloader` | component load; masters built lazily after adapter injection | **correctness-oriented, bounded support**: H2D-only, frozen denoiser, gradient checkpointing required; training prefetch is synchronous pull/LRU |
| LTX-2.3 | `transformer.transformer_blocks` / `LayerOffloadConductor` | after adapter/wrapper setup | **unsafe common implementation** |
| MiniMax-H3 | 50 `transformer.transformer_blocks` / `LayerOffloadConductor` | after adapter setup | **unsafe common implementation**; large swap counts are especially exposed to arena wrap |
| ACE-Step 1.5 | `transformer.decoder.layers` / `LayerOffloadConductor` | after adapter setup | **unsafe common implementation** |
| SenseNova U1.5 | none | refused during component load | **correctly unsupported**; non-zero `blocks_to_swap` raises before training |

The matrix describes training.  Inference also uses
`TransformerBlockOffloader`/`FluxBlockOffloader`; its H2D-only fixed-slot path is
separate from the broken conductor path and must not be used as evidence that
training prefetch works.

## Generation matrix

| Architecture | Generation block-swap path | Static verdict |
|---|---|---|
| SD1.5, SDXL | no block-loop driver | unsupported |
| Z-Image | native layer loop + `TransformerBlockOffloader` | wired for txt2img/img2img/inpaint; standard and optional H2D-only paths |
| Anima | native `.blocks` loop + explicit block list | wired; standard and optional H2D-only paths |
| Lens | native `transformer_blocks` loop + explicit block list | wired; standard and optional H2D-only paths |
| Ideogram 4 | native `.layers` loop | wired; standard and optional H2D-only paths |
| MiniT2I | MM-JiT `double_blocks` loop + explicit block list | wired; preamble/single blocks remain resident by design |
| Krea 2 | no `_block_offloader` consumer in the model loop | unsupported |
| FLUX.2 | `Flux2BlockSwapWrapper`, unified dual/single indices | wired; standard and optional H2D-only paths |
| LTX-2.3 | `Ltx2BlockLoopWrapper` | wired and forces H2D-only frozen-weight mode |
| MiniMax-H3 | `MiniMaxH3BlockLoopWrapper` | wired; forces standard sidecar-aware swap with pageable staging |
| ACE-Step 1.5 | no generation block-loop driver | unsupported |
| SenseNova U1.5 | no block-swap driver | unsupported; other component/phase eviction is a different mechanism |
| MiniMax Music 3 | no training or generation block-swap driver | unsupported |

The generic registry clamps a generation request to `[0, num_blocks - 1]` and
supports explicit non-`.layers` block lists.  Model loops listed as wired call
`wait_for_block(i)` before and `submit_move_blocks_forward(i)` after the actual
block, so their boundary placement is statically correct.

### Generation transfer efficiency

The three forward-only modes have different hot paths:

- **standard + pageable model weights** (the default when
  `use_pinned_memory=False`) uses two pinned staging sets but performs host-side
  `Event.synchronize()` calls inside the per-tensor swap loop.  It overlaps
  portions of D2H/H2D but repeatedly stalls Python and does not coalesce a
  block's tensors;
- **standard + fully pinned weights** removes those per-tensor host waits after
  one-time buffer creation, but retains D2H traffic even though inference
  weights are immutable;
- **H2D-only, ring size >= 2** keeps permanent CPU masters, coalesces a block
  into one flat copy, and uses compute/transfer events to prefetch `ring_size`
  blocks ahead.  This is the best current steady-state schedule when the block
  has one weight dtype and no quantization sidecars.

The H2D-only implementation has a step-boundary hole: `_h2d_submit()` stops
prefetching when `next_i` reaches the end of the swappable list and clears the
slot.  On the next denoise step, `_h2d_wait()` self-heals that empty/mismatched
slot with an immediate copy followed by `torch.cuda.synchronize()`.  The FLUX
variant has the same behavior.  Standard forward-only swap already wraps its
last submission to the first swappable block, so this regression is specific
to the nominal fast path.  Ring wrap-prefetch should be implemented and measured
before describing multi-step H2D-only overhead as minimal.

MiniMax-H3 deliberately uses standard, sidecar-aware swapping and
`use_pinned_memory=False`; its large per-block compute can hide much of the
cost, as previous generation probes observed, but the implementation still
pays per-tensor host waits and both transfer directions.  That observation is
not proof that the transfer schedule itself is optimal.

## Common `LayerOffloadConductor` defects

### C1. Registered hooks never reach the prefetch scheduler

`register_hooks()` installs a forward pre-hook which loads and immediately
waits for only the current block.  The only call to
`LayerOffloadStrategy.should_prefetch()` is in `forward_layer()`, but none of
the nine architecture paths calls that method: they invoke their native block
loop and rely exclusively on registered hooks.  Consequently
`enable_prefetch=True` is currently descriptive configuration, not behavior.

### C2. Forward never evicts a loaded block

The registered forward hooks have no post-hook.  `forward_layer()` also leaves
its intended eviction branch as `pass`.  Every initially offloaded layer is
therefore moved to GPU as forward progresses and stays there until its backward
hook runs.  At the forward/backward boundary all swapped layers are GPU
resident, so the implementation does not enforce its promised peak weight
residency.

### C3. D2H eviction is not ordered after compute

`offload_layer_to_cpu()` submits D2H copies on `transfer_stream` without first
recording a compute-stream event and making the transfer stream wait for it.
It then makes the transfer stream wait on its own event, which is not a host or
compute-stream completion barrier, deletes the event, and calls
`layer.to('cpu')`.  This does not establish the required producer-to-copy
ordering and can race the block's backward kernels.

The newer generic and FLUX offloaders do record `compute_done` and call
`transfer_stream.wait_event(compute_done)`; that later fix was never applied to
the conductor.

### C4. Each load performs redundant transfers and allocations

`load_layer_to_gpu()` first calls `layer.to(device)`, which already transfers
every parameter, then copies every parameter from its CPU arena a second time.
The first move is outside the dedicated transfer-stream context.  Thus even the
nominal async path pays an untracked full block transfer plus allocator churn
before recording its event.

### C5. The CPU arena wraps over live masters

`RingBufferAllocator` wraps to buffer zero when its fixed capacity is exceeded.
No layer allocation is freed during initialization or during the run, and
`LayerOffloadConductor` needs every offloaded CPU master to remain valid.
Therefore wrap means two live parameters can alias the same bytes.  The default
8 GiB target is not a capacity proof for large models or large
`blocks_to_swap`; no overflow check exists.

The allocator also estimates its largest parameter with
`layer.parameters(recurse=False)` but allocates from recursive
`layer.named_parameters()`.  Blocks whose parameters live in child modules can
therefore bypass the sizing estimate.

### C6. Gradient-checkpointing dependency is implicit

Moving a parameter's storage between forward and backward is only safe when
the backward does not depend on the original forward's saved weight storage,
or when a proven recompute/residency protocol restores it.  The conductor does
not validate gradient checkpointing, does not install a backward pre-load hook,
and does not distinguish original forward from checkpoint recomputation.

## FLUX.2 H2D-only training path

The FLUX.2 policy gate correctly requires all of the following:

- `block_swap_h2d_only=True`;
- a frozen denoiser (LoRA or text-encoder-only training);
- transformer gradient checkpointing, force-enabled and verified;
- lazy master construction after adapters/processors are installed;
- frozen base weights in the CPU master while trainable adapter weights remain
  resident;
- a unified index across dual and single block lists.

This avoids the broken standard bidirectional training schedule.  However,
training uses `_h2d_ensure_resident()` as an order-agnostic LRU pull.  Every miss
queues H2D and immediately calls `torch.cuda.synchronize()`.  There is no async
next-block or reverse-order checkpoint-recompute prefetch.  It is a valid
correctness-first fallback, not a minimum-overhead implementation.

## Optimizer host-state ring buffers

The optimizer allocator is correctly **not** the recyclable layer arena.  It
allocates persistent, non-overlapping, pinned CPU buffers per parameter and
keeps `absmax*` scale tensors on GPU.  Current load/resume code copies into the
owned buffers, validates shape/dtype, performs a residency census, and fails
loudly instead of silently skipping CPU parameters.  These are materially
stronger correctness properties than the July audit described.

The hot path is nevertheless not a prefetched ring:

- each per-parameter fused-backward hook creates temporary GPU state with
  `.cuda(non_blocking=True)`;
- the update kernel consumes it on the same current stream;
- updated state is copied back on that stream;
- there is no look-ahead using the known reverse parameter/block order and no
  fixed reusable GPU state slots.

Stream order makes the update/write-back sequence coherent, and checkpoint
boundaries synchronize before serialization.  It does not hide the H2D latency
behind the preceding parameter's backward compute, and temporary allocations
remain in the hot path.  The name denotes host residency, not measured transfer
overlap.

## Reconciliation with the older audit

`docs/audits/BLOCK_SWAP.md` remains useful history but is not current status.
Later commits added event-based ordering to the generic/FLUX paired swap,
quantization-sidecar handling, lazy H2D master construction, and fail-loud
optimizer-state checks.  Conversely, its favorable description of the
`LayerOffloadConductor` prefetch path was based on intended methods rather than
the hook path actually called by models.  This report supersedes those claims.

## Verification status and next work

Static source tracing is complete for all 13 training architectures.  The
repository virtual environment currently cannot launch because its recorded
base interpreter (`C:\Users\<redacted-local-user>\AppData\Local\Programs\Python\Python311\python.exe`)
is absent, so the focused CPU suite could not be rerun in this checkout.  This
is an environment blocker, not a test pass or a product-code failure.

Do not optimize around the present conductor.  The safe sequence is:

1. refuse SD1.5/SDXL non-zero settings and fail closed on unsupported methods;
2. replace the conductor arena with persistent CPU masters plus fixed GPU
   slots, with explicit capacity and lifetime invariants;
3. drive forward and backward/checkpoint-recompute residency from the actual
   block loop or paired pre/post hooks;
4. use compute/transfer events for producer and consumer dependencies, with no
   hot-path host/global synchronization;
5. add frozen-base H2D-only prefetch first, then separately implement and prove
   updated-weight D2H persistence for full-parameter training;
6. add fixed GPU optimizer-state slots and prefetch the next reverse-order
   parameter while the current backward/update runs;
7. only then run bounded CUDA overlap probes and publish overhead figures.

Required CUDA evidence is listed in the companion plan.  At minimum it must
show transfer bytes, copy intervals, exposed consumer wait, overlap ratio,
iteration median/p95, peak VRAM, finite gradients, exactly-once updates, and
checkpoint/resume identity for one representative image DiT, FLUX.2, one video
DiT, and one optimizer-state path.
