# Block swap and ring-buffer revalidation plan (2026-09)

## Goal

Revalidate the current implementation of weight block swap and host-resident
optimizer-state ring buffers after the fixes that landed after the July audit.
For every training architecture, distinguish three separate questions:

1. are parameters, gradients, optimizer states, and checkpoints correct;
2. are H2D/D2H copies asynchronous and prefetched early enough to overlap
   useful compute;
3. is measured overhead close to the unavoidable transfer-time floor.

The similarly named `RingBufferAllocator` used by `LayerOffloadConductor` and
the AdamW/Lion host-state ring-buffer optimizers are independent mechanisms and
must be reported separately.

## Safety and scope

- Do not start or resume run 127 and do not restart either application server.
- Use static/CPU tests for every architecture. Use bounded GPU probes for
  representative implementation families instead of repeating long full-model
  runs where the same shared mechanism is used.
- Cap GPU probes before model allocation and run one foreground probe at a time.
- Never treat an absent checkpoint, unsupported training method, or unmeasured
  architecture as a pass.
- Preserve default-off behavior and numerical semantics unless a defect is
  demonstrated and fixed in its own commit.

## Phase 1: current implementation inventory

Build a source-backed matrix for all 13 entries in `ARCH_REGISTRY` covering:

- selected block list and offloader/conductor implementation;
- LoRA, ReLoRA, full-parameter, and ControlNet applicability;
- forward and backward load/evict hooks;
- standard bidirectional swap versus H2D-only eligibility;
- quantized weight sidecars and adapter parameters;
- optimizer-state placement and fused-update mode;
- checkpoint/resume synchronization boundary.

Reconcile the matrix with `docs/audits/BLOCK_SWAP.md`; mark stale statements
instead of carrying them forward as facts.

## Phase 2: correctness gates

Add or strengthen focused tests for these invariants:

- a block is resident before each forward and backward use;
- an updated full-parameter weight is copied back before eviction;
- frozen H2D-only masters are immutable and adapter parameters remain live;
- quantization scales/sidecars move with their weight;
- every trainable parameter updates exactly once or the run fails loudly;
- host-resident optimizer state is prefetched before its update, written back
  after it, and synchronized before save;
- mixed CPU/GPU state residency, optimizer reset, and resume preserve identity;
- block swap plus activation dispatch cannot double-offload activations.

Any common defect receives a fix commit before architecture claims are made.

## Phase 3: transfer and prefetch audit

For each implementation family, trace the CUDA stream/event dependency graph
and classify every synchronization as one of:

- required consumer dependency;
- step/checkpoint boundary;
- error fallback;
- avoidable global or host synchronization.

Measure representative CUDA paths with event timestamps and transfer counters:

- no swap, standard swap, and H2D-only where valid;
- ring sizes 1, 2, and 3 where configurable;
- pinned and pageable host paths;
- H2D/D2H bytes, copy duration, exposed host wait, compute duration, overlap
  ratio, peak allocated/reserved VRAM, and iteration median/p95;
- optimizer state on GPU versus bounded host-resident state.

An asynchronous API call alone is not evidence of overlap. Prefetch is only
effective when the copy begins before the consumer needs it and CUDA timing
shows copy/compute concurrency. Compare observed copy time with transferred
bytes and the previously measured pageable/pinned PCIe bandwidth range.

## Architecture grouping

| Group | Architectures | Primary implementation question |
|---|---|---|
| U-Net image | SD1.5, SDXL | whether block swap is genuinely supported or correctly refused |
| shared DiT image | Z-Image, Anima, Lens, Ideogram 4, MiniT2I, Krea 2 | block-list mapping, conductor/shared offloader correctness, quantized sidecars |
| FLUX.2 | FLUX.2 | dual/single boundary, backward prefetch, H2D-only training behavior |
| video | LTX-2.3, MiniMax-H3 | temporal block loop, backward order, large-block PCIe overlap |
| audio | ACE-Step 1.5 | 3-D workload path and transformer block ownership |
| dual-objective | SenseNova U1.5 | MoT half residency, flow/text phase changes, host-state interaction |

MiniMax Music 3 is generation-only and has no optimizer path; include only any
staged inference offload that shares the audited primitive.

## Acceptance

- Static architecture matrix has no unknown supported path.
- Relevant CPU/static tests pass for all shared paths.
- Representative GPU families show finite loss/gradients and no missing or
  duplicated updates.
- No correctness path depends on a silent `param.is_cuda` skip.
- No hot-path global `torch.cuda.synchronize()` remains unless measurement or a
  documented fallback justifies it.
- Prefetch claims include measured overlap; otherwise label the path synchronous
  or pull-based.
- Final report separates verified correctness, measured efficiency, unsupported
  combinations, missing assets, and future optimization work.

## Commit units

1. This plan.
2. Common block-swap correctness fixes and tests, one defect per commit.
3. Optimizer ring-buffer correctness fixes and tests, one defect per commit.
4. Architecture-family wiring or refusal fixes, one family per commit.
5. Bounded performance probes and raw results.
6. Final architecture matrix and residual backlog.
