# Unified offload transfer migration and validation (2026-09)

## Result

The immutable-weight mechanism is now shared. `FrozenSequentialTransferEngine`
owns fixed GPU slots, dtype-separated flat planes, CUDA stream/event ordering,
cross-step wrap prefetch, and transfer counters. `FrozenLruTransferEngine` uses
the same storage and copy mechanism for order-agnostic checkpoint
recomputation.

All forward-only `TransformerBlockOffloader` and `FluxBlockOffloader` instances
use the common engine, irrespective of the legacy `block_swap_h2d_only` flag.
This covers the generation block-swap routes for Z-Image, Anima, Lens,
Ideogram4, MiniT2I, FLUX.2, LTX-2.3, and MiniMax-H3. MiniMax-H3 now uses the
multi-plane path instead of forcing the bidirectional legacy swap: FP8 weights,
FP32 projections, and quantization sidecars retain their own dtype.

FLUX.2 frozen-base/LoRA training uses the common frozen LRU policy. Mutable
training across all supported DiT block-swap routes uses
`MutableLruTransferEngine`: persistent non-aliasing CPU masters, fixed
dtype-separated GPU slots, clean forward eviction, reverse recompute prefetch,
post-optimizer D2H, and a checkpoint flush boundary.

AdamW8bit_RingBuffer and Lion8bit_RingBuffer now pass host state directly to
their compiled extensions. The extensions already owned a dedicated transfer
stream and producer/consumer events; the Python layer had bypassed that path by
allocating a second CUDA copy and copying it back itself. Host moment buffers
are now zeroed after allocation, matching the GPU-resident initialization.

## CUDA evidence

Device: NVIDIA RTX 6000 Ada Generation. The probes are independent processes;
they do not call the running backend or touch a training run.

### Immutable weight engine

Command:

```text
venv/Scripts/python.exe backend/core/training/probes/offload_transfer_validation.py --dim 3072 --blocks 10 --batch 128 --repeats 3
```

Ten 36 MiB FP32 bundles were consumed by identical GEMMs.

| Mode | Median | p95/max of 3 | Peak allocated | Steady allocated |
|---|---:|---:|---:|---:|
| All weights resident | 1.219 ms | 117.520 ms | 0.3654 GiB | 0.3624 GiB |
| Ring 1 | 16.870 ms | 16.968 ms | 0.0490 GiB | 0.0460 GiB |
| Ring 2 | 14.740 ms | 14.806 ms | 0.0841 GiB | 0.0812 GiB |

- resident, ring 1, and ring 2 outputs were bit-for-bit equal;
- ring 2 improved median iteration time by 14.45% over the minimum-VRAM ring 1;
- ring 2 reduced peak allocated VRAM by 76.98% versus full residency;
- 30 acquisitions over three iterations had zero misses, proving that end-of-step
  wrap prefetch removed the former synchronous boundary repair;
- measured H2D was 1.125 GiB in 32 plane submissions: two initial primes plus
  one release/prefetch per block and iteration, exactly matching the schedule.

The resident p95 includes first-execution library warm-up and is retained rather
than silently discarded. Median is the relevant steady-state comparison.

### RB optimizer host state

Command:

```text
venv/Scripts/python.exe backend/core/training/probes/rb_optimizer_transfer_validation.py --rows 1024 --cols 4096 --steps 3
```

For both optimizers, GPU-resident and pinned-host runs produced bit-identical
parameters, quantized state bytes, and absmax metadata after three updates.
Host state preserves the allocator's parameter-shaped view while resident state
is flat; equality is checked over the flat byte sequence, and checkpoint/resume
shape compatibility remains covered by the existing suite.

| Optimizer | Host median | GPU-state median | Host steady VRAM reduction | Host peak reduction |
|---|---:|---:|---:|---:|
| AdamW8bit RB | 1.031 ms | 0.344 ms | 19.94% | 14.25% |
| Lion8bit RB | 0.696 ms | 0.190 ms | 11.09% | 7.68% |

The isolated one-parameter host-state path is deliberately slower because it
measures PCIe transfer with no later backward compute to hide it. The result is
not evidence of optimizer speedup. Its benefits are bounded persistent VRAM and
the ability to overlap writeback with subsequent parameter backward work.

The production fused-hook path was also measured with 24 BF16 linear layers
(100,663,296 parameters) in a separate process per arm:

```text
venv/Scripts/python.exe backend/core/training/probes/ringbuffer_overlap.py --arm {adamw_host,adamw_gpu,lion_host,lion_gpu} --tokens 256,4096
```

| Optimizer | State | 256-token median | 4096-token median | 4096 peak |
|---|---|---:|---:|---:|
| AdamW8bit RB | host | 23.574 ms | 37.122 ms | 0.6360 GiB |
| AdamW8bit RB | GPU | 6.327 ms | 38.089 ms | 0.8235 GiB |
| Lion8bit RB | host | 13.166 ms | 35.870 ms | 0.6345 GiB |
| Lion8bit RB | GPU | 3.373 ms | 34.569 ms | 0.7283 GiB |

At 4096 tokens, where later backward compute can cover state traffic, host
state was within 4% of GPU state (Adam 2.54% faster in this sample; Lion 3.77%
slower). Peak allocated fell by exactly the expected state sizes: 192 MiB for
Adam and 96 MiB for Lion. The 256-token arm confirms that insufficient compute
exposes PCIe cost. These are synthetic throughput results, not a claim that
host state universally accelerates a real architecture.

The design proposal considered replacing the extension staging with shared
Python-side fixed slots grouped by model block. The measured production path
already provides the required transfer stream/event overlap and exact bounded
peak reduction, while grouping would change optimizer ordering and require a
new C++/CUDA ABI. The audit therefore retains the native staging implementation:
sharing the model-weight Python allocator would add a second scheduler without
removing PCIe bytes. This is the documented implementation decision for design
sequence item 6, not an unimplemented migration step.

### Mutable training weights

Command, once per mode:

```text
venv/Scripts/python.exe backend/core/training/probes/mutable_offload_validation.py --mode {resident,ring1,ring2} --blocks 12 --swap 10 --dim 2048 --batch 32 --steps 3
```

All three modes produced parameter SHA-256
`b7c224fb003e0d02520002c23914c1f3327a4ad2a6555bbce01134da051436e7`
after three fused BF16 SGD updates.

| Mode | Median | Peak allocated | Steady allocated |
|---|---:|---:|---:|
| Resident | 8.364 ms | 0.2137 GiB | 0.2040 GiB |
| Ring 1 | 23.548 ms | 0.0731 GiB | 0.0634 GiB |
| Ring 2 | 18.690 ms | 0.0887 GiB | 0.0790 GiB |

Ring 2 reduced peak allocated memory by 58.48% versus residency and improved
median iteration time by 20.63% versus the minimum-memory ring 1. It issued 50
H2D bundles instead of ring 1's 55; both wrote 30 dirty bundles after updates.

The real MiniMax-H3 FP8-scaled checkpoint also passed a short joint video/audio
LoRA train step with 40 of 50 blocks swapped. Loss was exactly
`3.1203932762145996`, matching the recorded resident and legacy-swap runs, all
600 LoRA gradients were finite, peak allocated fell from the legacy path's
21.97 GiB to 8.13 GiB, and wall time was 6.20 s. This is a one-run architecture
gate, not a stable throughput benchmark.

## Test evidence

- shared engine state machine, multi-plane sidecars, wrap, variable sizes,
  cleanup, sequential and LRU policies: 9 tests;
- focused generic/FLUX/training-gate suite: 61 passed;
- MiniMax-H3 wrapper, W4A8 sidecar, and LoRA roundtrip suite: 35 passed;
- combined offload, activation-dispatch, RB optimizer, resume, video-threading,
  and training-gate suite: 290 passed plus 487 subtests;
- py_compile and a real import with CUDA initialization stubbed passed for every
  changed backend module.
- mutable engine state/writeback and real checkpoint recomputation: 3 tests;
- architecture wiring and checkpointing refusal for nine named routes plus
  FLUX.2 full-FT, including the MiniMax generation ring capability: 20 tests;
- final combined mutable/frozen transfer, activation dispatch, optimizer,
  checkpoint/resume, quantized-training, video threading, and architecture
  contract suite: 506 passed plus 449 subtests;
- the root-import layer strategy test passed independently.

A separate broad quantization-capability run passed 131 tests plus 222 subtests
and failed one unrelated pre-existing classification guard: the live
`/generate/img2txt` route is absent from that test's two route sets. This change
does not alter that endpoint or the classification table.

The complete backend suite was additionally sampled through 43% before its
42-GiB CPU test was stopped under the allowed long-test omission. Its first
independently reproduced failure is unrelated and pre-existing:
`adapter_execution_backend_cheap_test` rejects the unmodified
`core/adapters/composite.py` dispatch call. The focused suite above contains
all files touched by, or contract-adjacent to, this migration and is green.

## Binary preservation and rebuild decision

Before optimizer work, the compiled binaries were copied to the ignored
directory `local/offload_engine_backup_20260910_224033`.

| Binary | SHA-256 (working and backup) |
|---|---|
| adamw8bit_cuda_ext.pyd | `A2387C0BC611D6618029E61E5AA2756D9DC7BCE7CF7CA500297670D531068E18` |
| lion8bit_cuda.pyd | `7C8FCD84D7760A944F109FAEB5D07E9AC177D3E861E7C9CA2106A65C534D62E6` |

`git check-ignore` resolves the backup through `.gitignore: local/*`. Both JIT
loaders reported `ninja: no work to do`, and the post-test hashes still match
the backups. No C++/CUDA source changed, so no rebuild occurred or is required.

## Final audit findings

1. The immutable transfer engine is bit-exact, bounded, and removes the old
   end-of-denoise-step synchronous repair. No hot-path global CUDA synchronize
   remains in these shared sequential/LRU policies.
2. The common engine does not cast across dtypes. It coalesces each dtype plane
   and moves sidecars with the weight that owns them.
3. The RB Python hot path no longer duplicates the extension's staging. Its
   numerical parity is proven, but an isolated host-state update is slower than
   GPU-resident state; do not market it as a speed optimization.
4. `LayerOffloadConductor` is now an architecture adapter over the common
   mutable engine; the recycled CPU arena, duplicate `.to()` plus copy path,
   unreachable `forward_layer`, and dead activation allocator were deleted.
5. Z-Image initialization now occurs after adapter injection, as every other
   mutable route does. FLUX.2 full-FT is likewise deferred to the mutable engine;
   its frozen-base route remains on the immutable LRU policy.
6. Mutable swap fails closed without gradient checkpointing and at runtime if a
   block reaches backward without recomputation. Checkpoint save synchronizes
   CPU masters, interrupted/emergency saves resolve active slots, OOM recovery
   releases active slots, and Ideogram's second conductor participates in
   update observation, flush, and cleanup.
7. Whole-model RB optimizer overlap is effective when layer compute is large
   enough: the 4096-token synthetic path stayed within 4% of GPU-state time
   while removing the exact optimizer-state allocation. The short-compute arm
   remains the counterexample, so no universal speedup claim is made.
