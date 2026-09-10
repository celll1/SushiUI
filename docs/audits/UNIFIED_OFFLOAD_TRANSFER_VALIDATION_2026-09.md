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

FLUX.2 frozen-base/LoRA training uses the common LRU policy. Trainable adapters
remain resident; immutable base weights have no D2H writeback. FLUX.2 full-FT
still falls back to its mutable legacy policy.

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
the ability to overlap writeback with subsequent parameter backward work. A
whole-model iteration benchmark is still required before claiming an iteration
speed improvement for RB optimizer state.

## Test evidence

- shared engine state machine, multi-plane sidecars, wrap, variable sizes,
  cleanup, sequential and LRU policies: 9 tests;
- focused generic/FLUX/training-gate suite: 61 passed;
- MiniMax-H3 wrapper, W4A8 sidecar, and LoRA roundtrip suite: 35 passed;
- combined offload, activation-dispatch, RB optimizer, resume, video-threading,
  and training-gate suite: 290 passed plus 487 subtests;
- py_compile and a real import with CUDA initialization stubbed passed for every
  changed backend module.

A separate broad quantization-capability run passed 131 tests plus 222 subtests
and failed one unrelated pre-existing classification guard: the live
`/generate/img2txt` route is absent from that test's two route sets. This change
does not alter that endpoint or the classification table.

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

## Audit findings and remaining boundary

1. The immutable transfer engine is bit-exact, bounded, and removes the old
   end-of-denoise-step synchronous repair. No hot-path global CUDA synchronize
   remains in these shared sequential/LRU policies.
2. The common engine does not cast across dtypes. It coalesces each dtype plane
   and moves sidecars with the weight that owns them.
3. The RB Python hot path no longer duplicates the extension's staging. Its
   numerical parity is proven, but an isolated host-state update is slower than
   GPU-resident state; do not market it as a speed optimization.
4. `LayerOffloadConductor` remains a separate mutable full-parameter training
   implementation for Z-Image, Anima, Lens, Ideogram4, MiniT2I, Krea2,
   LTX-2.3, MiniMax-H3, and ACE-Step. The earlier audit's defects still apply:
   its hook path does not reach forward prefetch or forward eviction, performs
   duplicate moves, and its fixed CPU arena can alias live masters. It cannot be
   mechanically routed through an immutable engine because updated weights need
   D2H only after the fused optimizer event, while non-reentrant checkpoint
   recomputation occurs inside architecture-specific closures that module hooks
   do not see.
5. Consequently, the migration is complete for immutable generation weights,
   FLUX.2 frozen-base training, and native RB state transfer, but **not** for the
   nine mutable `LayerOffloadConductor` training routes or FLUX.2 full-FT. Those
   paths require a mutable policy plus explicit checkpoint-recompute and
   post-update callbacks per architecture. Claiming full all-architecture
   training completion from the present evidence would be incorrect.

The next safe unit is one image DiT and one video DiT with a mutable engine,
checkpoint flush/save-resume parity, and an optimizer-completion event. Only
after those two representatives pass should the remaining architecture adapters
be migrated and `LayerOffloadConductor` deleted.
