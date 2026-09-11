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
