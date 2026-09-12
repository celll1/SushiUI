# Tagger refactor plan (2026-09)

## Goal

Repair the tagger paths that can lose work or mislead the operator, then reduce
duplication and CPU/IO overhead without changing model outputs, checkpoint
contents, training schedules, or public API compatibility.

The scope covers the legacy WD tagger, the trainable SigLIP2 tagger, dataset
browsing and sidecar editing, tagger training configuration, and live training
monitoring. The two inference stacks remain separate because both have active
callers and different model contracts.

## Work units

1. Repair browser batch inference. Feed encoded image bytes to the inference
   manager, serialize tag names rather than result objects, acquire the shared
   GPU slot, keep blocking work off the API event loop, and make transport
   failures visible to the frontend.
2. Make bulk sidecar editing fail closed. Do not treat an unreadable sidecar as
   an empty one, bound concurrent reads, key category refreshes by tag identity,
   and avoid one React state replacement per successful file.
3. Consolidate output selection. Share one implementation between loaded and
   live-training models, materialize response dictionaries only for selected
   tags, preserve category top-1 and sort/tie behavior, and remove superseded
   internal branches while retaining the public compatibility field.
4. Make model loading and inference controls truthful. Present backend-detected
   model type, prevent stale metadata requests and stale auto-derived vocabulary
   paths, report calibration/status failures, distinguish basic and advanced
   controls, and make the live-training-model fallback explicit.
5. Remove frontend duplication. Centralize tag category metadata and metric
   merge/decimation, keep backend defaults authoritative, and remove the tracked
   backup source file.
6. Decompose trainer orchestration where a boundary can be characterized without
   changing numerical order: validation collection/evaluation, checkpoint bundle
   writing, and online-Danbooru state persistence. Remove unused helpers and
   duplicate optimizer-state detection.
7. Document the maintained tagger boundary and re-audit comments. Retain comments
   that explain numerical, compatibility, multiprocessing, or memory constraints;
   remove section labels and prose that only restates adjacent code.

## Compatibility boundaries

- Existing `/api/v1/tagger/*`, `/api/v1/tagger/siglip2/*`,
  `/api/v1/tagger/browser/*`, and `/api/v1/tagger-training/*` routes remain.
- `display_calibration` remains accepted during the compatibility period even
  though calibrated probabilities are already returned unconditionally.
- Tag ordering, threshold comparison (`>=`), per-tag fallback rules, category
  top-1 selection, and response field names remain unchanged.
- Full and LoRA checkpoint tensor names, metadata, optimizer state, vocabulary
  snapshots, scheduler step order, and emitted training events remain unchanged.
- Failed sidecar reads become explicit errors and can no longer be overwritten
  through bulk editing.

## Proof requirements

- Browser batch inference tests use a real temporary image and verify bytes enter
  the manager, tag names reach the sidecar, SSE reaches a terminal event, and the
  GPU coordination boundary is acquired.
- Shared post-processing is compared against characterization fixtures covering
  global/per-tag thresholds, unreliable tags, calibration, OOD adjustment,
  category top-1, missing vocabulary rows, and equal-probability ordering.
- Full/LoRA head growth produces identical keys, shapes, dtype, device, and tensor
  values on CPU; optimizer migration tests cover both FP32 and 8-bit state.
- Trainer extraction tests compare optimizer/scheduler counts, emitted event
  order, and the complete checkpoint artifact set in a temporary directory.
- Frontend changes are checked by source review and repository-owner type/build
  verification; this repository currently has no frontend test runner.
- Changed backend files pass `py_compile` and a CUDA-stubbed real import. Focused
  backend tests must not create a CUDA context. Real-model speed and VRAM checks
  remain user-feedback verification.

## Commit sequence

Each numbered work unit is committed independently after this plan. A unit is
split further when its correctness and UI changes have different rollback
boundaries.
