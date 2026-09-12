# Tagger and frontend decomposition plan (2026-09)

## Goal

Finish the remaining tagger hot-path cleanup before decomposing the largest
frontend and API modules. Preserve numerical results and public request/response
contracts unless a separately characterized defect requires a behavior change.

## Work units

1. Enforce the configured per-epoch collection cap for the Danbooru
   `train_count` path. Keep this correctness repair separate from structural
   sampler changes.
2. Replace the duplicated base-dataset and Danbooru label/mask builders with one
   pure implementation. Precompute normalized special-tag membership without
   changing label or mask values.
3. Stop `AsymmetricLossOptimized` from retaining forward intermediates on the
   module. Preserve operation order, reduction, forward values, and gradients.
4. Reuse invariant tag-metric indexing data and remove redundant dtype
   conversions at the trainer boundary. Preserve saved metric arrays and rolling
   F1 inputs.
5. Reduce ORM materialization in tagger dataset and LR-matrix scans, and make
   both paths use the same caption tag extraction rules where their contracts
   overlap. Preserve sample ordering and LR-matrix output ordering.
6. Extract shared queue/progress, persisted-control, and send-to-panel behavior
   from the four generation panels. Keep mode-specific request construction and
   image/mask editing in their owning panels.
7. Split the API and training-configuration monoliths along existing domain
   boundaries. Preserve every route path, schema/default source, exported
   frontend API name, and architecture-specific visibility rule.

## Proof requirements

- The `train_count` test must demonstrate that the configured cap is reached and
  the query is exhausted within the current epoch.
- Base and online label builders must be characterized across rating presence,
  both quality modes, aliases, absent vocabulary tags, and vocabulary growth.
- Loss tests must compare forward tensors and parameter gradients exactly on
  CPU, including masked and unreduced cases.
- Metric tests must compare every accumulator array and serialized field before
  and after the implementation change, including vocabulary growth.
- Dataset/LR scans must be tested with temporary database rows containing JSON
  tag data, content fallback, missing images, multiple captions, and caption-type
  filtering.
- Frontend work is verified by source review; build and type-check remain the
  repository owner's responsibility.
- Backend edits receive `py_compile`, a CUDA-stubbed real import, and focused
  CPU tests. No server process is started or restarted.

## Commit sequence

This plan is committed first. Each work unit receives its own commit; a unit may
be split when behavior repair and mechanical extraction need independent
rollback. Units 6 and 7 start only after units 1 through 5 pass their completion
audit.
