# Dataset workspace refactor plan

Status: implemented and regression-tested in September 2026. The compatibility
Viewer remains available for multi-field metadata, while Tag Workspace is the
shared buffered tag-editing surface.

## Outcome

Dataset files and their sidecars remain the durable source of truth. The
dataset database becomes a searchable index, and training snapshots become
revision-keyed derived data. The Dataset page becomes a fast registry and
training-readiness view; detailed editing uses one shared workspace instead of
the current Dataset Viewer and Tagger Browser implementing parallel workflows.

Existing dataset records, API clients, and training configurations remain
readable throughout the migration. Compatibility endpoints may delegate to the
new services, but they must not retain independent persistence semantics.

## Current defects to close first

1. Single-item and batch edits choose different TXT/JSON fields and write
   sidecars non-atomically.
2. Database caption updates, aggregate statistics, and sidecar writes can
   succeed independently.
3. Batch item IDs are not consistently constrained to the dataset in the URL,
   and cancellation state is process-global rather than job-scoped.
4. Tag filtering uses substring matching while presenting exact-tag semantics.
5. Caption identity is ambiguous when an item has multiple rows with the same
   caption type.
6. Preview names based only on a media basename collide across directories and
   datasets.
7. External sidecar edits and caption deletion do not reliably invalidate every
   training snapshot.

## Target ownership

### Sidecar service

One backend service owns sidecar discovery, parsing, field selection, and
atomic replacement. A write either reaches the intended sidecar and then
updates the index, or reports failure without claiming the edit was persisted.
The service preserves unrelated JSON fields and uses an explicit field policy
derived from the indexed caption rather than inventing different rules for
single and batch calls.

### Dataset repository

One query layer owns dataset-scoped item lookup, caption identity, selection,
statistics invalidation, and revision changes. Routes must not reproduce these
queries. A selected item is always addressed by both dataset and item identity
at mutation boundaries.

### Media preview service

Previews are addressed by dataset/item identity, never a client-supplied
absolute path. The service validates membership, creates bounded-size cached
images lazily, includes source modification state in the cache key, and uses a
collision-resistant name. Video and audio preview work uses bounded concurrency.

### Dataset workspace

The existing virtualized Tagger Browser becomes the shared inspection/editing
surface after its process-global root is replaced by an explicit workspace
identity. Edits remain local and dirty until an explicit save. Navigation has a
save/discard guard, requests are cancellable, and multi-item editing exposes
common and partial tags.

### Training snapshot

Every successful indexed mutation increments a dataset revision. Training
snapshot keys include that revision and the caption-selection contract. A
reconcile operation imports external sidecar changes and increments the same
revision. Preview and training call the same caption-selection and processing
functions.

## Delivery order

### Phase 0: correctness and persistence

- Introduce the canonical atomic sidecar service and route all Dataset writes
  through it.
- Make caption updates dataset-scoped and give caption rows an unambiguous
  logical key without discarding legacy rows.
- Scope batch queries to their dataset and replace global cancellation with
  operation IDs.
- Repair exact-tag filtering, aggregate calculation, response contracts, and
  preview collisions.
- Add regression tests covering failure ordering, cross-dataset IDs, duplicate
  caption types, and TXT/JSON compatibility.

### Phase 1: responsive inspection

- Add a dataset/item preview endpoint with size variants, validators, and a
  bounded cache.
- Return grid projections rather than complete item metadata.
- Add request cancellation, stale-response guards, debounced filters, cursor
  pagination, and virtualized rendering.
- Replace all-ID selection with a server-side selection descriptor consisting
  of the active query and excluded IDs.
- Load tag statistics and detailed metadata only on demand.

### Phase 2: one workspace

- Move the reusable thumbnail grid, media view, selection model, and sidecar
  editor behind shared components and services.
- Replace the Tagger Browser global root with a workspace token scoped to the
  registered dataset or explicit folder session.
- Add dirty-buffer editing, save/discard navigation, real keyboard shortcuts,
  and common/partial-tag bulk editing.
- Retire the unused legacy item browser and reduce the old Dataset Viewer to a
  compatibility entry point.

### Phase 3: registry and training readiness

- Make the Dataset landing page show cached counts immediately and obtain
  health details separately.
- Report missing media, stale/invalid sidecars, duplicate stems, caption
  coverage, reference-pair failures, and media metadata gaps.
- Add explicit open-folder/copy-path/editor-launch integration through local
  settings; tracked configuration never contains machine-local commands.
- Offer an incremental reconcile action and a documented change-journal hook
  for external editors.

### Phase 4: snapshot and scanner consolidation

- Extract scanning, probing, grouping, caption ingestion, and stale-row cleanup
  from the API route into testable services.
- Replace timestamp-derived pickle identity with a dataset revision and schema
  version. Caption deletion must invalidate a snapshot.
- Stream projected database rows during snapshot construction and avoid holding
  ORM objects plus equivalent dictionaries simultaneously.
- Remove the unused N+1 dataset loader and duplicate caption-processing call
  sites after compatibility tests prove the shared path.

## Compatibility rules

- Existing database files migrate in place and are backed up by the established
  database migration mechanism.
- Existing `/api/v1/datasets` response fields remain available during the
  transition. New lightweight projections use new endpoints or opt-in fields.
- Existing TXT files remain TXT files. Existing JSON files retain unrelated
  fields and their detected caption field. Ambiguous files are refused with an
  actionable error rather than silently rewritten into a new shape.
- Existing training YAML continues to resolve `dataset_id`; folder-path lookup
  remains a legacy fallback.
- Cache format changes use a new versioned namespace and do not reinterpret old
  files as current snapshots.

## Verification gates

Each implementation commit must include the narrowest regression test that
would have caught the repaired real failure. Backend changes require compile
and import checks with CUDA initialization stubbed. The final audit must cover:

- sidecar write failure and recovery;
- DB/index agreement after single and batch edits;
- cross-dataset mutation refusal;
- old TXT and known JSON field layouts;
- exact tag selection and selection descriptors;
- stale-response suppression under rapid navigation;
- preview cache invalidation and basename collisions;
- external-edit reconcile and training snapshot invalidation;
- old database/API compatibility.

Frontend build and type checking remain owner-run per repository policy.
