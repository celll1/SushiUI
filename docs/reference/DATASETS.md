# Dataset contract

This reference describes the dataset surface consumed by the current training
stack. It does not serve as a product roadmap for the dataset-management UI.

## Stored datasets

Dataset records and indexed items live in the dataset database. Training runs
refer to registered datasets by `dataset_id`; a legacy folder-path lookup exists
for backward compatibility but is not the preferred identity.

Files and sidecars are durable; `datasets.db` is their searchable index. Each
successful indexed mutation advances `Dataset.revision`, which identifies the
derived training snapshot without recounting or timestamp-scanning every row.

Each training dataset entry may carry:

- `dataset_id`;
- an ordered `caption_types` list;
- `filters`;
- `ve_reconstruction_mode`.

`backend/core/training/dataset_params.py` owns dataset-level defaults and
serialization. The Pydantic request models in `backend/api/routes.py` and the
schemas in `openapi.yaml` own the versioned API contract.

## Indexed items

The training loader consumes an indexed primary image, its selected caption,
and optional `reference_images`. Caption selection follows the dataset's
configured caption types and is included in the dataset cache key. Missing or
changed indexed files are handled by the dataset-drift checks rather than being
silently treated as the original dataset.

Reference-image meaning is architecture and method specific:

- SenseNova and FLUX.2 can consume per-item references on their supported
  training paths.
- SD1.5/SDXL reference-image training requires a configured vision encoder.
- ControlNet uses the condition/reference image required by its trainer.
- `ve_reconstruction_mode` may use the item's own image as its reference.

Architecture preflight remains authoritative and may refuse a combination even
when the dataset record contains references.

## Images, captions, and paths

The dataset service indexes supported image files and associated text captions
from user-selected directories. Paths stored in the local database are expected
to be machine-specific; paths committed to documentation or fixtures must use
synthetic examples such as `<DATASET_ROOT>/subject/image001.png`.

Do not commit dataset contents, captions containing private material, database
files, or machine-local absolute paths. Raw inventories and dataset-specific
analysis belong under `local/`.

## External editing and reconciliation

The Dataset page is primarily a registry and readiness surface. Configure an
editor command under Settings to launch a local dataset editor, or use Open
Folder / Copy Path. Commands and arguments remain in the local settings
database and are never tracked configuration. Launching uses an argument vector
without a shell; put `{dataset}` on its own argument line where the dataset path
belongs, or it is appended as the last argument.

After changing sidecars externally, use **Reconcile** in the Dataset editor or
call `POST /api/v1/datasets/{dataset_id}/scan?incremental=true`. This imports
changed captions, removes stale index rows, advances the dataset revision, and
invalidates old training snapshots. External tools should call this endpoint
after a completed batch rather than writing `datasets.db` directly. The normal
Scan action remains available for a full statistics rebuild.

The Health action is intentionally on demand. It reports missing media and
sidecars, invalid or stale sidecars, duplicate stems, reference failures,
caption coverage, and missing image metadata without making the registry list
pay that cost on every open.

## Inspection and editing surfaces

The Dataset editor has two intentionally different views:

- **Viewer** is the compatibility and metadata surface. It uses lightweight
  cursor pages and loads item details and tag statistics only when requested.
- **Tag Workspace** reuses the virtualized Tagger Browser grid and buffered
  tag editor. Its opaque workspace token is bound to the registered dataset;
  listing, image reads, and writes reject files that are present under the
  folder but absent from the index.

Tag Workspace saves use the same caption transaction as Dataset batch edits.
TXT datasets remain TXT, JSON fields retain unrelated data, and a successful
save updates the index, aggregate tag state, and dataset revision together.
The standalone Tagger Browser keeps its folder-only TXT workflow for backward
compatibility and is not treated as a registered dataset until opened through
the Dataset editor.

## Scanner ownership

`backend/core/datasets/scanning.py` owns index reconciliation for both the HTTP
Scan/Reconcile actions and training pre-flight rescans. The API layer maps
service errors to HTTP responses and supplies UI progress; training supplies
its cancellation callback directly without importing an API route.

The scanner groups equal basenames by dataset-relative directory, probes new
video/audio records without decoding complete media, removes vanished items,
and removes file-sourced captions that disappear during a sidecar refresh.
Caption changes rebuild tag statistics; a no-change incremental reconcile
retains the cached aggregate. Every material result advances
`Dataset.revision`, invalidating the derived training snapshot.

## Change checklist

When adding a dataset-level parameter, update all of the following:

1. `DatasetConfigItem` in `backend/api/routes.py`;
2. `DATASET_LEVEL_PARAMS` in `backend/core/training/dataset_params.py`;
3. the consuming trainer or loader;
4. `openapi.yaml` and the frontend type/UI when exposed.

See `backend/core/training/TRAINING_PARAMS_GUIDE.md` for detailed propagation
rules and `docs/guides/DYNAMIC_CROP_BUCKETING.md` for per-epoch crop behavior.
