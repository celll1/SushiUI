# Dataset contract

This reference describes the dataset surface consumed by the current training
stack. It does not serve as a product roadmap for the dataset-management UI.

## Stored datasets

Dataset records and indexed items live in the dataset database. Training runs
refer to registered datasets by `dataset_id`; a legacy folder-path lookup exists
for backward compatibility but is not the preferred identity.

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

## Change checklist

When adding a dataset-level parameter, update all of the following:

1. `DatasetConfigItem` in `backend/api/routes.py`;
2. `DATASET_LEVEL_PARAMS` in `backend/core/training/dataset_params.py`;
3. the consuming trainer or loader;
4. `openapi.yaml` and the frontend type/UI when exposed.

See `backend/core/training/TRAINING_PARAMS_GUIDE.md` for detailed propagation
rules and `docs/guides/DYNAMIC_CROP_BUCKETING.md` for per-epoch crop behavior.
