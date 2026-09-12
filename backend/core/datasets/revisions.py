"""Dataset revision ownership shared by mutations and training snapshots."""

from __future__ import annotations

DATASET_SNAPSHOT_SCHEMA_VERSION = 2


def bump_dataset_revision(dataset) -> int:
    dataset.revision = int(dataset.revision or 0) + 1
    return dataset.revision
