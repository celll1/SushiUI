"""Dataset-scoped batch selection and cancellation state."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from collections.abc import Iterable


class BatchJobConflict(ValueError):
    pass


class BatchJobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, tuple[int, threading.Event]] = {}
        self._dataset_jobs: dict[int, set[str]] = defaultdict(set)

    def start(self, dataset_id: int, requested_id: str | None = None) -> str:
        operation_id = requested_id or str(uuid.uuid4())
        with self._lock:
            if operation_id in self._jobs:
                raise BatchJobConflict(f"Batch operation already active: {operation_id}")
            self._jobs[operation_id] = (dataset_id, threading.Event())
            self._dataset_jobs[dataset_id].add(operation_id)
        return operation_id

    def finish(self, operation_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(operation_id, None)
            if job is None:
                return
            dataset_id, _ = job
            active = self._dataset_jobs.get(dataset_id)
            if active is not None:
                active.discard(operation_id)
                if not active:
                    self._dataset_jobs.pop(dataset_id, None)

    def cancel(self, dataset_id: int, operation_id: str | None = None) -> int:
        with self._lock:
            if operation_id is not None:
                job = self._jobs.get(operation_id)
                if job is None or job[0] != dataset_id:
                    return 0
                job[1].set()
                return 1
            operation_ids = tuple(self._dataset_jobs.get(dataset_id, ()))
            for active_id in operation_ids:
                self._jobs[active_id][1].set()
            return len(operation_ids)

    def is_cancelled(self, operation_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(operation_id)
            return bool(job and job[1].is_set())


batch_jobs = BatchJobRegistry()


def resolve_dataset_item_ids(db, dataset_id: int, requested_ids: Iterable[int]) -> list[int]:
    """Resolve a stable, deduplicated selection and reject foreign item IDs."""
    from database.models import DatasetItem

    requested = list(dict.fromkeys(int(item_id) for item_id in requested_ids))
    if not requested:
        return [
            row[0]
            for row in (
                db.query(DatasetItem.id)
                .filter(DatasetItem.dataset_id == dataset_id)
                .order_by(DatasetItem.id)
                .all()
            )
        ]

    _validate_dataset_item_ids(db, dataset_id, requested)
    return requested


def _validate_dataset_item_ids(db, dataset_id: int, requested: list[int]) -> None:
    from database.models import DatasetItem

    found: set[int] = set()
    for start in range(0, len(requested), 900):
        chunk = requested[start:start + 900]
        found.update(
            row[0]
            for row in db.query(DatasetItem.id).filter(
                DatasetItem.dataset_id == dataset_id,
                DatasetItem.id.in_(chunk),
            )
        )
    missing = [item_id for item_id in requested if item_id not in found]
    if missing:
        sample = ", ".join(str(item_id) for item_id in missing[:10])
        raise ValueError(
            f"Item IDs do not belong to dataset {dataset_id}: {sample}"
            + (" ..." if len(missing) > 10 else "")
        )


def resolve_dataset_selection(db, dataset_id: int, requested_ids, selection) -> list[int]:
    if selection is None:
        return resolve_dataset_item_ids(db, dataset_id, requested_ids)

    from database.models import DatasetItem
    from .queries import exact_tag_item_ids

    excluded = list(dict.fromkeys(int(item_id) for item_id in selection.excluded_ids))
    if excluded:
        _validate_dataset_item_ids(db, dataset_id, excluded)
    requested_tags = (
        [tag.strip() for tag in selection.tags.split(",") if tag.strip()]
        if selection.tags else []
    )
    if requested_tags:
        item_ids = exact_tag_item_ids(
            db, dataset_id, requested_tags, search=selection.search
        )
    else:
        query = db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
        if selection.search:
            query = query.filter(DatasetItem.base_name.like(f"%{selection.search}%"))
        item_ids = [row[0] for row in query.order_by(DatasetItem.id).all()]
    excluded_set = set(excluded)
    return [item_id for item_id in item_ids if item_id not in excluded_set]
