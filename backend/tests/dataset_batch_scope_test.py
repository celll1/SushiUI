import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.batch_operations import BatchReplaceTagRequest, batch_replace_tag
from core.datasets.batch_jobs import BatchJobRegistry, resolve_dataset_item_ids
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _dataset(db, name):
    dataset = Dataset(name=name, path=name)
    db.add(dataset)
    db.flush()
    return dataset


def _item(db, dataset, tmp_path, name, tags):
    image = tmp_path / f"{name}.webp"
    image.write_bytes(b"image-placeholder")
    item = DatasetItem(dataset_id=dataset.id, base_name=name, image_path=str(image))
    db.add(item)
    db.flush()
    db.add(DatasetCaption(item_id=item.id, caption_type="tags", content=tags))
    db.commit()
    return item


def test_selection_rejects_foreign_ids_and_preserves_requested_order(tmp_path):
    db = _session(tmp_path)
    first = _dataset(db, "first")
    second = _dataset(db, "second")
    a = _item(db, first, tmp_path, "a", "old")
    b = _item(db, first, tmp_path, "b", "old")
    foreign = _item(db, second, tmp_path, "foreign", "old")

    assert resolve_dataset_item_ids(db, first.id, [b.id, a.id, b.id]) == [b.id, a.id]
    with pytest.raises(ValueError, match=str(foreign.id)):
        resolve_dataset_item_ids(db, first.id, [a.id, foreign.id])


def test_batch_query_cannot_mutate_an_item_in_another_dataset(tmp_path):
    db = _session(tmp_path)
    first = _dataset(db, "first")
    second = _dataset(db, "second")
    foreign = _item(db, second, tmp_path, "foreign", "old")
    request = BatchReplaceTagRequest(
        item_ids=[foreign.id], from_tag="old", to_tag="new"
    )

    result = asyncio.run(
        batch_replace_tag(
            request,
            db,
            lambda *_: None,
            dataset_id=first.id,
            should_cancel=lambda: False,
        )
    )

    db.expire_all()
    caption = db.query(DatasetCaption).filter_by(item_id=foreign.id).one()
    assert result.updated_count == 0
    assert result.skipped_count == 1
    assert caption.content == "old"


def test_cancellation_is_scoped_by_operation_and_dataset():
    registry = BatchJobRegistry()
    first = registry.start(1, "first")
    second = registry.start(1, "second")
    other = registry.start(2, "other")

    assert registry.cancel(1, first) == 1
    assert registry.is_cancelled(first)
    assert not registry.is_cancelled(second)
    assert not registry.is_cancelled(other)
    assert registry.cancel(2, first) == 0

    assert registry.cancel(1) == 2
    assert registry.is_cancelled(second)
    assert not registry.is_cancelled(other)
