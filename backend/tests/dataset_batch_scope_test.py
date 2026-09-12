import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import api.batch_operations as batch_module
from api.batch_operations import (
    BatchBackfillTagDataRequest,
    BatchReorderTagsRequest,
    BatchReplaceTagRequest,
    BatchSelection,
    batch_backfill_tag_data,
    batch_reorder_tags,
    batch_replace_tag,
)
import core.datasets.captions as caption_service
from core.datasets.batch_jobs import BatchJobRegistry, resolve_dataset_item_ids, resolve_dataset_selection
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
    db.add(DatasetCaption(
        item_id=item.id,
        caption_type="tags",
        content=tags,
        is_tags_format=True,
    ))
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


def test_batch_sidecar_failure_rolls_back_index_and_revision(tmp_path, monkeypatch):
    db = _session(tmp_path)
    dataset = _dataset(db, "first")
    item = _item(db, dataset, tmp_path, "one", "old")
    monkeypatch.setattr(batch_module.taglist_cache, "get_categories_batch", lambda tags: {})
    monkeypatch.setattr(
        caption_service,
        "write_indexed_caption",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = asyncio.run(batch_replace_tag(
        BatchReplaceTagRequest(item_ids=[item.id], from_tag="old", to_tag="new"),
        db,
        lambda *_: None,
        dataset_id=dataset.id,
        should_cancel=lambda: False,
    ))

    db.expire_all()
    assert result.failed_count == 1
    assert db.query(DatasetCaption).filter_by(item_id=item.id).one().content == "old"
    assert db.get(Dataset, dataset.id).revision == 0


def test_batch_reorder_updates_sidecar_tag_data_and_revision(tmp_path, monkeypatch):
    db = _session(tmp_path)
    dataset = _dataset(db, "first")
    item = _item(db, dataset, tmp_path, "one", "general, character")
    monkeypatch.setattr(batch_module.taglist_cache, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        batch_module.taglist_cache,
        "get_category",
        lambda tag: {"general": "General", "character": "Character"}[tag],
    )
    monkeypatch.setattr(
        batch_module.taglist_cache,
        "get_categories_batch",
        lambda tags: {tag: {"general": "General", "character": "Character"}[tag] for tag in tags},
    )

    result = asyncio.run(batch_reorder_tags(
        BatchReorderTagsRequest(
            item_ids=[item.id], category_order=["Character", "General"]
        ),
        db,
        lambda *_: None,
        dataset_id=dataset.id,
        should_cancel=lambda: False,
    ))

    db.expire_all()
    caption = db.query(DatasetCaption).filter_by(item_id=item.id).one()
    assert result.updated_count == 1
    assert caption.content == "character, general"
    assert caption.tag_data == (
        '[{"tag": "character", "category": "Character"}, '
        '{"tag": "general", "category": "General"}]'
    )
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "character, general"
    assert db.get(Dataset, dataset.id).revision == 1


def test_backfill_advances_revision_at_each_committed_batch(tmp_path, monkeypatch):
    db = _session(tmp_path)
    dataset = _dataset(db, "first")
    first = _item(db, dataset, tmp_path, "one", "a")
    second = _item(db, dataset, tmp_path, "two", "b")
    monkeypatch.setattr(batch_module.taglist_cache, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        batch_module.taglist_cache,
        "get_categories_batch",
        lambda tags: {tag: "General" for tag in tags},
    )

    result = asyncio.run(batch_backfill_tag_data(
        BatchBackfillTagDataRequest(dataset_id=dataset.id, batch_size=1),
        db,
        lambda *_: None,
    ))

    db.expire_all()
    assert result.updated_count == 2
    assert db.get(Dataset, dataset.id).revision == 2
    assert db.get(DatasetCaption, first.captions[0].id).tag_data is not None
    assert db.get(DatasetCaption, second.captions[0].id).tag_data is not None


def test_query_selection_applies_filters_and_exclusions_on_the_server(tmp_path):
    db = _session(tmp_path)
    dataset = _dataset(db, "first")
    kept = _item(db, dataset, tmp_path, "wanted_a", "cat, solo")
    excluded = _item(db, dataset, tmp_path, "wanted_b", "cat, solo")
    _item(db, dataset, tmp_path, "other", "cat, solo")
    _item(db, dataset, tmp_path, "wanted_c", "dog")
    selection = BatchSelection(
        mode="query",
        search="wanted",
        tags="cat",
        excluded_ids=[excluded.id],
    )

    assert resolve_dataset_selection(db, dataset.id, [], selection) == [kept.id]


def test_query_selection_rejects_foreign_exclusions(tmp_path):
    db = _session(tmp_path)
    first = _dataset(db, "first")
    second = _dataset(db, "second")
    foreign = _item(db, second, tmp_path, "foreign", "tag")
    selection = BatchSelection(
        mode="query", excluded_ids=[foreign.id]
    )

    with pytest.raises(ValueError, match=str(foreign.id)):
        resolve_dataset_selection(db, first.id, [], selection)


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
