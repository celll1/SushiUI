import asyncio

import torch
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from core.datasets.queries import (
    dataset_item_cursor_page,
    dataset_item_page,
    exact_tag_item_ids,
)
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _item(db, dataset_id, name, content):
    item = DatasetItem(dataset_id=dataset_id, base_name=name, image_path=name)
    db.add(item)
    db.flush()
    db.add(DatasetCaption(item_id=item.id, caption_type="tags", content=content))
    return item


def test_tag_filter_matches_complete_case_insensitive_tokens_only(tmp_path):
    db = _session(tmp_path)
    dataset = Dataset(name="test", path="test")
    db.add(dataset)
    db.flush()
    cat = _item(db, dataset.id, "cat", "Cat, solo")
    _item(db, dataset.id, "catgirl", "catgirl, solo")
    db.commit()

    assert exact_tag_item_ids(db, dataset.id, ["cat"]) == [cat.id]


def test_tag_filter_requires_all_tokens_and_treats_like_metacharacters_literally(tmp_path):
    db = _session(tmp_path)
    dataset = Dataset(name="test", path="test")
    db.add(dataset)
    db.flush()
    exact = _item(db, dataset.id, "exact", "100%, under_score, solo")
    _item(db, dataset.id, "partial", "1000, underXscore, solo")
    _item(db, dataset.id, "missing", "100%, solo")
    db.commit()

    assert exact_tag_item_ids(db, dataset.id, ["100%", "under_score"]) == [exact.id]


def test_tag_filter_honors_filename_search_and_dataset_scope(tmp_path):
    db = _session(tmp_path)
    first = Dataset(name="first", path="first")
    second = Dataset(name="second", path="second")
    db.add_all([first, second])
    db.flush()
    wanted = _item(db, first.id, "wanted_01", "tag")
    _item(db, first.id, "other_01", "tag")
    _item(db, second.id, "wanted_02", "tag")
    db.commit()

    assert exact_tag_item_ids(db, first.id, ["tag"], search="wanted") == [wanted.id]


def test_grid_page_keeps_heavy_item_columns_deferred(tmp_path):
    db = _session(tmp_path)
    dataset = Dataset(name="test", path="test")
    db.add(dataset)
    db.flush()
    dataset_id = dataset.id
    _item(db, dataset_id, "first", "tag")
    _item(db, dataset_id, "second", "tag")
    db.commit()
    db.expunge_all()

    items, total = dataset_item_page(
        db,
        dataset_id,
        page=1,
        page_size=1,
        search=None,
        tags=None,
        grid_projection=True,
    )

    assert total == 2
    assert len(items) == 1
    assert {"exif_data", "related_images", "image_hash"} <= inspect(items[0]).unloaded


def test_grid_cursor_counts_only_when_requested_and_has_stable_boundaries(tmp_path):
    db = _session(tmp_path)
    dataset = Dataset(name="cursor", path="cursor")
    db.add(dataset)
    db.flush()
    expected = [_item(db, dataset.id, f"item_{index}", "tag").id for index in range(5)]
    db.commit()

    first, total, cursor, has_more = dataset_item_cursor_page(
        db,
        dataset.id,
        after_id=None,
        page_size=2,
        search=None,
        tags=None,
        include_total=True,
    )
    second, no_total, next_cursor, second_has_more = dataset_item_cursor_page(
        db,
        dataset.id,
        after_id=cursor,
        page_size=2,
        search=None,
        tags=None,
        include_total=False,
    )
    last, _, last_cursor, last_has_more = dataset_item_cursor_page(
        db,
        dataset.id,
        after_id=next_cursor,
        page_size=2,
        search=None,
        tags=None,
        include_total=False,
    )

    assert [item.id for item in first] == expected[:2]
    assert total == 5
    assert cursor == expected[1]
    assert has_more
    assert [item.id for item in second] == expected[2:4]
    assert no_total is None
    assert next_cursor == expected[3]
    assert second_has_more
    assert [item.id for item in last] == expected[4:]
    assert last_cursor is None
    assert not last_has_more


def test_lightweight_dataset_detail_omits_tag_statistics(tmp_path):
    from api.routes import get_dataset

    db = _session(tmp_path)
    dataset = Dataset(
        name="test",
        path="test",
        tag_statistics={"tag": {"count": 1, "category": "General"}},
    )
    db.add(dataset)
    db.commit()
    dataset_id = dataset.id
    db.expunge_all()

    result = asyncio.run(get_dataset(dataset_id, False, db))

    assert "tag_statistics" not in result


def test_caption_type_match_rate_is_weighted_by_row_count(tmp_path):
    from api.routes import get_dataset_caption_types

    db = _session(tmp_path)
    dataset = Dataset(name="test", path="test")
    db.add(dataset)
    db.flush()
    low = _item(db, dataset.id, "low", "low")
    low_caption = db.query(DatasetCaption).filter_by(item_id=low.id).one()
    low_caption.source_field = "low"
    low_caption.tag_match_rate = 0.0
    for index in range(3):
        item = _item(db, dataset.id, f"high_{index}", "high")
        caption = db.query(DatasetCaption).filter_by(item_id=item.id).one()
        caption.source_field = "high"
        caption.tag_match_rate = 1.0
    db.commit()

    result = asyncio.run(get_dataset_caption_types(dataset.id, db))

    tags = next(row for row in result["caption_types"] if row["caption_type"] == "tags")
    assert tags["total_count"] == 4
    assert tags["avg_match_rate"] == 0.75


def test_natural_language_preview_applies_the_training_caption_dropout(tmp_path):
    from api.routes import get_random_caption

    db = _session(tmp_path)
    dataset = Dataset(
        name="test",
        path="test",
        caption_processing={
            "caption_types": ["natural_language"],
            "caption_dropout_rate": 1.0,
        },
    )
    db.add(dataset)
    db.flush()
    item = _item(db, dataset.id, "sample", "A complete prose caption.")
    caption = db.query(DatasetCaption).filter_by(item_id=item.id).one()
    caption.caption_type = "natural_language"
    caption.is_tags_format = False
    db.commit()

    result = asyncio.run(get_random_caption(dataset.id, None, db))

    assert result["caption"] == ""
