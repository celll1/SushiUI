import asyncio

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.datasets.queries import exact_tag_item_ids
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
