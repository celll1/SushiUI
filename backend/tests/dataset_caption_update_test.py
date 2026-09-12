from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.datasets.captions import CaptionSelectionError, update_caption
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _dataset_item(db, tmp_path):
    media = tmp_path / "image.webp"
    media.write_bytes(b"image")
    dataset = Dataset(name="test", path=str(tmp_path), tag_statistics={})
    db.add(dataset)
    db.flush()
    item = DatasetItem(
        dataset_id=dataset.id,
        item_type="single",
        base_name="image",
        image_path=str(media),
    )
    db.add(item)
    db.commit()
    return dataset, item, media


def test_update_persists_sidecar_db_and_empty_cached_statistics(db, tmp_path):
    dataset, item, media = _dataset_item(db, tmp_path)

    result = update_caption(
        db,
        dataset_id=dataset.id,
        item_id=item.id,
        caption_type="tags",
        content="one, two",
        tag_data=[{"tag": "one", "category": "General"}],
        persist_sidecar=True,
    )

    assert media.with_suffix(".txt").read_text(encoding="utf-8") == "one, two"
    assert result.caption.is_tags_format is True
    assert result.caption.field_category == "training"
    assert dataset.total_tags == 1
    assert dataset.revision == 1
    assert dataset.tag_statistics == {
        "one": {"count": 1, "category": "General"},
        "two": {"count": 1, "category": "Unknown"},
    }


def test_caption_id_selects_one_of_multiple_caption_rows(db, tmp_path):
    dataset, item, _ = _dataset_item(db, tmp_path)
    first = DatasetCaption(item_id=item.id, caption_type="tags", content="first")
    second = DatasetCaption(item_id=item.id, caption_type="tags", content="second")
    db.add_all([first, second])
    db.commit()

    with pytest.raises(CaptionSelectionError, match="Multiple captions"):
        update_caption(
            db,
            dataset_id=dataset.id,
            item_id=item.id,
            caption_type="tags",
            content="ambiguous",
        )

    result = update_caption(
        db,
        dataset_id=dataset.id,
        item_id=item.id,
        caption_id=second.id,
        caption_type="tags",
        content="chosen",
    )
    assert result.caption.id == second.id
    assert first.content == "first"


def test_dataset_scope_rejects_foreign_item(db, tmp_path):
    dataset, item, _ = _dataset_item(db, tmp_path)

    with pytest.raises(LookupError, match="not found"):
        update_caption(
            db,
            dataset_id=dataset.id + 1,
            item_id=item.id,
            caption_type="tags",
            content="wrong dataset",
        )


def test_db_commit_failure_restores_existing_sidecar(db, tmp_path, monkeypatch):
    dataset, item, media = _dataset_item(db, tmp_path)
    sidecar = media.with_suffix(".txt")
    sidecar.write_text("before", encoding="utf-8")
    caption = DatasetCaption(
        item_id=item.id,
        caption_type="tags",
        content="before",
        is_tags_format=True,
    )
    db.add(caption)
    db.commit()

    def fail_commit():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(db, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="db unavailable"):
        update_caption(
            db,
            dataset_id=dataset.id,
            item_id=item.id,
            caption_id=caption.id,
            caption_type="tags",
            content="after",
            persist_sidecar=True,
        )

    assert sidecar.read_text(encoding="utf-8") == "before"
    assert db.get(DatasetCaption, caption.id).content == "before"


def test_db_commit_failure_removes_new_sidecar(db, tmp_path, monkeypatch):
    dataset, item, media = _dataset_item(db, tmp_path)

    monkeypatch.setattr(
        db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("db unavailable"))
    )
    with pytest.raises(RuntimeError, match="db unavailable"):
        update_caption(
            db,
            dataset_id=dataset.id,
            item_id=item.id,
            caption_type="tags",
            content="new",
            persist_sidecar=True,
        )

    assert not Path(media).with_suffix(".txt").exists()
