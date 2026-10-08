"""End-to-end contract for the route-independent dataset scanner."""

import asyncio
import json

import pytest

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_scan_service_indexes_media_caption_and_revision(tmp_path, monkeypatch):
    from core.datasets import scanning
    from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem
    import utils.dataset_structure_detector as structure_detector
    import utils.taglist_loader as taglist_loader

    Image.new("RGB", (8, 6), "red").save(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("1girl, solo, red_hair", encoding="utf-8")

    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    dataset = Dataset(name="scan", path=str(tmp_path), recursive=True)
    db.add(dataset)
    db.commit()

    monkeypatch.setattr(
        structure_detector,
        "detect_dataset_structure",
        lambda *args, **kwargs: {
            "structure_type": "single",
            "reference_suffixes": [],
            "target_suffixes": [],
            "confidence": 1.0,
        },
    )
    monkeypatch.setattr(
        taglist_loader,
        "load_all_tags",
        lambda *args, **kwargs: {"1girl", "solo", "red_hair"},
    )
    monkeypatch.setattr(scanning, "create_thumbnail", lambda *args, **kwargs: None)
    monkeypatch.setattr(scanning.taglist_cache, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        scanning.taglist_cache,
        "get_categories_batch",
        lambda tags: {tag: "General" for tag in tags},
    )
    progress = []

    result = asyncio.run(scanning.scan_dataset_index(
        dataset.id,
        db,
        progress=lambda step, total, message: progress.append((step, total, message)),
    ))

    db.expire_all()
    item = db.query(DatasetItem).one()
    caption = db.query(DatasetCaption).one()
    assert result["items_found"] == 1
    assert result["captions_found"] == 1
    assert (item.width, item.height) == (8, 6)
    assert caption.content == "1girl, solo, red_hair"
    assert caption.is_tags_format is True
    assert db.get(Dataset, dataset.id).revision == 1
    assert progress[-1][0] == progress[-1][1]

    (tmp_path / "sample.txt").unlink()
    second = asyncio.run(scanning.scan_dataset_index(dataset.id, db, incremental=True))

    db.expire_all()
    refreshed = db.get(Dataset, dataset.id)
    assert second["captions_updated"] == 1
    assert db.query(DatasetCaption).count() == 0
    assert refreshed.total_tags == 0
    assert refreshed.tag_statistics == {}
    assert refreshed.revision == 2


@pytest.mark.parametrize("fail_flush", [False, True])
def test_scan_handles_surrogates_and_aborts_failed_transactions(tmp_path, monkeypatch, fail_flush):
    from sqlalchemy import event
    from core.datasets import scanning
    from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem
    import utils.dataset_structure_detector as detector
    import utils.taglist_loader as loader

    Image.new("RGB", (8, 8), "white").save(tmp_path / "sample.png")
    (tmp_path / "sample.json").write_text(json.dumps({
        "title": "before\ud83dafter", "nested": {"key\ud83d": "pair\ud83d\ude00"},
    }), encoding="utf-8")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    DatasetBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False)()
    dataset = Dataset(name="synthetic", path=str(tmp_path), recursive=True)
    db.add(dataset)
    db.commit()
    monkeypatch.setattr(detector, "detect_dataset_structure", lambda *a, **k: {"structure_type": "single"})
    monkeypatch.setattr(loader, "load_all_tags", lambda *a, **k: set())
    monkeypatch.setattr(scanning, "create_thumbnail", lambda *a, **k: None)
    monkeypatch.setattr(scanning.taglist_cache, "initialize", lambda *a, **k: None)

    if fail_flush:
        def poison_item(session, _context, _instances):
            for item in session.new:
                if isinstance(item, DatasetItem):
                    item.base_name = "bad\ud83d"
        event.listen(db, "before_flush", poison_item)
        with pytest.raises(UnicodeEncodeError):
            asyncio.run(scanning.scan_dataset_index(dataset.id, db))
        assert db.is_active
        assert db.query(DatasetItem).count() == 0
        assert dataset.total_items == 0
        assert dataset.last_scanned_at is None
    else:
        asyncio.run(scanning.scan_dataset_index(dataset.id, db))
        fields = {row.caption_type: row.content for row in db.query(DatasetCaption)}
        assert fields == {"title": "before\ufffdafter", "nested.key\ufffd": "pair\U0001f600"}
        assert dataset.total_items == 1
    db.close()
