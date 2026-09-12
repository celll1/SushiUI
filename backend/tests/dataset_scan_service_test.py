"""End-to-end contract for the route-independent dataset scanner."""

import asyncio

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
