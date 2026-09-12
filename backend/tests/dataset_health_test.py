from datetime import datetime, timedelta
from types import SimpleNamespace

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.datasets.health import inspect_dataset_rows
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None


def _row(path, **overrides):
    values = {
        "base_name": path.stem,
        "image_path": str(path),
        "item_type": "single",
        "related_images": None,
        "width": 64,
        "height": 64,
        "total_captions": 1,
        "total_tags": 1,
        "has_file_caption": True,
        "indexed_at": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_health_reports_source_and_index_drift(tmp_path):
    valid = tmp_path / "valid.png"
    valid.write_bytes(b"png")
    valid.with_suffix(".txt").write_text("tag", encoding="utf-8")
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"png")
    invalid.with_suffix(".json").write_text("[]", encoding="utf-8")
    missing = tmp_path / "missing.png"

    result = inspect_dataset_rows(
        [_row(valid), _row(invalid), _row(missing)],
        last_scanned_at=datetime.utcnow() - timedelta(days=1),
    )

    assert result["healthy"] is False
    assert result["counts"]["missing_media"] == 1
    assert result["counts"]["missing_sidecar"] == 1
    assert result["counts"]["invalid_sidecar"] == 1
    assert result["counts"]["stale_sidecar"] == 2


def test_health_checks_reference_pairs_and_metadata(tmp_path):
    target = tmp_path / "pair_target.png"
    target.write_bytes(b"png")
    target.with_suffix(".txt").write_text("tag", encoding="utf-8")

    result = inspect_dataset_rows(
        [_row(target, item_type="reference", related_images={"reference": []}, width=None)],
        last_scanned_at=None,
    )

    assert result["counts"]["reference_failure"] == 1
    assert result["counts"]["metadata_gap"] == 1


def test_health_route_projects_caption_state(tmp_path):
    from api.routes import get_dataset_health

    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    media = tmp_path / "image.png"
    media.write_bytes(b"png")
    media.with_suffix(".txt").write_text("tag", encoding="utf-8")
    dataset = Dataset(name="test", path=str(tmp_path))
    db.add(dataset)
    db.flush()
    item = DatasetItem(
        dataset_id=dataset.id,
        base_name="image",
        image_path=str(media),
        width=64,
        height=64,
        total_captions=1,
        total_tags=1,
    )
    db.add(item)
    db.flush()
    db.add(DatasetCaption(item_id=item.id, caption_type="tags", content="tag", source="file"))
    db.commit()

    result = get_dataset_health(dataset.id, db)

    assert result["healthy"] is True
    assert result["counts"]["captioned_items"] == 1
    assert result["counts"]["tagged_items"] == 1
