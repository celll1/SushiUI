import asyncio

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None

from core.datasets.revisions import bump_dataset_revision
from core.training.train_runner import _compute_dataset_cache_key
from database.models import Dataset, DatasetBase, DatasetItem


def test_training_snapshot_key_uses_dataset_revision(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    dataset = Dataset(name="test", path=str(tmp_path))
    db.add(dataset)
    db.commit()

    before = _compute_dataset_cache_key(db, [dataset.id], ["tags"])
    bump_dataset_revision(dataset)
    db.commit()
    after = _compute_dataset_cache_key(db, [dataset.id], ["tags"])

    assert before != after
    assert after == _compute_dataset_cache_key(db, [dataset.id], ["tags"])
    assert after != _compute_dataset_cache_key(
        db, [dataset.id], ["natural_language", "tags"]
    )
    assert _compute_dataset_cache_key(
        db, [dataset.id], ["natural_language", "tags"]
    ) != _compute_dataset_cache_key(db, [dataset.id], ["tags", "natural_language"])


def test_reference_mutation_persists_json_and_bumps_revision(tmp_path):
    from api.routes import ReferenceImagesUpdateRequest, update_item_reference_images

    engine = create_engine(f"sqlite:///{tmp_path / 'refs.db'}")
    DatasetBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    dataset = Dataset(name="refs", path=str(tmp_path))
    db.add(dataset)
    db.flush()
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    target.write_bytes(b"target")
    reference.write_bytes(b"reference")
    item = DatasetItem(
        dataset_id=dataset.id,
        base_name="target",
        image_path=str(target),
        related_images={"reference": []},
    )
    db.add(item)
    db.commit()

    asyncio.run(update_item_reference_images(
        item.id,
        ReferenceImagesUpdateRequest(reference_images=[str(reference)]),
        db,
    ))
    db.expire_all()

    assert db.get(DatasetItem, item.id).related_images == {"reference": [str(reference)]}
    assert db.get(Dataset, dataset.id).revision == 1
