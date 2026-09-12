"""Behavior contracts for tagger browser batch inference and sidecars."""

from contextlib import asynccontextmanager
import asyncio
import json

import torch
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None


def test_prediction_tag_names_preserves_response_order_and_deduplicates():
    from core.tagger.browser_sidecars import prediction_tag_names

    result = {
        "tags": [{"tag": "1girl"}, {"tag": "blue_hair"}],
        "quality_top": {"tag": "masterpiece"},
        "rating_top": {"tag": "general"},
    }
    assert prediction_tag_names(result) == [
        "1girl", "blue_hair", "masterpiece", "general",
    ]

    result["quality_top"] = {"tag": "1girl"}
    assert prediction_tag_names(result) == ["1girl", "blue_hair", "general"]


def test_write_image_sidecar_replaces_complete_file(tmp_path):
    from core.tagger.browser_sidecars import read_image_sidecar, write_image_sidecar

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image placeholder")
    sidecar = tmp_path / "sample.txt"
    sidecar.write_text("old", encoding="utf-8")

    assert write_image_sidecar(str(image_path), ["a", "b"]) == str(sidecar)
    assert sidecar.read_text(encoding="utf-8") == "a, b"
    assert read_image_sidecar(str(image_path)) == (["a", "b"], "a, b")
    assert list(tmp_path.glob("*.tmp")) == []


def test_browser_batch_infer_uses_bytes_gpu_slot_and_writes_names(
    tmp_path, monkeypatch,
):
    from api import routes
    import core.gpu_coordinator as coordinator_module

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    workspace_id, _ = routes._create_browser_workspace(str(tmp_path))

    calls = {"slots": 0, "predict": 0}

    class FakeManager:
        status = {"loaded": True}

        def predict(self, image_bytes, *, use_ood_detection=False):
            calls["predict"] += 1
            assert isinstance(image_bytes, bytes)
            assert image_bytes.startswith(b"\x89PNG")
            assert use_ood_detection is True
            return {
                "tags": [{"tag": "red_background"}],
                "quality_top": {"tag": "high_quality"},
                "rating_top": {"tag": "general"},
            }

    class FakeCoordinator:
        @asynccontextmanager
        async def generation_slot(self, **kwargs):
            assert kwargs == {"estimated_peak_gb": 2.5, "timeout": 60.0}
            calls["slots"] += 1
            yield

    monkeypatch.setattr(routes, "get_siglip2_inference_manager", lambda: FakeManager())
    monkeypatch.setattr(coordinator_module, "gpu_coordinator", FakeCoordinator())

    async def run_request():
        response = await routes.browser_batch_infer(
            routes.BrowserBatchInferRequest(
                rel_paths=["sample.png"], overwrite=True, use_ood_detection=True,
                workspace_id=workspace_id,
            )
        )
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return chunks

    chunks = asyncio.run(run_request())
    events = [
        json.loads(line[6:])
        for line in "".join(chunks).splitlines()
        if line.startswith("data: ")
    ]

    assert calls == {"slots": 1, "predict": 1}
    assert [event["type"] for event in events] == ["done", "complete"]
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == (
        "red_background, high_quality, general"
    )


def test_registered_workspace_saves_through_dataset_index(tmp_path, monkeypatch):
    from api import routes
    from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem

    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    dataset = Dataset(name="registered", path=str(tmp_path))
    db.add(dataset)
    db.flush()
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    Image.new("RGB", (4, 4), "blue").save(tmp_path / "not_indexed.png")
    sidecar_path = tmp_path / "sample.json"
    sidecar_path.write_text(
        json.dumps({"tags": "old", "private_metadata": {"keep": True}}),
        encoding="utf-8",
    )
    item = DatasetItem(
        dataset_id=dataset.id,
        base_name="sample",
        image_path=str(image_path),
    )
    db.add(item)
    db.flush()
    db.add(DatasetCaption(
        item_id=item.id,
        caption_type="tags",
        content="old",
        source_field="tags",
        is_tags_format=True,
    ))
    db.commit()
    workspace_id, _ = routes._create_browser_workspace(
        str(tmp_path), dataset_id=dataset.id
    )
    monkeypatch.setattr(routes.taglist_cache, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        routes.taglist_cache,
        "get_categories_batch",
        lambda tags: {tag: "General" for tag in tags},
    )

    asyncio.run(routes.browser_save_tags(
        routes.BrowserSaveTagsRequest(
            workspace_id=workspace_id,
            rel_path="sample.png",
            tags=["new", "second"],
        ),
        db,
    ))
    result = asyncio.run(routes.browser_get_tags("sample.png", workspace_id, db))
    listing = asyncio.run(routes.browser_list(
        recursive=False,
        include_tags=True,
        workspace_id=workspace_id,
        db=db,
    ))
    try:
        asyncio.run(routes.browser_image(
            "not_indexed.png",
            workspace_id=workspace_id,
            db=db,
        ))
    except routes.HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("Unindexed images must not be served by a dataset workspace")

    db.expire_all()
    assert result == {"tags": ["new", "second"], "raw": "new, second"}
    assert listing["images"] == [{
        "rel_path": "sample.png",
        "has_tags": True,
        "mtime": image_path.stat().st_mtime,
        "tags": ["new", "second"],
    }]
    assert db.query(DatasetCaption).filter_by(item_id=item.id).one().content == "new, second"
    assert db.get(Dataset, dataset.id).revision == 1
    assert json.loads(sidecar_path.read_text(encoding="utf-8")) == {
        "tags": "new, second",
        "private_metadata": {"keep": True},
    }
    assert not (tmp_path / "sample.txt").exists()
