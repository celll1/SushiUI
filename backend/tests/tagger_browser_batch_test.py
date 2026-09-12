"""Behavior contracts for tagger browser batch inference and sidecars."""

from contextlib import asynccontextmanager
import asyncio
import json

import torch
from PIL import Image

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
    from core.tagger.browser_sidecars import write_image_sidecar

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image placeholder")
    sidecar = tmp_path / "sample.txt"
    sidecar.write_text("old", encoding="utf-8")

    assert write_image_sidecar(str(image_path), ["a", "b"]) == str(sidecar)
    assert sidecar.read_text(encoding="utf-8") == "a, b"
    assert list(tmp_path.glob("*.tmp")) == []


def test_browser_batch_infer_uses_bytes_gpu_slot_and_writes_names(
    tmp_path, monkeypatch,
):
    from api import routes
    import core.gpu_coordinator as coordinator_module

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    routes._browser_root = str(tmp_path)

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
