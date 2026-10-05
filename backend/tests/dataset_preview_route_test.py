import asyncio
import io
from types import SimpleNamespace

import pytest
import httpx
from fastapi import FastAPI
from PIL import Image

from api import routes
from database import get_datasets_db
from database.models import DatasetItem


@pytest.fixture
def preview_client(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    Image.new("RGB", (800, 400), "red").save(source)
    item = DatasetItem(id=1, dataset_id=7, item_type="single", image_path=str(source))
    monkeypatch.setattr(routes, "_dataset_item_or_404", lambda *args: item)
    monkeypatch.setattr(routes, "settings", SimpleNamespace(
        thumbnails_dir=str(tmp_path), cache_dir=str(tmp_path / "cache")
    ))
    app = FastAPI()
    app.add_api_route("/datasets/{dataset_id}/items/{item_id}/preview",
                      routes.get_dataset_item_preview, methods=["GET"])
    app.dependency_overrides[get_datasets_db] = lambda: None
    async def request(url, **kwargs):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            return await client.get(url, **kwargs)

    return lambda url, **kwargs: asyncio.run(request(url, **kwargs))


@pytest.mark.parametrize("size", [128, 256, 512, None])
def test_preview_accepts_browser_query_and_returns_cached_webp(preview_client, size):
    url = "/datasets/7/items/1/preview"
    if size is not None:
        url += f"?size={size}"
    response = preview_client(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    with Image.open(io.BytesIO(response.content)) as image:
        expected = size if size is not None else routes.DATASET_DEFAULTS["preview_size"]
        assert image.size == (expected, expected // 2)
    cached = preview_client(url, headers={"If-None-Match": response.headers["etag"]})
    assert cached.status_code == 304


@pytest.mark.parametrize("size", ["0", "129", "256.5", "text"])
def test_preview_rejects_unsupported_query_sizes(preview_client, size):
    assert preview_client(f"/datasets/7/items/1/preview?size={size}").status_code == 422
