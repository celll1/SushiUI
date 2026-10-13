import importlib.util
import py_compile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
import yaml
import pytest
from PIL import Image
from pydantic import TypeAdapter, ValidationError

_MODULE_PATH = Path(__file__).parents[1] / "api" / "training_media.py"
_SPEC = importlib.util.spec_from_file_location("training_media_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MEDIA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MEDIA)

cached_debug_preview = _MEDIA.cached_debug_preview
cached_image_preview = _MEDIA.cached_image_preview
debug_image_urls = _MEDIA.debug_image_urls
debug_display_resolution = _MEDIA.debug_display_resolution
file_fingerprint = _MEDIA.file_fingerprint
find_debug_latent_file = _MEDIA.find_debug_latent_file
publish_sample_image = _MEDIA.publish_sample_image
SampleMetadata = _MEDIA.SampleMetadata
cached_sample_params = _MEDIA.cached_sample_params
PreviewSize = _MEDIA.PreviewSize

_REPO = Path(__file__).parents[2]


def test_sample_preview_is_sized_versioned_and_reused(tmp_path: Path):
    source = tmp_path / "step_000100_sample_0.png"
    Image.effect_noise((1024, 768), 80).convert("RGB").save(source)

    preview, etag = cached_image_preview(source, tmp_path / ".previews", 256)
    again, again_etag = cached_image_preview(source, tmp_path / ".previews", 256)

    assert preview == again
    assert etag == again_etag
    assert file_fingerprint(source) in preview.name
    with Image.open(preview) as image:
        assert image.format == "WEBP"
        assert max(image.size) == 256
    assert preview.stat().st_size < source.stat().st_size


@pytest.mark.parametrize("suffix,image_format", [(".webp", "WEBP"), (".png", "PNG")])
def test_sample_appears_only_after_encoding(tmp_path: Path, suffix, image_format):
    target = tmp_path / f"step_000100_sample_0{suffix}"
    info = SampleMetadata()
    info.add_text("prompt", 'test "prompt" <with> & markup')

    class ObservedImage:
        def save(self, pending, **kwargs):
            assert not target.exists()
            assert pending.suffix == ".tmp"
            assert kwargs["format"] == image_format
            Image.new("RGB", (8, 8), "red").save(pending, **kwargs)
            assert not target.exists()

    publish_sample_image(ObservedImage(), target, info)
    assert list(tmp_path.iterdir()) == [target]
    with Image.open(target) as image:
        assert image.format == image_format
    assert cached_sample_params(target) == {"prompt": 'test "prompt" <with> & markup'}


def test_failed_sample_png_keeps_previous_complete_file(tmp_path: Path):
    target = tmp_path / "step_000100_sample_0.png"
    Image.new("RGB", (8, 8), "blue").save(target)

    class FailedImage:
        def save(self, pending, **kwargs):
            pending.write_bytes(b"partial PNG")
            raise OSError("encoder failed")

    with pytest.raises(OSError, match="encoder failed"):
        publish_sample_image(FailedImage(), target, SampleMetadata())
    assert list(tmp_path.iterdir()) == [target]
    with Image.open(target) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_debug_manifest_lists_individual_images_and_renders_only_requested_kind(
    tmp_path: Path,
):
    debug_dir = tmp_path / "step_000200"
    debug_dir.mkdir()
    latent_file = debug_dir / "latents_t0.4023.pt"
    data = {
        "timestep": 0.40234375,
        "model_type": "sensenova_sdxl_chimera",
        "latents": torch.randn(1, 4, 128, 128),
        "noisy_latents": torch.randn(1, 4, 128, 128),
        "predicted_latent": torch.randn(1, 4, 128, 128),
    }
    torch.save(data, latent_file)

    found = find_debug_latent_file(debug_dir, 0.40230001)
    assert found == latent_file
    urls = debug_image_urls(139, 200, data, latent_file)
    assert set(urls) == {
        "latents_image", "noisy_latents_image", "predicted_latent_image"
    }
    assert all("/images/" in url and "v=" in url for url in urls.values())
    manifest_timestep = parse_qs(
        urlparse(urls["latents_image"]).query
    )["timestep"][0]
    assert manifest_timestep == "0.4023"
    assert find_debug_latent_file(debug_dir, float(manifest_timestep)) == latent_file

    preview, _etag = cached_debug_preview(latent_file, data, "target", 256)
    assert preview.is_file()
    assert list((debug_dir / ".previews").glob("*.webp")) == [preview]
    with Image.open(preview) as image:
        assert max(image.size) == 128


def test_preview_size_query_coerces_url_text_to_integer_enum():
    adapter = TypeAdapter(PreviewSize)
    assert adapter.validate_python("256") is PreviewSize.SMALL
    assert adapter.validate_python("512") is PreviewSize.MEDIUM
    with pytest.raises(ValidationError):
        adapter.validate_python("123")


def test_uncond_debug_preview_supports_tensor_decode_and_ignores_stale_webp(tmp_path):
    latent_file = tmp_path / "latents_t0.5000.pt"
    data = {"model_type": "qwen_image_21", "uncond_predicted_latent": torch.ones(1, 4, 8, 8)}
    torch.save(data, latent_file)
    urls = debug_image_urls(1, 2, data, latent_file)
    assert "/images/uncond_predicted_latent?" in urls["uncond_predicted_latent_image"]
    preview, _ = cached_debug_preview(latent_file, data, "uncond_predicted_latent", 256)
    assert preview.exists()
    decoded = tmp_path / "decode_t0.5000_uncond_pred_x0.webp"
    Image.new("RGB", (900, 600), "orange").save(decoded, format="WEBP")
    preview, _ = cached_debug_preview(latent_file, data, "uncond_predicted_latent", 512)
    with Image.open(preview) as image:
        assert image.size == (512, 341)
    assert "uncond_predicted_latent_image" not in debug_image_urls(1, 2, {}, latent_file)


def test_uncond_debug_api_manifest_and_legacy_base64(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from api.routes import visualize_debug_latent
    class Db:
        def query(self, _model): return self
        def filter(self, _condition): return self
        def first(self): return SimpleNamespace(output_dir=str(tmp_path))
    debug_dir = tmp_path / "debug" / "step_000200"
    debug_dir.mkdir(parents=True)
    torch.save({"timestep": 0.5, "loss": 0.1, "model_type": "qwen_image_21",
                "latents": torch.zeros(1, 4, 8, 8),
                "uncond_predicted_latent": torch.ones(1, 4, 8, 8),
                "uncond_prediction_status": "computed"}, debug_dir / "latents_t0.5000.pt")
    manifest = asyncio.run(visualize_debug_latent(1, 200, include_images=False, db=Db()))
    assert manifest["uncond_prediction_status"] == "computed"
    assert "/images/uncond_predicted_latent?" in manifest["image_urls"]["uncond_predicted_latent_image"]
    assert "uncond_predicted_latent_image" not in manifest
    legacy = asyncio.run(visualize_debug_latent(1, 200, include_images=True, db=Db()))
    assert legacy["uncond_predicted_latent_image"]


def test_debug_preview_prefers_saved_decode_webp(tmp_path: Path):
    latent_file = tmp_path / "latents_t7.pt"
    data = {"timestep": 7, "latents": torch.zeros(1, 4, 32, 32)}
    torch.save(data, latent_file)
    decoded = tmp_path / "decode_t7_target.webp"
    Image.new("RGB", (900, 600), "navy").save(decoded, format="WEBP")

    preview, _etag = cached_debug_preview(latent_file, data, "target", 512)
    with Image.open(preview) as image:
        assert image.size == (512, 341)
    assert debug_display_resolution(data, latent_file) == {
        "width": 900, "height": 600, "space": "image"
    }


def test_debug_resolution_uses_latent_grid_without_decode(tmp_path: Path):
    latent_file = tmp_path / "latents_t1.pt"
    assert debug_display_resolution(
        {"model_type": "qwen_image_21", "latents": torch.zeros(1, 16, 64, 80)},
        latent_file,
    ) == {"width": 80, "height": 64, "space": "latent"}


def test_openapi_and_frontend_use_compact_incremental_media_contract():
    spec = yaml.safe_load((_REPO / "openapi.yaml").read_text(encoding="utf-8"))
    paths = spec["paths"]
    sample_list = paths["/training/runs/{run_id}/samples"]["get"]
    debug_list = paths["/training/runs/{run_id}/debug-latents"]["get"]
    visualize = paths[
        "/training/runs/{run_id}/debug-latents/{step}/visualize"
    ]["get"]
    assert "since_step" in {p["name"] for p in sample_list["parameters"]}
    assert "since_step" in {p["name"] for p in debug_list["parameters"]}
    include = next(p for p in visualize["parameters"] if p["name"] == "include_images")
    assert include["schema"]["default"] is True
    assert "/training/runs/{run_id}/samples/{filename}/preview" in paths
    assert "/training/runs/{run_id}/debug-latents/{step}/images/{kind}" in paths

    api_source = (_REPO / "frontend" / "src" / "utils" / "api.ts").read_text(
        encoding="utf-8"
    )
    monitor_source = (
        _REPO / "frontend" / "src" / "components" / "training" / "TrainingMonitor.tsx"
    ).read_text(encoding="utf-8")
    assert "include_images: false" in api_source
    assert "preview_path?: string" in api_source
    assert 'viewMode !== "samples"' in monitor_source
    assert 'viewMode !== "debug"' in monitor_source
    assert "sampleUpgradeTimerRef.current = setTimeout" in monitor_source


def test_changed_backend_modules_compile(tmp_path: Path):
    for relative in (
        "backend/api/training_media.py",
        "backend/api/param_defaults.py",
        "backend/api/routes.py",
    ):
        source = _REPO / relative
        target = tmp_path / f"{source.stem}.pyc"
        py_compile.compile(str(source), cfile=str(target), doraise=True)
