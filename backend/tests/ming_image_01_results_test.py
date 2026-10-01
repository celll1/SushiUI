"""CPU RGBA persistence, group ownership and failure rollback contracts."""

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming result test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda

from api import ming_image_results as persistence
from core.models.ming_image_01.results import MingGenerationOutput, MingResultGroup
from database.models import GalleryBase, GeneratedImage
from config.settings import settings
from utils.image_utils import save_image_with_metadata


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "outputs_dir", str(tmp_path / "outputs"))
    monkeypatch.setattr(settings, "thumbnails_dir", str(tmp_path / "thumbnails"))
    engine = create_engine("sqlite:///:memory:")
    GalleryBase.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def output(profile="layer_decompose", batch=1):
    roles = ["image"] if profile == "generation_edit" else ["auxiliary_canvas", "layer_1", "layer_2"]
    return MingGenerationOutput(tuple(MingResultGroup(
        [Image.new("RGBA", (16, 16), (index * 40, b * 30, 90, 37 + index)) for index in range(len(roles))],
        roles.copy(), profile, 2, len(roles), 33 + b, 16, 16, "pil") for b in range(batch)))


def save(value, db, **kwargs):
    return persistence.persist_ming_results(value, db=db,
        params={"prompt": "layers", "steps": 2, "cfg_scale": 2., "ref_images": [Image.new("RGBA", (2, 2))]},
        generation_type="img2img", model_info={"type": "ming_image_01", "profile": value.groups[0].profile},
        model_name="Ming", model_hash="hash", **kwargs)


def assert_empty(db):
    assert db.query(GeneratedImage).count() == 0
    assert not list(Path(settings.outputs_dir).glob("*.png"))
    assert not list(Path(settings.thumbnails_dir).glob("*"))


def test_layer_groups_preserve_alpha_order_database_and_png_metadata(store):
    value = output(batch=2)
    response = save(value, store)
    assert len(response["images"]) == 6 and len(response["groups"]) == 2
    assert response["image"] == response["images"][0]
    assert store.query(GeneratedImage).count() == 6
    for batch, group in enumerate(response["groups"]):
        assert group["roles"] == value.groups[batch].roles
        for frame, filename in enumerate(group["filenames"]):
            with Image.open(Path(settings.outputs_dir) / filename) as actual:
                assert actual.mode == "RGBA" and actual.tobytes() == value.groups[batch].values[frame].tobytes()
                meta = json.loads(actual.text["ming_image_01_actual"])
                assert meta["group_id"] == group["group_id"]
                assert meta["frame_index"] == frame and meta["role"] == group["roles"][frame]
            row = store.get(GeneratedImage, group["image_ids"][frame])
            assert row.parameters["ming_image_01_actual"] == meta and row.seed == 33 + batch
            assert isinstance(row.parameters["ref_images"][0], str)
    second = save(value, store)
    assert not set(response["groups"][0]["filenames"]) & set(second["groups"][0]["filenames"])
    assert store.query(GeneratedImage).count() == 12
    assert not torch.cuda.is_initialized()


def test_design_and_skip_gallery_keep_explicit_groups_without_rows(store):
    response = save(output("generation_edit"), store, skip_gallery=True)
    assert response["groups"][0]["roles"] == ["image"]
    assert response["groups"][0]["image_ids"] == []
    assert response["filename"] == response["groups"][0]["filenames"][0]
    assert store.query(GeneratedImage).count() == 0
    assert not list(Path(settings.thumbnails_dir).glob("*"))


@pytest.mark.parametrize("failure", ["thumbnail", "flush", "commit", "cancel"])
def test_partial_group_failure_rolls_back_files_and_rows(store, monkeypatch, failure):
    if failure == "thumbnail":
        original = persistence.create_thumbnail
        def broken(path):
            original(path)
            raise RuntimeError("thumbnail failure")
        monkeypatch.setattr(persistence, "create_thumbnail", broken)
    elif failure in {"flush", "commit"}:
        def broken(*a, **k): raise RuntimeError("database failure")
        monkeypatch.setattr(store, failure, broken)
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return failure == "cancel" and calls >= 3
    with pytest.raises(RuntimeError):
        save(output(), store, cancelled=cancelled)
    assert not list(Path(settings.outputs_dir).glob("*.png"))
    assert not list(Path(settings.thumbnails_dir).glob("*"))
    with Session(store.bind) as verification:
        assert verification.query(GeneratedImage).count() == 0


@pytest.mark.parametrize("corruption", ["roles", "rgb", "dimensions", "latent", "profile"])
def test_invalid_native_output_is_rejected_before_writes(store, corruption):
    value = output()
    group = value.groups[0]
    if corruption == "roles": group.roles.reverse()
    elif corruption == "rgb": group.values[0] = group.values[0].convert("RGB")
    elif corruption == "dimensions": group.width = 32
    elif corruption == "latent": group.output_type = "latent"
    else: group.profile = "unknown"
    with pytest.raises(ValueError): save(value, store)
    assert_empty(store)


def test_explicit_filename_collision_never_overwrites_existing_png(store):
    image = output().primary
    name = save_image_with_metadata(image, {"seed": 1}, output_filename="fixed.png")
    before = (Path(settings.outputs_dir) / name).read_bytes()
    with pytest.raises(FileExistsError):
        save_image_with_metadata(Image.new("RGBA", image.size), {"seed": 1}, output_filename="fixed.png")
    assert (Path(settings.outputs_dir) / name).read_bytes() == before
    with pytest.raises(ValueError):
        save_image_with_metadata(image, {}, output_filename="../escape.png")


def test_persistence_refuses_loaded_profile_mismatch_before_writes(store):
    with pytest.raises(ValueError, match="loaded checkpoint"):
        persistence.persist_ming_results(output(), db=store, params={}, generation_type="img2img",
            model_info={"type": "ming_image_01", "profile": "generation_edit"}, model_name="Ming", model_hash="hash")
    assert_empty(store)


def test_route_finish_uses_owned_snapshot_and_preserves_source_mask_and_status(store, monkeypatch):
    from utils.image_utils import calculate_image_hash, encode_mask_to_base64
    source = Image.new("RGBA", (16, 16), (20, 30, 40, 128))
    mask = Image.new("L", source.size, 80)
    info = {"type": "ming_image_01", "profile": "layer_decompose", "source": "original-Ming",
            "model_hash": "original-hash", "dit_rewiring": {"enabled": True}}
    value = MingGenerationOutput(output().groups, info)
    info["source"] = "another-model"
    info["dit_rewiring"]["enabled"] = False
    assert value.model_info["dit_rewiring"]["enabled"] is True
    statuses = []
    monkeypatch.setattr(persistence, "complete_generation", lambda result, **kw: statuses.append((result, kw)))
    response = persistence.finish_ming_generation(value, db=store,
        params={"seed": 33, "steps": 2, "cfg_scale": 2., "generation_time": 0.1},
        generation_type="inpaint", generation_id=None, lora_configs=[], skip_gallery=False,
        source_image=source, mask_image=mask)
    assert len(response["images"]) == 3 and response["warnings"] == []
    assert statuses[0][0]["groups"] == response["groups"]
    assert statuses[0][0]["filename"] == response["image"]["filename"]
    for row in store.query(GeneratedImage).all():
        assert row.model_name == "original-Ming" and row.model_hash == "original-hash"
        assert row.source_image_hash == calculate_image_hash(source)
        assert row.mask_data == encode_mask_to_base64(mask)
        assert row.parameters["generation_time"] == 0.1
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("problem", ["missing_snapshot", "wrong_seed"])
def test_route_finish_rejects_missing_identity_before_writes(store, problem):
    value = output()
    if problem == "wrong_seed":
        value = MingGenerationOutput(value.groups, {"type": "ming_image_01", "profile": "layer_decompose"})
    with pytest.raises(ValueError):
        persistence.finish_ming_generation(value, db=store, params={"seed": -1},
            generation_type="txt2img", generation_id=None, lora_configs=[], skip_gallery=False)
    assert_empty(store)


def test_four_routes_handle_rich_output_before_generic_latent_or_single_image_save():
    import ast
    module = ast.parse((Path(__file__).resolve().parents[1] / "api/routes.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in module.body if isinstance(node, ast.AsyncFunctionDef)}
    for name in ("generate_txt2img", "generate_img2img", "generate_inpaint", "generate_outpaint"):
        function = functions[name]
        finish = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "finish_ming_generation"]
        assert len(finish) == 1
        generic = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id in {"store_latent", "save_image_with_metadata"}]
        assert generic and all(finish[0].lineno < call.lineno for call in generic)
        assert any(key.arg == "skip_gallery" for key in finish[0].keywords)
