"""Paired RGBA supervision records and real native CPU training integration."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming dataset test attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.dataset import inspect_item, load_pixels, pixel_cache_identity, prepare_record_batch, sidecar_path
from core.training.arch import get_arch_handler
from core.training.arch.base_arch import TrainStepContext
from ming_image_01_training_batch_test import components, trainer


def layer_files(root, count=2):
    root.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:32, :64]
    rgba = np.stack([x * 3, y * 7, (x+y) * 2, x * 4], axis=-1).astype(np.uint8)
    primary = root / "canvas.png"
    Image.fromarray(rgba).save(primary)
    paths = ["canvas.png"]
    for index in range(1, count):
        name = f"layer{index}.png"
        Image.fromarray(rgba).save(root / name)
        paths.append(name)
    spec = {"schema_version": 1, "profile": "layer_decompose",
        "targets": [{"path": path, "role": "auxiliary_canvas" if index == 0 else f"layer_{index}",
                     "weight": 0 if index == 0 else 1} for index, path in enumerate(paths)],
        "references": ["canvas.png"]}
    sidecar_path(primary).write_text(json.dumps(spec))
    return primary, spec


def test_paired_crop_flip_preserves_alpha_order_and_zero_auxiliary_weight(tmp_path):
    primary, _ = layer_files(tmp_path)
    record = inspect_item(primary, root=tmp_path, profile="layer_decompose")
    item = load_pixels(record, caption="word5", width=32, height=16,
                       crop_box=(16, 8, 32, 16), flip_horizontal=True)
    expected = ImageOps.mirror(Image.open(primary).crop((16, 8, 48, 24)))
    pixels = (item.pixels[:, 0] + 1) * 127.5
    torch.testing.assert_close(pixels, torch.from_numpy(np.asarray(expected).copy()).permute(2, 0, 1).float())
    assert np.array_equal(np.asarray(item.references[0]), np.asarray(expected))
    torch.testing.assert_close(item.pixels[:, 0], item.pixels[:, 1], rtol=0, atol=0)
    assert item.weights.tolist() == [0, 1]
    assert record.roles == ("auxiliary_canvas", "layer_1")
    assert record.bucket_key(32, 16) == ("layer_decompose", 32, 16, 2)


def test_design_rgb_is_promoted_to_opaque_rgba(tmp_path):
    primary = tmp_path / "plain.png"
    Image.new("RGB", (32, 32), (30, 40, 50)).save(primary)
    record = inspect_item(primary, root=tmp_path, profile="generation_edit")
    item = load_pixels(record, caption="word5", width=32, height=32)
    assert item.pixels.shape == (4, 1, 32, 32)
    assert torch.all(item.pixels[3] == 1) and not item.references


@pytest.mark.parametrize("problem", ["missing_sidecar", "roles", "alpha", "geometry", "escape", "missing_asset",
                                      "negative", "bool", "nan", "overflow", "sum_overflow", "underflow", "zero"])
def test_bad_layer_record_rejected_before_components_load(tmp_path, problem):
    root = tmp_path / "data"
    primary, spec = layer_files(root)
    if problem == "missing_sidecar": sidecar_path(primary).unlink()
    elif problem == "roles": spec["targets"][1]["role"] = "auxiliary_canvas"
    elif problem == "alpha": Image.new("RGB", (64, 32)).save(root / "layer1.png")
    elif problem == "geometry": Image.new("RGBA", (32, 32)).save(root / "layer1.png")
    elif problem == "escape":
        Image.new("RGBA", (64, 32)).save(tmp_path / "outside.png")
        spec["references"] = ["../outside.png"]
    elif problem == "missing_asset": spec["references"] = ["missing.png"]
    else:
        spec["targets"][1]["weight"] = {"negative": -1, "bool": True, "nan": float("nan"),
            "overflow": 1e40, "sum_overflow": 3e38, "underflow": 1e-50, "zero": 0}[problem]
        if problem == "sum_overflow": spec["targets"][0]["weight"] = 3e38
    if problem != "missing_sidecar": sidecar_path(primary).write_text(json.dumps(spec))
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_item(primary, root=root, profile="layer_decompose")


@pytest.mark.parametrize("asset", ["image", "manifest"])
def test_changed_asset_invalidates_preflight(tmp_path, asset):
    primary, spec = layer_files(tmp_path)
    record = inspect_item(primary, root=tmp_path, profile="layer_decompose")
    if asset == "image": Image.new("RGBA", (64, 32), (1, 2, 3, 4)).save(primary)
    else:
        spec["targets"][1]["weight"] = 2
        sidecar_path(primary).write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="changed after preflight"):
        load_pixels(record, caption="word5", width=32, height=32)


def test_duplicate_manifest_keys_refused(tmp_path):
    primary, _ = layer_files(tmp_path)
    sidecar_path(primary).write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate keys"):
        inspect_item(primary, root=tmp_path, profile="layer_decompose")


def test_pixel_cache_changes_for_pairing_and_supervision(tmp_path):
    primary, spec = layer_files(tmp_path)
    record = inspect_item(primary, root=tmp_path, profile="layer_decompose")
    args = dict(width=32, height=32, crop_box=(0, 0, 32, 32), flip_horizontal=False)
    key = pixel_cache_identity(record, **args)
    assert key != pixel_cache_identity(record, **{**args, "flip_horizontal": True})
    assert key != pixel_cache_identity(record, **{**args, "crop_box": (16, 0, 32, 32)})
    spec["targets"][1]["weight"] = 2
    sidecar_path(primary).write_text(json.dumps(spec))
    changed = inspect_item(primary, root=tmp_path, profile="layer_decompose")
    assert key != pixel_cache_identity(changed, **args)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_records_through_native_vae_conditioner_registered_handler_and_optimizer(tmp_path, profile):
    value = components(profile)
    items = []
    for index in range(2):
        root = tmp_path / str(index)
        primary, spec = layer_files(root)
        if profile == "generation_edit": sidecar_path(primary).unlink()
        elif index == 1:
            spec["references"].append("layer1.png")
            sidecar_path(primary).write_text(json.dumps(spec))
        record = inspect_item(primary, root=root, profile=profile)
        items.append(load_pixels(record, caption=f"word{index+5}", width=32, height=32))
    batch = prepare_record_batch(value, items, resolution=32)
    assert batch.clean.shape == (2, 16, 1 if profile == "generation_edit" else 2, 4, 4)
    if profile == "layer_decompose": assert [r.shape[1] for r in batch.reference_latents] == [1, 2]
    owner = trainer(value)
    owner.is_ming_image_01 = True
    optim = torch.optim.AdamW(value.transformer.parameters(), lr=.001)
    parameter = value.transformer.layers[0].attention.to_q.weight
    before = parameter.detach().clone()
    loss, *_ = get_arch_handler(owner).train_step(owner,
        TrainStepContext(latents=batch.clean, ming_batch=batch, timesteps=torch.tensor([.2, .8])))
    loss.backward()
    assert torch.isfinite(loss) and parameter.grad.abs().sum() > 0
    optim.step()
    assert not torch.equal(before, parameter) and not torch.cuda.is_initialized()


def test_frame_counts_are_distinct_buckets_and_cannot_batch(tmp_path):
    items = []
    for count in (2, 3):
        root = tmp_path / str(count)
        primary, _ = layer_files(root, count)
        record = inspect_item(primary, root=root, profile="layer_decompose")
        items.append(load_pixels(record, caption="word5", width=32, height=32))
    assert items[0].record.bucket_key(32, 32) != items[1].record.bucket_key(32, 32)
    with pytest.raises(ValueError, match="target-frame counts"):
        prepare_record_batch(components("layer_decompose"), items, resolution=32)


def test_shared_encode_image_retains_native_alpha(tmp_path):
    from core.training.base_trainer import BaseTrainer
    value = components("generation_edit")
    owner = trainer(value)
    owner.is_ming_image_01 = True
    owner.is_minit2i = owner.is_sensenova = False
    owner.vae, owner.vae_dtype = value.vae, torch.float32
    owner.arch = get_arch_handler(owner)
    low = BaseTrainer.encode_image(owner, Image.new("RGBA", (32, 32), (30, 40, 50, 0)),
                                   target_width=32, target_height=32, bucket_strategy="resize")
    high = BaseTrainer.encode_image(owner, Image.new("RGBA", (32, 32), (30, 40, 50, 255)),
                                    target_width=32, target_height=32, bucket_strategy="resize")
    assert low.shape == high.shape == (1, 16, 1, 4, 4) and not torch.equal(low, high)
    assert not torch.cuda.is_initialized()



def shared_dataset(root, count=2):
    from types import SimpleNamespace
    primary, spec = layer_files(root, count)
    paths = [primary, *[root / target["path"] for target in spec["targets"][1:]]]
    return SimpleNamespace(unique_id=str(root), dataset_root=root,
        items=[dict(image_path=str(path), caption="word5", width=64, height=32) for path in paths])


def test_common_discovery_excludes_layer_assets_and_is_transactional(tmp_path):
    from core.models.ming_image_01.dataset import annotate_datasets
    good = shared_dataset(tmp_path / "good", 3)
    bad = shared_dataset(tmp_path / "bad", 2)
    Image.new("RGBA", (64, 32)).save(bad.dataset_root / "loose.png")
    bad.items.append(dict(image_path=str(bad.dataset_root / "loose.png")))
    with pytest.raises(ValueError, match="primary images require"):
        annotate_datasets([good, bad], profile="layer_decompose")
    assert len(good.items) == 3 and all("_ming_record" not in item for item in good.items)
    assert annotate_datasets([good], profile="layer_decompose") == 1
    assert len(good.items) == 1 and good.items[0]["_ming_batch_signature"] == ("layer_decompose", 3)
    assert good.items[0]["reference_images"] == [str(good.dataset_root / "canvas.png")]


def test_common_epoch_reload_reestablishes_records_and_bucket_metadata(tmp_path):
    from types import MethodType, SimpleNamespace
    from core.training.base_trainer import BaseTrainer
    from core.training.bucketing import BucketManager
    from bucket_caption_epoch_sync_test import _Trainer
    dataset = shared_dataset(tmp_path)
    original = [dict(item) for item in dataset.items]
    dataset.reload_for_epoch = lambda epoch_num, run_id: [dict(item, caption=f"word{epoch_num+5}") for item in original]
    owner = _Trainer(pixel_align=16)
    owner._check_stop_requested = lambda: None
    owner.is_ming_image_01 = True
    owner.ming_image_01_components = SimpleNamespace(artifact=SimpleNamespace(profile="layer_decompose"))
    owner._annotate_ming_items = MethodType(BaseTrainer._annotate_ming_items, owner)
    owner._annotate_ming_items([dataset])
    manager = BucketManager(base_resolutions=[32], divisibility=16, strategy="resize")
    item = dataset.items[0]
    _, info = manager.assign_image_to_bucket(image_path=item["image_path"], width=64, height=32,
        dataset_unique_id=dataset.unique_id, caption=item["caption"],
        item_metadata=BaseTrainer._sensenova_bucket_metadata(item))
    before = info["_ming_record"]
    primary = Path(item["image_path"])
    spec = json.loads(sidecar_path(primary).read_text())
    spec["references"].append("layer1.png")
    spec["targets"][1]["weight"] = 2
    sidecar_path(primary).write_text(json.dumps(spec))
    pairs = owner._prepare_epoch_items([dataset], 1, 7, manager, [32])
    assert len(pairs) == 1 and pairs[0][0]["caption"] == info["caption"] == "word6"
    assert info["_ming_record"] != before and info["_ming_record"].weights == (0, 2)
    assert len(info["reference_images"]) == 2 and info["has_reference"]
    assert info["_ming_batch_signature"] == ("layer_decompose", 2)


def test_common_batch_split_preserves_planned_boundaries_and_every_occurrence(tmp_path):
    from core.models.ming_image_01.dataset import annotate_datasets, homogeneous_batches
    a, b = shared_dataset(tmp_path / "a", 2), shared_dataset(tmp_path / "b", 3)
    annotate_datasets([a, b], profile="layer_decompose")
    pa, pb = (a.items[0], a), (b.items[0], b)
    batches = homogeneous_batches([[pa, pb, pa], [pa]])
    assert batches == [[pa, pa], [pb], [pa]]
    assert sum(map(len, batches)) == 4
    assert homogeneous_batches(batches) == batches


def test_concept_plan_homogeneous_frames_and_reproducible_resume_identity(tmp_path):
    from core.models.ming_image_01.dataset import annotate_datasets, homogeneous_batches
    from core.training.concept_batch_order import ConceptOrderConfig, build_concept_batch_plan
    datasets = [shared_dataset(tmp_path / str(i), 2+i%2) for i in range(6)]
    annotate_datasets(datasets, profile="layer_decompose")
    pairs = [(dataset.items[0], dataset) for dataset in datasets]
    for item, _ in pairs:
        item["tag_data"] = [{"tag": "character_a", "category": "Character"}]
        item["is_tags_format"] = True
    config = ConceptOrderConfig.parse({"enabled": True, "min_items_per_concept": 2})
    first = build_concept_batch_plan(pairs, 4, config, 0, 42)
    rebuilt = build_concept_batch_plan(list(reversed(pairs)), 4, config, 0, 42)
    assert first.digest == rebuilt.digest and first.batches == rebuilt.batches
    assert homogeneous_batches(first.batches) == first.batches
    assert sum(map(len, first.batches)) == len(pairs)
    assert all(len({item["_ming_batch_signature"] for item, _ in batch}) == 1 for batch in first.batches)
    pairs[0][0]["_ming_batch_signature"] = ("layer_decompose", 4)
    changed = build_concept_batch_plan(pairs, 4, config, 0, 42)
    assert first.digest != changed.digest



def test_supervision_changes_invalidate_shared_resume_fingerprint_and_ledgers(tmp_path):
    from types import SimpleNamespace
    from core.models.ming_image_01.dataset import annotate_datasets
    from core.training.base_trainer import BaseTrainer
    from core.training.resume_batch_plan import make_plan, resolve_suffix
    from core.training.concept_batch_order import ConceptOrderConfig, build_concept_batch_plan
    dataset = shared_dataset(tmp_path)
    annotate_datasets([dataset], profile="layer_decompose")
    owner = SimpleNamespace(config={}, log_prefix="[Ming dataset]")
    first = BaseTrainer._compute_dataset_fingerprint(owner, [dataset])
    batch = [(dataset.items[0], dataset)]
    ledger = make_plan([batch], [dataset], "normal", "test", 0)
    config = ConceptOrderConfig.parse({"enabled": True, "min_items_per_concept": 2})
    concept = build_concept_batch_plan(batch, 2, config, 0, 42)
    primary = Path(dataset.items[0]["image_path"])
    spec = json.loads(sidecar_path(primary).read_text())
    spec["targets"][1]["weight"] = 2
    sidecar_path(primary).write_text(json.dumps(spec))
    annotate_datasets([dataset], profile="layer_decompose")
    changed = BaseTrainer._compute_dataset_fingerprint(owner, [dataset])
    assert first["image_paths_hash"] == changed["image_paths_hash"]
    assert BaseTrainer._check_dataset_fingerprint_changed(owner, first, changed)
    revised = make_plan([batch], [dataset], "normal", "test", 0)
    assert ledger["batches"] != revised["batches"] and ledger["history_manifest"] != revised["history_manifest"]
    with pytest.raises(ValueError, match="item conditions changed"):
        resolve_suffix(ledger, 0, [batch])
    assert resolve_suffix(revised, 0, [batch]) == [batch]
    assert concept.digest != build_concept_batch_plan(batch, 2, config, 0, 42).digest
    assert not BaseTrainer._check_dataset_fingerprint_changed(owner, changed, changed)


def test_random_shared_crop_is_replayable_without_consuming_global_rng(tmp_path):
    import random
    from types import SimpleNamespace
    from core.models.ming_image_01.dataset import load_shared_items, batch_crop_seed
    primary, _ = layer_files(tmp_path)
    record = inspect_item(primary, root=tmp_path, profile="layer_decompose")
    owner = SimpleNamespace(bucket_strategy="random_crop", run_seed=8765, _current_epoch=3, _current_batch_position=7)
    pair = ({"_ming_record": record, "width": 32, "height": 16, "caption": "word5"}, None)
    before = random.getstate()
    first = load_shared_items(owner, [pair])[0]
    assert random.getstate() == before
    random.random()
    later_state = random.getstate()
    replay = load_shared_items(owner, [pair], epoch=3, batch_position=7)[0]
    assert random.getstate() == later_state
    assert first.pixel_identity == replay.pixel_identity
    torch.testing.assert_close(first.pixels, replay.pixels, rtol=0, atol=0)
    assert np.array_equal(np.asarray(first.references[0]), np.asarray(replay.references[0]))
    seed = batch_crop_seed(owner, record, item_index=0, width=32, height=16, epoch=3, batch_position=7)
    for coordinate in ({"item_index": 1}, {"epoch": 4}, {"batch_position": 8}, {"width": 16}):
        args = dict(item_index=0, width=32, height=16, epoch=3, batch_position=7)
        args.update(coordinate)
        assert seed != batch_crop_seed(owner, record, **args)
    random.setstate(before)


def test_shared_crop_spec_is_authoritative_and_missing_seed_refuses(tmp_path):
    from types import SimpleNamespace
    from core.models.ming_image_01.dataset import load_shared_items
    primary, _ = layer_files(tmp_path)
    record = inspect_item(primary, root=tmp_path, profile="layer_decompose")
    owner = SimpleNamespace(bucket_strategy="random_crop")
    item = {"_ming_record": record, "width": 32, "height": 16, "caption": "word5"}
    with pytest.raises(ValueError, match="resolved run seed"):
        load_shared_items(owner, [(item, None)])
    item["_crop_spec"] = SimpleNamespace(crop_box=(16, 8, 32, 16))
    loaded = load_shared_items(owner, [(item, None)])[0]
    expected = load_pixels(record, caption="word5", width=32, height=16, crop_box=(16, 8, 32, 16))
    assert loaded.pixel_identity == expected.pixel_identity
    torch.testing.assert_close(loaded.pixels, expected.pixels, rtol=0, atol=0)

@pytest.mark.parametrize("item_type", ["single", "image", "video", "audio"])
def test_shared_item_type_contract(tmp_path, item_type):
    from types import SimpleNamespace
    from core.models.ming_image_01.dataset import annotate_datasets
    primary = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(primary)
    item = {"image_path": str(primary), "item_type": item_type}
    dataset = SimpleNamespace(dataset_root=tmp_path, items=[item])
    if item_type in {"single", "image"}:
        assert annotate_datasets([dataset], profile="generation_edit") == 1
        assert item["_ming_record"].profile == "generation_edit"
    else:
        with pytest.raises(ValueError, match="images, not video/audio"):
            annotate_datasets([dataset], profile="generation_edit")
        assert "_ming_record" not in item

@pytest.mark.parametrize("cancel", [False, True])
def test_annotation_progress_and_cancel_are_atomic(tmp_path, cancel):
    from types import SimpleNamespace
    from core.models.ming_image_01.dataset import annotate_datasets
    primary = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(primary)
    items = [{"image_path": str(primary), "item_type": "single"} for _ in range(2)]
    dataset = SimpleNamespace(dataset_root=tmp_path, items=items)
    reports, checks = [], []
    def stop_check():
        checks.append(1)
        if cancel and len(checks) == 2:
            raise KeyboardInterrupt("stop")
    def progress(**event):
        reports.append(event)
    if cancel:
        with pytest.raises(KeyboardInterrupt):
            annotate_datasets([dataset], profile="generation_edit", progress_callback=progress, stop_check=stop_check)
        assert all("_ming_record" not in item for item in items)
        assert [r["step"] for r in reports] == [0]
    else:
        annotate_datasets([dataset], profile="generation_edit", progress_callback=progress, stop_check=stop_check)
        assert [r["step"] for r in reports] == [0, 2]
        assert reports[-1]["detail"] == "Validating Ming images: 2/2"
    assert len(checks) == 2
