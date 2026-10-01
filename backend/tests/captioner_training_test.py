"""CPU dataset, tokenizer, artifact, cache, and exact-resume integration."""

from pathlib import Path
import sys
import threading
import io
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.captioner.artifact import export_bundle, load_bundle, save_bundle, tagger_identity
from core.captioner.concepts import ConceptSelector
from core.captioner.dataset import build_snapshot, split_records, verify_snapshot
from core.captioner.features import CaptionFeatures
from core.captioner.model import CaptionConfig, CompactCaptioner
from core.captioner.storage import FeatureCache, read_json
from core.captioner.tokenizer import CaptionTokenizer
from core.captioner.trainer import CaptionTrainer
from core.captioner.inference import CaptionInferenceManager
from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager
from core.gpu_coordinator import GPUCoordinator
from core.tagger.siglip2_tagger_model import SigLIP2TaggerFeatures
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_initialized()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    assert not torch.cuda.is_initialized()


class FakeTagger(nn.Module):
    is_naflex = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward_features(self, pixels, mask, shapes):
        pooled = pixels.mean(dim=1)
        return SigLIP2TaggerFeatures(pixels, pooled, pooled[:, :2])


class FakeProcessor:
    @property
    def image_processor(self):
        return self

    def to_dict(self):
        return {"type": "synthetic-cpu", "patch_size": 1}

    def __call__(self, images, return_tensors, max_num_patches):
        mean = torch.tensor(list(images[0].resize((1, 1)).getdata())[0], dtype=torch.float32) / 255
        return {"pixel_values": mean[None, None, :].expand(1, 4, 3).clone(),
                "pixel_attention_mask": torch.ones(1, 4, dtype=torch.long),
                "spatial_shapes": torch.tensor([[2, 2]])}


def manager():
    return SimpleNamespace(model=FakeTagger().eval(), model_type="full", is_naflex=True,
        processor=FakeProcessor(), vocabulary={"idx_to_tag": {0: "blue hair", 1: "red hair"},
        "tag_to_category": {"blue hair": "General", "red hair": "General"}})


def database(tmp_path, count=8):
    engine = create_engine(f"sqlite:///{tmp_path / 'dataset.db'}")
    DatasetBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    root = tmp_path / "images"
    root.mkdir()
    with factory() as db:
        dataset = Dataset(name="synthetic", path=str(root))
        db.add(dataset)
        db.flush()
        dataset_id = dataset.id
        for index in range(count):
            path = root / f"{index}.png"
            Image.new("RGB", (8, 8), (index * 25, 30, 60)).save(path)
            item = DatasetItem(dataset_id=dataset_id, base_name=str(index), image_path=str(path))
            db.add(item)
            db.flush()
            db.add_all([DatasetCaption(item_id=item.id, caption_type="tags", is_tags_format=True,
                                      content="blue hair"),
                        DatasetCaption(item_id=item.id, caption_type="natural_language", is_tags_format=False,
                                      content="A girl with blue hair.")])
        db.commit()
    return factory, [{"dataset_id": dataset_id, "input_tag_caption_types": ["tags"],
                      "target_caption_types": ["natural_language"], "filters": {}}]


def test_tokenizer_unicode_unknown_bytes_and_no_truncation():
    tokenizer = CaptionTokenizer.train(["A girl."] * 4, vocab_size=280, min_frequency=2)
    text = "青い髪の少女。 🐟"
    tokens = tokenizer.encode_target(text, max_length=100)
    assert tokenizer.unk_id not in tokens
    assert tokenizer.decode(tokens) == text
    copied = CaptionTokenizer.from_json(tokenizer.to_json())
    assert copied.encode_target(text, max_length=100) == tokens
    with pytest.raises(ValueError):
        tokenizer.encode_target(text, max_length=2)
    with pytest.raises(ValueError):
        tokenizer.encode_target("[BOS]", max_length=20)


def test_duplicate_groups_include_transitive_near_duplicates_and_source_groups():
    records = [{"image_hash": str(i), "pixel_hash": str(i), "difference_hash": value,
                "source_group": None} for i, value in enumerate([0, 1, 3, 1023, 2**32 - 1, 2**64 - 1])]
    records[4]["source_group"] = records[5]["source_group"] = "same-work"
    split = split_records(records, seed=42, val_ratio=0.1, test_ratio=0.1, near_duplicate_distance=1)
    assert len({r["split"] for r in split[:3]}) == 1
    assert split[4]["split"] == split[5]["split"]
    assert {r["split"] for r in split} == {"train", "val", "test"}
    again = split_records(records, seed=42, val_ratio=0.1, test_ratio=0.1, near_duplicate_distance=1)
    assert split == again


def test_snapshot_caption_join_filters_and_drift(tmp_path):
    factory, configs = database(tmp_path)
    with factory() as db:
        snapshot = build_snapshot(db, configs, seed=7, val_ratio=0.1, test_ratio=0.1,
                                  near_duplicate_distance=0, cancelled=lambda: False)
    assert len(snapshot["records"]) == 8
    assert snapshot["records"][0]["tags"] == ["blue hair"]
    verify_snapshot(snapshot)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(snapshot["records"][0]["path"])
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(snapshot)


def test_artifact_roundtrip_rejects_incompatible_and_corrupt_files(tmp_path):
    tagger = manager()
    identity = tagger_identity(tagger, max_num_patches=4)
    tokenizer = CaptionTokenizer.train(["A girl."], vocab_size=280, min_frequency=1)
    model = CompactCaptioner(CaptionConfig(3, 3, 2, tokenizer.vocab_size, 12, 3, 1, 24, 2, 32, 0.0,
                                          9, 0, 1, 2))
    selector = ConceptSelector(identity["names"], identity["categories"], max_concepts=2,
                               allowed_categories=["General"])
    root = tmp_path / "bundle"
    save_bundle(root, model=model, tokenizer=tokenizer, selector=selector, identity=identity,
                training_state={"step": 1}, snapshot_hash="synthetic", aliases={}, language="en")
    restored, _, _, _, state = load_bundle(root, expected_identity=identity, require_resume=True)
    assert state == {"step": 1}
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor)
    with pytest.raises(ValueError, match="do not match"):
        load_bundle(root, expected_identity={**identity, "max_num_patches": 8})
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(root)


def test_cache_capacity_and_dependency_invalidation(tmp_path):
    feature = CaptionFeatures(torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool),
                              torch.zeros(1, 2, 2), torch.ones(1, 3), torch.ones(1, 2))
    image_hash = "a" * 64
    cache = FeatureCache(tmp_path, identity={"weights": "a"}, max_bytes=100000)
    cache.put(image_hash, feature)
    torch.testing.assert_close(cache.get(image_hash).spatial_tokens, feature.spatial_tokens)
    other = FeatureCache(tmp_path, identity={"weights": "b"}, max_bytes=100000)
    assert other.get(image_hash) is None
    full = FeatureCache(tmp_path, identity={"weights": "c"}, max_bytes=1)
    full.put(image_hash, feature)
    assert full.get(image_hash) is None


def test_stopped_run_resumes_exactly_on_cpu(tmp_path):
    factory, configs = database(tmp_path)
    config = {"epochs": 2, "batch_size": 2, "hidden_dim": 12, "heads": 3, "layers": 1,
              "ffn_dim": 24, "visual_queries": 2, "max_length": 40, "vocab_size": 280,
              "val_ratio": 0.0, "test_ratio": 0.0, "near_duplicate_distance": 0,
              "save_every_steps": 100, "max_num_patches": 4, "max_concepts": 2, "dropout": 0.1}
    full_root, resumed_root = tmp_path / "full", tmp_path / "resumed"
    def run(root, stop, progress, resume=None):
        return CaptionTrainer(run_id="synthetic", output_dir=root, config=config, dataset_configs=configs,
            tagger_manager=manager(), session_factory=factory, progress=progress, stop_event=stop,
            alias_table={}, resume_path=resume).run()
    run(full_root, threading.Event(), lambda event: None)
    stopped = threading.Event()
    def progress(event):
        if event.get("phase") == "training" and event["step"] == 1:
            stopped.set()
    partial = run(resumed_root, stopped, progress)
    assert partial["stopped"] and partial["step"] == 1
    path = resumed_root / read_json(resumed_root / "latest.json")["checkpoint"]
    complete = run(resumed_root, threading.Event(), lambda event: None, path)
    assert complete["step"] == 8 and not complete["stopped"]
    full_path = full_root / read_json(full_root / "latest.json")["checkpoint"]
    resumed_path = resumed_root / read_json(resumed_root / "latest.json")["checkpoint"]
    first = load_bundle(full_path)[0]
    second = load_bundle(resumed_path)[0]
    for key, tensor in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[key], tensor, atol=0, rtol=0)


def test_trainer_allocation_cannot_overlap_generation():
    coordinator = GPUCoordinator()
    coordinator._active_generations = 1
    with pytest.raises(RuntimeError, match="serving generation"):
        with coordinator.trainer_allocation():
            pytest.fail("allocation admitted while generation is active")


def test_inference_borrows_encoder_and_export_has_no_optimizer(tmp_path):
    tagger = SigLIP2InferenceManager()
    for key, value in vars(manager()).items():
        setattr(tagger, key, value)
    tagger.device = "cpu"
    identity = tagger_identity(tagger, max_num_patches=4)
    tokenizer = CaptionTokenizer.train(["A girl."], vocab_size=280, min_frequency=1)
    model = CompactCaptioner(CaptionConfig(3, 3, 2, tokenizer.vocab_size, 12, 3, 1, 24, 2, 32, 0.0,
                                          9, 0, 1, 2))
    selector = ConceptSelector(identity["names"], identity["categories"], max_concepts=2,
                               allowed_categories=["General"])
    root, exported = tmp_path / "checkpoint", tmp_path / "export"
    save_bundle(root, model=model, tokenizer=tokenizer, selector=selector, identity=identity,
                training_state={"step": 1}, snapshot_hash="synthetic", aliases={}, language="en")
    export_bundle(root, exported)
    assert not (exported / "resume.pt").exists()
    with pytest.raises(ValueError, match="Incomplete"):
        load_bundle(exported, require_resume=True)
    inference = CaptionInferenceManager(tagger)
    assert inference.load(exported)["loaded"]
    image = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(image, format="PNG")
    encoder = tagger.model
    output = inference.predict(image.getvalue(), tag_source="merged", tags=["blue hair", "unknown"],
                               max_new_tokens=3, device="cpu", cancelled=lambda: False)
    assert tagger.model is encoder and not encoder.training
    assert output["used_tags"][0]["name"] == "blue hair"
    assert output["used_tags"][0]["score"] is None
    assert output["warnings"] == ["unknown_tag:unknown"]
    assert 1 <= output["token_count"] <= 3
    assert output["timings_ms"]["total"] > 0
    with pytest.raises(InterruptedError):
        inference.predict(image.getvalue(), tag_source="predicted", tags=[], max_new_tokens=3,
                          device="cpu", cancelled=lambda: True)
    tagger.model = FakeTagger().eval()
    tagger.model.weight.data.fill_(2)
    with pytest.raises(ValueError, match="no longer matches"):
        inference.predict(image.getvalue(), tag_source="predicted", tags=[], max_new_tokens=3,
                          device="cpu", cancelled=lambda: False)


def test_borrow_blocks_unload_until_request_finishes():
    tagger = SigLIP2InferenceManager()
    for key, value in vars(manager()).items():
        setattr(tagger, key, value)
    requested, finished = threading.Event(), threading.Event()
    def unload():
        requested.set()
        tagger.unload()
        finished.set()
    with tagger.borrow_caption_tagger():
        thread = threading.Thread(target=unload)
        thread.start()
        assert requested.wait(1)
        assert not finished.wait(0.05)
        assert tagger.model is not None
    thread.join(2)
    assert finished.is_set() and tagger.model is None
