"""Checkpoint-bound collection state without any network or image reads."""

import json
import threading
from types import SimpleNamespace

import pytest

from core.tagger.danbooru_sampler import DanbooruSampleBuffer
from core.tagger.danbooru_vocab_expander import VocabExpander
from core.tagger.foveatag.collection_runtime import save_collection_runtime, load_collection_runtime
from core.tagger.foveatag.shared_bundle import commit_native_bundle, verify_native_bundle
from foveatag_shared_bundle_test import bundle


def buffer():
    value = DanbooruSampleBuffer.__new__(DanbooruSampleBuffer)
    value._cycle_lock = threading.RLock()
    value._dynamic_tags = ["anonymous_a"]
    value._dynamic_last_used = {"anonymous_a": 11.}
    value._cooc_active_tags = ["anonymous_b"]
    value._cooc_last_used = {"anonymous_b": 12.}
    value._query_tags = ["anonymous_c"]
    value._query_last_used = {"anonymous_c": 13.}
    value._collect_count = {"anonymous_a": 7}
    value._cooc_collected = {"anonymous_b": 8}
    value._exhausted_tags = {"anonymous_a"}
    value._expanded_via_query = {"anonymous_c"}
    value._last_query_resolve = 14.
    value._low_f1_unavailable = {"anonymous_d"}
    value._native_expanded_tags = {"anonymous_e"}
    value._expander = VocabExpander()
    value._expander.propose({"anonymous_f"})
    value._pending_epoch_progress = None
    value._downloaded_ids = {9}
    value._cycle_gen = 0
    value._vocabulary = SimpleNamespace(tag_to_idx={name.replace("_", " "): index for index, name in enumerate(
        ("anonymous_a", "anonymous_b", "anonymous_c"))})
    return value


def test_native_online_cap_rejects_before_decode(monkeypatch):
    from io import BytesIO
    from PIL import Image
    from core.tagger.foveatag.processor import FoveaProcessor
    from core.tagger.foveatag.views import ViewConfig

    encoded = BytesIO()
    Image.new("RGB", (32, 32)).save(encoded, format="PNG")
    value = buffer()
    value._client = SimpleNamespace(download_inmemory=lambda post: (encoded.getvalue(), "png", []))
    value._quality_tag_enable = False
    value._processor = FoveaProcessor(ViewConfig(), seed=1, training=True, max_source_pixels=512)
    original_open = Image.open

    def header_only(*args, **kwargs):
        image = original_open(*args, **kwargs)
        image.load = lambda: pytest.fail("oversized native image decoded")
        return image

    monkeypatch.setattr(Image, "open", header_only)
    assert value._process_post({"id": 1}) is None


def test_native_processor_cap_is_enforced():
    from PIL import Image
    from core.tagger.foveatag.processor import FoveaProcessor
    from core.tagger.foveatag.views import ViewConfig
    with pytest.raises(ValueError, match="positive"):
        FoveaProcessor(ViewConfig(), seed=1, training=True, max_source_pixels=0)
    processor = FoveaProcessor(ViewConfig(), seed=1, training=False, max_source_pixels=512)
    with pytest.raises(ValueError, match="pixel cap"):
        processor(images=[Image.new("RGB", (32, 32))], return_tensors="pt")


def test_snapshot_restores_pending_proposals_run_caps_and_same_epoch_quotas(tmp_path):
    source = buffer()
    path = tmp_path / "step_000008_collection_runtime.json"
    save_collection_runtime(source, path, step=8, epoch=2)
    saved = load_collection_runtime(path, step=8)
    assert saved["dynamic_tags"] == source.snapshot_dynamic_tags()
    target = buffer()
    target._expanded_via_query.clear()
    target._expander = VocabExpander()
    target._collect_count.clear()
    target._cooc_collected.clear()
    target.restore_native_runtime(saved)
    target.reset_download_cycle(epoch=2)
    assert target._collect_count == source._collect_count
    assert target._cooc_collected == source._cooc_collected
    assert target._expanded_via_query == source._expanded_via_query
    assert target._expander.consume_pending() == ["anonymous_f"]
    target._expander.propose({"anonymous_f"})
    assert not target._expander.has_pending()
    target.reset_download_cycle(epoch=3)
    assert not target._collect_count and not target._cooc_collected


def test_invalid_collection_runtime_refuses_before_restore(tmp_path):
    path = tmp_path / "latest_collection_runtime.json"
    save_collection_runtime(buffer(), path, step=8, epoch=2)
    with pytest.raises(ValueError, match="missing or invalid"):
        load_collection_runtime(path, step=9)
    document = json.loads(path.read_text())
    document["expander"]["already_proposed"] = []
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="missing or invalid"):
        load_collection_runtime(path, step=8)


def test_collection_runtime_required_and_integrity_bound(tmp_path):
    bundle(tmp_path, "latest", 8)
    state_path = tmp_path / "latest_state.json"
    state = json.loads(state_path.read_text())
    state["native_runtime"]["collection_enabled"] = True
    state["epoch"] = 2
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="lacks selected collection"):
        commit_native_bundle(tmp_path, "latest", optional_suffixes=[])
    path = tmp_path / "latest_collection_runtime.json"
    save_collection_runtime(buffer(), path, step=8, epoch=2)
    commit_native_bundle(tmp_path, "latest", optional_suffixes=["_collection_runtime.json"])
    assert path.name in verify_native_bundle(tmp_path, "latest")["files"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="integrity differs"):
        verify_native_bundle(tmp_path, "latest")


def test_actual_native_mixed_loop_commits_collection_runtime(tmp_path):
    from dataclasses import asdict
    from torch.utils.data import DataLoader
    from backend.api.param_defaults import TAGGER_TRAINING_DEFAULTS
    from core.gpu_coordinator import gpu_coordinator
    from core.tagger.danbooru_sampler import MixedDataLoader
    from core.tagger.foveatag.encoder import FoveaEncoder, FoveaEncoderConfig
    from core.tagger.foveatag.pretraining_checkpoint import save_pretrained_encoder
    from core.tagger.tagger_dataset import tagger_collate_fn
    from core.tagger.tagger_trainer import TaggerTrainer, _load_training_state
    from core.tagger.tag_vocabulary import TagVocabulary
    from foveatag_shared_replay_test import SyntheticNativeDataset
    encoder = FoveaEncoder(FoveaEncoderConfig(widths=(8, 16, 32, 32), depths=(1, 1, 1, 1),
        attention_heads=(2, 2), scale_dropout=0))
    path = save_pretrained_encoder(tmp_path / "encoder.safetensors", encoder,
        run_contract={"encoder": asdict(encoder.config)}, epoch=1, global_step=1, validation={})
    config = dict(TAGGER_TRAINING_DEFAULTS)
    config.update(architecture="foveatag", vision_encoder_path=path, training_method="full",
        fovea_learned_routing=False,
        fovea_feature_dim=16, fovea_overview_size=32, fovea_local_size=32, fovea_local_views=2,
        epochs=1, batch_size=2, num_workers=0, optimizer="adamw", mixed_precision="fp32",
        gradient_checkpointing=False, save_tag_metrics=False, save_ood_reference=False,
        train_f1_eval_every_n_steps=0, save_every_n_steps=1, keep_last_n_checkpoints=0,
        danbooru_injection_interval=1)
    vocabulary = TagVocabulary()
    vocabulary.tag_to_idx = {f"anonymous_{i}": i for i in range(5)}
    vocabulary.idx_to_tag = {index: name for name, index in vocabulary.tag_to_idx.items()}
    vocabulary.tag_to_category = {name: "General" for name in vocabulary.tag_to_idx}
    source = SyntheticNativeDataset()
    collection = buffer()
    collection._native_expanded_tags = {"anonymous_4"}
    sample = source[0]
    queued = [(sample[0], sample[1], sample[2], ["anonymous_4"]) for _ in range(4)]
    def drain(count):
        if len(queued) < count:
            return None
        values = queued[:count]
        del queued[:count]
        return values
    collection.drain_batch = drain
    loader = MixedDataLoader(DataLoader(source, batch_size=2, collate_fn=tagger_collate_fn), collection,
        injection_interval=1, vocabulary=vocabulary)
    trainer = TaggerTrainer("synthetic", config, vocabulary, str(tmp_path / "run"))
    try:
        trainer.train(loader, None, source.processor)
    finally:
        gpu_coordinator.unregister_trainer(trainer._coordinator_handle)
        trainer._coordinator_handle.detach()
    document = verify_native_bundle(tmp_path / "run", "latest")
    assert "latest_collection_runtime.json" in document["files"]
    saved = load_collection_runtime(tmp_path / "run/latest_collection_runtime.json", step=document["global_step"], epoch=1)
    assert saved["query_expanded"] == ["anonymous_c"]
    assert saved["expanded_tags"] == ["anonymous_4"]
    state = json.loads((tmp_path / "run/latest_state.json").read_text())
    assert state["native_runtime"]["expanded_tags"] == ["anonymous_4"]
    assert trainer._optimizer_steps_completed == state["global_step"] + 2
    config = dict(config, epochs=2, enable_danbooru_augmentation=False)
    resumed = TaggerTrainer("synthetic", config, vocabulary, str(tmp_path / "run"))
    plain_loader = DataLoader(source, batch_size=2, collate_fn=tagger_collate_fn)
    try:
        resumed.train(plain_loader, None, source.processor,
            resume_state=_load_training_state(str(tmp_path / "run"), "latest"), resume_ckpt_name="latest")
    finally:
        gpu_coordinator.unregister_trainer(resumed._coordinator_handle)
        resumed._coordinator_handle.detach()
    document = verify_native_bundle(tmp_path / "run", "latest")
    state = json.loads((tmp_path / "run/latest_state.json").read_text())
    assert not state["native_runtime"]["collection_enabled"]
    assert state["native_runtime"]["expanded_tags"] == ["anonymous_4"]
    assert "latest_collection_runtime.json" not in document["files"]


def test_atomic_collection_save_failure_does_not_publish(tmp_path, monkeypatch):
    import core.tagger.foveatag.collection_runtime as runtime
    def unavailable(*args, **kwargs):
        raise OSError("synthetic unavailable storage")
    monkeypatch.setattr(runtime.os, "replace", unavailable)
    path = tmp_path / "latest_collection_runtime.json"
    with pytest.raises(OSError, match="unavailable storage"):
        save_collection_runtime(buffer(), path, step=8, epoch=2)
    assert not path.exists()


def test_collection_epoch_must_match_model_epoch(tmp_path):
    path = tmp_path / "latest_collection_runtime.json"
    save_collection_runtime(buffer(), path, step=8, epoch=2)
    with pytest.raises(ValueError, match="missing or invalid"):
        load_collection_runtime(path, step=8, epoch=3)


def test_resume_reconciles_pool_entry_before_worker_proposal(tmp_path):
    source = buffer()
    source._expander = VocabExpander()
    source._vocabulary.tag_to_idx.pop("anonymous c")
    path = tmp_path / "latest_collection_runtime.json"
    save_collection_runtime(source, path, step=8, epoch=2)
    target = buffer()
    target._vocabulary = source._vocabulary
    target.restore_native_runtime(load_collection_runtime(path, step=8))
    assert target._expander.consume_pending() == ["anonymous c"]


@pytest.mark.parametrize("method", ["full", "lora"])
def test_native_head_growth_preserves_existing_rows_and_optimizer(method):
    import torch
    from core.tagger.danbooru_vocab_expander import expand_vocab_and_head
    from core.tagger.tag_vocabulary import TagVocabulary
    from foveatag_shared_export_test import model, batch
    instance = model(method)
    optimizer = torch.optim.AdamW(instance.trainable_parameters(), lr=.001)
    instance(batch()).square().mean().backward()
    optimizer.step()
    vocabulary = TagVocabulary()
    vocabulary.tag_to_idx = {f"anonymous_{index}": index for index in range(5)}
    vocabulary.idx_to_tag = {index: name for name, index in vocabulary.tag_to_idx.items()}
    vocabulary.tag_to_category = {name: "General" for name in vocabulary.tag_to_idx}
    old_weight = instance.head.weight.detach().clone()
    old_momentum = optimizer.state[instance.head.weight]["exp_avg"].clone()
    assert expand_vocab_and_head(["anonymous_new"], vocabulary, instance, optimizer) == 1
    torch.testing.assert_close(instance.head.weight[:5], old_weight, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[instance.head.weight]["exp_avg"][:5], old_momentum, rtol=0, atol=0)
    assert torch.count_nonzero(optimizer.state[instance.head.weight]["exp_avg"][5]) == 0
    assert instance(batch()).shape == (1, 6)
    optimizer.zero_grad()
    instance(batch()).square().mean().backward()
    optimizer.step()
