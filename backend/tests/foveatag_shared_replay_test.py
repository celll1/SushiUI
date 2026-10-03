"""Native views, sampler positions and common-loop resume remain reproducible."""

from dataclasses import asdict
import json
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

from api.param_defaults import TAGGER_TRAINING_DEFAULTS
from core.tagger.foveatag.encoder import FoveaEncoder, FoveaEncoderConfig
from core.tagger.foveatag.pretraining_checkpoint import save_pretrained_encoder
from core.tagger.foveatag.processor import FoveaProcessor
from core.tagger.foveatag.shared_replay import NativeReplayCursor, native_epoch_loader, native_dataset_fingerprint
from core.tagger.foveatag.views import ViewConfig
from core.tagger.tagger_dataset import tagger_collate_fn


class SyntheticNativeDataset(Dataset):
    def __init__(self):
        self.processor = FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=2),
                                        seed=42, training=True)
        self._samples = [(f"anonymous-source-{index}", []) for index in range(10)]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        views = self.processor(images=[Image.new("RGB", (96, 48), (index * 20, 32, 64))],
                               return_tensors="pt", sample_index=index)["fovea_views"]
        label = torch.zeros(5)
        label[index % 5] = 1
        return views, torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.int64), label, torch.ones(5)


def test_indexed_views_ignore_global_rng_and_change_with_epoch():
    processor = SyntheticNativeDataset().processor
    image = Image.new("RGB", (512, 384))
    torch.manual_seed(1)
    first = processor(images=[image], return_tensors="pt", sample_index=7)["fovea_views"]
    state = torch.get_rng_state().clone()
    torch.manual_seed(900)
    second = processor(images=[image], return_tensors="pt", sample_index=7)["fovea_views"]
    torch.testing.assert_close(first.local_geometry, second.local_geometry, rtol=0, atol=0)
    torch.set_rng_state(state)
    processor(images=[image], return_tensors="pt", sample_index=7)
    assert torch.equal(state, torch.get_rng_state())
    processor.set_epoch(2)
    third = processor(images=[image], return_tensors="pt", sample_index=7)["fovea_views"]
    assert not torch.equal(first.local_geometry, third.local_geometry)


def test_partial_epoch_is_exact_suffix_without_global_rng_consumption():
    source = SyntheticNativeDataset()
    loader = DataLoader(source, batch_size=2, collate_fn=tagger_collate_fn)
    state = torch.get_rng_state().clone()
    complete = list(native_epoch_loader(loader, seed=42, epoch=3, base_batches=0))
    assert torch.equal(state, torch.get_rng_state())
    torch.manual_seed(810)
    resumed = list(native_epoch_loader(loader, seed=42, epoch=3, base_batches=2))
    assert len(resumed) == len(complete) - 2
    for expected, actual in zip(complete[2:], resumed):
        torch.testing.assert_close(expected[0].overview_images, actual[0].overview_images, rtol=0, atol=0)
        torch.testing.assert_close(expected[0].local_geometry, actual[0].local_geometry, rtol=0, atol=0)
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)


def test_fingerprint_covers_selected_source_order_not_labels():
    source = SyntheticNativeDataset()
    initial = native_dataset_fingerprint(Subset(source, [3, 1, 7]))
    source._samples[3] = (source._samples[3][0], ["anonymous"])
    assert native_dataset_fingerprint(Subset(source, [3, 1, 7])) == initial
    assert native_dataset_fingerprint(Subset(source, [1, 3, 7])) != initial
    assert initial["total_item_count"] == 3


def test_cursor_does_not_treat_injection_updates_as_source_batches():
    cursor = NativeReplayCursor()
    cursor.complete(0, is_injection=False, readable=True, injection_interval=1)
    assert cursor.pending_injection
    cursor.complete(1, is_injection=True, readable=True, injection_interval=1)
    assert cursor.base_batches == cursor.valid_base_batches == 1
    assert not cursor.pending_injection
    cursor.complete(2, is_injection=False, readable=False, injection_interval=1)
    assert cursor.base_batches == 2
    assert cursor.valid_base_batches == 1
    assert cursor.last_batch_idx == 2


@pytest.mark.parametrize("cut", [2, 3, 6])
def test_mixed_loader_resumes_pending_injection_and_original_cadence(cut):
    from types import SimpleNamespace
    from core.tagger.danbooru_sampler import MixedDataLoader
    source = SyntheticNativeDataset()
    loader = DataLoader(source, batch_size=2, collate_fn=tagger_collate_fn)
    buffer = SimpleNamespace(reset_download_cycle=lambda epoch: None)
    mixed = MixedDataLoader(native_epoch_loader(loader, seed=42, epoch=1, base_batches=0),
                             buffer=buffer, injection_interval=2)
    injection = tagger_collate_fn([source[0], source[1]])
    mixed._build_injection_batch = lambda: injection
    complete = list(mixed)
    cursor = NativeReplayCursor()
    for index, (payload, is_injection) in enumerate(complete[:cut]):
        cursor.complete(index, is_injection=is_injection, readable=payload is not None, injection_interval=2)
    resumed = mixed.rewrap(native_epoch_loader(loader, seed=42, epoch=1, base_batches=cursor.base_batches),
                            base_step_offset=cursor.valid_base_batches, pending_injection=cursor.pending_injection)
    resumed._build_injection_batch = lambda: injection
    actual = list(resumed)
    expected = complete[cut:]
    assert [item[1] for item in actual] == [item[1] for item in expected]
    for (expected_batch, _), (actual_batch, _) in zip(expected, actual):
        torch.testing.assert_close(expected_batch[0].overview_images, actual_batch[0].overview_images, rtol=0, atol=0)


def test_native_training_state_uses_completed_not_fetched_position(tmp_path):
    from core.tagger.tagger_trainer import _save_training_state, _load_training_state
    cursor = NativeReplayCursor(base_batches=2, valid_base_batches=2, last_batch_idx=1)
    _save_training_state(str(tmp_path), "latest", 1, 2, 2, 0., .35,
                         native_runtime={"cursor": cursor.to_dict()})
    with pytest.raises(ValueError, match="no readable commit"):
        _load_training_state(str(tmp_path), "latest")
    state = json.loads((tmp_path / "latest_state.json").read_text())
    assert state["batch_idx"] == 1
    assert not (tmp_path / "latest_state.json.tmp").exists()


def test_native_rng_roundtrip_includes_numpy_and_current_cuda_stream(monkeypatch):
    from core.tagger.tagger_trainer import _capture_rng, _restore_rng
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.tensor([7, 9], dtype=torch.uint8))
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: restored_cuda.append(value.clone()))
    snapshot = _capture_rng(native=True)
    expected = (random.random(), torch.rand(3), np.random.rand(3))
    _restore_rng(json.loads(json.dumps(snapshot)))
    assert random.random() == expected[0]
    torch.testing.assert_close(torch.rand(3), expected[1], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(3), expected[2])
    assert restored_cuda[0].tolist() == [7, 9]


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("metrics_enabled", [False, True])
@pytest.mark.parametrize("f1_enabled", [False, True])
@pytest.mark.parametrize("exposure_enabled", [False, True])
def test_actual_common_loop_stop_resume_matches_uninterrupted(tmp_path, method, metrics_enabled, f1_enabled, exposure_enabled):
    from core.tagger.tagger_trainer import TaggerTrainer, _load_training_state
    from core.tagger.tag_vocabulary import TagVocabulary
    from core.gpu_coordinator import gpu_coordinator
    encoder_config = FoveaEncoderConfig(widths=(8, 16, 32, 32), depths=(1, 1, 1, 1),
        attention_heads=(2, 2), scale_dropout=.2)
    encoder = FoveaEncoder(encoder_config)
    path = save_pretrained_encoder(tmp_path / "encoder.safetensors", encoder,
        run_contract={"encoder": asdict(encoder_config)}, epoch=1, global_step=1, validation={})
    config = dict(TAGGER_TRAINING_DEFAULTS)
    config.update(architecture="foveatag", vision_encoder_path=str(path), training_method=method,
        fovea_feature_dim=16, fovea_overview_size=32, fovea_local_size=32, fovea_local_views=2,
        epochs=1, batch_size=2, num_workers=0, optimizer="adamw", mixed_precision="fp32",
        gradient_checkpointing=False, head_lr_multiplier=1., warmup_steps=2,
        save_every_n_steps=1, keep_last_n_checkpoints=0, save_tag_metrics=metrics_enabled, save_ood_reference=False,
        train_f1_eval_every_n_steps=1 if f1_enabled else 0,
        train_f1_threshold_search_every_n_steps=2 if f1_enabled else 0,
        danbooru_train_count_enable=exposure_enabled)
    vocabulary = TagVocabulary()
    vocabulary.tag_to_idx = {f"anonymous-{i}": i for i in range(5)}
    vocabulary.idx_to_tag = {i: name for name, i in vocabulary.tag_to_idx.items()}
    vocabulary.tag_to_category = {name: "General" for name in vocabulary.tag_to_idx}

    traces = {}
    def run(directory, *, stop=False, resume=False):
        torch.manual_seed(15)
        random.seed(15)
        np.random.seed(15)
        source = SyntheticNativeDataset()
        loader = DataLoader(source, batch_size=2, collate_fn=tagger_collate_fn)
        trainer = None
        events = []
        def callback(run_id, event, values):
            if event == "train_f1":
                events.append({key: values[key] for key in ("step", "train_f1", "train_precision", "train_recall", "threshold")})
            if stop and event == "checkpoint" and values.get("step") == 2:
                trainer.stop()
        trainer = TaggerTrainer("synthetic", config, vocabulary, str(directory), progress_callback=callback)
        resume_name = "step_000002" if f1_enabled else "latest"
        state = _load_training_state(str(directory), resume_name) if resume else None
        try:
            trainer.train(loader, None, source.processor, resume_state=state,
                          resume_ckpt_name=resume_name if resume else None)
            result = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
        finally:
            gpu_coordinator.unregister_trainer(trainer._coordinator_handle)
            trainer._coordinator_handle.detach()
        traces[(directory.name, resume)] = events
        return result

    expected = run(tmp_path / "continuous")
    run(tmp_path / "interrupted", stop=True)
    saved = _load_training_state(str(tmp_path / "interrupted"), "latest")
    assert saved["global_step"] == 2
    assert saved["batch_idx"] == 1
    assert saved["native_runtime"]["cursor"]["base_batches"] == 2
    actual = run(tmp_path / "interrupted", resume=True)
    if f1_enabled:
        assert traces[("continuous", False)] == traces[("interrupted", False)] + traces[("interrupted", True)]
    if metrics_enabled or f1_enabled or exposure_enabled:
        from core.tagger.tag_metrics_accumulator import TagMetricsAccumulator
        from core.tagger.foveatag.metrics_runtime import restore_metrics_runtime, HISTOGRAMS, VECTORS, COUNTERS
        metrics = []
        for directory in ("continuous", "interrupted"):
            value = TagMetricsAccumulator(5)
            state = _load_training_state(str(tmp_path / directory), "latest")
            restore_metrics_runtime(value, tmp_path / directory / "latest_metrics_runtime.npz", step=state["global_step"])
            metrics.append(value)
        for key in HISTOGRAMS + VECTORS:
            np.testing.assert_array_equal(getattr(metrics[0], key), getattr(metrics[1], key))
        for key in COUNTERS:
            assert getattr(metrics[0], key) == getattr(metrics[1], key)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
