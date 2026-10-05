"""Registry-wide exposure contracts and architecture-neutral batch boundaries."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import random

import pytest
import torch

from core.training.arch import ARCH_REGISTRY
from core.training.concept_batch_order import ConceptOrderConfig, build_concept_batch_plan
from core.training.concept_support import CONCEPT_ORDER_SUPPORT, validate_concept_order
from core.training.resume_batch_plan import make_plan, resolve_suffix
from core.training.sample_occurrence import OccurrenceItem, text_buffer_key
from concept_sample_multiplier_test import dataset


def concept():
    return ConceptOrderConfig.parse({"enabled": True, "batch_multiplier": 3,
        "min_items_per_concept": 1, "focus_batches": 1, "local_swap_window": 0})


def test_registry_api_and_ui_cover_every_architecture():
    from api.arch_capabilities import TRAINING_DECLARED_ARCHS, training_feature_unsupported_reason
    assert set(ARCH_REGISTRY) == set(CONCEPT_ORDER_SUPPORT) == set(TRAINING_DECLARED_ARCHS)
    for arch, contract in CONCEPT_ORDER_SUPPORT.items():
        assert bool(training_feature_unsupported_reason(arch, "concept_batch_order", "lora")) != contract.supported
    source = (Path(__file__).parents[2] / "frontend/src/components/training/TrainingConfig.tsx").read_text(encoding="utf-8")
    assert 'unsupportedTrainingFeature("concept_batch_order")' in source
    assert 'setError(conceptOrderUnsupported)' in source


@pytest.mark.parametrize("arch", sorted(ARCH_REGISTRY))
def test_public_api_concept_contract_for_every_architecture(monkeypatch, arch):
    from api.routes import TrainingRunCreateRequest
    from core.model_loader import ModelLoader
    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda _: arch)
    arguments = dict(base_model_path="test-model", training_method="lora", total_steps=4,
        text_encoding_mode="swap_onthefly", concept_batch_order={"enabled": True, "batch_multiplier": 2})
    if CONCEPT_ORDER_SUPPORT[arch].supported:
        request = TrainingRunCreateRequest(**arguments)
        assert request.concept_batch_order.batch_multiplier == 2
    else:
        with pytest.raises(ValueError, match="concept_batch_order.*unsupported"):
            TrainingRunCreateRequest(**arguments)


@pytest.mark.parametrize("arch", [name for name, support in CONCEPT_ORDER_SUPPORT.items() if support.supported])
def test_image_architectures_preserve_exposure_caption_buffer_and_resume(arch):
    from core.training.cpu_te_prefetch import CpuTextEncoderPrefetcher
    ds = dataset(count=4)
    validate_concept_order(arch, concept(), text_encoding_mode="cpu_prefetch", datasets=[ds])
    plan = build_concept_batch_plan([(item, ds) for item in ds.items], 2, concept(), 0, 42)
    assert set(Counter(item["image_path"] for batch in plan.batches for item, _ in batch).values()) == {3}
    worker = CpuTextEncoderPrefetcher(lambda captions, _: [(torch.tensor([len(cap)]), None) for cap in captions],
        plan.batches, prefetch_depth=1)
    worker.start()
    try:
        for batch in plan.batches:
            _, payload = worker.next(timeout=10)
            assert set(payload) == {text_buffer_key(item) for item, _ in batch}
            assert [payload[text_buffer_key(item)][2] for item, _ in batch] == [item["caption"] for item, _ in batch]
    finally:
        worker.stop()
    ledger = make_plan(plan.batches, [ds], "concept", "contract", 0)
    suffix = resolve_suffix(ledger, 2, [[(item, ds) for item in ds.items]])
    assert [[item["caption"] for item, _ in batch] for batch in suffix] == [
        [item["caption"] for item, _ in batch] for batch in plan.batches[2:]]


def test_sensenova_ordinary_batch_prefix_keeps_occurrence_captions(monkeypatch):
    from core.training.arch.sensenova import SenseNovaArchHandler
    from core.training.ops import sensenova_ops
    from core.training.sensenova_tasks import build_task_homogeneous_batches
    ds = dataset(count=4)
    plan = build_concept_batch_plan([(item, ds) for item in ds.items], 2, concept(), 0, 42)
    batches = build_task_homogeneous_batches(plan.batches, 2, random.Random(42))
    assert all(isinstance(item, OccurrenceItem) for batch in batches for item, _ in batch)
    seen = []
    monkeypatch.setattr(sensenova_ops, "encode_prompts", lambda trainer, prompts, **kw: seen.append((prompts, kw)))
    handler = SenseNovaArchHandler()
    for batch in batches:
        prompts = [item["caption"] for item, _ in batch]
        handler.encode_prompts(SimpleNamespace(), prompts, cfg_null=[True, False])
        assert seen[-1] == (prompts, {"requires_grad": False, "reference_image_paths": None, "cfg_null": [True, False]})
    assert len({caption for prompts, _ in seen for caption in prompts}) > 1
    ds.items[0]["_sensenova_task_views"] = [{"task": "t2i"}]
    with pytest.raises(ValueError, match="task_views"):
        validate_concept_order("sensenova", concept(), text_encoding_mode="onthefly_gpu", datasets=[ds])


def test_custom_signature_prevents_incompatible_batches_and_stale_resume():
    ds = dataset(count=4)
    for index, item in enumerate(ds.items):
        item["_concept_batch_signature"] = (str(index % 2),)
    plan = build_concept_batch_plan([(item, ds) for item in ds.items], 2, concept(), 0, 42)
    assert all(len({item["_concept_batch_signature"] for item, _ in batch}) == 1 for batch in plan.batches)
    ledger = make_plan(plan.batches, [ds], "concept", "signature", 0)
    ds.items[0]["_concept_batch_signature"] = ("changed",)
    with pytest.raises(ValueError):
        resolve_suffix(ledger, 0, [[(item, ds) for item in ds.items]])


def test_undeclared_arch_and_native_yue2_do_not_ignore_settings():
    with pytest.raises(ValueError, match="no declared"):
        validate_concept_order("future-native", concept(), text_encoding_mode="onthefly_gpu")
    from core.training.ops.yue2_ops import train_abc_ar_loop
    trainer = SimpleNamespace(config={"concept_batch_order": {"enabled": True, "batch_multiplier": 2}})
    with pytest.raises(ValueError, match="unsupported for yue2"):
        train_abc_ar_loop(trainer, datasets=[])


@pytest.mark.parametrize("arch", [name for name, support in CONCEPT_ORDER_SUPPORT.items() if not support.supported])
def test_yaml_native_arch_refusal_precedes_dataset_work(monkeypatch, arch):
    from core.model_loader import ModelLoader
    from core.training.train_runner import _prepare_training_process_config
    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda _: arch)
    config = {"config": {"process": [{"train": {"text_encoding_mode": "onthefly_gpu",
        "concept_batch_order": {"enabled": True, "batch_multiplier": 2}}, "network": {"type": "lora"}}]}}
    with pytest.raises(ValueError, match="unsupported"):
        _prepare_training_process_config(config, "test-model")


def test_sensenova_task_views_and_vae_yaml_are_refused_before_loading(monkeypatch):
    from core.model_loader import ModelLoader
    from core.training.train_runner import _prepare_training_process_config
    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda _: "sensenova")
    config = {"config": {"process": [{"train": {"text_encoding_mode": "onthefly_gpu",
        "concept_batch_order": {"enabled": True, "batch_multiplier": 2}}, "network": {"type": "lora"},
        "datasets": [{"task_views": [{"task": "t2i"}]}]}]}}
    with pytest.raises(ValueError, match="task_views"):
        _prepare_training_process_config(config, "test-model")
    config["config"]["process"][0]["network"]["type"] = "vae_decoder"
    with pytest.raises(ValueError, match="VAE-only"):
        _prepare_training_process_config(config, "test-model")
