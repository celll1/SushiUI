"""Sample exposure plans, independent captions and checkpoint-bound suffixes."""

from collections import Counter
import json
import random
from types import SimpleNamespace

import pytest
import torch

from core.training.concept_batch_order import ConceptOrderConfig, build_concept_batch_plan
from core.training.resume_batch_plan import make_plan, rebase_plan, resolve_suffix, save_plan, load_plan
from core.training.sample_occurrence import occurrence_batches, text_buffer_key
from core.training.occurrence_checkpoint import save_pending, restore_pending


def dataset(name="a", count=8):
    config = {"normalize_tags": False, "shuffle_tokens": True, "shuffle_per_epoch": True,
              "tag_dropout_rate": 0.15, "tag_dropout_per_epoch": True}
    ds = SimpleNamespace(unique_id=name, caption_config=config)
    ds.items = [{"image_path": f"{name}/{i}.png", "raw_caption": "hero, red, blue, green, white, black",
                 "caption": "legacy", "is_tags_format": True,
                 "tag_data": [{"tag": "hero", "category": "Character"}] + [
                     {"tag": color, "category": "General"} for color in ("red", "blue", "green", "white", "black")],
                 "bucket_width": 512, "bucket_height": 512,
                 "_caption_processing_config": config} for i in range(count)]
    return ds


def plan(ds, *, epoch=0, seed=7, **kwargs):
    config = ConceptOrderConfig.parse({"enabled": True, "min_items_per_concept": 1,
        "focus_batches": 2, "local_swap_window": 0, "batch_multiplier": 3, **kwargs})
    built = build_concept_batch_plan([(item, d) for d in ds for item in d.items], 2, config, epoch, seed)
    return built, config


def test_sample_counts_rebatching_and_segment_locality():
    ds = dataset()
    built, config = plan([ds])
    counts = Counter(item["image_path"] for batch in built.batches for item, _ in batch)
    assert set(counts.values()) == {3}
    segments = {}
    for batch in built.batches:
        assert len({item["image_path"] for item, _ in batch}) == len(batch)
        assert len({(item["bucket_width"], item["bucket_height"]) for item, _ in batch}) == 1
        for item, _ in batch:
            segments.setdefault(item["image_path"], set()).add(item["_sample_occurrence"]["segment"])
    assert all(len(values) == 1 for values in segments.values())
    base_pairs = {tuple(item["image_path"] for item, _ in batch) for batch in built.batches
                  if batch[0][0]["_sample_occurrence"]["kind"] == "base"}
    assert any(tuple(item["image_path"] for item, _ in batch) not in base_pairs for batch in built.batches)
    assert built.state(config, 7, None)["multiplier_samples"] == 16
    assert all(item["caption"] == "legacy" for item in ds.items)


def test_caption_context_is_repeatable_independent_and_does_not_advance_global_rng():
    ds = dataset()
    built, _ = plan([ds])
    random.seed(1337)
    state = random.getstate()
    captions = [item.get("caption") for batch in built.batches for item, _ in batch]
    assert random.getstate() == state
    assert captions == [item["caption"] for batch in built.batches for item, _ in batch]
    keys = [text_buffer_key(item) for batch in built.batches for item, _ in batch]
    assert len(set(keys)) == len(keys)
    assert len(set(captions)) > 1


def test_random_multiplicity_determinism_epoch_variation_and_bounds():
    ds = dataset(count=50)
    args = {"batch_multiplier_mode": "uniform_int", "batch_multiplier_min": 1, "batch_multiplier_max": 4}
    a, _ = plan([ds], **args)
    b, _ = plan([ds], **args)
    c, _ = plan([ds], epoch=1, **args)
    assert a.digest == b.digest and a.digest != c.digest
    counts = Counter(item["image_path"] for batch in a.batches for item, _ in batch)
    assert set(counts.values()) == {1, 2, 3, 4}


def test_default_multiplier_preserves_v1_plan_hash_and_rng():
    ds = dataset()
    a, ca = plan([ds], batch_multiplier=1)
    b, cb = plan([ds], batch_multiplier_mode="uniform_int", batch_multiplier_min=1, batch_multiplier_max=1)
    assert ca.fingerprint() == cb.fingerprint()
    assert a.digest == b.digest
    assert a.state(ca, 7, None)["version"] == 1
    assert all("_sample_occurrence" not in item for batch in a.batches for item, _ in batch)


@pytest.mark.parametrize("kwargs", [
    {"batch_multiplier": True}, {"batch_multiplier": 0}, {"batch_multiplier": 65},
    {"batch_multiplier_mode": "bad"}, {"batch_multiplier_min": 3, "batch_multiplier_max": 2},
])
def test_invalid_multiplier_is_rejected(kwargs):
    with pytest.raises(ValueError):
        plan([dataset()], **kwargs)


def test_ledger_suffix_captions_and_added_dataset_multiplier_survive_rebase(tmp_path):
    a, b = dataset(), dataset("b")
    built, config = plan([a])
    old = make_plan(built.batches, [a], "concept", config.fingerprint(), 0, batch_size=2)
    assert old["version"] == 2
    assert load_plan(tmp_path, save_plan(tmp_path, "run", old)) == old
    for cursor in range(len(built.batches) + 1):
        suffix = resolve_suffix(old, cursor, built.batches)
        assert [(item["_sample_occurrence"], item.get("caption")) for batch in suffix for item, _ in batch] == [
            (item["_sample_occurrence"], item.get("caption")) for batch in built.batches[cursor:] for item, _ in batch]
    added, _ = plan([a, b])
    candidate = make_plan(added.batches, [a, b], "concept", config.fingerprint(), 0, batch_size=2)
    revised, stats = rebase_plan(old, 3, candidate)
    assert revised["batches"][:3] == old["batches"][:3]
    assert stats["added_occurrences"] == len(b.items) * 3
    retained = [[entry for entry in batch if entry[0] == "a"] for batch in revised["batches"][3:]]
    assert [batch for batch in retained if batch] == old["batches"][3:]
    resolved = resolve_suffix(revised, 3, added.batches)
    assert sum(ds.unique_id == "b" for batch in resolved for _, ds in batch) == 24


def test_training_draws_survive_prefill_and_resume_without_global_rng_changes():
    built, _ = plan([dataset()])
    torch.manual_seed(12)
    initial = torch.get_rng_state()
    def draws(batches):
        result = []
        for batch in occurrence_batches(batches, "cpu"):
            result.append((torch.rand(len(batch)).tolist(), torch.randn(len(batch)).tolist(), random.random()))
        return result
    continuous = draws(built.batches)
    assert torch.equal(initial, torch.get_rng_state())
    torch.rand(20)
    random.random()
    assert continuous[3:] == draws(built.batches[3:])
    assert len({tuple(row[0]) for row in continuous}) > 1


def test_background_once_and_bucket_tails_are_not_padded_with_duplicates():
    ds = dataset(count=3)
    background = {"image_path": "background", "raw_caption": "scenery", "is_tags_format": False,
                  "bucket_width": 512, "bucket_height": 512}
    ds.items[0]["bucket_width"] = 768
    ds.items.append(background)
    built, _ = plan([ds])
    counts = Counter(item["image_path"] for batch in built.batches for item, _ in batch)
    assert counts["background"] == 1
    assert counts["a/0.png"] == 3
    assert all(len({item["image_path"] for item, _ in batch}) == len(batch) for batch in built.batches)


def test_partial_gas_gradient_sidecar_continues_optimizer_exactly(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    owner = SimpleNamespace(_concept_multiplier_active=True, _grad_accum_steps=2,
        output_dir=tmp_path, run_name="run", use_grad_scaler=False,
        _build_ema_param_name_map=lambda: {id(p): n for n, p in model.named_parameters()})
    value = torch.tensor([[0.2, 0.7]])
    (model(value).square().mean() / 2).backward()
    reference = save_pending(owner, [opt], 1)
    saved_optimizer = opt.state_dict()
    saved_optimizer["_sushi_opt_class"] = type(opt).__name__
    saved_optimizer["_sushi_param_names"] = [list(dict(model.named_parameters()))]
    expected_gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    saved_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    (model(value * 2).square().mean() / 2).backward()
    opt.step()
    result = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.load_state_dict(saved_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    opt.zero_grad(set_to_none=True)
    from core.training.base_trainer import BaseTrainer
    owner.log_prefix = "[Exposure test]"
    assert BaseTrainer._load_optimizer_state_by_parameter_name(owner, opt, saved_optimizer,
        saved_optimizer["_sushi_param_names"], saved_optimizer["_sushi_param_names"], "pre-update")
    restore_pending(owner, [opt], reference)
    assert all(torch.equal(p.grad, expected_gradients[n]) for n, p in model.named_parameters())
    (model(value * 2).square().mean() / 2).backward()
    opt.step()
    assert all(torch.equal(p, result[n]) for n, p in model.named_parameters())
    owner._grad_accum_steps = 3
    with pytest.raises(ValueError, match="settings"):
        restore_pending(owner, [opt], reference)


def test_public_schema_and_config_roundtrip_preserve_multiplier_when_disabled():
    import yaml
    from api.routes import TrainingRunCreateRequest
    from core.training.training_config import TrainingConfigGenerator
    value = {"enabled": False, "batch_multiplier": 3, "batch_multiplier_mode": "uniform_int",
             "batch_multiplier_min": 2, "batch_multiplier_max": 4}
    request = TrainingRunCreateRequest(training_method="lora", base_model_path="example.safetensors",
                                       total_steps=4, concept_batch_order=value)
    config = TrainingConfigGenerator.generate_lora_config(request.model_dump(),
        run_name="run", base_model_path="example.safetensors", output_dir="example-output")
    train = yaml.safe_load(config)["config"]["process"][0]["train"]
    for key, expected in value.items():
        assert train["concept_batch_order"][key] == expected
    assert TrainingRunCreateRequest.model_json_schema()["$defs"]["ConceptBatchOrderRequest"][
        "properties"]["batch_multiplier"]["maximum"] == 64
    with pytest.raises(ValueError, match="exposure|multiplier"):
        TrainingRunCreateRequest(training_method="lora", base_model_path="example.safetensors",
            total_steps=4, concept_batch_order={**value, "enabled": True}, text_encoding_mode="pre_encoded_cache")


def test_cpu_prefetch_uses_occurrence_key_without_caption_aliasing():
    from core.training.cpu_te_prefetch import CpuTextEncoderPrefetcher
    built, _ = plan([dataset()])
    def encode(captions, lyrics):
        return [(torch.tensor([len(caption)]), None) for caption in captions]
    worker = CpuTextEncoderPrefetcher(encode, built.batches, prefetch_depth=2)
    worker.start()
    try:
        for batch in built.batches:
            _, payload = worker.next(timeout=10)
            assert set(payload) == {text_buffer_key(item) for item, _ in batch}
            assert all(payload[text_buffer_key(item)][2] == item.get("caption") for item, _ in batch)
    finally:
        worker.stop()


def test_yaml_multiplier_cache_preflight_precedes_model_loading():
    from core.training.train_runner import _prepare_training_process_config
    config = {"config": {"process": [{"train": {
        "concept_batch_order": {"enabled": True, "batch_multiplier": 2},
        "text_encoding_mode": "pre_encoded_cache"}, "network": {"type": "lora"}}]}}
    with pytest.raises(ValueError, match="per-exposure"):
        _prepare_training_process_config(config, "nonexistent-model")
