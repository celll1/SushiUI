"""Native CPU cache integrity, model-content identity and atomic cancellation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming cache attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda

from core.models.ming_image_01.cache import MingTensorCache, identity_key, module_fingerprint, latent_identity, conditioning_identity


def expected(tensors): return {name: (value.shape, value.dtype) for name, value in tensors.items()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_frames_and_sparse_ragged_references_roundtrip(tmp_path, dtype):
    cache = MingTensorCache(tmp_path)
    key = identity_key({"test": "latent"})
    tensors = {"clean": torch.randn(3, 16, 3, 4, 4).to(dtype),
               "reference_0": torch.randn(16, 1, 4, 4).to(dtype),
               "reference_2": torch.randn(16, 2, 4, 4).to(dtype)}
    cache.write("latents", key, tensors)
    loaded = cache.read("latents", key, expected=expected(tensors))
    assert set(loaded) == set(tensors) and "reference_1" not in loaded
    for name in tensors: torch.testing.assert_close(loaded[name], tensors[name], rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_frozen_dual_stream_roundtrip_and_trainability_gate(tmp_path):
    cache = MingTensorCache(tmp_path)
    key = identity_key({"test": "conditioning"})
    tensors = {"queries": torch.randn(2, 4, 8), "direct": torch.randn(2, 7, 24)}
    frozen = SimpleNamespace(cacheable=True)
    cache.write("conditioning", key, tensors, conditioner=frozen)
    result = cache.read("conditioning", key, expected=expected(tensors), conditioner=frozen)
    for name in tensors: torch.testing.assert_close(result[name], tensors[name], rtol=0, atol=0)
    frozen.cacheable = False
    with pytest.raises(ValueError, match="completely frozen"):
        cache.read("conditioning", key, expected=expected(tensors), conditioner=frozen)
    with pytest.raises(ValueError, match="completely frozen"):
        cache.write("conditioning", key, tensors, conditioner=frozen)


@pytest.mark.parametrize("problem", ["truncated", "payload", "shape", "dtype", "role", "identity"])
def test_corrupt_or_incompatible_cache_is_a_miss(tmp_path, problem):
    cache = MingTensorCache(tmp_path)
    key = identity_key({"test": "integrity"})
    tensors = {"clean": torch.zeros(1, 16, 1, 4, 4)}
    cache.write("latents", key, tensors)
    path = tmp_path / "latents" / (key + ".safetensors")
    wanted = expected(tensors)
    if problem == "truncated": path.write_bytes(b"broken")
    elif problem == "payload":
        raw = bytearray(path.read_bytes()); raw[-1] = 1; path.write_bytes(raw)
    elif problem == "shape": wanted["clean"] = ((1, 16, 2, 4, 4), torch.float32)
    elif problem == "dtype": wanted["clean"] = (tensors["clean"].shape, torch.bfloat16)
    elif problem == "role": wanted = {"queries": wanted["clean"]}
    else:
        other = identity_key({"test": "other"})
        path.rename(path.with_name(other + ".safetensors")); key = other
    assert cache.read("latents", key, expected=wanted) is None


def test_cancelled_replacement_preserves_existing_cache_and_removes_temporary(tmp_path):
    cache = MingTensorCache(tmp_path)
    key = identity_key({"test": "cancel"})
    original = {"clean": torch.zeros(1, 16, 1, 4, 4)}
    cache.write("latents", key, original)
    with pytest.raises(RuntimeError, match="cancelled"):
        cache.write("latents", key, {"clean": torch.ones_like(original["clean"])}, cancelled=lambda: True)
    result = cache.read("latents", key, expected=expected(original))
    torch.testing.assert_close(result["clean"], original["clean"], rtol=0, atol=0)
    assert not list(tmp_path.rglob("*.tmp"))


def test_content_fingerprints_and_semantic_inputs_invalidate_keys():
    model = torch.nn.Linear(4, 3)
    first = module_fingerprint(model)
    with torch.no_grad(): model.weight[0, 0] += 1
    second = module_fingerprint(model)
    assert first != second
    latent = dict(pixels=["a", "b"], vae_weights=first, vae_config={"scale": 1}, dtype=torch.float32, execution={"tiling": False})
    key = latent_identity(**latent)
    assert key != latent_identity(**{**latent, "vae_weights": second})
    assert key != latent_identity(**{**latent, "pixels": ["b", "a"]})
    assert key != latent_identity(**{**latent, "dtype": torch.bfloat16})
    args = dict(pixels=["a"], captions=["word5"], artifact_config="processor-tokenizer-config",
        conditioner_weights=first, resolution=32, dtype=torch.float32, topology="dense", execution={"attention": "eager"})
    key = conditioning_identity(**args)
    for field, value in {"captions": ["word6"], "conditioner_weights": second,
                         "resolution": 64, "topology": "unet", "artifact_config": "other-processor", "execution": {"attention": "other"}}.items():
        assert key != conditioning_identity(**{**args, field: value})


@pytest.mark.parametrize("problem", ["nan", "channels", "reference_grid", "unknown_role"])
def test_invalid_native_tensors_cannot_publish(tmp_path, problem):
    cache = MingTensorCache(tmp_path)
    tensors = {"clean": torch.zeros(1, 16, 1, 4, 4)}
    if problem == "nan": tensors["clean"][0, 0, 0, 0, 0] = float("nan")
    elif problem == "channels": tensors["clean"] = tensors["clean"][:, :3]
    elif problem == "reference_grid": tensors["reference_0"] = torch.zeros(16, 1, 2, 2)
    else: tensors["video"] = tensors["clean"]
    with pytest.raises(ValueError): cache.write("latents", identity_key({"test": problem}), tensors)
    assert not list(tmp_path.rglob("*.safetensors"))



def cached_record_setup(tmp_path, profile="layer_decompose", trainable=False):
    from core.models.ming_image_01.dataset import inspect_item, load_pixels
    from core.models.ming_image_01.cache import snapshot_cache_context
    from ming_image_01_dataset_test import layer_files, sidecar_path
    from ming_image_01_training_batch_test import components
    root = tmp_path / "data"
    primary, _ = layer_files(root, 3)
    if profile == "generation_edit": sidecar_path(primary).unlink()
    record = inspect_item(primary, root=root, profile=profile)
    item = load_pixels(record, caption="word5", width=32, height=32)
    value = components(profile)
    value.conditioner.set_trainable({"language", "projections"} if trainable else set())
    value.artifact.config_fingerprint = identity_key({"tiny": profile})
    cache = MingTensorCache(tmp_path / "cache")
    context = snapshot_cache_context(value, cache, execution={"native": 1, "attention": "eager", "tiling": False},
        latent_enabled=True, conditioning_enabled=not trainable)
    return value, item, context


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_record_cache_hit_skips_real_vae_and_frozen_conditioner_and_keeps_loss(tmp_path, monkeypatch, profile):
    from core.models.ming_image_01.dataset import prepare_record_latents
    from core.models.ming_image_01.cache import expected_conditioning
    from core.training.ops.ming_image_01_ops import train_step
    from ming_image_01_training_batch_test import trainer
    value, item, context = cached_record_setup(tmp_path, profile)
    first = prepare_record_latents(value, [item], resolution=32, cache_context=context).for_step()
    assert len(list((tmp_path / "cache").rglob("*.safetensors"))) == 2
    def forbidden_encoder(*a, **k): raise AssertionError("Cache hit reexecuted a native encoder")
    monkeypatch.setattr(value.vae, "encode", forbidden_encoder)
    monkeypatch.setattr(value.conditioner, "forward", forbidden_encoder)
    second = prepare_record_latents(value, [item], resolution=32, cache_context=context).for_step()
    for a, b in ((first.clean, second.clean), (first.conditioning.queries, second.conditioning.queries),
                 (first.conditioning.direct, second.conditioning.direct)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert first.reference_latents is None if profile == "generation_edit" else len(second.reference_latents) == 1
    if second.reference_latents is not None:
        torch.testing.assert_close(first.reference_latents[0], second.reference_latents[0], rtol=0, atol=0)
    owner = trainer(value, _active_mnt_noise=torch.ones_like(first.clean))
    a = train_step(owner, first, timesteps=torch.tensor([.4]))[0]
    b = train_step(owner, second, timesteps=torch.tensor([.4]))[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    b.backward()
    assert value.transformer.layers[0].attention.to_q.weight.grad.abs().sum() > 0
    assert not torch.cuda.is_initialized()


def test_trainable_conditioner_uses_cached_vae_but_retains_new_encoder_graph(tmp_path, monkeypatch):
    from core.models.ming_image_01.dataset import prepare_record_latents
    from core.training.ops.ming_image_01_ops import train_step
    from ming_image_01_training_batch_test import trainer
    value, item, context = cached_record_setup(tmp_path, trainable=True)
    first = prepare_record_latents(value, [item], resolution=32, cache_context=context).for_step()
    def forbidden_encoder(*a, **k): raise AssertionError("Latent cache hit reexecuted VAE")
    monkeypatch.setattr(value.vae, "encode", forbidden_encoder)
    second = prepare_record_latents(value, [item], resolution=32, cache_context=context).for_step()
    assert second.conditioning.direct.requires_grad and second.conditioning.queries.requires_grad
    assert not list((tmp_path / "cache" / "conditioning").glob("*.safetensors"))
    train_step(trainer(value), second, timesteps=torch.tensor([.4]))[0].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in value.conditioner.parameters())


def test_caption_edit_reuses_latents_but_recomputes_frozen_conditioning(tmp_path, monkeypatch):
    from dataclasses import replace
    from core.models.ming_image_01.dataset import prepare_record_latents
    value, item, context = cached_record_setup(tmp_path)
    first = prepare_record_latents(value, [item], resolution=32, cache_context=context).for_step()
    def forbidden_encoder(*a, **k): raise AssertionError("Caption edit invalidated VAE cache")
    monkeypatch.setattr(value.vae, "encode", forbidden_encoder)
    changed = prepare_record_latents(value, [replace(item, caption="word6")], resolution=32, cache_context=context).for_step()
    assert not torch.equal(first.conditioning.direct, changed.conditioning.direct)
    assert len(list((tmp_path / "cache" / "conditioning").glob("*.safetensors"))) == 2


def test_weight_mutation_refuses_stale_snapshot_before_cache_read(tmp_path):
    from core.models.ming_image_01.dataset import prepare_record_latents
    value, item, context = cached_record_setup(tmp_path)
    prepare_record_latents(value, [item], resolution=32, cache_context=context).for_step()
    with torch.no_grad(): next(value.conditioner.parameters()).add_(1)
    with pytest.raises(ValueError, match="weights changed"):
        prepare_record_latents(value, [item], resolution=32, cache_context=context)



def test_fresh_model_context_and_common_preparation_reuse_published_cache(tmp_path, monkeypatch):
    import copy
    from core.models.ming_image_01.dataset import prepare_record_latents, prepare_shared_latents, shared_inputs_for_mnt
    from core.models.ming_image_01.cache import snapshot_cache_context
    from ming_image_01_training_batch_test import trainer
    value, item, context = cached_record_setup(tmp_path)
    first = prepare_record_latents(value, [item], resolution=32, cache_context=context).for_step()
    restored = copy.deepcopy(value)
    second_context = snapshot_cache_context(restored, context["cache"], execution=context["execution"],
        latent_enabled=True, conditioning_enabled=True)
    def forbidden_encoder(*a, **k): raise AssertionError("Restored-model cache hit executed encoder")
    monkeypatch.setattr(restored.vae, "encode", forbidden_encoder)
    monkeypatch.setattr(restored.conditioner, "forward", forbidden_encoder)
    monkeypatch.setattr(restored.vae, "to", forbidden_encoder)
    monkeypatch.setattr(restored.conditioner, "to", forbidden_encoder)
    owner = trainer(restored)
    owner.ming_image_01_components, owner.ming_cache_context = restored, second_context
    owner.bucket_strategy, owner.ming_prompt_resolution = "resize", 32
    pair = ({"_ming_record": item.record, "width": 32, "height": 32, "caption": "word5"}, SimpleNamespace(unique_id="test"))
    prepared = prepare_shared_latents(owner, [pair])
    packed, aux, refs = shared_inputs_for_mnt(owner, prepared)
    from core.models.ming_image_01.training_data import native_step_batch
    second = native_step_batch(prepared.clean, packed, aux, refs)
    torch.testing.assert_close(first.clean, second.clean, rtol=0, atol=0)
    torch.testing.assert_close(first.conditioning.queries, second.conditioning.queries, rtol=0, atol=0)
    torch.testing.assert_close(first.conditioning.direct, second.conditioning.direct, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()



def test_native_mixed_query_autocast_output_is_canonical_without_losing_gradients(monkeypatch):
    from ming_image_01_conditioning_test import make_conditioner
    value = make_conditioner()
    queries = torch.randn(1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    direct = torch.randn(1, 5, 24, requires_grad=True)
    monkeypatch.setattr(value.native, "get_condition_embeds_for_image_gen", lambda **kw: (queries, direct))
    result = value(input_ids=torch.ones(1, 5, dtype=torch.long), attention_mask=torch.ones(1, 5, dtype=torch.long))
    assert result.queries.dtype == result.direct.dtype == next(value.parameters()).dtype
    torch.testing.assert_close(result.queries, queries.float(), rtol=0, atol=0)
    (result.queries.sum()+result.direct.sum()).backward()
    assert torch.all(queries.grad == 1) and torch.all(direct.grad == 1)


def test_native_vae_transfer_failure_still_attempts_cpu_cleanup(tmp_path, monkeypatch):
    from core.models.ming_image_01.dataset import prepare_record_latents
    value, item, _ = cached_record_setup(tmp_path)
    calls = []
    def transfer(**kw):
        calls.append(str(kw["device"]))
        if len(calls) == 1: raise RuntimeError("transfer failed")
        return value.vae
    monkeypatch.setattr(value.vae, "to", transfer)
    with pytest.raises(RuntimeError, match="transfer failed"):
        prepare_record_latents(value, [item], resolution=32, vae_device=torch.device("cpu"))
    assert calls == ["cpu", "cpu"]
