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
