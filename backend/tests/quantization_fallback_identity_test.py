from __future__ import annotations

import os
import sys

import torch
import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.vram_optimization import (
    _cached_runtime_quantization,
    _discard_runtime_quantization_cache,
    _quantize_text_encoder,
    _quantize_transformer,
    _quantize_unet,
    offload_cached_runtime_quantization,
    restore_runtime_quantization_sources,
)
from core import vram_optimization


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def to(self, *args, **kwargs):
        self.last_move = args[0] if args else kwargs.get("device")
        return self


def test_unsupported_quantization_returns_original_models():
    unet = _Model()
    transformer = _Model()
    text_encoder = _Model()

    assert _quantize_unet(unet, "unsupported") is unet
    assert _quantize_transformer(transformer, "unsupported") is transformer
    assert _quantize_text_encoder(text_encoder, "unsupported") is text_encoder


def test_copy_failure_returns_original_models(monkeypatch):
    def fail_copy(model):
        raise RuntimeError("copy failed")

    monkeypatch.setattr("core.vram_optimization.copy.deepcopy", fail_copy)
    unet = _Model()
    transformer = _Model()
    text_encoder = _Model()

    assert _quantize_unet(unet, "fp8_e4m3fn") is unet
    assert _quantize_transformer(transformer, "fp8_e4m3fn") is transformer
    assert _quantize_text_encoder(text_encoder, "fp8_e4m3fn") is text_encoder


def test_runtime_quantization_cache_reuses_one_identity_and_evicts_on_change():
    source = _Model()
    owner = {"transformer": source}
    created = []

    def quantize(_source, mode):
        candidate = _Model()
        candidate.mode = mode
        created.append(candidate)
        return candidate

    first, applied, cached = _cached_runtime_quantization(
        source, "fp8_e4m3fn", quantize,
        cache_owner=owner, cache_identity="model-a", component_name="transformer",
    )
    assert applied and not cached
    assert owner["transformer"] is first

    restore_runtime_quantization_sources(owner)
    assert owner["transformer"] is source

    reused, applied, cached = _cached_runtime_quantization(
        source, "fp8_e4m3fn", quantize,
        cache_owner=owner, cache_identity="model-a", component_name="transformer",
    )
    assert applied and cached
    assert reused is first
    assert len(created) == 1

    replacement, applied, cached = _cached_runtime_quantization(
        source, "fp8_e5m2", quantize,
        cache_owner=owner, cache_identity="model-b", component_name="transformer",
    )
    assert applied and not cached
    assert replacement is not first
    assert len(created) == 2
    cache = owner["_runtime_fp8_cache"]
    assert list(cache) == ["transformer"]
    assert cache["transformer"]["model"] is replacement
    assert cache["transformer"]["source"] is source


def test_failed_runtime_quantization_never_caches_or_replaces_source():
    source = _Model()
    owner = {"text_encoder": source}

    result, applied, cached = _cached_runtime_quantization(
        source, "fp8_e4m3fn", lambda model, _mode: model,
        cache_owner=owner, cache_identity="model-a", component_name="text_encoder",
    )

    assert result is source
    assert not applied and not cached
    assert owner["text_encoder"] is source
    assert owner["_runtime_fp8_cache"] == {}


def test_runtime_quantization_cache_offloads_and_discards_active_copy():
    source = _Model()
    owner = {"transformer": source}
    candidate, _, _ = _cached_runtime_quantization(
        source, "fp8_e4m3fn", lambda _model, _mode: _Model(),
        cache_owner=owner, cache_identity="model-a", component_name="transformer",
    )

    offload_cached_runtime_quantization(owner, "transformer")
    assert candidate.last_move == "cpu"

    _discard_runtime_quantization_cache(owner, "transformer")
    assert owner["transformer"] is source
    assert "_runtime_fp8_cache" not in owner


def test_successful_weight_only_quantization_returns_a_distinct_fp8_model():
    source = _Model()

    quantized = _quantize_transformer(source, "fp8_e4m3fn")

    assert quantized is not source
    assert source.linear.weight.dtype == torch.float32
    assert quantized.linear.weight.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    ("move_name", "quantizer_name", "component_name"),
    [
        ("move_zimage_text_encoder_to_gpu", "_quantize_text_encoder", "text_encoder"),
        ("move_flux2_text_encoder_to_gpu", "_quantize_text_encoder", "text_encoder"),
        ("move_zimage_transformer_to_gpu", "_quantize_transformer", "transformer"),
        ("move_flux2_transformer_to_gpu", "_quantize_transformer", "transformer"),
    ],
)
def test_runtime_fp8_move_paths_reuse_the_cached_active_object(
    monkeypatch, move_name, quantizer_name, component_name
):
    source = _Model()
    owner = {component_name: source}
    candidate = _Model()
    calls = []

    def quantize(model, mode):
        calls.append((model, mode))
        return candidate

    monkeypatch.setattr(vram_optimization, quantizer_name, quantize)
    move = getattr(vram_optimization, move_name)

    first = move(
        source, "fp8_e4m3fn", cache_owner=owner, cache_identity="model-a"
    )
    restore_runtime_quantization_sources(owner)
    second = move(
        source, "fp8_e4m3fn", cache_owner=owner, cache_identity="model-a"
    )

    assert first is candidate and second is candidate
    assert owner[component_name] is candidate
    assert calls == [(source, "fp8_e4m3fn")]
    assert candidate.last_move == "cuda:0"
