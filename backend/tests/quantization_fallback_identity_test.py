from __future__ import annotations

import os
import sys

import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.vram_optimization import (
    _quantize_text_encoder,
    _quantize_transformer,
    _quantize_unet,
)


class _Model(torch.nn.Module):
    pass


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
