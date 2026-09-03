"""Unit tests for adapter checkpoint codec detection and key normalization."""

import os
import sys
import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.adapters import (  # noqa: E402
    CodecRegistry,
    CodecSpec,
    detect_adapter_codec,
    normalize_adapter_keys,
)


def test_detect_sushiui_canonical_lora():
    tensors = {
        "model.diffusion_model.proj.lora_down.weight": torch.randn(4, 8),
        "model.diffusion_model.proj.lora_up.weight": torch.randn(8, 4),
        "model.diffusion_model.proj.alpha": torch.tensor(8.0),
    }
    spec = detect_adapter_codec(tensors)
    assert spec.algorithm == "lora"
    assert not spec.weight_decompose
    assert spec.format == "sushiui_canonical"
    assert spec.rank == 4
    assert spec.alpha == 8.0


def test_detect_sushiui_canonical_dora():
    tensors = {
        "model.diffusion_model.proj.lora_down.weight": torch.randn(4, 8),
        "model.diffusion_model.proj.lora_up.weight": torch.randn(8, 4),
        "model.diffusion_model.proj.dora_scale": torch.randn(8),
    }
    spec = detect_adapter_codec(tensors)
    assert spec.algorithm == "lora"
    assert spec.weight_decompose is True
    assert spec.format == "sushiui_canonical"


def test_detect_loha_and_lokr():
    loha_tensors = {
        "layer.hada_w1_a": torch.randn(8, 4),
        "layer.hada_w1_b": torch.randn(4, 8),
        "layer.hada_w2_a": torch.randn(8, 4),
        "layer.hada_w2_b": torch.randn(4, 8),
    }
    spec_loha = detect_adapter_codec(loha_tensors)
    assert spec_loha.algorithm == "loha"
    assert spec_loha.format == "sushiui_canonical"

    lokr_tensors = {
        "layer.lokr_w1": torch.randn(4, 4),
        "layer.lokr_w2_a": torch.randn(2, 2),
        "layer.lokr_w2_b": torch.randn(2, 2),
    }
    spec_lokr = detect_adapter_codec(lokr_tensors)
    assert spec_lokr.algorithm == "lokr"
    assert spec_lokr.format == "sushiui_canonical"


def test_detect_lycoris_kohya_metadata():
    metadata = {
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": '{"algo": "loha"}',
        "lora_rank": "8",
        "lora_alpha": "4.0",
    }
    tensors = {
        "lora_unet_double_blocks_0_img_attn_qkv.hada_w1_a": torch.randn(8, 8),
    }
    spec = detect_adapter_codec(tensors, metadata=metadata)
    assert spec.algorithm == "loha"
    assert spec.format == "lycoris_kohya"
    assert spec.rank == 8
    assert spec.alpha == 4.0


def test_detect_and_normalize_diffusers_peft():
    peft_tensors = {
        "base_model.model.transformer.attn.q.lora_A.weight": torch.randn(4, 8),
        "base_model.model.transformer.attn.q.lora_B.weight": torch.randn(8, 4),
    }
    spec = detect_adapter_codec(peft_tensors)
    assert spec.algorithm == "lora"
    assert spec.format == "diffusers_peft"

    normalized = normalize_adapter_keys(peft_tensors, spec)
    assert "transformer.attn.q.lora_down.weight" in normalized
    assert "transformer.attn.q.lora_up.weight" in normalized
    assert "base_model.model.transformer.attn.q.lora_A.weight" not in normalized


def test_detect_unknown_format():
    random_tensors = {
        "unrelated_layer.weight": torch.randn(4, 4),
        "bias": torch.zeros(4),
    }
    spec = detect_adapter_codec(random_tensors)
    assert spec.algorithm == "unknown"
    assert spec.format == "unknown"
