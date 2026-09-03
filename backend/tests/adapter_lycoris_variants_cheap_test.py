"""LyCORIS adapter variants (LoHa, LoKr, DoRA) unit and composite tests."""

import os
import sys
import pytest
import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.adapters import (  # noqa: E402
    CompositeAdapterLayer,
    DoRALinearLayer,
    LoHaLinearLayer,
    LoKrLinearLayer,
    LoRALinearLayer,
)


def test_loha_linear_layer_forward_and_strength():
    torch.manual_seed(42)
    base = nn.Linear(8, 6)
    loha = LoHaLinearLayer(base, rank=4, alpha=4.0, lora_name="test_loha")

    x = torch.randn(2, 8)
    out_base = base(x)
    out_loha = loha(x)
    delta = loha.forward_delta(x)

    assert torch.allclose(out_loha, out_base + delta, atol=1e-6)

    # strength = 0 gives zero delta
    loha.set_adapter_strength(0.0)
    delta_zero = loha.forward_delta(x)
    assert torch.allclose(delta_zero, torch.zeros_like(delta_zero), atol=1e-7)

    # branch tensors
    tensors = loha.branch_tensors()
    assert set(tensors.keys()) == {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "alpha"}
    assert tensors["hada_w1_a"].shape == (6, 4)
    assert tensors["hada_w1_b"].shape == (4, 8)


def test_lokr_linear_layer_forward_and_strength():
    torch.manual_seed(42)
    base = nn.Linear(8, 6)
    lokr = LoKrLinearLayer(base, rank=2, alpha=2.0, lora_name="test_lokr")

    x = torch.randn(2, 8)
    out_base = base(x)
    out_lokr = lokr(x)
    delta = lokr.forward_delta(x)

    assert torch.allclose(out_lokr, out_base + delta, atol=1e-6)

    # strength = 0 gives zero delta
    lokr.set_adapter_strength(0.0)
    delta_zero = lokr.forward_delta(x)
    assert torch.allclose(delta_zero, torch.zeros_like(delta_zero), atol=1e-7)

    # branch tensors
    tensors = lokr.branch_tensors()
    assert "lokr_w1" in tensors
    assert "lokr_w2_a" in tensors
    assert "lokr_w2_b" in tensors


def test_lokr_decompose_both_stores_w1_low_rank():
    """``lokr_w1_a``/``lokr_w1_b`` is a real upstream form; the two factors must
    multiply back to the same ``(out_l, in_m)`` block the full ``lokr_w1`` is."""
    torch.manual_seed(42)
    base = nn.Linear(8, 6)
    plain = LoKrLinearLayer(base, rank=2, alpha=2.0, lora_name="plain")
    both = LoKrLinearLayer(base, rank=2, alpha=2.0, lora_name="both",
                           decompose_both=True)

    assert set(both.branch_tensors()) == {
        "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "alpha"}
    assert both.lokr_w1_a.shape[0] == plain.lokr_w1.shape[0]
    assert both.lokr_w1_b.shape[1] == plain.lokr_w1.shape[1]
    assert both.compute_delta_weight().shape == (6, 8)

    with pytest.raises(ValueError):
        LoKrLinearLayer(base, rank=0, alpha=2.0, lora_name="x", decompose_both=True)


def test_dora_linear_layer_strength_zero_identity():
    """DoRA must satisfy exact strength-zero identity: W_eff = W0 when delta = 0."""
    torch.manual_seed(42)
    base = nn.Linear(8, 6)
    lora_branch = LoRALinearLayer(base, rank=2, alpha=2.0, lora_name="sub_lora")
    lora_branch.lora_up.weight.data.normal_()
    dora = DoRALinearLayer(base, lora_branch)

    x = torch.randn(2, 8)
    out_base = base(x)

    # At strength 0, output must match base output
    dora.set_adapter_strength(0.0)
    out_dora = dora(x)
    assert torch.allclose(out_dora, out_base, atol=1e-6)

    # At strength 1, output deviates from base
    dora.set_adapter_strength(1.0)
    out_dora_active = dora(x)
    assert not torch.allclose(out_dora_active, out_base, atol=1e-4)

    # branch tensors include dora_scale
    tensors = dora.branch_tensors()
    assert "dora_scale" in tensors
    assert tensors["dora_scale"].shape == (6,)


def test_composite_supports_loha_lokr_and_dora_branches():
    """CompositeAdapterLayer must allow simultaneous stacking of LoRA, LoHa, LoKr, and DoRA."""
    torch.manual_seed(42)
    base = nn.Linear(8, 6)
    parent = nn.Sequential(base)

    composite = CompositeAdapterLayer.attach(parent, 0)
    assert isinstance(parent[0], CompositeAdapterLayer)

    lora = LoRALinearLayer(base, rank=2, alpha=2.0, lora_name="lora")
    loha = LoHaLinearLayer(base, rank=2, alpha=2.0, lora_name="loha")
    lokr = LoKrLinearLayer(base, rank=2, alpha=2.0, lora_name="lokr")
    dora_inner = LoRALinearLayer(base, rank=2, alpha=2.0, lora_name="dora_inner")
    dora = DoRALinearLayer(base, dora_inner)

    composite.add_branch("branch_lora", lora)
    composite.add_branch("branch_loha", loha)
    composite.add_branch("branch_lokr", lokr)
    composite.add_branch("branch_dora", dora)

    assert len(composite) == 4
    assert composite.branch_names == ("branch_lora", "branch_loha", "branch_lokr", "branch_dora")

    x = torch.randn(3, 8)
    out = composite(x)
    assert out.shape == (3, 6)

    # Individual deactivation works
    composite.set_active("branch_loha", False)
    assert not composite.is_active("branch_loha")
    out_partial = composite(x)
    assert out_partial.shape == (3, 6)

    # Detach restores base
    composite.clear_branches()
    composite.detach(parent, 0)
    assert parent[0] is base
