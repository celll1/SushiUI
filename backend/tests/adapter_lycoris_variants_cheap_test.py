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


# -- export -> from_tensors -> load, bit-identically -------------------------
# Two constructor arguments that no checkpoint stores: ``alpha`` (the layer
# built one and used it) and LoKr's ``factor``. Both are silent -- shapes match
# or the scale is merely wrong -- so each has a row here.

_LOAD_D_IN, _LOAD_D_OUT = 12, 12


def _randomised(layer):
    """Every factor non-zero: the shipped init leaves one at zero, and a
    zero delta makes any round-trip assertion pass."""
    with torch.no_grad():
        for tensor in layer.branch_tensors().values():
            if isinstance(tensor, nn.Parameter):
                tensor.copy_(torch.randn(tensor.shape))
    return layer


def _base_linear(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(_LOAD_D_IN, _LOAD_D_OUT)


@pytest.mark.parametrize("cls", [LoHaLinearLayer, LoKrLinearLayer])
def test_a_round_trip_with_alpha_not_equal_to_rank_is_bit_identical(cls):
    """``branch_tensors()["alpha"]`` is a FRESHLY BUILT tensor, so the old
    ``load_tensors`` copied the checkpoint's alpha into a throwaway and left the
    layer at the constructor's. MEASURED on this fixture: a file saved at
    alpha=8, rank=4 loaded into a layer built at alpha=rank applied a delta
    exactly 0.5x the correct one -- shapes fine, image wrong.
    """
    base = _base_linear()
    saved = _randomised(cls(base, rank=4, alpha=8.0, lora_name="src"))
    expected = saved.compute_delta_weight().detach().clone()
    tensors = saved.export_tensors()

    rebuilt = cls.from_tensors(base, tensors, lora_name="dst")

    assert rebuilt.alpha == 8.0 and rebuilt.scale == 2.0
    assert torch.equal(rebuilt.compute_delta_weight(), expected)


@pytest.mark.parametrize("factor,factors", [(2, ((2, 6), (2, 6))),
                                            (-1, ((3, 4), (3, 4)))])
@pytest.mark.parametrize("rank", [0, 4])
def test_a_lokr_written_with_a_foreign_factor_rebuilds_from_its_tensors(
        factor, factors, rank):
    """``factor`` is not stored in a checkpoint, so a loader that re-derived the
    split from ``factorization(out_features, -1)`` would allocate other shapes
    and ``copy_`` would raise."""
    base = _base_linear()
    saved = _randomised(LoKrLinearLayer(base, rank=rank, alpha=8.0,
                                        lora_name="src", factor=factor))
    assert saved.factors == factors
    expected = saved.compute_delta_weight().detach().clone()

    rebuilt = LoKrLinearLayer.from_tensors(base, saved.export_tensors(),
                                           lora_name="dst")

    assert rebuilt.factors == factors
    assert rebuilt.rank == rank and rebuilt.scale == saved.scale
    assert torch.equal(rebuilt.compute_delta_weight(), expected)


def test_a_lokr_with_both_operands_factored_rebuilds():
    base = _base_linear()
    saved = _randomised(LoKrLinearLayer(base, rank=2, alpha=8.0, lora_name="src",
                                        decompose_both=True))
    rebuilt = LoKrLinearLayer.from_tensors(base, saved.export_tensors())
    assert rebuilt.decompose_both and rebuilt.rank == 2
    assert torch.equal(rebuilt.compute_delta_weight(),
                       saved.compute_delta_weight())


def test_a_lora_round_trip_still_takes_its_alpha_from_the_tensors_or_the_caller():
    """``LoRALinearLayer`` deliberately does not export ``alpha``, so its
    precedence lives in ``from_tensors`` alone."""
    base = _base_linear()
    saved = _randomised(LoRALinearLayer(base, rank=4, alpha=8.0, lora_name="src"))
    tensors = saved.export_tensors()
    assert "alpha" not in tensors

    assert LoRALinearLayer.from_tensors(base, tensors).alpha == 4.0
    assert LoRALinearLayer.from_tensors(base, tensors, alpha=8.0).alpha == 8.0
    with_tensor = dict(tensors, alpha=torch.tensor(8.0))
    rebuilt = LoRALinearLayer.from_tensors(base, with_tensor, alpha=1.0)
    assert rebuilt.alpha == 8.0
    assert torch.equal(rebuilt.compute_delta_weight(), saved.compute_delta_weight())


@pytest.mark.parametrize("cls", [LoHaLinearLayer, LoKrLinearLayer])
def test_a_use_scalar_layer_cannot_be_built_from_a_checkpoint(cls):
    """No file carries ``scalar``, and a layer built from one would multiply
    the whole delta by zero (see ``LoHaLinearLayer``)."""
    base = _base_linear()
    saved = _randomised(cls(base, rank=4, alpha=8.0, lora_name="src"))
    with pytest.raises(ValueError, match="scalar"):
        cls.from_tensors(base, saved.export_tensors(), use_scalar=True)
    assert cls.from_tensors(base, saved.export_tensors()).scalar is None


@pytest.mark.parametrize("cls,name,shape", [
    (LoHaLinearLayer, "hada_w1_b", (4, _LOAD_D_IN + 1)),
    (LoHaLinearLayer, "hada_w2_a", (_LOAD_D_OUT + 1, 4)),
    (LoKrLinearLayer, "lokr_w1", (5, 3)),
])
def test_geometry_that_disagrees_with_the_base_is_refused(cls, name, shape):
    base = _base_linear()
    tensors = _randomised(cls(base, rank=4, alpha=8.0, lora_name="src")).export_tensors()
    tensors[name] = torch.randn(shape)
    with pytest.raises(ValueError):
        cls.from_tensors(base, tensors)


def test_loading_does_not_draw_from_the_global_rng():
    """``from_tensors`` skips the kaiming init of factors it is about to
    overwrite; the draw would otherwise advance the generation seed once per
    wrapped target."""
    base = _base_linear()
    tensors = _randomised(
        LoHaLinearLayer(base, rank=4, alpha=8.0, lora_name="src")).export_tensors()

    torch.manual_seed(1234)
    before = torch.randn(3)
    torch.manual_seed(1234)
    LoHaLinearLayer.from_tensors(base, tensors)
    assert torch.equal(torch.randn(3), before)


@pytest.mark.parametrize("cls", [LoHaLinearLayer, LoKrLinearLayer])
def test_load_tensors_applies_the_checkpoints_alpha_to_the_layer(cls):
    """The training-resume path, where the layer already exists: ``alpha`` is a
    freshly built tensor in ``branch_tensors()``, so copying into it mutated a
    throwaway and left ``scale`` at the constructor's."""
    base = _base_linear()
    saved = _randomised(cls(base, rank=4, alpha=8.0, lora_name="src"))
    expected = saved.compute_delta_weight().detach().clone()

    stale = cls(base, rank=4, alpha=4.0, lora_name="dst")   # alpha == rank
    stale.load_tensors(saved.export_tensors())

    assert stale.alpha == 8.0 and stale.scale == 2.0
    assert torch.equal(stale.compute_delta_weight(), expected)


def test_a_lora_layer_ignores_alpha_on_load_exactly_as_before():
    """``LoRALinearLayer`` has no ``alpha`` among its branch tensors, so the
    spec-constant machinery must leave its load path untouched."""
    base = _base_linear()
    saved = _randomised(LoRALinearLayer(base, rank=4, alpha=8.0, lora_name="src"))
    layer = LoRALinearLayer(base, rank=4, alpha=4.0, lora_name="dst")
    layer.load_tensors(dict(saved.export_tensors(), alpha=torch.tensor(8.0)))
    assert layer.alpha == 4.0 and layer.scale == 1.0
    assert layer.spec_constants() == ()
