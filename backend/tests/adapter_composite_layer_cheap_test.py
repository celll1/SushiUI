"""``CompositeAdapterLayer``: two adapters over one base module, CPU, ~1 s.

Every architecture refuses adapter stacking today because
``LoRALinearLayer.__init__`` reads ``in_features``/``out_features`` into locals
and never exposes them, so the wrapper cannot wrap a wrapper. This file gates
the wrapper that replaces it: one owned base, an ordered set of named branches,
and add/remove/restrengthen/deactivate without rewrapping anything.

Nothing here wires the composite into an architecture -- adoption is a later
commit per architecture -- so the gates are the mechanism's own invariants plus
the two cross-cutting ones a wrapper class can break silently:

  * the offloaders in ``core.memory_management.block_offloading`` select by
    ``__class__.__name__.endswith("Linear")``, so a delegating ``.weight`` on a
    ``*Linear``-named wrapper enrols the base weight TWICE and the paired swap
    then restores the outgoing block's weights;
  * the INT8 gates refuse conversion while wrappers are present, and an
    unrecognised wrapper lets the quantizer cast the adapter's own branches.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_composite_layer_cheap_test.py -v
"""

import os
import sys
from types import SimpleNamespace

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from core.adapters import (  # noqa: E402
    CompositeAdapterLayer,
    LoRALinearLayer,
    MiniMaxH3LoRALinearLayer,
)

D_IN, D_OUT = 6, 5
RANK, ALPHA = 3, 6.0  # alpha != rank on purpose: the scale must be 2.0, not 1.0
STRENGTH = 0.7  # non-dyadic on purpose: reassociating the scale changes the bits


def _base(dtype=torch.float32, seed=7):
    generator = torch.Generator().manual_seed(seed)
    linear = nn.Linear(D_IN, D_OUT)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator) * 0.1)
        linear.bias.copy_(torch.randn(linear.bias.shape, generator=generator) * 0.1)
    return linear.to(dtype)


def _randomise(layer, seed):
    """``lora_up`` initialises to zeros; without this every forward comparison is
    vacuous and a transposed round trip still passes."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for weight in (layer.lora_down.weight, layer.lora_up.weight):
            weight.copy_(torch.randn(weight.shape, generator=generator) * 0.3)
    return layer


def _lora(base, seed, cls=LoRALinearLayer, name="branch"):
    return _randomise(cls(base, rank=RANK, alpha=ALPHA, lora_name=name), seed)


def _x(rows=4, dtype=torch.float32, seed=99):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, D_IN, generator=generator).to(dtype)


class _DenseBranch(nn.Module):
    """A third algebra stub: satisfies the branch protocol and nothing else.

    Deliberately NOT a ``LoRALinearLayer`` subclass -- the composite must drive
    branches off ``forward_delta`` alone, and the INT8 gate must recognise the
    composite itself rather than a branch's class name.
    """

    def __init__(self, seed=3):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.delta_weight = nn.Parameter(
            torch.randn(D_OUT, D_IN, generator=generator) * 0.05)

    def forward_delta(self, x):
        return F.linear(x, self.delta_weight)


class _ExtendedTensorBranch(LoRALinearLayer):
    def __init__(self, base):
        super().__init__(base, rank=RANK, alpha=ALPHA, lora_name="extended")
        self.hada_w1_a = nn.Parameter(torch.randn(RANK, D_IN))
        self.dora_scale = nn.Parameter(torch.randn(D_OUT))

    def branch_tensors(self):
        tensors = super().branch_tensors()
        tensors.update({
            "hada_w1_a": self.hada_w1_a,
            "dora_scale": self.dora_scale,
            "dora_scale_alias": self.dora_scale,
        })
        return tensors


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_q = _base()
        self.attn_k = _base()


# ---------------------------------------------------------------- numerics


def test_single_branch_is_bit_identical_to_the_lora_wrapper():
    base = _base()
    x = _x()

    reference = _lora(base, seed=11)
    reference.scale = (ALPHA / RANK) * STRENGTH  # exactly what the loaders write

    composite = CompositeAdapterLayer(base)
    composite.add_branch("a", _lora(base, seed=11), strength=STRENGTH)

    assert composite.get_branch("a").scale == reference.scale
    assert torch.equal(reference(x), composite(x))  # atol=0.0


@pytest.mark.parametrize("cls", [LoRALinearLayer, MiniMaxH3LoRALinearLayer])
def test_forward_delta_is_the_second_term_of_forward(cls):
    """Both algebras' ``forward_delta`` must not drift from their ``forward``."""
    base = _base()
    x = _x()
    layer = _lora(base, seed=12, cls=cls)

    assert torch.equal(layer(x), base(x) + layer.forward_delta(x))


def test_two_branches_sum_against_the_analytic_delta():
    base = _base()
    x = _x()
    first, second = _lora(base, seed=21), _lora(base, seed=22)

    composite = CompositeAdapterLayer(base)
    composite.add_branch("first", first, strength=STRENGTH)
    composite.add_branch("second", second, strength=0.5)

    def delta(layer, strength):
        return (ALPHA / RANK) * strength * (
            x @ layer.lora_down.weight.T @ layer.lora_up.weight.T)

    expected = base(x) + delta(first, STRENGTH) + delta(second, 0.5)
    assert torch.allclose(composite(x), expected, atol=1e-6, rtol=0)

    single = CompositeAdapterLayer(base)
    single.add_branch("first", first, strength=STRENGTH)
    assert not torch.allclose(composite(x), single(x))  # the second branch bites


def test_two_branch_order_independence_is_exact_three_is_associative():
    base = _base()
    x = _x()
    a, b, c = _lora(base, seed=31), _lora(base, seed=32), _lora(base, seed=33)

    def build(order):
        composite = CompositeAdapterLayer(base)
        for name, branch in order:
            composite.add_branch(name, branch)
        return composite

    # Deltas are summed before the base is added once, and fp addition COMMUTES,
    # so two branches are order-independent bit for bit.
    assert torch.equal(build([("a", a), ("b", b)])(x),
                       build([("b", b), ("a", a)])(x))

    # Three or more only associate, hence a tolerance rather than equality:
    # measured 1.19e-07 max absolute difference on outputs of magnitude 4.75
    # (one fp32 ULP), stated at 1e-06.
    assert torch.allclose(build([("a", a), ("b", b), ("c", c)])(x),
                          build([("c", c), ("b", b), ("a", a)])(x),
                          atol=1e-6, rtol=0)


def test_deactivating_and_restrengthening_do_not_rewrap():
    parent = _Block()
    base = parent.attn_q
    x = _x()

    composite = CompositeAdapterLayer.attach(parent, "attn_q")
    composite.add_branch("a", _lora(base, seed=41), strength=1.0)
    composite.add_branch("b", _lora(base, seed=42), strength=1.0)
    identity = id(parent.attn_q)

    only_a = CompositeAdapterLayer(base)
    only_a.add_branch("a", composite.get_branch("a"))

    composite.set_active("b", False)
    assert torch.equal(composite(x), only_a(x))
    assert id(parent.attn_q) == identity

    composite.set_active("b", True)
    composite.set_strength("a", 0.0)
    assert composite.get_branch("a").scale == 0.0
    assert composite.get_strength("a") == 0.0
    assert id(parent.attn_q) == identity
    assert composite.branch_names == ("a", "b")


def test_zero_active_branches_returns_the_base_output_unchanged():
    base = _base()
    x = _x()
    composite = CompositeAdapterLayer(base)
    composite.add_branch("a", _lora(base, seed=51))
    composite.set_active("a", False)

    assert torch.equal(composite(x), base(x))


def test_both_shipped_algebras_work_as_branches():
    """The MiniMax-H3 algebra exists because its forward runs WITHOUT autocast.

    Under exactly that condition the stock branch raises and the H3 branch does
    not, which is why the composite dispatches on ``forward_delta`` and never on
    a branch's class.
    """
    base = _base(dtype=torch.bfloat16)
    x = _x(dtype=torch.bfloat16)

    h3 = CompositeAdapterLayer(base)
    h3.add_branch("h3", _lora(base, seed=61, cls=MiniMaxH3LoRALinearLayer))
    assert torch.isfinite(h3(x)).all()
    assert not torch.equal(h3(x), base(x))

    stock = CompositeAdapterLayer(base)
    stock.add_branch("stock", _lora(base, seed=61))
    with pytest.raises(RuntimeError):
        stock(x)

    mixed = CompositeAdapterLayer(_base())
    mixed.add_branch("stock", _lora(mixed.base_module, seed=62))
    mixed.add_branch("h3", _lora(mixed.base_module, seed=63,
                                 cls=MiniMaxH3LoRALinearLayer))
    mixed.add_branch("dense", _DenseBranch())
    assert len(mixed) == 3


def test_trainable_parameters_derive_from_branch_tensors_without_duplicates():
    branch = _ExtendedTensorBranch(_base())

    parameters = list(branch.trainable_parameters())

    assert parameters == [
        branch.lora_down.weight,
        branch.lora_up.weight,
        branch.hada_w1_a,
        branch.dora_scale,
    ]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)


# ---------------------------------------------------------------- structure


def test_in_and_out_features_are_exposed_from_the_owned_base():
    """The absence of these two on ``LoRALinearLayer`` IS the stacking defect."""
    base = _base()
    composite = CompositeAdapterLayer(base)

    assert composite.in_features == base.in_features
    assert composite.out_features == base.out_features
    assert composite.weight is base.weight
    assert composite.bias is base.bias
    assert composite.base_module is base

    with pytest.raises(AttributeError):
        LoRALinearLayer(_lora(base, seed=71), rank=RANK, alpha=ALPHA, lora_name="x")


def test_attach_is_idempotent_and_add_remove_keeps_the_same_wrapper():
    parent = _Block()
    base = parent.attn_q

    composite = CompositeAdapterLayer.attach(parent, "attn_q")
    again = CompositeAdapterLayer.attach(parent, "attn_q")
    assert again is composite
    assert composite.original_module is base

    composite.add_branch("a", _lora(base, seed=81))
    composite.add_branch("b", _lora(base, seed=82))
    assert composite.branch_names == ("a", "b")

    removed = composite.remove_branch("a")
    assert parent.attn_q is composite
    assert composite.branch_names == ("b",)
    assert not composite.has_branch("a")
    assert removed is not None
    assert composite.original_module is base


def test_restore_returns_the_original_object_by_identity():
    parent = _Block()
    base = parent.attn_q
    composite = CompositeAdapterLayer.attach(parent, "attn_q")
    composite.add_branch("a", _lora(base, seed=91))
    composite.add_branch("b", _lora(base, seed=92))

    composite.clear_branches()
    assert composite.branch_names == ()
    assert composite.original_module is base

    returned = composite.detach(parent, "attn_q")
    assert returned is base
    assert parent.attn_q is base
    assert not any(isinstance(m, CompositeAdapterLayer) for m in parent.modules())


def test_integer_parent_slots_work():
    """Anima's adaln_modulation_* and llm_adapter targets are Sequential indices."""
    parent = nn.Sequential(_base(), nn.GELU(), _base())
    base = parent[2]
    x = _x()

    composite = CompositeAdapterLayer.attach(parent, 2)
    assert parent[2] is composite
    composite.add_branch("a", _lora(base, seed=101), strength=STRENGTH)
    assert not torch.equal(composite(x), base(x))

    assert composite.detach(parent, 2) is base
    assert parent[2] is base


def test_a_branch_built_against_a_different_base_is_refused():
    """The stale-module splice, refused at add time instead of at forward time."""
    base, other = _base(), _base(seed=8)
    composite = CompositeAdapterLayer(base)

    with pytest.raises(ValueError):
        composite.add_branch("stale", _lora(other, seed=111))

    with pytest.raises(TypeError):
        composite.add_branch("not-a-branch", nn.Linear(D_IN, D_OUT))

    composite.add_branch("a", _lora(base, seed=112))
    with pytest.raises(ValueError):
        composite.add_branch("a", _lora(base, seed=113))

    composite.add_branch("dense", _DenseBranch())
    with pytest.raises(TypeError):
        composite.set_strength("dense", 0.5)  # no set_adapter_strength: refuse, do not improvise


# ------------------------------------------------- block-offload name gate


def _linear_selected(block):
    from core.memory_management.block_offloading import TransformerBlockOffloader
    return TransformerBlockOffloader._linear_weight_modules(block)


def test_the_composite_is_not_selected_by_the_offloaders_linear_name_test():
    from core.memory_management.block_offloading import linear_weight_dtypes

    assert not CompositeAdapterLayer.__name__.endswith("Linear")

    block = _Block()
    base = block.attn_q
    composite = CompositeAdapterLayer.attach(block, "attn_q")
    composite.add_branch("a", _lora(base, seed=121))

    paths = linear_weight_dtypes(block)
    assert "attn_q" not in paths
    assert "attn_q.original_module" in paths

    pointers = [m.weight.data_ptr() for m in _linear_selected(block)]
    assert len(pointers) == len(set(pointers)), "a weight was enrolled twice"


def test_a_linear_named_composite_would_double_enrol_the_base_weight():
    """Why the name ends in ``Layer``: this subclass differs ONLY in its name."""

    class _CompositeAdapterLinear(CompositeAdapterLayer):
        pass

    block = _Block()
    base = block.attn_q
    composite = _CompositeAdapterLinear.attach(block, "attn_q")
    composite.add_branch("a", _lora(base, seed=131))

    pointers = [m.weight.data_ptr() for m in _linear_selected(block)]
    assert pointers.count(base.weight.data_ptr()) == 2


# ------------------------------------------------------------ INT8 gates


def test_lora_wrapped_count_recognises_a_composite_with_a_non_lora_branch():
    from core.models.common.int8_runtime_quantize import lora_wrapped_count

    model = nn.Sequential(_base(), _base())
    composite = CompositeAdapterLayer.attach(model, 0)
    composite.add_branch("dense", _DenseBranch())

    # The old flat ``type(m).__name__ == "LoRALinearLayer"`` test returns 0 here.
    assert lora_wrapped_count(model) == 1

    composite.add_branch("a", _lora(composite.base_module, seed=141))
    composite.add_branch("b", _lora(composite.base_module, seed=142))
    # Roots, not branches: one hidden Linear slot is one refusal, not three.
    assert lora_wrapped_count(model) == 1


def test_lora_wrapped_count_is_unchanged_for_plain_wrappers():
    from core.models.common.int8_runtime_quantize import lora_wrapped_count

    model = _Block()
    model.attn_q = _lora(model.attn_q, seed=151)
    model.attn_k = _lora(model.attn_k, seed=152)
    assert lora_wrapped_count(model) == 2
    assert lora_wrapped_count(nn.Sequential(_base(), _base())) == 0


def test_the_int8_converter_refuses_a_composite_wrapped_model():
    from core.models.common.int8_runtime_quantize import (
        LoraWrappedError, quantize_linears_in_place)

    model = nn.Sequential(_base(), _base())
    composite = CompositeAdapterLayer.attach(model, 0)
    composite.add_branch("dense", _DenseBranch())

    with pytest.raises(LoraWrappedError):
        quantize_linears_in_place(model, arch="anima", label="test transformer")


def test_the_vram_preflight_refuses_a_composite_wrapped_model(monkeypatch):
    """The pre-flight refuses over the WHOLE component set before the first layer
    of the first component is touched, so it must see the composite too."""
    import api.generation_status as status
    from core.vram_optimization import apply_runtime_int8_quantization

    recorded = []
    monkeypatch.setattr(status, "add_warning",
                        lambda message, code=None, **kw: recorded.append((code, message)))

    model = nn.Sequential(_base(), _base())
    composite = CompositeAdapterLayer.attach(model, 0)
    composite.add_branch("dense", _DenseBranch())

    returned, converted = apply_runtime_int8_quantization(
        SimpleNamespace(), model, "anima", "int8", label="test transformer")

    assert converted is False
    assert returned is model
    # The LoRA refusal specifically, not whichever other refusal happens to fire.
    assert [code for code, _m in recorded] == ["quantization_fallback"]
    assert "because LoRAs are loaded" in recorded[0][1]
    assert isinstance(model[0], CompositeAdapterLayer)
    assert type(model[1]) is nn.Linear  # the unwrapped sibling was not converted


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
