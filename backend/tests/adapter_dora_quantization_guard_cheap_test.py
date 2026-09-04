"""The legacy FP8 quantizers must not cast a weight-decomposed adapter's base.

``AdapterSession`` refuses a DoRA over a weight-only quantized base at install
time, but on Z-Image the legacy FP8 pass runs LATER: ``_load_lora_zimage`` is
line 700 of the generate function and ``move_zimage_transformer_to_gpu`` is line
812. ``_quantize_transformer`` deep-copies the tree and casts every
``nn.Linear`` weight, which over a wrapped tree includes the DoRA wrapper's own
BASE -- the weight whose direction and norm the magnitude epilogue divides by.
Nothing in the session can see that happen, so the guard lives in the quantizer.

Same precedence as ``_lens_quantization_with_lora``: the adapter wins and the
request is warned. Keyed on the DECOMPOSED family alone, so LoRA/LoHa/LoKr
quantization behaviour is byte-unchanged -- which the last row asserts.

CPU only; nothing here loads a model.

Run with:
    venv/Scripts/python.exe -m pytest \
        backend/tests/adapter_dora_quantization_guard_cheap_test.py -v
"""

import os
import sys

import pytest
import torch
from torch import nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.adapters import (  # noqa: E402
    CompositeAdapterLayer, DoRALinearLayer, new_adapter_branch,
)
from core.vram_optimization import (  # noqa: E402
    _anima_quantize_fp8, _decomposed_adapter_branches, _quantize_transformer,
)

D = 8
FP8 = "fp8_e4m3fn"


class _Tree(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(D, D, bias=False)


def _wrapped(algorithm="lora", weight_decompose=True):
    """One target covered by a composite holding one branch, as a loader leaves
    it -- so the walk below sees the real shape, not a bare layer."""
    tree = _Tree()
    composite = CompositeAdapterLayer.attach(tree.attn, "to_q")
    branch = new_adapter_branch(algorithm, composite.original_module, rank=2,
                                alpha=4.0, name="to_q", dtype=torch.float32,
                                weight_decompose=weight_decompose)
    composite.add_branch("0:test", branch, strength=1.0)
    return tree


@pytest.fixture
def warned(monkeypatch):
    seen = []
    import core.vram_optimization as vram

    monkeypatch.setattr(vram, "_add_generation_warning",
                        lambda message, code=None: seen.append((code, message)))
    return seen


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_the_probe_counts_only_decomposed_branches(algorithm):
    assert _decomposed_adapter_branches(_wrapped(algorithm)) == 1
    assert _decomposed_adapter_branches(
        _wrapped(algorithm, weight_decompose=False)) == 0
    assert _decomposed_adapter_branches(_Tree()) == 0


@pytest.mark.parametrize("quantize", [_quantize_transformer, _anima_quantize_fp8])
def test_an_fp8_pass_over_a_dora_branch_is_dropped_and_warned(quantize, warned):
    tree = _wrapped()
    args = (tree, FP8) if quantize is _quantize_transformer else (tree, FP8, "T")
    result = quantize(*args)

    assert result is tree, "the tree was replaced, so it was quantized anyway"
    base = tree.attn.to_q.original_module
    assert base.weight.dtype == torch.float32
    assert [code for code, _m in warned] == ["quantization_fallback"]
    assert "weight-decomposed" in warned[0][1]


def test_the_base_weight_the_epilogue_divides_by_is_what_is_at_stake():
    """WHY the guard exists, as a number rather than an assertion about types:
    casting the base to fp8 moves the DoRA delta, because the epilogue reads
    ``W_base`` twice -- once inside ``||W_base + delta||`` and once in the
    subtraction."""
    tree = _wrapped()
    branch = tree.attn.to_q.get_branch("0:test")
    assert isinstance(branch, DoRALinearLayer)
    with torch.no_grad():
        branch.branch.lora_up.weight.normal_()
        branch.dora_scale.mul_(1.3)
    before = branch.compute_delta_weight().clone()

    base = branch.original_module
    with torch.no_grad():
        base.weight.data = base.weight.data.to(torch.float8_e4m3fn)
    after = branch.compute_delta_weight()

    moved = float((after - before).abs().max()) / float(before.abs().max())
    assert moved > 1e-3, moved


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
@pytest.mark.parametrize("quantize", [_quantize_transformer, _anima_quantize_fp8])
def test_an_additive_branch_still_quantizes_exactly_as_before(algorithm,
                                                              quantize, warned):
    """The guard must not become "no adapters under FP8": the additive families
    never read the base weight, and their behaviour here is unchanged."""
    tree = _wrapped(algorithm, weight_decompose=False)
    args = (tree, FP8) if quantize is _quantize_transformer else (tree, FP8, "T")
    result = quantize(*args)

    assert result is not tree, "the additive request was dropped too"
    assert [code for code, _m in warned] == []
