"""Correctness gates for the adapter algebras against the fp32 oracle. CPU, ~10 s.

The layers were written without a reference to check them against.
``core.adapters.reference`` is that reference -- an independent fp32
implementation sharing no code with them -- and every case compares a forward,
its input gradient and every parameter gradient against it.

Covers the design doc's correctness gates a CPU test can reach: fp32/fp16/bf16,
strengths 0 / 1 / fractional / >1 / negative, runtime versus merge including
``alpha != rank``, and gradient checkpointing on and off.

Every algebra zero-initialises one factor, so its initial delta is zero and
every equivalence assertion here would pass however wrong the algebra is.
``_build`` randomises that factor and ``_assert_delta_is_exercised`` runs first.
"""

import ast
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from core.adapters import (  # noqa: E402
    TUCKER_TENSOR_NAMES,
    DoRALinearLayer,
    LoHaLinearLayer,
    LoKrLinearLayer,
    LoRALinearLayer,
    factorization,
)
from core.adapters.execution.probe import ORACLE_TOLERANCE  # noqa: E402
from core.adapters.reference import (  # noqa: E402
    adapter_delta_weight,
    dora_effective_delta_weight,
)

#: 10 = 2*5 and 12 = 3*4, so LoKr factors BOTH sides non-trivially; a square or
#: prime shape would collapse its Kronecker structure and hide an operand swap.
D_IN, D_OUT = 12, 10
#: ...which is exactly why the DoRA magnitude axis needs a SQUARE case of its
#: own: that is where a wrong reshape does not raise.
SQUARE = (D_IN, D_IN)
RANK, ALPHA = 4, 8.0          # alpha != rank throughout: the scale must be 2.0
STRENGTHS = (0.0, 1.0, 0.35, 1.8, -0.6)

#: Worst relative error measured with this harness over 40 seeds x 3
#: (rank, alpha) pairs x 5 strengths x all six algebras, taken over the forward,
#: forward_delta, the input gradient and every parameter gradient: 8.2e-7 fp32,
#: 2.7e-3 fp16, 2.9e-2 bf16 -- 3 to 7 ulps of each dtype (eps 1.2e-7 / 9.8e-4 /
#: 7.8e-3), every one of the three a LoKr-under-DoRA factor gradient. Headroom
#: is therefore 2.4x / 2.3x / 1.7x, not the uniform 3x an earlier and narrower
#: sweep suggested. The seeds the tests themselves use are fixed, so the gate is
#: deterministic; above these values is a real disagreement, not rounding.
#:
#: DEFINED in ``core.adapters.execution.probe``, because the phase-4 backend
#: probe admits a candidate against the same bar and a second copy of these
#: numbers would drift. Pinned below, so loosening them there fails HERE.
TOLERANCE = ORACLE_TOLERANCE
assert TOLERANCE == {torch.float32: 2e-6, torch.float16: 6e-3,
                     torch.bfloat16: 5e-2}, (
    "the measured oracle tolerances moved; re-measure the sweep in this file's "
    "docstring before changing them")
DTYPES = tuple(TOLERANCE)

#: Randomisation applied to the factor that starts as a no-op. Sized by the
#: SMALLEST delta of the eleven rows (LoHa's, a Hadamard product of two small
#: factors), so that even at strength 0.35 the branch clears ``MIN_MOVE``.
FACTOR_STD = 2.0

#: Relative perturbation of ``dora_scale`` away from the base's own norms, which
#: is where the epilogue is an identity. 0.05 left the column-magnitude DoHa row
#: at 0.095 -- just under ``MIN_MOVE``.
DORA_JITTER = 0.12

#: Fake-assertion guard: the branch must move the base output by at least this
#: fraction, so a comparison cannot pass with the branch effectively absent.
#: Against the tolerances above that is a factor of 50,000 (fp32) and 17 (fp16),
#: but only 2 in bf16. MEASURED resolution, by scaling every delta by 1+e: bf16
#: catches e=5% on all six algebras, 2% on one, 1% on none, while fp16 catches
#: 1% on all six. The bf16 arm is therefore a "the low-dtype path does not
#: structurally diverge" check rather than a numerical gate -- which still
#: catches every structural defect (operand swap, factor crossover, doubled
#: scale, dropped strength, wrong norm axis, permuted dora_scale), each of those
#: being a >=2x error.
MIN_MOVE = 0.10


#: The factors each STORED FORM must export, independent of the layer classes:
#: a branch that silently stops exporting one has to fail, and a set derived
#: from what the layer exported could not notice.
FACTORS = {
    "lora": ("lora_down.weight", "lora_up.weight"),
    "loha": ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"),
    "loha_scalar": ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "scalar"),
    "lokr": ("lokr_w1", "lokr_w2_a", "lokr_w2_b"),
    "lokr_full": ("lokr_w1", "lokr_w2"),
    "lokr_both": ("lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"),
    "lokr_both_scalar": ("lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b",
                         "scalar"),
}


@dataclass(frozen=True)
class Algebra:
    name: str
    algorithm: str                # what the oracle dispatches on
    weight_decompose: bool
    zero_init: Tuple[str, ...]    # factors that start as a no-op
    variant: str = ""             # FACTORS key; defaults to ``algorithm``
    #: Constructor arguments past rank/alpha, as pairs so the row stays hashable.
    options: Tuple[Tuple[str, Any], ...] = ()
    rank: Optional[int] = None    # forced rank; the full LoKr form is rank 0

    @property
    def parameter_names(self) -> FrozenSet[str]:
        names = FACTORS[self.variant or self.algorithm]
        if not self.weight_decompose:
            return frozenset(names)
        return frozenset([f"branch.{name}" for name in names] + ["dora_scale"])


#: Table-driven so Phase 3's weight-decomposed pairs are rows, not copies. The
#: decomposed rows exercise the epilogue only; the capability matrix in
#: ``core.training.arch.base_arch`` still refuses every one of them.
#:
#: The last five rows are forms that exist upstream and that this repo either
#: mis-scaled or did not model: the full/full LoKr whose scale is 1 rather than
#: the stored ``alpha``, ``decompose_both``, and the trained ``scalar``.
ALGEBRAS = (
    Algebra("lora", "lora", False, ("lora_up.weight",)),
    Algebra("loha", "loha", False, ("hada_w2_a",)),
    Algebra("lokr", "lokr", False, ("lokr_w2_a",)),
    Algebra("dora", "lora", True, ("lora_up.weight",)),
    Algebra("doha", "loha", True, ("hada_w2_a",)),
    Algebra("dokr", "lokr", True, ("lokr_w2_a",)),
    Algebra("loha_scalar", "loha", False, ("scalar",), "loha_scalar",
            (("use_scalar", True),)),
    Algebra("lokr_full", "lokr", False, ("lokr_w2",), "lokr_full", (), 0),
    Algebra("lokr_both", "lokr", False, ("lokr_w2_a",), "lokr_both",
            (("decompose_both", True),)),
    Algebra("lokr_both_scalar", "lokr", False, ("scalar",), "lokr_both_scalar",
            (("decompose_both", True), ("use_scalar", True))),
    Algebra("dokr_full", "lokr", True, ("lokr_w2",), "lokr_full", (), 0),
)
BY_NAME = {algebra.name: algebra for algebra in ALGEBRAS}
DECOMPOSED = [a for a in ALGEBRAS if a.weight_decompose]
_LAYER_CLASS = {"lora": LoRALinearLayer, "loha": LoHaLinearLayer,
                "lokr": LoKrLinearLayer}


def _base_linear(dtype: torch.dtype, generator: torch.Generator,
                 shape: Tuple[int, int] = (D_IN, D_OUT)) -> nn.Linear:
    base = nn.Linear(*shape)
    with torch.no_grad():
        base.weight.copy_(torch.randn(base.weight.shape, generator=generator) * 0.2)
        base.bias.copy_(torch.randn(base.bias.shape, generator=generator) * 0.1)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base.to(dtype)


def _layer(algebra: Algebra, base: nn.Linear, rank: int = RANK,
           alpha: float = ALPHA, dtype: torch.dtype = torch.float32):
    return _LAYER_CLASS[algebra.algorithm](
        base, rank=rank if algebra.rank is None else algebra.rank, alpha=alpha,
        lora_name="gate", lora_dtype=dtype, **dict(algebra.options))


def _dora_scale(base: nn.Linear, axis: str,
                generator: torch.Generator) -> torch.Tensor:
    """A PERTURBED magnitude vector in the requested orientation.

    Left at the base's own norms, ``W_adapter == W0`` and the delta is
    near-zero: the same trap as a zero factor. ``"row"`` is upstream's
    ``wd_on_out=True`` default, ``"column"`` its ``(1, in)`` alternative.
    """
    weight = base.weight.detach().to(torch.float32)
    if axis == "row":
        scale = torch.linalg.vector_norm(weight, ord=2, dim=1)
    else:
        scale = torch.linalg.vector_norm(weight, ord=2, dim=0, keepdim=True)
    jitter = 1.0 + DORA_JITTER * torch.randn(scale.shape, generator=generator)
    return (scale * jitter).to(base.weight.dtype)


def _build(algebra: Algebra, dtype: torch.dtype = torch.float32, seed: int = 11,
           rank: int = RANK, alpha: float = ALPHA,
           shape: Tuple[int, int] = (D_IN, D_OUT), dora_axis: str = "row"):
    """A base and an adapter layer whose delta is NOT zero.

    Randomising ``algebra.zero_init`` is the whole point: with the shipped
    initialisation the delta is exactly zero, and then every comparison in this
    file passes against any oracle at all.
    """
    generator = torch.Generator().manual_seed(seed)
    # The layer classes' own kaiming init draws from the GLOBAL rng, so two
    # "identical" builds are only identical if this is seeded too.
    torch.manual_seed(seed)
    base = _base_linear(dtype, generator, shape)
    layer = _layer(algebra, base, rank, alpha, dtype)
    tensors = layer.branch_tensors()
    with torch.no_grad():
        for name in algebra.zero_init:
            tensor = tensors[name]
            drawn = torch.randn(tensor.shape, generator=generator) * FACTOR_STD
            if tensor.ndim == 0:
                # A lone scalar can land near zero and make the branch vanish,
                # failing MIN_MOVE for a reason that is not a defect. The floor
                # also covers use_scalar leaving the other factors small.
                drawn = drawn.abs() + FACTOR_STD
            tensor.copy_(drawn.to(tensor.dtype))
    if algebra.weight_decompose:
        layer = DoRALinearLayer(
            base, layer, dora_scale=_dora_scale(base, dora_axis, generator))
    return base, layer


def _branch_parameters(layer: nn.Module) -> Dict[str, nn.Parameter]:
    """Every trainable tensor of the branch, base excluded.

    ``branch_tensors()`` also yields a non-Parameter ``alpha`` for LoHa/LoKr
    (LyCORIS carries it as a tensor), hence the isinstance filter.
    """
    if isinstance(layer, DoRALinearLayer):
        found = {f"branch.{name}": tensor
                 for name, tensor in layer.branch.branch_tensors().items()
                 if isinstance(tensor, nn.Parameter)}
        found["dora_scale"] = layer.dora_scale
        return found
    return {name: tensor for name, tensor in layer.branch_tensors().items()
            if isinstance(tensor, nn.Parameter)}


def _oracle_delta(algebra: Algebra, tensors: Dict[str, torch.Tensor],
                  base_weight: torch.Tensor, strength: float,
                  rank: int = RANK, alpha: float = ALPHA) -> torch.Tensor:
    if not algebra.weight_decompose:
        return adapter_delta_weight(algebra.algorithm, tensors, rank=rank,
                                    alpha=alpha, strength=strength)
    inner = {name[len("branch."):]: tensor for name, tensor in tensors.items()
             if name.startswith("branch.")}
    unit = adapter_delta_weight(algebra.algorithm, inner, rank=rank, alpha=alpha)
    return dora_effective_delta_weight(base_weight, unit, tensors["dora_scale"],
                                       strength=strength)


def _relative_error(got: torch.Tensor, expected: torch.Tensor) -> float:
    got32, expected32 = got.to(torch.float32), expected.to(torch.float32)
    scale = max(expected32.abs().max().item(), 1e-12)
    return (got32 - expected32).abs().max().item() / scale


def _assert_delta_is_exercised(layer: nn.Module, base: nn.Linear,
                               x: torch.Tensor, label: str) -> float:
    """The branch must actually move the output before any equivalence claim.

    Guards the defect this repo has hit: with the shipped zero-initialised
    factor the delta is exactly zero and every comparison passes for the wrong
    reason.
    """
    with torch.no_grad():
        moved = _relative_error(layer(x), base(x))
    assert moved >= MIN_MOVE, (
        f"{label}: the branch moves the output by {moved:.3g}, below "
        f"MIN_MOVE={MIN_MOVE} -- this comparison would pass with the branch "
        f"absent, so it proves nothing")
    return moved


def _delta_weight(layer: nn.Module) -> torch.Tensor:
    return layer.compute_delta_weight().detach().to(torch.float32)


def _assert_matches_oracle(algebra: Algebra, base: nn.Linear, layer: nn.Module,
                           dtype: torch.dtype, strength: float,
                           rank: int = RANK, alpha: float = ALPHA) -> None:
    """Forward, input gradient and EVERY parameter gradient against the oracle."""
    tolerance = TOLERANCE[dtype]
    d_in, d_out = base.in_features, base.out_features
    layer.set_adapter_strength(strength)

    generator = torch.Generator().manual_seed(5)
    x = (torch.randn(4, d_in, generator=generator) * 0.5).to(dtype).requires_grad_()
    grad_out = (torch.randn(4, d_out, generator=generator) * 0.5).to(dtype)

    if strength != 0.0:
        _assert_delta_is_exercised(layer, base, x.detach(),
                                   f"{algebra.name}/{dtype}/s={strength}")
    out = layer(x)
    (out * grad_out).sum().backward()

    parameters = _branch_parameters(layer)
    # Before the oracle, so a branch that stops exporting a factor fails here
    # rather than as a KeyError inside it.
    assert set(parameters) == algebra.parameter_names, \
        f"{algebra.name}: the branch exports {sorted(parameters)}"
    oracle_leaves = {name: tensor.detach().to(torch.float32).clone().requires_grad_()
                     for name, tensor in parameters.items()}
    base_w = base.weight.detach().to(torch.float32)
    base_b = base.bias.detach().to(torch.float32)
    x32 = x.detach().to(torch.float32).clone().requires_grad_()

    delta = _oracle_delta(algebra, oracle_leaves, base_w, strength, rank, alpha)
    out_ref = F.linear(x32, base_w + delta, base_b)
    (out_ref * grad_out.to(torch.float32)).sum().backward()

    label = f"{algebra.name} {dtype} s={strength}"
    assert _relative_error(out, out_ref) <= tolerance, f"{label}: forward"
    assert _relative_error(x.grad, x32.grad) <= tolerance, f"{label}: input grad"
    # forward_delta is the CompositeAdapterLayer branch protocol, i.e. what
    # every generation backend actually calls; LoRA computes it separately from
    # its own forward, so it needs its own comparison.
    with torch.no_grad():
        composed = base(x.detach()) + layer.forward_delta(x.detach())
    assert _relative_error(composed, out_ref) <= tolerance, f"{label}: forward_delta"
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"{label}: no gradient reached {name}"
        error = _relative_error(parameter.grad, oracle_leaves[name].grad)
        assert error <= tolerance, f"{label}: grad {name} off by {error:.3e}"


@pytest.mark.parametrize("algebra", ALGEBRAS, ids=lambda a: a.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_forward_input_grad_and_every_parameter_grad_match_the_oracle(algebra, dtype):
    for strength in STRENGTHS:
        base, layer = _build(algebra, dtype)
        _assert_matches_oracle(algebra, base, layer, dtype, strength)


@pytest.mark.parametrize("algebra", ALGEBRAS, ids=lambda a: a.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_strength_zero_is_an_exact_identity(algebra, dtype):
    """Not "close to": ``torch.equal`` against the bare base output.

    DoRA is the case a merely-plausible implementation fails: scaling the delta
    and then renormalising leaves ``s = 0`` off the base by the renormalisation
    round trip. Measured on fp32 with the pre-fix code, in the configuration
    ``_build`` produces, that residual is 1.0e-1 RELATIVE -- a 10% error, not a
    rounding nit. It collapses to 2.6e-8 only while ``dora_scale`` still equals
    the base's own row norms, which is exactly the degenerate case ``_build``
    perturbs away from.
    """
    base, layer = _build(algebra, dtype)
    x = (torch.randn(4, D_IN, generator=torch.Generator().manual_seed(3)) * 0.5).to(dtype)

    layer.set_adapter_strength(1.0)
    _assert_delta_is_exercised(layer, base, x, f"{algebra.name}/{dtype}")

    layer.set_adapter_strength(0.0)
    delta = layer.forward_delta(x)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(layer(x), base(x))


@pytest.mark.parametrize("algebra", ALGEBRAS, ids=lambda a: a.name)
def test_negative_strength_reflects_the_positive_one(algebra):
    """``W_eff(s) = W_base + s * (W_adapter - W_base)`` makes ``-s`` the exact
    reflection of ``s`` for the decomposed families too, which upstream DoRA's
    merge-then-interpolate order does not give."""
    base, layer = _build(algebra)
    layer.set_adapter_strength(0.6)
    x = torch.randn(4, D_IN, generator=torch.Generator().manual_seed(9)) * 0.5
    _assert_delta_is_exercised(layer, base, x, algebra.name)
    positive = _delta_weight(layer)
    layer.set_adapter_strength(-0.6)
    assert torch.equal(_delta_weight(layer), -positive)


@pytest.mark.parametrize("algebra", ALGEBRAS, ids=lambda a: a.name)
@pytest.mark.parametrize("rank,alpha,strength", [
    # alpha != rank AND a non-unit strength at once: a scale applied in the
    # wrong place cancels out whenever either of the two is 1.
    (RANK, ALPHA, 0.35),
    (RANK, float(RANK), 1.0),
    (2, 1.0, 1.8),
])
def test_runtime_matches_the_merged_weight(algebra, rank, alpha, strength):
    """Folding the delta into the base must reproduce the runtime forward, and
    subtracting it again must restore the base.

    Near-tautological for five of the six: LoHa, LoKr and DoRA all define
    ``forward`` as ``base(x) + F.linear(x, compute_delta_weight())``, so runtime
    against merged differs only by float associativity. Only the LoRA row runs a
    genuinely different path (``lora_up(lora_down(x))``); the content of this
    test for the rest is the oracle comparison on the delta weight below.
    """
    base, layer = _build(algebra, torch.float32, rank=rank, alpha=alpha)
    layer.set_adapter_strength(strength)
    x = torch.randn(4, D_IN, generator=torch.Generator().manual_seed(9)) * 0.5

    _assert_delta_is_exercised(layer, base, x,
                               f"{algebra.name}/r{rank}/a{alpha}/s{strength}")
    delta = _delta_weight(layer)

    oracle = _oracle_delta(
        algebra,
        {name: tensor.detach().to(torch.float32)
         for name, tensor in _branch_parameters(layer).items()},
        base.weight.detach().to(torch.float32), strength, rank, alpha)
    assert _relative_error(delta, oracle) <= TOLERANCE[torch.float32]

    merged = nn.Linear(D_IN, D_OUT)
    with torch.no_grad():
        merged.weight.copy_(base.weight + delta)
        merged.bias.copy_(base.bias)
    assert _relative_error(layer(x), merged(x)) <= TOLERANCE[torch.float32]

    with torch.no_grad():
        merged.weight.sub_(delta)
    assert _relative_error(merged(x), base(x)) <= TOLERANCE[torch.float32]


@pytest.mark.parametrize("algebra", [a for a in ALGEBRAS if a.weight_decompose],
                         ids=lambda a: a.name)
def test_dora_neutralises_a_strength_folded_into_its_inner_branch(algebra):
    """The epilogue needs the inner branch at UNIT strength.

    Every generation loader folds the request strength into the branch it builds
    (``CompositeAdapterLayer.add_branch``), which for DoRA would put it inside
    ``v`` BEFORE the magnitude epilogue and then apply it again on the
    interpolation. ``DoRALinearLayer.__init__`` resets the branch, so the
    tampered layer must be indistinguishable from the clean one.
    """
    x = torch.randn(4, D_IN, generator=torch.Generator().manual_seed(9)) * 0.5
    base, clean = _build(algebra)
    clean.set_adapter_strength(0.5)
    _assert_delta_is_exercised(clean, base, x, algebra.name)

    tampered_base, rebuilt = _build(algebra)          # same seed, same tensors
    rebuilt.branch.set_adapter_strength(2.5)          # what a loader would do
    tampered = DoRALinearLayer(tampered_base, rebuilt.branch,
                               dora_scale=clean.dora_scale.detach())
    tampered.set_adapter_strength(0.5)

    assert torch.equal(tampered(x), clean(x))


@pytest.mark.parametrize("algebra", ALGEBRAS, ids=lambda a: a.name)
def test_gradient_checkpointing_reproduces_every_parameter_gradient(algebra):
    """Two adapter layers in series, recomputed in backward, must land the same
    gradients on every factor as the plain run."""
    x = torch.randn(4, D_IN, generator=torch.Generator().manual_seed(4)) * 0.5

    def stack(seed):
        first_base, first = _build(algebra, seed=seed)
        second_base, second = _build(algebra, seed=seed + 1, shape=(D_OUT, D_IN))
        for layer in (first, second):
            layer.set_adapter_strength(0.75)
        _assert_delta_is_exercised(first, first_base, x, algebra.name)
        _assert_delta_is_exercised(second, second_base, F.gelu(first(x)).detach(),
                                   algebra.name)
        return nn.Sequential(first, nn.GELU(), second)

    plain, wrapped = stack(21), stack(21)

    plain(x).square().sum().backward()
    checkpoint(wrapped, x, use_reentrant=False).square().sum().backward()

    for (name, a), (_, b) in zip(plain.named_parameters(), wrapped.named_parameters()):
        if a.grad is None and b.grad is None:
            continue
        assert torch.equal(a.grad, b.grad), \
            f"{algebra.name}: {name} differs under checkpointing"


@pytest.mark.parametrize("algebra", ALGEBRAS, ids=lambda a: a.name)
def test_the_shipped_initialisation_really_does_hide_a_broken_algebra(algebra):
    """The guard's own gate: without ``_build``'s randomisation the delta is
    exactly zero, so every equivalence assertion in this file would be vacuous.
    """
    generator = torch.Generator().manual_seed(11)
    base = _base_linear(torch.float32, generator)
    layer = _layer(algebra, base)
    if algebra.weight_decompose:
        layer = DoRALinearLayer(base, layer)
    x = torch.randn(4, D_IN, generator=generator) * 0.5

    # Zero for the additive families; for the decomposed ones dora_scale starts
    # at the base's own row norms, so the epilogue is an identity up to fp32
    # rounding rather than exactly zero.
    assert _delta_weight(layer).abs().max().item() < 1e-6
    with pytest.raises(AssertionError):
        _assert_delta_is_exercised(layer, base, x, algebra.name)


def test_oracle_module_does_not_import_training_or_api():
    """``core.adapters.reference`` is not re-exported from the package, so the
    process probe in ``adapter_layering_test`` never imports it; this is its
    layering gate."""
    source = os.path.join(_BACKEND, "core", "adapters", "reference.py")
    with open(source, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [name for name in imported
                 if name == "api" or name.startswith(("api.", "core.training"))]
    assert offenders == [], f"reference.py must not import {offenders}"


# -- upstream conventions, verified against LyCORIS 4.0.0 -------------------
# Commit 03270a3839102e63b48578c80e7c024036de74d7. Each of these was WRONG here
# and shape-compatible, so a third-party checkpoint loaded and denoised at the
# wrong numbers rather than failing.


@pytest.mark.parametrize("options,rank,expected", [
    ((), RANK, ALPHA / RANK),                              # w2 factored
    ((("decompose_both", True),), RANK, ALPHA / RANK),     # both factored
    ((), 0, 1.0),                                          # both full
])
def test_lokr_scale_comes_from_the_tensor_set_not_from_the_stored_alpha(
        options, rank, expected):
    """Upstream overrides ``alpha = lora_dim`` when both operands are stored
    full, so its scale there is exactly 1 -- and it writes that ``lora_dim``
    into the file's ``alpha``. Reading that back as "no rank, so use alpha
    bare" scaled the whole adapter by ``lora_dim``: 8x in this fixture.
    """
    base = _base_linear(torch.float32, torch.Generator().manual_seed(1))
    layer = LoKrLinearLayer(base, rank=rank, alpha=ALPHA, lora_name="s",
                            **dict(options))
    assert layer.scale == expected
    assert layer.branch_tensors()["alpha"].item() == ALPHA


def test_the_full_lokr_form_ignores_alpha_entirely():
    """The invariant behind the row above: with no factored operand there is no
    rank to divide by, so the delta cannot depend on ``alpha`` at all."""
    deltas = []
    for alpha in (1.0, ALPHA):
        base, layer = _build(BY_NAME["lokr_full"], alpha=alpha)
        _assert_delta_is_exercised(
            layer, base,
            torch.randn(4, D_IN, generator=torch.Generator().manual_seed(9)) * 0.5,
            f"lokr_full/a{alpha}")
        deltas.append(_delta_weight(layer))
    assert torch.equal(*deltas)


@pytest.mark.parametrize("dimension,expected", [
    (127, (1, 127)), (320, (16, 20)), (360, (18, 20)), (768, (24, 32)),
    (1024, (32, 32)), (1280, (32, 40)), (3072, (48, 64)), (18432, (128, 144)),
])
def test_factorization_pins_the_default_factor(dimension, expected):
    """Upstream's own docstring table is stale: the CODE gives 360 -> (18, 20),
    not the (8, 45) it documents."""
    assert factorization(dimension) == expected == factorization(dimension, -1)


def test_the_default_factor_agrees_with_the_previous_balanced_search():
    """``factor=-1`` is the only path any shipped code took, and upstream's
    algorithm agrees with the ``isqrt`` search it replaces on every dimension
    -- so adopting upstream moves no existing LoKr."""
    def balanced(n):
        for i in range(int(math.isqrt(n)), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n

    assert [factorization(d) for d in range(2, 4097)] == \
        [balanced(d) for d in range(2, 4097)]


@pytest.mark.parametrize("dimension,factor,expected", [
    (1024, 64, (16, 64)),   # m <= n, so not the (64, 16) this repo returned
    (1024, 8, (8, 128)),
    (768, 8, (8, 96)),
    (30, 4, (3, 10)),       # 30 % 4 != 0: the divisor search caps at 4
    (12, 8, (3, 4)),
])
def test_factorization_pins_an_explicit_factor(dimension, factor, expected):
    assert factorization(dimension, factor) == expected


def test_an_explicit_factor_reaches_both_dimensions():
    """It reached ``out_features`` alone, so the input side stayed balanced and
    the Kronecker structure differed from upstream's for the same config."""
    layer = LoKrLinearLayer(nn.Linear(256, 1024), rank=0, alpha=1.0,
                            lora_name="f", factor=64)
    assert layer.factors == ((16, 64), (4, 64))
    assert tuple(layer.lokr_w1.shape) == (16, 4)
    assert tuple(layer.lokr_w2.shape) == (64, 64)


@pytest.mark.parametrize("algebra", DECOMPOSED, ids=lambda a: a.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_column_magnitudes_on_a_square_weight_match_the_oracle(algebra, dtype):
    """``wd_on_out=False`` stores ``(1, in)`` and norms per INPUT column.

    Square is the dangerous shape: the old ``dora_scale.view(-1, 1)`` raised
    whenever ``in != out`` and reshaped silently otherwise -- and every
    attention ``to_q``/``to_k``/``to_v``/``to_out`` is square.

    This IS two implementations agreeing on one convention; what makes the
    convention right is that it agrees with upstream ``_weight_decompose``,
    checked at ``03270a38``, which no test here can perform.
    """
    for strength in STRENGTHS:
        base, layer = _build(algebra, dtype, shape=SQUARE, dora_axis="column")
        assert tuple(layer.dora_scale.shape) == (1, D_IN)
        _assert_matches_oracle(algebra, base, layer, dtype, strength)


def test_column_magnitudes_are_not_the_old_row_reshape():
    """MEASURED on this fixture: reading a ``(1, in)`` vector as row magnitudes
    is 47.6% off on the delta weight, and 45-68% off on the layer output
    depending on the probe input. Not a rounding nit -- a different image from
    the one the file was trained for."""
    base, layer = _build(BY_NAME["dora"], shape=SQUARE, dora_axis="column")
    correct = _delta_weight(layer)

    w0 = base.weight.detach().to(torch.float32)
    v = w0 + layer.branch_delta_weight()
    row_norm = torch.norm(v, p=2, dim=1, keepdim=True)
    as_rows = layer.dora_scale.detach().reshape(-1, 1) * (v / row_norm) - w0

    assert _relative_error(as_rows, correct) > 0.3


@pytest.mark.parametrize("shape", [(D_IN + 1, 1), (1, D_IN + 1), (2, 2),
                                   (D_IN, 2), (1, 1, D_IN)])
def test_a_dora_scale_of_any_other_shape_is_refused(shape):
    """Refused, not reshaped: the shape is the only record of ``wd_on_out``."""
    base = _base_linear(torch.float32, torch.Generator().manual_seed(1), SQUARE)
    branch = LoRALinearLayer(base, rank=RANK, alpha=ALPHA, lora_name="d")
    with pytest.raises(ValueError):
        DoRALinearLayer(base, branch, dora_scale=torch.ones(shape))
    with pytest.raises(ValueError):
        dora_effective_delta_weight(base.weight, torch.zeros_like(base.weight),
                                    torch.ones(shape))


@pytest.mark.parametrize("name", ["loha_scalar", "lokr_both_scalar"])
def test_the_trained_scalar_sets_the_effective_strength(name):
    """``scalar`` must actually scale the delta.

    No file carries the key -- upstream folds it into the saved
    ``w1``/``hada_w1_a`` and its reader forces ``scalar := 1`` -- so what this
    pins is the WRITE side: a serializer built on ``branch_tensors()`` that
    emitted ``scalar`` bare, beside an unfolded ``w1``, would hand every other
    reader an adapter ``1/scalar`` too strong."""
    base, layer = _build(BY_NAME[name])
    scalar = float(layer.scalar.detach())
    assert scalar != 1.0
    trained = _delta_weight(layer)

    with torch.no_grad():
        layer.scalar.fill_(1.0)          # what dropping the tensor would give
    ignored = _delta_weight(layer)

    assert _relative_error(ignored * scalar, trained) <= TOLERANCE[torch.float32]
    assert _relative_error(ignored, trained) > 0.1


@pytest.mark.parametrize("name", sorted(TUCKER_TENSOR_NAMES))
def test_a_tucker_tensor_is_refused_rather_than_dropped(name):
    """They exist only for a target with kernel dims and also transpose
    ``hada_w1_a``; a Linear branch that ignored them would apply a different
    algebra at matching shapes."""
    _, layer = _build(BY_NAME["loha"])
    with pytest.raises(ValueError, match="Tucker"):
        layer.load_tensors({name: torch.zeros(2, 2, 1, 1)})


def test_loha_and_lokr_carry_alpha_as_a_branch_tensor_but_lora_does_not():
    """LoRA deliberately omits ``alpha`` (a spec constant the saving adapter
    owns); LoHa/LoKr emit it as a fresh non-Parameter scalar, which no optimizer
    can train and which ``load_tensors`` would copy into a throwaway. Convention
    under review; recorded so the asymmetry is not silent."""
    base = _base_linear(torch.float32, torch.Generator().manual_seed(1))
    lora = LoRALinearLayer(base, rank=RANK, alpha=ALPHA, lora_name="l")
    assert "alpha" not in lora.branch_tensors()
    for cls in (LoHaLinearLayer, LoKrLinearLayer):
        tensors = cls(base, rank=RANK, alpha=ALPHA, lora_name="l").branch_tensors()
        assert not isinstance(tensors["alpha"], nn.Parameter)


# -- the same oracle, reached through the checkpoint builder ----------------
# ``build_adapter_branch`` derives geometry, alpha and stored form from the
# TENSORS. A builder that transposes a factor, picks the wrong Kronecker split
# or applies the scale twice produces a correctly shaped branch, so only a
# numeric comparison catches it.

#: ``scalar`` is a training-side tensor no file carries (upstream folds it into
#: ``w1`` at save and forces ``scalar := 1`` at load), so the builder refuses to
#: reconstruct one and those two rows cannot round-trip through it.
BUILDABLE = tuple(a for a in ALGEBRAS if "scalar" not in a.variant)


@pytest.mark.parametrize("algebra", BUILDABLE, ids=lambda a: a.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_a_branch_built_from_tensors_matches_the_oracle(algebra, dtype):
    from core.adapters import SHAPE_MISMATCH, TensorGroup, build_adapter_branch

    for strength in STRENGTHS:
        base, layer = _build(algebra, dtype)
        group = TensorGroup("gate", dict(layer.export_tensors()))
        # LoRA exports no ``alpha``, so its scale must come from the metadata
        # arm of the precedence; LoHa/LoKr carry the tensor and it wins.
        branch = build_adapter_branch(base, group, metadata_alpha=ALPHA,
                                      lora_dtype=dtype, lora_name="gate")
        assert branch is not SHAPE_MISMATCH
        _assert_matches_oracle(algebra, base, branch, dtype, strength)


@pytest.mark.parametrize("algebra", BUILDABLE, ids=lambda a: a.name)
def test_a_built_branch_reproduces_the_source_layer_bit_for_bit(algebra):
    from core.adapters import TensorGroup, build_adapter_branch

    base, layer = _build(algebra)
    expected = _delta_weight(layer)
    branch = build_adapter_branch(base, TensorGroup("gate", dict(layer.export_tensors())),
                                  metadata_alpha=ALPHA)
    assert torch.equal(_delta_weight(branch), expected)

# -- the decomposition axis, past what the oracle can witness ---------------


@pytest.mark.parametrize("algebra", DECOMPOSED, ids=lambda a: a.name)
def test_a_decomposed_branch_cannot_ride_a_block_swap(algebra):
    """``branch_survives_block_swap`` asks whether a block offloader's
    name-based walk reaches every tensor the branch owns. For a DoRA over an
    ordinary LoRA the answer must be NO for the right reason: the two factors
    ARE ``nn.Linear`` weights and would ride, and ``dora_scale`` is a bare
    parameter that would not -- so the wrapper is what makes the answer flip,
    not the algebra underneath it."""
    from core.adapters import branch_survives_block_swap

    _base, layer = _build(algebra)
    assert branch_survives_block_swap(layer) is False
    assert branch_survives_block_swap(layer.branch) is (algebra.algorithm == "lora")

    # And the magnitude is exactly the tensor left behind.
    carried = {id(m.weight) for m in layer.modules()
               if m.__class__.__name__.endswith("Linear")
               and getattr(m, "weight", None) is not None}
    base_ids = {id(p) for p in layer.original_module.parameters()}
    stranded = [p for p in layer.parameters()
                if id(p) not in carried and id(p) not in base_ids]
    if algebra.algorithm == "lora":
        assert [id(p) for p in stranded] == [id(layer.dora_scale)]
    else:
        assert id(layer.dora_scale) in {id(p) for p in stranded}


@pytest.mark.parametrize("algebra", DECOMPOSED, ids=lambda a: a.name)
@pytest.mark.parametrize("base_dtype", [torch.float16, torch.bfloat16])
def test_the_magnitude_takes_the_branch_dtype_not_the_bases(algebra, base_dtype):
    """A trained magnitude with no fp32 master, while its factors have one, is
    the asymmetry this resolves: ``new_adapter_branch`` builds the whole branch
    -- factors and magnitude -- in the run's ``lora_dtype``."""
    from core.adapters import new_adapter_branch

    base = _base_linear(base_dtype, torch.Generator().manual_seed(1))
    layer = new_adapter_branch(algebra.algorithm, base, rank=RANK, alpha=ALPHA,
                               dtype=torch.float32, weight_decompose=True)
    assert layer.dora_scale.dtype == torch.float32
    # Its own factors, not ``branch.parameters()`` -- that also yields the
    # wrapped base's fp16 weight.
    assert all(p.dtype == torch.float32
               for p in layer.branch.trainable_parameters())

    # A load-time build takes the branch dtype the LOADER chose. On ten of the
    # eleven that is the base weight's own, so a generation load is unmoved;
    # Lens's prefers the bias, so there it can differ. Either way the magnitude
    # rides with the factors.
    _b2, loaded = _build(algebra, base_dtype)
    assert loaded.dora_scale.dtype == base_dtype


@pytest.mark.parametrize("algebra", DECOMPOSED, ids=lambda a: a.name)
def test_rank_alpha_and_name_delegate_through_the_wrapper(algebra):
    """A loader reads and WRITES these off a built branch -- Z-Image logs
    rank/alpha, MiniMax-H3's fused-QKV split assigns them -- and the wrapper has
    none of its own. A write must reach the inner branch's own scale rule, not
    a copy of it here."""
    _base, layer = _build(algebra)
    assert (layer.rank, layer.alpha) == (layer.branch.rank, layer.branch.alpha)
    assert layer.lora_name == layer.branch.lora_name

    layer.alpha = 2 * ALPHA
    assert layer.alpha == layer.branch.alpha == 2 * ALPHA
    # LoKr takes its rank off its tensors, so only its alpha is writable here.
    if algebra.algorithm != "lokr":
        assert layer.branch.scale == 2 * ALPHA / layer.branch.rank


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_the_row_magnitude_convention_agrees_with_peft(dtype):
    """The one axis check that is not two mirrors of one convention.

    ``layers.py`` and ``reference.py`` both read a ``(out, 1)`` ``dora_scale``
    as per-output-row magnitudes, and the oracle discloses that it shares that
    reading rather than deriving it, so their agreement proves nothing. PEFT is
    a third implementation written by neither: it norms along ``dim=1``, stores
    one magnitude per output row, and diffusers' Kohya converter maps
    ``dora_scale`` straight onto it.

    Forward only: PEFT detaches the weight norm from the graph (DoRA paper 4.3)
    where this repo takes the exact gradient, so the backwards differ by
    construction. Recorded in the design doc, not asserted away here.
    """
    dora = pytest.importorskip("peft.tuners.lora.dora")

    generator = torch.Generator().manual_seed(17)
    base = nn.Linear(D_IN, D_OUT, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.randn(base.weight.shape, generator=generator) * 0.2)
    base = base.to(dtype)
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    inner = LoRALinearLayer(base, rank=RANK, alpha=ALPHA, lora_name="peft",
                            lora_dtype=dtype)
    with torch.no_grad():
        inner.lora_up.weight.copy_(
            (torch.randn(inner.lora_up.weight.shape, generator=generator)
             * FACTOR_STD).to(dtype))
    magnitude = _dora_scale(base, "row", generator)
    ours = DoRALinearLayer(base, inner, dora_scale=magnitude.clone())

    theirs = dora.DoraLinearLayer(fan_in_fan_out=False)
    theirs.weight = nn.Parameter(magnitude.detach().clone().to(dtype))

    x = (torch.randn(4, D_IN, generator=generator) * 0.5).to(dtype)
    _assert_delta_is_exercised(ours, base, x, f"peft/{dtype}")
    with torch.no_grad():
        expected = base(x) + theirs(x, lora_A=inner.lora_down, lora_B=inner.lora_up,
                                    scaling=ALPHA / RANK, base_layer=base)
        assert _relative_error(ours(x), expected) <= TOLERANCE[dtype]

