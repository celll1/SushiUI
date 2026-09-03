"""Correctness gates for the adapter algebras against the fp32 oracle. CPU, ~10 s.

The layer classes in ``core.adapters.layers`` were written without a reference
to check them against, so this file supplies one: ``core.adapters.reference``
is an INDEPENDENT fp32 implementation (explicit outer-product sums, explicit
Kronecker block assembly) that shares no code with the layers, and every case
below compares a layer's forward, its input gradient and EVERY one of its
parameter gradients against it.

Covers the "Correctness gates" bullets of
``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` that a CPU test can reach: fp32 /
fp16 / bf16, strengths 0 / 1 / fractional / >1 / negative, runtime versus
merge/unmerge including ``alpha != rank``, and gradient checkpointing on/off.
Block swap, the fused-optimizer census, quantized bases and the per-arch round
trip are gated elsewhere.

THE FAKE-ASSERTION TRAP. Every algebra zero-initialises one factor so its
initial delta is exactly zero -- and a zero delta makes every equivalence
assertion here pass no matter how wrong the algebra is. ``_build`` therefore
randomises that factor, and ``_assert_delta_is_exercised`` runs before each
equivalence assertion.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_oracle_gate_cheap_test.py -v
"""

import ast
import os
import sys
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

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
    DoRALinearLayer,
    LoHaLinearLayer,
    LoKrLinearLayer,
    LoRALinearLayer,
)
from core.adapters.reference import (  # noqa: E402
    adapter_delta_weight,
    dora_effective_delta_weight,
)

#: 10 = 2*5 and 12 = 3*4, so LoKr factors BOTH sides non-trivially; a square or
#: prime shape would collapse its Kronecker structure and hide an operand swap.
D_IN, D_OUT = 12, 10
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
TOLERANCE = {
    torch.float32: 2e-6,
    torch.float16: 6e-3,
    torch.bfloat16: 5e-2,
}
DTYPES = tuple(TOLERANCE)

#: Randomisation applied to the zero-initialised factor. Sized by the SMALLEST
#: delta of the six algebras (LoHa's, a Hadamard product of two small factors),
#: so that even at strength 0.35 the branch clears ``MIN_MOVE``.
FACTOR_STD = 2.0

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


#: The factors each algebra must export, independent of the layer classes: a
#: branch that silently stops exporting one has to fail, and a set derived from
#: what the layer exported could not notice.
FACTORS = {
    "lora": ("lora_down.weight", "lora_up.weight"),
    "loha": ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"),
    "lokr": ("lokr_w1", "lokr_w2_a", "lokr_w2_b"),
}


@dataclass(frozen=True)
class Algebra:
    name: str
    algorithm: str                # what the oracle dispatches on
    weight_decompose: bool
    zero_init: Tuple[str, ...]    # factors the layer zero-initialises

    @property
    def parameter_names(self) -> FrozenSet[str]:
        names = FACTORS[self.algorithm]
        if not self.weight_decompose:
            return frozenset(names)
        return frozenset([f"branch.{name}" for name in names] + ["dora_scale"])


#: Table-driven so Phase 3's weight-decomposed pairs are rows, not copies. The
#: decomposed rows exercise the epilogue only; the capability matrix in
#: ``core.training.arch.base_arch`` still refuses every one of them.
ALGEBRAS = (
    Algebra("lora", "lora", False, ("lora_up.weight",)),
    Algebra("loha", "loha", False, ("hada_w2_a",)),
    Algebra("lokr", "lokr", False, ("lokr_w2_a",)),
    Algebra("dora", "lora", True, ("lora_up.weight",)),
    Algebra("doha", "loha", True, ("hada_w2_a",)),
    Algebra("dokr", "lokr", True, ("lokr_w2_a",)),
)
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


def _build(algebra: Algebra, dtype: torch.dtype = torch.float32, seed: int = 11,
           rank: int = RANK, alpha: float = ALPHA,
           shape: Tuple[int, int] = (D_IN, D_OUT)):
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
    layer = _LAYER_CLASS[algebra.algorithm](
        base, rank=rank, alpha=alpha, lora_name="gate", lora_dtype=dtype)
    tensors = layer.branch_tensors()
    with torch.no_grad():
        for name in algebra.zero_init:
            tensor = tensors[name]
            tensor.copy_((torch.randn(tensor.shape, generator=generator) * FACTOR_STD
                          ).to(tensor.dtype))
    if algebra.weight_decompose:
        layer = DoRALinearLayer(base, layer)
        with torch.no_grad():
            # dora_scale left at the base's row norms makes W_adapter == W0 and
            # the delta near-zero: the same trap as the zero factor.
            layer.dora_scale.mul_(
                (1.0 + 0.05 * torch.randn(layer.dora_scale.shape, generator=generator)
                 ).to(layer.dora_scale.dtype))
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


@pytest.mark.parametrize("algebra", ALGEBRAS, ids=lambda a: a.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_forward_input_grad_and_every_parameter_grad_match_the_oracle(algebra, dtype):
    tolerance = TOLERANCE[dtype]
    for strength in STRENGTHS:
        base, layer = _build(algebra, dtype)
        layer.set_adapter_strength(strength)

        generator = torch.Generator().manual_seed(5)
        x = (torch.randn(4, D_IN, generator=generator) * 0.5).to(dtype).requires_grad_()
        grad_out = (torch.randn(4, D_OUT, generator=generator) * 0.5).to(dtype)

        if strength != 0.0:
            _assert_delta_is_exercised(layer, base, x.detach(),
                                       f"{algebra.name}/{dtype}/s={strength}")
        out = layer(x)
        (out * grad_out).sum().backward()

        parameters = _branch_parameters(layer)
        # Before the oracle, so a branch that stops exporting a factor fails
        # here rather than as a KeyError inside it.
        assert set(parameters) == algebra.parameter_names,             f"{algebra.name}: the branch exports {sorted(parameters)}"
        oracle_leaves = {name: tensor.detach().to(torch.float32).clone().requires_grad_()
                         for name, tensor in parameters.items()}
        base_w = base.weight.detach().to(torch.float32)
        base_b = base.bias.detach().to(torch.float32)
        x32 = x.detach().to(torch.float32).clone().requires_grad_()

        delta = _oracle_delta(algebra, oracle_leaves, base_w, strength)
        out_ref = F.linear(x32, base_w + delta, base_b)
        (out_ref * grad_out.to(torch.float32)).sum().backward()

        label = f"{algebra.name} {dtype} s={strength}"
        assert _relative_error(out, out_ref) <= tolerance, f"{label}: forward"
        assert _relative_error(x.grad, x32.grad) <= tolerance, f"{label}: input grad"
        # forward_delta is the CompositeAdapterLayer branch protocol, i.e. what
        # every generation backend actually calls; LoRA computes it separately
        # from its own forward, so it needs its own comparison.
        with torch.no_grad():
            composed = base(x.detach()) + layer.forward_delta(x.detach())
        assert _relative_error(composed, out_ref) <= tolerance, f"{label}: forward_delta"
        for name, parameter in parameters.items():
            assert parameter.grad is not None, f"{label}: no gradient reached {name}"
            error = _relative_error(parameter.grad, oracle_leaves[name].grad)
            assert error <= tolerance, f"{label}: grad {name} off by {error:.3e}"


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
    layer = _LAYER_CLASS[algebra.algorithm](
        base, rank=RANK, alpha=ALPHA, lora_name="fresh")
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


# -- current-behaviour records, NOT endorsements ----------------------------
# Each documents a convention the parallel upstream-LyCORIS check owns, so that
# changing one is deliberate and visible in a diff.

def test_lokr_unfactored_form_scales_by_alpha_alone():
    """rank 0 selects the full ``lokr_w2`` and the scale becomes ``alpha`` --
    neither ``alpha/rank`` nor 1.0. Convention under review."""
    generator = torch.Generator().manual_seed(1)
    base = _base_linear(torch.float32, generator)
    layer = LoKrLinearLayer(base, rank=0, alpha=8.0, lora_name="full")
    assert layer.scale == 8.0
    assert "lokr_w2" in layer.branch_tensors()
    with torch.no_grad():
        layer.lokr_w2.copy_(torch.randn(layer.lokr_w2.shape, generator=generator) * 0.3)
    oracle = adapter_delta_weight(
        "lokr",
        {name: tensor.detach() for name, tensor in layer.branch_tensors().items()},
        rank=0, alpha=8.0)
    x = torch.randn(4, D_IN, generator=torch.Generator().manual_seed(9)) * 0.5
    _assert_delta_is_exercised(layer, base, x, "lokr/rank0")
    assert _relative_error(_delta_weight(layer), oracle) <= TOLERANCE[torch.float32]


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
