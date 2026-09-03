"""Tensor grouping: key spellings, partial groups, aliases, row splitting. CPU, ~2 s.

``core.adapters.groups`` has no production caller yet (design doc, phase 2), so
this file is its whole gate. Three things it catches that a shape check cannot:

  * ``.lokr_w1_a`` read as ``.lokr_w1`` plus a stray ``_a``;
  * an incomplete group silently promoted to a complete one, or ``partial``
    turning into a refusal;
  * a LoKr fused-QKV row split that is not a Kronecker product at all. That one
    is checked NUMERICALLY, against the fp32 oracle, because a wrong split has
    the right shape.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_tensor_group_cheap_test.py -v
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from core.adapters import (  # noqa: E402
    ADAPTER_SUFFIXES,
    SHAPE_MISMATCH,
    LoHaLinearLayer,
    LoKrLinearLayer,
    LoRALinearLayer,
    TensorGroup,
    build_adapter_branch,
    group_adapter_tensors,
    split_adapter_suffix,
    split_group_on_out_rows,
)
from core.adapters.reference import (  # noqa: E402
    loha_delta_weight, lokr_delta_weight, lora_delta_weight)

#: 12 = 3*4 both ways, so LoKr factors non-trivially and L=3 is coprime to the
#: n=2 split below -- which is the case the split must refuse.
D_IN, D_OUT = 12, 12
RANK, ALPHA = 4, 8.0
STEM = "blocks.0.attn.to_q"


def _t(*shape, seed=0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _lora_tensors():
    return {"lora_down.weight": _t(RANK, D_IN, seed=1),
            "lora_up.weight": _t(D_OUT, RANK, seed=2)}


def _loha_tensors():
    return {"hada_w1_a": _t(D_OUT, RANK, seed=3), "hada_w1_b": _t(RANK, D_IN, seed=4),
            "hada_w2_a": _t(D_OUT, RANK, seed=5), "hada_w2_b": _t(RANK, D_IN, seed=6)}


def _lokr_tensors(factored=True):
    """out 12 -> (3, 4), in 12 -> (3, 4)."""
    tensors = {"lokr_w1": _t(3, 3, seed=7)}
    if factored:
        tensors["lokr_w2_a"] = _t(4, RANK, seed=8)
        tensors["lokr_w2_b"] = _t(RANK, 4, seed=9)
    else:
        tensors["lokr_w2"] = _t(4, 4, seed=10)
    return tensors


ALGORITHMS = {"lora": _lora_tensors, "loha": _loha_tensors, "lokr": _lokr_tensors}

#: canonical name -> the spellings a checkpoint may use for it.
SPELLINGS = {
    "lora_down.weight": (".lora_down.weight", ".lora_A.weight",
                         ".lora_A.default.weight"),
    "lora_up.weight": (".lora_up.weight", ".lora_B.weight",
                       ".lora_B.default.weight"),
}


def _keys(tensors, dialect=0, stem=STEM):
    """Write a canonical tensor dict out under one dialect's key spellings."""
    written = {}
    for name, tensor in tensors.items():
        spellings = SPELLINGS.get(name, (f".{name}",))
        written[stem + spellings[min(dialect, len(spellings) - 1)]] = tensor
    return written


# -- suffix table ----------------------------------------------------------


def test_no_suffix_is_a_suffix_of_another():
    """What makes the longest-first order safe rather than merely tidy: while
    this holds, ``endswith`` is unambiguous; the first entry that breaks it is
    the one the ordering exists for."""
    for a in ADAPTER_SUFFIXES:
        for b in ADAPTER_SUFFIXES:
            assert a == b or not a.endswith(b), f"{a!r} ends with {b!r}"


@pytest.mark.parametrize("suffix,canonical", sorted(ADAPTER_SUFFIXES.items()))
def test_every_suffix_resolves_to_its_own_canonical_name(suffix, canonical):
    assert split_adapter_suffix(STEM + suffix) == (STEM, canonical)


@pytest.mark.parametrize("suffix", [".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a",
                                    ".lokr_w2_b"])
def test_a_factored_lokr_suffix_is_never_read_as_the_full_one(suffix):
    """``.lokr_w1_a`` is a DIFFERENT tensor from ``.lokr_w1``: reading it as the
    full operand would allocate ``(out_l, in_m)`` for an ``(out_l, rank)``
    factor and silently mis-shape the Kronecker split."""
    stem, name = split_adapter_suffix(STEM + suffix)
    assert (stem, name) == (STEM, suffix[1:])


@pytest.mark.parametrize("key", ["blocks.0.attn.to_q.weight", "alpha",
                                 ".lora_down.weight", "lora_down.weight",
                                 "blocks.0.norm.bias"])
def test_a_non_adapter_key_matches_nothing(key):
    assert split_adapter_suffix(key) is None


# -- grouping --------------------------------------------------------------


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
@pytest.mark.parametrize("dialect", [0, 1, 2])
@pytest.mark.parametrize("dora", [False, True])
def test_every_spelling_of_every_algorithm_groups_the_same(algorithm, dialect, dora):
    tensors = ALGORITHMS[algorithm]()
    tensors["alpha"] = torch.tensor(ALPHA)
    if dora:
        tensors["dora_scale"] = torch.ones(D_OUT)

    result = group_adapter_tensors(_keys(tensors, dialect))

    assert result.unmatched == ()
    assert result.partial == {}
    group = result.groups[STEM]
    assert dict(group) == tensors
    assert group.algorithm == algorithm
    assert group.weight_decompose is dora
    assert group.missing() == ()
    assert group.alpha == ALPHA
    assert group.rank == RANK


def test_a_kohya_stem_survives_grouping_and_a_stem_of_hook_translates_it():
    raw = _keys(_lora_tensors(), stem="lora_unet_blocks_0_attn_to_q")
    plain = group_adapter_tensors(raw)
    assert set(plain.groups) == {"lora_unet_blocks_0_attn_to_q"}

    def stem_of(stem):
        if not stem.startswith("lora_unet_"):
            return None
        return stem[len("lora_unet_"):].replace("_", ".")

    translated = group_adapter_tensors(raw, stem_of)
    assert set(translated.groups) == {"blocks.0.attn.to.q"}
    assert group_adapter_tensors(raw, lambda _stem: None).unmatched == tuple(raw)


def test_an_incomplete_group_lands_in_partial_and_not_in_groups():
    """Never raised on -- see ``GroupingResult``."""
    tensors = _loha_tensors()
    tensors.pop("hada_w2_b")
    result = group_adapter_tensors(_keys(tensors))

    assert result.groups == {}
    assert result.partial[STEM].missing() == ("hada_w2_b",)
    assert result.unmatched == ()


@pytest.mark.parametrize("dropped,missing", [
    ("lora_up.weight", ("lora_up.weight",)),
    ("lora_down.weight", ("lora_down.weight",)),
])
def test_a_half_lora_pair_is_partial(dropped, missing):
    tensors = _lora_tensors()
    tensors.pop(dropped)
    result = group_adapter_tensors(_keys(tensors))
    assert result.groups == {} and result.partial[STEM].missing() == missing


@pytest.mark.parametrize("tensors,missing", [
    ({"lokr_w1": _t(3, 3)}, ("lokr_w2",)),
    ({"lokr_w1": _t(3, 3), "lokr_w2_a": _t(4, RANK)}, ("lokr_w2_b",)),
    ({"lokr_w2": _t(4, 4)}, ("lokr_w1",)),
])
def test_an_incomplete_lokr_operand_is_partial(tensors, missing):
    """Either operand may be stored full OR factored, so completeness is
    per-operand rather than a fixed name list."""
    assert TensorGroup(STEM, dict(tensors)).missing() == missing


def test_a_group_of_only_alpha_is_partial_not_an_unknown_success():
    group = group_adapter_tensors({f"{STEM}.alpha": torch.tensor(4.0)})
    assert group.groups == {}
    assert group.partial[STEM].algorithm == "unknown"


def test_unmatched_keys_are_reported_and_not_grouped():
    tensors = _keys(_lora_tensors())
    tensors["blocks.0.attn.to_q.weight"] = _t(4, 4)
    result = group_adapter_tensors(tensors)
    assert result.unmatched == ("blocks.0.attn.to_q.weight",)
    assert set(result.groups[STEM]) == set(_lora_tensors())


# -- legacy aliases --------------------------------------------------------


def test_the_legacy_aliases_return_the_same_objects_as_the_canonical_names():
    """Transitional, and load-bearing: the eleven architecture parsers read
    ``weights["down"]``, so they can move onto ``TensorGroup`` without their
    branch builders changing in the same commit."""
    tensors = _lora_tensors()
    tensors["alpha"] = torch.tensor(ALPHA)
    group = group_adapter_tensors(_keys(tensors)).groups[STEM]

    assert group["down"] is group["lora_down.weight"] is tensors["lora_down.weight"]
    assert group["up"] is group["lora_up.weight"] is tensors["lora_up.weight"]
    assert group["alpha"] is tensors["alpha"]
    assert "down" in group and "up" in group and "alpha" in group
    assert group.get("down") is group["down"]
    assert "mid" not in group and "nonsense" not in group


def test_iteration_yields_canonical_names_only():
    """``__contains__`` accepts aliases, ``__iter__`` does not emit them, so
    ``dict(group)`` is exactly what ``load_tensors`` consumes."""
    group = group_adapter_tensors(_keys(_lora_tensors())).groups[STEM]
    assert sorted(group) == ["lora_down.weight", "lora_up.weight"]
    assert len(group) == 2


# -- fused-QKV row splitting -----------------------------------------------

_ORACLE = {"lora": lora_delta_weight, "loha": loha_delta_weight,
           "lokr": lokr_delta_weight}


def _delta(group):
    return _ORACLE[group.algorithm](dict(group), rank=group.rank or None,
                                    alpha=group.alpha)


@pytest.mark.parametrize("algorithm", ["lora", "loha"])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_a_row_split_reconstructs_the_fused_delta_exactly(algorithm, n):
    """``delta[rows] = up[rows, :] @ down``: the ``_b`` factors are shared, so
    the split is exact, not approximate."""
    tensors = ALGORITHMS[algorithm]()
    tensors["alpha"] = torch.tensor(ALPHA)
    group = TensorGroup(STEM, tensors)
    inner = D_OUT // n

    parts = split_group_on_out_rows(group, n, inner)
    assert parts is not None and len(parts) == n
    fused = _delta(group)
    for index, part in parts.items():
        assert torch.equal(_delta(part), fused[index * inner:(index + 1) * inner])


def _kronecker_residual(chunk, M=3, N=4):
    """How far ``chunk`` is from the NEAREST Kronecker product with an ``(*, M)``
    (x) ``(*, N)`` column split: ``min`` over every row factorization of
    ``s2/s1`` of the Van Loan rearrangement, which is exactly 0 for a true
    Kronecker product."""
    rows = chunk.shape[0]
    residuals = []
    for p in (p for p in range(1, rows + 1) if rows % p == 0):
        q = rows // p
        rearranged = chunk.reshape(p, q, M, N).permute(0, 2, 1, 3).reshape(p * M, q * N)
        singular = torch.linalg.svdvals(rearranged.to(torch.float64))
        residuals.append((singular[1] / singular[0]).item())
    return min(residuals)


@pytest.mark.parametrize("n,inner", [(2, 6), (4, 3)])
def test_a_lokr_row_split_that_is_not_a_kronecker_product_is_refused(n, inner):
    """The case that would otherwise ship a numerically wrong MiniMax-H3
    adapter. ``kron(w1, w2)`` puts row ``i*K + k`` at ``w1[i] (x) w2[k]``, so a
    piece spanning a partial ``i`` is reproduced by no ``(a, b)`` sharing the
    PARENT'S column split -- which is what ``_kronecker_residual`` measures, and
    the qualifier matters because every matrix is a degenerate
    ``kron(1x1, itself)`` under some other split. Shown numerically rather than
    by shape. MEASURED worst piece on this fixture: 0.31 (n=2) and 0.27 (n=4)
    away from the nearest Kronecker product with the parent's (3, 4) split.

    n=4 is also the case that shows why the whole split must be judged, not one
    piece: its FIRST piece lands inside block ``i=0`` and is a clean Kronecker
    product; its second straddles ``i=0`` and ``i=1`` and is not.
    """
    group = TensorGroup(STEM, _lokr_tensors(factored=False))
    assert group["lokr_w1"].shape[0] % n != 0
    assert split_group_on_out_rows(group, n, inner) is None

    fused = _delta(group)
    worst = max(_kronecker_residual(fused[i * inner:(i + 1) * inner])
                for i in range(n))
    assert worst > 1e-3, (
        "every piece of this split is a Kronecker product, so the refusal is "
        "not merely conservative but wrong")


def test_the_refusal_is_conservative_where_every_piece_lands_inside_one_block():
    """Recorded, not fixed: when ``inner`` divides ``w2.shape[0]`` every piece
    lies within one ``i`` block and IS ``kron(w1[i:i+1], w2[k0:k1])``, yet
    ``n`` does not divide ``w1.shape[0]`` and the split is refused. Refusing a
    representable case costs a feature; accepting an unrepresentable one ships
    wrong numbers."""
    group = TensorGroup(STEM, _lokr_tensors(factored=False))
    fused = _delta(group)
    assert max(_kronecker_residual(fused[i * 2:(i + 1) * 2]) for i in range(6)) < 1e-6
    assert split_group_on_out_rows(group, 6, 2) is None


@pytest.mark.parametrize("factored", [True, False])
def test_a_block_aligned_lokr_row_split_is_exact(factored):
    """``n`` divides ``w1.shape[0]``: each piece covers whole ``i`` blocks and
    is ``kron(w1[rows], w2)``."""
    group = TensorGroup(STEM, _lokr_tensors(factored))
    parts = split_group_on_out_rows(group, 3, 4)
    assert parts is not None
    fused = _delta(group)
    for index, part in parts.items():
        assert torch.equal(_delta(part), fused[index * 4:(index + 1) * 4])
        assert part["lokr_w1"].shape == (1, 3)


@pytest.mark.parametrize("kwargs", [
    {"n": 0, "inner": 4}, {"n": 3, "inner": 0}, {"n": 3, "inner": 5},
])
def test_a_row_split_that_does_not_cover_the_output_is_refused(kwargs):
    group = TensorGroup(STEM, _lora_tensors())
    assert split_group_on_out_rows(group, **kwargs) is None


def test_a_dora_or_tucker_or_partial_group_is_never_split():
    """``dora_scale``'s ``(1, in)`` form has no row axis at all, so slicing it
    on rows would be wrong for half of the files that carry one."""
    base = _lora_tensors()
    for extra in ({"dora_scale": torch.ones(D_OUT)},
                  {"hada_t1": torch.ones(2, 2, 1, 1)}):
        tensors = dict(base)
        tensors.update(extra)
        assert split_group_on_out_rows(TensorGroup(STEM, tensors), 3, 4) is None
    half = {"lora_down.weight": base["lora_down.weight"]}
    assert split_group_on_out_rows(TensorGroup(STEM, half), 3, 4) is None


# -- build_adapter_branch --------------------------------------------------


def _linear(out_features=D_OUT, in_features=D_IN):
    linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear.weight.copy_(_t(out_features, in_features, seed=31) * 0.1)
    return linear


@pytest.mark.parametrize("algorithm,cls", [("lora", LoRALinearLayer),
                                           ("loha", LoHaLinearLayer),
                                           ("lokr", LoKrLinearLayer)])
def test_the_builder_dispatches_on_the_group_algorithm(algorithm, cls):
    group = TensorGroup(STEM, ALGORITHMS[algorithm]())
    branch = build_adapter_branch(_linear(), group)
    assert isinstance(branch, cls)
    assert branch.lora_name == STEM


def test_the_builder_honours_the_alpha_precedence():
    """Per-key tensor, then file metadata, then rank -- the order Z-Image's
    codec set in phase 0."""
    tensors = _lora_tensors()
    assert build_adapter_branch(_linear(), TensorGroup(STEM, dict(tensors))).alpha \
        == float(RANK)
    assert build_adapter_branch(_linear(), TensorGroup(STEM, dict(tensors)),
                                metadata_alpha=3.0).alpha == 3.0
    with_tensor = dict(tensors, alpha=torch.tensor(ALPHA))
    assert build_adapter_branch(_linear(), TensorGroup(STEM, with_tensor),
                                metadata_alpha=3.0).alpha == ALPHA


def test_the_builder_accepts_a_layer_class_override():
    from core.adapters import MiniMaxH3LoRALinearLayer

    branch = build_adapter_branch(_linear(), TensorGroup(STEM, _lora_tensors()),
                                  layer_cls=MiniMaxH3LoRALinearLayer)
    assert isinstance(branch, MiniMaxH3LoRALinearLayer)


def test_the_builder_wraps_a_dora_group_in_the_decomposition_epilogue():
    from core.adapters import DoRALinearLayer

    tensors = dict(_lora_tensors(), dora_scale=torch.ones(D_OUT))
    branch = build_adapter_branch(_linear(), TensorGroup(STEM, tensors))
    assert isinstance(branch, DoRALinearLayer)
    assert isinstance(branch.branch, LoRALinearLayer)


@pytest.mark.parametrize("mutate", [
    lambda t: t.pop("lora_up.weight"),
    lambda t: t.__setitem__("lora_up.weight", _t(D_OUT + 1, RANK)),
    lambda t: t.__setitem__("lora_down.weight", _t(RANK, D_IN + 1)),
    lambda t: t.__setitem__("hada_t1", torch.ones(2, 2, 1, 1)),
    lambda t: t.__setitem__("lora_mid.weight", torch.ones(RANK, RANK)),
])
def test_a_group_the_base_cannot_take_is_SHAPE_MISMATCH_not_an_exception(mutate):
    """One target whose shapes disagree is a module to skip, which is what the
    eleven loaders already do with this sentinel."""
    tensors = _lora_tensors()
    mutate(tensors)
    assert (build_adapter_branch(_linear(), TensorGroup(STEM, tensors))
            is SHAPE_MISMATCH)


@pytest.mark.parametrize("label,tensors", [
    # Each of these ESCAPED the builder before: the exception came from a shape
    # index or ``Tensor.item()`` rather than from a validated check, so widening
    # the caught set is the fix and these four rows are the gate. Once a backend
    # is wired onto this, one malformed target in a foreign file must not turn
    # "skip it and report lora_partial" into a 500 out of a generation request.
    ("1-D hada_w1_a -> IndexError", {
        "hada_w1_a": _t(RANK), "hada_w1_b": _t(RANK, D_IN),
        "hada_w2_a": _t(D_OUT, RANK), "hada_w2_b": _t(RANK, D_IN)}),
    ("0-D lora_down -> IndexError", {
        "lora_down.weight": torch.tensor(1.0), "lora_up.weight": _t(D_OUT, RANK)}),
    ("2-element alpha -> RuntimeError from item()", {
        "lora_down.weight": _t(RANK, D_IN), "lora_up.weight": _t(D_OUT, RANK),
        "alpha": torch.tensor([8.0, 4.0])}),
    ("rank-0 lora -> ZeroDivisionError from alpha/rank", {
        "lora_down.weight": _t(0, D_IN), "lora_up.weight": _t(D_OUT, 0)}),
])
def test_a_malformed_tensor_set_is_SHAPE_MISMATCH_not_an_escaping_exception(
        label, tensors):
    assert (build_adapter_branch(_linear(), TensorGroup(STEM, tensors))
            is SHAPE_MISMATCH), label


def test_a_rank_zero_group_is_refused_where_the_scale_is_alpha_over_rank():
    """A rank-0 LoHa builds cleanly and applies an exactly zero delta -- the
    quietest possible failure. LoKr is the exception the rule needs: its
    full/full form IS rank 0 and scales by 1."""
    loha = {"hada_w1_a": _t(D_OUT, 0), "hada_w1_b": _t(0, D_IN),
            "hada_w2_a": _t(D_OUT, 0), "hada_w2_b": _t(0, D_IN)}
    group = TensorGroup(STEM, loha)
    assert group.missing() == () and group.rank == 0
    assert build_adapter_branch(_linear(), group) is SHAPE_MISMATCH

    full_lokr = TensorGroup(STEM, _lokr_tensors(factored=False))
    assert full_lokr.rank == 0
    branch = build_adapter_branch(_linear(), full_lokr)
    assert isinstance(branch, LoKrLinearLayer) and branch.scale == 1.0


def test_a_foreign_lokr_factorization_builds_rather_than_raising():
    """``factor`` is not stored in a checkpoint, so the geometry must come off
    the tensors: ``factorization(12, -1)`` is (3, 4) and this file is (2, 6)."""
    tensors = {"lokr_w1": _t(2, 2, seed=41), "lokr_w2": _t(6, 6, seed=42)}
    branch = build_adapter_branch(_linear(), TensorGroup(STEM, tensors))
    assert isinstance(branch, LoKrLinearLayer)
    assert branch.factors == ((2, 6), (2, 6))


# -- to_spec ---------------------------------------------------------------


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
@pytest.mark.parametrize("dora", [False, True])
def test_to_spec_normalizes_a_group_and_validates(algorithm, dora):
    tensors = ALGORITHMS[algorithm]()
    tensors["alpha"] = torch.tensor(ALPHA)
    if dora:
        tensors["dora_scale"] = torch.ones(D_OUT)

    spec = TensorGroup(STEM, tensors).to_spec(architecture="zimage").validate()

    assert spec.algorithm == algorithm and spec.weight_decompose is dora
    assert spec.rank == RANK and spec.alpha == ALPHA and spec.scale == ALPHA / RANK
    assert spec.architecture == "zimage" and spec.use_tucker is False


def test_a_full_lokr_carries_no_rank_and_therefore_no_alpha():
    """Upstream overrides ``alpha = lora_dim`` for the full/full form and the
    layer's ``scale`` ignores it, so keeping it would only trip
    ``validate()``'s "alpha with no rank"."""
    tensors = dict(_lokr_tensors(factored=False), alpha=torch.tensor(ALPHA))
    spec = TensorGroup(STEM, tensors).to_spec().validate()
    assert spec.rank is None and spec.alpha is None and spec.scale is None


def test_a_tucker_group_reaches_the_spec_refusal():
    tensors = dict(_loha_tensors(), hada_t1=torch.ones(2, 2, 1, 1))
    group = TensorGroup(STEM, tensors)
    assert group.use_tucker and group.to_spec().use_tucker
    with pytest.raises(Exception, match="Tucker"):
        group.to_spec().validate()
