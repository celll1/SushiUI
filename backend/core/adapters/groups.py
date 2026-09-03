"""Tensor grouping: checkpoint keys -> one stem's factor set -> a branch.

The eleven component-loader architectures each hand-write this: parse a key
into ``(module_path, suffix)``, bucket by module path, drop anything that is
not a complete ``down``/``up`` pair, then build a ``LoRALinearLayer``. Four of
them spell the drop identically (``if "down" in v and "up" in v``). This module
is the one implementation they migrate onto, extended to the LyCORIS factor
sets. NOTHING IMPORTS IT YET -- see the design doc's phase 2.

``TensorGroup`` answers to the legacy ``"down"``/``"up"``/``"alpha"`` spellings
as well as the canonical ones, so that migration can be additive rather than
eleven simultaneous rewrites. Why that matters, the removal condition, and the
refusals below are recorded in ``docs/guides/LYCORIS_ADAPTER_DESIGN.md``,
phase 2.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .layers import (TUCKER_TENSOR_NAMES, DoRALinearLayer, LoHaLinearLayer,
                     LoKrLinearLayer, LoRALinearLayer)
from .session import SHAPE_MISMATCH
from .spec import (ALGORITHM_LOHA, ALGORITHM_LOKR, ALGORITHM_LORA,
                   ALGORITHM_UNKNOWN, OPTION_USE_TUCKER, RANK_REQUIRED,
                   AdapterSpec)
from .targets import lora_branch_dtype

__all__ = [
    "ADAPTER_SUFFIXES",
    "GroupingResult",
    "TensorGroup",
    "build_adapter_branch",
    "group_adapter_tensors",
    "split_adapter_suffix",
    "split_group_on_out_rows",
]


#: Checkpoint suffix -> canonical tensor name, the LyCORIS 4.0.0 set plus the
#: PEFT spellings. The canonical names are what ``branch_tensors()`` produces,
#: so a group can be handed straight to ``load_tensors``.
ADAPTER_SUFFIXES: Mapping[str, str] = {
    ".lora_down.weight": "lora_down.weight",
    ".lora_up.weight": "lora_up.weight",
    ".lora_A.weight": "lora_down.weight",
    ".lora_B.weight": "lora_up.weight",
    ".lora_A.default.weight": "lora_down.weight",
    ".lora_B.default.weight": "lora_up.weight",
    ".lora_mid.weight": "lora_mid.weight",
    ".hada_w1_a": "hada_w1_a",
    ".hada_w1_b": "hada_w1_b",
    ".hada_w2_a": "hada_w2_a",
    ".hada_w2_b": "hada_w2_b",
    ".hada_t1": "hada_t1",
    ".hada_t2": "hada_t2",
    ".lokr_w1": "lokr_w1",
    ".lokr_w1_a": "lokr_w1_a",
    ".lokr_w1_b": "lokr_w1_b",
    ".lokr_w2": "lokr_w2",
    ".lokr_w2_a": "lokr_w2_a",
    ".lokr_w2_b": "lokr_w2_b",
    ".lokr_t2": "lokr_t2",
    ".dora_scale": "dora_scale",
    ".alpha": "alpha",
}

#: Longest first, so ``.lokr_w1_a`` cannot be read as ``.lokr_w1`` plus a stray
#: ``_a``. ``endswith`` is anchored and so is safe on today's table either way;
#: the order is what keeps a suffix that IS a suffix of another one -- the next
#: entry someone adds -- from resolving to the shorter name.
_SUFFIXES_LONGEST_FIRST: Tuple[str, ...] = tuple(
    sorted(ADAPTER_SUFFIXES, key=len, reverse=True))

#: Legacy spellings the eleven architecture parsers use today. Transitional:
#: delete them once no branch builder reads ``weights["down"]``.
_LEGACY_ALIASES: Mapping[str, str] = {
    "down": "lora_down.weight",
    "up": "lora_up.weight",
    "mid": "lora_mid.weight",
    "alpha": "alpha",
}

_LOHA_NAMES = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
_LORA_NAMES = ("lora_down.weight", "lora_up.weight")

_LAYER_CLASS = {
    ALGORITHM_LORA: LoRALinearLayer,
    ALGORITHM_LOHA: LoHaLinearLayer,
    ALGORITHM_LOKR: LoKrLinearLayer,
}


def split_adapter_suffix(key: str) -> Optional[Tuple[str, str]]:
    """``(stem, canonical name)`` for a key that names an adapter tensor."""
    for suffix in _SUFFIXES_LONGEST_FIRST:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], ADAPTER_SUFFIXES[suffix]
    return None


@dataclass
class TensorGroup(collections.abc.Mapping):
    """One stem's tensors, keyed by canonical name.

    A ``Mapping`` that ALSO answers to the legacy aliases: ``group["down"]`` is
    ``group["lora_down.weight"]`` and ``"down" in group`` is true, while
    iteration yields only canonical names. That asymmetry is the point -- it is
    what lets an architecture parser move onto this class without its branch
    builder changing in the same commit.
    """

    stem: str
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)

    # -- Mapping -----------------------------------------------------------

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[_LEGACY_ALIASES.get(name, name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return _LEGACY_ALIASES.get(name, name) in self.tensors

    # -- algebra -----------------------------------------------------------

    @property
    def algorithm(self) -> str:
        if any(name.startswith("hada_") for name in self.tensors):
            return ALGORITHM_LOHA
        if any(name.startswith("lokr_") for name in self.tensors):
            return ALGORITHM_LOKR
        if any(name.startswith("lora_") for name in self.tensors):
            return ALGORITHM_LORA
        return ALGORITHM_UNKNOWN

    @property
    def weight_decompose(self) -> bool:
        return "dora_scale" in self.tensors

    @property
    def use_tucker(self) -> bool:
        return bool(TUCKER_TENSOR_NAMES.intersection(self.tensors))

    @property
    def rank(self) -> int:
        """0 for the full/full LoKr, which has no rank to scale by."""
        algorithm = self.algorithm
        if algorithm == ALGORITHM_LORA and "lora_down.weight" in self.tensors:
            return int(self.tensors["lora_down.weight"].shape[0])
        if algorithm == ALGORITHM_LOHA and "hada_w1_a" in self.tensors:
            return int(self.tensors["hada_w1_a"].shape[1])
        if algorithm == ALGORITHM_LOKR:
            for name in ("lokr_w1_a", "lokr_w2_a"):
                if name in self.tensors:
                    return int(self.tensors[name].shape[1])
        return 0

    @property
    def alpha(self) -> Optional[float]:
        value = self.tensors.get("alpha")
        if value is None:
            return None
        return float(value.item()) if torch.is_tensor(value) else float(value)

    def missing(self) -> Tuple[str, ...]:
        """Required-but-absent canonical names; empty means complete."""
        algorithm = self.algorithm
        if algorithm == ALGORITHM_LORA:
            return tuple(n for n in _LORA_NAMES if n not in self.tensors)
        if algorithm == ALGORITHM_LOHA:
            return tuple(n for n in _LOHA_NAMES if n not in self.tensors)
        if algorithm == ALGORITHM_LOKR:
            absent: List[str] = []
            for full, (a, b) in (("lokr_w1", ("lokr_w1_a", "lokr_w1_b")),
                                 ("lokr_w2", ("lokr_w2_a", "lokr_w2_b"))):
                if full in self.tensors:
                    continue
                short = [n for n in (a, b) if n not in self.tensors]
                if short:
                    # Neither operand form is complete: name the whole operand
                    # when both factors are gone, the one factor otherwise.
                    absent.append(full if len(short) == 2 else short[0])
            return tuple(absent)
        # No algebra recognised -- never complete, whatever it holds.
        return _LORA_NAMES

    def to_spec(self, **kwargs) -> AdapterSpec:
        options = dict(kwargs.pop("options", None) or {})
        if self.use_tucker:
            options[OPTION_USE_TUCKER] = True
        rank = self.rank or None
        return AdapterSpec(
            algorithm=self.algorithm,
            weight_decompose=self.weight_decompose,
            rank=rank,
            # A rank-0 (full/full) LoKr's stored alpha is upstream's
            # ``lora_dim`` override and describes no scale -- the layer's
            # ``scale`` ignores it -- so carrying it here would only make
            # ``validate()`` refuse an alpha with no rank.
            alpha=None if rank is None else self.alpha,
            options=options,
            **kwargs,
        )


@dataclass(frozen=True)
class GroupingResult:
    """``groups`` are complete, ``partial`` are not, ``unmatched`` are the keys
    that named no adapter tensor at all.

    ``partial`` is RETURNED, never raised on -- every architecture drops
    incomplete groups silently today.
    """

    groups: Dict[str, TensorGroup] = field(default_factory=dict)
    partial: Dict[str, TensorGroup] = field(default_factory=dict)
    unmatched: Tuple[str, ...] = ()


def group_adapter_tensors(
    tensors: Mapping[str, torch.Tensor],
    stem_of: Optional[Callable[[str], Optional[str]]] = None,
) -> GroupingResult:
    """Bucket a checkpoint's tensors by stem.

    ``stem_of`` translates the suffix-stripped key into a module path and is
    where an architecture's key dialect lives; returning ``None`` means "not a
    key of mine" and sends it to ``unmatched``.
    """
    collected: Dict[str, TensorGroup] = {}
    unmatched: List[str] = []
    for key, tensor in tensors.items():
        split = split_adapter_suffix(key)
        if split is None:
            unmatched.append(key)
            continue
        raw_stem, name = split
        stem = raw_stem if stem_of is None else stem_of(raw_stem)
        if stem is None:
            unmatched.append(key)
            continue
        collected.setdefault(stem, TensorGroup(stem)).tensors[name] = tensor

    groups, partial = {}, {}
    for stem, group in collected.items():
        target = partial if group.missing() else groups
        target[stem] = group
    return GroupingResult(groups, partial, tuple(unmatched))


def split_group_on_out_rows(
    group: TensorGroup, n: int, inner: int,
) -> Optional[Dict[int, TensorGroup]]:
    """Split a group into ``n`` pieces of ``inner`` OUTPUT ROWS each.

    The engine-owned half of MiniMax-H3's fused-QKV mapping: ``delta[rows] =
    up[rows, :] @ down`` makes the row slice exact for ``lora`` (slice
    ``lora_up``) and for ``loha`` (slice ``hada_w1_a`` and ``hada_w2_a``), the
    ``_b`` factors being shared.

    ``lokr`` is different in kind: ``kron(w1, w2)`` puts row ``i*K + k`` at
    ``w1[i] (x) w2[k]``, so a piece is another Kronecker product UNDER THE
    PARENT'S OWN split only when it covers whole ``i`` blocks -- ``n`` dividing
    ``w1.shape[0]``. Anything else returns ``None``, because emitting the slice
    would be a numerically wrong adapter (measured 0.31 off at n=2). The
    refusal is conservative and the qualifier on "Kronecker product" is
    load-bearing; both are in the design doc, phase 2.

    Weight-decomposed and Tucker groups are refused too: ``dora_scale``'s
    ``(1, in)`` form has no row axis to slice at all.
    """
    if n <= 0 or inner <= 0 or group.missing() or group.use_tucker:
        return None
    if group.weight_decompose or "lora_mid.weight" in group:
        return None
    out_rows = n * inner

    def pieces(sliced_names: Tuple[str, ...], block: int) -> Dict[int, TensorGroup]:
        """Everything not row-sliced is shared by reference."""
        out = {}
        for index in range(n):
            tensors = dict(group.tensors)
            for name in sliced_names:
                tensors[name] = group[name][index * block:(index + 1) * block, :].contiguous()
            out[index] = TensorGroup(f"{group.stem}#{index}", tensors)
        return out

    algorithm = group.algorithm
    if algorithm == ALGORITHM_LORA:
        if int(group["lora_up.weight"].shape[0]) != out_rows:
            return None
        return pieces(("lora_up.weight",), inner)

    if algorithm == ALGORITHM_LOHA:
        if any(int(group[name].shape[0]) != out_rows
               for name in ("hada_w1_a", "hada_w2_a")):
            return None
        return pieces(("hada_w1_a", "hada_w2_a"), inner)

    if algorithm == ALGORITHM_LOKR:
        w1_name = "lokr_w1_a" if "lokr_w1" not in group else "lokr_w1"
        w2_name = "lokr_w2_a" if "lokr_w2" not in group else "lokr_w2"
        w1_rows = int(group[w1_name].shape[0])
        if w1_rows * int(group[w2_name].shape[0]) != out_rows or w1_rows % n != 0:
            return None
        return pieces((w1_name,), w1_rows // n)

    return None


def build_adapter_branch(
    base: nn.Module,
    group: TensorGroup,
    *,
    metadata_alpha: Optional[float] = None,
    layer_cls: Optional[type] = None,
    lora_dtype: Optional[torch.dtype] = None,
    lora_name: str = "",
):
    """A branch for ``base`` from one stem's tensors, or ``SHAPE_MISMATCH``.

    ``SHAPE_MISMATCH`` rather than an exception for the same reason the eleven
    loaders return it: one target whose shapes disagree is a module to skip,
    not a request to refuse.

    ``layer_cls`` overrides the algebra's default layer for an architecture
    that needs a subclass (MiniMax-H3 casts per call, having no autocast).

    EVERY read of the group's tensors happens inside the try: a rank-deficient
    factor, a 0-D weight or a two-element ``.alpha`` raises from a shape index
    or ``Tensor.item()``, not from a validated check. ``AttributeError`` is
    deliberately NOT caught -- a missing attribute is an engine bug, not a file
    defect, and swallowing it would silently apply nothing.
    """
    if group.missing() or group.use_tucker:
        return SHAPE_MISMATCH
    cls = layer_cls or _LAYER_CLASS.get(group.algorithm)
    if cls is None:
        return SHAPE_MISMATCH
    if lora_dtype is None:
        lora_dtype = lora_branch_dtype(base)

    try:
        # lora/loha scale by alpha/rank, so a rank-0 group is not a weak branch,
        # it is an exactly zero delta (and a ZeroDivisionError in the LoRA
        # layer). Same rule as ``AdapterSpec.validate``'s RANK_REQUIRED.
        if group.algorithm in RANK_REQUIRED and not group.rank:
            return SHAPE_MISMATCH
        # Per-key tensor wins, then file metadata, then rank -- the order
        # Z-Image's codec set. The tensor arm is inside ``from_tensors``.
        alpha = None if group.alpha is not None else metadata_alpha
        branch = cls.from_tensors(base, group, alpha=alpha, lora_dtype=lora_dtype,
                                  lora_name=lora_name or group.stem)
        if group.weight_decompose:
            branch = DoRALinearLayer(base, branch, dora_scale=group["dora_scale"])
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError,
            ZeroDivisionError):
        return SHAPE_MISMATCH
    return branch
