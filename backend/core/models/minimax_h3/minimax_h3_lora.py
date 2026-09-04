"""Inference-time LoRA support for the MiniMax-H3 DiT.

Loads a LoRA safetensors file and covers each matching vendored Linear with a
``CompositeAdapterLayer`` holding one ``MiniMaxH3LoRALinearLayer`` branch per
selected LoRA (forward-time addition, never a weight merge -- fully reversible
by restoring the original module), so two MiniMax-H3 LoRAs over one module sum.
The branch class is this architecture's own because its forward runs without
``torch.autocast``. This mirrors ``core.models.krea2.krea2_lora`` /
``core.models.anima.anima_lora`` in shape; see those modules for the general
pattern this one specialises.

TWO key conventions are supported, detected from the keys themselves.

1. The ComfyUI/"interchange" layout real MiniMax-H3 LoRAs ship in --

    diffusion_model.blocks.<N>.attn.qkv_proj.lora_A.weight
    diffusion_model.blocks.<N>.attn.qkv_proj.lora_B.weight
    diffusion_model.blocks.<N>.attn.qkv_proj.alpha            (optional)
    diffusion_model.token_refiner.blocks.<N>.<...>
    diffusion_model.final_layer.<...>

2. The sd-scripts native layout ``core.training.adapters.minimax_h3_adapter``
   writes for a LoRA trained inside this repo --

    lora_unet_transformer_blocks_<N>_attn_to_q.lora_down.weight
    lora_unet_transformer_blocks_<N>_attn_to_q.lora_up.weight
    lora_unet_transformer_blocks_<N>_attn_to_q.alpha

   These already target vendored module names one-to-one, so NONE of the three
   conversions below applies to them: no qkv fusion to split, no fc1 half swap.
   The only work is un-flattening the underscored stem, which is ambiguous in
   general and is therefore done against a table built from the training
   adapter's own scope constants (``_native_leaf_table``) rather than guessed.
   A stem that table cannot map, and a stem whose factor group is incomplete,
   both raise instead of being dropped -- a self-trained checkpoint that matched
   nothing used to be indistinguishable from a generation with no LoRA at all,
   and this repo writes every factor of every target it saves, so either is a
   real defect in the file. A COMPLETE LyCORIS group in this spelling is a
   target, not an incomplete pair.

Three conversions turn the Comfy layout into the vendored one (measured
against two real checkpoints -- see ``minimax_h3/loader.py``'s own DiT
state-dict mapping, which performs the identical three conversions on the
BASE weights and is the ground truth this module was checked against):

  (a) **qkv block-diagonal split.** ``attn.qkv_proj`` is one fused Linear
      whose LoRA ``lora_B`` is exactly block-diagonal across the three
      projections' OUTPUT rows -- but the RANK is split unevenly between
      to_q/to_k/to_v, differently in every block. The split is DERIVED from
      ``lora_B``'s own nonzero-column ranges per output third, never assumed
      to be an equal three-way split of the rank; a malformed or unexpected
      LoRA raises rather than silently producing a wrong split.
      See ``_split_qkv``. A LoHa/LoKr goes through the engine's
      ``split_group_on_out_rows`` instead (``_split_qkv_targets``), which is
      exact for LoHa and REFUSES a LoKr whose ``w1`` rows are not divisible by
      three rather than emitting a different factorization.

  (b) **fc1 SwiGLU half swap.** Comfy stores ``mlp.fc1`` as ``[gate; up]``;
      the vendored ``ff.net.0.proj`` (SwiGLU) expects ``[up; gate]``. This is
      a ROW PERMUTATION of the OUT axis only -- the input-side factors are
      untouched -- so it moves ``lora_B``, or ``hada_w1_a``/``hada_w2_a``, or a
      LoKr's ``w1`` and only when its rows are even. Getting this
      backwards is silent: shapes match, the load is clean, and the gate
      delta lands in the up path. See ``_swap_fc1_group``.

  (c) **Scale.** Final per-module scale is ``(alpha / rank) * user_strength``,
      exactly as ``krea2_lora.apply_lora_group`` computes it -- where
      ``rank`` is the FUSED qkv stem's TOTAL rank (before the block-diagonal
      split), not any individual projection's post-split rank: the ratio is
      what the original module was scaled by, and each split piece inherits
      that same ratio so the sum of the three pieces reproduces the
      original, undivided delta.

      Alpha resolution differs BY CONVENTION, and deliberately so:

        * Comfy: per-key ``.alpha``, else ``alpha = rank`` (scale 1.0). File
          metadata is NOT a fallback tier here. These checkpoints drop alpha
          and bake a flat multiplier straight into ``lora_B`` instead of
          relying on alpha/rank (real ``lightx2v_turbo_4step``: no per-key
          alphas, ``ss_network_alpha: 'Dynamic'``, and a ``conversion`` note
          saying so), so honouring a numeric ``ss_network_alpha`` alongside
          would silently double-attenuate them.
        * Native: per-key ``.alpha``, else file
          ``lora_alpha``/``ss_network_alpha``, else rank. The training adapter
          writes both, and its metadata alpha means the alpha/rank ratio.

Save format reference: ``core/training/adapters/minimax_h3_adapter.py``
(sd-scripts native, the OUTPUT side of convention 2 above).
"""

from __future__ import annotations

import collections.abc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn
from safetensors import safe_open

from core.adapters.groups import (TensorGroup, declared_groups,
                                  group_adapter_tensors,
                                  split_group_on_out_rows)
from core.adapters.spec import ALGORITHM_LOHA, ALGORITHM_LOKR, ALGORITHM_LORA


_PREFIX = "diffusion_model."
_NATIVE_PREFIX = "lora_unet_"
_QKV_SUFFIX = ".attn.qkv_proj"
_FC1_SUFFIX = ".mlp.fc1"


# ---------------------------------------------------------------------------
# Raw key parsing
# ---------------------------------------------------------------------------

def _comfy_stem(raw_stem: str) -> Optional[str]:
    """Suffix-stripped key -> comfy module stem, or None for a foreign key."""
    if not raw_stem.startswith(_PREFIX):
        return None
    return raw_stem[len(_PREFIX):]


def _native_stem(raw_stem: str) -> Optional[str]:
    """Suffix-stripped key -> flattened sd-scripts stem, or None."""
    if not raw_stem.startswith(_NATIVE_PREFIX):
        return None
    return raw_stem[len(_NATIVE_PREFIX):]


_NATIVE_STEM_RE = re.compile(r"^transformer_blocks_(\d+)_(.+)$")


_NATIVE_LEAF_TABLE: Optional[Dict[str, str]] = None


def _native_leaf_table() -> Dict[str, str]:
    """``{flattened_leaf: dotted_leaf}`` for every leaf the training adapter
    can target, derived from ITS constants so the two cannot drift apart.

    Un-flattening ``attn_to_out_0`` back to ``attn.to_out.0`` is ambiguous by
    inspection (``to_out_0`` could be an attribute of that name); the table is
    what makes it exact. Memoized: it is consulted once per stem, and the
    import is deferred (training package, imported from a generation module).
    """
    global _NATIVE_LEAF_TABLE
    if _NATIVE_LEAF_TABLE is None:
        from core.training.adapters.minimax_h3_adapter import _ATTN_LEAVES, _FF_LEAVES

        leaves = [f"attn.{leaf}" for leaf in _ATTN_LEAVES] + [f"ff.{leaf}" for leaf in _FF_LEAVES]
        _NATIVE_LEAF_TABLE = {leaf.replace(".", "_"): leaf for leaf in leaves}
    return _NATIVE_LEAF_TABLE


def _native_stem_to_module_path(stem: str, table: Dict[str, str]) -> Optional[str]:
    match = _NATIVE_STEM_RE.match(stem)
    if match is None:
        return None
    leaf = table.get(match.group(2))
    if leaf is None:
        return None
    return f"transformer_blocks.{match.group(1)}.{leaf}"


def _metadata_alpha(metadata: Optional[Dict[str, str]]) -> Optional[float]:
    """File-level ``lora_alpha``/``ss_network_alpha``, or ``None``.

    The NATIVE branch's middle alpha tier (per-key tensor -> file metadata ->
    rank), reached only when a native checkpoint carries no per-key ``.alpha``.
    The comfy branch must not consult this -- see the module docstring, (c).
    """
    for key in ("lora_alpha", "ss_network_alpha"):
        value = (metadata or {}).get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_lora_safetensors(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Load a LoRA safetensors file. Returns ``(raw_state_dict, metadata)``."""
    raw: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    return raw, metadata


def _group_raw(raw: Dict[str, torch.Tensor]) -> Dict[str, TensorGroup]:
    """COMPLETE comfy factor groups, whatever the algebra.

    ``group_adapter_tensors`` already drops the incomplete ones, which
    ``count_declared_branches`` still counts, so a truncated file is refused
    unapplied rather than applying with a target missing.
    """
    return group_adapter_tensors(raw, _comfy_stem).groups


@dataclass
class ConvertedTarget(collections.abc.Mapping):
    """One vendored target's factors plus the pair that DEFINES its scale.

    ``scale_rank`` is the FUSED qkv stem's rank, which after a compact split is
    NOT this piece's own tensor rank (module docstring, (c)). A LyCORIS row
    split keeps the rank columns, so for those the two agree.

    A ``Mapping`` over ``group``, answering the legacy ``["down"]``/
    ``["scale_ratio"]`` spellings its callers still use.
    """

    group: TensorGroup
    alpha: float
    scale_rank: int
    #: The file-metadata alpha tier: native only, never comfy (docstring, (c)).
    metadata_alpha: Optional[float] = None

    _SCALARS = ("scale_ratio", "alpha", "scale_rank")

    @property
    def scale_ratio(self) -> float:
        return self.alpha / self.scale_rank

    def __getitem__(self, name: str) -> Any:
        if name in self._SCALARS:
            return self.scale_ratio if name == "scale_ratio" else getattr(self, name)
        return self.group[name]

    def __iter__(self) -> Iterator[str]:
        names = list(self.group.tensors)
        yield from names
        for name in self._SCALARS:
            if name not in names:
                yield name

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, name: object) -> bool:
        return name in self._SCALARS or name in self.group


def count_declared_branches(raw: Dict[str, torch.Tensor]) -> int:
    """Branches this file declares (see ``declared_groups``), counting a fused
    comfy ``qkv_proj`` stem as the THREE targets ``_split_qkv`` turns it into.

    Does not run the conversion, so unlike ``normalise_lora_state_dict`` it
    never raises.
    """
    native = any(key.startswith(_NATIVE_PREFIX) for key in raw)
    stems = declared_groups(raw, _native_stem if native else _comfy_stem)
    if native:
        return len(stems)
    return sum(3 if stem.endswith(_QKV_SUFFIX) else 1 for stem in stems)


# ---------------------------------------------------------------------------
# (a) qkv block-diagonal split
# ---------------------------------------------------------------------------

def _try_derive_compact_qkv_ranges(
    up: torch.Tensor, r: int, inner: int,
) -> Optional[list]:
    """Attempt to read a PER-COMPONENT rank-column split off ``up``'s own
    block-diagonal structure. Returns three ``(lo, hi)`` ranges that are each
    contiguous, mutually disjoint and together cover ``0..r`` -- or ``None``
    when ``up`` is not shaped this way (measured live on the real
    ``fl2va_4step_lora`` checkpoint: a genuinely dense/shared qkv adapter,
    where every rank column contributes to all three projections at once,
    not one that was assembled by fusing three separately-ranked adapters).
    ``None`` is not a refusal -- ``_split_qkv``'s caller falls back to the
    always-exact shared-``down`` split below, never to a guess.
    """
    ranges = []
    for block in range(3):
        rows = up[block * inner:(block + 1) * inner, :]
        nonzero_cols = torch.nonzero(rows.abs().sum(dim=0) != 0, as_tuple=True)[0]
        if nonzero_cols.numel() == 0:
            return None
        lo = int(nonzero_cols.min().item())
        hi = int(nonzero_cols.max().item()) + 1
        expected = torch.arange(lo, hi, device=nonzero_cols.device)
        sorted_cols = torch.sort(nonzero_cols).values
        if sorted_cols.numel() != expected.numel() or not torch.equal(sorted_cols, expected):
            return None
        ranges.append((lo, hi))
    for i in range(2):
        if ranges[i][1] != ranges[i + 1][0]:
            return None
    if ranges[0][0] != 0 or ranges[-1][1] != r:
        return None
    return ranges


def _split_qkv(
    stem: str, down: torch.Tensor, up: torch.Tensor,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], int]:
    """Split a fused qkv LoRA into three ``(down, up)`` pairs.

    TWO EXACT paths, tried in order -- neither is a guess:

    1. **Compact (block-diagonal) split**, when ``up``'s own structure
       supports it (measured on the real ``lightx2v_turbo_4step`` checkpoint:
       50 main + 2 token-refiner stems, 31 distinct rank-triples, e.g.
       ``(2,2,2)`` up to ``(38,52,4)`` -- NEVER an equal three-way ``r // 3``
       split): each projection gets only its own rank-column slice of
       ``down``, derived from ``up``'s nonzero-column ranges
       (``_try_derive_compact_qkv_ranges``). Smaller per-target rank, exact.

    2. **General (shared-down) split**, used whenever (1) does not apply
       (measured live on the real ``fl2va_4step_lora`` checkpoint: its
       ``up`` is dense, every rank column active in every third). ``down``
       (lora_A, the shared INPUT projection over the fused Linear's 5376
       input features) is identical for every one of q/k/v regardless of
       ``up``'s structure -- the identity ``delta[rows] = up[rows, :] @ down``
       holds unconditionally, so keeping the FULL rank ``r`` and slicing
       ``up`` by OUTPUT ROW ONLY reproduces the fused delta exactly, with
       nothing guessed and nothing dropped. This is what a normal (not
       resize-fused) qkv-targeting LoRA looks like.

    Returns ``({"to_q": (down, up), "to_k": ..., "to_v": ...}, total_rank)``;
    ``total_rank`` is the FUSED stem's rank, for the caller's alpha/rank scale
    (see the module docstring, point (c)) -- unaffected by which path ran.
    """
    r = int(down.shape[0])
    if up.shape[1] != r:
        raise ValueError(
            f"{stem}: lora_A rank {r} does not match lora_B's {up.shape[1]} input columns."
        )
    total_out = int(up.shape[0])
    if total_out % 3 != 0:
        raise ValueError(
            f"{stem}: fused qkv lora_B has {total_out} output rows, not divisible by 3 -- cannot "
            f"be a [q_all | k_all | v_all] fused qkv projection."
        )
    inner = total_out // 3

    ranges = _try_derive_compact_qkv_ranges(up, r, inner)
    parts: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    if ranges is not None:
        for name, block, (lo, hi) in zip(("to_q", "to_k", "to_v"), range(3), ranges):
            parts[name] = (
                down[lo:hi, :].contiguous(),
                up[block * inner:(block + 1) * inner, lo:hi].contiguous(),
            )
    else:
        for name, block in zip(("to_q", "to_k", "to_v"), range(3)):
            parts[name] = (
                down.clone(),
                up[block * inner:(block + 1) * inner, :].contiguous(),
            )
    return parts, r


_QKV_NAMES = ("to_q", "to_k", "to_v")


def _fused_out_rows(group: TensorGroup) -> int:
    """The fused Linear's output rows, from whichever factors carry that axis."""
    algorithm = group.algorithm
    if algorithm == ALGORITHM_LORA:
        return int(group["lora_up.weight"].shape[0])
    if algorithm == ALGORITHM_LOHA:
        return int(group["hada_w1_a"].shape[0])
    if algorithm == ALGORITHM_LOKR:
        w1 = group["lokr_w1"] if "lokr_w1" in group else group["lokr_w1_a"]
        w2 = group["lokr_w2"] if "lokr_w2" in group else group["lokr_w2_a"]
        return int(w1.shape[0]) * int(w2.shape[0])
    raise ValueError(f"unrecognised adapter algebra {algorithm!r} on a fused stem")


def _own_shared_tensors(parent: TensorGroup, piece: TensorGroup) -> TensorGroup:
    """Clone what ``split_group_on_out_rows`` shares BY REFERENCE.

    ``from_tensors`` ADOPTS a group's tensors as parameter storage, so without
    this the three qkv branches alias one buffer and any in-place write reaches
    all three. ``_split_qkv`` already clones ``lora_down`` for the same reason.
    """
    tensors = {name: (value.clone() if value is parent.tensors.get(name) else value)
               for name, value in piece.tensors.items()}
    return TensorGroup(piece.stem, tensors)


def _qkv_split_refusal(stem: str, group: TensorGroup) -> str:
    """Why a WELL-FORMED LyCORIS group cannot be split three ways.

    Kept distinct from the truncated/corrupt wording: nothing is wrong with the
    file, its factorization simply does not line up with the fused projection.
    """
    if group.use_tucker or group.weight_decompose or "lora_mid.weight" in group:
        return (f"{stem}: a fused qkv {group.algorithm} carrying a Tucker core, a "
                f"LoCon mid factor or a DoRA magnitude has no output-row split -- "
                f"those factors mix or rescale rows, so no contiguous third is a "
                f"sub-adapter of the fused projection.")
    if group.algorithm == ALGORITHM_LOKR:
        w1 = group["lokr_w1"] if "lokr_w1" in group else group["lokr_w1_a"]
        return (f"{stem}: a fused qkv LoKr cannot be split into to_q/to_k/to_v. "
                f"kron(w1, w2) puts output row i*K + k at w1[i] (x) w2[k], so a "
                f"contiguous third is another Kronecker product only when 3 "
                f"divides w1's {int(w1.shape[0])} rows, and it does not. The file "
                f"is well formed; its factorization does not line up with this "
                f"architecture's fused qkv projection.")
    return (f"{stem}: a fused qkv {group.algorithm} adapter could not be split "
            f"into to_q/to_k/to_v -- its factor shapes disagree with the fused "
            f"projection's output rows.")


def _split_qkv_targets(
    stem: str, group: TensorGroup,
) -> List[Tuple[str, TensorGroup, int]]:
    """``(target leaf, its factor group, the scale-defining rank)`` x 3.

    LoRA keeps ``_split_qkv``'s two exact paths byte for byte. Every other
    algebra goes through the engine's ``split_group_on_out_rows``, which is
    exact for LoHa and refuses a LoKr whose ``w1`` rows are not divisible by 3
    rather than emitting a wrong factorization.
    """
    # A LoRA piece is rebuilt from down/up alone, so an extra factor would be
    # DROPPED rather than refused -- and the non-fused path refuses it.
    if group.weight_decompose or "lora_mid.weight" in group:
        raise ValueError(_qkv_split_refusal(stem, group))
    if group.algorithm == ALGORITHM_LORA:
        parts, rank_total = _split_qkv(stem, group["down"], group["up"])
        return [(name,
                 TensorGroup(f"{stem}#{name}", {"lora_down.weight": parts[name][0],
                                                "lora_up.weight": parts[name][1]}),
                 rank_total)
                for name in _QKV_NAMES]

    total_out = _fused_out_rows(group)
    if total_out % 3 != 0:
        raise ValueError(
            f"{stem}: fused qkv adapter has {total_out} output rows, not divisible "
            f"by 3 -- cannot be a [q_all | k_all | v_all] fused qkv projection.")
    pieces = split_group_on_out_rows(group, 3, total_out // 3)
    if pieces is None:
        raise ValueError(_qkv_split_refusal(stem, group))
    rank = group.rank or 1
    return [(name, _own_shared_tensors(group, pieces[index]), rank)
            for index, name in enumerate(_QKV_NAMES)]


# ---------------------------------------------------------------------------
# (b) fc1 SwiGLU half swap
# ---------------------------------------------------------------------------

def _swap_fc1_halves(up: torch.Tensor) -> torch.Tensor:
    """Comfy ``[gate; up]`` -> vendored SwiGLU ``[up; gate]``, row permutation of
    ``lora_B`` only. Mirrors ``loader._map_dit_state_dict``'s base-weight swap."""
    gate, up_half = up.chunk(2, dim=0)
    return torch.cat([up_half, gate], dim=0).contiguous()


def _fc1_row_factors(group: TensorGroup) -> Tuple[str, ...]:
    """The factor names carrying the OUT axis, or ``()`` when the half swap is
    not expressible in this group's stored form.

    Generalized from ``lora_up``: the swap is a permutation of the output rows,
    so it also moves ``hada_w1_a``/``hada_w2_a``. For a LoKr it moves ``w1``
    alone, and only when ``w1``'s row count is even -- kron puts row ``i*K + k``
    at ``w1[i] (x) w2[k]``, so the half boundary is a block boundary exactly
    then.
    """
    algorithm = group.algorithm
    if algorithm == ALGORITHM_LORA:
        return ("lora_up.weight",)
    if algorithm == ALGORITHM_LOHA:
        return ("hada_w1_a", "hada_w2_a")
    if algorithm == ALGORITHM_LOKR:
        name = "lokr_w1" if "lokr_w1" in group else "lokr_w1_a"
        return (name,) if int(group[name].shape[0]) % 2 == 0 else ()
    return ()


def _fc1_swap_refusal(stem: str, group: TensorGroup) -> str:
    """Why a WELL-FORMED group cannot take the fc1 half swap. Same distinction
    as ``_qkv_split_refusal``: the file is fine, its factorization is not."""
    if group.algorithm == ALGORITHM_LOKR:
        w1 = group["lokr_w1"] if "lokr_w1" in group else group["lokr_w1_a"]
        return (f"{stem}: an fc1 LoKr cannot be converted to the vendored SwiGLU "
                f"order. The comfy [gate; up] -> [up; gate] swap is a permutation "
                f"of the output rows, which kron(w1, w2) expresses only when w1's "
                f"{int(w1.shape[0])} rows are even, and they are not. The file is "
                f"well formed; its factorization does not line up with this "
                f"architecture's SwiGLU projection.")
    return (f"{stem}: an fc1 {group.algorithm} adapter has no factor carrying "
            f"the output-row axis, so the vendored SwiGLU half swap cannot be "
            f"expressed in its stored form.")


def _swap_fc1_group(stem: str, group: TensorGroup) -> TensorGroup:
    """The fc1 half swap applied to a whole factor group."""
    names = _fc1_row_factors(group)
    if not names:
        raise ValueError(_fc1_swap_refusal(stem, group))
    tensors = dict(group.tensors)
    for name in names:
        tensors[name] = _swap_fc1_halves(tensors[name])
    return TensorGroup(group.stem, tensors)


# ---------------------------------------------------------------------------
# Full conversion: raw state dict -> {vendored_target_path: ConvertedTarget}
# ---------------------------------------------------------------------------

def _resolve_alpha(alpha_tensor, metadata_alpha: Optional[float], rank: int) -> float:
    """Per-key ``.alpha`` tensor -> ``metadata_alpha`` -> rank. Comfy callers
    pass ``metadata_alpha=None`` (module docstring, (c))."""
    if alpha_tensor is not None:
        return float(alpha_tensor.item())
    if metadata_alpha is not None:
        return metadata_alpha
    return float(rank)


def _normalise_comfy(raw: Dict[str, torch.Tensor]) -> Dict[str, ConvertedTarget]:
    """ComfyUI/interchange layout -> vendored targets. Takes no metadata: this
    branch's alpha is per-key or rank, never the file's (module docstring, (c))."""
    from core.models.minimax_h3.loader import _rename_dit_key

    targets: Dict[str, ConvertedTarget] = {}

    def record(path: str, stem: str, target: ConvertedTarget) -> None:
        if path in targets:
            raise ValueError(f"duplicate LoRA target {path!r} (from stem {stem!r})")
        targets[path] = target

    for stem, group in _group_raw(raw).items():
        mapped = _rename_dit_key(stem + ".weight")
        if not mapped.endswith(".weight"):
            raise ValueError(f"{stem}: unexpected renamed key {mapped!r} (expected a .weight suffix)")
        mapped = mapped[: -len(".weight")]

        if stem.endswith(_QKV_SUFFIX):
            base = mapped.split(".attn.qkv_proj")[0] + ".attn."
            for name, piece, rank_total in _split_qkv_targets(stem, group):
                alpha_value = _resolve_alpha(group.tensors.get("alpha"), None, rank_total)
                record(base + name, stem, ConvertedTarget(piece, alpha_value, rank_total))
            continue

        if stem.endswith(_FC1_SUFFIX):
            group = _swap_fc1_group(stem, group)

        rank = group.rank
        alpha_value = _resolve_alpha(group.tensors.get("alpha"), None, rank)
        record(mapped, stem, ConvertedTarget(group, alpha_value, rank or 1))

    return targets


def _normalise_native(
    raw: Dict[str, torch.Tensor], metadata_alpha: Optional[float],
) -> Dict[str, ConvertedTarget]:
    """sd-scripts native (this repo's own trainer output) -> vendored targets.

    One-to-one with the vendored module names: no qkv split, no fc1 half swap.
    Only the underscored stem is un-flattened, against the training adapter's
    own leaf table.
    """
    collected = group_adapter_tensors(raw, _native_stem)
    table = _native_leaf_table()
    targets: Dict[str, ConvertedTarget] = {}
    unmapped: list = []
    # An incomplete group is a target the file names and no builder can apply,
    # whatever its algebra; a COMPLETE LoHa/LoKr one is not incomplete.
    incomplete: list = sorted(collected.partial)
    for stem, group in collected.groups.items():
        module_path = _native_stem_to_module_path(stem, table)
        if module_path is None:
            unmapped.append(stem)
            continue
        rank = group.rank
        if group.algorithm == ALGORITHM_LORA and int(group["up"].shape[1]) != rank:
            raise ValueError(
                f"{stem}: lora_down rank {rank} does not match lora_up's "
                f"{group['up'].shape[1]} columns."
            )
        if module_path in targets:
            raise ValueError(f"duplicate LoRA target {module_path!r} (from stem {stem!r})")
        alpha_value = _resolve_alpha(group.tensors.get("alpha"), metadata_alpha, rank)
        targets[module_path] = ConvertedTarget(group, alpha_value, rank or 1,
                                               metadata_alpha)

    if incomplete:
        raise ValueError(
            f"{len(incomplete)} sd-scripts LoRA stem(s) carry an incomplete factor "
            f"group (first few: {incomplete[:5]}); this repo's trainer writes every "
            f"factor of every target it saves, so the file is truncated or corrupt."
        )
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} sd-scripts LoRA stem(s) name no MiniMax-H3 LoRA target "
            f"(first few: {sorted(unmapped)[:5]}); recognised leaves are "
            f"{sorted(table)}."
        )
    return targets


def normalise_lora_state_dict(
    raw: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, ConvertedTarget]:
    """Group + convert a raw LoRA state dict into vendored targets.

    Returns ``{vendored_module_path: ConvertedTarget}``, which reads as the
    ``{"down", "up", "scale_ratio", "alpha", "scale_rank"}`` mapping it replaced
    and additionally carries the whole factor ``group``, whatever its algebra.
    ``scale_ratio`` is ``alpha / scale_rank`` (``scale_rank`` being the FUSED
    rank for a qkv split's three pieces, see point (c) in the module docstring);
    the caller multiplies by the user-supplied LoRA strength. The pair is
    carried alongside the ratio because after a compact qkv split it is NOT this
    branch's own tensor rank, and a branch that only knows the ratio cannot be
    restrengthened later without recomputing it from the wrong pair.

    The convention is read off the keys. ``metadata`` supplies the middle alpha
    tier (per-key tensor -> file metadata -> rank) on the NATIVE branch only;
    the comfy branch never consults it (module docstring, (c)).
    """
    has_comfy = any(key.startswith(_PREFIX) for key in raw)
    has_native = any(key.startswith(_NATIVE_PREFIX) for key in raw)
    if has_comfy and has_native:
        raise ValueError(
            "LoRA mixes the ComfyUI (diffusion_model.*) and sd-scripts (lora_unet_*) key "
            "conventions; they need different conversions and cannot be applied together."
        )
    if has_native:
        return _normalise_native(raw, _metadata_alpha(metadata))
    return _normalise_comfy(raw)


# ---------------------------------------------------------------------------
# Variant guard
# ---------------------------------------------------------------------------

_VARIANT_TOKENS = ("ref2va", "fl2va")


def _detect_variant_token(text: str) -> Optional[str]:
    lowered = text.lower()
    for token in _VARIANT_TOKENS:
        if token in lowered:
            return token
    return None


def check_variant_compatibility(
    metadata: Dict[str, str], lora_path: str, current_variant: Optional[str],
    warn: Callable[[str, str], None],
) -> None:
    """Refuse (raise) or warn when a LoRA's declared/implied variant conflicts
    with the loaded checkpoint's variant.

    ``fl2va`` and ``ref2va`` checkpoints are byte-size-identical with
    identical keys and shapes, so a wrong-variant LoRA is UNDETECTABLE by key
    or shape -- it loads clean and applies silently wrong. When
    ``metadata["base_model"]`` names a variant explicitly, a mismatch is a
    hard refusal. When metadata carries no variant (real ``F1``'s case), fall
    back to a filename substring check and only WARN -- a filename is not
    proof, but it is the only signal left.

    On a ``hybrid`` checkpoint a LoRA WARNS and loads (design section 5.3
    allowed either; the repo owner chose warn-over-refuse). What it cannot do
    is state which merge it was trained for, so the caveat is surfaced through
    ``warn`` -- the same channel as the undeclared case, which reaches the
    generation's ``warnings[]`` and not only the console. A LoRA that DECLARES
    ``fl2va``/``ref2va`` is still refused below: that guard predates the merge.
    """
    current = (current_variant or "").lower()
    base_model = str(metadata.get("base_model", "") or "")
    name = Path(lora_path).name

    if current == "hybrid":
        warn(
            f"LoRA '{lora_path}' is being applied to a merged (hybrid) MiniMax-H3 checkpoint. A "
            f"hybrid is an fl2va base carrying ref2va AdaLN blocks over a block range; no LoRA "
            f"metadata names an AdaLN recipe, and every MiniMax-H3 partition shares its keys and "
            f"shapes, so no LoRA can state which merge it was trained for and its weights cannot "
            f"reveal one. Nothing about a LoRA on a merged checkpoint was measured.",
            "minimax_h3_lora_hybrid_unmeasured",
        )
        # Fall through ONLY for a LoRA that names a partition: that declaration
        # contradicts the merge it is being applied to, and refusing it is the
        # pre-existing guard, not this one. Everything else is warned and loads.
        if not (base_model and _detect_variant_token(base_model)):
            return

    if base_model:
        declared = _detect_variant_token(base_model)
        if declared is not None and current and declared != current:
            raise ValueError(
                f"LoRA '{lora_path}' declares base_model={base_model!r} (variant={declared!r}), "
                f"but the loaded MiniMax-H3 checkpoint is the {current!r} variant. fl2va and "
                f"ref2va checkpoints are byte-size-identical with identical keys and shapes -- a "
                f"wrong-variant LoRA cannot be detected from its weights and would apply "
                f"silently wrong. Refusing to load this LoRA."
            )
        return

    declared = _detect_variant_token(name)
    if declared is not None and current and declared != current:
        warn(
            f"LoRA '{lora_path}' carries no base_model metadata; its filename names the "
            f"{declared!r} variant but the loaded checkpoint is {current!r}. fl2va and ref2va "
            f"checkpoints are byte-size-identical with identical keys and shapes, so a "
            f"wrong-variant LoRA cannot be detected from its contents -- verify this LoRA was "
            f"trained for the loaded variant before trusting its output.",
            "minimax_h3_lora_variant_ambiguous",
        )
    elif declared is None:
        warn(
            f"LoRA '{lora_path}' carries no base_model metadata and its filename does not name a "
            f"variant (fl2va/ref2va). fl2va and ref2va checkpoints are byte-size-identical with "
            f"identical keys and shapes, so a wrong-variant LoRA cannot be detected from its "
            f"contents -- verify this LoRA was trained for the loaded variant "
            f"({current or 'unknown'}) before trusting its output.",
            "minimax_h3_lora_variant_unknown",
        )


# ---------------------------------------------------------------------------
# Rank-variation-across-blocks detection (block swap interaction)
# ---------------------------------------------------------------------------

_BLOCK_LEAF_RE = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")


def detect_rank_variation(targets: Dict[str, "ConvertedTarget"]) -> Dict[str, bool]:
    """``{leaf_name: True}`` for every ``transformer_blocks.*`` leaf whose rank
    differs between blocks.

    Only ``transformer_blocks.*`` entries matter here:
    ``TransformerBlockOffloader._build_weight_swap_jobs``
    (``core/memory_management/block_offloading.py``) pairs an incoming and an
    outgoing block's Linear weights by name + shape + dtype, and only the
    block stack is ever swapped -- ``token_refiner`` and ``final_layer``
    leaves are moved whole, never paired block-to-block.

    LoHa/LoKr branches are skipped: they own no ``nn.Linear`` for that pairing
    to reach at all (see ``BLOCK_SWAP_ADAPTER_ORDER``).
    """
    ranks_by_leaf: Dict[str, set] = {}
    for module_path, target in targets.items():
        m = _BLOCK_LEAF_RE.match(module_path)
        if not m or target.group.algorithm != ALGORITHM_LORA:
            continue
        leaf = m.group(2)
        ranks_by_leaf.setdefault(leaf, set()).add(target.group.rank)
    return {leaf: len(ranks) > 1 for leaf, ranks in ranks_by_leaf.items()}


# ---------------------------------------------------------------------------
# Apply / restore (inference)
# ---------------------------------------------------------------------------

def build_lora_branch(base: nn.Module, target: "ConvertedTarget",
                      module_path: str) -> nn.Module:
    """One branch over ``base``, or ``SHAPE_MISMATCH``.

    The algebra is the file's: ``build_adapter_branch`` dispatches on the tensor
    names. Only the LoRA branch takes this architecture's per-call activation
    cast -- MiniMax-H3's forward runs without ``torch.autocast``, and LoHa/LoKr
    already cast their merged delta inside ``reference_delta``.

    The branch dtype is per target, never bf16 by hand: this tree is mixed
    precision (Fp8Linear block stack, fp32 AdaLN and head projections).
    """
    from core.adapters import (SHAPE_MISMATCH, MiniMaxH3LoRALinearLayer,
                               build_adapter_branch, lora_branch_dtype)

    is_lora = target.group.algorithm == ALGORITHM_LORA
    branch = build_adapter_branch(
        base, target.group,
        metadata_alpha=target.metadata_alpha,
        layer_cls=MiniMaxH3LoRALinearLayer if is_lora else None,
        lora_dtype=lora_branch_dtype(base),
        lora_name=module_path)
    if branch is SHAPE_MISMATCH:
        return branch
    if is_lora:
        # The scale-defining pair, NOT the branch's own tensor rank: after a
        # compact qkv split the ratio belongs to the FUSED stem (docstring, (c)).
        # ``set_adapter_strength`` then recomputes ``alpha / rank * strength``.
        branch.alpha = target.alpha
        branch.rank = target.scale_rank
    return branch


def apply_lora_group(
    transformer: nn.Module,
    targets: Dict[str, "ConvertedTarget"],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: set,
    branch_name: str = "lora",
) -> Tuple[int, list]:
    """Add one named branch per target to the covering ``CompositeAdapterLayer``.

    Returns ``(applied_count, missing_target_paths)``. Each target Linear is
    covered ONCE by a composite and each selected LoRA adds a NAMED branch, so
    two LoRAs over the same module SUM instead of the second being dropped;
    ``branch_name`` must be unique within the request (``add_branch`` refuses a
    duplicate). The branch itself stays a ``MiniMaxH3LoRALinearLayer`` -- this
    architecture's forward runs without ``torch.autocast`` and needs that
    class's per-call activation cast -- and the composite drives it through
    ``forward_delta``, never through its class.

    Target resolution is KEY-DRIVEN (``_resolve_leaf`` over the checkpoint's own
    module paths), which is also what restore walks, so load and unload cannot
    disagree about which slot a target lives in.

    ``lora_original_modules.setdefault`` records only the FIRST original seen
    for a module path, so ``restore_originals`` always reaches the un-LoRA'd
    module.

    The caller decides how loudly to report unmatched targets (a LoRA trained
    against a different scope, e.g. only ``attention``, legitimately leaves
    ``ff``/``adaln`` targets unmatched by the MODEL side -- but a target this
    function cannot even RESOLVE against the live module tree is a real
    problem worth surfacing).
    """
    from core.adapters import (
        SHAPE_MISMATCH,
        CompositeAdapterLayer,
        is_lora_wrappable_linear,
    )
    from core.training.adapters.minimax_h3_adapter import _resolve_leaf

    applied = 0
    missing: list = []
    for module_path, weights in targets.items():
        resolved = _resolve_leaf(transformer, module_path)
        if resolved is None:
            missing.append(module_path)
            continue
        parent, attr, current = resolved

        true_original = (current.original_module
                         if isinstance(current, CompositeAdapterLayer) else current)
        if not is_lora_wrappable_linear(true_original):
            missing.append(module_path)
            continue

        wrapper = build_lora_branch(true_original, weights, module_path)
        if wrapper is SHAPE_MISMATCH:
            missing.append(module_path)
            continue
        lora_original_modules.setdefault(module_path, true_original)

        composite = CompositeAdapterLayer.attach(parent, attr)
        composite.add_branch(branch_name, wrapper, strength=strength)
        wrapped_keys.add(module_path)
        applied += 1

    return applied, missing


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: set,
) -> int:
    """Revert every composite-covered module to its pre-LoRA original.

    Driven by what is INSTALLED at each recorded path, so a second call is a
    no-op rather than a re-splice. Clears ``wrapped_keys`` but NOT
    ``lora_original_modules``; that map's owner decides its lifetime
    (``MiniMaxH3Mixin._minimax_h3_lora_state``).
    """
    from core.adapters import CompositeAdapterLayer, set_module_slot
    from core.training.adapters.minimax_h3_adapter import _resolve_leaf

    restored = 0
    for module_path in list(wrapped_keys):
        resolved = _resolve_leaf(transformer, module_path)
        if resolved is None:
            continue
        parent, attr, current = resolved
        if not isinstance(current, CompositeAdapterLayer):
            continue
        set_module_slot(parent, attr,
                        lora_original_modules.get(module_path, current.original_module))
        restored += 1
    wrapped_keys.clear()
    return restored
