"""Inference-time LoRA support for the MiniMax-H3 DiT.

Loads a LoRA safetensors file and wraps the matching vendored Linear modules
with ``MiniMaxH3LoRALinearLayer`` (forward-time addition, never a weight
merge -- fully reversible by restoring the original module). This mirrors
``core.models.krea2.krea2_lora`` / ``core.models.anima.anima_lora`` in shape;
see those modules for the general pattern this one specialises.

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
   A stem that table cannot map, and a stem missing its down or up half, both
   raise instead of being dropped -- a self-trained checkpoint that matched
   nothing used to be indistinguishable from a generation with no LoRA at all,
   and this repo writes both halves for every target it saves, so either is a
   real defect in the file.

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
      See ``_split_qkv``.

  (b) **fc1 SwiGLU half swap.** Comfy stores ``mlp.fc1`` as ``[gate; up]``;
      the vendored ``ff.net.0.proj`` (SwiGLU) expects ``[up; gate]``. This is
      a ROW PERMUTATION of ``lora_B`` only -- ``lora_A`` (the down
      projection, over the INPUT dimension) is untouched. Getting this
      backwards is silent: shapes match, the load is clean, and the gate
      delta lands in the up path. See ``_swap_fc1_halves``.

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

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from safetensors import safe_open


_PREFIX = "diffusion_model."
_NATIVE_PREFIX = "lora_unet_"
_QKV_SUFFIX = ".attn.qkv_proj"
_FC1_SUFFIX = ".mlp.fc1"


# ---------------------------------------------------------------------------
# Raw key parsing
# ---------------------------------------------------------------------------

def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """``(comfy_module_stem, tag)`` for a recognised key, ``tag`` in
    ``{"down", "up", "alpha"}``; ``None`` for anything else (dropped)."""
    if not key.startswith(_PREFIX):
        return None
    rest = key[len(_PREFIX):]
    for suffix, tag in ((".lora_A.weight", "down"), (".lora_B.weight", "up"), (".alpha", "alpha")):
        if rest.endswith(suffix):
            return rest[: -len(suffix)], tag
    return None


def _parse_native_key(key: str) -> Optional[Tuple[str, str]]:
    """``(flattened_stem, tag)`` for an sd-scripts native key, else ``None``."""
    if not key.startswith(_NATIVE_PREFIX):
        return None
    rest = key[len(_NATIVE_PREFIX):]
    for suffix, tag in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
        if rest.endswith(suffix):
            return rest[: -len(suffix)], tag
    return None


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


def _group_raw(raw: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        stem, tag = parsed
        grouped.setdefault(stem, {})[tag] = tensor
    return {s: v for s, v in grouped.items() if "down" in v and "up" in v}


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


# ---------------------------------------------------------------------------
# (b) fc1 SwiGLU half swap
# ---------------------------------------------------------------------------

def _swap_fc1_halves(up: torch.Tensor) -> torch.Tensor:
    """Comfy ``[gate; up]`` -> vendored SwiGLU ``[up; gate]``, row permutation of
    ``lora_B`` only. Mirrors ``loader._map_dit_state_dict``'s base-weight swap."""
    gate, up_half = up.chunk(2, dim=0)
    return torch.cat([up_half, gate], dim=0).contiguous()


# ---------------------------------------------------------------------------
# Full conversion: comfy raw state dict -> {vendored_target_path: {down, up, scale_ratio}}
# ---------------------------------------------------------------------------

def _resolve_alpha(alpha_tensor, metadata_alpha: Optional[float], rank: int) -> float:
    """Per-key ``.alpha`` tensor -> ``metadata_alpha`` -> rank. Comfy callers
    pass ``metadata_alpha=None`` (module docstring, (c))."""
    if alpha_tensor is not None:
        return float(alpha_tensor.item())
    if metadata_alpha is not None:
        return metadata_alpha
    return float(rank)


def _normalise_comfy(raw: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    """ComfyUI/interchange layout -> vendored targets. Takes no metadata: this
    branch's alpha is per-key or rank, never the file's (module docstring, (c))."""
    from core.models.minimax_h3.loader import _rename_dit_key

    grouped_raw = _group_raw(raw)
    targets: Dict[str, Dict[str, Any]] = {}

    for stem, weights in grouped_raw.items():
        down = weights["down"]
        up = weights["up"]
        alpha_tensor = weights.get("alpha")

        mapped = _rename_dit_key(stem + ".weight")
        if not mapped.endswith(".weight"):
            raise ValueError(f"{stem}: unexpected renamed key {mapped!r} (expected a .weight suffix)")
        mapped = mapped[: -len(".weight")]

        if stem.endswith(_QKV_SUFFIX):
            parts, rank_total = _split_qkv(stem, down, up)
            scale_ratio = _resolve_alpha(alpha_tensor, None, rank_total) / rank_total
            base = mapped.split(".attn.qkv_proj")[0] + ".attn."
            for name, (d, u) in parts.items():
                target = base + name
                if target in targets:
                    raise ValueError(f"duplicate LoRA target {target!r} (from stem {stem!r})")
                targets[target] = {"down": d, "up": u, "scale_ratio": scale_ratio}
            continue

        rank = int(down.shape[0])
        scale_ratio = _resolve_alpha(alpha_tensor, None, rank) / rank

        if stem.endswith(_FC1_SUFFIX):
            up = _swap_fc1_halves(up)

        if mapped in targets:
            raise ValueError(f"duplicate LoRA target {mapped!r} (from stem {stem!r})")
        targets[mapped] = {"down": down, "up": up, "scale_ratio": scale_ratio}

    return targets


def _normalise_native(
    raw: Dict[str, torch.Tensor], metadata_alpha: Optional[float],
) -> Dict[str, Dict[str, Any]]:
    """sd-scripts native (this repo's own trainer output) -> vendored targets.

    One-to-one with the vendored module names: no qkv split, no fc1 half swap.
    Only the underscored stem is un-flattened, against the training adapter's
    own leaf table.
    """
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        parsed = _parse_native_key(key)
        if parsed is None:
            continue
        stem, tag = parsed
        grouped.setdefault(stem, {})[tag] = tensor

    table = _native_leaf_table()
    targets: Dict[str, Dict[str, Any]] = {}
    unmapped: list = []
    incomplete: list = []
    for stem, weights in grouped.items():
        if "down" not in weights or "up" not in weights:
            incomplete.append(stem)
            continue
        module_path = _native_stem_to_module_path(stem, table)
        if module_path is None:
            unmapped.append(stem)
            continue
        down = weights["down"]
        up = weights["up"]
        rank = int(down.shape[0])
        if int(up.shape[1]) != rank:
            raise ValueError(
                f"{stem}: lora_down rank {rank} does not match lora_up's {up.shape[1]} columns."
            )
        if module_path in targets:
            raise ValueError(f"duplicate LoRA target {module_path!r} (from stem {stem!r})")
        targets[module_path] = {
            "down": down,
            "up": up,
            "scale_ratio": _resolve_alpha(weights.get("alpha"), metadata_alpha, rank) / rank,
        }

    if incomplete:
        raise ValueError(
            f"{len(incomplete)} sd-scripts LoRA stem(s) carry only one of lora_down/lora_up "
            f"(first few: {sorted(incomplete)[:5]}); this repo's trainer writes both for every "
            f"target it saves, so the file is truncated or corrupt."
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
) -> Dict[str, Dict[str, Any]]:
    """Group + convert a raw LoRA state dict into vendored targets.

    Returns ``{vendored_module_path: {"down": Tensor, "up": Tensor,
    "scale_ratio": float}}``. ``scale_ratio`` is ``alpha / rank`` (rank being
    the FUSED rank for a qkv split's three pieces, see point (c) in the module
    docstring); the caller multiplies by the user-supplied LoRA strength.

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


def detect_rank_variation(targets: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """``{leaf_name: True}`` for every ``transformer_blocks.*`` leaf whose rank
    (down.shape[0]) differs between blocks.

    Only ``transformer_blocks.*`` entries matter here:
    ``TransformerBlockOffloader._build_weight_swap_jobs``
    (``core/memory_management/block_offloading.py``) pairs an incoming and an
    outgoing block's Linear weights by name + shape + dtype, and only the
    block stack is ever swapped -- ``token_refiner`` and ``final_layer``
    leaves are moved whole, never paired block-to-block.
    """
    ranks_by_leaf: Dict[str, set] = {}
    for module_path, weights in targets.items():
        m = _BLOCK_LEAF_RE.match(module_path)
        if not m:
            continue
        leaf = m.group(2)
        ranks_by_leaf.setdefault(leaf, set()).add(int(weights["down"].shape[0]))
    return {leaf: len(ranks) > 1 for leaf, ranks in ranks_by_leaf.items()}


# ---------------------------------------------------------------------------
# Apply / restore (inference)
# ---------------------------------------------------------------------------

def apply_lora_group(
    transformer: nn.Module,
    targets: Dict[str, Dict[str, Any]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: set,
    shadowed: Optional[list] = None,
) -> Tuple[int, list]:
    """Wrap matching vendored modules with ``MiniMaxH3LoRALinearLayer``.

    Returns ``(applied_count, missing_target_paths)``. A target an EARLIER
    LoRA in the same request already wrapped is SKIPPED and appended to
    ``shadowed`` when the caller supplies a list: neither ``LoRALinearLayer``
    nor its ``MiniMaxH3LoRALinearLayer`` subclass exposes
    ``in_features``/``out_features``, so a wrapper cannot wrap a wrapper, and
    the previous shape here (unwrap to the true original, wrap that) silently
    discarded the earlier LoRA on every shared target. Additive composition is
    ``CompositeAdapterLinear`` work (LYCORIS_ADAPTER_DESIGN Phase 1); until
    then the caller refuses a fully shadowed stack rather than faking it.

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
        LoRALinearLayer,
        MiniMaxH3LoRALinearLayer,
        is_lora_wrappable_linear,
        lora_branch_dtype,
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

        if isinstance(current, LoRALinearLayer):
            if shadowed is not None:
                shadowed.append(module_path)
            continue
        if is_lora_wrappable_linear(current):
            true_original = current
        else:
            missing.append(module_path)
            continue

        lora_original_modules.setdefault(module_path, true_original)

        down = weights["down"]
        up = weights["up"]
        rank = int(down.shape[0])
        scale = float(weights["scale_ratio"]) * strength

        # `rank`/`alpha` passed to the constructor only size lora_down/lora_up;
        # the ratio they'd imply (alpha/rank == 1.0 here) is discarded below in
        # favour of the FUSED stem's own scale_ratio (module docstring, (c)).
        wrapper = MiniMaxH3LoRALinearLayer(true_original, rank=rank, alpha=rank, lora_name=module_path)
        device = true_original.weight.device
        # MIXED PRECISION: the block stack's attn/ff Linears are Fp8Linear
        # (branch defaults to bf16); adaln_proj.linear / norm_out.linear /
        # proj_out / audio_proj_out are real float32 Linears (branch matches
        # that fp32). Never hardcode bf16 here -- see the module + adapter
        # docstrings on why the H3 tree is mixed precision.
        compute_dtype = lora_branch_dtype(true_original)
        with torch.no_grad():
            wrapper.lora_down.weight.data = down.to(device=device, dtype=compute_dtype)
            wrapper.lora_up.weight.data = up.to(device=device, dtype=compute_dtype)
        wrapper.lora_down = wrapper.lora_down.to(dtype=compute_dtype)
        wrapper.lora_up = wrapper.lora_up.to(dtype=compute_dtype)
        wrapper.scale = scale

        if isinstance(attr, int):
            parent[attr] = wrapper
        else:
            setattr(parent, attr, wrapper)
        wrapped_keys.add(module_path)
        applied += 1

    return applied, missing


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: set,
) -> int:
    """Revert every wrapped module to its pre-LoRA original.

    Clears ``wrapped_keys`` but NOT ``lora_original_modules``; that map's owner
    decides its lifetime (``MiniMaxH3Mixin._minimax_h3_lora_state``).
    """
    from core.training.adapters.minimax_h3_adapter import _resolve_leaf

    restored = 0
    for module_path in list(wrapped_keys):
        if module_path not in lora_original_modules:
            continue
        resolved = _resolve_leaf(transformer, module_path)
        if resolved is None:
            continue
        parent, attr, _current = resolved
        original = lora_original_modules[module_path]
        if isinstance(attr, int):
            parent[attr] = original
        else:
            setattr(parent, attr, original)
        restored += 1
    wrapped_keys.clear()
    return restored
