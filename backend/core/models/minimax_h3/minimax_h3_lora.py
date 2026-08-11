"""Inference-time LoRA support for the MiniMax-H3 DiT.

Loads a LoRA safetensors file and wraps the matching vendored Linear modules
with ``MiniMaxH3LoRALinearLayer`` (forward-time addition, never a weight
merge -- fully reversible by restoring the original module). This mirrors
``core.models.krea2.krea2_lora`` / ``core.models.anima.anima_lora`` in shape;
see those modules for the general pattern this one specialises.

Only ONE key convention is supported: the ComfyUI/"interchange" layout real
MiniMax-H3 LoRAs ship in --

    diffusion_model.blocks.<N>.attn.qkv_proj.lora_A.weight
    diffusion_model.blocks.<N>.attn.qkv_proj.lora_B.weight
    diffusion_model.blocks.<N>.attn.qkv_proj.alpha            (optional)
    diffusion_model.token_refiner.blocks.<N>.<...>
    diffusion_model.final_layer.<...>

This is NOT the format ``core.training.adapters.minimax_h3_adapter`` writes
for a LoRA trained inside this repo (sd-scripts native, already targeting
vendored module names one-to-one, no fusion). Loading a self-trained
checkpoint through this module is out of scope here and is a follow-up; a
``lora_unet_*`` key is simply unmatched by ``_parse_key`` and dropped like any
other unrecognised key, the same silent-drop convention
``anima_lora.normalise_lora_state_dict`` documents.

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
      original, undivided delta. ``alpha`` absent means ``alpha = rank``
      (scale 1.0) -- some real checkpoints drop alpha and bake a flat
      multiplier directly into ``lora_B`` instead of relying on alpha/rank,
      and re-applying a nonexistent alpha/rank ratio on top would silently
      double-attenuate them.

Save format reference: ``core/training/adapters/minimax_h3_adapter.py``
(sd-scripts native, the OUTPUT side -- not read by this module, see above).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from safetensors import safe_open


_PREFIX = "diffusion_model."
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

def normalise_lora_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    """Group + convert a raw comfy LoRA state dict into vendored targets.

    Returns ``{vendored_module_path: {"down": Tensor, "up": Tensor,
    "scale_ratio": float}}``. ``scale_ratio`` is ``alpha / rank`` (rank being
    the FUSED rank for a qkv split's three pieces, see point (c) in the
    module docstring); the caller multiplies by the user-supplied LoRA
    strength.
    """
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
            alpha_value = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank_total)
            scale_ratio = alpha_value / rank_total
            base = mapped.split(".attn.qkv_proj")[0] + ".attn."
            for name, (d, u) in parts.items():
                target = base + name
                if target in targets:
                    raise ValueError(f"duplicate LoRA target {target!r} (from stem {stem!r})")
                targets[target] = {"down": d, "up": u, "scale_ratio": scale_ratio}
            continue

        rank = int(down.shape[0])
        alpha_value = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
        scale_ratio = alpha_value / rank

        if stem.endswith(_FC1_SUFFIX):
            up = _swap_fc1_halves(up)

        if mapped in targets:
            raise ValueError(f"duplicate LoRA target {mapped!r} (from stem {stem!r})")
        targets[mapped] = {"down": down, "up": up, "scale_ratio": scale_ratio}

    return targets


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
    """
    current = (current_variant or "").lower()
    base_model = str(metadata.get("base_model", "") or "")
    name = Path(lora_path).name

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
) -> Tuple[int, list]:
    """Wrap matching vendored modules with ``MiniMaxH3LoRALinearLayer``.

    Stacking-safe (unwraps an existing wrapper to recover the true original
    so a second LoRA composes rather than replaces the first) and
    unload-safe (``lora_original_modules.setdefault`` records only the FIRST
    original seen for a module path, so ``restore_originals`` always reaches
    the un-LoRA'd module regardless of how many LoRAs were stacked on it).

    Returns ``(applied_count, missing_target_paths)``; the caller decides how
    loudly to report unmatched targets (a LoRA trained against a different
    scope, e.g. only ``attention``, legitimately leaves ``ff``/``adaln``
    targets unmatched by the MODEL side -- but a target this function cannot
    even RESOLVE against the live module tree is a real problem worth
    surfacing).
    """
    from core.training.adapters.base_adapter import is_lora_wrappable_linear, lora_branch_dtype
    from core.training.adapters.minimax_h3_adapter import MiniMaxH3LoRALinearLayer, _resolve_leaf
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    applied = 0
    missing: list = []
    for module_path, weights in targets.items():
        resolved = _resolve_leaf(transformer, module_path)
        if resolved is None:
            missing.append(module_path)
            continue
        parent, attr, current = resolved

        if isinstance(current, LoRALinearLayer):
            true_original = current.original_module
        elif is_lora_wrappable_linear(current):
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
    """Revert every wrapped module to its pre-LoRA original."""
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
