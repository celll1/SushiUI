"""LoRA support for the LTX-2.3 joint audio+video DiT (generation side).

Key codec is the exact inverse of the trainer's save format
(``core.training.adapters.ltx2_adapter.Ltx2LoRAAdapter.save_checkpoint``):
sd-scripts native ``lora_unet_<module path with dots flattened to
underscores>.{lora_down.weight, lora_up.weight, alpha}``.

The flattening is NOT inverted here (``to_out_0`` -> ``to_out.0`` and
``transformer_blocks_0`` -> ``transformer_blocks.0`` are not separable without a
token table). Instead every candidate target is enumerated from the LIVE model
with the TRAINER's own iterator and flattened forward into the same key space,
so the two sides cannot drift: a stem matches iff the trainer would have written
it.

Components an LTX-2.3 LoRA can contain: the DiT only. ``Ltx2LoRAAdapter
.apply_lora_to_text_encoders`` returns 0 — Gemma-3 and the text connectors are
frozen — so there are no text-encoder tensors to route anywhere.
"""

from typing import Any, Dict, Optional, Set, Tuple

import torch
from torch import nn
from safetensors import safe_open


# Every scope the trainer can be configured to save, so a file trained with a
# non-default scope (ff / audio / av_cross) round-trips too.
FULL_SCOPE: Dict[str, bool] = {
    "attention": True,
    "ff": True,
    "audio": True,
    "av_cross": True,
}

_SDSCRIPTS_PREFIX = "lora_unet_"
# Prefixes third-party exports put in front of a DOTTED DiT module path.
_DOTTED_PREFIXES = ("diffusion_model.", "transformer.")
_SUFFIXES = (
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".alpha", "alpha"),
)


def flatten_module_path(module_path: str) -> str:
    """Module path -> the stem the trainer writes after ``lora_unet_``."""
    return module_path.replace(".", "_")


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """``(flattened stem, "down"|"up"|"alpha")`` for a recognised key, else None."""
    for suffix, tag in _SUFFIXES:
        if not key.endswith(suffix):
            continue
        stem = key[: -len(suffix)]
        if stem.startswith(_SDSCRIPTS_PREFIX):
            return stem[len(_SDSCRIPTS_PREFIX):], tag
        for prefix in _DOTTED_PREFIXES:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        if "." in stem:
            return flatten_module_path(stem), tag
        return None
    return None


def normalise_lora_state_dict(
    raw: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Group raw tensors by flattened module stem, keeping complete pairs only."""
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        stem, tag = parsed
        grouped.setdefault(stem, {})[tag] = tensor
    return {stem: v for stem, v in grouped.items() if "down" in v and "up" in v}


def detect_lora_format(raw: Dict[str, torch.Tensor]) -> str:
    n_sd = sum(1 for k in raw if k.startswith(_SDSCRIPTS_PREFIX))
    n_dotted = sum(1 for k in raw if k.startswith(_DOTTED_PREFIXES))
    if n_sd and n_dotted:
        return "mixed"
    if n_sd:
        return "sd-scripts"
    if n_dotted:
        return "dotted"
    return "unknown"


def load_lora_safetensors(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], str]:
    """Return ``(tensors, safetensors metadata, format label)``."""
    raw: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    return raw, metadata, detect_lora_format(raw)


def _iter_targets(transformer: nn.Module):
    """The trainer's own target iterator at full scope — single source of truth
    for both which modules are targets and what they are called."""
    from core.training.adapters.ltx2_adapter import iter_ltx2_lora_targets

    return iter_ltx2_lora_targets(transformer, FULL_SCOPE)


def _set_module(parent: Any, attr: Any, module: nn.Module) -> None:
    if isinstance(attr, int):
        parent[attr] = module
    else:
        setattr(parent, attr, module)


def metadata_alpha(metadata: Optional[Dict[str, str]]) -> Optional[float]:
    """Alpha from safetensors metadata, for files carrying no per-key ``.alpha``.

    ``Ltx2LoRAAdapter.save_checkpoint`` writes BOTH a per-key ``.alpha`` tensor
    and a ``lora_alpha`` metadata entry, so this is the fallback for third-party
    files; the per-key tensor still wins. ``ss_network_alpha`` is the sd-scripts
    spelling of the same field. Without it an ``alpha != rank`` file would apply
    at scale 1.0 instead of the scale it was trained at.
    """
    for key in ("lora_alpha", "ss_network_alpha"):
        raw = (metadata or {}).get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    file_alpha: Optional[float] = None,
) -> Tuple[int, list, int]:
    """Wrap every matching DiT Linear with ``LoRALinearLayer``.

    Returns ``(applied, unmatched_stems, occupied)``. ``occupied`` counts
    targets this file matched that an EARLIER LoRA in the same request already
    wraps; those are LEFT ALONE and the caller refuses or warns. Re-wrapping
    would silently drop the earlier branch (last-wins), and nesting is not
    available because ``LoRALinearLayer`` reads ``in_features``/``out_features``
    off what it is handed and so cannot wrap a wrapper. Summing branches is the
    shared composite wrapper's job (LYCORIS_ADAPTER_DESIGN Phase 1).

    Alpha precedence per adapter: per-key ``.alpha`` tensor, then the file's
    metadata alpha, then the rank (i.e. scale 1.0).
    """
    from core.training.adapters.sd15_adapter import LoRALinearLayer
    from core.training.adapters.base_adapter import lora_branch_dtype

    applied = 0
    occupied = 0
    matched: Set[str] = set()
    for module_path, parent, attr, current in _iter_targets(transformer):
        stem = flatten_module_path(module_path)
        weights = grouped.get(stem)
        if weights is None:
            continue
        matched.add(stem)
        if isinstance(current, LoRALinearLayer):
            occupied += 1
            continue

        down, up = weights["down"], weights["up"]
        alpha_tensor = weights.get("alpha")
        lora_original_modules.setdefault(module_path, current)

        rank = int(down.shape[0])
        if alpha_tensor is not None:
            alpha_value = float(alpha_tensor.item())
        elif file_alpha is not None:
            alpha_value = float(file_alpha)
        else:
            alpha_value = float(rank)

        # LoRALinearLayer escapes the offloader's movers only because its class
        # name ends in "Layer": ending in "Linear" would enrol the base weight a
        # SECOND time through its `.weight` property (sd15_adapter.py:102-107),
        # a double-swap that silently restores the outgoing block's weights.
        wrapper = LoRALinearLayer(current, rank=rank, alpha=alpha_value,
                                  lora_name=module_path)
        # Adapter follows its BASE MODULE's current device, which under block
        # swap differs per block (resident blocks on GPU, swapped-out blocks on
        # CPU). That is what lets the offloader carry the adapter with the block
        # it belongs to; see _load_lora_ltx2's block-swap contract.
        device = current.weight.device
        dtype = lora_branch_dtype(current)
        with torch.no_grad():
            wrapper.lora_down.weight.data = down.to(device=device, dtype=dtype)
            wrapper.lora_up.weight.data = up.to(device=device, dtype=dtype)
        wrapper.lora_down = wrapper.lora_down.to(dtype=dtype)
        wrapper.lora_up = wrapper.lora_up.to(dtype=dtype)
        wrapper.scale = (alpha_value / rank) * strength

        _set_module(parent, attr, wrapper)
        wrapped_keys.add(module_path)
        applied += 1

    return applied, sorted(set(grouped) - matched), occupied


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
) -> int:
    """Revert every wrapped DiT Linear to its pre-LoRA original module.

    Device-agnostic by construction: the saved object is the original module,
    and putting it back is a parent-attribute assignment, so a block whose
    weights are currently on the host (swapped out) restores exactly like a
    resident one.
    """
    restored = 0
    for module_path, parent, attr, _current in _iter_targets(transformer):
        original = lora_original_modules.get(module_path)
        if original is not None:
            _set_module(parent, attr, original)
            restored += 1
    lora_original_modules.clear()
    wrapped_keys.clear()
    return restored


def swappable_block_weight_footprints(blocks, blocks_to_swap: int) -> list:
    """Total Linear-weight ``numel`` per swappable block, in block order.

    Mirrors ``TransformerBlockOffloader._linear_weight_modules``' selection
    (class name ends in ``Linear`` and carries a weight), which is what the
    coalesced H2D-only path measures when it asserts that every swappable block
    is identically structured. A LoRA that covers only some blocks breaks that
    equality, and this is how the caller detects it before the first forward.
    """
    num_blocks = len(blocks)
    first = max(0, num_blocks - max(0, blocks_to_swap))
    out = []
    for idx in range(first, num_blocks):
        total = 0
        for _name, module in blocks[idx].named_modules():
            if not module.__class__.__name__.endswith("Linear"):
                continue
            weight = getattr(module, "weight", None)
            if weight is not None:
                total += weight.numel()
        out.append(total)
    return out
