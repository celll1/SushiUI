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

from core.adapters.groups import (TensorGroup, declared_groups,
                                  group_adapter_tensors, split_adapter_suffix)


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


def flatten_module_path(module_path: str) -> str:
    """Module path -> the stem the trainer writes after ``lora_unet_``."""
    return module_path.replace(".", "_")


def _ltx2_stem(raw_stem: str) -> Optional[str]:
    """Suffix-stripped key -> the flattened stem, or None for a foreign key."""
    if raw_stem.startswith(_SDSCRIPTS_PREFIX):
        return raw_stem[len(_SDSCRIPTS_PREFIX):]
    for prefix in _DOTTED_PREFIXES:
        if raw_stem.startswith(prefix):
            raw_stem = raw_stem[len(prefix):]
            break
    # A dotted stem is a module path; a bare one names no LTX-2.3 target.
    return flatten_module_path(raw_stem) if "." in raw_stem else None


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """``(flattened stem, canonical tensor name)`` for a recognised key, else None."""
    split = split_adapter_suffix(key)
    if split is None:
        return None
    stem = _ltx2_stem(split[0])
    return None if stem is None else (stem, split[1])


def declared_branch_count(raw: Dict[str, torch.Tensor]) -> int:
    """Branches this file declares to LTX-2.3; see ``declared_groups``."""
    return len(declared_groups(raw, _ltx2_stem))


def normalise_lora_state_dict(
    raw: Dict[str, torch.Tensor],
) -> Dict[str, TensorGroup]:
    """Group raw tensors by flattened module stem, keeping complete pairs only.

    ``TensorGroup`` answers to ``["down"]``/``["up"]``/``.get("alpha")``, which
    is what the branch builder reads.
    """
    grouped = group_adapter_tensors(raw, _ltx2_stem).groups
    return {stem: g for stem, g in grouped.items() if "down" in g and "up" in g}


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


def iter_lora_slots(transformer: nn.Module):
    """``(module_path, parent, slot)`` for every slot a LoRA may cover.

    ONE enumerator for both ``apply_lora_group`` and ``restore_originals``, so
    they cannot disagree once a target holds more than one branch.

    Two halves, because the trainer's iterator alone is not the whole set once a
    composite exists. It supplies the codec (a stem matches iff the trainer
    would have written it), but it selects by a predicate that a
    ``CompositeAdapterLayer`` fails, so an occupied target VANISHES from it --
    and its feed-forward branch, which walks ``ff.named_modules()`` and only
    skips past a ``LoRALinearLayer`` subtree, then descends INTO the composite
    and offers the adapter's own ``lora_down``/``lora_up`` as targets. Composite
    roots are found structurally instead, and everything underneath one is
    dropped by path prefix against those roots -- real paths, not a path shape,
    because ``branches.<i>`` collides with the index slots this architecture
    genuinely has (``to_out.0``, ``ff.net.2``).

    Callers materialise this before mutating: it reads the live tree.
    """
    from core.adapters import CompositeAdapterLayer

    composites: Dict[str, Tuple[nn.Module, Any]] = {}
    for parent_path, parent in transformer.named_modules():
        for slot, child in parent.named_children():
            if isinstance(child, CompositeAdapterLayer):
                path = f"{parent_path}.{slot}" if parent_path else slot
                composites[path] = (parent, slot)

    inside = tuple(f"{path}." for path in composites)
    for module_path, parent, attr, _current in _iter_targets(transformer):
        if module_path in composites or module_path.startswith(inside):
            continue
        yield module_path, parent, attr
    for path, (parent, slot) in composites.items():
        yield path, parent, slot


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
    branch_name: str = "lora",
) -> Tuple[int, list]:
    """Add one named branch per matching DiT Linear.

    Each target is covered ONCE by a ``CompositeAdapterLayer`` and each selected
    LoRA adds a named branch to it, so two LoRAs over the same module sum
    instead of the second being refused. ``branch_name`` must be unique within
    the request; ``add_branch`` refuses a duplicate.

    Returns ``(applied, unmatched_stems)``.

    Alpha precedence per adapter: per-key ``.alpha`` tensor, then the file's
    metadata alpha, then the rank (i.e. scale 1.0).
    """
    from core.adapters import (
        CompositeAdapterLayer, LoRALinearLayer, get_module_slot, lora_branch_dtype,
    )

    applied = 0
    matched: Set[str] = set()
    for module_path, parent, attr in list(iter_lora_slots(transformer)):
        stem = flatten_module_path(module_path)
        weights = grouped.get(stem)
        if weights is None:
            continue
        matched.add(stem)

        installed = get_module_slot(parent, attr)
        current = (installed.original_module
                   if isinstance(installed, CompositeAdapterLayer) else installed)
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

        # Neither the composite nor LoRALinearLayer ends in "Linear", which is
        # what keeps them out of the offloader's movers: a Linear-named wrapper
        # would enrol the base weight a SECOND time through its `.weight`
        # property (core/adapters/layers.py), a double-swap that silently
        # restores the outgoing block's weights.
        branch = LoRALinearLayer(current, rank=rank, alpha=alpha_value,
                                 lora_name=module_path)
        # Adapter follows its BASE MODULE's current device, which under block
        # swap differs per block (resident blocks on GPU, swapped-out blocks on
        # CPU). That is what lets the offloader carry the adapter with the block
        # it belongs to; see _load_lora_ltx2's block-swap contract.
        device = current.weight.device
        dtype = lora_branch_dtype(current)
        with torch.no_grad():
            branch.lora_down.weight.data = down.to(device=device, dtype=dtype)
            branch.lora_up.weight.data = up.to(device=device, dtype=dtype)
        branch.lora_down = branch.lora_down.to(dtype=dtype)
        branch.lora_up = branch.lora_up.to(dtype=dtype)

        # attach() is idempotent, and add_branch refolds the strength into the
        # branch's own scale -- multiplying it onto the delta afterwards is
        # different arithmetic and loses bit-identity with the single-LoRA
        # numerics this replaces.
        composite = CompositeAdapterLayer.attach(parent, attr)
        composite.add_branch(branch_name, branch, strength=strength)
        wrapped_keys.add(module_path)
        applied += 1

    return applied, sorted(set(grouped) - matched)


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
) -> int:
    """Revert every composite-covered DiT Linear to its pre-LoRA original module.

    Driven by what is INSTALLED (``iter_lora_slots``), not by map membership, so
    it removes exactly the wrappers this request put there, branches and all.

    Device-agnostic by construction: the saved object is the original module,
    and putting it back is a parent-attribute assignment, so a block whose
    weights are currently on the host (swapped out) restores exactly like a
    resident one.
    """
    from core.adapters import CompositeAdapterLayer, get_module_slot

    restored = 0
    for module_path, parent, attr in list(iter_lora_slots(transformer)):
        current = get_module_slot(parent, attr)
        if not isinstance(current, CompositeAdapterLayer):
            continue
        _set_module(parent, attr,
                    lora_original_modules.get(module_path, current.original_module))
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
