"""LoRA support for the Krea 2 DiT (training target enumeration + inference apply).

Krea 2 is a single-stream flow-matching MMDiT (``transformer_blocks`` ModuleList)
with an internal text-fusion sub-transformer (``text_fusion``) and a Qwen3-VL text
encoder that is ALWAYS FROZEN (no TE LoRA — mirrors the ideogram4 Qwen3-VL policy).

Target scope (controlled by the ``scope`` dict):
  attn:        transformer_blocks.{N}.attn.{to_q,to_k,to_v,to_gate,to_out.0}
  mlp:         transformer_blocks.{N}.ff.{gate,up,down}         (SwiGLU)
  text_fusion: text_fusion.{layerwise_blocks,refiner_blocks}.{N}.attn.{...} / .ff.{...}
               + text_fusion.projector                          (default OFF)
  proj:        img_in, txt_in.linear_1/linear_2, final_layer.linear,
               time_embed.linear_1/linear_2, time_mod_proj      (default OFF)

~264 plain nn.Linear targets total; the default scope (attn+mlp) covers the 28
main blocks. Keys use a reversible ``.``<->``__`` encoding (module names contain
only single underscores, so ``__`` is unambiguously the path separator) — the same
scheme as minit2i_lora. Targets are plain nn.Linear on a bf16 checkpoint; a
weight-only quantized checkpoint keeps its Fp8Linear / Int8Linear base and those
are wrapped too. Forward-time addition, fully reversible.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Mapping, Optional, Tuple

import torch
from torch import nn

from core.adapters.groups import (TensorGroup, declared_groups,
                                  group_adapter_tensors, split_adapter_suffix)


def _flatten(module_path: str) -> str:
    return module_path.replace(".", "__")


def _restore(flat: str) -> str:
    return flat.replace("__", ".")


def _krea2_stem(raw_stem: str) -> Optional[str]:
    """Suffix-stripped key -> module path, or None for a foreign key."""
    if not raw_stem.startswith("lora_unet_"):
        return None
    return _restore(raw_stem[len("lora_unet_"):])


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """``(module_path, canonical tensor name)`` for a recognised key, else None."""
    split = split_adapter_suffix(key)
    if split is None:
        return None
    module_path = _krea2_stem(split[0])
    return None if module_path is None else (module_path, split[1])


def declared_branch_count(raw: Dict[str, torch.Tensor]) -> int:
    """Branches this file declares to Krea 2; see ``declared_groups``."""
    return len(declared_groups(raw, _krea2_stem))


def normalise_lora_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, TensorGroup]:
    """Down/up groups by module path. ``TensorGroup`` answers to
    ``["down"]``/``["up"]``/``.get("alpha")``, which is what the builder reads."""
    grouped = group_adapter_tensors(raw, _krea2_stem).groups
    return {m: g for m, g in grouped.items() if "down" in g and "up" in g}


def detect_lora_format(raw: Mapping[str, torch.Tensor]) -> str:
    """The key-format label for one file.

    ``AdapterSession`` reads the safetensors itself; this is the only part of
    the file load that is Krea 2's.
    """
    return "sd-scripts" if any(k.startswith("lora_unet_") for k in raw) else "unknown"


DEFAULT_SCOPE: Dict[str, bool] = {"attn": True, "mlp": True, "text_fusion": False, "proj": False}
_FULL_SCOPE: Dict[str, bool] = {k: True for k in DEFAULT_SCOPE}


def parse_scope_csv(scope_csv: Optional[str]) -> Dict[str, bool]:
    scope = {k: False for k in DEFAULT_SCOPE}
    if not scope_csv:
        return dict(DEFAULT_SCOPE)
    for tok in scope_csv.split(","):
        tok = tok.strip()
        if tok in scope:
            scope[tok] = True
    if not any(scope.values()):
        return dict(DEFAULT_SCOPE)
    return scope


def _is_target(m) -> bool:
    """True for a module a LoRA can wrap: a plain Linear, EITHER quantized
    Linear (weight-only e4m3 or int8), or an adapter wrapper an earlier LoRA in
    the same request already installed.

    ``Fp8Linear`` and ``Int8Linear`` are ``nn.Module``s, NOT ``nn.Linear``
    subclasses, so both must be named explicitly. Omitting one is silent: the
    iterator simply yields no targets for those layers and the session reports a
    small ``applied`` count without raising -- which on a quantized checkpoint
    looks exactly like "the LoRA had no effect".

    ``CompositeAdapterLayer`` is here for the same reason: drop it and a second
    selected LoRA skips every occupied target and reports zero matches as if its
    keys were wrong."""
    from core.adapters import CompositeAdapterLayer, LoRALinearLayer
    try:
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        return isinstance(m, (nn.Linear, Fp8Linear, Int8Linear, LoRALinearLayer,
                              CompositeAdapterLayer))
    except Exception:
        return isinstance(m, (nn.Linear, LoRALinearLayer, CompositeAdapterLayer))


_ATTN_ATTRS = ("to_q", "to_k", "to_v", "to_gate")
_MLP_ATTRS = ("gate", "up", "down")


def _emit_attn(block, prefix, out):
    attn = getattr(block, "attn", None)
    if attn is None:
        return
    for attr in _ATTN_ATTRS:
        m = getattr(attn, attr, None)
        if _is_target(m):
            out.append((f"{prefix}.attn.{attr}", attn, attr, m))
    to_out = getattr(attn, "to_out", None)
    if isinstance(to_out, nn.ModuleList) and len(to_out) > 0 and _is_target(to_out[0]):
        out.append((f"{prefix}.attn.to_out.0", to_out, 0, to_out[0]))


def _emit_mlp(block, prefix, out):
    ff = getattr(block, "ff", None)
    if ff is None:
        return
    for attr in _MLP_ATTRS:
        m = getattr(ff, attr, None)
        if _is_target(m):
            out.append((f"{prefix}.ff.{attr}", ff, attr, m))


def iter_krea2_lora_targets(
    transformer: nn.Module,
    scope: Optional[Dict[str, bool]] = None,
) -> Generator[Tuple[str, Any, Any, nn.Module], None, None]:
    """Yield (module_path, parent, attr_or_idx, current_module) per LoRA target.

    ONE enumerator for both load and unload, so the two cannot disagree about a
    slot once a target can hold more than one branch.

    module_path is relative to the Krea2Transformer2DModel
    (e.g. "transformer_blocks.0.attn.to_q"). attr_or_idx is a str for normal
    attributes or an int for ModuleList children (to_out[0]) -- address it with
    ``core.adapters.get_module_slot`` / ``set_module_slot``, which take either;
    ``setattr(parent, 0, module)`` raises TypeError.
    """
    scope = scope if scope is not None else DEFAULT_SCOPE
    want_attn = bool(scope.get("attn", False))
    want_mlp = bool(scope.get("mlp", False))
    want_fusion = bool(scope.get("text_fusion", False))
    want_proj = bool(scope.get("proj", False))

    out: list = []

    for i, block in enumerate(getattr(transformer, "transformer_blocks", [])):
        prefix = f"transformer_blocks.{i}"
        if want_attn:
            _emit_attn(block, prefix, out)
        if want_mlp:
            _emit_mlp(block, prefix, out)

    if want_fusion:
        fusion = getattr(transformer, "text_fusion", None)
        if fusion is not None:
            for group in ("layerwise_blocks", "refiner_blocks"):
                for i, block in enumerate(getattr(fusion, group, [])):
                    prefix = f"text_fusion.{group}.{i}"
                    _emit_attn(block, prefix, out)
                    _emit_mlp(block, prefix, out)
            projector = getattr(fusion, "projector", None)
            if _is_target(projector):
                out.append(("text_fusion.projector", fusion, "projector", projector))

    if want_proj:
        # Top-level input/output projections.
        if _is_target(getattr(transformer, "img_in", None)):
            out.append(("img_in", transformer, "img_in", transformer.img_in))
        te = getattr(transformer, "time_embed", None)
        if te is not None:
            for attr in ("linear_1", "linear_2"):
                m = getattr(te, attr, None)
                if _is_target(m):
                    out.append((f"time_embed.{attr}", te, attr, m))
        if _is_target(getattr(transformer, "time_mod_proj", None)):
            out.append(("time_mod_proj", transformer, "time_mod_proj", transformer.time_mod_proj))
        txt_in = getattr(transformer, "txt_in", None)
        if txt_in is not None:
            for attr in ("linear_1", "linear_2"):
                m = getattr(txt_in, attr, None)
                if _is_target(m):
                    out.append((f"txt_in.{attr}", txt_in, attr, m))
        final = getattr(transformer, "final_layer", None)
        if final is not None and _is_target(getattr(final, "linear", None)):
            out.append(("final_layer.linear", final, "linear", final.linear))

    for entry in out:
        yield entry


def flatten_to_key(module_path: str) -> str:
    """Module path -> sd-scripts-style LoRA key stem ('lora_unet_<flat>')."""
    return f"lora_unet_{_flatten(module_path)}"


# ---------------------------------------------------------------------------
# Apply (inference). The LIFETIME -- resolve, parse, refuse, install, restore --
# belongs to ``core.adapters.AdapterSession``; what is Krea 2's is the target
# scope, the key codec and one branch.
# ---------------------------------------------------------------------------

def iter_krea2_lora_slots(transformer: nn.Module):
    """``(parent, slot, module_path)`` over the FULL scope, for ``AdapterSession``.

    Full scope on both the load and the unload path. Application is
    lookup-driven -- a target the file names no key for gets no branch -- so
    enumerating every group applies exactly what the checkpoint's own scope
    would, and restore reaches a composite installed from any scope.
    """
    for module_path, parent, attr, _current in iter_krea2_lora_targets(
            transformer, _FULL_SCOPE):
        yield parent, attr, module_path


def build_lora_branch(base: nn.Module, weights: Dict[str, torch.Tensor],
                      module_path: str) -> nn.Module:
    """One branch over ``base``, at the file's own alpha/rank scale.

    The request strength is NOT folded in here: ``add_branch(strength=)`` refolds
    it into this branch's own scale, and multiplying it onto the delta instead is
    different arithmetic that loses bit-identity with the single-LoRA numerics.
    """
    from core.adapters import LoRALinearLayer, lora_branch_dtype

    down, up = weights["down"], weights["up"]
    alpha_tensor = weights.get("alpha")
    rank = int(down.shape[0])
    alpha_value = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
    branch = LoRALinearLayer(base, rank=rank, alpha=alpha_value, lora_name=module_path)
    device = base.weight.device
    # Never the base weight's own dtype: over an int8 base that would quantize
    # the branch to 8 levels, over an e4m3 one it would round most of it away.
    compute_dtype = lora_branch_dtype(base)
    with torch.no_grad():
        branch.lora_down.weight.data = down.to(device=device, dtype=compute_dtype)
        branch.lora_up.weight.data = up.to(device=device, dtype=compute_dtype)
    branch.lora_down = branch.lora_down.to(dtype=compute_dtype)
    branch.lora_up = branch.lora_up.to(dtype=compute_dtype)
    return branch
