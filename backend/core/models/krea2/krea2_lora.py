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

from typing import Any, Dict, Generator, Optional, Set, Tuple

import torch
from torch import nn
from safetensors import safe_open


def _flatten(module_path: str) -> str:
    return module_path.replace(".", "__")


def _restore(flat: str) -> str:
    return flat.replace("__", ".")


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    if key.startswith("lora_unet_"):
        rest = key[len("lora_unet_"):]
        for suffix, tag in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
            if rest.endswith(suffix):
                return _restore(rest[: -len(suffix)]), tag
    return None


def normalise_lora_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        module_path, suffix = parsed
        grouped.setdefault(module_path, {})[suffix] = tensor
    return {m: v for m, v in grouped.items() if "down" in v and "up" in v}


def load_lora_safetensors(path: str) -> Tuple[Dict[str, torch.Tensor], str]:
    raw: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    fmt = "sd-scripts" if any(k.startswith("lora_unet_") for k in raw) else "unknown"
    return raw, fmt


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
    Linear (weight-only e4m3 or int8), or an already-wrapped LoRALinearLayer.

    ``Fp8Linear`` and ``Int8Linear`` are ``nn.Module``s, NOT ``nn.Linear``
    subclasses, so both must be named explicitly. Omitting one is silent: the
    iterator simply yields no targets for those layers and ``apply_lora_group``
    reports a small ``applied`` count without raising -- which on a quantized
    checkpoint looks exactly like "the LoRA had no effect"."""
    from core.adapters import LoRALinearLayer
    try:
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        return isinstance(m, (nn.Linear, Fp8Linear, Int8Linear, LoRALinearLayer))
    except Exception:
        return isinstance(m, (nn.Linear, LoRALinearLayer))


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

    module_path is relative to the Krea2Transformer2DModel
    (e.g. "transformer_blocks.0.attn.to_q"). attr_or_idx is a str for normal
    attributes or an int for ModuleList children (to_out[0]) — use _set_module().
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


def _set_module(parent: Any, attr: Any, module: nn.Module) -> None:
    if isinstance(attr, int):
        parent[attr] = module
    else:
        setattr(parent, attr, module)


def flatten_to_key(module_path: str) -> str:
    """Module path -> sd-scripts-style LoRA key stem ('lora_unet_<flat>')."""
    return f"lora_unet_{_flatten(module_path)}"


# ---------------------------------------------------------------------------
# Apply / restore (inference)
# ---------------------------------------------------------------------------

def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    scope: Optional[Dict[str, bool]] = None,
) -> int:
    """Wrap matching modules with LoRALinearLayer (stackable, reversible)."""
    from core.adapters import LoRALinearLayer

    effective_scope = scope if scope is not None else _FULL_SCOPE
    applied = 0
    for module_path, parent, attr, linear in iter_krea2_lora_targets(transformer, effective_scope):
        weights = grouped.get(module_path)
        if weights is None:
            continue
        down, up = weights["down"], weights["up"]
        alpha_tensor = weights.get("alpha")
        true_original = linear.original_module if isinstance(linear, LoRALinearLayer) else linear
        lora_original_modules.setdefault(module_path, true_original)
        rank = int(down.shape[0])
        alpha_value = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
        wrapper = LoRALinearLayer(true_original, rank=rank, alpha=alpha_value, lora_name=module_path)
        device = true_original.weight.device
        # Compute dtype for the LoRA branch: the base weight's dtype when it is a
        # normal float, else bf16. Both quantized bases take the bf16 branch --
        # e4m3 by the "float8" test, int8 because an integer dtype is not
        # floating point at all -- which is also the dtype their own forward
        # produces from a bf16 activation.
        if (true_original.weight.dtype.is_floating_point and
                "float8" not in str(true_original.weight.dtype)):
            compute_dtype = true_original.weight.dtype
        else:
            compute_dtype = torch.bfloat16
        with torch.no_grad():
            wrapper.lora_down.weight.data = down.to(device=device, dtype=compute_dtype)
            wrapper.lora_up.weight.data = up.to(device=device, dtype=compute_dtype)
        wrapper.lora_down = wrapper.lora_down.to(dtype=compute_dtype)
        wrapper.lora_up = wrapper.lora_up.to(dtype=compute_dtype)
        wrapper.scale = (alpha_value / rank) * strength
        _set_module(parent, attr, wrapper)
        wrapped_keys.add(module_path)
        applied += 1
    return applied


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
) -> int:
    """Revert every wrapped module to its pre-LoRA original."""
    restored = 0
    for module_path, parent, attr, _linear in iter_krea2_lora_targets(transformer, _FULL_SCOPE):
        if module_path in lora_original_modules:
            _set_module(parent, attr, lora_original_modules[module_path])
            restored += 1
    wrapped_keys.clear()
    return restored
