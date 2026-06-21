"""LoRA support for MiniT2I (target enumeration + inference apply).

Targets (mirroring the reference DEFAULT_TARGET_MODULES = qkv/attn_proj/w1/w2/w3/
txt_embedder/pooled_embedder), grouped by scope:
  attn:      double_blocks.N.{img_qkv,txt_qkv,img_attn_proj,txt_attn_proj},
             txt_preamble_blocks.N.{qkv,attn_proj}
  mlp:       double_blocks.N.{img_mlp,txt_mlp}.{w1,w2,w3}, txt_preamble_blocks.N.mlp.{w1,w2,w3}
  txt_embed: txt_embedder, pooled_embedder

Module paths are relative to the MiniT2IMMJiTModel (e.g. model.net.double_blocks.0.img_qkv).
Keys use a reversible "."<->"__" encoding ("lora_unet_<path with . as __>") — module
names contain only single underscores, so "__" is unambiguously the path separator.
All targets are plain nn.Linear (no fp8). Forward-time addition, fully reversible.
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
    if not key.startswith("lora_unet_"):
        return None
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


DEFAULT_SCOPE: Dict[str, bool] = {"attn": True, "mlp": True, "txt_embed": True}
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


def _net(transformer: nn.Module) -> Optional[nn.Module]:
    # MiniT2IMMJiTModel.model.net (MMJiT)
    model = getattr(transformer, "model", None)
    return getattr(model, "net", None) if model is not None else None


def iter_minit2i_lora_targets(
    transformer: nn.Module,
    scope: Optional[Dict[str, bool]] = None,
) -> Generator[Tuple[str, Any, Any, nn.Module], None, None]:
    """Yield (module_path, parent, attr, current_module) for each LoRA target.

    module_path is relative to `transformer` (e.g. "model.net.double_blocks.0.img_qkv").
    """
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    scope = scope if scope is not None else DEFAULT_SCOPE
    want_attn = bool(scope.get("attn", False))
    want_mlp = bool(scope.get("mlp", False))
    want_txt_embed = bool(scope.get("txt_embed", False))
    is_target = lambda m: isinstance(m, (nn.Linear, LoRALinearLayer))

    net = _net(transformer)
    if net is None:
        return

    def emit_block(block, prefix):
        # attention
        if want_attn:
            for attr in ("img_qkv", "txt_qkv", "img_attn_proj", "txt_attn_proj", "qkv", "attn_proj"):
                m = getattr(block, attr, None)
                if is_target(m):
                    yield f"{prefix}.{attr}", block, attr, m
        if want_mlp:
            for mlp_name in ("img_mlp", "txt_mlp", "mlp"):
                mlp = getattr(block, mlp_name, None)
                if mlp is not None:
                    for wname in ("w1", "w2", "w3"):
                        m = getattr(mlp, wname, None)
                        if is_target(m):
                            yield f"{prefix}.{mlp_name}.{wname}", mlp, wname, m

    for i, block in enumerate(getattr(net, "double_blocks", [])):
        yield from emit_block(block, f"model.net.double_blocks.{i}")
    for i, block in enumerate(getattr(net, "txt_preamble_blocks", [])):
        yield from emit_block(block, f"model.net.txt_preamble_blocks.{i}")

    if want_txt_embed:
        for attr in ("txt_embedder", "pooled_embedder"):
            m = getattr(net, attr, None)
            if is_target(m):
                yield f"model.net.{attr}", net, attr, m


def _set_module(parent: Any, attr: Any, module: nn.Module) -> None:
    if isinstance(attr, int):
        parent[attr] = module
    else:
        setattr(parent, attr, module)


def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    scope: Optional[Dict[str, bool]] = None,
) -> int:
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    effective_scope = scope if scope is not None else _FULL_SCOPE
    applied = 0
    for module_path, parent, attr, linear in iter_minit2i_lora_targets(transformer, effective_scope):
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
        compute_dtype = true_original.weight.dtype if true_original.weight.dtype.is_floating_point else torch.float32
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
    restored = 0
    for module_path, parent, attr, _linear in iter_minit2i_lora_targets(transformer, _FULL_SCOPE):
        if module_path in lora_original_modules:
            _set_module(parent, attr, lora_original_modules[module_path])
            restored += 1
    wrapped_keys.clear()
    return restored


def flatten_to_key(module_path: str) -> str:
    """Module path -> sd-scripts-style LoRA key stem ('lora_unet_<flat>')."""
    return f"lora_unet_{_flatten(module_path)}"
