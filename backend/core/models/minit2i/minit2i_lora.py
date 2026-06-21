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


# Transformer LoRA keys use the "lora_unet_" prefix; FLAN-T5 (text encoder)
# LoRA keys use "lora_te_". Both can live in one safetensors; normalise_lora_state_dict
# namespaces TE module paths with "te::" so apply routes them to the right module.
TE_KEY_PREFIX = "lora_te_"
TE_NAMESPACE = "te::"


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    for prefix, ns in (("lora_unet_", ""), (TE_KEY_PREFIX, TE_NAMESPACE)):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            for suffix, tag in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
                if rest.endswith(suffix):
                    return ns + _restore(rest[: -len(suffix)]), tag
            return None
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
    fmt = "sd-scripts" if any(k.startswith("lora_unet_") or k.startswith(TE_KEY_PREFIX) for k in raw) else "unknown"
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


# ---- FLAN-T5 (text encoder) LoRA scope ----
TE_DEFAULT_SCOPE: Dict[str, bool] = {"attn": True, "ff": True}
_TE_FULL_SCOPE: Dict[str, bool] = {k: True for k in TE_DEFAULT_SCOPE}


def parse_te_scope_csv(scope_csv: Optional[str]) -> Dict[str, bool]:
    scope = {k: False for k in TE_DEFAULT_SCOPE}
    if not scope_csv:
        return dict(TE_DEFAULT_SCOPE)
    for tok in scope_csv.split(","):
        tok = tok.strip()
        if tok in scope:
            scope[tok] = True
    if not any(scope.values()):
        return dict(TE_DEFAULT_SCOPE)
    return scope


def iter_minit2i_te_lora_targets(
    text_encoder: nn.Module,
    scope: Optional[Dict[str, bool]] = None,
) -> Generator[Tuple[str, Any, Any, nn.Module], None, None]:
    """Yield (module_path, parent, attr, current_module) for each FLAN-T5 LoRA target.

    module_path is relative to the T5EncoderModel
    (e.g. "encoder.block.0.layer.0.SelfAttention.q"). Targets:
      attn: SelfAttention.{q,k,v,o}
      ff:   DenseReluDense.{wi,wi_0,wi_1,wo} (gated-gelu uses wi_0/wi_1)
    """
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    scope = scope if scope is not None else TE_DEFAULT_SCOPE
    want_attn = bool(scope.get("attn", False))
    want_ff = bool(scope.get("ff", False))
    is_target = lambda m: isinstance(m, (nn.Linear, LoRALinearLayer))

    encoder = getattr(text_encoder, "encoder", None)
    if encoder is None:
        return
    for i, block in enumerate(getattr(encoder, "block", [])):
        layers = getattr(block, "layer", None)
        if layers is None:
            continue
        if want_attn and len(layers) >= 1:
            sa = getattr(layers[0], "SelfAttention", None)
            if sa is not None:
                for attr in ("q", "k", "v", "o"):
                    m = getattr(sa, attr, None)
                    if is_target(m):
                        yield f"encoder.block.{i}.layer.0.SelfAttention.{attr}", sa, attr, m
        if want_ff and len(layers) >= 2:
            ff = getattr(layers[-1], "DenseReluDense", None)
            if ff is not None:
                for attr in ("wi", "wi_0", "wi_1", "wo"):
                    m = getattr(ff, attr, None)
                    if is_target(m):
                        yield f"encoder.block.{i}.layer.1.DenseReluDense.{attr}", ff, attr, m


def flatten_to_te_key(module_path: str) -> str:
    """T5 module path -> sd-scripts-style LoRA key stem ('lora_te_<flat>')."""
    return f"{TE_KEY_PREFIX}{_flatten(module_path)}"


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


def _apply_group(
    targets,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    namespace: str,
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
) -> int:
    """Wrap each target Linear with a LoRALinearLayer carrying the grouped weights.

    grouped keys are namespaced ("" for transformer, "te::" for the text encoder);
    `namespace` selects which entries this pass consumes so a single state dict can
    hold both transformer and TE LoRA without collision.
    """
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    applied = 0
    for module_path, parent, attr, linear in targets:
        key = namespace + module_path
        weights = grouped.get(key)
        if weights is None:
            continue
        down, up = weights["down"], weights["up"]
        alpha_tensor = weights.get("alpha")
        true_original = linear.original_module if isinstance(linear, LoRALinearLayer) else linear
        lora_original_modules.setdefault(key, true_original)
        rank = int(down.shape[0])
        alpha_value = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
        wrapper = LoRALinearLayer(true_original, rank=rank, alpha=alpha_value, lora_name=key)
        device = true_original.weight.device
        compute_dtype = true_original.weight.dtype if true_original.weight.dtype.is_floating_point else torch.float32
        with torch.no_grad():
            wrapper.lora_down.weight.data = down.to(device=device, dtype=compute_dtype)
            wrapper.lora_up.weight.data = up.to(device=device, dtype=compute_dtype)
        wrapper.lora_down = wrapper.lora_down.to(dtype=compute_dtype)
        wrapper.lora_up = wrapper.lora_up.to(dtype=compute_dtype)
        wrapper.scale = (alpha_value / rank) * strength
        _set_module(parent, attr, wrapper)
        wrapped_keys.add(key)
        applied += 1
    return applied


def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    scope: Optional[Dict[str, bool]] = None,
) -> int:
    return _apply_group(
        iter_minit2i_lora_targets(transformer, scope if scope is not None else _FULL_SCOPE),
        grouped, "", strength, lora_original_modules, wrapped_keys,
    )


def apply_te_lora_group(
    text_encoder: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    scope: Optional[Dict[str, bool]] = None,
) -> int:
    return _apply_group(
        iter_minit2i_te_lora_targets(text_encoder, scope if scope is not None else _TE_FULL_SCOPE),
        grouped, TE_NAMESPACE, strength, lora_original_modules, wrapped_keys,
    )


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    text_encoder: Optional[nn.Module] = None,
) -> int:
    restored = 0
    for module_path, parent, attr, _linear in iter_minit2i_lora_targets(transformer, _FULL_SCOPE):
        if module_path in lora_original_modules:
            _set_module(parent, attr, lora_original_modules[module_path])
            restored += 1
    if text_encoder is not None:
        for module_path, parent, attr, _linear in iter_minit2i_te_lora_targets(text_encoder, _TE_FULL_SCOPE):
            key = TE_NAMESPACE + module_path
            if key in lora_original_modules:
                _set_module(parent, attr, lora_original_modules[key])
                restored += 1
    wrapped_keys.clear()
    return restored


def flatten_to_key(module_path: str) -> str:
    """Module path -> sd-scripts-style LoRA key stem ('lora_unet_<flat>')."""
    return f"lora_unet_{_flatten(module_path)}"
