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
All targets are plain nn.Linear (no fp8). Each is covered ONCE by a
CompositeAdapterLayer holding one named branch per selected LoRA (forward-time
addition, fully reversible), so two LoRAs over one module SUM.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, Optional, Tuple

import torch
from torch import nn

from core.adapters.groups import (TensorGroup, declared_groups,
                                  group_adapter_tensors, split_adapter_suffix)


def _flatten(module_path: str) -> str:
    return module_path.replace(".", "__")


def _restore(flat: str) -> str:
    return flat.replace("__", ".")


# Transformer LoRA keys use the "lora_unet_" prefix; FLAN-T5 (text encoder)
# LoRA keys use "lora_te_". Both can live in one safetensors; normalise_lora_state_dict
# namespaces TE module paths with "te::" so apply routes them to the right module.
TE_KEY_PREFIX = "lora_te_"
TE_NAMESPACE = "te::"


def _minit2i_stem(raw_stem: str) -> Optional[str]:
    """Suffix-stripped key -> namespaced module path, or None for a foreign key."""
    for prefix, ns in (("lora_unet_", ""), (TE_KEY_PREFIX, TE_NAMESPACE)):
        if raw_stem.startswith(prefix):
            return ns + _restore(raw_stem[len(prefix):])
    return None


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """``(namespaced module path, canonical tensor name)``, or None."""
    split = split_adapter_suffix(key)
    if split is None:
        return None
    module_path = _minit2i_stem(split[0])
    return None if module_path is None else (module_path, split[1])


def declared_group_stems(raw: Dict[str, torch.Tensor]) -> Dict[str, TensorGroup]:
    """The groups this file declares, keyed like ``normalise_lora_state_dict``
    so the per-pass counter can split them on ``TE_NAMESPACE``; see
    ``declared_groups``."""
    return declared_groups(raw, _minit2i_stem)


def normalise_lora_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, TensorGroup]:
    """COMPLETE factor groups by namespaced module path, whatever the algebra.

    ``group_adapter_tensors`` already drops the incomplete ones; a down/up
    filter on top would silently drop every LoHa and LoKr group.
    """
    return group_adapter_tensors(raw, _minit2i_stem).groups


def detect_lora_format(keys: Iterable[str]) -> str:
    """"sd-scripts" / "unknown", from the key names alone."""
    return "sd-scripts" if any(
        k.startswith("lora_unet_") or k.startswith(TE_KEY_PREFIX) for k in keys
    ) else "unknown"


def alpha_from_metadata(metadata: Optional[Dict[str, str]]) -> Optional[float]:
    """File-level LoRA alpha, or None.

    Second rung of the precedence per-key ``.alpha`` tensor -> file metadata ->
    rank. Without it a trainer that records alpha only in metadata (kohya's
    ``ss_network_alpha``) silently applies at scale 1.0 instead of alpha/rank.
    """
    if not metadata:
        return None
    for key in ("lora_alpha", "ss_network_alpha"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


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

    ONE enumerator for both load and unload of the TEXT-ENCODER half, so the two
    cannot disagree about a slot once a target can hold more than one branch.

    module_path is relative to the T5EncoderModel
    (e.g. "encoder.block.0.layer.0.SelfAttention.q"). Targets:
      attn: SelfAttention.{q,k,v,o}
      ff:   DenseReluDense.{wi,wi_0,wi_1,wo} (gated-gelu uses wi_0/wi_1)
    """
    from core.adapters import CompositeAdapterLayer, LoRALinearLayer

    scope = scope if scope is not None else TE_DEFAULT_SCOPE
    want_attn = bool(scope.get("attn", False))
    want_ff = bool(scope.get("ff", False))
    # A composite is a target too: drop it and a second selected LoRA skips every
    # occupied slot and reports zero matches as if its keys were wrong.
    is_target = lambda m: isinstance(
        m, (nn.Linear, LoRALinearLayer, CompositeAdapterLayer))

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

    ONE enumerator for both load and unload of the TRANSFORMER half.

    module_path is relative to `transformer` (e.g. "model.net.double_blocks.0.img_qkv").
    """
    from core.adapters import CompositeAdapterLayer, LoRALinearLayer

    scope = scope if scope is not None else DEFAULT_SCOPE
    want_attn = bool(scope.get("attn", False))
    want_mlp = bool(scope.get("mlp", False))
    want_txt_embed = bool(scope.get("txt_embed", False))
    is_target = lambda m: isinstance(
        m, (nn.Linear, LoRALinearLayer, CompositeAdapterLayer))

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


def iter_minit2i_lora_slots(transformer: nn.Module):
    """``(parent, slot, module_path)`` over the FULL transformer scope, for
    ``AdapterSession``. Module paths are bare, so they cannot collide with the
    namespaced text-encoder ones."""
    for module_path, parent, attr, _current in iter_minit2i_lora_targets(
            transformer, _FULL_SCOPE):
        yield parent, attr, module_path


def iter_minit2i_te_lora_slots(text_encoder: nn.Module):
    """``(parent, slot, TE_NAMESPACE + module_path)`` over the FULL FLAN-T5 scope.

    The namespace is carried in the module path itself, so the session's
    per-component originals maps read exactly as the single map they replace and
    a text-encoder path can never be mistaken for a transformer one.
    """
    for module_path, parent, attr, _current in iter_minit2i_te_lora_targets(
            text_encoder, _TE_FULL_SCOPE):
        yield parent, attr, TE_NAMESPACE + module_path


def branch_dtype(base: nn.Module) -> torch.dtype:
    """MiniT2I's own branch dtype: every target is a plain ``nn.Linear``, so the
    base weight's dtype IS the compute dtype. Deliberately not
    ``core.adapters.lora_branch_dtype``, whose non-float fallback is bf16."""
    dtype = base.weight.dtype
    return dtype if dtype.is_floating_point else torch.float32


def build_lora_branch(
    base: nn.Module,
    group: TensorGroup,
    key: str,
    default_alpha: Optional[float] = None,
) -> nn.Module:
    """One branch for one target, built and not installed, or ``SHAPE_MISMATCH``.

    The algebra is the group's: ``build_adapter_branch`` dispatches on the tensor
    names. Alpha precedence: per-key ``.alpha`` tensor, then file metadata
    (``default_alpha``), then rank. The strength is NOT folded here --
    ``CompositeAdapterLayer.add_branch(strength=)`` does it, and multiplying it
    onto the delta instead loses bit-identity with the single-LoRA numerics.
    """
    from core.adapters import build_adapter_branch

    return build_adapter_branch(base, group, metadata_alpha=default_alpha,
                                lora_dtype=branch_dtype(base), lora_name=key)


def flatten_to_key(module_path: str) -> str:
    """Module path -> sd-scripts-style LoRA key stem ('lora_unet_<flat>')."""
    return f"lora_unet_{_flatten(module_path)}"
