"""LoRA key codec and target inventory for Qwen-Image 2.1."""

from core.models.krea2.krea2_lora import (
    build_lora_branch,
    declared_branch_count as _declared_branch_count,
    detect_lora_format,
    flatten_to_key,
    normalise_lora_state_dict,
)


def _cond_as_shared_keys(raw):
    return {
        key.replace("lora_cond_unet_", "lora_unet_", 1): value
        for key, value in raw.items() if key.startswith("lora_cond_unet_")
    }


def declared_branch_count(raw):
    return _declared_branch_count(raw) + _declared_branch_count(_cond_as_shared_keys(raw))


def normalise_cond_lora_state_dict(raw):
    if any(key.startswith("lora_unet_") or key.startswith("lora_uncond_unet_") for key in raw):
        raise ValueError("Qwen cond/base checkpoint contains shared or uncond LoRA tensors")
    mapped = _cond_as_shared_keys(raw)
    if not mapped:
        raise ValueError("Qwen cond/base checkpoint has no conditional LoRA tensors")
    return normalise_lora_state_dict(mapped)


def build_cond_lora_branch(base, group, module_path):
    from core.adapters import build_adapter_branch, lora_branch_dtype
    from core.models.qwen_image_21.branch_lora import QwenCondLoRALinearLayer

    return build_adapter_branch(
        base, group, layer_cls=QwenCondLoRALinearLayer,
        lora_dtype=lora_branch_dtype(base), lora_name=module_path,
    )


def iter_lora_slots(transformer):
    import torch.nn as nn
    from core.adapters import CompositeAdapterLayer, LoRALinearLayer
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    target_types = (nn.Linear, ConvRotInt8Linear, LoRALinearLayer, CompositeAdapterLayer)
    for index, block in enumerate(getattr(transformer, "transformer_blocks", ())):
        attention = getattr(block, "attn", None)
        if attention is None:
            continue
        prefix = f"transformer_blocks.{index}.attn"
        for attr in ("to_q", "to_k", "to_v"):
            if isinstance(getattr(attention, attr, None), target_types):
                yield attention, attr, f"{prefix}.{attr}"
        to_out = getattr(attention, "to_out", None)
        if isinstance(to_out, (nn.ModuleList, nn.Sequential)) and to_out and isinstance(to_out[0], target_types):
            yield to_out, 0, f"{prefix}.to_out.0"


__all__ = [
    "build_lora_branch",
    "build_cond_lora_branch",
    "declared_branch_count",
    "detect_lora_format",
    "flatten_to_key",
    "iter_lora_slots",
    "normalise_lora_state_dict",
    "normalise_cond_lora_state_dict",
]
