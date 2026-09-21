"""LoRA key codec and target inventory for Qwen-Image 2.1."""

from core.models.krea2.krea2_lora import (
    build_lora_branch,
    declared_branch_count,
    detect_lora_format,
    flatten_to_key,
    normalise_lora_state_dict,
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
    "declared_branch_count",
    "detect_lora_format",
    "flatten_to_key",
    "iter_lora_slots",
    "normalise_lora_state_dict",
]
