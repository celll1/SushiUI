"""Native DiT adapter slots and reversible canonical checkpoint keys."""
from core.models.krea2.krea2_lora import (
    build_lora_branch, declared_branch_count, detect_lora_format,
    flatten_to_key, normalise_lora_state_dict,
)


def iter_lora_slots(transformer, *, attention=True, projection=True):
    from torch import nn
    from core.adapters import CompositeAdapterLayer, LoRALinearLayer
    from core.adapters.targets import is_lora_wrappable_linear

    def supported(module):
        return isinstance(module, (CompositeAdapterLayer, LoRALinearLayer)) or is_lora_wrappable_linear(module)

    if attention:
        for group in ("noise_refiner", "context_refiner", "layers"):
            for index, block in enumerate(getattr(transformer, group, ())):
                module = block.attention
                prefix = f"{group}.{index}.attention"
                for attr in ("to_q", "to_k", "to_v"):
                    if supported(getattr(module, attr, None)):
                        yield module, attr, f"{prefix}.{attr}"
                output = module.to_out
                if isinstance(output, (nn.ModuleList, nn.Sequential)) and len(output) and supported(output[0]):
                    yield output, 0, f"{prefix}.to_out.0"
    if projection:
        module = getattr(transformer, "cap_embedder", None)
        if isinstance(module, nn.Sequential) and len(module) > 1 and supported(module[1]):
            yield module, 1, "cap_embedder.1"


__all__ = ["build_lora_branch", "declared_branch_count", "detect_lora_format",
           "flatten_to_key", "iter_lora_slots", "normalise_lora_state_dict"]
