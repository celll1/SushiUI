"""Ming frozen ConvRot execution policy and resume function contract."""
import torch

MING_BLOCK_GROUPS = ("noise_refiner", "context_refiner", "layers")
BASE_FORWARD = "convrot_int8_bf16_backward_v1"
MODES = {"auto", "dequant", "transient_bf16", "cached_bf16", "prefetch_bf16"}
OPTIMIZED = MODES - {"auto", "dequant"}


def settings(config, *, method):
    from api.param_defaults import TRAINING_DEFAULTS
    mode = config.get("qwen_convrot_training_forward", TRAINING_DEFAULTS["qwen_convrot_training_forward"])
    blocks = config.get("qwen_convrot_backward_cache_blocks", TRAINING_DEFAULTS["qwen_convrot_backward_cache_blocks"])
    depth = config.get("qwen_convrot_backward_prefetch_depth", TRAINING_DEFAULTS["qwen_convrot_backward_prefetch_depth"])
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Ming ConvRot training forward policy is invalid")
    if type(blocks) is not int or not 1 <= blocks <= 8:
        raise ValueError("Ming ConvRot backward cache blocks must be an integer in [1, 8]")
    if type(depth) is not int or not 0 <= depth <= 7:
        raise ValueError("Ming ConvRot backward prefetch depth must be an integer in [0, 7]")
    if method != "lora" and mode in OPTIMIZED:
        raise ValueError("Ming optimized ConvRot forward requires a frozen LoRA base")
    if (mode == "prefetch_bf16" or (mode == "auto" and method == "lora")) and depth >= blocks:
        raise ValueError("Ming ConvRot prefetch depth must be smaller than cache blocks")
    return mode, blocks, depth


def resolve_policy(owner, model):
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    mode, blocks, depth = settings(owner.config, method="lora")
    packed = any(type(child) is ConvRotInt8Linear for child in model.modules())
    compatible = packed and owner.training_dtype == torch.bfloat16 and torch.device(owner.device).type == "cuda"
    if mode == "auto":
        mode = "prefetch_bf16" if compatible else "dequant"
    if mode in OPTIMIZED and not compatible:
        raise ValueError("Ming optimized ConvRot training requires a ConvRot base, CUDA device and BF16 training")
    if mode in {"cached_bf16", "prefetch_bf16"} and owner.blocks_to_swap > 0:
        raise ValueError("Ming cached ConvRot backward cannot be combined with block swap")
    return mode, blocks, depth


def configure_training(owner, model):
    from .quantization import configure_frozen_quantized_training
    mode, blocks, depth = resolve_policy(owner, model)
    report = configure_frozen_quantized_training(model)
    owner.ming_convrot_training_forward = mode
    owner.ming_convrot_backward_prefetch_cache = None
    if mode == "dequant":
        return report
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.common import quantized_frozen_training as shared
    model.to(device=owner.device)
    for child in model.modules():
        if type(child) is ConvRotInt8Linear:
            child._force_dequant = False
    cache_bytes = 0
    if mode == "transient_bf16":
        count = shared.enable_frozen_training_fused(model, label="Ming training transformer")
    elif mode == "cached_bf16":
        count, cache_bytes = shared.enable_frozen_training_cached_backward(model, dtype=owner.training_dtype,
            label="Ming training transformer")
    else:
        count, cache_bytes, cache = shared.enable_frozen_training_prefetch_backward(model,
            dtype=owner.training_dtype, cache_blocks=blocks, prefetch_depth=depth,
            block_groups=MING_BLOCK_GROUPS, label="Ming training transformer")
        owner.ming_convrot_backward_prefetch_cache = cache
    if not count:
        raise ValueError("Ming optimized ConvRot policy found no eligible frozen layers")
    return {**report, "execution": BASE_FORWARD, "policy": mode,
            "eligible_layers": count, "backward_cache_bytes": cache_bytes,
            "cache_blocks": blocks if mode == "prefetch_bf16" else None,
            "prefetch_depth": depth if mode == "prefetch_bf16" else None}


def checkpoint_metadata(owner):
    mode = getattr(owner, "ming_convrot_training_forward", "dequant")
    return {"ming_base_forward": BASE_FORWARD} if mode in OPTIMIZED else {}


def validate_resume_metadata(owner, metadata):
    expected = checkpoint_metadata(owner).get("ming_base_forward", "")
    actual = metadata.get("ming_base_forward", "")
    if actual != expected:
        raise ValueError("Ming LoRA resume base-forward contract does not match this run; select the saved ConvRot policy")



def validate_generation_metadata(model, metadata):
    actual = metadata.get("ming_base_forward", "")
    if not actual:
        return
    if actual != BASE_FORWARD:
        raise ValueError("Ming adapter declares an unsupported base-forward contract")
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    linears = [child for child in model.modules() if type(child) is ConvRotInt8Linear]
    if not linears or any(child.compute_dtype != torch.bfloat16 for child in linears):
        raise ValueError("Ming ConvRot-forward adapter requires a ConvRot base loaded in BF16")
