"""Pre-load contract for aligned H3 Full training."""


def preflight_full_training(config, model_path, *, bundle_vae_explicit_false=False):
    from api.param_defaults import TRAINING_DEFAULTS
    from core.training.vae_swap import resolve_vae_swap_source, initialization_mode

    if not config.get("train_unet", TRAINING_DEFAULTS["train_unet"]):
        raise ValueError("H3 aligned Full training requires train_unet=true")
    for name in ("train_text_encoder", "train_image_encoder"):
        if config.get(name, TRAINING_DEFAULTS[name]):
            raise ValueError(f"H3 aligned Full training requires {name}=false")
    if bundle_vae_explicit_false or config.get("bundle_vae") is False:
        raise ValueError("H3 aligned Full training requires a bundled target codec")
    if initialization_mode(config, "minimax_h3") != "paired_spatial":
        raise ValueError("H3 aligned Full training requires paired_spatial initialization")
    if not model_path:
        raise ValueError("H3 aligned Full training requires a base checkpoint")

    from .video_artifact import native_metadata, preflight_checkpoint, _headers
    if native_metadata(model_path) is not None:
        preflight_checkpoint(model_path)
        return
    if not resolve_vae_swap_source(config):
        raise ValueError("H3 Full training requires vae_swap_source or an aligned Full checkpoint")
    from safetensors import SafetensorError
    try:
        _, shapes, dtypes = _headers(model_path)
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"Cannot inspect H3 Full checkpoint: {exc}") from exc
    matrices = {key for key, shape in shapes.items() if len(shape) == 2 and key.endswith(".weight")}
    if not matrices:
        raise ValueError("H3 Full training requires an unquantized DiT safetensors checkpoint")
    if (any(dtypes[key] not in ("BF16", "F16", "F32") for key in matrices)
            or any(key.endswith((".comfy_quant", ".qweight", ".weight_scale", ".weight_scale_inv"))
                   for key in shapes)):
        raise ValueError("H3 Full training requires unquantized floating-point DiT weights; use LoRA for quantized weights")
