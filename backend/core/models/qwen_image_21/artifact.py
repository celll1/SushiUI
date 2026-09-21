"""Split single-file artifact contract for Qwen-Image 2.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from core.models.common.single_file_format import read_state_dict

from .vendor import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel


MODEL_TYPE = "qwen_image_21"
FORMAT_VERSION = "1"
DEFAULT_TRANSFORMER_CONFIG = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": 64,
    "num_layers": 32,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "context_in_dim": 4096,
    "mlp_ratio": 3,
    "axes_dims_rope": [16, 56, 56],
    "eps": 1e-6,
    "causal_condition": True,
}


@dataclass(frozen=True)
class ArtifactManifest:
    root: str
    transformer: str
    text_encoder: str
    vae: str
    processor: str
    scheduler: str | None
    transformer_config: dict[str, Any]
    text_encoder_config: str
    vae_config: dict[str, Any]


def _absolute(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def load_manifest(path: str) -> ArtifactManifest:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Qwen-Image 2.1 manifest not found: {candidate}")
    with candidate.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("model_type") != MODEL_TYPE:
        raise ValueError(f"{candidate}: model_type must be {MODEL_TYPE!r}")
    if str(raw.get("format_version")) != FORMAT_VERSION:
        raise ValueError(f"{candidate}: unsupported format_version {raw.get('format_version')!r}")
    root = candidate.parent
    components = raw.get("components") or {}
    required = ("transformer", "text_encoder", "vae", "processor", "text_encoder_config")
    missing = [name for name in required if not components.get(name)]
    if missing:
        raise ValueError(f"{candidate}: missing component path(s): {', '.join(missing)}")
    return ArtifactManifest(
        root=str(root),
        transformer=_absolute(root, components["transformer"]),
        text_encoder=_absolute(root, components["text_encoder"]),
        vae=_absolute(root, components["vae"]),
        processor=_absolute(root, components["processor"]),
        scheduler=_absolute(root, components.get("scheduler")),
        transformer_config=dict(raw.get("transformer_config") or DEFAULT_TRANSFORMER_CONFIG),
        text_encoder_config=_absolute(root, components["text_encoder_config"]),
        vae_config=dict(raw.get("vae_config") or {}),
    )


def artifact_metadata(component: str, config: dict[str, Any], *, variant: str = "bf16") -> dict[str, str]:
    return {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "component": component,
        "variant": variant,
        "config": json.dumps(config, sort_keys=True, default=str),
        "format": "pt",
    }


def _validate_component(path: str, metadata: dict[str, str], expected: str) -> str:
    if metadata.get("model_type") != MODEL_TYPE or metadata.get("component") != expected:
        raise ValueError(
            f"{path}: expected {MODEL_TYPE}/{expected}, got "
            f"{metadata.get('model_type')!r}/{metadata.get('component')!r}"
        )
    variant = metadata.get("variant", "bf16")
    if variant not in {"bf16", "int8_convrot"}:
        raise ValueError(f"{path}: unsupported variant {variant!r}")
    return variant


def _marker_config(marker: torch.Tensor, path: str, layer: str) -> dict[str, int]:
    try:
        fields = json.loads(bytes(marker.detach().cpu().reshape(-1).tolist()).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"{path}: invalid ConvRot marker for {layer}") from exc
    expected = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    if fields != expected:
        raise ValueError(f"{path}: unsupported ConvRot marker for {layer}: {fields!r}")
    return {"convrot_groupsize": 256, "marker_numel": int(marker.numel())}


def convrot_layers(state: dict[str, torch.Tensor], path: str) -> dict[str, dict[str, int]]:
    layers: dict[str, dict[str, int]] = {}
    for key, marker in state.items():
        if not key.endswith(".comfy_quant"):
            continue
        layer = key[: -len(".comfy_quant")]
        weight = state.get(f"{layer}.weight")
        scale = state.get(f"{layer}.weight_scale")
        if weight is None or scale is None:
            raise ValueError(f"{path}: ConvRot layer {layer!r} is missing weight or weight_scale")
        if weight.ndim != 2 or weight.dtype is not torch.int8:
            raise ValueError(f"{path}: ConvRot layer {layer!r} must carry a 2-D int8 weight")
        if weight.shape[1] % 256 or scale.dtype is not torch.float32 or scale.numel() != weight.shape[0]:
            raise ValueError(f"{path}: invalid ConvRot geometry for {layer!r}")
        state[f"{layer}.weight_scale"] = scale.reshape(-1).contiguous()
        layers[layer] = _marker_config(marker, path, layer)
    return layers


def _finish_load(module: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> torch.nn.Module:
    info = module.load_state_dict(state, strict=False, assign=True)
    missing = [key for key in info.missing_keys if not key.endswith(".weight_scale") and not key.endswith(".comfy_quant")]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{label}: state mismatch; missing={missing[:10]}, unexpected={info.unexpected_keys[:10]}"
        )
    return module.eval()


def load_transformer(path: str, config: dict[str, Any], dtype: torch.dtype) -> tuple[torch.nn.Module, str]:
    from accelerate import init_empty_weights

    state, metadata = read_state_dict(path)
    variant = _validate_component(path, metadata, "transformer")
    with init_empty_weights():
        model = QwenImage21Transformer2DModel(**config)
    if variant == "int8_convrot":
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime, swap_linears_to_convrot_int8

        layers = convrot_layers(state, path)
        require_convrot_int8_runtime()
        swapped = swap_linears_to_convrot_int8(model, state, layers, dtype)
        if swapped != len(layers):
            raise ValueError(f"{path}: installed {swapped}/{len(layers)} ConvRot layers")
    model = _finish_load(model, state, "Qwen-Image 2.1 transformer")
    if variant == "bf16":
        model.to(dtype=dtype)
    return model.to("cpu"), variant


def load_text_encoder(path: str, config_path: str, dtype: torch.dtype) -> tuple[torch.nn.Module, str]:
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    state, metadata = read_state_dict(path)
    variant = _validate_component(path, metadata, "text_encoder")
    config = Qwen3VLConfig.from_json_file(config_path)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    if variant == "int8_convrot":
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime, swap_linears_to_convrot_int8

        layers = convrot_layers(state, path)
        require_convrot_int8_runtime()
        swapped = swap_linears_to_convrot_int8(model, state, layers, dtype)
        if swapped != len(layers):
            raise ValueError(f"{path}: installed {swapped}/{len(layers)} ConvRot layers")
    model = _finish_load(model, state, "Qwen-Image 2.1 text encoder")
    if variant == "bf16":
        model.to(dtype=dtype)
    return model.to("cpu"), variant


def load_vae(path: str, config: dict[str, Any], dtype: torch.dtype) -> torch.nn.Module:
    from accelerate import init_empty_weights

    state, metadata = read_state_dict(path)
    _validate_component(path, metadata, "vae")
    with init_empty_weights():
        model = AutoencoderKLQwenImage21(**config)
    return _finish_load(model, state, "Qwen-Image 2.1 VAE").to(dtype=dtype, device="cpu")


def is_qwen_image_21_artifact(path: str) -> bool:
    try:
        load_manifest(path)
        return True
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        return False
