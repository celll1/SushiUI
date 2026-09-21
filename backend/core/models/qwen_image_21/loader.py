"""Load Qwen-Image 2.1 from a Diffusers directory or SushiUI split artifact."""

from __future__ import annotations

import json
import os

import torch

from .artifact import load_manifest, load_text_encoder, load_transformer, load_vae
from .vendor import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel


def _source_components(model_path: str, dtype: torch.dtype) -> dict:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    required = ("transformer", "text_encoder", "vae", "processor")
    missing = [name for name in required if not os.path.isdir(os.path.join(model_path, name))]
    if missing:
        raise ValueError(f"{model_path}: missing Qwen-Image 2.1 subfolder(s): {', '.join(missing)}")
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    vae = AutoencoderKLQwenImage21.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_path, "processor"))
    scheduler_dir = os.path.join(model_path, "scheduler")
    scheduler = (
        FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_dir)
        if os.path.isdir(scheduler_dir)
        else FlowMatchEulerDiscreteScheduler()
    )
    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "processor": processor,
        "scheduler": scheduler,
        "transformer_variant": "bf16",
        "text_encoder_variant": "bf16",
    }


def _artifact_components(model_path: str, dtype: torch.dtype, load_text_encoder_component: bool) -> dict:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLProcessor

    manifest = load_manifest(model_path)
    transformer, transformer_variant = load_transformer(
        manifest.transformer, manifest.transformer_config, dtype
    )
    text_encoder = None
    text_encoder_variant = None
    if load_text_encoder_component:
        text_encoder, text_encoder_variant = load_text_encoder(
            manifest.text_encoder, manifest.text_encoder_config, dtype
        )
    vae = load_vae(manifest.vae, manifest.vae_config, dtype)
    processor = Qwen3VLProcessor.from_pretrained(manifest.processor)
    scheduler = (
        FlowMatchEulerDiscreteScheduler.from_pretrained(manifest.scheduler)
        if manifest.scheduler and os.path.isdir(manifest.scheduler)
        else FlowMatchEulerDiscreteScheduler()
    )
    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "processor": processor,
        "scheduler": scheduler,
        "transformer_variant": transformer_variant,
        "text_encoder_variant": text_encoder_variant,
        "manifest": manifest,
    }


def load_qwen_image_21_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    *,
    load_text_encoder: bool = True,
) -> dict:
    """Return CPU-resident components consumed by ``PipelineManager`` and trainers."""
    manifest_path = os.path.join(model_path, "manifest.json") if os.path.isdir(model_path) else ""
    if manifest_path and os.path.isfile(manifest_path):
        components = _artifact_components(model_path, torch_dtype, load_text_encoder)
        components["companion_path"] = os.path.abspath(model_path)
    elif os.path.isdir(model_path):
        components = _source_components(model_path, torch_dtype)
        components["companion_path"] = os.path.abspath(model_path)
        if not load_text_encoder:
            components["text_encoder"] = None
    elif os.path.isfile(model_path):
        from core.models.common.single_file_format import is_index_path
        if is_index_path(model_path):
            with open(model_path, encoding="utf-8") as handle:
                metadata = (json.load(handle).get("metadata") or {})
        else:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
        if metadata.get("model_type") != "qwen_image_21" or metadata.get("component") != "transformer":
            raise ValueError(f"{model_path}: not a Qwen-Image 2.1 transformer checkpoint")
        companion_path = metadata.get("companion_path")
        if not companion_path or os.path.abspath(companion_path) == os.path.abspath(model_path):
            raise ValueError(
                f"{model_path}: training checkpoint has no usable companion_path for TE/VAE/processor"
            )
        components = load_qwen_image_21_components(
            companion_path, torch_dtype=torch_dtype, load_text_encoder=load_text_encoder
        )
        config = json.loads(metadata.get("config") or "{}")
        transformer, variant = load_transformer(model_path, config, torch_dtype)
        components["transformer"] = transformer
        components["transformer_variant"] = variant
        components["checkpoint_path"] = model_path
    else:
        raise ValueError(
            "Qwen-Image 2.1 uses a split component artifact. Select its directory containing manifest.json."
        )

    for name in ("transformer", "text_encoder", "vae"):
        module = components.get(name)
        if module is not None:
            module.to("cpu")
            module.eval()
    components.update(
        type="qwen_image_21",
        vae_scale_factor=16,
        latent_channels=64,
        pixel_alignment=32,
    )
    return components


def build_pipeline(components: dict) -> QwenImage21Pipeline:
    text_encoder = components.get("text_encoder")
    if text_encoder is None:
        raise RuntimeError("Qwen-Image 2.1 text encoder was not loaded")
    return QwenImage21Pipeline(
        scheduler=components["scheduler"],
        vae=components["vae"],
        text_encoder=text_encoder,
        processor=components["processor"],
        transformer=components["transformer"],
    )
