"""Load a standard SDXL single-file checkpoint for style-bridge training."""

from __future__ import annotations

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors import safe_open
from transformers import CLIPTextConfig, CLIPTextModel


_TE1_PREFIX = "conditioner.embedders.0.transformer.text_model."


def load_sdxl_style_pipeline(path: str, dtype: torch.dtype = torch.bfloat16) -> StableDiffusionXLPipeline:
    # diffusers' single-file converter still expects CLIPTextModel.text_model;
    # Transformers 5 moved TE1's layers to the CLIPTextModel root.
    from core.transformers_clip_compat import install as install_clip_compat
    install_clip_compat()
    pipeline = StableDiffusionXLPipeline.from_single_file(
        path, text_encoder=None, torch_dtype=dtype, use_safetensors=True,
    )
    config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    text_encoder = CLIPTextModel(config)
    with safe_open(path, framework="pt", device="cpu") as artifact:
        keys = [key for key in artifact.keys() if key.startswith(_TE1_PREFIX)]
        state = {key[len(_TE1_PREFIX):]: artifact.get_tensor(key) for key in keys}
    missing, unexpected = text_encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"SDXL TE1 weights do not match: missing={missing[:5]}, unexpected={unexpected[:5]}")
    pipeline.text_encoder = text_encoder.to(dtype=dtype).eval().requires_grad_(False)
    if pipeline.tokenizer is None:
        raise ValueError("SDXL checkpoint loader did not resolve the first tokenizer")
    return pipeline
