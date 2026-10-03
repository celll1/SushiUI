"""On-demand SDXL generation from a trained style IP-Adapter artifact."""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import os
import threading

import torch
from PIL import Image
from transformers import Siglip2ImageProcessor

from core.inference.attention_processors import set_attention_processor, set_style_ip_context
from core.model_loader import ModelLoader
from .sdxl_ip_adapter import load_sdxl_style_ip_adapter
from .sdxl_ip_runtime import load_sdxl_style_pipeline
from .style_artifacts import ENCODER_SUFFIX, METADATA_SUFFIX, load_style_encoder


def sha256_file(path: str) -> str:
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class SDXLStyleGenerationService:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def generate(self, *, model_path: str, encoder_path: str, adapter_path: str,
                 reference: Image.Image, prompt: str, negative_prompt: str,
                 width: int, height: int, steps: int, guidance_scale: float,
                 style_strength: float, seed: int) -> str:
        if not model_path.endswith(".safetensors") or not encoder_path.endswith(ENCODER_SUFFIX) \
                or not adapter_path.endswith(".safetensors"):
            raise ValueError("Expected SDXL, style encoder, and bridge .safetensors artifacts")
        for path in (model_path, encoder_path, adapter_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        metadata_path = encoder_path[:-len(ENCODER_SUFFIX)] + METADATA_SUFFIX
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(metadata_path)
        if not torch.cuda.is_available():
            raise RuntimeError("SDXL style generation requires a CUDA GPU")
        with self._lock:
            model_sha = sha256_file(model_path)
            encoder_sha = sha256_file(encoder_path)
            pipe = None
            style_model = None
            adapter = None
            try:
                pipe = load_sdxl_style_pipeline(model_path)
                v_prediction = ModelLoader.detect_v_prediction(model_path)
                if v_prediction:
                    ModelLoader._configure_v_prediction_scheduler(pipe)
                pipe.to("cuda")
                set_attention_processor(pipe.unet)
                adapter = load_sdxl_style_ip_adapter(
                    pipe.unet, adapter_path, style_encoder_sha256=encoder_sha,
                    sdxl_model_sha256=model_sha,
                ).to("cuda", dtype=torch.float32).eval()
                with open(metadata_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                processor = Siglip2ImageProcessor.from_dict(metadata["preprocessing"])
                inputs = processor(images=[reference], max_num_patches=int(metadata["style_max_num_patches"]),
                                   return_tensors="pt")
                style_model = load_style_encoder(encoder_path, "sdpa").to(
                    "cuda", dtype=torch.bfloat16).eval()
                with torch.inference_mode():
                    embedding = style_model.embed(
                        inputs["pixel_values"].to("cuda", dtype=torch.bfloat16),
                        inputs["pixel_attention_mask"].to("cuda"),
                        inputs["spatial_shapes"].to("cuda"),
                    ).float()
                    context = adapter.context(embedding, scale=style_strength,
                                              cfg_batch=guidance_scale > 1.0)
                    set_style_ip_context(pipe.unet, context)
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    image = pipe(
                        prompt=prompt, negative_prompt=negative_prompt,
                        width=width, height=height, num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        guidance_rescale=0.7 if v_prediction else 0.0,
                        generator=generator,
                    ).images[0]
                output = io.BytesIO()
                image.save(output, format="PNG")
                return base64.b64encode(output.getvalue()).decode("ascii")
            finally:
                if pipe is not None:
                    set_style_ip_context(pipe.unet, None)
                del adapter, style_model, pipe
                gc.collect()
                torch.cuda.empty_cache()


sdxl_style_generation_service = SDXLStyleGenerationService()
