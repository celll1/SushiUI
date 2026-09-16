"""Generation backend for SenseNova SDXL Chimera."""

from __future__ import annotations

import random
import time

import torch


class SenseNovaSDXLChimeraMixin:
    def _generate_img2txt_sensenova_sdxl_chimera(
        self, params, image, progress_callback=None
    ) -> tuple:
        """Run text output through the frozen understanding-only component."""
        components = self.sensenova_sdxl_chimera_components
        if not components:
            raise RuntimeError("SenseNova SDXL Chimera components not loaded")
        from transformers import StoppingCriteria, StoppingCriteriaList
        from core.models.sensenova.vendor.utils import load_image_native

        understanding = components["understanding"]
        transformer = understanding["transformer"]
        tokenizer = understanding["tokenizer"]
        max_new_tokens = int(params["max_new_tokens"])
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.SystemRandom().randint(0, 2**31 - 1)

        manager = self
        class _CancelAndProgress(StoppingCriteria):
            def __init__(inner_self):
                inner_self.generated = 0

            def __call__(inner_self, _input_ids, _scores, **_kwargs):
                inner_self.generated += 1
                if progress_callback is not None:
                    progress_callback(
                        min(inner_self.generated, max_new_tokens),
                        max_new_tokens,
                        "Generating text",
                    )
                return bool(manager.cancel_requested)

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(params.get("do_sample", False)),
            "stopping_criteria": StoppingCriteriaList([_CancelAndProgress()]),
        }
        if generation_config["do_sample"]:
            generation_config.update({
                "temperature": float(params.get("temperature", 0.7)),
                "top_p": float(params.get("top_p", 0.9)),
            })
            if params.get("top_k") is not None:
                generation_config["top_k"] = int(params["top_k"])
        if params.get("repetition_penalty") is not None:
            generation_config["repetition_penalty"] = float(params["repetition_penalty"])

        pixel_values = grid_hw = None
        try:
            transformer.to(self.device)
            if progress_callback is not None:
                progress_callback(0, max_new_tokens, "Encoding image")
            started = time.perf_counter()
            pixel_values, grid_hw = load_image_native(
                image,
                transformer.patch_size,
                transformer.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=2048 * 2048,
                upscale=False,
            )
            pixel_values = pixel_values.to(device=self.device, dtype=torch.bfloat16)
            grid_hw = grid_hw.to(self.device)
            preprocess_seconds = time.perf_counter() - started
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            started = time.perf_counter()
            with torch.inference_mode():
                response = transformer.chat(
                    tokenizer,
                    pixel_values,
                    params["instruction"],
                    generation_config,
                    history=None,
                    return_history=False,
                    grid_hw=grid_hw,
                    verbose=False,
                )
            return response, seed, {
                "preprocess_seconds": preprocess_seconds,
                "generation_seconds": time.perf_counter() - started,
            }
        finally:
            del pixel_values, grid_hw
            transformer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_txt2img_sensenova_sdxl_chimera(
        self, params, progress_callback=None, step_callback=None
    ) -> tuple:
        components = self.sensenova_sdxl_chimera_components
        if not components:
            raise RuntimeError("SenseNova SDXL Chimera components not loaded")
        from api.param_defaults import GENERATION_DEFAULTS
        from core.models.sensenova_sdxl_chimera import pipeline_ops

        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
        height = int(params.get("height") or GENERATION_DEFAULTS["height"])
        width = int(params.get("width") or GENERATION_DEFAULTS["width"])
        steps = int(params.get("steps") or GENERATION_DEFAULTS["steps"])
        cfg_scale = float(params.get("cfg_scale", GENERATION_DEFAULTS["cfg_scale"]))
        prompt = str(params.get("prompt") or "")
        negative_prompt = str(params.get("negative_prompt") or "")
        understanding = components["understanding"]
        transformer = understanding["transformer"]
        tokenizer = understanding["tokenizer"]
        bridge = components["condition_bridge"]
        unet = components["unet"]
        vae = components["vae"]
        device = self.device
        dtype = next(unet.parameters()).dtype

        positive = negative = latents = None
        try:
            transformer.to(device)
            bridge.to(device=device, dtype=dtype)
            with torch.inference_mode():
                positive = pipeline_ops.build_conditioning(
                    transformer, tokenizer, bridge, prompt
                )
                if cfg_scale > 1.0:
                    negative = pipeline_ops.build_conditioning(
                        transformer, tokenizer, bridge, negative_prompt
                    )
            transformer.to("cpu")
            bridge.to("cpu")
            unet.to(device)

            def report(step, total, current):
                if getattr(self, "cancel_requested", False):
                    raise RuntimeError("Generation cancelled by user")
                if progress_callback is not None:
                    progress_callback(step, total, current, None, current)

            latents = pipeline_ops.sample_txt2img_latents(
                unet,
                positive,
                negative,
                height=height,
                width=width,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                timestep_shift=float(params.get("timestep_shift", GENERATION_DEFAULTS["timestep_shift"])),
                cfg_mode=str(params.get("chimera_cfg_mode", "sequential")),
                original_height=int(params.get("original_height") or height),
                original_width=int(params.get("original_width") or width),
                crop_top=int(params.get("crop_top") or 0),
                crop_left=int(params.get("crop_left") or 0),
                attention_backend=str(params.get("attention_type") or "normal"),
                progress_callback=report,
            )
            unet.to("cpu")
            vae.to(device)
            with torch.inference_mode():
                image = pipeline_ops.decode_latents(vae, latents)
            return image, seed, 0
        finally:
            from core.models.sensenova_sdxl_chimera.attention_processor import (
                clear_chimera_attention_caches,
            )

            clear_chimera_attention_caches(unet)
            for component in (transformer, bridge, unet, vae):
                try:
                    component.to("cpu")
                except Exception as exc:
                    print(f"[Chimera] component offload failed: {exc}")
            del positive, negative, latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
