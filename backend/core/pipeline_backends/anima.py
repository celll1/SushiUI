from typing import Dict, Any, Optional, List, Callable
from PIL import Image
import torch
import json
import os
import sys
import gc
import random
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
from config.settings import settings
from extensions.base_extension import BaseExtension
from core.model_loader import ModelLoader, ModelSource
from core.prompts.processors import PromptEditingProcessor
from core.inference.schedulers import get_scheduler
from core.inference.custom_sampling import custom_sampling_loop, custom_img2img_sampling_loop, custom_inpaint_sampling_loop

class AnimaMixin:
    """AnimaMixin: anima backend methods extracted verbatim from pipeline.py."""

    def _anima_resolve_dtype(self, dtype_str: Optional[str] = None) -> torch.dtype:
        if dtype_str == "fp16":
            return torch.float16
        if dtype_str == "fp32":
            return torch.float32
        return torch.bfloat16

    def _load_lora_anima(self, lora_configs: List[Dict]) -> int:
        """Wrap target Linear modules of the Anima DiT with LoRA adapters.

        Supports stacking multiple LoRAs on the same module (each subsequent
        wrap takes the existing wrapper's true original as its base, so
        unload always returns to the un-LoRA'd model).
        """
        from core.models.anima.anima_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs:
            return 0
        if not self.anima_components:
            print("[Anima LoRA] WARNING: components not loaded")
            return 0

        transformer = self.anima_components["transformer"]
        if not hasattr(self, "_anima_lora_original_modules"):
            self._anima_lora_original_modules: Dict[str, torch.nn.Linear] = {}
            self._anima_lora_wrapped_keys: set = set()

        total_applied = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[Anima LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw)
                print(f"[Anima LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} keys={len(raw)} matched_modules={len(grouped)} strength={strength}")
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._anima_lora_original_modules, self._anima_lora_wrapped_keys,
                )
                print(f"[Anima LoRA]   wrapped {applied} module(s)")
                total_applied += applied
            except Exception as e:
                print(f"[Anima LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total_applied

    def _unload_lora_anima(self) -> int:
        """Restore every Anima DiT Linear to its pre-LoRA original."""
        from core.models.anima.anima_lora import restore_originals
        if not getattr(self, "_anima_lora_wrapped_keys", None):
            return 0
        if not self.anima_components:
            return 0
        transformer = self.anima_components["transformer"]
        restored = restore_originals(
            transformer, self._anima_lora_original_modules, self._anima_lora_wrapped_keys,
        )
        print(f"[Anima LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    @staticmethod
    def _anima_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs from a generation params dict.

        Returns a dict consumed by anima_pipeline_ops._apply_advanced_cfg.
        Missing keys fall back to no-op defaults (constant CFG, no rescale,
        no threshold).
        """
        return {
            "cfg_schedule_type": params.get("cfg_schedule_type", "constant"),
            "cfg_schedule_min": params.get("cfg_schedule_min", 1.0),
            "cfg_schedule_max": params.get("cfg_schedule_max"),
            "cfg_schedule_power": params.get("cfg_schedule_power", 2.0),
            "cfg_rescale_snr_alpha": params.get("cfg_rescale_snr_alpha", 0.0),
            "dynamic_threshold_percentile": params.get("dynamic_threshold_percentile", 0.0),
            "dynamic_threshold_mimic_scale": params.get("dynamic_threshold_mimic_scale", 1.0),
            "developer_mode": params.get("developer_mode", False),
        }

    @staticmethod
    def _anima_encode_nag_neg(params, encode_prompt, text_encoder, qwen3_tokenizer,
                              t5_tokenizer, enc_device, compute_dtype):
        """Encode the NAG-negative prompt using the SAME encoder path as the
        positive / negative prompt. Returns the embeds dict, or None when NAG is
        not active (so the generation path is unchanged by default).

        NAG is active only when nag_enable is set, nag_scale > 1, and a
        NAG-negative prompt string is provided (falls back to the regular
        negative_prompt when nag_negative_prompt is empty, matching the other
        backends' behaviour of guiding away from the negative context).
        """
        from core.inference.nag_dit import nag_active

        nag_enable = params.get("nag_enable", False)
        nag_scale = float(params.get("nag_scale", 1.0) or 1.0)
        nag_negative_prompt = params.get("nag_negative_prompt", "") or ""
        if not nag_negative_prompt:
            nag_negative_prompt = params.get("negative_prompt", "") or ""

        # nag_active gates on enable + scale; require a non-empty negative text too.
        if not (nag_enable and abs(nag_scale - 1.0) > 1e-5 and nag_negative_prompt):
            return None

        neg = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                            nag_negative_prompt, device=enc_device, dtype=compute_dtype)
        # Sanity: only proceed if nag_active agrees (defensive, mirrors reference).
        return neg if nag_active(nag_enable, nag_scale, neg.get("prompt_embeds")) else None

    @staticmethod
    def _anima_build_nag_wrapper(params, transformer, nag_neg_embeds):
        """Build an AnimaNAGWrapper for the conditional pass, or None when NAG
        is inactive (nag_neg_embeds is None). OFF by default."""
        if nag_neg_embeds is None:
            return None
        from core.inference.nag_anima import AnimaNAGWrapper
        nag_scale = float(params.get("nag_scale", 5.0) or 5.0)
        nag_tau = float(params.get("nag_tau", 2.5) or 2.5)
        nag_alpha = float(params.get("nag_alpha", 0.25) or 0.25)
        return AnimaNAGWrapper(
            transformer, nag_neg_embeds,
            nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha,
        )

    @staticmethod
    def _anima_negpip_active(params) -> bool:
        """True when NegPip should auto-activate: the positive OR the negative
        prompt carries a NEGATIVE emphasis weight. OFF by default otherwise, so
        positive-only prompts take the byte-identical default path."""
        from core.inference.negpip_anima import negpip_active
        return negpip_active(params.get("prompt", "") or "",
                             params.get("negative_prompt", "") or "")

    @staticmethod
    def _anima_build_negpip(params, cond_transformer, uncond_transformer,
                            cond_embeds, uncond_embeds, t5_tokenizer,
                            device, compute_dtype):
        """Build the (cond, uncond) NegPip transformer wrappers, or (cond, None).

        ``cond_transformer`` is whatever already drives the COND pass (the raw
        transformer, or an ``AnimaNAGWrapper`` when NAG is active) — the returned
        cond wrapper arms the POSITIVE prompt's signed weights around it, folding
        into NAG when present. ``uncond_transformer`` (raw transformer) is wrapped
        with the NEGATIVE prompt's signed weights (a negative weight there is a
        double-negative that re-affirms). Returns the possibly-wrapped
        (cond, uncond) transformers; wrappers must be ``.restore()``d after use.

        Only called when ``_anima_negpip_active`` is true, so the default path is
        untouched. If neither prompt yields a non-unit weight vector aligned to
        its T5 tokens, the corresponding pass is left as-is.
        """
        from core.inference.negpip_anima import (
            build_anima_negpip_weights, AnimaNegPipWrapper,
        )

        pos_w = build_anima_negpip_weights(
            params.get("prompt", "") or "", cond_embeds["t5_input_ids"],
            t5_tokenizer, device, compute_dtype,
        )
        cond_wrapped = cond_transformer
        if pos_w is not None:
            cond_wrapped = AnimaNegPipWrapper(cond_transformer, pos_w)

        uncond_wrapped = None
        if uncond_embeds is not None:
            neg_w = build_anima_negpip_weights(
                params.get("negative_prompt", "") or "", uncond_embeds["t5_input_ids"],
                t5_tokenizer, device, compute_dtype,
            )
            if neg_w is not None:
                uncond_wrapped = AnimaNegPipWrapper(uncond_transformer, neg_w)

        return cond_wrapped, uncond_wrapped

    def _anima_move(self, component_name: str, target_device: str,
                     quantization: Optional[str] = None):
        """Move a named Anima component to the given device.

        For GPU moves the specialized helpers in core.vram_optimization apply
        optional FP8 quantization to text_encoder / transformer; CPU moves use
        the plain .to('cpu') path. The (possibly quantized) component is
        written back into self.anima_components so subsequent calls see the
        quantized copy.
        """
        from core.vram_optimization import (
            move_anima_text_encoder_to_gpu, move_anima_text_encoder_to_cpu,
            move_anima_transformer_to_gpu, move_anima_transformer_to_cpu,
            move_anima_vae_to_gpu, move_anima_vae_to_cpu,
        )

        comp = self.anima_components.get(component_name)
        if comp is None:
            return comp

        try:
            if component_name == "text_encoder":
                if target_device == "cpu":
                    move_anima_text_encoder_to_cpu(comp)
                else:
                    comp = move_anima_text_encoder_to_gpu(comp, quantization)
                    self.anima_components["text_encoder"] = comp
            elif component_name == "transformer":
                if target_device == "cpu":
                    move_anima_transformer_to_cpu(comp)
                else:
                    comp = move_anima_transformer_to_gpu(comp, quantization)
                    self.anima_components["transformer"] = comp
            elif component_name == "vae":
                if target_device == "cpu":
                    move_anima_vae_to_cpu(comp)
                else:
                    move_anima_vae_to_gpu(comp)
            else:
                # Fallback for unknown components
                if hasattr(comp, "to"):
                    comp.to(target_device)
        except Exception as e:
            print(f"[Anima] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _generate_txt2img_anima(self, params: Dict[str, Any],
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded. Please load an Anima model first.")

        print("[Anima] Starting txt2img generation")
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_txt2img, vae_decode_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        # Seed
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        ancestral_seed = params.get("ancestral_seed", -1)
        if ancestral_seed == -1:
            ancestral_seed = random.randint(0, 2147483647)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        height = int(params.get("height", 512))
        width = int(params.get("width", 512))
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))

        # Optional FP8 quantization (matches FLUX.2 / Z-Image pattern)
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        # Snap to patch_spatial * vae_scale_factor
        snap = transformer.patch_spatial * 8
        height = (height // snap) * snap
        width = (width // snap) * snap

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: text encoding
            if not cpu_text_encoding:
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            # NegPip auto-activates on any negative emphasis weight. When active
            # we encode CLEAN embeddings (skip_emphasis) so the signed V scaling
            # carries all the emphasis; otherwise the default emphasis path runs.
            use_negpip = self._anima_negpip_active(params)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype,
                                  skip_emphasis=use_negpip)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype,
                                       skip_emphasis=use_negpip)
            nag_neg = self._anima_encode_nag_neg(
                params, encode_prompt, text_encoder, qwen3_tokenizer, t5_tokenizer,
                enc_device, compute_dtype,
            )
            if not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                # Move CPU-encoded embeddings to GPU for denoising
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
                if nag_neg is not None:
                    nag_neg = _embeds_to_gpu(nag_neg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 2: denoising
            transformer = self._anima_move("transformer", device, transformer_quantization)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]

            # Optional NAG (Normalized Attention Guidance). OFF by default.
            nag_wrapper = self._anima_build_nag_wrapper(params, transformer, nag_neg)
            # Optional NegPip (signed per-token V scale). OFF by default; folds
            # into NAG on the cond pass and wraps the raw transformer on uncond.
            base_cond = nag_wrapper if nag_wrapper is not None else transformer
            cond_driver = base_cond
            negpip_uncond = None
            if use_negpip:
                cond_driver, negpip_uncond = self._anima_build_negpip(
                    params, base_cond, transformer, cond, uncond,
                    t5_tokenizer, device, compute_dtype,
                )
            # Restore only the NegPip wrappers we actually created (cond_driver
            # differs from base_cond only when a NegPip cond wrapper was built).
            negpip_cond = cond_driver if cond_driver is not base_cond else None
            try:
                latents = sample_txt2img(
                    transformer=transformer, scheduler=scheduler,
                    cond_embeds=cond, uncond_embeds=uncond,
                    height=height, width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator, device=device, dtype=compute_dtype,
                    step_callback=(progress_callback or step_callback),
                    advanced_cfg=self._anima_advanced_cfg(params),
                    spectrum_params=params,
                    nag_transformer=cond_driver if cond_driver is not transformer else None,
                    negpip_uncond_transformer=negpip_uncond,
                )
            finally:
                for w in (negpip_uncond, negpip_cond, nag_wrapper):
                    if w is not None and hasattr(w, "restore"):
                        w.restore()
            if applied_lora_count:
                self._unload_lora_anima()
            self._anima_move("transformer", "cpu")
            del cond, uncond
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: VAE decode
            self._anima_move("vae", device)
            self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
            images = vae_decode_latents(vae, latents)
            del latents
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Anima] txt2img completed")
            return images[0], seed, ancestral_seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Ensure all components are back on CPU even if an error occurred
            for _comp in ("text_encoder", "transformer", "vae"):
                try:
                    self._anima_move(_comp, "cpu")
                except Exception:
                    pass

    def _generate_img2img_anima(self, params: Dict[str, Any], init_image: Image.Image,
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded.")

        print("[Anima] Starting img2img generation")
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_img2img, vae_encode_image, vae_decode_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.7))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        # Resize init image to match desired width/height (snapped)
        width = int(params.get("width", init_image.width))
        height = int(params.get("height", init_image.height))
        snap = transformer.patch_spatial * 8
        width = (width // snap) * snap
        height = (height // snap) * snap
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Encode init image
            self._anima_move("vae", device)
            init_latents = vae_encode_image(vae, init_image, device, compute_dtype)
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Text encoding
            if not cpu_text_encoding:
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            use_negpip = self._anima_negpip_active(params)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype,
                                  skip_emphasis=use_negpip)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype,
                                       skip_emphasis=use_negpip)
            nag_neg = self._anima_encode_nag_neg(
                params, encode_prompt, text_encoder, qwen3_tokenizer, t5_tokenizer,
                enc_device, compute_dtype,
            )
            if not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
                if nag_neg is not None:
                    nag_neg = _embeds_to_gpu(nag_neg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Denoise
            transformer = self._anima_move("transformer", device, transformer_quantization)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]

            # Optional NAG (Normalized Attention Guidance). OFF by default.
            nag_wrapper = self._anima_build_nag_wrapper(params, transformer, nag_neg)
            # Optional NegPip (signed per-token V scale). OFF by default.
            base_cond = nag_wrapper if nag_wrapper is not None else transformer
            cond_driver = base_cond
            negpip_uncond = None
            if use_negpip:
                cond_driver, negpip_uncond = self._anima_build_negpip(
                    params, base_cond, transformer, cond, uncond,
                    t5_tokenizer, device, compute_dtype,
                )
            negpip_cond = cond_driver if cond_driver is not base_cond else None
            try:
                latents = sample_img2img(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents,
                    cond_embeds=cond, uncond_embeds=uncond,
                    num_inference_steps=num_inference_steps,
                    denoising_strength=denoising_strength,
                    guidance_scale=guidance_scale,
                    generator=generator, device=device, dtype=compute_dtype,
                    step_callback=(progress_callback or step_callback),
                    advanced_cfg=self._anima_advanced_cfg(params),
                    spectrum_params=params,
                    nag_transformer=cond_driver if cond_driver is not transformer else None,
                    negpip_uncond_transformer=negpip_uncond,
                )
            finally:
                for w in (negpip_uncond, negpip_cond, nag_wrapper):
                    if w is not None and hasattr(w, "restore"):
                        w.restore()
            if applied_lora_count:
                self._unload_lora_anima()
            self._anima_move("transformer", "cpu")
            del cond, uncond, init_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode
            self._anima_move("vae", device)
            self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
            images = vae_decode_latents(vae, latents)
            del latents
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Anima] img2img completed")
            return images[0], seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            for _comp in ("text_encoder", "transformer", "vae"):
                try:
                    self._anima_move(_comp, "cpu")
                except Exception:
                    pass

    def _generate_inpaint_anima(self, params: Dict[str, Any],
                                 init_image: Image.Image, mask_image: Image.Image,
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded.")

        print("[Anima] Starting inpaint generation")
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_inpaint, vae_encode_image, vae_decode_latents,
            make_mask_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        width = int(params.get("width", init_image.width))
        height = int(params.get("height", init_image.height))
        snap = transformer.patch_spatial * 8
        width = (width // snap) * snap
        height = (height // snap) * snap
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)

        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Encode init image
            self._anima_move("vae", device)
            init_latents = vae_encode_image(vae, init_image, device, compute_dtype)
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mask_latents = make_mask_latents(
                mask_image, init_latents.shape[-2], init_latents.shape[-1],
                device, compute_dtype,
            )

            # Text encoding
            if not cpu_text_encoding:
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            use_negpip = self._anima_negpip_active(params)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype,
                                  skip_emphasis=use_negpip)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype,
                                       skip_emphasis=use_negpip)
            nag_neg = self._anima_encode_nag_neg(
                params, encode_prompt, text_encoder, qwen3_tokenizer, t5_tokenizer,
                enc_device, compute_dtype,
            )
            if not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
                if nag_neg is not None:
                    nag_neg = _embeds_to_gpu(nag_neg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Denoise
            transformer = self._anima_move("transformer", device, transformer_quantization)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]

            # Optional NAG (Normalized Attention Guidance). OFF by default.
            nag_wrapper = self._anima_build_nag_wrapper(params, transformer, nag_neg)
            # Optional NegPip (signed per-token V scale). OFF by default.
            base_cond = nag_wrapper if nag_wrapper is not None else transformer
            cond_driver = base_cond
            negpip_uncond = None
            if use_negpip:
                cond_driver, negpip_uncond = self._anima_build_negpip(
                    params, base_cond, transformer, cond, uncond,
                    t5_tokenizer, device, compute_dtype,
                )
            negpip_cond = cond_driver if cond_driver is not base_cond else None
            try:
                latents = sample_inpaint(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents, mask_latents=mask_latents,
                    cond_embeds=cond, uncond_embeds=uncond,
                    num_inference_steps=num_inference_steps,
                    denoising_strength=denoising_strength,
                    guidance_scale=guidance_scale,
                    generator=generator, device=device, dtype=compute_dtype,
                    step_callback=(progress_callback or step_callback),
                    advanced_cfg=self._anima_advanced_cfg(params),
                    spectrum_params=params,
                    nag_transformer=cond_driver if cond_driver is not transformer else None,
                    negpip_uncond_transformer=negpip_uncond,
                )
            finally:
                for w in (negpip_uncond, negpip_cond, nag_wrapper):
                    if w is not None and hasattr(w, "restore"):
                        w.restore()
            if applied_lora_count:
                self._unload_lora_anima()
            self._anima_move("transformer", "cpu")
            del cond, uncond, init_latents, mask_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode
            self._anima_move("vae", device)
            self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
            images = vae_decode_latents(vae, latents)
            del latents
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Anima] inpaint completed")
            return images[0], seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            for _comp in ("text_encoder", "transformer", "vae"):
                try:
                    self._anima_move(_comp, "cpu")
                except Exception:
                    pass
