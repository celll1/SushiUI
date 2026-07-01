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

class Ideogram4Mixin:
    """Ideogram4Mixin: ideogram4 backend methods extracted verbatim from pipeline.py."""

    def _load_lora_ideogram4(self, lora_configs: List[Dict]) -> int:
        """Wrap Ideogram 4 transformer Linear/Fp8Linear modules with LoRA adapters.

        Applies the conditional-branch LoRA to `transformer`; if the checkpoint
        also carries unconditional-branch keys (lora_uncond_*) and the
        unconditional transformer is loaded, those are applied to it too.
        Must be called after the transformer(s) are staged on GPU.
        """
        from core.models.ideogram4.ideogram4_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs or not self.ideogram4_components:
            return 0

        transformer = self.ideogram4_components["transformer"]
        uncond = self.ideogram4_components.get("unconditional_transformer")
        if not hasattr(self, "_ideogram4_lora_orig"):
            self._ideogram4_lora_orig: Dict[str, torch.nn.Module] = {}
            self._ideogram4_lora_keys: set = set()
            self._ideogram4_lora_orig_uncond: Dict[str, torch.nn.Module] = {}
            self._ideogram4_lora_keys_uncond: set = set()

        total = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[Ideogram4 LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw, branch="cond")
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._ideogram4_lora_orig, self._ideogram4_lora_keys,
                )
                print(f"[Ideogram4 LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} cond_modules={len(grouped)} wrapped={applied} strength={strength}")
                total += applied

                if uncond is not None:
                    grouped_u = normalise_lora_state_dict(raw, branch="uncond")
                    if grouped_u:
                        applied_u = apply_lora_group(
                            uncond, grouped_u, strength,
                            self._ideogram4_lora_orig_uncond, self._ideogram4_lora_keys_uncond,
                        )
                        print(f"[Ideogram4 LoRA]   uncond wrapped {applied_u} module(s)")
                        total += applied_u
            except Exception as e:
                print(f"[Ideogram4 LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total

    def _unload_lora_ideogram4(self) -> int:
        """Restore every Ideogram 4 transformer Linear to its pre-LoRA original."""
        from core.models.ideogram4.ideogram4_lora import restore_originals
        if not self.ideogram4_components:
            return 0
        restored = 0
        if getattr(self, "_ideogram4_lora_keys", None):
            restored += restore_originals(
                self.ideogram4_components["transformer"],
                self._ideogram4_lora_orig, self._ideogram4_lora_keys,
            )
        uncond = self.ideogram4_components.get("unconditional_transformer")
        if uncond is not None and getattr(self, "_ideogram4_lora_keys_uncond", None):
            restored += restore_originals(
                uncond, self._ideogram4_lora_orig_uncond, self._ideogram4_lora_keys_uncond,
            )
        if restored:
            print(f"[Ideogram4 LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    def _ideogram4_move(self, component_name: str, target_device: str):
        """Move a named Ideogram 4 component to the target device.

        Weights are already weight-only FP8 (Fp8Linear buffers move with .to());
        plain .to() is sufficient — no extra quantization step is applied.
        """
        comp = self.ideogram4_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[Ideogram4] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    @staticmethod
    def _ideogram4_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs for Ideogram 4 generation.

        Consumed by ideogram4_pipeline_ops._blend_guidance (standard CFG blend
        plus optional schedule / SNR-rescale / dynamic thresholding).
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

    def _ideogram4_common_params(self, params: Dict[str, Any], default_w: int, default_h: int):
        """Resolve shared Ideogram 4 generation parameters."""
        from core.models.ideogram4.ideogram4_resolution import normalize_resolution, latent_grid

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        req_width = int(params.get("width", default_w))
        req_height = int(params.get("height", default_h))
        width, height = normalize_resolution(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Ideogram4] Resolution aligned: {req_width}x{req_height} -> {width}x{height}")
        grid_h, grid_w = latent_grid(width, height)

        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "num_inference_steps": int(params.get("steps", 28)),
            "guidance_scale": float(params.get("cfg_scale", 7.0)),
            "mu": float(params.get("ideogram4_mu", 0.0)),
            "std": float(params.get("ideogram4_std", 1.5)),
            "max_sequence_length": int(params.get("ideogram4_max_seq_len", 512)),
            "width": width,
            "height": height,
            "grid_h": grid_h,
            "grid_w": grid_w,
        }

    @torch.no_grad()
    def _ideogram4_encode(self, prompt, grid_h, grid_w, max_sequence_length, device, dtype):
        """Stage the text encoder to GPU, encode the prompt, then free it back to CPU."""
        from core.models.ideogram4.ideogram4_pipeline_ops import encode_prompt

        self._ideogram4_move("text_encoder", device)
        text_encoder = self.ideogram4_components["text_encoder"]
        tokenizer = self.ideogram4_components["tokenizer"]
        cond = encode_prompt(
            text_encoder, tokenizer, prompt,
            grid_h=grid_h, grid_w=grid_w,
            max_sequence_length=max_sequence_length, device=device,
        )
        self._ideogram4_move("text_encoder", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Cast conditioning to the transformer compute dtype (halves memory; matches RMSNorm dtype).
        cond["llm_features"] = cond["llm_features"].to(dtype)
        cond["neg_llm_features"] = cond["neg_llm_features"].to(dtype)
        return cond

    def _ideogram4_setup_block_swap(self, transformer, blocks_to_swap: int,
                                    use_pinned_memory: bool, device: str):
        """Attach a block-swap offloader to one Ideogram 4 transformer.

        The transformer starts on CPU; the offloader keeps the first
        (num_layers - blocks_to_swap) blocks resident on GPU and streams the rest
        per forward. Non-block (auxiliary) modules are moved to GPU here since the
        shared offloader only auto-moves Z-Image-named aux modules.
        """
        from core.memory_management import create_block_offloader_for_model

        # Auxiliary modules (everything except the swappable block list) stay on GPU.
        for name, child in transformer.named_children():
            if name != "layers":
                child.to(device)
        for p in transformer.parameters(recurse=False):
            if p.device.type != "cuda":
                p.data = p.data.to(device)
        for b in transformer.buffers(recurse=False):
            if b.device.type != "cuda":
                b.data = b.data.to(device)

        offloader = create_block_offloader_for_model(
            transformer=transformer,
            blocks_to_swap=blocks_to_swap,
            device=torch.device(device),
            target_dtype=torch.bfloat16,
            use_pinned_memory=use_pinned_memory,
        )
        transformer._block_offloader = offloader
        offloader.prepare_block_devices_before_forward()
        return offloader

    def _ideogram4_stage_transformers(self, device: str, params: Optional[Dict[str, Any]] = None):
        """Place both transformers on GPU for the denoise loop.

        With block swap enabled, each transformer streams its blocks (per-model
        offloader) instead of being fully resident, roughly halving the resident
        footprint of the two 9.3B FP8 transformers at the cost of CPU<->GPU traffic.
        """
        params = params or {}
        enable_block_swap = bool(params.get("enable_block_swap", False))
        num_layers = len(self.ideogram4_components["transformer"].layers)
        blocks_to_swap = int(params.get("blocks_to_swap", 20))
        blocks_to_swap = max(0, min(blocks_to_swap, num_layers - 1))
        use_pinned_memory = bool(params.get("use_pinned_memory", False))

        self._ideogram4_offloaders = []
        if enable_block_swap and blocks_to_swap > 0:
            print(f"[Ideogram4] Block swap enabled: {blocks_to_swap}/{num_layers} blocks per transformer "
                  f"(pinned_memory={use_pinned_memory})")
            for comp_name in ("transformer", "unconditional_transformer"):
                t = self.ideogram4_components[comp_name]
                off = self._ideogram4_setup_block_swap(t, blocks_to_swap, use_pinned_memory, device)
                self._ideogram4_offloaders.append((comp_name, off))
        else:
            self._ideogram4_move("transformer", device)
            self._ideogram4_move("unconditional_transformer", device)
        return (
            self.ideogram4_components["transformer"],
            self.ideogram4_components["unconditional_transformer"],
        )

    def _ideogram4_unstage_transformers(self):
        # Tear down any block-swap offloaders, then return both transformers to CPU.
        offloaders = getattr(self, "_ideogram4_offloaders", [])
        for comp_name, off in offloaders:
            t = self.ideogram4_components.get(comp_name)
            if t is not None and hasattr(t, "_block_offloader"):
                try:
                    delattr(t, "_block_offloader")
                except Exception:
                    pass
            cleanup = getattr(off, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        self._ideogram4_offloaders = []
        self._ideogram4_move("transformer", "cpu")
        self._ideogram4_move("unconditional_transformer", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ideogram4_cleanup(self):
        # Strip any leftover block-swap offloaders (e.g. if setup raised mid-way).
        for _comp in ("transformer", "unconditional_transformer"):
            t = (self.ideogram4_components or {}).get(_comp)
            if t is not None and hasattr(t, "_block_offloader"):
                try:
                    delattr(t, "_block_offloader")
                except Exception:
                    pass
        self._ideogram4_offloaders = []
        for _comp in ("text_encoder", "transformer", "unconditional_transformer", "vae"):
            try:
                self._ideogram4_move(_comp, "cpu")
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_txt2img_ideogram4(self, params: Dict[str, Any],
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded. Please load an Ideogram 4 model first.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            prepare_latents, denoise_loop, vae_decode,
        )

        print("[Ideogram4] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, 1024, 1024)
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
            )

            print("[Ideogram4] Stage 2: Prepare latents...")
            latents = prepare_latents(
                cfg["grid_h"], cfg["grid_w"], dtype=torch.float32, device=device, seed=cfg["seed"],
            )

            print("[Ideogram4] Stage 3: Denoising (dual-branch)...")
            transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            try:
                latents = denoise_loop(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, latents=latents, cond=cond,
                    guidance_scale=cfg["guidance_scale"], num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=cfg["height"], width=cfg["width"],
                    mu=cfg["mu"], std=cfg["std"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                if applied_lora:
                    self._unload_lora_ideogram4()
                self._ideogram4_unstage_transformers()
            del cond

            print("[Ideogram4] Stage 4: VAE decode...")
            self._ideogram4_move("vae", device)
            self._apply_vae_tiling(self.ideogram4_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"])
            del latents
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup()

    def _generate_img2img_ideogram4(self, params: Dict[str, Any], init_image: Image.Image,
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            vae_encode, denoise_loop_img2img, vae_decode,
        )

        print("[Ideogram4] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.7))
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
            )

            print("[Ideogram4] Stage 2: Encoding init image...")
            self._ideogram4_move("vae", device)
            init_latents = vae_encode(
                self.ideogram4_components["vae"], init_image, cfg["height"], cfg["width"],
                device=device, dtype=torch.float32,
            )
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] Stage 3: Denoising (SDEdit)...")
            transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            try:
                latents = denoise_loop_img2img(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, init_latents=init_latents, denoising_strength=denoising_strength,
                    cond=cond, guidance_scale=cfg["guidance_scale"],
                    num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=cfg["height"], width=cfg["width"],
                    mu=cfg["mu"], std=cfg["std"], seed=cfg["seed"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                if applied_lora:
                    self._unload_lora_ideogram4()
                self._ideogram4_unstage_transformers()
            del cond, init_latents

            print("[Ideogram4] Stage 4: VAE decode...")
            self._ideogram4_move("vae", device)
            self._apply_vae_tiling(self.ideogram4_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"])
            del latents
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup()

    def _generate_inpaint_ideogram4(self, params: Dict[str, Any],
                                    init_image: Image.Image, mask_image: Image.Image,
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            vae_encode, denoise_loop_inpaint, vae_decode, prepare_mask_latent,
        )

        print("[Ideogram4] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        width, height = cfg["width"], cfg["height"]
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)
        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
            )

            print("[Ideogram4] Stage 2: Encoding init image + mask...")
            self._ideogram4_move("vae", device)
            init_latents = vae_encode(
                self.ideogram4_components["vae"], init_image, height, width,
                device=device, dtype=torch.float32,
            )
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask_latent = prepare_mask_latent(
                mask_image, cfg["grid_h"], cfg["grid_w"], device=device, dtype=torch.float32,
            )

            print("[Ideogram4] Stage 3: Denoising (repaint)...")
            transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            try:
                latents = denoise_loop_inpaint(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, init_latents=init_latents, mask_latent=mask_latent,
                    denoising_strength=denoising_strength, cond=cond,
                    guidance_scale=cfg["guidance_scale"], num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=height, width=width,
                    mu=cfg["mu"], std=cfg["std"], seed=cfg["seed"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                if applied_lora:
                    self._unload_lora_ideogram4()
                self._ideogram4_unstage_transformers()
            del cond, init_latents, mask_latent

            print("[Ideogram4] Stage 4: VAE decode...")
            self._ideogram4_move("vae", device)
            self._apply_vae_tiling(self.ideogram4_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"])
            del latents
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup()
