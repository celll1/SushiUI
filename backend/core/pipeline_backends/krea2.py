"""Krea 2 generation backend (Krea2Mixin) for PipelineManager.

Single-stream flow-matching MMDiT with a Qwen3-VL text encoder (12-layer
hidden-state stack) and the Qwen-Image VAE (16 latent channels, f8). VRAM staging
mirrors the other large DiT archs: encode text (TE -> GPU -> free to CPU), stage
the ~26GB transformer on GPU for the denoise loop, then VAE-decode. CFG uses the
standard UI ``cfg_scale`` mapped to the Krea guidance convention
(``guidance = cfg_scale - 1``); the distilled/turbo checkpoint disables CFG and
pins a fixed timestep shift.
"""

from typing import Dict, Any, Optional, List
from PIL import Image
import random
import torch

from config.settings import settings

# Latent geometry: Qwen-Image VAE 8x downscale + 2x2 patchify => 16px token grid.
GRID_ALIGN = 16


def _normalize_resolution(width: int, height: int) -> tuple:
    """Round width/height up to a multiple of the 16px token grid."""
    w = ((int(width) + GRID_ALIGN - 1) // GRID_ALIGN) * GRID_ALIGN
    h = ((int(height) + GRID_ALIGN - 1) // GRID_ALIGN) * GRID_ALIGN
    return max(w, GRID_ALIGN), max(h, GRID_ALIGN)


class Krea2Mixin:
    """Krea 2 backend methods for PipelineManager."""

    def _krea2_move(self, component_name: str, target_device: str):
        comp = self.krea2_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[Krea2] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _krea2_apply_attention_backend(self, transformer, params: Dict[str, Any]):
        """Stamp the inference attention backend (native/flash/sage) on the transformer.

        The vendored forward propagates ``_attn_backend`` to every Krea2Attention each
        step; the conduit handles the GQA (48Q/12KV) downgrade rules per backend.
        """
        from core.attention import normalize_backend
        backend = normalize_backend(params.get("attention_type", settings.attention_type))
        transformer._attn_backend = backend
        print(f"[Krea2] Attention backend: {backend} "
              f"(from attention_type={params.get('attention_type')!r})")

    @staticmethod
    def _krea2_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
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

    def _krea2_common_params(self, params: Dict[str, Any], default_w: int, default_h: int):
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        req_w = int(params.get("width", default_w))
        req_h = int(params.get("height", default_h))
        width, height = _normalize_resolution(req_w, req_h)
        if (width, height) != (req_w, req_h):
            print(f"[Krea2] Resolution aligned: {req_w}x{req_h} -> {width}x{height}")

        vae_scale = int(self.krea2_components.get("vae_scale_factor", 8))
        patch_size = int(self.krea2_components.get("patch_size", 2))
        grid_h = height // (vae_scale * patch_size)
        grid_w = width // (vae_scale * patch_size)
        is_distilled = bool(self.krea2_components.get("is_distilled", False))

        # UI cfg_scale -> Krea guidance (cfg_scale - 1). Distilled checkpoint: no CFG.
        cfg_scale = float(params.get("cfg_scale", 4.5))
        guidance = 0.0 if is_distilled else max(cfg_scale - 1.0, 0.0)
        default_steps = 8 if is_distilled else 28

        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", "") or "",
            "num_inference_steps": int(params.get("steps", default_steps)),
            "guidance": guidance,
            "max_sequence_length": int(params.get("krea2_max_seq_len", 512)),
            "width": width,
            "height": height,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "patch_size": patch_size,
            "is_distilled": is_distilled,
            "num_channels_latents": int(self.krea2_components["transformer"].config["in_channels"]) // (patch_size ** 2),
        }

    @torch.no_grad()
    def _krea2_encode(self, prompt, negative_prompt, cfg, device, dtype):
        """Stage the TE to GPU, encode positive (+ negative when CFG on), free TE to CPU."""
        from core.models.krea2.krea2_pipeline_ops import encode_prompt

        select_layers = self.krea2_components["text_encoder_select_layers"]
        max_len = cfg["max_sequence_length"]

        self._krea2_move("text_encoder", device)
        te = self.krea2_components["text_encoder"]
        tok = self.krea2_components["tokenizer"]

        prompt_embeds, prompt_mask = encode_prompt(te, tok, prompt, select_layers, max_len, device)
        neg_embeds = neg_mask = None
        if cfg["guidance"] > 0.0:
            neg_prompt = negative_prompt if (negative_prompt and negative_prompt.strip()) else ""
            neg_embeds, neg_mask = encode_prompt(te, tok, neg_prompt, select_layers, max_len, device)
            neg_embeds = neg_embeds.to(dtype)
        self._krea2_move("text_encoder", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return prompt_embeds.to(dtype), prompt_mask, neg_embeds, neg_mask

    def _krea2_cleanup(self):
        for _c in ("text_encoder", "transformer", "vae"):
            try:
                self._krea2_move(_c, "cpu")
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_txt2img_krea2(self, params: Dict[str, Any],
                                progress_callback=None, step_callback=None) -> tuple:
        if not self.krea2_components:
            raise RuntimeError("Krea 2 components not loaded. Please load a Krea 2 model first.")
        from core.models.krea2.krea2_pipeline_ops import (
            prepare_latents_txt2img, denoise_loop, vae_decode,
        )

        print("[Krea2] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._krea2_common_params(params, 1024, 1024)
        scheduler = self.krea2_components["scheduler"]
        advanced_cfg = self._krea2_advanced_cfg(params)

        try:
            print("[Krea2] Stage 1: Text encoding...")
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._krea2_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg, device, dtype)

            print("[Krea2] Stage 2: Prepare latents...")
            latents = prepare_latents_txt2img(
                cfg["num_channels_latents"], cfg["grid_h"], cfg["grid_w"], cfg["patch_size"],
                dtype=torch.float32, device=device, seed=cfg["seed"])

            print("[Krea2] Stage 3: Denoising...")
            transformer = self._krea2_move("transformer", device)
            self._krea2_apply_attention_backend(transformer, params)
            try:
                latents = denoise_loop(
                    transformer, scheduler, latents, prompt_embeds, prompt_mask,
                    neg_embeds, neg_mask, cfg["guidance"], cfg["num_inference_steps"],
                    cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], cfg["is_distilled"], device,
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                self._krea2_move("transformer", "cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print("[Krea2] Stage 4: VAE decode...")
            self._krea2_move("vae", device)
            self._apply_vae_tiling(self.krea2_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.krea2_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], cfg["patch_size"])
            self._krea2_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Krea2] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Krea2] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._krea2_cleanup()

    def _generate_img2img_krea2(self, params: Dict[str, Any], init_image: Image.Image,
                                progress_callback=None, step_callback=None) -> tuple:
        if not self.krea2_components:
            raise RuntimeError("Krea 2 components not loaded.")
        from core.models.krea2.krea2_pipeline_ops import (
            vae_encode, denoise_loop_img2img, vae_decode,
        )

        print("[Krea2] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._krea2_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.7))
        scheduler = self.krea2_components["scheduler"]
        advanced_cfg = self._krea2_advanced_cfg(params)

        try:
            print("[Krea2] Stage 1: Text encoding...")
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._krea2_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg, device, dtype)

            print("[Krea2] Stage 2: Encoding init image...")
            self._krea2_move("vae", device)
            init_latents = vae_encode(
                self.krea2_components["vae"], init_image, cfg["height"], cfg["width"],
                cfg["patch_size"], device=device, dtype=torch.float32)
            self._krea2_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Krea2] Stage 3: Denoising (SDEdit)...")
            transformer = self._krea2_move("transformer", device)
            self._krea2_apply_attention_backend(transformer, params)
            try:
                latents = denoise_loop_img2img(
                    transformer, scheduler, init_latents, denoising_strength,
                    prompt_embeds, prompt_mask, neg_embeds, neg_mask,
                    cfg["guidance"], cfg["num_inference_steps"],
                    cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], cfg["is_distilled"], device,
                    seed=cfg["seed"], progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                self._krea2_move("transformer", "cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print("[Krea2] Stage 4: VAE decode...")
            self._krea2_move("vae", device)
            self._apply_vae_tiling(self.krea2_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.krea2_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], cfg["patch_size"])
            self._krea2_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Krea2] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Krea2] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._krea2_cleanup()

    def _generate_inpaint_krea2(self, params: Dict[str, Any],
                                init_image: Image.Image, mask_image: Image.Image,
                                progress_callback=None, step_callback=None) -> tuple:
        if not self.krea2_components:
            raise RuntimeError("Krea 2 components not loaded.")
        from core.models.krea2.krea2_pipeline_ops import (
            vae_encode, denoise_loop_inpaint, vae_decode, prepare_mask_latent,
        )

        print("[Krea2] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._krea2_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        scheduler = self.krea2_components["scheduler"]
        advanced_cfg = self._krea2_advanced_cfg(params)

        width, height = cfg["width"], cfg["height"]
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)
        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        try:
            print("[Krea2] Stage 1: Text encoding...")
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._krea2_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg, device, dtype)

            print("[Krea2] Stage 2: Encoding init image + mask...")
            self._krea2_move("vae", device)
            init_latents = vae_encode(
                self.krea2_components["vae"], init_image, height, width,
                cfg["patch_size"], device=device, dtype=torch.float32)
            self._krea2_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask_latent = prepare_mask_latent(
                mask_image, cfg["grid_h"], cfg["grid_w"], device=device, dtype=torch.float32)

            print("[Krea2] Stage 3: Denoising (repaint)...")
            transformer = self._krea2_move("transformer", device)
            self._krea2_apply_attention_backend(transformer, params)
            try:
                latents = denoise_loop_inpaint(
                    transformer, scheduler, init_latents, mask_latent, denoising_strength,
                    prompt_embeds, prompt_mask, neg_embeds, neg_mask,
                    cfg["guidance"], cfg["num_inference_steps"],
                    cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], cfg["is_distilled"], device,
                    seed=cfg["seed"], progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                self._krea2_move("transformer", "cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print("[Krea2] Stage 4: VAE decode...")
            self._krea2_move("vae", device)
            self._apply_vae_tiling(self.krea2_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.krea2_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], cfg["patch_size"])
            self._krea2_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Krea2] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Krea2] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._krea2_cleanup()
