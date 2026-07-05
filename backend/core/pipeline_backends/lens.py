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

class LensMixin:
    """LensMixin: lens backend methods extracted verbatim from pipeline.py."""

    def _load_lora_lens(self, lora_configs: List[Dict]) -> int:
        """Wrap target Linear modules of the Lens transformer with LoRA adapters.

        Must be called after the transformer is on GPU (and optionally quantised).
        Supports stacking multiple LoRAs on the same module.
        """
        from core.models.lens.lens_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs:
            return 0
        if not self.lens_components:
            print("[Lens LoRA] WARNING: components not loaded")
            return 0

        transformer = self.lens_components["transformer"]
        if not hasattr(self, "_lens_lora_original_modules"):
            self._lens_lora_original_modules: Dict[str, torch.nn.Linear] = {}
            self._lens_lora_wrapped_keys: set = set()

        total_applied = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength  = float(cfg.get("strength", 1.0))
            resolved  = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[Lens LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped  = normalise_lora_state_dict(raw)
                print(f"[Lens LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} keys={len(raw)} matched_modules={len(grouped)} "
                      f"strength={strength}")
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._lens_lora_original_modules, self._lens_lora_wrapped_keys,
                )
                print(f"[Lens LoRA]   wrapped {applied} module(s)")
                total_applied += applied
            except Exception as e:
                print(f"[Lens LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total_applied

    def _unload_lora_lens(self) -> int:
        """Restore every Lens transformer Linear to its pre-LoRA original."""
        from core.models.lens.lens_lora import restore_originals
        if not getattr(self, "_lens_lora_wrapped_keys", None):
            return 0
        if not self.lens_components:
            return 0
        transformer = self.lens_components["transformer"]
        restored = restore_originals(
            transformer, self._lens_lora_original_modules, self._lens_lora_wrapped_keys,
        )
        print(f"[Lens LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    @staticmethod
    def _lens_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs for Lens generation.

        Returns a dict consumed by lens_pipeline_ops._apply_advanced_cfg_lens.
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
    def _lens_encode_nag(params: Dict[str, Any], text_encoder, tokenizer,
                         negative_prompt: str, enc_device, dtype,
                         max_sequence_length: int,
                         skip_emphasis: bool = False) -> Optional[Dict[str, Any]]:
        """Encode the NAG-negative prompt and build nag_params for the denoise loop.

        Returns None when NAG is inactive (nag_enable off, or nag_scale<=1), keeping the
        default generation path byte-identical. The nag-negative prompt falls back to the
        CFG negative prompt when unset (matching the FLUX.2 backend).
        """
        nag_enable = bool(params.get("nag_enable", False))
        nag_scale = float(params.get("nag_scale", 5.0))
        if not nag_enable or nag_scale <= 1.0:
            return None
        from core.models.lens.lens_pipeline_ops import encode_nag_negative

        nag_neg_prompt = params.get("nag_negative_prompt", "") or negative_prompt or ""
        nag_features, nag_mask = encode_nag_negative(
            text_encoder, tokenizer, nag_neg_prompt,
            device=enc_device, dtype=dtype, max_length=max_sequence_length,
            skip_emphasis=skip_emphasis,
        )
        return {
            "nag_features": nag_features,
            "nag_mask": nag_mask,
            "nag_scale": nag_scale,
            "nag_tau": float(params.get("nag_tau", 2.5)),
            "nag_alpha": float(params.get("nag_alpha", 0.25)),
        }

    @staticmethod
    def _lens_negpip_params(prompt: str, negative_prompt: str,
                            nag_negative_prompt: str, max_sequence_length: int,
                            ) -> Optional[Dict[str, Any]]:
        """Auto-activate NegPip when any prompt carries a negative emphasis weight.

        Returns None (default path byte-identical) unless the positive, negative, or
        nag-negative prompt contains a negative weight (e.g. "(worst quality:-1)").
        When active, the raw (emphasis-bearing) prompt strings are carried through to
        the denoise loop, which builds the per-context signed V weights there.
        """
        from core.prompts.prompt_parser import prompt_has_negative_weight

        candidates = [prompt, negative_prompt, nag_negative_prompt]
        if not any(prompt_has_negative_weight(p or "") for p in candidates):
            return None
        return {
            "prompt": prompt or "",
            "negative_prompt": negative_prompt or "",
            "nag_negative_prompt": nag_negative_prompt or negative_prompt or "",
            "max_length": max_sequence_length,
        }

    def _reload_lens_text_encoder(self) -> None:
        """Reload the Lens text encoder from disk (~4 s).

        Called lazily at the start of each generation when the text encoder has
        been freed after the previous encoding stage to reclaim ~9.7 GB of mxfp4
        CUDA memory.
        """
        from core.models.lens.lens_loader import reload_lens_text_encoder
        model_path = self.lens_components.get("base_dir") or (self.current_model_info or {}).get("source", "")
        transformer = self.lens_components.get("transformer")
        selected_layers = (
            tuple(transformer.config.selected_layer_index)
            if transformer is not None else None
        )
        te = reload_lens_text_encoder(
            model_path,
            torch_dtype=torch.bfloat16,
            selected_layers=selected_layers,
        )
        self.lens_components["text_encoder"] = te

    def _lens_move(self, component_name: str, target_device: str,
                   quantization: Optional[str] = None):
        """Move a Lens component to the target device.

        GPU moves delegate to specialized helpers in core.vram_optimization
        that apply optional FP8 quantization.  The (possibly quantized)
        component is written back into self.lens_components.
        """
        from core.vram_optimization import (
            move_lens_text_encoder_to_gpu, move_lens_text_encoder_to_cpu,
            move_lens_transformer_to_gpu, move_lens_transformer_to_cpu,
            move_lens_vae_to_gpu, move_lens_vae_to_cpu,
        )

        comp = self.lens_components.get(component_name)
        if comp is None:
            return comp

        try:
            if component_name == "text_encoder":
                if target_device == "cpu":
                    move_lens_text_encoder_to_cpu(comp)
                else:
                    comp = move_lens_text_encoder_to_gpu(comp, quantization)
                    self.lens_components["text_encoder"] = comp
            elif component_name == "transformer":
                if target_device == "cpu":
                    move_lens_transformer_to_cpu(comp)
                else:
                    comp = move_lens_transformer_to_gpu(comp, quantization)
                    self.lens_components["transformer"] = comp
            elif component_name == "vae":
                if target_device == "cpu":
                    move_lens_vae_to_cpu(comp)
                else:
                    move_lens_vae_to_gpu(comp)
        except Exception as e:
            print(f"[Lens] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _lens_setup_block_swap(self, transformer, blocks_to_swap: int,
                               use_pinned_memory: bool, device: str,
                               h2d_only: bool = False, ring_size: int = 2):
        """Attach a block-swap offloader to the Lens transformer.

        The offloader keeps the first (num_layers - blocks_to_swap) blocks
        resident on GPU and streams the rest per forward. Non-block (auxiliary)
        modules are moved to GPU here since the shared offloader only auto-moves
        Z-Image-named aux modules.
        """
        from core.memory_management import create_block_offloader_for_model

        # Auxiliary modules (everything except the swappable block list) stay on GPU.
        for name, child in transformer.named_children():
            if name != "transformer_blocks":
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
            h2d_only=h2d_only,
            ring_size=ring_size,
            block_list=transformer.transformer_blocks,
        )
        transformer._block_offloader = offloader
        offloader.prepare_block_devices_before_forward()
        return offloader

    def _lens_set_attention_backend(self, transformer, params: Dict[str, Any]) -> str:
        """Stamp the inference attention backend on every LensJointAttention module.

        Reads the canonical inference key ``attention_type`` (values
        ``normal|sage|flash``; falls back to the app-wide default), normalizes it
        (``normal``->``native``; ``sla`` passthrough preserved), and sets
        ``_attention_backend`` on each ``LensJointAttention`` via a class-name scan
        that mirrors the trainer's ``_setup_attention_backend_lens``. The vendor
        forward reads this attr and routes through ``dispatch_attention``. The
        conduit itself handles the masked case by auto-downgrading mask-incapable
        kernels (flash/sage) to native.
        """
        from core.attention import normalize_backend

        attention_type = params.get("attention_type", settings.attention_type)
        backend = normalize_backend(attention_type)
        n = 0
        for m in transformer.modules():
            if type(m).__name__ == "LensJointAttention":
                m._attention_backend = backend
                n += 1
        print(f"[Lens] Attention backend '{backend}' set on {n} module(s)")
        return backend

    def _lens_stage_transformer(self, params: Dict[str, Any], device: str,
                                transformer_quantization: Optional[str]):
        """Place the Lens transformer on GPU for denoising.

        When block swap is enabled, the transformer streams its blocks (per-model
        offloader) instead of being fully resident. Otherwise the whole
        transformer is moved to GPU (default path, unchanged).
        Returns the (possibly quantized) transformer from self.lens_components.
        """
        enable_block_swap = bool(params.get("enable_block_swap", False))
        transformer = self.lens_components["transformer"]
        num_layers = len(transformer.transformer_blocks)
        blocks_to_swap = int(params.get("blocks_to_swap", 20))
        blocks_to_swap = max(0, min(blocks_to_swap, num_layers - 1))
        use_pinned_memory = bool(params.get("use_pinned_memory", False))
        h2d_only = bool(params.get("block_swap_h2d_only", False))
        ring_size = int(params.get("block_swap_ring_size", 2))

        self._lens_offloader = None
        if enable_block_swap and blocks_to_swap > 0:
            print(f"[Lens] Block swap enabled: {blocks_to_swap}/{num_layers} blocks "
                  f"(pinned_memory={use_pinned_memory}, h2d_only={h2d_only}, ring_size={ring_size})")
            # Optional quantization is applied in place; the transformer stays on CPU
            # and only its aux modules + resident blocks are staged to GPU by the offloader.
            transformer = self._lens_move("transformer", device, transformer_quantization)
            transformer = self.lens_components["transformer"]
            self._lens_offloader = self._lens_setup_block_swap(
                transformer, blocks_to_swap, use_pinned_memory, device,
                h2d_only=h2d_only, ring_size=ring_size,
            )
        else:
            transformer = self._lens_move("transformer", device, transformer_quantization)
            transformer = self.lens_components["transformer"]
        return transformer

    def _lens_unstage_transformer(self):
        """Tear down any block-swap offloader, then return the transformer to CPU."""
        transformer = (self.lens_components or {}).get("transformer")
        offloader = getattr(self, "_lens_offloader", None)
        if transformer is not None and hasattr(transformer, "_block_offloader"):
            try:
                delattr(transformer, "_block_offloader")
            except Exception:
                pass
        if offloader is not None:
            cleanup = getattr(offloader, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        self._lens_offloader = None
        self._lens_move("transformer", "cpu")

    def _generate_txt2img_lens(self, params: Dict[str, Any],
                                progress_callback=None, step_callback=None,
                                ) -> tuple:
        if not self.lens_components:
            raise RuntimeError("Lens components not loaded. Please load a Lens model first.")

        from core.models.lens.lens_pipeline_ops import (
            encode_prompt, prepare_latents, denoise_loop, vae_decode,
        )
        from core.models.lens.lens_resolution import align_to_grid

        print("[Lens] Starting txt2img generation")

        device = self.device
        dtype = torch.bfloat16

        # Lazy reload: text encoder is freed after each generation to reclaim
        # the ~9.7 GB of mxfp4 CUDA memory.  Reload it here before encoding.
        if self.lens_components.get("text_encoder") is None:
            self._reload_lens_text_encoder()

        transformer = self.lens_components["transformer"]
        text_encoder = self.lens_components["text_encoder"]
        tokenizer = self.lens_components["tokenizer"]
        vae = self.lens_components["vae"]
        scheduler = self.lens_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")
        max_sequence_length = 512

        req_width = int(params.get("width", 1024))
        req_height = int(params.get("height", 1024))
        width, height = align_to_grid(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Lens] Resolution aligned: {req_width}×{req_height} → {width}×{height}")

        latent_h = height // 16
        latent_w = width // 16

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: Text encoding
            print("[Lens] Stage 1: Text encoding...")
            if not cpu_text_encoding:
                text_encoder = self._lens_move("text_encoder", device, text_encoder_quantization)
            negpip_params = self._lens_negpip_params(
                prompt, negative_prompt,
                params.get("nag_negative_prompt", "") or "", max_sequence_length,
            )
            encoder_features, encoder_mask = encode_prompt(
                text_encoder, tokenizer, prompt, negative_prompt,
                device=enc_device, dtype=dtype, max_length=max_sequence_length,
                skip_emphasis=negpip_params is not None,
            )
            nag_params = self._lens_encode_nag(
                params, text_encoder, tokenizer, negative_prompt,
                enc_device, dtype, max_sequence_length,
                skip_emphasis=negpip_params is not None,
            )
            if not cpu_text_encoding:
                self._lens_move("text_encoder", "cpu")
            if cpu_text_encoding:
                encoder_features = [f.to(device) for f in encoder_features]
                encoder_mask = encoder_mask.to(device)
                if nag_params is not None:
                    nag_params["nag_features"] = [f.to(device) for f in nag_params["nag_features"]]
                    nag_params["nag_mask"] = nag_params["nag_mask"].to(device)

            # Free mxfp4 CUDA buffers (~9.7 GB) — not needed during denoising.
            # Will be reloaded lazily at the start of the next generation.
            import gc as _gc
            self.lens_components["text_encoder"] = None
            text_encoder = None
            _gc.collect()
            torch.cuda.empty_cache()

            # Stage 2: Prepare latents
            latents = prepare_latents(height, width, dtype=dtype, device=device, seed=seed)

            # Stage 3: Denoising
            print("[Lens] Stage 3: Denoising...")
            transformer = self._lens_stage_transformer(params, device, transformer_quantization)
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_lens(lora_configs) if lora_configs else 0
            transformer = self.lens_components["transformer"]
            self._lens_set_attention_backend(transformer, params)
            try:
                latents = denoise_loop(
                    transformer=transformer, scheduler=scheduler,
                    latents=latents, encoder_features=encoder_features, encoder_mask=encoder_mask,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    latent_h=latent_h, latent_w=latent_w,
                    progress_callback=progress_callback,
                    advanced_cfg=self._lens_advanced_cfg(params),
                    spectrum_params=params,
                    nag_params=nag_params,
                    negpip_params=negpip_params,
                    tokenizer=tokenizer,
                )
            finally:
                if applied_lora_count:
                    self._unload_lora_lens()
            self._lens_unstage_transformer()
            del encoder_features, encoder_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 4: VAE decode
            print("[Lens] Stage 4: VAE decode...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            self._apply_vae_tiling(vae_gpu, getattr(self, "_vae_tiling", False))
            image = vae_decode(vae_gpu, latents, latent_h, latent_w)
            del latents
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Lens] txt2img completed")
            return image, seed, 0

        except Exception as e:
            print(f"[Lens] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Always free text encoder CUDA buffers on exit (normal or exception).
            # Next generation will reload it lazily before encoding.
            if self.lens_components.get("text_encoder") is not None:
                import gc as _gc
                self.lens_components["text_encoder"] = None
                _gc.collect()
            # Strip any leftover block-swap offloader (e.g. if setup/denoise raised mid-way).
            _t = (self.lens_components or {}).get("transformer")
            if _t is not None and hasattr(_t, "_block_offloader"):
                try:
                    delattr(_t, "_block_offloader")
                except Exception:
                    pass
            self._lens_offloader = None
            for _comp in ("transformer", "vae"):
                try:
                    self._lens_move(_comp, "cpu")
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_img2img_lens(self, params: Dict[str, Any], init_image: Image.Image,
                                progress_callback=None, step_callback=None,
                                ) -> tuple:
        if not self.lens_components:
            raise RuntimeError("Lens components not loaded.")

        from core.models.lens.lens_pipeline_ops import (
            encode_prompt, vae_encode, denoise_loop_img2img, vae_decode,
        )
        from core.models.lens.lens_resolution import align_to_grid

        print("[Lens] Starting img2img generation")

        device = self.device
        dtype = torch.bfloat16

        if self.lens_components.get("text_encoder") is None:
            self._reload_lens_text_encoder()

        transformer = self.lens_components["transformer"]
        text_encoder = self.lens_components["text_encoder"]
        tokenizer = self.lens_components["tokenizer"]
        vae = self.lens_components["vae"]
        scheduler = self.lens_components["scheduler"]

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
        max_sequence_length = 512

        req_width = int(params.get("width", init_image.width))
        req_height = int(params.get("height", init_image.height))
        width, height = align_to_grid(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Lens] Resolution aligned: {req_width}×{req_height} → {width}×{height}")
        latent_h = height // 16
        latent_w = width // 16

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: Text encoding
            print("[Lens] Stage 1: Text encoding...")
            if not cpu_text_encoding:
                text_encoder = self._lens_move("text_encoder", device, text_encoder_quantization)
            negpip_params = self._lens_negpip_params(
                prompt, negative_prompt,
                params.get("nag_negative_prompt", "") or "", max_sequence_length,
            )
            encoder_features, encoder_mask = encode_prompt(
                text_encoder, tokenizer, prompt, negative_prompt,
                device=enc_device, dtype=dtype, max_length=max_sequence_length,
                skip_emphasis=negpip_params is not None,
            )
            nag_params = self._lens_encode_nag(
                params, text_encoder, tokenizer, negative_prompt,
                enc_device, dtype, max_sequence_length,
                skip_emphasis=negpip_params is not None,
            )
            if not cpu_text_encoding:
                self._lens_move("text_encoder", "cpu")
            if cpu_text_encoding:
                encoder_features = [f.to(device) for f in encoder_features]
                encoder_mask = encoder_mask.to(device)
                if nag_params is not None:
                    nag_params["nag_features"] = [f.to(device) for f in nag_params["nag_features"]]
                    nag_params["nag_mask"] = nag_params["nag_mask"].to(device)

            # Free mxfp4 CUDA buffers (~9.7 GB) — not needed during denoising.
            import gc as _gc
            self.lens_components["text_encoder"] = None
            text_encoder = None
            _gc.collect()
            torch.cuda.empty_cache()

            # Stage 2: Encode init image
            print("[Lens] Stage 2: Encoding init image...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            init_latents = vae_encode(vae_gpu, init_image, height, width, device=device, dtype=dtype)
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: Denoising (SDEdit)
            print("[Lens] Stage 3: Denoising...")
            transformer = self._lens_stage_transformer(params, device, transformer_quantization)
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_lens(lora_configs) if lora_configs else 0
            transformer = self.lens_components["transformer"]
            self._lens_set_attention_backend(transformer, params)
            try:
                latents = denoise_loop_img2img(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents, denoising_strength=denoising_strength,
                    encoder_features=encoder_features, encoder_mask=encoder_mask,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    latent_h=latent_h, latent_w=latent_w, seed=seed,
                    progress_callback=progress_callback,
                    advanced_cfg=self._lens_advanced_cfg(params),
                    spectrum_params=params,
                    nag_params=nag_params,
                    negpip_params=negpip_params,
                    tokenizer=tokenizer,
                )
            finally:
                if applied_lora_count:
                    self._unload_lora_lens()
            self._lens_unstage_transformer()
            del encoder_features, encoder_mask, init_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 4: VAE decode
            print("[Lens] Stage 4: VAE decode...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            self._apply_vae_tiling(vae_gpu, getattr(self, "_vae_tiling", False))
            image = vae_decode(vae_gpu, latents, latent_h, latent_w)
            del latents
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Lens] img2img completed")
            return image, seed, 0

        except Exception as e:
            print(f"[Lens] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            if self.lens_components.get("text_encoder") is not None:
                import gc as _gc
                self.lens_components["text_encoder"] = None
                _gc.collect()
            # Strip any leftover block-swap offloader (e.g. if setup/denoise raised mid-way).
            _t = (self.lens_components or {}).get("transformer")
            if _t is not None and hasattr(_t, "_block_offloader"):
                try:
                    delattr(_t, "_block_offloader")
                except Exception:
                    pass
            self._lens_offloader = None
            for _comp in ("transformer", "vae"):
                try:
                    self._lens_move(_comp, "cpu")
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_inpaint_lens(self, params: Dict[str, Any],
                                init_image: Image.Image, mask_image: Image.Image,
                                progress_callback=None, step_callback=None,
                                ) -> tuple:
        if not self.lens_components:
            raise RuntimeError("Lens components not loaded.")

        from core.models.lens.lens_pipeline_ops import (
            encode_prompt, vae_encode, denoise_loop_inpaint, vae_decode, prepare_mask_latent,
        )
        from core.models.lens.lens_resolution import align_to_grid

        print("[Lens] Starting inpaint generation (repaint)")

        device = self.device
        dtype = torch.bfloat16

        if self.lens_components.get("text_encoder") is None:
            self._reload_lens_text_encoder()

        transformer = self.lens_components["transformer"]
        text_encoder = self.lens_components["text_encoder"]
        tokenizer = self.lens_components["tokenizer"]
        vae = self.lens_components["vae"]
        scheduler = self.lens_components["scheduler"]

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
        max_sequence_length = 512

        req_width = int(params.get("width", init_image.width))
        req_height = int(params.get("height", init_image.height))
        width, height = align_to_grid(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Lens] Resolution aligned: {req_width}×{req_height} → {width}×{height}")
        latent_h = height // 16
        latent_w = width // 16

        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)

        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: Text encoding
            print("[Lens] Stage 1: Text encoding...")
            if not cpu_text_encoding:
                text_encoder = self._lens_move("text_encoder", device, text_encoder_quantization)
            negpip_params = self._lens_negpip_params(
                prompt, negative_prompt,
                params.get("nag_negative_prompt", "") or "", max_sequence_length,
            )
            encoder_features, encoder_mask = encode_prompt(
                text_encoder, tokenizer, prompt, negative_prompt,
                device=enc_device, dtype=dtype, max_length=max_sequence_length,
                skip_emphasis=negpip_params is not None,
            )
            nag_params = self._lens_encode_nag(
                params, text_encoder, tokenizer, negative_prompt,
                enc_device, dtype, max_sequence_length,
                skip_emphasis=negpip_params is not None,
            )
            if not cpu_text_encoding:
                self._lens_move("text_encoder", "cpu")
            if cpu_text_encoding:
                encoder_features = [f.to(device) for f in encoder_features]
                encoder_mask = encoder_mask.to(device)
                if nag_params is not None:
                    nag_params["nag_features"] = [f.to(device) for f in nag_params["nag_features"]]
                    nag_params["nag_mask"] = nag_params["nag_mask"].to(device)

            # Free mxfp4 CUDA buffers (~9.7 GB) — not needed during denoising.
            import gc as _gc
            self.lens_components["text_encoder"] = None
            text_encoder = None
            _gc.collect()
            torch.cuda.empty_cache()

            # Stage 2: Encode init image + prepare mask
            print("[Lens] Stage 2: Encoding init image...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            init_latents = vae_encode(vae_gpu, init_image, height, width, device=device, dtype=dtype)
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mask_latent = prepare_mask_latent(mask_image, latent_h, latent_w, device=device, dtype=dtype)

            # Stage 3: Denoising with repaint
            print("[Lens] Stage 3: Denoising (repaint)...")
            transformer = self._lens_stage_transformer(params, device, transformer_quantization)
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_lens(lora_configs) if lora_configs else 0
            transformer = self.lens_components["transformer"]
            self._lens_set_attention_backend(transformer, params)
            try:
                latents = denoise_loop_inpaint(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents, mask_latent=mask_latent,
                    denoising_strength=denoising_strength,
                    encoder_features=encoder_features, encoder_mask=encoder_mask,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    latent_h=latent_h, latent_w=latent_w, seed=seed,
                    progress_callback=progress_callback,
                    advanced_cfg=self._lens_advanced_cfg(params),
                    spectrum_params=params,
                    nag_params=nag_params,
                    negpip_params=negpip_params,
                    tokenizer=tokenizer,
                )
            finally:
                if applied_lora_count:
                    self._unload_lora_lens()
            self._lens_unstage_transformer()
            del encoder_features, encoder_mask, init_latents, mask_latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 4: VAE decode
            print("[Lens] Stage 4: VAE decode...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            self._apply_vae_tiling(vae_gpu, getattr(self, "_vae_tiling", False))
            image = vae_decode(vae_gpu, latents, latent_h, latent_w)
            del latents
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Lens] inpaint completed")
            return image, seed, 0

        except Exception as e:
            print(f"[Lens] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            if self.lens_components.get("text_encoder") is not None:
                import gc as _gc
                self.lens_components["text_encoder"] = None
                _gc.collect()
            # Strip any leftover block-swap offloader (e.g. if setup/denoise raised mid-way).
            _t = (self.lens_components or {}).get("transformer")
            if _t is not None and hasattr(_t, "_block_offloader"):
                try:
                    delattr(_t, "_block_offloader")
                except Exception:
                    pass
            self._lens_offloader = None
            for _comp in ("transformer", "vae"):
                try:
                    self._lens_move(_comp, "cpu")
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
