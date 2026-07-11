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
import time as _time
from core.inference.generation_timing import generation_timer

class ZImageMixin:
    """ZImageMixin: zimage backend methods extracted verbatim from pipeline.py."""

    def _load_lora_zimage(self, lora_configs: List[Dict]):
        """Load LoRAs for Z-Image Transformer

        Args:
            lora_configs: List of LoRA configurations

        Note:
            Z-Image uses component-based architecture (not pipeline-based).
            LoRAs wrap original linear layers (forward-time addition, not weight merging).
            This allows LoRAs to be unloaded by restoring original modules.
            Based on training implementation in lora_trainer.py:674-708
        """
        if not lora_configs:
            return

        if not self.zimage_components:
            print("[Z-Image LoRA] WARNING: Z-Image components not loaded")
            return

        transformer = self.zimage_components["transformer"]

        # Store original modules for unloading (first time only)
        if not hasattr(self, '_zimage_lora_original_modules'):
            self._zimage_lora_original_modules = {}
            self._zimage_lora_wrapped_modules = set()  # Track which modules have LoRA

        # Use global lora_manager instance (has user-configured additional_dirs)
        from core.extensions.lora_manager import lora_manager

        print(f"[Z-Image LoRA] Loading {len(lora_configs)} LoRA(s)...")

        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config.get("path", "")
            lora_strength = lora_config.get("strength", 1.0)

            # Resolve path using LoRAManager (checks lora_dir + additional_dirs)
            resolved_path = lora_manager._resolve_lora_path(lora_path)

            if resolved_path is None:
                print(f"[Z-Image LoRA] WARNING: LoRA file not found: {lora_path}")
                print(f"[Z-Image LoRA]   Searched in: {lora_manager.lora_dir}")
                print(f"[Z-Image LoRA]   Additional dirs: {lora_manager.additional_dirs}")
                continue

            print(f"[Z-Image LoRA] Loading LoRA {i+1}/{len(lora_configs)}: {lora_path} (strength={lora_strength})")

            # Load LoRA weights
            from safetensors import safe_open

            try:
                with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
                    lora_state_dict = {key: f.get_tensor(key) for key in f.keys()}

                print(f"[Z-Image LoRA] Loaded {len(lora_state_dict)} tensors from {lora_path}")

                # Apply LoRA to transformer attention modules
                # Target modules: to_q, to_k, to_v, to_out.0 in ZImageAttention
                applied_count = 0

                # Find all attention modules
                for attn_name, attn_module in transformer.named_modules():
                    if "ZImageAttention" not in attn_module.__class__.__name__:
                        continue

                    # Apply to to_q, to_k, to_v
                    for attr_name in ["to_q", "to_k", "to_v"]:
                        if hasattr(attn_module, attr_name):
                            original_linear = getattr(attn_module, attr_name)

                            if isinstance(original_linear, torch.nn.Linear):
                                # Build LoRA key prefix
                                lora_key_prefix = f"transformer.{attn_name}.{attr_name}"
                                lora_down_key = f"{lora_key_prefix}.lora_down.weight"
                                lora_up_key = f"{lora_key_prefix}.lora_up.weight"

                                # Check if LoRA weights exist for this module
                                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                    lora_down_weight = lora_state_dict[lora_down_key]
                                    lora_up_weight = lora_state_dict[lora_up_key]

                                    # Load alpha if present
                                    lora_alpha_key = f"{lora_key_prefix}.alpha"
                                    lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                    # Wrap with LoRA layer
                                    module_key = f"{attn_name}.{attr_name}"
                                    wrapped_module = self._wrap_with_lora(
                                        attn_module,
                                        attr_name,
                                        original_linear,
                                        lora_down_weight,
                                        lora_up_weight,
                                        lora_strength,
                                        lora_alpha,
                                        module_key
                                    )
                                    if wrapped_module is not None:
                                        applied_count += 1

                    # Apply to to_out.0 (ModuleList)
                    if hasattr(attn_module, "to_out") and isinstance(attn_module.to_out, torch.nn.ModuleList):
                        if len(attn_module.to_out) > 0 and isinstance(attn_module.to_out[0], torch.nn.Linear):
                            original_linear = attn_module.to_out[0]

                            lora_key_prefix = f"transformer.{attn_name}.to_out.0"
                            lora_down_key = f"{lora_key_prefix}.lora_down.weight"
                            lora_up_key = f"{lora_key_prefix}.lora_up.weight"

                            if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                lora_down_weight = lora_state_dict[lora_down_key]
                                lora_up_weight = lora_state_dict[lora_up_key]

                                # Load alpha if present
                                lora_alpha_key = f"{lora_key_prefix}.alpha"
                                lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                # Wrap with LoRA layer (to_out is ModuleList, replace [0])
                                module_key = f"{attn_name}.to_out.0"
                                wrapped_module = self._wrap_with_lora(
                                    attn_module.to_out,
                                    0,  # ModuleList index
                                    original_linear,
                                    lora_down_weight,
                                    lora_up_weight,
                                    lora_strength,
                                    lora_alpha,
                                    module_key
                                )
                                if wrapped_module is not None:
                                    applied_count += 1

                print(f"[Z-Image LoRA] Applied LoRA to {applied_count} modules")

            except Exception as e:
                print(f"[Z-Image LoRA] ERROR: Failed to load LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()

    def _wrap_with_lora(self, parent_module, attr_name, original_linear, lora_down_weight, lora_up_weight, strength, alpha, module_key):
        """Wrap a linear layer with LoRA

        Args:
            parent_module: Parent module containing the linear layer
            attr_name: Attribute name or index (for ModuleList)
            original_linear: Original linear layer
            lora_down_weight: LoRA down projection weight [rank, in_features]
            lora_up_weight: LoRA up projection weight [out_features, rank]
            strength: LoRA strength multiplier
            alpha: LoRA alpha parameter
            module_key: Unique key for this module (for tracking)

        Returns:
            Wrapped LoRA module or None if failed
        """
        # Import LoRALinearLayer from training adapters (model-agnostic wrapper class)
        from core.training.adapters.sd15_adapter import LoRALinearLayer
        import numpy as np

        # Get true original module (unwrap if it's already a LoRA wrapper)
        LoRALinearLayerClass = LoRALinearLayer  # Same class, just alias for clarity

        if isinstance(original_linear, LoRALinearLayerClass):
            # Already wrapped - extract the original module
            true_original = original_linear.original_module
            print(f"[Z-Image LoRA DEBUG] Detected existing LoRA wrapper, extracting original module")
        else:
            true_original = original_linear

        # Save original module (first time only)
        if module_key not in self._zimage_lora_original_modules:
            self._zimage_lora_original_modules[module_key] = true_original

        # Compute rank and alpha value
        rank = lora_down_weight.shape[0]
        alpha_value = alpha.item() if alpha is not None else rank

        # Create LoRA wrapper using the true original module
        # lora_name is required parameter, use module_key for identification
        lora_wrapper = LoRALinearLayer(
            true_original, rank=rank, alpha=alpha_value, lora_name=module_key
        )

        # Load pretrained LoRA weights
        device = true_original.weight.device
        dtype = true_original.weight.dtype

        with torch.no_grad():
            lora_wrapper.lora_down.weight.data = lora_down_weight.to(device=device, dtype=dtype)
            lora_wrapper.lora_up.weight.data = lora_up_weight.to(device=device, dtype=dtype)

        # Apply strength by adjusting scaling (override the default scale)
        lora_wrapper.scale = (alpha_value / rank) * strength

        # Replace in parent module
        if isinstance(attr_name, int):
            # ModuleList index
            parent_module[attr_name] = lora_wrapper
        else:
            # Attribute name
            setattr(parent_module, attr_name, lora_wrapper)

        # Track wrapped modules
        self._zimage_lora_wrapped_modules.add(module_key)

        print(f"[Z-Image LoRA DEBUG] Wrapped {module_key}: alpha={alpha_value:.1f}, rank={rank}, strength={strength:.2f}, scaling={lora_wrapper.scaling:.4f}")

        return lora_wrapper

    def _unload_lora_zimage(self):
        """Unload LoRAs from Z-Image Transformer

        Restores original linear layers by removing LoRA wrappers.
        """
        if not hasattr(self, '_zimage_lora_original_modules'):
            print("[Z-Image LoRA] No LoRAs loaded")
            return

        if not self.zimage_components:
            print("[Z-Image LoRA] WARNING: Z-Image components not loaded")
            return

        transformer = self.zimage_components["transformer"]
        unloaded_count = 0

        print(f"[Z-Image LoRA] Unloading LoRAs ({len(self._zimage_lora_wrapped_modules)} modules)...")

        # Restore original modules
        for attn_name, attn_module in transformer.named_modules():
            if "ZImageAttention" not in attn_module.__class__.__name__:
                continue

            # Restore to_q, to_k, to_v
            for attr_name in ["to_q", "to_k", "to_v"]:
                module_key = f"{attn_name}.{attr_name}"
                if module_key in self._zimage_lora_original_modules:
                    original_module = self._zimage_lora_original_modules[module_key]
                    setattr(attn_module, attr_name, original_module)
                    unloaded_count += 1

            # Restore to_out.0 (ModuleList)
            if hasattr(attn_module, "to_out") and isinstance(attn_module.to_out, torch.nn.ModuleList):
                module_key = f"{attn_name}.to_out.0"
                if module_key in self._zimage_lora_original_modules:
                    original_module = self._zimage_lora_original_modules[module_key]
                    attn_module.to_out[0] = original_module
                    unloaded_count += 1

        # Clear wrapped modules tracking (but keep original modules for future loads)
        self._zimage_lora_wrapped_modules.clear()

        print(f"[Z-Image LoRA] Unloaded {unloaded_count} LoRA modules")
        print(f"[Z-Image LoRA] Original modules preserved for future LoRA loads")

    def _zimage_cleanup(self, gen_succeeded=True, keep_te=False, keep_transformer=False, keep_vae=False):
        """Safety-net CPU offload for Z-Image components.

        On the happy path each generate function already offloads text_encoder,
        transformer and vae to CPU inline via move_zimage_*_to_cpu(). This helper
        is called from a `finally` block in every generate entry point so that an
        exception raised mid-generation (denoise loop, VAE decode, etc.) cannot
        leave the transformer (largest component) or TE resident on GPU.

        Idempotent: re-running after the happy-path cleanup is a cheap no-op
        (.to("cpu") on an already-CPU module). Never raises - a failure here
        must not mask the original exception from the caller's try/except.

        Block Swap: mirrors the existing happy-path behavior, which offloads the
        transformer via move_zimage_transformer_to_cpu() without a separate
        offloader.cleanup() step (the offloader's hooks live on the transformer
        itself via transformer._block_offloader and don't need explicit teardown
        here), so this safety net does not fight the offloader's block residency
        management - it only ensures the transformer as a whole lands on CPU.

        Keep-models-hot (see core/keep_hot.py): ``gen_succeeded`` and the
        ``keep_*`` flags let a successful generation that opted into
        keep_models_hot skip forcing a component back to CPU here when the
        caller already decided (and is tracking via keep_hot.mark_resident)
        that it should stay GPU-resident for the next generation. On any
        failed generation (``gen_succeeded=False``) the keep flags are
        ignored entirely and every component is force-offloaded, exactly as
        before this feature existed -- never trust GPU residency after an
        exception. Defaults are chosen so a call with no arguments reproduces
        the pre-keep-hot behavior (force-offload everything).
        """
        try:
            from core.vram_optimization import (
                move_zimage_text_encoder_to_cpu,
                move_zimage_transformer_to_cpu,
                move_zimage_vae_to_cpu,
            )

            components = getattr(self, "zimage_components", None) or {}

            _kh_skip = set()
            if gen_succeeded:
                if keep_te:
                    _kh_skip.add("text_encoder")
                if keep_transformer:
                    _kh_skip.add("transformer")
                if keep_vae:
                    _kh_skip.add("vae")

            text_encoder = components.get("text_encoder")
            if text_encoder is not None and "text_encoder" not in _kh_skip:
                try:
                    move_zimage_text_encoder_to_cpu(text_encoder)
                except Exception as e:
                    print(f"[Z-Image] cleanup: failed to offload text_encoder to CPU: {e}")

            transformer = components.get("transformer")
            if transformer is not None and "transformer" not in _kh_skip:
                try:
                    move_zimage_transformer_to_cpu(transformer)
                except Exception as e:
                    print(f"[Z-Image] cleanup: failed to offload transformer to CPU: {e}")

            vae = components.get("vae")
            if vae is not None and "vae" not in _kh_skip:
                try:
                    move_zimage_vae_to_cpu(vae)
                except Exception as e:
                    print(f"[Z-Image] cleanup: failed to offload vae to CPU: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Z-Image] cleanup: unexpected error during safety-net cleanup: {e}")

    def _get_zimage_scheduler(self, sampler: str):
        """
        Get appropriate Flow Match scheduler for Z-Image based on sampler selection

        Z-Image uses Flow Matching schedulers (different from SD/SDXL).
        Maps user-selected sampler to compatible Flow Match scheduler.

        Sampler mapping:
        - euler → FlowMatchEulerDiscreteScheduler (stochastic_sampling=False)
        - euler_a → FlowMatchEulerDiscreteScheduler (stochastic_sampling=True)
        - heun → FlowMatchHeunDiscreteScheduler

        Args:
            sampler: User-selected sampler name (e.g., "euler", "heun")

        Returns:
            Configured Flow Match scheduler instance
        """
        from diffusers.schedulers import (
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
        )

        base_scheduler = self.zimage_components["scheduler"]
        config = base_scheduler.config

        # Map sampler to Flow Match scheduler class
        if sampler == "heun":
            scheduler_class = FlowMatchHeunDiscreteScheduler
            print(f"[Z-Image] Using FlowMatchHeunDiscreteScheduler for sampler '{sampler}'")
            return scheduler_class.from_config(config)
        else:
            # Euler/Euler a: use FlowMatchEulerDiscreteScheduler with stochastic_sampling flag
            is_ancestral = sampler in ["euler_a", "dpm2_a"]
            print(f"[Z-Image] Using FlowMatchEulerDiscreteScheduler for sampler '{sampler}' (stochastic={is_ancestral})")

            # Create config dict and enable stochastic_sampling for ancestral samplers
            scheduler_config = dict(config)
            scheduler_config["stochastic_sampling"] = is_ancestral

            return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    def _generate_txt2img_zimage(self, params: Dict[str, Any], progress_callback=None, step_callback=None) -> tuple[Image.Image, int]:
        """Generate image from text using Z-Image

        Args:
            params: Generation parameters
            progress_callback: Legacy callback (not used for Z-Image)
            step_callback: Step callback (not used for Z-Image)

        Returns:
            tuple: (image, actual_seed)
        """
        if not self.zimage_components:
            raise RuntimeError("Z-Image components not loaded. Please load a Z-Image model first.")

        print("[Z-Image] Starting txt2img generation")

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        # Match the ACTUAL offloader-creation guard: the transformer streaming
        # offloader is created whenever enable_block_swap is set (the blocks_to_swap
        # count is clamped, 0 -> a fully-resident no-op offloader), regardless of the
        # count. Keying transformer-hot eligibility on enable_block_swap alone keeps
        # the invariant "transformer kept hot => no offloader attached" exact.
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False))

        def _kh_offload_zimage():
            comps = getattr(self, "zimage_components", None) or {}
            for _kh_key in ("text_encoder", "transformer", "vae"):
                _kh_comp = comps.get(_kh_key)
                if _kh_comp is not None:
                    try:
                        _kh_comp.to("cpu")
                    except Exception:
                        pass

        invalidate_if_model_changed(self, params, offload_fn=_kh_offload_zimage)

        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.zimage_components.get("text_encoder"))
            if not _kh_has_loras and not _kh_is_block_swapped:
                _kh_total_bytes += component_nbytes(self.zimage_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.zimage_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_has_loras and not _kh_is_block_swapped
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:

            # Extract components
            transformer = self.zimage_components["transformer"]
            vae = self.zimage_components["vae"]
            text_encoder = self.zimage_components["text_encoder"]
            tokenizer = self.zimage_components["tokenizer"]

            # Get scheduler based on user-selected sampler
            # Z-Image uses Flow Match schedulers (different from SD/SDXL)
            sampler = params.get("sampler", "euler")
            scheduler = self._get_zimage_scheduler(sampler)

            # Set attention backend based on global settings or params
            attention_type = params.get("attention_type", settings.attention_type)

            # Only switch if attention type has changed (avoid redundant switching overhead)
            if attention_type != self.current_attention_type:
                print(f"[Z-Image] Switching attention backend: {self.current_attention_type} -> {attention_type}")
                # normalize_backend ("normal"->"native"; "sla" preserved) + set on
                # BOTH module identities (dual-module hazard). See helper docstring.
                from core.models.zimage_transformer import set_zimage_attention_backend
                set_zimage_attention_backend(attention_type)
                self.current_attention_type = attention_type
            else:
                print(f"[Z-Image] Attention backend already set to: {attention_type} (skipping)")
                from core.models.zimage_transformer import set_zimage_attention_backend
                set_zimage_attention_backend(attention_type)  # Ensure it's set (for safety)

            # Load or unload LoRAs
            lora_configs = params.get("loras", [])
            print(f"[Z-Image] DEBUG: lora_configs received: {lora_configs}")
            print(f"[Z-Image] DEBUG: lora_configs type: {type(lora_configs)}")
            print(f"[Z-Image] DEBUG: lora_configs length: {len(lora_configs) if lora_configs else 0}")

            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()
                # Load new LoRAs
                self._load_lora_zimage(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Determine ancestral seed for database storage (stochastic_sampling uses internal RNG)
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                # Generate random seed for reproducibility tracking
                actual_ancestral_seed = random.randint(0, 2147483647)
                print(f"[Z-Image] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                # Use specified seed
                actual_ancestral_seed = ancestral_seed
                print(f"[Z-Image] Using specified ancestral seed: {ancestral_seed}")

            # Z-Image parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            num_inference_steps = params.get("steps", 8)  # Turbo default: 8 steps
            max_sequence_length = params.get("max_sequence_length", 512)

            # Z-Image supports CFG (guidance_scale)
            # CFG=1.0: no CFG (positive only)
            # CFG!=1.0: CFG enabled
            guidance_scale = params.get("cfg_scale", 3.5)

            print(f"[Z-Image] Generating {width}x{height} image")
            print(f"[Z-Image] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
            print(f"[Z-Image] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                log_device_status,
                move_zimage_text_encoder_to_gpu,
                move_zimage_text_encoder_to_cpu,
                move_zimage_transformer_to_gpu,
                move_zimage_transformer_to_cpu,
                move_zimage_vae_to_gpu,
                move_zimage_vae_to_cpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")  # Transformer (U-Net equivalent)
            text_encoder_quantization = params.get("text_encoder_quantization")  # Text Encoder (Z-Image only)

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            if not is_resident(self, "text_encoder", _kh_model_key):
                text_encoder = move_zimage_text_encoder_to_gpu(text_encoder, text_encoder_quantization)
            log_device_status("Ready for Z-Image text encoding", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance = \
                self._zimage_encode_prompt(
                    text_encoder, tokenizer, prompt, negative_prompt,
                    guidance_scale, max_sequence_length, text_encoder_quantization
                )

            # NAG: encode the nag-negative prompt while the text encoder is still on GPU
            # (None when NAG is off -> generation path is unchanged).
            nag_negative_embeds_list = self._zimage_encode_nag_negative(
                text_encoder, tokenizer, params, prompt, max_sequence_length,
                text_encoder_quantization
            )

            # Offload Text Encoder to CPU to free VRAM (unless kept hot -- see
            # core/keep_hot.py; the finally block below records the residency).
            if not _kh_keep_te:
                move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: Denoising Loop
            # ============================================================
            # Block Swap parameters
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 20)
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_swap_h2d_only = params.get("block_swap_h2d_only", False)
            block_swap_ring_size = int(params.get("block_swap_ring_size", 2))

            if not enable_block_swap:
                # Normal mode: move entire Transformer to GPU
                if not is_resident(self, "transformer", _kh_model_key):
                    transformer = move_zimage_transformer_to_gpu(transformer, transformer_quantization)

                # DEBUG: Verify LoRA is still applied after GPU move
                if lora_configs:
                    for attn_name, attn_module in transformer.named_modules():
                        if "ZImageAttention" in attn_module.__class__.__name__:
                            if hasattr(attn_module, "to_q"):
                                weight_norm = attn_module.to_q.weight.data.norm().item()
                                print(f"[Z-Image LoRA DEBUG] After GPU move, first attention to_q weight norm: {weight_norm:.4f}")
                            break

                log_device_status("Ready for Z-Image denoising loop", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })
            else:
                # Block Swap mode: keep Transformer on CPU for Block Swap initialization
                print("[Z-Image] Block Swap enabled - keeping Transformer on CPU for Block Swap initialization")

                # Create block offloader
                from core.memory_management import create_block_offloader_for_model

                block_offloader = create_block_offloader_for_model(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    h2d_only=block_swap_h2d_only,
                    ring_size=block_swap_ring_size,
                )

                # Attach block offloader to transformer
                transformer._block_offloader = block_offloader

                # Prepare block devices (this moves blocks to GPU/CPU according to strategy)
                block_offloader.prepare_block_devices_before_forward()

                log_device_status("Ready for Z-Image denoising loop (Block Swap enabled)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })

            # Training-free reference-style transfer setup. OFF by default
            # (style_transfer/style_transfers absent -> (None, None, None, None, "stack"), no-op
            # below). ``style_refs`` is populated (and style_cfg/style_ref_x0/style_eps_ref left
            # None) ONLY when ``params["style_transfers"]`` carries 2+ references -- a single
            # reference (via either key) always resolves through the
            # style_cfg/style_ref_x0/style_eps_ref triple, so that code path (both here and inside
            # ``_zimage_denoising_loop``/``_zimage_style_step``) is untouched. txt2img has no other
            # VAE-encode stage (unlike img2img/inpaint), so the VAE is briefly staged to GPU here
            # just to encode the reference(s), then offloaded again.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                if not is_resident(self, "vae", _kh_model_key):
                    move_zimage_vae_to_gpu(vae)
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._zimage_style_configs(
                        params, vae, height, width, torch.device(self.device), generator=generator
                    )
                move_zimage_vae_to_cpu(vae)
                if style_refs is not None:
                    print(f"[Z-Image] Multi-reference style transfer active: {len(style_refs)} refs, "
                          f"combine_mode={style_combine_mode}")
                elif style_cfg is not None:
                    print(f"[Z-Image] Style transfer active: ref_k_strength={style_cfg.ref_k_strength}, "
                          f"adain_strength={style_cfg.adain_strength}, block_range={style_cfg.block_range}")

            latents = self._zimage_denoising_loop(
                transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
                height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
                generator, progress_callback, step_callback,
                spectrum_params=params,
                nag_negative_embeds_list=nag_negative_embeds_list,
                nag_params=params,
                style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                style_refs=style_refs, style_combine_mode=style_combine_mode,
            )

            # Offload Transformer to CPU to free VRAM for VAE (unless kept hot --
            # only possible when block swap was NOT active, see setup above).
            if not _kh_keep_transformer:
                move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 3: VAE Decode
            # ============================================================
            if not is_resident(self, "vae", _kh_model_key):
                move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding (unless kept hot)
            if not _kh_keep_vae:
                move_zimage_vae_to_cpu(vae)

            # Clear intermediate tensors from GPU memory
            del prompt_embeds_list, negative_prompt_embeds_list, latents
            torch.cuda.empty_cache()  # Release PyTorch's VRAM cache

            log_device_status("VAE decode complete, all components offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            print("[Z-Image] Generation completed")
            _kh_gen_succeeded = True

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image generation failed: {str(e)}")
        finally:
            if not _kh_gen_succeeded:
                clear_resident(self)
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    discard_resident(self, "vae")
            self._zimage_cleanup(
                gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te,
                keep_transformer=_kh_keep_transformer,
                keep_vae=_kh_keep_vae,
            )

    def _generate_img2img_zimage(self, params: Dict[str, Any], init_image: Image.Image, progress_callback=None, step_callback=None) -> tuple[Image.Image, int]:
        """Generate image from image using Z-Image

        Args:
            params: Generation parameters
            init_image: Input PIL image
            progress_callback: Legacy callback (not used for Z-Image)
            step_callback: Step callback (not used for Z-Image)

        Returns:
            tuple: (image, actual_seed)
        """
        if not self.zimage_components:
            raise RuntimeError("Z-Image components not loaded. Please load a Z-Image model first.")

        print("[Z-Image] Starting img2img generation")

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        # Match the ACTUAL offloader-creation guard (created whenever
        # enable_block_swap is set; blocks_to_swap count is clamped). Keeps the
        # invariant "transformer kept hot => no offloader attached" exact.
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False))

        def _kh_offload_zimage():
            comps = getattr(self, "zimage_components", None) or {}
            for _kh_key in ("text_encoder", "transformer", "vae"):
                _kh_comp = comps.get(_kh_key)
                if _kh_comp is not None:
                    try:
                        _kh_comp.to("cpu")
                    except Exception:
                        pass

        invalidate_if_model_changed(self, params, offload_fn=_kh_offload_zimage)

        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.zimage_components.get("text_encoder"))
            if not _kh_has_loras and not _kh_is_block_swapped:
                _kh_total_bytes += component_nbytes(self.zimage_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.zimage_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_has_loras and not _kh_is_block_swapped
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            # Extract components
            transformer = self.zimage_components["transformer"]
            vae = self.zimage_components["vae"]
            text_encoder = self.zimage_components["text_encoder"]
            tokenizer = self.zimage_components["tokenizer"]

            # Get scheduler based on user-selected sampler
            # Z-Image uses Flow Match schedulers (different from SD/SDXL)
            sampler = params.get("sampler", "euler")
            scheduler = self._get_zimage_scheduler(sampler)

            # Set attention backend
            attention_type = params.get("attention_type", settings.attention_type)
            if attention_type != self.current_attention_type:
                print(f"[Z-Image] Switching attention backend: {self.current_attention_type} -> {attention_type}")
                # normalize_backend ("normal"->"native"; "sla" preserved) + set on
                # BOTH module identities (dual-module hazard). See helper docstring.
                from core.models.zimage_transformer import set_zimage_attention_backend
                set_zimage_attention_backend(attention_type)
                self.current_attention_type = attention_type
            else:
                print(f"[Z-Image] Attention backend already set to: {attention_type} (skipping)")
                from core.models.zimage_transformer import set_zimage_attention_backend
                set_zimage_attention_backend(attention_type)

            # Load or unload LoRAs
            lora_configs = params.get("loras", [])
            if lora_configs:
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()
                self._load_lora_zimage(lora_configs)
            else:
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Determine ancestral seed for database storage (stochastic_sampling uses internal RNG)
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                # Generate random seed for reproducibility tracking
                actual_ancestral_seed = random.randint(0, 2147483647)
                print(f"[Z-Image] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                # Use specified seed
                actual_ancestral_seed = ancestral_seed
                print(f"[Z-Image] Using specified ancestral seed: {ancestral_seed}")

            # Z-Image parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            num_inference_steps = params.get("steps", 8)
            max_sequence_length = params.get("max_sequence_length", 512)
            guidance_scale = params.get("cfg_scale", 3.5)

            # img2img specific parameters
            denoising_strength = params.get("denoising_strength", 0.75)

            print(f"[Z-Image] Generating {width}x{height} image from input image")
            print(f"[Z-Image] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}, Strength: {denoising_strength}")
            print(f"[Z-Image] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                log_device_status,
                move_zimage_text_encoder_to_gpu,
                move_zimage_text_encoder_to_cpu,
                move_zimage_transformer_to_gpu,
                move_zimage_transformer_to_cpu,
                move_zimage_vae_to_gpu,
                move_zimage_vae_to_cpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            if not is_resident(self, "text_encoder", _kh_model_key):
                text_encoder = move_zimage_text_encoder_to_gpu(text_encoder, text_encoder_quantization)
            log_device_status("Ready for Z-Image text encoding", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance = \
                self._zimage_encode_prompt(
                    text_encoder, tokenizer, prompt, negative_prompt,
                    guidance_scale, max_sequence_length, text_encoder_quantization
                )

            # NAG: encode the nag-negative prompt while the text encoder is still on GPU
            # (None when NAG is off -> generation path is unchanged).
            nag_negative_embeds_list = self._zimage_encode_nag_negative(
                text_encoder, tokenizer, params, prompt, max_sequence_length,
                text_encoder_quantization
            )

            # Offload Text Encoder to CPU (unless kept hot -- TE is not touched
            # again in this generation, so this is also TE's keep-hot exit point).
            if not _kh_keep_te:
                move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: VAE Encode Input Image
            # ============================================================
            if not is_resident(self, "vae", _kh_model_key):
                move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE encode (img2img)", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # Resize input image if needed
            if init_image.size != (width, height):
                print(f"[Z-Image] Resizing input image from {init_image.size} to {width}x{height}")
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

            # Prepare image tensor
            import numpy as np
            image_array = np.array(init_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(device=self.device, dtype=vae.dtype)

            # Encode to latent space
            # Z-Image VAE uses encoder -> quant_conv -> sample (not encode method)
            with torch.no_grad():
                h = vae.encoder(image_tensor)
                if vae.quant_conv is not None:
                    h = vae.quant_conv(h)
                mean, logvar = torch.chunk(h, 2, dim=1)
                std = torch.exp(0.5 * logvar)

                # Generate noise with generator
                noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
                init_latents = mean + std * noise

                # Z-Image VAE scaling factor (apply scaling and shift)
                if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
                    init_latents = init_latents * vae.config.scaling_factor
                else:
                    # Fallback: assume standard scaling
                    init_latents = init_latents * 0.13025

                # Clean up intermediate tensors
                del h, mean, logvar, std

            print(f"[Z-Image] Encoded input image to latents: {init_latents.shape}")

            # Training-free reference-style transfer setup (see the txt2img comment above for the
            # single-ref/multi-ref routing invariant). VAE is already resident on GPU from the
            # init-image encode above, so the reference encode(s) piggyback on it before the
            # offload below.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._zimage_style_configs(
                        params, vae, height, width, torch.device(self.device), generator=generator
                    )
                if style_refs is not None:
                    print(f"[Z-Image] Multi-reference style transfer active: {len(style_refs)} refs, "
                          f"combine_mode={style_combine_mode}")
                elif style_cfg is not None:
                    print(f"[Z-Image] Style transfer active: ref_k_strength={style_cfg.ref_k_strength}, "
                          f"adain_strength={style_cfg.adain_strength}, block_range={style_cfg.block_range}")

            # Offload VAE to CPU after encoding
            move_zimage_vae_to_cpu(vae)

            # ============================================================
            # Stage 3: Add Noise to Latents (Flow Matching Style)
            # ============================================================
            device = torch.device(self.device)

            # Calculate VAE scale factor for dynamic shift
            if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            else:
                vae_scale_factor = 8

            # Calculate dynamic shift
            from core.zimage_utils import calculate_shift
            image_seq_len = (init_latents.shape[2] // 2) * (init_latents.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )

            # Set scheduler parameters
            scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            # Prepare full timesteps first
            scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
            timesteps = scheduler.timesteps

            # Calculate timestep to start from (based on strength)
            init_timestep = int(num_inference_steps * denoising_strength)
            t_start = max(num_inference_steps - init_timestep, 0)

            # Get partial timesteps for img2img
            timesteps_img2img = timesteps[t_start:]

            print(f"[Z-Image] img2img: Using {len(timesteps_img2img)}/{len(timesteps)} timesteps (t_start={t_start}, strength={denoising_strength})")

            # Add noise to init_latents at the starting timestep
            noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=torch.float32)

            # Flow Matching noise addition
            # Check if scheduler has add_noise method
            if hasattr(scheduler, 'add_noise'):
                print(f"[Z-Image] Using scheduler.add_noise() for noise addition")
                noised_latents = scheduler.add_noise(init_latents, noise, timesteps_img2img[0:1])
            else:
                # Manual flow matching noise addition: x_t = (1 - t) * x_0 + t * noise
                # Normalize timestep to [0, 1] range (Z-Image: 1000=start/noisy, 0=end/clean)
                t_normalized = timesteps_img2img[0].item() / 1000.0
                print(f"[Z-Image] Manual flow matching noise addition: t={timesteps_img2img[0].item():.1f}, t_norm={t_normalized:.3f}")
                noised_latents = (1.0 - t_normalized) * init_latents + t_normalized * noise

            print(f"[Z-Image] Noised latents shape: {noised_latents.shape}, dtype: {noised_latents.dtype}")

            # ============================================================
            # Stage 4: Denoising Loop
            # ============================================================
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 20)
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_swap_h2d_only = params.get("block_swap_h2d_only", False)
            block_swap_ring_size = int(params.get("block_swap_ring_size", 2))

            if not enable_block_swap:
                if not is_resident(self, "transformer", _kh_model_key):
                    transformer = move_zimage_transformer_to_gpu(transformer, transformer_quantization)
                log_device_status("Ready for Z-Image denoising loop (img2img)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })
            else:
                print("[Z-Image] Block Swap enabled - keeping Transformer on CPU for Block Swap initialization")
                from core.memory_management import create_block_offloader_for_model
                block_offloader = create_block_offloader_for_model(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    h2d_only=block_swap_h2d_only,
                    ring_size=block_swap_ring_size,
                )
                transformer._block_offloader = block_offloader
                block_offloader.prepare_block_devices_before_forward()
                log_device_status("Ready for Z-Image denoising loop (Block Swap enabled, img2img)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })

            # Run denoising loop with noised latents and partial timesteps
            latents = self._zimage_denoising_loop(
                transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
                height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
                generator, progress_callback, step_callback,
                init_latents=noised_latents,
                timesteps_override=timesteps_img2img,
                spectrum_params=params,
                nag_negative_embeds_list=nag_negative_embeds_list,
                nag_params=params,
                style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                style_refs=style_refs, style_combine_mode=style_combine_mode,
            )

            # Offload Transformer to CPU (unless kept hot -- only possible when
            # block swap was NOT active, see keep-hot setup above)
            if not _kh_keep_transformer:
                move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 5: VAE Decode
            # ============================================================
            # NOTE: VAE was already staged to GPU once for input-image encoding
            # (Stage 2) and offloaded again there unconditionally -- that offload
            # is a within-generation VRAM-relief step, not the keep-hot exit
            # boundary, so it is intentionally left untouched by keep-hot. This
            # reload IS a normal re-stage every time (never resident-skipped)
            # because the mid-generation offload above always runs.
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding (unless kept hot -- this is VAE's
            # true keep-hot exit point for this generation)
            if not _kh_keep_vae:
                move_zimage_vae_to_cpu(vae)

            # Clear intermediate tensors
            del prompt_embeds_list, negative_prompt_embeds_list, init_latents, noised_latents, latents
            torch.cuda.empty_cache()

            log_device_status("VAE decode complete, all components offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            print("[Z-Image] img2img generation completed")
            _kh_gen_succeeded = True

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] img2img generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image img2img generation failed: {str(e)}")
        finally:
            if not _kh_gen_succeeded:
                clear_resident(self)
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    discard_resident(self, "vae")
            self._zimage_cleanup(
                gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te,
                keep_transformer=_kh_keep_transformer,
                keep_vae=_kh_keep_vae,
            )

    def _generate_inpaint_zimage(
        self, params: dict, init_image, mask_image, progress_callback=None, step_callback=None
    ) -> tuple:
        """
        Generate inpainted image using Z-Image model.

        Inpaint = img2img + mask blending
        - Encode init_image to latents
        - Add noise based on denoising_strength
        - Denoise with mask blending at each step
        - Decode back to image

        Args:
            params: Generation parameters (prompt, steps, cfg_scale, etc.)
            init_image: PIL Image (area to inpaint)
            mask_image: PIL Image (white = inpaint, black = keep)
            progress_callback: Progress callback function
            step_callback: Step callback function

        Returns:
            (generated_image, seed)
        """
        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        # Inpaint does not apply LoRA, but the LoRA hazard gate is kept uniform
        # across every keep-hot entry point regardless of per-arch LoRA support.
        _kh_has_loras = bool(params.get("loras") or [])
        # Match the ACTUAL offloader-creation guard (created whenever
        # enable_block_swap is set; blocks_to_swap count is clamped). Keeps the
        # invariant "transformer kept hot => no offloader attached" exact.
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False))

        def _kh_offload_zimage():
            comps = getattr(self, "zimage_components", None) or {}
            for _kh_key in ("text_encoder", "transformer", "vae"):
                _kh_comp = comps.get(_kh_key)
                if _kh_comp is not None:
                    try:
                        _kh_comp.to("cpu")
                    except Exception:
                        pass

        invalidate_if_model_changed(self, params, offload_fn=_kh_offload_zimage)

        _kh_total_bytes = 0
        if _kh_requested:
            _comps = getattr(self, "zimage_components", None) or {}
            _kh_total_bytes += component_nbytes(_comps.get("text_encoder"))
            if not _kh_has_loras and not _kh_is_block_swapped:
                _kh_total_bytes += component_nbytes(_comps.get("transformer"))
            _kh_total_bytes += component_nbytes(_comps.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_has_loras and not _kh_is_block_swapped
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            # Get components
            text_encoder = self.zimage_components["text_encoder"]
            tokenizer = self.zimage_components["tokenizer"]
            transformer = self.zimage_components["transformer"]
            vae = self.zimage_components["vae"]
            scheduler = self.zimage_components["scheduler"]

            # Get parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            num_inference_steps = params.get("steps", 8)
            guidance_scale = params.get("cfg_scale", 3.5)
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            seed = params.get("seed", -1)
            denoising_strength = params.get("denoising_strength", 0.75)
            mask_blur = params.get("mask_blur", 0)
            max_sequence_length = params.get("max_sequence_length", 256)

            # Generate seed
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Determine ancestral seed for database storage (stochastic_sampling uses internal RNG)
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                # Generate random seed for reproducibility tracking
                actual_ancestral_seed = random.randint(0, 2147483647)
                print(f"[Z-Image] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                # Use specified seed
                actual_ancestral_seed = ancestral_seed
                print(f"[Z-Image] Using specified ancestral seed: {ancestral_seed}")

            print(f"[Z-Image] Starting inpaint generation")
            print(f"[Z-Image] Generating {width}x{height} inpainted image")
            print(f"[Z-Image] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}, Strength: {denoising_strength}")
            print(f"[Z-Image] Mask blur: {mask_blur}")
            print(f"[Z-Image] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                log_device_status,
                move_zimage_text_encoder_to_gpu,
                move_zimage_text_encoder_to_cpu,
                move_zimage_transformer_to_gpu,
                move_zimage_transformer_to_cpu,
                move_zimage_vae_to_gpu,
                move_zimage_vae_to_cpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            if not is_resident(self, "text_encoder", _kh_model_key):
                text_encoder = move_zimage_text_encoder_to_gpu(text_encoder, text_encoder_quantization)
            log_device_status("Ready for Z-Image text encoding", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance = \
                self._zimage_encode_prompt(
                    text_encoder, tokenizer, prompt, negative_prompt,
                    guidance_scale, max_sequence_length, text_encoder_quantization
                )

            # NAG: encode the nag-negative prompt while the text encoder is still on GPU
            # (None when NAG is off -> generation path is unchanged).
            nag_negative_embeds_list = self._zimage_encode_nag_negative(
                text_encoder, tokenizer, params, prompt, max_sequence_length,
                text_encoder_quantization
            )

            # Offload Text Encoder to CPU (unless kept hot -- TE is not touched
            # again in this generation, so this is also TE's keep-hot exit point).
            if not _kh_keep_te:
                move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: VAE Encode Input Image and Mask
            # ============================================================
            if not is_resident(self, "vae", _kh_model_key):
                move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE encode (inpaint)", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # Resize input image and mask if needed
            if init_image.size != (width, height):
                print(f"[Z-Image] Resizing input image from {init_image.size} to {width}x{height}")
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

            if mask_image.size != (width, height):
                print(f"[Z-Image] Resizing mask from {mask_image.size} to {width}x{height}")
                mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)

            # Apply mask blur if requested
            if mask_blur > 0:
                from PIL import ImageFilter
                mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                print(f"[Z-Image] Applied Gaussian blur to mask (radius={mask_blur})")

            # Prepare image tensor
            import numpy as np
            image_array = np.array(init_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(device=self.device, dtype=vae.dtype)

            # Prepare mask tensor (white = 1 = inpaint, black = 0 = keep)
            mask_array = np.array(mask_image.convert('L')).astype(np.float32) / 255.0  # Grayscale
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # 1CHW
            mask_tensor = mask_tensor.to(device=self.device, dtype=vae.dtype)

            # Encode input image to latent space
            with torch.no_grad():
                h = vae.encoder(image_tensor)
                if vae.quant_conv is not None:
                    h = vae.quant_conv(h)
                mean, logvar = torch.chunk(h, 2, dim=1)
                std = torch.exp(0.5 * logvar)

                # Generate noise with generator
                noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
                init_latents = mean + std * noise

                # Z-Image VAE scaling factor
                if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
                    init_latents = init_latents * vae.config.scaling_factor
                else:
                    init_latents = init_latents * 0.13025

                # Store original latents for mask blending
                original_latents = init_latents.clone()

                # Clean up intermediate tensors
                del h, mean, logvar, std

            # Resize mask to latent dimensions (downsample by VAE scale factor)
            # Z-Image VAE: 8x downsampling -> latent is 1/8 of image size
            latent_height = init_latents.shape[2]
            latent_width = init_latents.shape[3]
            mask_latent = torch.nn.functional.interpolate(
                mask_tensor, size=(latent_height, latent_width), mode='nearest'
            )

            print(f"[Z-Image] Encoded input image to latents: {init_latents.shape}")
            print(f"[Z-Image] Mask latent shape: {mask_latent.shape}")

            # Training-free reference-style transfer setup (see the txt2img comment above for the
            # single-ref/multi-ref routing invariant). VAE is already resident on GPU from the
            # init-image encode above, so the reference encode(s) piggyback on it before the
            # offload below.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._zimage_style_configs(
                        params, vae, height, width, torch.device(self.device), generator=generator
                    )
                if style_refs is not None:
                    print(f"[Z-Image] Multi-reference style transfer active: {len(style_refs)} refs, "
                          f"combine_mode={style_combine_mode}")
                elif style_cfg is not None:
                    print(f"[Z-Image] Style transfer active: ref_k_strength={style_cfg.ref_k_strength}, "
                          f"adain_strength={style_cfg.adain_strength}, block_range={style_cfg.block_range}")

            # Offload VAE to CPU after encoding
            move_zimage_vae_to_cpu(vae)

            # ============================================================
            # Stage 3: Add Noise to Latents (Flow Matching Style)
            # ============================================================
            device = torch.device(self.device)

            # Calculate VAE scale factor for dynamic shift
            if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            else:
                vae_scale_factor = 8

            # Calculate dynamic shift
            from core.zimage_utils import calculate_shift
            image_seq_len = (init_latents.shape[2] // 2) * (init_latents.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )

            # Set scheduler parameters
            scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            # Prepare full timesteps first
            scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
            timesteps = scheduler.timesteps

            # Calculate timestep to start from (based on strength)
            init_timestep = int(num_inference_steps * denoising_strength)
            t_start = max(num_inference_steps - init_timestep, 0)

            # Get partial timesteps for inpaint
            timesteps_inpaint = timesteps[t_start:]

            print(f"[Z-Image] inpaint: Using {len(timesteps_inpaint)}/{len(timesteps)} timesteps (t_start={t_start}, strength={denoising_strength})")

            # Save original unnoised latents (for mask blending in loop)
            original_latents = init_latents.clone()

            # Add noise to init_latents at the starting timestep
            noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=torch.float32)

            # Flow Matching noise addition (apply to entire image, mask blending happens in loop)
            if hasattr(scheduler, 'add_noise'):
                print(f"[Z-Image] Using scheduler.add_noise() for noise addition")
                noised_latents = scheduler.add_noise(init_latents, noise, timesteps_inpaint[0:1])
            else:
                # Manual flow matching noise addition: x_t = (1 - t) * x_0 + t * noise
                t_normalized = timesteps_inpaint[0].item() / 1000.0
                print(f"[Z-Image] Manual flow matching noise addition: t={timesteps_inpaint[0].item():.1f}, t_norm={t_normalized:.3f}")
                noised_latents = (1.0 - t_normalized) * init_latents + t_normalized * noise

            print(f"[Z-Image] Noised latents shape: {noised_latents.shape}, dtype: {noised_latents.dtype}")

            # ============================================================
            # Stage 4: Denoising Loop with Mask Blending
            # ============================================================
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 20)
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_swap_h2d_only = params.get("block_swap_h2d_only", False)
            block_swap_ring_size = int(params.get("block_swap_ring_size", 2))

            if not enable_block_swap:
                if not is_resident(self, "transformer", _kh_model_key):
                    transformer = move_zimage_transformer_to_gpu(transformer, transformer_quantization)
                log_device_status("Ready for Z-Image denoising loop (inpaint)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })
            else:
                print("[Z-Image] Block Swap enabled - keeping Transformer on CPU for Block Swap initialization")
                from core.memory_management import create_block_offloader_for_model
                block_offloader = create_block_offloader_for_model(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    h2d_only=block_swap_h2d_only,
                    ring_size=block_swap_ring_size,
                )
                transformer._block_offloader = block_offloader
                block_offloader.prepare_block_devices_before_forward()
                log_device_status("Ready for Z-Image denoising loop (Block Swap enabled, inpaint)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })

            # Run denoising loop with mask blending
            latents = self._zimage_denoising_loop(
                transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
                height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
                generator, progress_callback, step_callback,
                init_latents=noised_latents,
                timesteps_override=timesteps_inpaint,
                mask_latent=mask_latent,
                original_latents=original_latents,
                spectrum_params=params,
                nag_negative_embeds_list=nag_negative_embeds_list,
                nag_params=params,
                style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                style_refs=style_refs, style_combine_mode=style_combine_mode,
            )

            # Offload Transformer to CPU (unless kept hot -- only possible when
            # block swap was NOT active, see keep-hot setup above)
            if not _kh_keep_transformer:
                move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 5: VAE Decode
            # ============================================================
            # NOTE: VAE was already staged to GPU once for input-image/mask
            # encoding (Stage 2) and offloaded again there unconditionally --
            # that offload is a within-generation VRAM-relief step, not the
            # keep-hot exit boundary, so it is intentionally left untouched.
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding (unless kept hot -- this is
            # VAE's true keep-hot exit point for this generation)
            if not _kh_keep_vae:
                move_zimage_vae_to_cpu(vae)

            # Clear intermediate tensors
            del prompt_embeds_list, negative_prompt_embeds_list, init_latents, original_latents, noised_latents, mask_latent, latents
            torch.cuda.empty_cache()

            log_device_status("VAE decode complete, all components offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            print("[Z-Image] inpaint generation completed")
            _kh_gen_succeeded = True

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] inpaint generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image inpaint generation failed: {str(e)}")
        finally:
            if not _kh_gen_succeeded:
                clear_resident(self)
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    discard_resident(self, "vae")
            self._zimage_cleanup(
                gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te,
                keep_transformer=_kh_keep_transformer,
                keep_vae=_kh_keep_vae,
            )

    def _zimage_encode_single(self, text_encoder, tokenizer, prompts, max_sequence_length,
                              has_fp8_weights, device):
        """Encode a list of prompt strings with the Qwen text encoder (penultimate layer),
        returning a list of per-prompt embeddings masked by their attention mask. Shared
        helper so the NAG-negative prompt uses the exact same encoder path as pos/neg."""
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            ))
        inputs = tokenizer(
            formatted, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        masks = inputs.attention_mask.to(device).bool()
        with torch.no_grad():
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    embeds = text_encoder(input_ids=input_ids, attention_mask=masks,
                                          output_hidden_states=True).hidden_states[-2]
            else:
                embeds = text_encoder(input_ids=input_ids, attention_mask=masks,
                                      output_hidden_states=True).hidden_states[-2]
        return [embeds[i][masks[i]] for i in range(len(embeds))]

    def _zimage_encode_nag_negative(self, text_encoder, tokenizer, params, prompt,
                                    max_sequence_length, text_encoder_quantization=None):
        """Encode the NAG-negative prompt (only when NAG is active) using the same encoder
        path as the positive/negative prompts. Returns a list of embeddings (one per prompt)
        or None when NAG is off. Gated by nag_enable AND nag_scale>1.
        """
        nag_enable = params.get("nag_enable", False)
        try:
            nag_scale = float(params.get("nag_scale", 1.0))
        except (TypeError, ValueError):
            nag_scale = 1.0
        if not nag_enable or abs(nag_scale - 1.0) <= 1e-5:
            return None
        nag_negative_prompt = params.get("nag_negative_prompt", "")
        if nag_negative_prompt is None:
            nag_negative_prompt = ""

        device = next(text_encoder.parameters()).device
        has_fp8_weights = False
        if text_encoder_quantization and text_encoder_quantization.startswith('fp8_'):
            for module in text_encoder.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        has_fp8_weights = True
                        break

        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
        nag_neg_list = [nag_negative_prompt for _ in prompt_list]
        print(f"[Z-Image NAG] Encoding NAG-negative prompt (scale={nag_scale})")
        return self._zimage_encode_single(
            text_encoder, tokenizer, nag_neg_list, max_sequence_length, has_fp8_weights, device
        )

    def _zimage_encode_prompt(
        self, text_encoder, tokenizer, prompt, negative_prompt,
        guidance_scale, max_sequence_length, text_encoder_quantization=None
    ):
        """
        Stage 1: Text Encoding for Z-Image
        Encodes prompt and negative prompt using Qwen text encoder.
        Text encoder is on GPU when this is called, and will be moved to CPU after.

        Returns:
            prompt_embeds_list: List of text embeddings (one per image)
            negative_prompt_embeds_list: List of negative embeddings (if CFG enabled)
            do_classifier_free_guidance: bool
        """
        _t_phase = _time.perf_counter()
        device = next(text_encoder.parameters()).device

        # Check if Text Encoder has FP8 weights
        has_fp8_weights = False
        if text_encoder_quantization and text_encoder_quantization.startswith('fp8_'):
            for module in text_encoder.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        has_fp8_weights = True
                        break

        # Format prompts using Qwen chat template
        if isinstance(prompt, str):
            prompt = [prompt]

        # CFG is enabled when guidance_scale is not 1.0 (consistent with SD/SDXL)
        # CFG=1.0 or CFG=0.0: no CFG (positive only)
        # CFG!=1.0 and CFG!=0.0: CFG enabled
        # Note: CFG=0.0 is treated as "positive only" (same as CFG=1.0)
        do_classifier_free_guidance = abs(guidance_scale - 1.0) > 1e-5 and abs(guidance_scale) > 1e-5

        print(f"[Z-Image] Encoding prompt with Text Encoder on {device}")

        formatted_prompts = []
        for p in prompt:
            messages = [{"role": "user", "content": p}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted_prompts.append(formatted_prompt)

        # Tokenize prompts
        text_inputs = tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Encode prompts (use penultimate layer output)
        # For FP8 quantized Text Encoder, use autocast for mixed precision
        with torch.no_grad():
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    prompt_embeds = text_encoder(
                        input_ids=text_input_ids,
                        attention_mask=prompt_masks,
                        output_hidden_states=True,
                    ).hidden_states[-2]
            else:
                prompt_embeds = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=prompt_masks,
                    output_hidden_states=True,
                ).hidden_states[-2]

        # Extract embeddings per prompt (masked by attention mask)
        prompt_embeds_list = []
        for i in range(len(prompt_embeds)):
            prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])

        # Encode negative prompts if CFG is enabled
        negative_prompt_embeds_list = []
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]

            neg_formatted = []
            for p in negative_prompt:
                messages = [{"role": "user", "content": p}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                neg_formatted.append(formatted_prompt)

            neg_inputs = tokenizer(
                neg_formatted,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )

            neg_input_ids = neg_inputs.input_ids.to(device)
            neg_masks = neg_inputs.attention_mask.to(device).bool()

            with torch.no_grad():
                if has_fp8_weights:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        neg_embeds = text_encoder(
                            input_ids=neg_input_ids,
                            attention_mask=neg_masks,
                            output_hidden_states=True,
                        ).hidden_states[-2]
                else:
                    neg_embeds = text_encoder(
                        input_ids=neg_input_ids,
                        attention_mask=neg_masks,
                        output_hidden_states=True,
                    ).hidden_states[-2]

            for i in range(len(neg_embeds)):
                negative_prompt_embeds_list.append(neg_embeds[i][neg_masks[i]])

        print(f"[Z-Image] Text encoding complete: {len(prompt_embeds_list)} prompts encoded")

        generation_timer.add("text_encode", _time.perf_counter() - _t_phase)
        return prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance

    def _zimage_prepare_style_reference(self, vae, style_image, height: int, width: int,
                                        device, generator=None):
        """VAE-encode a style reference image to the SAME (non-packed) latent shape used by
        the Z-Image denoising loop's own ``latents`` tensor -- ``(1, C, H_lat, W_lat)`` -- using
        the identical encode path as img2img/inpaint's own init-image encoding (encoder ->
        quant_conv -> mean/logvar reparameterize -> scaling_factor). Returns a float32 tensor
        (the "clean" x0 reference latent, re-noised per-step by ``_zimage_style_step``)."""
        import numpy as np
        from PIL import Image as PILImage

        if style_image.mode != "RGB":
            style_image = style_image.convert("RGB")
        if style_image.size != (width, height):
            style_image = style_image.resize((width, height), PILImage.Resampling.LANCZOS)
        image_array = np.array(style_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        image_tensor = image_tensor * 2.0 - 1.0
        image_tensor = image_tensor.to(device=device, dtype=vae.dtype)

        with torch.no_grad():
            h = vae.encoder(image_tensor)
            if vae.quant_conv is not None:
                h = vae.quant_conv(h)
            mean, logvar = torch.chunk(h, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            ref_x0 = mean + std * noise
            if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
                ref_x0 = ref_x0 * vae.config.scaling_factor
            else:
                ref_x0 = ref_x0 * 0.13025
            del h, mean, logvar, std
        return ref_x0.float()

    def _zimage_style_triple(self, style_dict: Dict[str, Any], vae, height: int, width: int,
                              device, generator, seed, ref_index: int = 0):
        """Build a single ``(StyleTransferConfig, ref_x0, eps_ref)`` triple from one
        ``style_transfer`` dict.

        ``axes_dims`` is filled in from ``core.models.zimage_transformer.ROPE_AXES_DIMS``,
        Z-Image's RoPE axis split (t, h, w) = (32, 48, 48) -- the SAME axis layout Krea2 uses, so
        the shared ``frequency_scale_vector`` real-valued per-head-dim scale curve applies
        UNCHANGED even though Z-Image's RoPE is implemented via complex ``view_as_complex``
        multiplication rather than Krea2's real cos/sin pairs: ``apply_rotary_emb`` here rotates
        each adjacent-slot ``(2i, 2i+1)`` real pair by one complex frequency, and a uniform real
        scale applied to a real-valued tensor AFTER that rotation is just a per-channel magnitude
        scale -- it doesn't care whether the rotation was expressed as a 2x2 real matrix or a
        complex multiply, only that both slots of a rotary pair get the SAME scale. The
        ``repeat_interleave(2)`` pairing in ``frequency_scale_vector`` exactly matches Z-Image's
        own ``view_as_complex(x.reshape(..., -1, 2))`` pairing (adjacent real slots), so no
        ones-vector fallback is needed here (unlike architectures whose RoPE axis order can't be
        cleanly recovered).

        ``ref_index`` decorrelates the fixed re-noising noise tensor across multiple simultaneous
        references (each ref would otherwise draw the EXACT same ``eps_ref`` from the
        ``seed+991`` offset, since that offset does not depend on which reference is being
        prepared). ``ref_index=0`` (the default, used by the single-ref path) reproduces the
        pre-multi-ref ``seed+991`` offset exactly. The VAE reparameterization noise inside
        ``_zimage_prepare_style_reference`` needs no separate decorrelation: it consumes the
        SAME shared ``generator`` object passed in by the caller, whose state naturally advances
        (and thus decorrelates) across successive per-ref calls."""
        from diffusers.utils.torch_utils import randn_tensor
        from core.inference.reference_style import style_config_from_dict
        from core.models.zimage_transformer import ROPE_AXES_DIMS

        cfg = style_config_from_dict(style_dict)
        cfg.axes_dims = tuple(ROPE_AXES_DIMS)

        ref_x0 = self._zimage_prepare_style_reference(
            vae, style_dict["image"], height, width, device, generator=generator
        )

        ref_seed = None if seed is None or seed < 0 else (int(seed) + 991 + ref_index) % (2 ** 32)
        ref_generator = torch.Generator(device=device).manual_seed(ref_seed) if ref_seed is not None else None
        eps_ref = randn_tensor(ref_x0.shape, generator=ref_generator, device=device, dtype=ref_x0.dtype)
        return cfg, ref_x0, eps_ref

    def _zimage_style_config(self, params: Dict[str, Any], vae, height: int, width: int,
                              device, generator=None):
        """Build a ``(StyleTransferConfig, ref_x0, eps_ref)`` triple from
        ``params["style_transfer"]`` (assembled by
        ``generation_utils.process_controlnet_configs``), or ``(None, None, None)`` when no
        style reference is attached (byte-identical default path -- the caller never installs
        ``transformer._style_ctx`` in that case). Single-reference path, BYTE-IDENTICAL to the
        pre-multi-ref implementation (delegates to ``_zimage_style_triple`` with ``ref_index=0``,
        which reproduces the original ``seed+991`` re-noising offset exactly)."""
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None

        seed = params.get("seed", -1)
        return self._zimage_style_triple(style_dict, vae, height, width, device, generator, seed, ref_index=0)

    def _zimage_style_configs(self, params: Dict[str, Any], vae, height: int, width: int,
                               device, generator=None):
        """Build the full style-transfer configuration for Z-Image generation, covering both
        the single-reference path (legacy ``(style_cfg, style_ref_x0, style_eps_ref)`` triple,
        exactly as ``_zimage_style_config`` would return) and the multi-reference path
        (``style_refs``, a list of per-ref triples, populated ONLY when
        ``params["style_transfers"]`` has more than one entry). A single-entry
        ``style_transfers`` list is intentionally routed through the single-ref triple instead
        (``style_refs`` stays ``None``), so the pre-multi-ref code path executes
        byte-identically end to end.

        Returns ``(style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode)``.
        """
        style_list = params.get("style_transfers")
        if style_list and len(style_list) > 1:
            seed = params.get("seed", -1)
            combine_mode = str(params.get("style_combine_mode", "stack") or "stack")
            refs = []
            for idx, style_dict in enumerate(style_list):
                if not style_dict or not style_dict.get("image"):
                    continue
                refs.append(self._zimage_style_triple(
                    style_dict, vae, height, width, device, generator, seed, ref_index=idx
                ))
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                cfg, x0, eps = refs[0]
                return cfg, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._zimage_style_config(
            params, vae, height, width, device, generator=generator
        )
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

    def _zimage_style_step(
        self, transformer, style_cfg, style_ref_x0, style_eps_ref,
        t, latents, prompt_embeds_list, negative_prompt_embeds_list,
        apply_cfg: bool, guidance_scale: float, has_fp8_weights: bool,
        step_idx: int, num_inference_steps: int,
        style_refs=None, style_combine_mode: str = "stack",
    ):
        """One style-active denoise step for Z-Image: bypasses the batched
        ``[negative; positive(; nag_negative)]`` CFG fast path entirely for this step (mirrors
        FLUX.2's ``_flux2_style_step``, the closest architectural precedent for a training-free
        style-transfer two-pass loop). A REF capture forward (the style reference re-noised to
        the CURRENT sigma, using the SAME ``x_t = (1-sigma)*x0 + sigma*eps`` convention this
        loop's own img2img/inpaint noising uses, ``sigma = t/1000``) stashes post-RoPE
        image-token (PREFIX) Q/K/V per joint block; the COND forward then reads/injects them.
        The UNCOND forward (when CFG is active) is ALWAYS run with the style context disarmed
        (untouched), matching the Krea2/FLUX.2 wiring. NAG, NegPip and FBCache are bypassed for
        this step -- mutually exclusive with style transfer (same reasoning as FLUX.2: FBCache
        additionally needs full-generation-level exclusion since a cache hit skips layers[1:]
        and would desync the per-block style store across steps -- enforced by the caller,
        ``_zimage_denoising_loop``, disabling FBCache for the whole generation whenever style
        transfer is requested). Block Swap still applies unchanged since the transformer's own
        ``forward()`` layer loop (and its ``_block_offloader`` calls) runs exactly the same way
        for these calls as for the normal batched path.

        ``style_refs`` (optional, multi-reference): a list of ``(StyleTransferConfig, ref_x0,
        ref_eps)`` triples, one per reference image, each keeping its OWN config (block_range,
        strengths, freq curve, step gating). Only consulted when it has 2+ entries -- callers
        route ``len(style_refs) <= 1`` through the ``style_cfg``/``style_ref_x0``/
        ``style_eps_ref`` single-ref path below instead (untouched), so single-ref behavior stays
        byte-identical. ``style_combine_mode`` selects how the N refs combine ("stack" or
        "common_concept", see ``core.inference.reference_style.inject_kv_multi``).

        Returns the final signed noise_pred (``-model_output.squeeze(2)``, same convention the
        non-style branch produces) ready for ``scheduler.step``.
        """
        from core.inference.reference_style import StyleContext

        batch_size = len(prompt_embeds_list)
        sigma_now = float(t.item()) / 1000.0

        input_dtype = torch.bfloat16 if has_fp8_weights else next(transformer.parameters()).dtype
        timestep = t.expand(latents.shape[0]).to(input_dtype)
        timestep = (1000 - timestep) / 1000

        latents_list = list(latents.to(input_dtype).unsqueeze(2).unbind(dim=0))

        def _forward(x_list, cap_list):
            with torch.no_grad():
                if has_fp8_weights:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        return transformer(x_list, timestep, cap_list)[0]
                return transformer(x_list, timestep, cap_list)[0]

        try:
            if style_refs is not None and len(style_refs) > 1:
                # Multi-reference (N>1): one capture forward PER reference (each skipped if not
                # step-active this step, mirroring the single-ref ``is_step_active`` gate applied
                # per-ref instead of globally), then a single StyleContext holding the full
                # ``refs`` list for the conditional forward. len(style_refs) <= 1 is NEVER routed
                # here by the caller (see docstring) so this branch does not affect single-ref
                # behavior at all.
                active_refs = []
                for cfg_i, x0_i, eps_i in style_refs:
                    if not cfg_i.is_step_active(step_idx, num_inference_steps):
                        continue
                    ref_t_i = (1.0 - sigma_now) * x0_i + sigma_now * eps_i
                    progress_i = cfg_i.step_progress(step_idx, num_inference_steps)
                    # Per-item lists: the reference image is IDENTICAL (same ref) for every batch
                    # item, only the caption differs -- exactly how the target's own ``latents``
                    # are shared across the pos/neg/nag groups in the normal batched path.
                    ref_item_i = ref_t_i[0].to(input_dtype).unsqueeze(1)  # (C, 1, H, W)
                    ref_list_i = [ref_item_i for _ in range(batch_size)]

                    capture_ctx_i = StyleContext(mode="capture", config=cfg_i, progress=progress_i)
                    transformer._style_ctx = capture_ctx_i
                    _forward(ref_list_i, prompt_embeds_list)
                    active_refs.append((capture_ctx_i.store, cfg_i))

                if active_refs:
                    overall_progress = active_refs[0][1].step_progress(step_idx, num_inference_steps)
                    inject_ctx = StyleContext(
                        mode="inject", config=active_refs[0][1], refs=active_refs,
                        combine_mode=style_combine_mode, progress=overall_progress,
                    )
                    transformer._style_ctx = inject_ctx
                    cond_out = _forward(latents_list, prompt_embeds_list)
                else:
                    cond_out = _forward(latents_list, prompt_embeds_list)
            else:
                ref_t = (1.0 - sigma_now) * style_ref_x0 + sigma_now * style_eps_ref
                progress = style_cfg.step_progress(step_idx, num_inference_steps)

                # Per-item lists: the reference image is IDENTICAL (same style ref) for every
                # batch item, only the caption differs -- exactly how the target's own
                # ``latents`` are shared across the pos/neg/nag groups in the normal batched path
                # (repeat, not distinct content).
                ref_item = ref_t[0].to(input_dtype).unsqueeze(1)  # (C, 1, H, W)
                ref_list = [ref_item for _ in range(batch_size)]

                capture_ctx = StyleContext(mode="capture", config=style_cfg, progress=progress)
                transformer._style_ctx = capture_ctx
                _forward(ref_list, prompt_embeds_list)

                inject_ctx = StyleContext(mode="inject", config=style_cfg, store=capture_ctx.store, progress=progress)
                transformer._style_ctx = inject_ctx
                cond_out = _forward(latents_list, prompt_embeds_list)
        finally:
            transformer._style_ctx = None
            # Defense-in-depth: if a style forward raised after stamping but before
            # the transformer's own end-of-forward clear, the per-layer attention
            # _style_ctx/block_idx would be left stale. With transformer._style_ctx
            # now None a subsequent forward skips both stamping AND clearing, so a
            # stale per-layer ctx could otherwise fire on a later (even style-off)
            # generation. Clear the per-layer attention ctx here too.
            for _layer in getattr(transformer, "layers", []):
                _attn = getattr(_layer, "attention", None)
                if _attn is not None:
                    _attn._style_ctx = None
                    _attn.block_idx = None

        # --- CFG-decoupled style guidance (single-ref only) ---
        # Disabled by default (style_guidance_scale None/<=0, or the multi-ref
        # path -- style_refs with 2+ entries is out of scope here): this block is
        # skipped entirely and cond_out stays exactly the styled cond prediction
        # above -- byte-identical to before this feature (zero extra forwards).
        # Enabled (>0), single-ref, AND CFG is actually being applied this step
        # (apply_cfg -- with no uncond pred there is nothing to decouple style
        # from): run one more forward -- the SAME latents_list/prompt_embeds_list
        # as the styled cond_out above -- but with transformer._style_ctx already
        # cleared by the finally block above (and the per-layer ctx/block_idx
        # reset alongside it), so this is a plain no-style cond forward (cond_ns).
        # Z-Image's own combine below is:
        #   noise_pred = neg + guidance_scale * (pos - neg)
        # Rewriting the cond term to pos' = cond_ns + (lambda/guidance_scale) *
        # (cond_s - cond_ns) makes that SAME combine reproduce the
        # style-guidance target:
        #   neg + guidance_scale*(pos' - neg)
        # = neg + guidance_scale*(cond_ns-neg) + guidance_scale*(lambda/guidance_scale)*(cond_s-cond_ns)
        # = neg + guidance_scale*(cond_ns - neg) + lambda*(cond_s - cond_ns)
        # -- prompt guidance stays at guidance_scale (this step's, already resolved
        # by the caller's CFG-truncation schedule before this function was
        # called), style strength is lambda, decoupled from guidance_scale, exactly
        # like the SDXL/Anima prototypes. FBCache is already forced off for the
        # whole generation whenever style transfer is active (see
        # ``_zimage_denoising_loop``), so this extra forward cannot desync a
        # cache. Guarded on guidance_scale > 1e-6 (else this block is skipped and
        # cond_out is used as-is, i.e. the plain styled-cond pass).
        style_guidance_active = (
            style_cfg is not None
            and not (style_refs is not None and len(style_refs) > 1)
            and style_cfg.style_guidance_scale is not None
            and style_cfg.style_guidance_scale > 0
            and apply_cfg
            and guidance_scale > 1e-6
        )
        cond_ns_out = None
        if style_guidance_active:
            cond_ns_out = _forward(latents_list, prompt_embeds_list)

        if apply_cfg:
            uncond_out = _forward(latents_list, negative_prompt_embeds_list)
            combined = []
            for j in range(batch_size):
                neg = uncond_out[j].float()
                pos = cond_out[j].float()
                if style_guidance_active:
                    pos_ns = cond_ns_out[j].float()
                    lam = style_cfg.style_guidance_scale
                    pos = pos_ns + (lam / guidance_scale) * (pos - pos_ns)
                combined.append(neg + guidance_scale * (pos - neg))
            noise_pred = torch.stack(combined, dim=0)
        else:
            noise_pred = torch.stack([o.float() for o in cond_out], dim=0)

        return -noise_pred.squeeze(2)

    def _zimage_denoising_loop(
        self, transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
        height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
        generator, progress_callback, step_callback,
        init_latents: Optional[torch.Tensor] = None,
        timesteps_override: Optional[torch.Tensor] = None,
        mask_latent: Optional[torch.Tensor] = None,
        original_latents: Optional[torch.Tensor] = None,
        spectrum_params: Optional[Dict[str, Any]] = None,
        nag_negative_embeds_list: Optional[List[torch.Tensor]] = None,
        nag_params: Optional[Dict[str, Any]] = None,
        style_cfg=None,
        style_ref_x0: Optional[torch.Tensor] = None,
        style_eps_ref: Optional[torch.Tensor] = None,
        style_refs=None,
        style_combine_mode: str = "stack",
    ):
        """
        Stage 2: Denoising Loop for Z-Image
        Runs the transformer denoising loop with flow matching.
        Transformer is on GPU when this is called, and will be moved to CPU after.

        Args:
            init_latents: Optional initial latents for img2img/inpaint (already noised)
            timesteps_override: Optional timesteps for img2img/inpaint (partial timesteps from t_start)
            mask_latent: Optional mask for inpainting (1 = inpaint, 0 = keep original)
            original_latents: Optional original unnoised latents for inpaint blending
            style_refs: Optional multi-reference (N>1) list of ``(StyleTransferConfig, ref_x0,
                ref_eps)`` triples -- see ``_zimage_style_step``'s docstring. Only consulted when
                it has 2+ entries; ``style_cfg``/``style_ref_x0``/``style_eps_ref`` drive the
                (untouched) single-ref path otherwise.
            style_combine_mode: "stack" or "common_concept", see
                ``core.inference.reference_style.inject_kv_multi``.

        Returns:
            latents: Denoised latents (torch.Tensor)
        """
        _t_phase = _time.perf_counter()
        # Import calculate_shift from local zimage_utils (with fallback)
        try:
            from core.zimage_utils import calculate_shift
        except ImportError:
            # Fallback implementation if zimage_utils is not available
            def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
                m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                b = base_shift - m * base_seq_len
                mu = image_seq_len * m + b
                return mu

        # Use self.device instead of transformer device (Block Swap may have weights on CPU)
        device = torch.device(self.device)

        print(f"[Z-Image] Starting denoising loop on {device}")

        # Calculate VAE scale factor
        vae = self.zimage_components["vae"]
        if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            vae_scale_factor = 8
        vae_scale = vae_scale_factor * 2

        # Calculate latent dimensions
        height_latent = 2 * (int(height) // vae_scale)
        width_latent = 2 * (int(width) // vae_scale)
        batch_size = len(prompt_embeds_list)
        shape = (batch_size, transformer.in_channels, height_latent, width_latent)

        # Initialize latents (use init_latents if provided for img2img, otherwise random for txt2img)
        if init_latents is not None:
            latents = init_latents.to(device=device, dtype=torch.float32)
            print(f"[Z-Image] Starting from noised input image latents (img2img)")
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            print(f"[Z-Image] Starting from random latents (txt2img)")

        # Calculate dynamic shift for flow matching
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # Use local calculate_shift implementation (from zimage_utils.py or fallback)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )

        # Set scheduler parameters
        scheduler.sigma_min = 0.0

        # Prepare timesteps (use override if provided for img2img, otherwise calculate normally)
        if timesteps_override is not None:
            timesteps = timesteps_override
            print(f"[Z-Image] Using {len(timesteps)} timesteps for img2img (strength-based, from t_start)")
        else:
            # Only FlowMatchEulerDiscreteScheduler supports 'mu' parameter
            # FlowMatchHeunDiscreteScheduler does not support it
            if hasattr(scheduler, '__class__') and 'Euler' in scheduler.__class__.__name__:
                scheduler_kwargs = {"mu": mu}
                scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
                print(f"[Z-Image] Denoising loop: {num_inference_steps} steps requested, {len(scheduler.timesteps)} timesteps generated, shift={mu:.3f}")
            else:
                # Heun or other schedulers: no mu parameter
                scheduler.set_timesteps(num_inference_steps, device=device)
                print(f"[Z-Image] Denoising loop: {num_inference_steps} steps requested, {len(scheduler.timesteps)} timesteps generated (scheduler: {scheduler.__class__.__name__})")
            timesteps = scheduler.timesteps

        # Detect FP8 quantization (check once before loop)
        has_fp8_weights = False
        for module in transformer.modules():
            if isinstance(module, torch.nn.Linear):
                if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    has_fp8_weights = True
                    print(f"[Z-Image] Detected FP8 quantized Transformer (dtype: {module.weight.dtype})")
                    print(f"[Z-Image] Will use autocast for mixed precision inference")
                    break
        if not has_fp8_weights:
            print(f"[Z-Image] Transformer not quantized (BF16 inference)")

        # Spectrum output-mode acceleration: forecast the per-step (post-CFG) velocity
        # on skip steps to avoid the transformer evaluation. Output mode only (block mode
        # is U-Net-specific). Disabled for too-few steps.
        spectrum = None
        if spectrum_params is not None:
            from core.inference.spectrum_forecaster import build_output_forecaster
            spectrum = build_output_forecaster(spectrum_params, len(timesteps), label="Z-Image")

        # Training-free reference-style transfer: active only when a style reference was
        # attached (style_cfg is not None -- built by ``_zimage_style_config``, byte-identical
        # no-op otherwise). ``style_multi_active`` covers the multi-reference (N>1) path built by
        # ``_zimage_style_configs`` -- ``style_cfg``/``style_refs`` are mutually exclusive (never
        # both set), so this OR simply widens the "style transfer is on" gate to cover both.
        style_active_single = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
        style_multi_active = style_refs is not None and len(style_refs) > 1
        style_active = style_active_single or style_multi_active

        # First Block Cache (FBCache): dynamic per-step residual reuse. Mutually exclusive with:
        #   (a) Spectrum -- both target the same trajectory redundancy; combining compounds error.
        #   (b) Block Swap -- a cache hit skips layers[1:], which would desync the block-swap
        #       rotation (the offloader expects every layer to run each step).
        #   (c) Style transfer -- a cache hit skips layers[1:], which would desync the per-block
        #       style store (the style hook needs EVERY joint layer to run its capture+inject pair
        #       every active step, and a cache hit would silently reuse a stale non-style residual).
        # FBCache runs only when ALL THREE are off. CFG is BATCHED here (one transformer forward per
        # step over the whole [neg; pos(; nag)] batch), so a SINGLE FirstBlockCache instance is
        # correct: the first-block residual and cached full residual span the entire batch and the
        # same batch layout recurs every step.
        from core.inference.fbcache import build_fbcache, fbcache_active
        fbcache = None
        if spectrum_params is not None and fbcache_active(spectrum_params):
            _fb_bs = bool(spectrum_params.get("enable_block_swap", False)) and \
                int(spectrum_params.get("blocks_to_swap", 0)) > 0
            if spectrum is not None:
                print("[FBCache] Z-Image disabled: Spectrum is enabled (same redundancy target)")
            elif _fb_bs:
                print("[FBCache] Z-Image disabled: Block Swap is enabled (layer skip desyncs rotation)")
            elif style_active:
                print("[FBCache] Z-Image disabled: Style transfer is enabled (layer skip desyncs the per-block style store)")
            else:
                fbcache = build_fbcache(spectrum_params, label="Z-Image")
        # Ensure no stale cache leaks into this forward stream.
        if hasattr(transformer, "_fbcache"):
            transformer._fbcache = None

        # NAG (Normalized Attention Guidance) setup. Active only when a nag-negative
        # embedding list was provided (nag_enable AND nag_scale>1, resolved at encode time).
        # When inactive, nag_negative_embeds_list is None and the loop is byte-identical.
        from core.inference.nag_dit import nag_active as _nag_active_fn
        _nag_p = nag_params or {}
        try:
            nag_scale = float(_nag_p.get("nag_scale", 1.0))
        except (TypeError, ValueError):
            nag_scale = 1.0
        nag_on = _nag_active_fn(_nag_p.get("nag_enable", False), nag_scale,
                                nag_negative_embeds_list)
        if nag_on:
            nag_tau = float(_nag_p.get("nag_tau", 2.5))
            nag_alpha = float(_nag_p.get("nag_alpha", 0.25))
            print(f"[Z-Image NAG] Active: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}")
        # Ensure any stale request is cleared before the loop.
        if hasattr(transformer, "_nag_request"):
            transformer._nag_request = None

        # NegPip (signed-value attention) setup. AUTO-ACTIVATES (no toggle) only when a prompt
        # carries a NEGATIVE emphasis weight (e.g. "(worst quality:-1)"). When no negative weight
        # is present, negpip_on is False and _negpip_request stays None -> the transformer forward
        # is byte-identical (positive-only default path unchanged). Per-context signed weight
        # rows are aligned to each caption's real token count (the encoder's masked length, i.e.
        # the length of the corresponding embeds row), so a negatively-weighted token's V scale
        # lands on exactly its caption token. Composes with NAG (image-prefix output guidance) and
        # Spectrum (post-CFG forecast: NegPip only touches evaluated steps).
        negpip_on = False
        negpip_pos_rows = negpip_neg_rows = negpip_nag_rows = None
        try:
            from core.prompts.prompt_parser import prompt_has_negative_weight
            from core.inference.negpip_zimage import build_zimage_caption_weights
            _np = nag_params or {}
            _pos_prompt = _np.get("prompt", "")
            _neg_prompt = _np.get("negative_prompt", "") or ""
            _nag_neg_prompt = _np.get("nag_negative_prompt", "") or ""
            _pos_list = [_pos_prompt] if isinstance(_pos_prompt, str) else list(_pos_prompt)
            _has_neg = (prompt_has_negative_weight(_pos_prompt)
                        or prompt_has_negative_weight(_neg_prompt)
                        or prompt_has_negative_weight(_nag_neg_prompt))
            if _has_neg:
                _tok = self.zimage_components["tokenizer"]
                _wdtype = torch.float32

                def _rows_for(prompt_str, embeds_list):
                    # One weight row per caption item, length == that caption's real token count
                    # (== embeds row length == encoder masked length). Same prompt string per item
                    # in this pipeline (batch shares one prompt), so build once per item length.
                    rows = []
                    for e in (embeds_list or []):
                        rows.append(build_zimage_caption_weights(
                            prompt_str, _tok, e.shape[0], e.device, _wdtype))
                    return rows

                negpip_pos_rows = _rows_for(_pos_prompt if isinstance(_pos_prompt, str)
                                            else (_pos_list[0] if _pos_list else ""),
                                            prompt_embeds_list)
                if do_classifier_free_guidance and negative_prompt_embeds_list:
                    negpip_neg_rows = _rows_for(_neg_prompt, negative_prompt_embeds_list)
                if nag_negative_embeds_list:
                    negpip_nag_rows = _rows_for(_nag_neg_prompt, nag_negative_embeds_list)
                negpip_on = True
                print(f"[Z-Image NegPip] Active: negative emphasis weight detected "
                      f"(pos={negpip_pos_rows is not None}, neg={negpip_neg_rows is not None}, "
                      f"nag={negpip_nag_rows is not None})")
        except Exception as _np_err:
            print(f"[Z-Image NegPip] Setup skipped ({_np_err}); positive-only path unchanged")
            negpip_on = False
        if hasattr(transformer, "_negpip_request"):
            transformer._negpip_request = None

        # Attach the FBCache to the transformer for the whole loop (None -> forward unchanged).
        if fbcache is not None:
            transformer._fbcache = fbcache

        # Denoising loop with progress callback
        # Note: Heun scheduler generates 2*steps-1 timesteps (39 for 20 steps)
        # We normalize progress to user-requested num_inference_steps for UI consistency
        for i, t in enumerate(timesteps):
            if self.cancel_requested:
                print("[Z-Image] Generation cancelled by user")
                raise RuntimeError("Generation cancelled by user")
            # Skip last step if t=0 (flow matching termination)
            if t == 0 and i == len(timesteps) - 1:
                print(f"[Z-Image] Step {i+1}/{len(timesteps)} | t={t.item():.2f} | Skipping last step (flow matching termination)")
                continue

            # Calculate normalized step for progress bar (map timestep index to user-requested steps)
            # For Heun: len(timesteps)=39, num_inference_steps=20 → normalize i to 0-19 range
            normalized_step = int((i / len(timesteps)) * num_inference_steps)

            # step_callback fires before the model forward so step-range LoRA
            # hooks see the next step index correctly. progress_callback (which
            # carries the preview payload) is deferred until after the forward
            # so we can hand it pred_x0 in addition to the raw latents.
            if step_callback:
                step_callback(normalized_step, num_inference_steps)

            # Normalize timestep to [0, 1]
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            t_norm = timestep[0].item()

            # CFG truncation logic (disable CFG after certain timestep)
            # Default value from Z-Image: DEFAULT_CFG_TRUNCATION = 1.0
            current_guidance_scale = guidance_scale
            cfg_truncation = 1.0  # Z-Image default
            if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
                if t_norm > cfg_truncation:
                    current_guidance_scale = 1.0  # Set to 1.0 (no CFG) instead of 0.0

            # Apply CFG when guidance_scale is not 1.0 (consistent with SD/SDXL)
            apply_cfg = do_classifier_free_guidance and abs(current_guidance_scale - 1.0) > 1e-5

            # Training-free reference-style transfer: bypasses the batched
            # [negative;positive(;nag_negative)] CFG fast path (and Spectrum/NAG/NegPip/FBCache)
            # entirely for this step -- see ``_zimage_style_step``'s docstring. Takes priority over
            # Spectrum's skip-step forecast since a forecast has no attention to inject style into.
            # Multi-reference (N>1): active this step when ANY of the N refs is step-active (each
            # ref's own ``is_step_active`` gate, mirroring the single-ref check applied per-ref
            # instead of globally) -- when none are active, this step falls through to the normal
            # batched path below exactly like the single-ref "step outside range" case does.
            if style_multi_active:
                style_active_step = any(
                    cfg_i.is_step_active(normalized_step, num_inference_steps) for _s, cfg_i in style_refs
                )
            elif style_active_single:
                style_active_step = style_cfg.is_step_active(normalized_step, num_inference_steps)
            else:
                style_active_step = False

            # Spectrum: forecast the post-CFG velocity on skip steps (skip transformer + CFG)
            spectrum_skip = not style_active_step and spectrum is not None and not spectrum.is_anchor(i)
            if style_active_step:
                noise_pred = self._zimage_style_step(
                    transformer, style_cfg, style_ref_x0, style_eps_ref,
                    t, latents, prompt_embeds_list, negative_prompt_embeds_list,
                    apply_cfg, current_guidance_scale, has_fp8_weights,
                    normalized_step, num_inference_steps,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
                if spectrum is not None:
                    spectrum.record(i, noise_pred)
            elif spectrum_skip:
                noise_pred = spectrum.forecast(i)
            else:
                # Prepare model input (concat positive + negative if CFG)
                # Note: For FP8 quantization, keep input in BF16/FP16, don't convert to FP8
                if has_fp8_weights:
                    # FP8 quantized: use BF16 input (autocast will handle conversion)
                    input_dtype = torch.bfloat16
                else:
                    # Normal case: use transformer's dtype
                    transformer_dtype = next(transformer.parameters()).dtype
                    input_dtype = transformer_dtype

                # NAG batch layout: when active, append the nag-negative caption group so
                # its captions evolve through the blocks. Image latents are duplicated for
                # that group (identical image, different caption -> guidance driver).
                #   CFG on  -> groups [neg, pos, nag_neg]   (repeat x3, NAG on cond only)
                #   CFG off -> groups [pos, nag_neg]        (repeat x2, NAG on pos)
                nag_this_step = nag_on
                # NegPip: assemble the signed weight rows in the SAME batch order as
                # prompt_embeds_model_input (one row per caption item). Each context uses its own
                # signed weights: pos subtracts, neg (uncond) re-affirms via double-negative.
                negpip_rows_this_step = None
                if apply_cfg:
                    if nag_this_step:
                        latent_model_input = latents.to(input_dtype).repeat(3, 1, 1, 1)
                        prompt_embeds_model_input = (
                            negative_prompt_embeds_list + prompt_embeds_list
                            + nag_negative_embeds_list
                        )
                        timestep_model_input = timestep.repeat(3)
                        if negpip_on:
                            negpip_rows_this_step = (
                                (negpip_neg_rows or [None] * len(negative_prompt_embeds_list))
                                + (negpip_pos_rows or [None] * len(prompt_embeds_list))
                                + (negpip_nag_rows or [None] * len(nag_negative_embeds_list))
                            )
                    else:
                        latent_model_input = latents.to(input_dtype).repeat(2, 1, 1, 1)
                        # CFG input order: [negative, positive] (consistent with SD/SDXL)
                        prompt_embeds_model_input = negative_prompt_embeds_list + prompt_embeds_list
                        timestep_model_input = timestep.repeat(2)
                        if negpip_on:
                            negpip_rows_this_step = (
                                (negpip_neg_rows or [None] * len(negative_prompt_embeds_list))
                                + (negpip_pos_rows or [None] * len(prompt_embeds_list))
                            )
                else:
                    if nag_this_step:
                        latent_model_input = latents.to(input_dtype).repeat(2, 1, 1, 1)
                        prompt_embeds_model_input = prompt_embeds_list + nag_negative_embeds_list
                        timestep_model_input = timestep.repeat(2)
                        if negpip_on:
                            negpip_rows_this_step = (
                                (negpip_pos_rows or [None] * len(prompt_embeds_list))
                                + (negpip_nag_rows or [None] * len(nag_negative_embeds_list))
                            )
                    else:
                        latent_model_input = latents.to(input_dtype)
                        prompt_embeds_model_input = prompt_embeds_list
                        timestep_model_input = timestep
                        if negpip_on:
                            negpip_rows_this_step = (
                                negpip_pos_rows or [None] * len(prompt_embeds_list)
                            )

                # Add channel dimension and split into list
                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                # Install the NAG request so ZImageAttention applies guidance in the joint
                # layers only; the transformer forward converts it into the live context and
                # clears it. We also clear it here in a finally as a safety net.
                if nag_this_step:
                    transformer._nag_request = {
                        "group_size": batch_size,
                        "has_cfg": bool(apply_cfg),
                        "scale": nag_scale,
                        "tau": nag_tau,
                        "alpha": nag_alpha,
                    }

                # NegPip: install the signed weight rows (batch order matches the caption list
                # built above). Converted into the live NegPipContext by the transformer forward,
                # which also clears it; we clear again in finally as a safety net.
                if negpip_on and negpip_rows_this_step is not None:
                    transformer._negpip_request = {"weight_rows": negpip_rows_this_step}

                # FBCache: hand the transformer the current step index so its forward can gate
                # warmup and index the per-step decision (mirrors how _block_offloader is attached).
                if fbcache is not None:
                    transformer._fbcache_step = i

                # Transformer forward pass
                # For FP8 quantized models, use autocast to handle mixed precision
                try:
                    with torch.no_grad():
                        if has_fp8_weights:
                            # FP8: use autocast for automatic mixed precision
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                model_out_list = transformer(
                                    latent_model_input_list,
                                    timestep_model_input,
                                    prompt_embeds_model_input,
                                )[0]
                        else:
                            # Normal: no autocast needed
                            model_out_list = transformer(
                                latent_model_input_list,
                                timestep_model_input,
                                prompt_embeds_model_input,
                            )[0]
                finally:
                    if nag_this_step:
                        transformer._nag_request = None
                        from core.models.zimage_transformer import ZImageAttention
                        ZImageAttention._nag_ctx = None
                    if negpip_on:
                        transformer._negpip_request = None
                        from core.models.zimage_transformer import ZImageAttention
                        ZImageAttention._negpip_ctx = None

                # Apply CFG if enabled
                if apply_cfg:
                    # CFG output order matches input: [negative, positive]
                    neg_out = model_out_list[:batch_size]  # negative (uncond)
                    pos_out = model_out_list[batch_size:]  # positive (cond)
                    noise_pred = []
                    for j in range(batch_size):
                        neg = neg_out[j].float()
                        pos = pos_out[j].float()
                        # Standard CFG formula (consistent with SD/SDXL)
                        # pred = uncond + guidance_scale * (cond - uncond)
                        pred = neg + current_guidance_scale * (pos - neg)
                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    # No CFG. When NAG is on the batch is [pos_nag, nag_neg]; the pos group
                    # is already NAG-guided, so keep only the first batch_size outputs.
                    out_slice = model_out_list[:batch_size] if nag_this_step else model_out_list
                    noise_pred = torch.stack([out.float() for out in out_slice], dim=0)

                # Scheduler step (flow matching with stochastic_sampling if enabled)
                noise_pred = -noise_pred.squeeze(2)
                if spectrum is not None:
                    spectrum.record(i, noise_pred)

            # Predicted clean latent for preview: x_t = (1-σ)·x_0 + σ·noise,
            # v = noise - x_0, so x_0 = x_t - σ·v. σ is t_norm (the timestep
            # already normalised to [0, 1] above). Note that the sign flip on
            # noise_pred above gives us the standard-direction velocity, so the
            # straight subtraction is the right formula here.
            try:
                preview_pred_x0 = (latents.float() - t_norm * noise_pred.float()).to(latents.dtype)
            except Exception:
                preview_pred_x0 = None

            if progress_callback:
                try:
                    progress_callback(normalized_step, num_inference_steps, latents,
                                       None, preview_pred_x0)
                except Exception as e:
                    print(f"[Z-Image] Progress callback error: {e}")

            latents = scheduler.step(
                noise_pred.to(torch.float32), t, latents,
                return_dict=False
            )[0]

            # Inpaint mask blending: blend denoised latents with noised original latents
            if mask_latent is not None and original_latents is not None:
                # For inpaint, non-masked area should also be noised at current timestep
                # then blended with denoised latents
                original_latents_device = original_latents.to(device=latents.device, dtype=latents.dtype)
                mask_latent_device = mask_latent.to(device=latents.device, dtype=latents.dtype)

                # Add noise to original latents at current timestep
                # This ensures non-masked area follows the same noise schedule
                if i < len(timesteps) - 1:  # Not the last step
                    next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor([0.0], device=device)
                    # Generate noise for original latents
                    noise_for_original = torch.randn_like(original_latents_device)

                    # Flow Matching: add noise at next timestep level
                    t_next_normalized = next_t.item() / 1000.0
                    noised_original = (1.0 - t_next_normalized) * original_latents_device + t_next_normalized * noise_for_original
                else:
                    # Last step: use clean original latents
                    noised_original = original_latents_device

                # Blend: mask * denoised + (1 - mask) * noised_original
                latents = mask_latent_device * latents + (1.0 - mask_latent_device) * noised_original

            if normalized_step % 5 == 0 or normalized_step == num_inference_steps - 1:
                print(f"[Z-Image] Step {normalized_step+1}/{num_inference_steps} | t={t_norm:.3f} | CFG={current_guidance_scale:.1f}")

        print(f"[Z-Image] Denoising loop complete")

        # FBCache cleanup: detach the cache + step so it never leaks into a later forward
        # (e.g. VAE-adjacent or a subsequent generation reusing this transformer instance).
        if fbcache is not None:
            print(f"[FBCache] Z-Image summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
        if hasattr(transformer, "_fbcache"):
            transformer._fbcache = None
        if hasattr(transformer, "_fbcache_step"):
            transformer._fbcache_step = None

        generation_timer.add("denoise", _time.perf_counter() - _t_phase)
        return latents

    def _zimage_decode_latents(self, vae, latents):
        """
        Stage 3: VAE Decode for Z-Image
        Decodes latents to images using VAE.
        VAE is on GPU when this is called, and will be moved to CPU after.

        Returns:
            images: List of PIL images
        """
        _t_phase = _time.perf_counter()
        device = next(vae.parameters()).device

        print(f"[Z-Image] Decoding latents with VAE on {device}")
        self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))

        # Apply VAE scaling and shift
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        latents = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor

        # Decode latents
        with torch.no_grad():
            image = vae.decode(latents, return_dict=False)[0]

        # Convert to PIL images
        from PIL import Image
        image = (image / 2 + 0.5).clamp(0, 1)
        _cf = getattr(self, "_color_flatten_strength", 0)
        if _cf and _cf > 0:
            from core.inference.color_flatten import flatten_chroma
            image = flatten_chroma(image, _cf)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        images = [Image.fromarray(img) for img in image]

        print(f"[Z-Image] VAE decode complete: {len(images)} images generated")

        generation_timer.add("vae_decode", _time.perf_counter() - _t_phase)
        return images
