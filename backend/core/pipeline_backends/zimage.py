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
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type
                self.current_attention_type = attention_type
            else:
                print(f"[Z-Image] Attention backend already set to: {attention_type} (skipping)")
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type  # Ensure it's set (for safety)

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

            # Offload Text Encoder to CPU to free VRAM
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

            if not enable_block_swap:
                # Normal mode: move entire Transformer to GPU
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
                    use_pinned_memory=use_pinned_memory
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

            latents = self._zimage_denoising_loop(
                transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
                height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
                generator, progress_callback, step_callback,
                spectrum_params=params,
                nag_negative_embeds_list=nag_negative_embeds_list,
                nag_params=params
            )

            # Offload Transformer to CPU to free VRAM for VAE
            move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 3: VAE Decode
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding
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

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image generation failed: {str(e)}")

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
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type
                self.current_attention_type = attention_type
            else:
                print(f"[Z-Image] Attention backend already set to: {attention_type} (skipping)")
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type

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

            # Offload Text Encoder to CPU
            move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: VAE Encode Input Image
            # ============================================================
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

            if not enable_block_swap:
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
                    use_pinned_memory=use_pinned_memory
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
                nag_params=params
            )

            # Offload Transformer to CPU
            move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 5: VAE Decode
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding
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

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] img2img generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image img2img generation failed: {str(e)}")

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

            # Offload Text Encoder to CPU
            move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: VAE Encode Input Image and Mask
            # ============================================================
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

            if not enable_block_swap:
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
                    use_pinned_memory=use_pinned_memory
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
                nag_params=params
            )

            # Offload Transformer to CPU
            move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 5: VAE Decode
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding
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

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] inpaint generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image inpaint generation failed: {str(e)}")

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

        return prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance

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
        nag_params: Optional[Dict[str, Any]] = None
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

        Returns:
            latents: Denoised latents (torch.Tensor)
        """
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

            # Spectrum: forecast the post-CFG velocity on skip steps (skip transformer + CFG)
            spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
            if spectrum_skip:
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
                if apply_cfg:
                    if nag_this_step:
                        latent_model_input = latents.to(input_dtype).repeat(3, 1, 1, 1)
                        prompt_embeds_model_input = (
                            negative_prompt_embeds_list + prompt_embeds_list
                            + nag_negative_embeds_list
                        )
                        timestep_model_input = timestep.repeat(3)
                    else:
                        latent_model_input = latents.to(input_dtype).repeat(2, 1, 1, 1)
                        # CFG input order: [negative, positive] (consistent with SD/SDXL)
                        prompt_embeds_model_input = negative_prompt_embeds_list + prompt_embeds_list
                        timestep_model_input = timestep.repeat(2)
                else:
                    if nag_this_step:
                        latent_model_input = latents.to(input_dtype).repeat(2, 1, 1, 1)
                        prompt_embeds_model_input = prompt_embeds_list + nag_negative_embeds_list
                        timestep_model_input = timestep.repeat(2)
                    else:
                        latent_model_input = latents.to(input_dtype)
                        prompt_embeds_model_input = prompt_embeds_list
                        timestep_model_input = timestep

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

        return latents

    def _zimage_decode_latents(self, vae, latents):
        """
        Stage 3: VAE Decode for Z-Image
        Decodes latents to images using VAE.
        VAE is on GPU when this is called, and will be moved to CPU after.

        Returns:
            images: List of PIL images
        """
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
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        images = [Image.fromarray(img) for img in image]

        print(f"[Z-Image] VAE decode complete: {len(images)} images generated")

        return images
