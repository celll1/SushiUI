"""
ControlNet Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- ControlNetSD15Adapter: SD1.5 Standard ControlNet / LLLite
- ControlNetSDXLAdapter: SDXL Standard ControlNet / LLLite [Phase 3]

Implements the third training mode alongside LoRA and Full Parameter.
ControlNet training freezes UNet/VAE/TE entirely and only trains
the ControlNet module.

References:
- diffusers ControlNetModel (Apache-2 license)
- sd-scripts (Apache-2 license) by kohya-ss (LLLite implementation)

Author: Claude (2026-01-26)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer, log_verbose
from .adapters import ControlNetSD15Adapter, ControlNetSDXLAdapter


class ControlNetTrainer(BaseTrainer):
    """
    ControlNet Trainer for SD1.5/SDXL models.

    Uses model-specific adapters for ControlNet creation, training,
    and checkpoint management.

    Supports:
    - Standard ControlNet (diffusers ControlNetModel)
    - ControlNet-LLLite (kohya-ss sd-scripts compatible) [Phase 2]
    """

    def __init__(
        self,
        controlnet_type: str = "standard",
        controlnet_pretrained_path: Optional[str] = None,
        init_from_unet: bool = True,
        # LLLite parameters (Phase 2)
        lllite_conditioning_channels: int = 32,
        lllite_rank: int = 64,
        # Condition generation (Phase 4)
        condition_preprocessors: Optional[List[str]] = None,
        condition_cache_mode: str = "on_the_fly",
        **kwargs
    ):
        """
        Initialize ControlNet Trainer.

        Args:
            controlnet_type: "standard" (diffusers ControlNetModel) or "lllite" (sd-scripts compatible)
            controlnet_pretrained_path: Path to existing ControlNet checkpoint for resume
            init_from_unet: Initialize ControlNet weights from base UNet (standard only)
            lllite_conditioning_channels: Number of conditioning channels for LLLite (Phase 2)
            lllite_rank: Rank for LLLite linear layers (Phase 2)
            condition_preprocessors: List of controlnet-aux preprocessor types (Phase 4)
            condition_cache_mode: "pre_generate" or "on_the_fly" (Phase 4)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # ControlNet-specific settings (set before super().__init__)
        self.controlnet_type = controlnet_type
        self.controlnet_pretrained_path = controlnet_pretrained_path
        self.init_from_unet = init_from_unet
        self.lllite_conditioning_channels = lllite_conditioning_channels
        self.lllite_rank = lllite_rank
        self.condition_preprocessors = condition_preprocessors
        self.condition_cache_mode = condition_cache_mode

        # ControlNet module storage (set by _create_controlnet)
        self.controlnet: Optional[nn.Module] = None

        # ControlNet training does NOT train UNet/TE
        self.train_unet = False
        self.train_text_encoder = False
        self.train_image_encoder = False

        # Flag to signal base_trainer to load condition images
        self.use_condition_images = True

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[ControlNet Trainer]"

        # Validate model type (only SD1.5/SDXL supported)
        if self.is_zimage or self.is_flux2:
            model_type = "Z-Image" if self.is_zimage else "FLUX.2"
            raise ValueError(
                f"ControlNet training is only supported for SD1.5 and SDXL models. "
                f"Detected model type: {model_type}"
            )
        if self.is_deus:
            raise ValueError(
                f"ControlNet training is only supported for SD1.5 and SDXL models. "
                f"Detected model type: DEUS"
            )

        # Freeze all base model components
        self._freeze_base_models()

        # Create model-specific adapter
        self._create_adapter()

        # Create ControlNet using adapter
        self._create_controlnet()

        print(f"{self.log_prefix} Initialized")
        print(f"{self.log_prefix} ControlNet type: {self.controlnet_type}")
        print(f"{self.log_prefix} Model type: {'SDXL' if self.is_sdxl else 'SD1.5'}")

    def _freeze_base_models(self):
        """Freeze all base model components (UNet, VAE, TE)."""
        print(f"{self.log_prefix} Freezing all base model components...")

        if self.unet is not None:
            self.unet.requires_grad_(False)
            self.unet.eval()
            print(f"  UNet: frozen")

        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
            print(f"  VAE: frozen")

        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
            print(f"  Text Encoder 1: frozen")

        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)
            self.text_encoder_2.eval()
            print(f"  Text Encoder 2: frozen")

    def _create_adapter(self):
        """Create model-specific ControlNet adapter based on detected model type."""
        if self.is_sdxl:
            self.adapter = ControlNetSDXLAdapter(self, self.controlnet_type)
            print(f"{self.log_prefix} Using ControlNetSDXLAdapter ({self.controlnet_type})")
        else:
            self.adapter = ControlNetSD15Adapter(self, self.controlnet_type)
            print(f"{self.log_prefix} Using ControlNetSD15Adapter ({self.controlnet_type})")

    def _create_controlnet(self):
        """Create ControlNet model using adapter."""
        print(f"{self.log_prefix} Creating ControlNet...")

        self.controlnet = self.adapter.create_controlnet(
            init_from_unet=self.init_from_unet,
            pretrained_path=self.controlnet_pretrained_path,
        )

        # Enable gradient checkpointing for ControlNet
        if hasattr(self.controlnet, 'enable_gradient_checkpointing'):
            self.controlnet.enable_gradient_checkpointing()
            print(f"{self.log_prefix} Gradient checkpointing enabled for ControlNet")

        print(f"{self.log_prefix} ControlNet created successfully")

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters from ControlNet.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters(self.controlnet)

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save ControlNet checkpoint.

        Standard: saves as diffusers-compatible directory
        LLLite: saves as sd-scripts compatible .safetensors [Phase 2]

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        if self.controlnet_type == "standard":
            # Directory format: {run_name}_controlnet_step_001000/
            checkpoint_path = self.output_dir / f"{self.run_name}_controlnet_step_{step:06d}"
        else:
            # LLLite: single file format
            checkpoint_path = self.output_dir / f"{self.run_name}_lllite_step_{step:06d}.safetensors"

        self.adapter.save_checkpoint(self.controlnet, step, epoch, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load ControlNet checkpoint for resume training.

        Args:
            checkpoint_path: Path to checkpoint directory or file

        Returns:
            Step number from checkpoint
        """
        print(f"{self.log_prefix} Loading checkpoint: {checkpoint_path}")
        step = self.adapter.load_checkpoint(self.controlnet, checkpoint_path)
        print(f"{self.log_prefix} Loaded checkpoint from step {step}")
        return step

    # ============================================================
    # Sample Generation (ControlNet-aware)
    # ============================================================

    def _load_sample_condition_image(self, condition_image_path: "Optional[str]" = None) -> "Optional[Image.Image]":
        """
        Load condition image for sample generation during training.

        Priority:
        1. Per-prompt condition_image_path argument
        2. First dataset item's reference_images[0] (fallback)

        Uses path-based caching to avoid reloading the same image across calls.

        Args:
            condition_image_path: Path to condition image (per-prompt, optional)

        Returns:
            PIL Image or None if no condition image available
        """
        from PIL import Image

        # Initialize path-based cache
        if not hasattr(self, '_condition_image_cache'):
            self._condition_image_cache = {}  # path -> PIL.Image

        # Option 1: Per-prompt condition image path
        if condition_image_path:
            # Check cache
            if condition_image_path in self._condition_image_cache:
                return self._condition_image_cache[condition_image_path]

            p = Path(condition_image_path)
            if p.exists():
                try:
                    img = Image.open(str(p)).convert("RGB")
                    print(f"{self.log_prefix} [Sample] Loaded condition image from per-prompt path: {p}")
                    self._condition_image_cache[condition_image_path] = img
                    return img
                except Exception as e:
                    print(f"{self.log_prefix} [Sample] Failed to load condition image from {p}: {e}")
            else:
                print(f"{self.log_prefix} [Sample] Condition image path not found: {p}")

        # Option 2: First dataset item's reference image (fallback)
        fallback_key = "__dataset_fallback__"
        if fallback_key in self._condition_image_cache:
            return self._condition_image_cache[fallback_key]

        datasets = getattr(self, '_training_datasets', None)
        if datasets:
            for ds in datasets:
                items = ds.get("items", [])
                for item in items:
                    ref_images = item.get("reference_images", [])
                    if ref_images:
                        ref_path = Path(ref_images[0])
                        if ref_path.exists():
                            try:
                                img = Image.open(str(ref_path)).convert("RGB")
                                print(f"{self.log_prefix} [Sample] Loaded condition image from dataset: {ref_path}")
                                self._condition_image_cache[fallback_key] = img
                                return img
                            except Exception as e:
                                print(f"{self.log_prefix} [Sample] Failed to load {ref_path}: {e}")

        print(f"{self.log_prefix} [Sample] WARNING: No condition image found for sample generation")
        # Cache None for fallback to avoid repeated warnings
        self._condition_image_cache[fallback_key] = None
        return None

    def generate_sample(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        current_step: int = 0,
        schedule_type: str = "uniform",
        condition_image_path: "Optional[str]" = None,
    ) -> "Image.Image":
        """
        Generate sample image during ControlNet training.

        Overrides BaseTrainer.generate_sample() to apply the trained ControlNet
        during sample generation, allowing visual verification of training progress.

        Args:
            condition_image_path: Per-prompt condition image path (optional).
                If not provided, falls back to dataset's first reference image.

        Standard ControlNet: Sets pipeline.controlnet and passes controlnet_images
        LLLite: Applies patches to UNet before sampling, removes after
        """
        # Load condition image (per-prompt path or fallback to dataset)
        loaded_condition = self._load_sample_condition_image(condition_image_path)

        # No condition image available: fall back to base (no ControlNet)
        if loaded_condition is None:
            print(f"{self.log_prefix} [Sample] No condition image, falling back to base generate_sample()")
            return super().generate_sample(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                current_step=current_step, schedule_type=schedule_type,
            )

        # Dispatch to type-specific implementation
        if self.controlnet_type == "standard":
            return self._generate_sample_standard(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                current_step=current_step, schedule_type=schedule_type,
                condition_image=loaded_condition,
            )
        elif self.controlnet_type == "lllite":
            return self._generate_sample_lllite(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                current_step=current_step, schedule_type=schedule_type,
                condition_image=loaded_condition,
            )
        else:
            # Unknown type: fall back to base
            return super().generate_sample(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                current_step=current_step, schedule_type=schedule_type,
            )

    def _generate_sample_standard(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        current_step: int = 0,
        schedule_type: str = "uniform",
        condition_image: "Optional[Image.Image]" = None,
    ) -> "Image.Image":
        """
        Generate sample with Standard ControlNet (ControlNetModel).

        Sets pipeline.controlnet and passes controlnet_images to custom_sampling_loop().

        Args:
            condition_image: Pre-loaded PIL Image for conditioning.
        """
        from PIL import Image
        import random

        print(f"{self.log_prefix} [Sample] Generating with Standard ControlNet: {prompt[:50]}...")

        from core.inference.custom_sampling import custom_sampling_loop
        from core.inference.schedulers import get_scheduler

        # Resize condition image to sample dimensions
        condition_image = condition_image.resize((width, height), Image.LANCZOS)

        # Set models to eval mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()
        self.controlnet.eval()

        try:
            # ========================================
            # STEP 1: Create Temporary Pipeline with ControlNet
            # ========================================
            class TempPipeline:
                def __init__(self, unet, vae, text_encoder, text_encoder_2,
                             scheduler, tokenizer, tokenizer_2, controlnet):
                    self.unet = unet
                    self.vae = vae
                    self.text_encoder = text_encoder
                    self.text_encoder_2 = text_encoder_2
                    self.scheduler = scheduler
                    self.tokenizer = tokenizer
                    self.tokenizer_2 = tokenizer_2
                    self.controlnet = controlnet
                    self.vae_scale_factor = 8
                    self.image_processor = None

            # Map schedule_type
            schedule_type_mapped = schedule_type
            if schedule_type == "sgm_uniform":
                schedule_type_mapped = "uniform"

            class SchedulerContainer:
                def __init__(self, scheduler):
                    self.scheduler = scheduler

            scheduler_container = SchedulerContainer(self.original_scheduler)
            scheduler = get_scheduler(
                pipeline=scheduler_container,
                sampler="euler",
                schedule_type=schedule_type_mapped
            )

            pipeline = TempPipeline(
                unet=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder,
                text_encoder_2=getattr(self, 'text_encoder_2', None),
                scheduler=scheduler,
                tokenizer=self.tokenizer,
                tokenizer_2=getattr(self, 'tokenizer_2', None),
                controlnet=self.controlnet,
            )

            # ========================================
            # STEP 2: Text Encoding
            # ========================================
            self.move_text_encoder_to_gpu()

            if self.is_sdxl:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt("", requires_grad=False)
            else:
                prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds = self.encode_prompt("", requires_grad=False)
                pooled_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            # Pad negative embeddings to match positive (prompt chunking)
            if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                seq_len_diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
                padding = torch.zeros(
                    (negative_prompt_embeds.shape[0], seq_len_diff, negative_prompt_embeds.shape[2]),
                    dtype=negative_prompt_embeds.dtype,
                    device=negative_prompt_embeds.device
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)

            self.move_text_encoder_to_cpu()
            torch.cuda.empty_cache()

            # ========================================
            # STEP 3: Create Generator
            # ========================================
            if seed < 0:
                actual_seed = random.randint(0, 2**32 - 1)
            else:
                actual_seed = seed

            generator = torch.Generator(device=self.device).manual_seed(actual_seed)

            # ========================================
            # STEP 4: Call custom_sampling_loop with ControlNet
            # ========================================
            self.move_main_model_to_gpu()
            self.move_vae_to_gpu()

            is_v_prediction = pipeline.scheduler.config.get("prediction_type") == "v_prediction"
            guidance_rescale = 0.7 if is_v_prediction else 0.0

            log_verbose(f"{self.log_prefix} [Sample] Standard ControlNet active, condition_scale=1.0")

            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                image = custom_sampling_loop(
                    pipeline=pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    width=width,
                    height=height,
                    generator=generator,
                    ancestral_generator=None,
                    latents=None,
                    prompt_embeds_callback=None,
                    progress_callback=None,
                    step_callback=None,
                    developer_mode=False,
                    cfg_schedule_type="constant",
                    cfg_schedule_min=1.0,
                    cfg_schedule_max=None,
                    cfg_schedule_power=2.0,
                    cfg_rescale_snr_alpha=0.0,
                    dynamic_threshold_percentile=0.0,
                    dynamic_threshold_mimic_scale=1.0,
                    nag_enable=False,
                    nag_scale=5.0,
                    nag_tau=3.5,
                    nag_alpha=0.25,
                    nag_sigma_end=0.0,
                    nag_negative_prompt_embeds=None,
                    nag_negative_pooled_prompt_embeds=None,
                    attention_type="normal",
                    # ControlNet parameters
                    controlnet_images=[condition_image],
                    controlnet_conditioning_scale=1.0,
                    control_guidance_start=0.0,
                    control_guidance_end=1.0,
                )

                self.move_main_model_to_cpu()
                self.move_vae_to_cpu()
                torch.cuda.empty_cache()

                log_verbose(f"{self.log_prefix} [Sample] Standard ControlNet sample generated (seed: {actual_seed})")
                return image

        except Exception as e:
            print(f"{self.log_prefix} [Sample] ERROR: {type(e).__name__}: {str(e)}")
            print(f"{self.log_prefix} [Sample] Sample generation failed - training will continue")

            from PIL import Image
            placeholder = Image.new("RGB", (width, height), color=(255, 255, 255))
            return placeholder

        finally:
            # Restore training mode
            self.unet.train()
            self.vae.train()
            self.text_encoder.train()
            if self.text_encoder_2 is not None:
                self.text_encoder_2.train()
            self.controlnet.train()
            self.move_main_model_to_gpu()

    def _generate_sample_lllite(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        current_step: int = 0,
        schedule_type: str = "uniform",
        condition_image: "Optional[Image.Image]" = None,
    ) -> "Image.Image":
        """
        Generate sample with LLLite ControlNet.

        Applies LLLite patches to UNet before sampling, removes after.

        Args:
            condition_image: Pre-loaded PIL Image for conditioning.
        """
        from PIL import Image
        import random
        import torchvision.transforms.functional as TF

        print(f"{self.log_prefix} [Sample] Generating with LLLite ControlNet: {prompt[:50]}...")

        from core.inference.custom_sampling import custom_sampling_loop
        from core.inference.schedulers import get_scheduler

        # Resize condition image to sample dimensions
        condition_image = condition_image.resize((width, height), Image.LANCZOS)

        # Set models to eval mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()
        self.controlnet.eval()

        lllite_patched = False

        try:
            # ========================================
            # STEP 1: Create Temporary Pipeline (no controlnet attr)
            # ========================================
            if self.is_sdxl:
                class TempPipeline:
                    def __init__(self, unet, vae, text_encoder, text_encoder_2,
                                 scheduler, tokenizer, tokenizer_2):
                        self.unet = unet
                        self.vae = vae
                        self.text_encoder = text_encoder
                        self.text_encoder_2 = text_encoder_2
                        self.scheduler = scheduler
                        self.tokenizer = tokenizer
                        self.tokenizer_2 = tokenizer_2
                        self.vae_scale_factor = 8
                        self.image_processor = None
            else:
                class TempPipeline:
                    def __init__(self, unet, vae, text_encoder, scheduler, tokenizer):
                        self.unet = unet
                        self.vae = vae
                        self.text_encoder = text_encoder
                        self.scheduler = scheduler
                        self.tokenizer = tokenizer
                        self.vae_scale_factor = 8
                        self.image_processor = None

            schedule_type_mapped = schedule_type
            if schedule_type == "sgm_uniform":
                schedule_type_mapped = "uniform"

            class SchedulerContainer:
                def __init__(self, scheduler):
                    self.scheduler = scheduler

            scheduler_container = SchedulerContainer(self.original_scheduler)
            scheduler = get_scheduler(
                pipeline=scheduler_container,
                sampler="euler",
                schedule_type=schedule_type_mapped
            )

            if self.is_sdxl:
                pipeline = TempPipeline(
                    unet=self.unet,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    text_encoder_2=self.text_encoder_2,
                    scheduler=scheduler,
                    tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2,
                )
            else:
                pipeline = TempPipeline(
                    unet=self.unet,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    scheduler=scheduler,
                    tokenizer=self.tokenizer,
                )

            # ========================================
            # STEP 2: Text Encoding
            # ========================================
            self.move_text_encoder_to_gpu()

            if self.is_sdxl:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt("", requires_grad=False)
            else:
                prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds = self.encode_prompt("", requires_grad=False)
                pooled_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            # Pad negative embeddings to match positive (prompt chunking)
            if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                seq_len_diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
                padding = torch.zeros(
                    (negative_prompt_embeds.shape[0], seq_len_diff, negative_prompt_embeds.shape[2]),
                    dtype=negative_prompt_embeds.dtype,
                    device=negative_prompt_embeds.device
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)

            self.move_text_encoder_to_cpu()
            torch.cuda.empty_cache()

            # ========================================
            # STEP 3: Create Generator
            # ========================================
            if seed < 0:
                actual_seed = random.randint(0, 2**32 - 1)
            else:
                actual_seed = seed

            generator = torch.Generator(device=self.device).manual_seed(actual_seed)

            # ========================================
            # STEP 4: Apply LLLite patches and call custom_sampling_loop
            # ========================================
            self.move_main_model_to_gpu()
            self.move_vae_to_gpu()

            # Prepare condition tensor [1, 3, H, W] in [0, 1] range
            cond_tensor = TF.to_tensor(condition_image).unsqueeze(0).to(
                device=self.device, dtype=self.training_dtype
            )

            # Apply LLLite patches to UNet
            self.controlnet.apply_patches(self.unet, cond_tensor)
            lllite_patched = True
            log_verbose(f"{self.log_prefix} [Sample] LLLite patches applied to UNet")

            is_v_prediction = pipeline.scheduler.config.get("prediction_type") == "v_prediction"
            guidance_rescale = 0.7 if is_v_prediction else 0.0

            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                image = custom_sampling_loop(
                    pipeline=pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    width=width,
                    height=height,
                    generator=generator,
                    ancestral_generator=None,
                    latents=None,
                    prompt_embeds_callback=None,
                    progress_callback=None,
                    step_callback=None,
                    developer_mode=False,
                    cfg_schedule_type="constant",
                    cfg_schedule_min=1.0,
                    cfg_schedule_max=None,
                    cfg_schedule_power=2.0,
                    cfg_rescale_snr_alpha=0.0,
                    dynamic_threshold_percentile=0.0,
                    dynamic_threshold_mimic_scale=1.0,
                    nag_enable=False,
                    nag_scale=5.0,
                    nag_tau=3.5,
                    nag_alpha=0.25,
                    nag_sigma_end=0.0,
                    nag_negative_prompt_embeds=None,
                    nag_negative_pooled_prompt_embeds=None,
                    attention_type="normal",
                    # No controlnet params for LLLite (patches already applied)
                )

                # Remove LLLite patches
                self.controlnet.remove_patches(self.unet)
                lllite_patched = False

                self.move_main_model_to_cpu()
                self.move_vae_to_cpu()
                torch.cuda.empty_cache()

                log_verbose(f"{self.log_prefix} [Sample] LLLite ControlNet sample generated (seed: {actual_seed})")
                return image

        except Exception as e:
            print(f"{self.log_prefix} [Sample] ERROR: {type(e).__name__}: {str(e)}")
            print(f"{self.log_prefix} [Sample] Sample generation failed - training will continue")

            from PIL import Image
            placeholder = Image.new("RGB", (width, height), color=(255, 255, 255))
            return placeholder

        finally:
            # Remove LLLite patches if still applied
            if lllite_patched and hasattr(self.controlnet, '_is_patched') and self.controlnet._is_patched:
                self.controlnet.remove_patches(self.unet)

            # Restore training mode
            self.unet.train()
            self.vae.train()
            self.text_encoder.train()
            if self.text_encoder_2 is not None:
                self.text_encoder_2.train()
            self.controlnet.train()
            self.move_main_model_to_gpu()
