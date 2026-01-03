"""
Base Trainer for SushiUI Training Framework

This module contains shared training logic that is common across different
training methods (LoRA, Full Parameter, etc.).

Architecture:
- Supports SD1.5, SDXL, and Z-Image models
- Component-based approach (individual UNet, VAE, TextEncoder loading)
- Gradient checkpointing for memory efficiency
- Mixed precision training support
- SNR-weighted loss calculation
- Latent caching for faster training
- Bucketing support for variable resolutions
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import numpy as np
import gc
import math
from abc import ABC, abstractmethod


# ============================================================
# Utility Functions
# ============================================================

def print_vram_usage(label: str = ""):
    """
    Print detailed VRAM usage statistics.

    Args:
        label: Optional label to identify the checkpoint
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

    print(f"[VRAM] {label if label else 'Current'}")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Peak:      {max_allocated:.2f} GB")


def get_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """Get memory usage of a tensor in MB."""
    return tensor.element_size() * tensor.nelement() / 1024**2


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch.dtype.

    Args:
        dtype_str: String like "fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e5m2"

    Returns:
        torch.dtype
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }

    if dtype_str not in dtype_map:
        print(f"[Trainer] WARNING: Unknown dtype '{dtype_str}', defaulting to fp16")
        return torch.float16

    return dtype_map[dtype_str]


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR (Signal-to-Noise Ratio) from diffusion timesteps.

    SNR = alpha_bar / (1 - alpha_bar)

    Args:
        noise_scheduler: DDPMScheduler instance
        timesteps: Tensor of timesteps [batch_size]

    Returns:
        SNR values [batch_size]
    """
    # Get alpha_bar for each timestep
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
    alpha_bar = alphas_cumprod[timesteps].float()

    # SNR = alpha / (1 - alpha)
    snr = alpha_bar / (1.0 - alpha_bar)

    return snr


def apply_snr_weight(loss, timesteps, noise_scheduler, min_snr_gamma=5.0):
    """
    Apply Min-SNR gamma weighting to loss.

    Reference: "Efficient Diffusion Training via Min-SNR Weighting Strategy"
    https://arxiv.org/abs/2303.09556

    This reweights the loss to ensure all timesteps contribute equally to training,
    preventing the model from overfitting to high-noise timesteps.

    Args:
        loss: Unreduced loss tensor [batch_size, ...]
        timesteps: Tensor of timesteps [batch_size]
        noise_scheduler: DDPMScheduler instance
        min_snr_gamma: Minimum SNR gamma value (default: 5.0, standard for SD/SDXL)

    Returns:
        Weighted loss (same shape as input)
    """
    snr = compute_snr(noise_scheduler, timesteps)

    # Min-SNR gamma weighting: min(SNR, gamma) / SNR
    # This clamps the weight for low-noise (high SNR) timesteps
    mse_loss_weights = torch.clamp(snr, max=min_snr_gamma) / snr

    # Reshape to match loss dimensions [batch_size, 1, 1, 1]
    while mse_loss_weights.dim() < loss.dim():
        mse_loss_weights = mse_loss_weights.unsqueeze(-1)

    # Apply weighting
    weighted_loss = loss * mse_loss_weights

    return weighted_loss


def get_target_from_prediction_type(
    noise_scheduler,
    prediction_type: str,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Get the target tensor based on prediction type (LEGACY - DDPM only).

    DEPRECATED: Use add_noise_unified() and get_target_unified() instead.

    Args:
        noise_scheduler: DDPMScheduler instance
        prediction_type: "epsilon" (noise), "v_prediction", or "sample"
        latents: Original latents [B, C, H, W]
        noise: Sampled noise [B, C, H, W]
        timesteps: Timesteps [B]

    Returns:
        Target tensor for loss calculation
    """
    if prediction_type == "epsilon":
        # Predict noise (most common for SD/SDXL)
        return noise

    elif prediction_type == "v_prediction":
        # Predict velocity (v = alpha_bar * noise - sqrt(1 - alpha_bar) * latents)
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device)
        alpha_bar = alphas_cumprod[timesteps].float()

        # Reshape alpha_bar to [B, 1, 1, 1]
        while alpha_bar.dim() < latents.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        velocity = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * latents
        return velocity

    elif prediction_type == "sample":
        # Predict original sample (less common)
        return latents

    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")


def add_noise_unified(
    noise_process: str,
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Add noise to latents using specified noise process (Unified Framework).

    Args:
        noise_process: "ddpm" or "flow"
        noise_scheduler: Noise scheduler instance (DDPMScheduler or FlowMatchEulerDiscreteScheduler)
        latents: Original latents [B, C, H, W]
        noise: Sampled noise [B, C, H, W]
        timesteps: Timesteps (discrete for DDPM, continuous [0,1] for Flow)

    Returns:
        Noisy latents
    """
    if noise_process == "ddpm":
        # DDPM: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # timesteps are discrete [0, num_train_timesteps)
        return noise_scheduler.add_noise(latents, noise, timesteps)

    elif noise_process == "flow":
        # Flow Matching: x_t = (1 - t) * noise + t * x_0
        # timesteps are continuous [0, 1]
        t = timesteps.float()
        while t.dim() < latents.dim():
            t = t.unsqueeze(-1)

        noisy_latents = (1.0 - t) * noise + t * latents
        return noisy_latents

    else:
        raise ValueError(f"Unknown noise_process: {noise_process}")


def get_target_unified(
    noise_process: str,
    prediction_target: str,
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Get the training target based on noise process and prediction target (Unified Framework).

    Args:
        noise_process: "ddpm" or "flow"
        prediction_target: "epsilon", "velocity", or "sample"
        noise_scheduler: Noise scheduler instance
        latents: Original latents [B, C, H, W]
        noise: Sampled noise [B, C, H, W]
        timesteps: Timesteps (discrete for DDPM, continuous [0,1] for Flow)

    Returns:
        Target tensor for loss calculation
    """
    if noise_process == "ddpm":
        # DDPM noise process with discrete timesteps
        if prediction_target == "epsilon":
            # Predict noise
            return noise

        elif prediction_target == "velocity":
            # Predict velocity: v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x_0
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device)
            alpha_bar = alphas_cumprod[timesteps].float()

            while alpha_bar.dim() < latents.dim():
                alpha_bar = alpha_bar.unsqueeze(-1)

            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

            velocity = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * latents
            return velocity

        elif prediction_target == "sample":
            # Predict original sample
            return latents

        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    elif noise_process == "flow":
        # Flow Matching with continuous timesteps [0, 1]
        if prediction_target == "epsilon":
            # Predict noise (Flow + epsilon is unusual but supported)
            return noise

        elif prediction_target == "velocity":
            # Predict velocity: v = x_0 - noise (constant direction in flow matching)
            return latents - noise

        elif prediction_target == "sample":
            # Predict original sample
            return latents

        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    else:
        raise ValueError(f"Unknown noise_process: {noise_process}")


def predict_original_latent_unified(
    noise_process: str,
    prediction_target: str,
    noise_scheduler,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Predict original latent from model prediction (Unified Framework).

    Used for regularization losses (SNR, Energy) and reconstruction loss monitoring.

    Args:
        noise_process: "ddpm" or "flow"
        prediction_target: "epsilon", "velocity", or "sample"
        noise_scheduler: Noise scheduler instance
        noisy_latents: Noisy latents [B, C, H, W]
        model_pred: Model prediction [B, C, H, W]
        timesteps: Timesteps (discrete for DDPM, continuous [0,1] for Flow)

    Returns:
        Predicted original latent [B, C, H, W]
    """
    if noise_process == "ddpm":
        # DDPM: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        alpha_bar = alphas_cumprod[timesteps]

        while alpha_bar.dim() < noisy_latents.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        if prediction_target == "epsilon":
            # model_pred = noise, solve for x_0: x_0 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)
            predicted_latent = (noisy_latents - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar
        elif prediction_target == "velocity":
            # model_pred = v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0
            # Solve for x_0: x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
            predicted_latent = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * model_pred
        elif prediction_target == "sample":
            # model_pred = x_0 directly
            predicted_latent = model_pred
        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    elif noise_process == "flow":
        # Flow Matching: x_t = (1 - t) * noise + t * x_0
        t = timesteps.float()
        while t.dim() < noisy_latents.dim():
            t = t.unsqueeze(-1)

        if prediction_target == "epsilon":
            # model_pred = noise, solve for x_0: x_0 = (x_t - (1 - t) * noise) / t
            # Avoid division by zero at t=0
            epsilon = 1e-8
            predicted_latent = (noisy_latents - (1.0 - t) * model_pred) / (t + epsilon)
        elif prediction_target == "velocity":
            # model_pred = v = x_0 - noise
            # x_t = (1 - t) * noise + t * x_0
            # Let noise = x_0 - v, then x_t = (1 - t) * (x_0 - v) + t * x_0 = x_0 - (1 - t) * v
            # Solve for x_0: x_0 = x_t + (1 - t) * v
            predicted_latent = noisy_latents + (1.0 - t) * model_pred
        elif prediction_target == "sample":
            # model_pred = x_0 directly
            predicted_latent = model_pred
        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    else:
        raise ValueError(f"Unknown noise_process: {noise_process}")

    return predicted_latent


# ============================================================
# Base Trainer Class
# ============================================================

class BaseTrainer(ABC):
    """
    Abstract base trainer class with shared logic for all training methods.

    Subclasses must implement:
    - setup_trainable_parameters()
    - save_checkpoint()
    - load_checkpoint()
    - find_latest_checkpoint() (optional)
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        run_name: str = None,
        run_id: Optional[int] = None,  # Database run ID for metrics logging
        learning_rate: float = 1e-4,
        device: str = "cuda",
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        debug_vram: bool = False,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
        # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
        prompt_chunking_mode: str = "a1111",  # "a1111", "sd_scripts", "nobos"
        max_prompt_chunks: int = 0,  # 0 = unlimited
        # Component-specific learning rates
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        # Block Swap settings (training VRAM optimization)
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = False,
        # Fused optimizer groups (for any optimizer with Block Swap)
        num_optimizer_groups: int = 0,
        # Optimizer options and hyperparameters
        optimizer_is_paged: bool = False,
        optimizer_cautious: bool = False,
        optimizer_beta1: Optional[float] = None,
        optimizer_beta2: Optional[float] = None,
        optimizer_epsilon: Optional[float] = None,
        optimizer_weight_decay: Optional[float] = None,
        # Schedule-Free optimizer options (RingBuffer optimizers only)
        optimizer_schedule_free: bool = False,
        optimizer_warmup_steps: int = 0,
        optimizer_schedule_free_r: float = 0.0,
        optimizer_schedule_free_weight_lr_power: float = 2.0,
        optimizer_use_radam: bool = False,
    ):
        """
        Initialize base trainer.

        Args:
            model_path: Path to base Stable Diffusion model
            output_dir: Directory to save checkpoints
            run_name: Training run name (for checkpoint filename generation)
            learning_rate: Learning rate
            device: Device to use (cuda/cpu)
            weight_dtype: Model weight dtype (fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2)
            training_dtype: Training/activation dtype (fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            output_dtype: Output dtype for safetensors (fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            vae_dtype: VAE-specific dtype (fp16, fp32, bf16) - SDXL VAE works fine with fp16
            mixed_precision: Enable mixed precision training (autocast)
            debug_vram: Enable detailed VRAM profiling (default: False)
            use_flash_attention: Enable Flash Attention for training (faster, lower memory)
            min_snr_gamma: Min-SNR gamma value for loss weighting (default: 5.0, 0 to disable)
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.run_name = run_name or Path(output_dir).name
        self.run_id = run_id  # Database run ID (for dual logging: TensorBoard + DB)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Component-specific learning rates
        self.unet_lr = unet_lr if unet_lr is not None else learning_rate
        self.text_encoder_lr = text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_1_lr = text_encoder_1_lr if text_encoder_1_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_2_lr = text_encoder_2_lr if text_encoder_2_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate

        # Block Swap settings (training VRAM optimization)
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory = use_pinned_memory

        # Fused optimizer settings (for Block Swap compatibility)
        self.num_optimizer_groups = num_optimizer_groups
        self.use_fused_backward = False  # Adafactor per-parameter updates
        self.fused_optimizer_groups = None  # FusedOptimizerGroups instance (for any optimizer)

        # Optimizer options and hyperparameters (defaults will be used if None)
        self.optimizer_is_paged = optimizer_is_paged
        self.optimizer_cautious = optimizer_cautious
        self.optimizer_beta1 = optimizer_beta1
        self.optimizer_beta2 = optimizer_beta2
        self.optimizer_epsilon = optimizer_epsilon
        self.optimizer_weight_decay = optimizer_weight_decay

        # Schedule-Free optimizer options (RingBuffer optimizers only)
        self.optimizer_schedule_free = optimizer_schedule_free
        self.optimizer_warmup_steps = optimizer_warmup_steps
        self.optimizer_schedule_free_r = optimizer_schedule_free_r
        self.optimizer_schedule_free_weight_lr_power = optimizer_schedule_free_weight_lr_power
        self.optimizer_use_radam = optimizer_use_radam

        # Convert dtype strings to torch.dtype
        self.weight_dtype = get_torch_dtype(weight_dtype)
        self.training_dtype = get_torch_dtype(training_dtype)
        self.output_dtype = get_torch_dtype(output_dtype)
        self.vae_dtype = get_torch_dtype(vae_dtype)
        self.mixed_precision = mixed_precision
        self.debug_vram = debug_vram
        self.use_flash_attention = use_flash_attention
        self.min_snr_gamma = min_snr_gamma

        # Prompt chunking settings (SD/SDXL only)
        self.prompt_chunking_mode = prompt_chunking_mode
        self.max_prompt_chunks = max_prompt_chunks

        # Regularization losses (to prevent overbaking)
        self.snr_regularization_loss = None
        self.energy_regularization_loss = None
        self.config = {}  # Will be set by subclass for factory function access

        # Legacy dtype for compatibility
        self.dtype = self.weight_dtype

        # Log prefix for subclass override
        self.log_prefix = "[Trainer]"

        print(f"[Trainer] Precision settings:")
        print(f"  Weight dtype: {weight_dtype} ({self.weight_dtype})")
        print(f"  Training dtype: {training_dtype} ({self.training_dtype})")
        print(f"  Output dtype: {output_dtype} ({self.output_dtype})")
        print(f"  VAE dtype: {vae_dtype} ({self.vae_dtype})")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Loss calculation: Always FP32 for numerical stability")
        print(f"  Min-SNR gamma: {min_snr_gamma} ({'enabled' if min_snr_gamma > 0 else 'disabled'})")

        # Initialize tensorboard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = self.output_dir / "tensorboard" / timestamp
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        print(f"{self.log_prefix} Initializing on {self.device}")
        print(f"{self.log_prefix} Tensorboard logs: {tensorboard_dir}")
        print(f"{self.log_prefix} Loading model from {model_path}")

        # Load model components
        self._load_model_components()

    def _load_model_components(self):
        """Load model components (dispatcher for different model types)."""
        # Detect model type
        from core.model_loader import ModelLoader
        model_type = ModelLoader.detect_model_type(self.model_path)
        self.is_zimage = (model_type == "zimage")
        self.is_sdxl = False

        if self.is_zimage:
            self._load_zimage_components()
        else:
            self._load_sd_sdxl_components()

    def _load_zimage_components(self):
        """Load Z-Image model components."""
        print(f"{self.log_prefix} Detected Z-Image model")
        print(f"{self.log_prefix} Loading Z-Image components from {self.model_path}")

        from core.model_loader import ModelLoader
        components = ModelLoader.load_zimage_from_diffusers(
            model_path=self.model_path,
            device="cpu",
            torch_dtype=self.weight_dtype
        )

        # Store components
        self.transformer_original = components["transformer"]
        self.vae = components["vae"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.scheduler = components["scheduler"]

        # Z-Image specific: no text_encoder_2, no unet
        self.text_encoder_2 = None
        self.tokenizer_2 = None
        self.unet = None
        self.noise_scheduler = self.scheduler

        # Convert VAE to vae_dtype
        self.vae = self.vae.to(dtype=self.vae_dtype)

        # Wrap transformer with BatchedZImageWrapperOptimized
        from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
        print(f"{self.log_prefix} Wrapping Z-Image Transformer with BatchedZImageWrapperOptimized")
        self.transformer = BatchedZImageWrapperOptimized(self.transformer_original)
        print(f"{self.log_prefix} Phase 2 optimization: Complete batched processing")

        # Setup Flash Attention if enabled
        if self.use_flash_attention:
            self._setup_flash_attention_zimage()

        # Enable gradient checkpointing for Transformer (CRITICAL for VRAM reduction)
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            print(f"{self.log_prefix} Gradient checkpointing enabled for Z-Image Transformer")
        else:
            print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for Z-Image Transformer")

        # Enable gradient checkpointing for Text Encoder
        if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable()
            print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder (Qwen3)")

        # Freeze all base weights (full parameter training will unfreeze specific layers later)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # Setup Block Swap if enabled (before moving to GPU)
        self.layer_offload_conductor = None  # Will be initialized if blocks_to_swap > 0

        if self.blocks_to_swap > 0:
            print(f"{self.log_prefix} Block Swap enabled for training: {self.blocks_to_swap} blocks")
            print(f"{self.log_prefix} Using LayerOffloadConductor (Ring Buffer implementation)")
            print(f"{self.log_prefix} Pinned memory: {self.use_pinned_memory}")

            # Import new ring buffer implementation
            from core.memory_management import LayerOffloadConductor

            # Check if transformer has layers attribute
            if not hasattr(self.transformer_original, 'layers'):
                raise ValueError(
                    f"Transformer must have 'layers' attribute for Block Swap. "
                    f"Found: {type(self.transformer_original)}"
                )

            # Initialize Layer Offload Conductor
            self.layer_offload_conductor = LayerOffloadConductor(
                layers=self.transformer_original.layers,
                blocks_to_swap=self.blocks_to_swap,
                device=self.device,
                use_pinned_memory=self.use_pinned_memory,
                cpu_buffer_size_mb=8192,  # 8GB CPU buffer for layer params
                activation_buffer_size_mb=4096,  # 4GB CPU buffer for activations
                enable_prefetch=True,  # Enable prefetching next layer
                enable_activation_offload=False  # Disable for now (experimental)
            )

            # Attach to transformer for reference
            self.transformer_original._layer_offload_conductor = self.layer_offload_conductor

            # Register hooks for automatic layer swapping
            self.layer_offload_conductor.register_hooks()

            print(f"{self.log_prefix} LayerOffloadConductor initialized successfully")
            print(f"{self.log_prefix} Ring buffer allocation strategy enabled")
        else:
            print(f"{self.log_prefix} Block Swap disabled (blocks_to_swap=0)")
            # Move Transformer to GPU normally
            print(f"{self.log_prefix} Moving Transformer to {self.device}...")
            self.transformer_original.to(self.device)
            # Note: self.transformer.transformer is the same object as self.transformer_original
            # No need to call self.transformer.to(device) again

        print(f"{self.log_prefix} Z-Image model loaded successfully")
        print(f"{self.log_prefix} Scheduler type: {self.scheduler.__class__.__name__}")
        print(f"{self.log_prefix} VAE latent channels: {self.vae.config.latent_channels}")

    def _load_sd_sdxl_components(self):
        """Load SD/SDXL model components."""
        is_safetensors = self.model_path.endswith('.safetensors')

        if is_safetensors:
            print(f"{self.log_prefix} Loading from safetensors file")
            # Try SDXL first, fall back to SD1.5
            try:
                print(f"{self.log_prefix} Trying SDXL pipeline...")
                temp_pipeline = StableDiffusionXLPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                )
                is_sdxl_model = True
            except Exception as e:
                print(f"{self.log_prefix} Not SDXL, trying SD1.5 pipeline...")
                temp_pipeline = StableDiffusionPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                )
                is_sdxl_model = False

            # Extract components
            self.vae = temp_pipeline.vae
            self.text_encoder = temp_pipeline.text_encoder
            self.tokenizer = temp_pipeline.tokenizer
            self.unet = temp_pipeline.unet

            # Save original scheduler for inference (sample generation)
            # This preserves the model's original scheduler config (prediction_type, timestep_spacing, etc.)
            self.original_scheduler = temp_pipeline.scheduler

            # Use DDPMScheduler for training
            self.noise_scheduler = DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
                prediction_type="epsilon"
            )

            # SDXL-specific components
            if is_sdxl_model:
                self.text_encoder_2 = temp_pipeline.text_encoder_2
                self.tokenizer_2 = temp_pipeline.tokenizer_2
            else:
                self.text_encoder_2 = None
                self.tokenizer_2 = None

            del temp_pipeline
            self.vae = self.vae.to(dtype=self.vae_dtype)

        else:
            print(f"{self.log_prefix} Loading from diffusers directory")
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype
            )

            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype
            )

            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_path,
                subfolder="tokenizer"
            )

            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_path,
                subfolder="unet",
                torch_dtype=self.dtype
            )

            # Save original scheduler for inference (sample generation)
            from diffusers.schedulers import EulerDiscreteScheduler
            self.original_scheduler = EulerDiscreteScheduler.from_pretrained(
                self.model_path,
                subfolder="scheduler"
            )

            # Use DDPMScheduler for training
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.model_path,
                subfolder="scheduler"
            )

            # Check for SDXL
            if (Path(self.model_path) / "text_encoder_2").exists():
                self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder_2",
                    torch_dtype=self.dtype
                )
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                    self.model_path,
                    subfolder="tokenizer_2"
                )
                is_sdxl_model = True
            else:
                self.text_encoder_2 = None
                self.tokenizer_2 = None
                is_sdxl_model = False

        # Store SDXL flag
        self.is_sdxl = is_sdxl_model

        # No transformer for SD/SDXL
        self.transformer = None
        self.transformer_original = None

        # Setup Flash Attention if enabled
        if self.use_flash_attention:
            self._setup_flash_attention_sd_sdxl()

        # Enable gradient checkpointing for U-Net (CRITICAL for VRAM reduction)
        if hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()
            print(f"{self.log_prefix} Gradient checkpointing enabled for U-Net")
        else:
            print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for U-Net")

        # Enable gradient checkpointing for Text Encoders
        if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable()
            print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder 1")

        if self.text_encoder_2 is not None:
            if hasattr(self.text_encoder_2, 'gradient_checkpointing_enable'):
                self.text_encoder_2.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder 2")

        print(f"{self.log_prefix} {'SDXL' if self.is_sdxl else 'SD1.5'} model loaded successfully")

    def _setup_flash_attention_zimage(self):
        """Setup Flash Attention for Z-Image models."""
        import sys

        # The transformer is loaded via importlib with module name "zimage_transformer"
        # We need to access the ACTUAL module used, not core.models.zimage_transformer
        if 'zimage_transformer' in sys.modules:
            zimage_transformer_module = sys.modules['zimage_transformer']
            ZImageAttention = zimage_transformer_module.ZImageAttention
            print(f"{self.log_prefix} Setting Flash Attention backend for Z-Image...")
            print(f"{self.log_prefix} [DEBUG] Module: {zimage_transformer_module.__name__}")
            ZImageAttention._attention_backend = "flash"
            print(f"{self.log_prefix} [OK] Flash Attention enabled: {ZImageAttention._attention_backend}")
        else:
            # Fallback: Try core.models.zimage_transformer (for inference pipeline)
            from core.models.zimage_transformer import ZImageAttention
            print(f"{self.log_prefix} Setting Flash Attention backend for Z-Image (fallback)...")
            ZImageAttention._attention_backend = "flash"
            print(f"{self.log_prefix} [OK] Flash Attention enabled: {ZImageAttention._attention_backend}")

    def _setup_flash_attention_sd_sdxl(self):
        """Setup Flash Attention for SD/SDXL models."""
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            print(f"{self.log_prefix} Setting Flash Attention for SD/SDXL UNet...")
            self.unet.set_attn_processor(AttnProcessor2_0())
            print(f"{self.log_prefix} [OK] Flash Attention enabled for UNet")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to enable Flash Attention: {e}")

    # ============================================================
    # Abstract Methods (must be implemented by subclasses)
    # ============================================================

    @abstractmethod
    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        """
        Setup trainable parameters for the model.

        Returns:
            List of parameter groups for optimizer (each with 'params' and 'lr')
        """
        pass

    @abstractmethod
    def save_checkpoint(self, step: int, epoch: int):
        """
        Save training checkpoint.

        Args:
            step: Current training step
            epoch: Current epoch
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint (must be implemented by subclass).

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Step number from checkpoint
        """
        raise NotImplementedError("load_checkpoint() must be implemented by subclass")

    def save_training_state(self, step: int, epoch: int, batch_idx: int):
        """
        Save training state (epoch progress, batch index, random state) to JSON file.

        This is saved separately from the model checkpoint to keep checkpoint files lightweight.
        Enables mid-epoch resume without re-processing already trained batches.

        Args:
            step: Current global step
            epoch: Current epoch (0-indexed)
            batch_idx: Current batch index within epoch (next batch to process)
        """
        import json
        import random
        import re

        # Extract short name from run_name (same logic as checkpoint saving)
        match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
        if match:
            short_name = match.group(1)
        else:
            short_name = self.run_name

        state_file = self.output_dir / f"{short_name}_step_{step}_state.json"

        state = {
            "global_step": step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "random_state": random.getstate(),  # Save Python random state for batch shuffle reproducibility
        }

        with open(state_file, 'w') as f:
            # Convert random_state tuple to list for JSON serialization
            state_serializable = state.copy()
            random_state = state["random_state"]
            state_serializable["random_state"] = {
                "version": random_state[0],
                "state": list(random_state[1]),  # Convert tuple to list
                "gauss_next": random_state[2],
            }
            json.dump(state_serializable, f, indent=2)

        print(f"{self.log_prefix} Saved training state to {state_file.name}")

    def load_training_state(self, step: int) -> Optional[dict]:
        """
        Load training state from JSON file.

        Args:
            step: Step number to load state for

        Returns:
            Dict with keys: global_step, epoch, batch_idx, random_state
            None if state file not found
        """
        import json
        import random
        import re

        # Extract short name from run_name (same logic as checkpoint saving)
        match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
        if match:
            short_name = match.group(1)
        else:
            short_name = self.run_name

        state_file = self.output_dir / f"{short_name}_step_{step}_state.json"

        if not state_file.exists():
            print(f"{self.log_prefix} No training state file found: {state_file.name}")
            return None

        with open(state_file, 'r') as f:
            state = json.load(f)

        # Restore random_state from serialized format
        random_state_dict = state["random_state"]
        state["random_state"] = (
            random_state_dict["version"],
            tuple(random_state_dict["state"]),  # Convert list back to tuple
            random_state_dict["gauss_next"],
        )

        print(f"{self.log_prefix} Loaded training state: epoch={state['epoch']}, batch_idx={state['batch_idx']}")
        return state

    def find_latest_checkpoint(self) -> Optional[Tuple[str, int]]:
        """
        Find the latest checkpoint in output directory.

        Returns:
            Tuple of (checkpoint_path, step) or None if no checkpoints exist
        """
        # Search for checkpoint files with pattern: {run_name}_step_{step}.safetensors
        checkpoint_files = list(self.output_dir.glob("*_step_*.safetensors"))

        # Search for training state files with pattern: {run_name}_step_{step}_state.json
        state_files = list(self.output_dir.glob("*_step_*_state.json"))

        if not checkpoint_files and not state_files:
            print(f"{self.log_prefix} No checkpoints found in {self.output_dir}")
            return None

        # Helper to extract step number from filename
        def get_step(path):
            try:
                # Extract step number from filename: {run_name}_step_{step}.safetensors or {run_name}_step_{step}_state.json
                # Split by "_step_" and take the next part (remove "_state" suffix if present)
                step_str = path.stem.split("_step_")[-1].replace("_state", "")
                return int(step_str)
            except (ValueError, IndexError):
                return 0

        # Find latest step from both sources
        latest_checkpoint_step = 0
        latest_checkpoint_path = None
        latest_state_step = 0

        if checkpoint_files:
            latest_checkpoint_path = max(checkpoint_files, key=get_step)
            latest_checkpoint_step = get_step(latest_checkpoint_path)

        if state_files:
            latest_state_path = max(state_files, key=get_step)
            latest_state_step = get_step(latest_state_path)

        # Debug: Print all checkpoints
        print(f"{self.log_prefix} Found checkpoint files:")
        for ckpt in sorted(checkpoint_files, key=get_step):
            step_num = get_step(ckpt)
            print(f"{self.log_prefix}   - {ckpt.name} → step {step_num}")

        print(f"{self.log_prefix} Found training state files:")
        for state in sorted(state_files, key=get_step):
            step_num = get_step(state)
            print(f"{self.log_prefix}   - {state.name} → step {step_num}")

        # Use the latest step (state.json takes priority as it represents interrupted training)
        if latest_state_step > latest_checkpoint_step:
            print(f"{self.log_prefix} Latest state.json: step {latest_state_step}")
            print(f"{self.log_prefix} Latest safetensors: step {latest_checkpoint_step}")
            print(f"{self.log_prefix} WARNING: State file is newer than checkpoint - this should not happen")
            print(f"{self.log_prefix} Using checkpoint step {latest_checkpoint_step}")
            step = latest_checkpoint_step
        else:
            step = max(latest_checkpoint_step, latest_state_step)

        if latest_checkpoint_path is None:
            print(f"{self.log_prefix} ERROR: No safetensors checkpoint found")
            return None

        print(f"{self.log_prefix} Selected latest checkpoint: {latest_checkpoint_path.name} (step {step})")
        return (str(latest_checkpoint_path), step)

    def _cleanup_old_checkpoints(self, max_step_saves_to_keep: int):
        """
        Delete old checkpoints, keeping only the most recent N checkpoints.

        Args:
            max_step_saves_to_keep: Maximum number of checkpoints to keep (0 = keep all)
        """
        if max_step_saves_to_keep <= 0:
            return

        # Find all checkpoint files
        checkpoint_files = list(self.output_dir.glob("*_step_*.safetensors"))
        if len(checkpoint_files) <= max_step_saves_to_keep:
            return

        # Sort by step number
        def get_step(path):
            try:
                step_str = path.stem.split("_step_")[-1]
                return int(step_str)
            except (ValueError, IndexError):
                return 0

        checkpoint_files.sort(key=get_step, reverse=True)

        # Delete old checkpoints
        checkpoints_to_delete = checkpoint_files[max_step_saves_to_keep:]
        for checkpoint_path in checkpoints_to_delete:
            # Also delete associated .pt file (optimizer state) and _state.json file (training state)
            pt_path = checkpoint_path.with_suffix(".pt")
            state_json_path = checkpoint_path.parent / f"{checkpoint_path.stem}_state.json"

            print(f"{self.log_prefix} Deleting old checkpoint: {checkpoint_path.name}")
            checkpoint_path.unlink()

            if pt_path.exists():
                print(f"{self.log_prefix} Deleting old optimizer state: {pt_path.name}")
                pt_path.unlink()

            if state_json_path.exists():
                print(f"{self.log_prefix} Deleting old training state: {state_json_path.name}")
                state_json_path.unlink()

    # ============================================================
    # Optimizer Setup
    # ============================================================

    def setup_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "constant",
        total_steps: int = 1000,
    ):
        """
        Setup optimizer and learning rate scheduler.

        Args:
            optimizer_type: Optimizer type (adamw, adamw8bit, adafactor, etc.)
            lr_scheduler_type: LR scheduler type (constant, cosine, etc.)
            total_steps: Total training steps
        """
        # Get trainable parameters from subclass
        param_groups = self.setup_trainable_parameters()

        print(f"{self.log_prefix} Setting up optimizer: {optimizer_type}")
        print(f"{self.log_prefix} LR scheduler: {lr_scheduler_type}")

        # Create optimizer using factory
        from .optimizer_factory import OptimizerFactory
        try:
            # Use hyperparameters from config, or fall back to defaults
            weight_decay = self.optimizer_weight_decay if self.optimizer_weight_decay is not None else 0.01
            beta1 = self.optimizer_beta1 if self.optimizer_beta1 is not None else 0.9
            beta2 = self.optimizer_beta2 if self.optimizer_beta2 is not None else 0.999
            eps = self.optimizer_epsilon if self.optimizer_epsilon is not None else 1e-8

            # Lion optimizers use 'lion_betas' kwarg instead of 'betas', and don't have epsilon
            optimizer_kwargs = {
                "weight_decay": weight_decay,
            }
            if "lion" in optimizer_type.lower():
                optimizer_kwargs["lion_betas"] = (beta1, beta2)
                # Lion doesn't use epsilon
            else:
                optimizer_kwargs["betas"] = (beta1, beta2)
                optimizer_kwargs["eps"] = eps

            # Pass cautious and Schedule-Free options to RingBuffer optimizers
            if "ringbuffer" in optimizer_type.lower():
                optimizer_kwargs["cautious"] = self.optimizer_cautious
                optimizer_kwargs["schedule_free"] = self.optimizer_schedule_free
                optimizer_kwargs["warmup_steps"] = self.optimizer_warmup_steps
                optimizer_kwargs["r"] = self.optimizer_schedule_free_r
                optimizer_kwargs["weight_lr_power"] = self.optimizer_schedule_free_weight_lr_power
                optimizer_kwargs["use_radam"] = self.optimizer_use_radam

            self.optimizer = OptimizerFactory.create_optimizer(
                optimizer_type=optimizer_type,
                params=param_groups,
                learning_rate=self.learning_rate,
                **optimizer_kwargs,
            )
        except (ValueError, ImportError) as e:
            print(f"{self.log_prefix} ERROR: {e}")
            print(f"{self.log_prefix} Falling back to AdamW")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                eps=1e-8,
            )

        # Setup LR scheduler
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        self.lr_scheduler = get_diffusers_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        # Setup fused backward/optimizer groups if Block Swap is enabled
        if self.blocks_to_swap > 0:
            if self.num_optimizer_groups > 0:
                # Validate compatibility: Block Swap + Fused Optimizer Groups + 8bit optimizer
                if optimizer_type.lower() in ["adamw8bit", "lion8bit", "adafactor8bit"]:
                    raise ValueError(
                        f"Block Swap + Fused Optimizer Groups is incompatible with 8-bit optimizers ({optimizer_type}). "
                        f"8-bit optimizers cannot handle CPU parameters that Block Swap creates. "
                        f"Options: (1) Use Adafactor without num_optimizer_groups (fused backward pass), "
                        f"(2) Use non-8bit optimizer (AdamW, Lion, etc.) with num_optimizer_groups, "
                        f"(3) Disable Block Swap (blocks_to_swap=0)"
                    )

                # Fused optimizer groups: works with non-8bit optimizers only
                self._setup_fused_optimizer_groups(optimizer_type, total_steps, lr_scheduler_type)
            elif optimizer_type.lower() in ["adafactor", "adamw8bit"]:
                # Fused backward pass: Adafactor or AdamW8bit
                self._setup_fused_backward_pass(optimizer_type)

    def _setup_fused_backward_pass(self, optimizer_type: str):
        """
        Setup fused backward pass for Block Swap compatibility.

        Registers post-accumulate-grad hooks that update parameters immediately
        after gradients are computed, before Block Swap moves them to CPU.

        Works with Adafactor or AdamW8bit optimizers (PyTorch 2.1+).

        Args:
            optimizer_type: Optimizer type ("adafactor" or "adamw8bit")
        """
        print(f"{self.log_prefix} Setting up fused backward pass for {optimizer_type}...")

        # Check PyTorch version
        import torch
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            print(f"{self.log_prefix} WARNING: PyTorch 2.1+ required for fused backward pass")
            print(f"{self.log_prefix} Current version: {torch.__version__}")
            print(f"{self.log_prefix} Fused backward pass disabled")
            return

        # Patch optimizer with step_param method
        if optimizer_type.lower() == "adafactor":
            from .optimizers.adafactor_fused import patch_adafactor_fused
            patch_adafactor_fused(self.optimizer)
        elif optimizer_type.lower() == "adamw8bit":
            from .optimizers.adamw8bit_fused import patch_adamw8bit_fused
            patch_adamw8bit_fused(self.optimizer)
        elif optimizer_type.lower() == "adamw8bit_ringbuffer":
            # AdamW8bit_RingBuffer has built-in hook support via patch_adamw8bit_ringbuffer
            from .optimizers.adamw8bit_ringbuffer import patch_adamw8bit_ringbuffer
            # Note: patch_adamw8bit_ringbuffer registers hooks itself, so we don't need the loop below
            patch_adamw8bit_ringbuffer(self.unet, self.optimizer)
            self.use_fused_backward = True
            print(f"{self.log_prefix} AdamW8bit_RingBuffer hooks registered via patch_adamw8bit_ringbuffer")
            return  # Skip the hook registration loop below
        elif optimizer_type.lower() == "lion8bit_ringbuffer":
            # Lion8bit_RingBuffer has built-in hook support via register_lion8bit_fused_backward
            from .optimizers.lion8bit_ringbuffer import register_lion8bit_fused_backward
            # Note: register_lion8bit_fused_backward registers hooks itself, so we don't need the loop below
            register_lion8bit_fused_backward(self.optimizer, self.unet)
            self.use_fused_backward = True
            print(f"{self.log_prefix} Lion8bit_RingBuffer hooks registered via register_lion8bit_fused_backward")
            return  # Skip the hook registration loop below

        # Register hooks for all trainable parameters
        hooks_registered = 0
        for param_group in self.optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, pg=param_group):
                        """Hook called when gradient is ready for this parameter"""
                        # Gradient clipping (if enabled)
                        # Note: We don't use max_grad_norm here because it's set to 0 by default
                        # and clipping is already handled elsewhere

                        # Update THIS parameter immediately (while on GPU)
                        self.optimizer.step_param(tensor, pg)

                        # Clear gradient to save memory
                        tensor.grad = None

                    # Register hook: called when gradient for THIS parameter is ready
                    parameter.register_post_accumulate_grad_hook(__grad_hook)
                    hooks_registered += 1

        self.use_fused_backward = True
        print(f"{self.log_prefix} Registered {hooks_registered} fused backward hooks")
        print(f"{self.log_prefix} Optimizer.step() and zero_grad() will be called by hooks automatically")

    def _setup_fused_optimizer_groups(self, optimizer_type: str, total_steps: int, lr_scheduler_type: str):
        """
        Setup fused optimizer groups for Block Swap compatibility.

        Works with ANY optimizer (AdamW, AdamW8bit, Lion8bit, etc.) by dividing
        parameters into groups and updating each group when all its gradients are ready.

        Args:
            optimizer_type: Optimizer type (adamw, adamw8bit, etc.)
            total_steps: Total training steps
            lr_scheduler_type: LR scheduler type
        """
        print(f"{self.log_prefix} Setting up fused optimizer groups...")

        # Check PyTorch version
        import torch
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            print(f"{self.log_prefix} WARNING: PyTorch 2.1+ required for fused optimizer groups")
            print(f"{self.log_prefix} Current version: {torch.__version__}")
            print(f"{self.log_prefix} Fused optimizer groups disabled")
            return

        # Get trainable parameters from current optimizer
        trainable_params = []
        for param_group in self.optimizer.param_groups:
            trainable_params.extend(param_group["params"])

        # Create multiple optimizers by dividing parameters into groups
        from .optimizers.fused_optimizer_groups import create_optimizer_groups, FusedOptimizerGroups

        optimizers = create_optimizer_groups(
            params=trainable_params,
            optimizer_type=optimizer_type,
            num_groups=self.num_optimizer_groups,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Replace self.optimizer with first optimizer (for compatibility)
        self.optimizer = optimizers[0]

        # Create LR schedulers for all optimizers
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        lr_schedulers = []
        for optimizer in optimizers:
            lr_scheduler = get_diffusers_scheduler(
                lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps,
            )
            lr_schedulers.append(lr_scheduler)

        # Replace self.lr_scheduler with first scheduler (for compatibility)
        self.lr_scheduler = lr_schedulers[0]

        # Store all schedulers for stepping
        self.lr_schedulers = lr_schedulers

        # Create FusedOptimizerGroups instance
        self.fused_optimizer_groups = FusedOptimizerGroups(
            optimizers=optimizers,
            max_grad_norm=0.0  # Gradient clipping handled by hook
        )

        # Register hooks
        self.fused_optimizer_groups.register_hooks()

        print(f"{self.log_prefix} Fused optimizer groups setup complete")
        print(f"{self.log_prefix} Optimizer.step() and zero_grad() will be called by hooks automatically")

    # ============================================================
    # Prompt Encoding
    # ============================================================

    def encode_prompt(self, prompt: str, requires_grad: bool = False):
        """
        Encode text prompt to embeddings with chunking support for long prompts (>75 tokens).

        Args:
            prompt: Text prompt to encode
            requires_grad: Whether to enable gradient computation for text encoders

        Returns:
            For SD1.5: text_embeddings tensor
            For SDXL: tuple of (text_embeddings, pooled_embeddings)
        """
        # Check prompt length - use tokenizer_2 for SDXL as it determines chunking
        tokenizer = self.tokenizer_2 if self.is_sdxl else self.tokenizer
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]

        # If prompt is short (<=75 tokens), use simple encoding
        if len(tokens) <= 75:
            return self._encode_prompt_simple(prompt, requires_grad)

        # Long prompt - use chunking
        return self._encode_prompt_chunked(prompt, requires_grad)

    def _encode_prompt_simple(self, prompt: str, requires_grad: bool = False):
        """
        Encode short prompt (<=75 tokens) using standard method.
        """
        if self.is_sdxl:
            # SDXL: Two text encoders
            text_inputs_1 = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_inputs_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

            with context_manager:
                # CRITICAL: Both text encoders must use hidden_states[-2] (penultimate layer)
                # This matches diffusers' StableDiffusionXLPipeline.encode_prompt() implementation
                encoder_output_1 = self.text_encoder(
                    text_inputs_1.input_ids.to(self.device),
                    output_hidden_states=True,
                )
                text_embeddings_1 = encoder_output_1.hidden_states[-2]

                encoder_output_2 = self.text_encoder_2(
                    text_inputs_2.input_ids.to(self.device),
                    output_hidden_states=True,
                )
                text_embeddings_2 = encoder_output_2.hidden_states[-2]
                pooled_embeddings = encoder_output_2[0]

                text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)

                return text_embeddings, pooled_embeddings
        else:
            # SD1.5: Single text encoder
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

            with context_manager:
                text_embeddings = self.text_encoder(
                    text_inputs.input_ids.to(self.device),
                )[0]

                return text_embeddings

    def _encode_prompt_chunked(self, prompt: str, requires_grad: bool = False):
        """
        Encode long prompt (>75 tokens) using chunking.
        Splits prompt into 75-token chunks and concatenates embeddings.
        """
        tokenizer = self.tokenizer_2 if self.is_sdxl else self.tokenizer
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]

        # Split tokens into 75-token chunks
        chunk_size = 75
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(chunk_tokens)

        # Limit chunks if max_prompt_chunks is set
        if self.max_prompt_chunks > 0 and len(chunks) > self.max_prompt_chunks:
            chunks = chunks[:self.max_prompt_chunks]

        # Encode each chunk
        chunk_embeds_list = []
        pooled_embeddings = None

        context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

        with context_manager:
            for idx, chunk_tokens in enumerate(chunks):
                # Decode tokens back to text
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                # Encode chunk
                if self.is_sdxl:
                    # SDXL: Encode with both text encoders
                    text_inputs_1 = self.tokenizer(
                        chunk_text,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_inputs_2 = self.tokenizer_2(
                        chunk_text,
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    encoder_output_1 = self.text_encoder(
                        text_inputs_1.input_ids.to(self.device),
                        output_hidden_states=True,
                    )
                    text_embeddings_1 = encoder_output_1.hidden_states[-2]

                    encoder_output_2 = self.text_encoder_2(
                        text_inputs_2.input_ids.to(self.device),
                        output_hidden_states=True,
                    )
                    text_embeddings_2 = encoder_output_2.hidden_states[-2]

                    # Use pooled embeddings from first chunk only
                    if idx == 0:
                        pooled_embeddings = encoder_output_2[0]

                    chunk_embeds = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)
                    chunk_embeds_list.append(chunk_embeds)
                else:
                    # SD1.5: Single text encoder
                    text_inputs = self.tokenizer(
                        chunk_text,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_embeddings = self.text_encoder(
                        text_inputs.input_ids.to(self.device),
                    )[0]

                    chunk_embeds_list.append(text_embeddings)

        # Concatenate chunks based on chunking mode
        if self.prompt_chunking_mode == "a1111":
            # A1111 mode: concatenate all chunks as-is
            text_embeddings = torch.cat(chunk_embeds_list, dim=1)
        elif self.prompt_chunking_mode == "sd_scripts":
            # sd-scripts mode: strip BOS/EOS between chunks
            processed_chunks = []
            for idx, chunk_emb in enumerate(chunk_embeds_list):
                if len(chunk_embeds_list) == 1:
                    processed_chunks.append(chunk_emb)
                elif idx == 0:
                    # First chunk: remove EOS (last token before padding)
                    processed_chunks.append(chunk_emb[:, :-1, :])
                elif idx == len(chunk_embeds_list) - 1:
                    # Last chunk: remove BOS (first token)
                    processed_chunks.append(chunk_emb[:, 1:, :])
                else:
                    # Middle chunks: remove both BOS and EOS
                    processed_chunks.append(chunk_emb[:, 1:-1, :])
            text_embeddings = torch.cat(processed_chunks, dim=1)
        else:  # nobos
            # NoBOS mode: strip all BOS/EOS tokens
            processed_chunks = []
            for chunk_emb in chunk_embeds_list:
                # Remove first (BOS) and last (EOS) tokens
                processed_chunks.append(chunk_emb[:, 1:-1, :])
            text_embeddings = torch.cat(processed_chunks, dim=1)

        if self.is_sdxl:
            return text_embeddings, pooled_embeddings
        else:
            return text_embeddings

    def encode_prompt_zimage(
        self,
        prompt: str,
        max_sequence_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt using Qwen3 text encoder with chat template (Z-Image).

        Args:
            prompt: Text prompt
            max_sequence_length: Maximum sequence length

        Returns:
            Tuple of (prompt_embeds, attention_mask)
        """
        # Format with Qwen chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # Tokenize
        text_inputs = self.tokenizer(
            formatted_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device).bool()

        # Encode with penultimate layer
        with torch.no_grad():
            encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            prompt_embeds = encoder_output.hidden_states[-2]

        # Extract and detach outputs
        result_embeds = prompt_embeds[0].detach()
        result_mask = attention_mask[0].detach()

        # Free intermediate tensors to prevent VRAM accumulation
        del input_ids, encoder_output, prompt_embeds, attention_mask

        return result_embeds, result_mask

    def encode_caption(self, caption: str, requires_grad: bool = False):
        """
        Unified caption encoding for all architectures.

        Returns:
            Tuple of (embeddings, auxiliary_data):
            - Z-Image: (prompt_embeds, attention_mask)
            - SD1.5: (text_embeddings, None)
            - SDXL: (text_embeddings, pooled_embeddings)
        """
        if self.is_zimage:
            return self.encode_prompt_zimage(caption)
        elif self.is_sdxl:
            text_emb, pooled_emb = self.encode_prompt(caption, requires_grad=requires_grad)
            return text_emb, pooled_emb
        else:
            text_emb = self.encode_prompt(caption, requires_grad=requires_grad)
            return text_emb, None

    # ============================================================
    # VRAM Management (swap mode for all architectures)
    # ============================================================

    def move_text_encoder_to_gpu(self):
        """Move Text Encoder(s) to GPU for encoding."""
        if self.text_encoder is not None:
            self.text_encoder.to(self.device)
        if self.is_sdxl and self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.device)

    def move_text_encoder_to_cpu(self):
        """Move Text Encoder(s) to CPU to free VRAM."""
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")
        if self.is_sdxl and self.text_encoder_2 is not None:
            self.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    def move_main_model_to_gpu(self):
        """Move main model (U-Net or Transformer) to GPU for training."""
        if self.is_zimage:
            if self.transformer_original is not None:
                self.transformer_original.to(self.device)
        else:
            if self.unet is not None:
                self.unet.to(self.device)

    def move_main_model_to_cpu(self):
        """Move main model (U-Net or Transformer) to CPU to free VRAM."""
        if self.is_zimage:
            if self.transformer_original is not None:
                self.transformer_original.to("cpu")
        else:
            if self.unet is not None:
                self.unet.to("cpu")
        torch.cuda.empty_cache()

    def move_vae_to_gpu(self):
        """Move VAE to GPU for encoding/decoding."""
        if self.vae is not None:
            self.vae.to(device=self.device, dtype=self.vae_dtype)

    def move_vae_to_cpu(self):
        """Move VAE to CPU to free VRAM."""
        if self.vae is not None:
            self.vae.to(device="cpu", dtype=self.vae_dtype)
        torch.cuda.empty_cache()

    # ============================================================
    # Image Encoding
    # ============================================================

    def encode_image(
        self,
        image: Image.Image,
        target_size: int = 512,
        target_width: int = None,
        target_height: int = None,
        bucket_strategy: str = "crop"
    ) -> torch.Tensor:
        """
        Encode image to latents.

        Args:
            image: PIL Image
            target_size: Square target size (deprecated, use target_width/height)
            target_width: Target width (for bucketing)
            target_height: Target height (for bucketing)
            bucket_strategy: Strategy for fitting image to target size
                - "resize": Direct resize (may distort aspect ratio, fastest)
                - "crop": Aspect ratio preserving resize + center crop (default)
                - "random_crop": Random crop at original resolution (no downscale, for tiled inference training)

        Returns:
            Latent tensor
        """
        image = image.convert("RGB")

        # Determine target dimensions
        if target_width is not None and target_height is not None:
            width, height = target_width, target_height
        else:
            width, height = target_size, target_size

        img_width, img_height = image.size

        if img_width * img_height > 5000 * 5000:
            print(f"[encode_image] Resizing large image {img_width}x{img_height} -> {width}x{height}")

        # Apply bucketing strategy
        if bucket_strategy == "resize":
            # Direct resize (may distort aspect ratio)
            image = image.resize((width, height), Image.LANCZOS)

        elif bucket_strategy == "crop":
            # Aspect ratio preserving resize + center crop (default)
            scale = max(width / img_width, height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            image = image.resize((new_width, new_height), Image.LANCZOS)

            # Center crop
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))

        elif bucket_strategy == "random_crop":
            # Random crop at original resolution (no resize)
            # Enables model to learn inference on partial regions of large images (for tiled inference)
            import random

            # If image is smaller than target, resize it first
            if img_width < width or img_height < height:
                scale = max(width / img_width, height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                image = image.resize((new_width, new_height), Image.LANCZOS)
                img_width, img_height = new_width, new_height

            # Random crop from original (or upscaled) resolution
            max_left = img_width - width
            max_top = img_height - height
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0
            image = image.crop((left, top, left + width, top + height))

        else:
            raise ValueError(f"Unknown bucket_strategy: {bucket_strategy}. Must be 'resize', 'crop', or 'random_crop'")

        if image.size != (width, height):
            print(f"[encode_image] ERROR: Final image size {image.size} != target {(width, height)}")

        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = (image_array - 0.5) * 2.0

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

        vae_device = next(self.vae.parameters()).device
        image_tensor = image_tensor.to(device=vae_device, dtype=self.vae.dtype)

        # Encode to latents
        with torch.no_grad():
            if self.is_zimage:
                # Z-Image VAE
                h = self.vae.encoder(image_tensor)
                if self.vae.quant_conv is not None:
                    h = self.vae.quant_conv(h)
                mean, logvar = torch.chunk(h, 2, dim=1)
                latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                shift_factor = self.vae.config.shift_factor if self.vae.config.shift_factor is not None else 0.0
                latents = self.vae.config.scaling_factor * (latents - shift_factor)
                # Clean up intermediate tensors
                del h, mean, logvar
            else:
                # SD/SDXL VAE
                encoder_output = self.vae.encode(image_tensor)
                latents = encoder_output.latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                # Clean up intermediate tensors
                del encoder_output

        # Clean up image_tensor before moving latents to CPU
        del image_tensor

        # Convert to training dtype and move to CPU immediately to free VRAM
        latents = latents.to(dtype=self.training_dtype, device='cpu')

        return latents

    # ============================================================
    # Training Step
    # ============================================================

    def train_step(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor = None,
        timesteps: Optional[torch.Tensor] = None,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        profile_vram: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform single training step (SD/SDXL).

        Args:
            latents: Image latents [B, C, H, W]
            text_embeddings: Text prompt embeddings
            pooled_embeddings: Pooled text embeddings (SDXL only)
            timesteps: Optional timesteps tensor. If None, sample uniformly from [0, num_train_timesteps)
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output
            profile_vram: If True, print VRAM usage

        Returns:
            (loss_tensor, loss_value) - Loss tensor with grad and scalar value
        """
        if profile_vram:
            print_vram_usage("[train_step] Start")

        # Sample noise
        noise = torch.randn_like(latents)

        if profile_vram:
            print_vram_usage("[train_step] After noise generation")

        # Sample random timestep (or use provided timesteps)
        batch_size = latents.shape[0]

        # Determine noise process from trainer config (set by train_runner.py)
        noise_process = getattr(self, 'noise_process', 'ddpm')  # Default: ddpm for backward compatibility

        if timesteps is None:
            if noise_process == "ddpm":
                # DDPM: sample discrete timesteps [0, num_train_timesteps)
                if self.timestep_sampler is not None:
                    # Use timestep sampler: sample from [0, 1] then scale to discrete timesteps
                    timesteps_continuous = self.timestep_sampler.sample(batch_size, self.device)
                    timesteps = (timesteps_continuous * self.noise_scheduler.config.num_train_timesteps).long()
                    timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
                else:
                    # Legacy behavior: sample uniformly from [0, num_train_timesteps)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (batch_size,),
                        device=self.device,
                    ).long()
            elif noise_process == "flow":
                # Flow Matching: sample continuous timesteps [0, 1]
                if self.timestep_sampler is not None:
                    # Use timestep sampler (already returns [0, 1])
                    timesteps = self.timestep_sampler.sample(batch_size, self.device)
                else:
                    # Uniform sampling from [0, 1]
                    timesteps = torch.rand((batch_size,), device=self.device)
        else:
            # MNT: timesteps provided externally
            if noise_process == "ddpm":
                # Convert flow-matching timesteps [0, 1] to discrete timesteps for DDPM
                # timesteps in [0, 1] -> scale to [0, num_train_timesteps)
                timesteps = (timesteps * self.noise_scheduler.config.num_train_timesteps).long()
                timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
            elif noise_process == "flow":
                # Flow matching: timesteps are already [0, 1]
                pass

        # Add noise to latents using unified framework
        noisy_latents = add_noise_unified(
            noise_process=noise_process,
            noise_scheduler=self.noise_scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )

        # Prepare added_cond_kwargs for SDXL
        added_cond_kwargs = None
        if self.is_sdxl and pooled_embeddings is not None:
            latent_height, latent_width = latents.shape[2], latents.shape[3]
            image_height, image_width = latent_height * 8, latent_width * 8

            original_size = (image_height, image_width)
            crops_coords_top_left = (0, 0)
            target_size = (image_height, image_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=pooled_embeddings.dtype, device=self.device)
            add_time_ids = add_time_ids.repeat(batch_size, 1)

            added_cond_kwargs = {
                "text_embeds": pooled_embeddings,
                "time_ids": add_time_ids
            }

        if profile_vram:
            print_vram_usage("[train_step] Before UNet forward")

        # Enable gradients for gradient checkpointing
        noisy_latents.requires_grad_(True)
        text_embeddings.requires_grad_(True)
        if pooled_embeddings is not None:
            pooled_embeddings.requires_grad_(True)

        # Predict noise using UNet
        if self.mixed_precision:
            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                if self.is_sdxl and added_cond_kwargs is not None:
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        text_embeddings,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                else:
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        text_embeddings
                    ).sample
        else:
            if self.is_sdxl and added_cond_kwargs is not None:
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    text_embeddings,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
            else:
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    text_embeddings
                ).sample

        if profile_vram:
            print_vram_usage("[train_step] After UNet forward")

        # Get target based on unified framework
        prediction_target = getattr(self, 'prediction_target', 'epsilon')  # Default: epsilon for backward compatibility
        target = get_target_unified(
            noise_process=noise_process,
            prediction_target=prediction_target,
            noise_scheduler=self.noise_scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )

        # Calculate loss (always in fp32)
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Apply Min-SNR gamma weighting
        if self.min_snr_gamma > 0:
            loss_per_sample_weighted = apply_snr_weight(loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma)
        else:
            loss_per_sample_weighted = loss_per_sample

        mse_loss = loss_per_sample_weighted.mean()

        # Add SNR and/or Energy regularization if enabled (can use both simultaneously)
        regularization_loss = torch.tensor(0.0, device=self.device)

        # Compute predicted latent once (used by both regularization losses)
        predicted_latent_for_reg = None
        if self.snr_regularization_loss is not None or self.energy_regularization_loss is not None:
            # Compute predicted latent from model_pred (keep gradients for backprop)
            predicted_latent_for_reg = predict_original_latent_unified(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noise_scheduler=self.noise_scheduler,
                noisy_latents=noisy_latents,
                model_pred=model_pred,
                timesteps=timesteps,
            )

        # SNR regularization (周波数領域の過剰デノイズ抑制)
        if self.snr_regularization_loss is not None:
            # Convert timesteps to continuous [0, 1] for regularization
            if noise_process == "ddpm":
                timesteps_continuous = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
            else:  # flow
                timesteps_continuous = timesteps.float()  # Already [0, 1]

            snr_reg_loss = self.snr_regularization_loss(
                predicted_latent_for_reg,
                latents,
                timesteps_continuous
            )
            regularization_loss = regularization_loss + snr_reg_loss

        # Energy regularization (空間領域のエネルギー保存)
        if self.energy_regularization_loss is not None:
            # Convert timesteps to continuous [0, 1] for regularization
            if noise_process == "ddpm":
                timesteps_continuous = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
            else:  # flow
                timesteps_continuous = timesteps.float()  # Already [0, 1]

            energy_reg_loss = self.energy_regularization_loss(
                predicted_latent_for_reg,
                latents,
                timesteps_continuous
            )
            regularization_loss = regularization_loss + energy_reg_loss

        # Total loss
        loss = mse_loss + regularization_loss

        # Calculate reconstruction loss for monitoring
        with torch.no_grad():
            # Reuse predicted_latent_for_reg if already computed, otherwise compute it
            if predicted_latent_for_reg is not None:
                predicted_latent_for_recon = predicted_latent_for_reg.detach()
            else:
                predicted_latent_for_recon = predict_original_latent_unified(
                    noise_process=noise_process,
                    prediction_target=prediction_target,
                    noise_scheduler=self.noise_scheduler,
                    noisy_latents=noisy_latents,
                    model_pred=model_pred,
                    timesteps=timesteps,
                )

            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
            recon_loss = recon_loss_per_sample.mean()

        if profile_vram:
            print_vram_usage("[train_step] After loss calculation")

        # Debug save if requested
        if debug_save_path is not None:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            timestep_value = timesteps[0].item()

            with torch.no_grad():
                predicted_latent = predict_original_latent_unified(
                    noise_process=noise_process,
                    prediction_target=prediction_target,
                    noise_scheduler=self.noise_scheduler,
                    noisy_latents=noisy_latents,
                    model_pred=model_pred,
                    timesteps=timesteps,
                )

            debug_data = {
                'latents': latents[0:1].detach().cpu(),
                'noisy_latents': noisy_latents[0:1].detach().cpu(),
                'predicted_noise': model_pred[0:1].detach().cpu(),
                'actual_noise': noise[0:1].detach().cpu(),
                'predicted_latent': predicted_latent[0:1].detach().cpu(),
                'timestep': timestep_value,
                'loss': loss_per_sample_weighted[0].item(),
                'loss_batch_mean': loss.item(),
                'loss_unweighted': loss_per_sample[0].item(),
                'recon_loss': recon_loss_per_sample[0].item(),
                'recon_loss_batch_mean': recon_loss.item(),
                'batch_size': batch_size,
                'min_snr_gamma': self.min_snr_gamma,
            }

            if debug_captions is not None and len(debug_captions) > 0:
                debug_data['caption'] = debug_captions[0]
                debug_data['all_captions'] = debug_captions

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:04d}.pt")
            del predicted_latent

        # Return loss tensor (with gradient) and reconstruction loss value
        # IMPORTANT: Do NOT call .item() on loss here - it breaks the computation graph!
        # The training loop will call .backward() on the loss tensor.
        recon_loss_value = recon_loss.item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        # But keep 'loss' tensor for backward pass
        del noise, noisy_latents, model_pred, recon_loss
        if self.is_sdxl and added_cond_kwargs is not None:
            del added_cond_kwargs

        return loss, recon_loss_value

    def train_step_zimage(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        profile_vram: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform single training step (Z-Image).

        Args:
            latents: Image latents [B, C, H, W]
            prompt_embeds: Prompt embeddings [B, seq_len, 2560]
            attention_mask: Attention mask [B, seq_len]
            timesteps: Timesteps for this batch [B]. If None, sampled uniformly from [0, 1]
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output
            profile_vram: If True, print VRAM usage

        Returns:
            Tuple of (loss tensor, reconstruction loss value)
        """
        if profile_vram:
            print_vram_usage("[train_step_zimage] Start")

        # Z-Image uses Flow Matching with velocity prediction
        noise_process = getattr(self, 'noise_process', 'flow')  # Z-Image default: flow
        prediction_target = getattr(self, 'prediction_target', 'velocity')  # Z-Image default: velocity

        # Sample random timesteps from [0, 1] if not provided
        batch_size = latents.shape[0]
        if timesteps is None:
            if self.timestep_sampler is not None:
                # Use timestep sampler (returns [0, 1] for flow matching)
                timesteps = self.timestep_sampler.sample(batch_size, self.device)
            else:
                # Legacy behavior: uniform sampling from [0, 1]
                timesteps = torch.rand(batch_size, device=self.device)

        # Sample noise (standard normal distribution)
        noise = torch.randn_like(latents)

        # Add noise using unified framework
        noisy_latents = add_noise_unified(
            noise_process=noise_process,
            noise_scheduler=self.noise_scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )

        if profile_vram:
            print_vram_usage("[train_step_zimage] Before Transformer forward")

        # Note: Gradient checkpointing automatically manages requires_grad
        # No need to manually set requires_grad_(True) - PyTorch handles this
        # prompt_embeds is always detached (from encode_prompt_zimage with no_grad)
        # attention_mask is bool type, does not need gradients

        # Add frame dimension for Z-Image: [B, C, H, W] -> [B, C, 1, H, W]
        noisy_latents_4d = noisy_latents.unsqueeze(2)

        # Predict velocity using Z-Image Transformer
        if self.mixed_precision:
            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                model_pred, _ = self.transformer(
                    x=noisy_latents_4d,
                    t=timesteps,
                    cap_feats=prompt_embeds,
                    cap_mask=attention_mask,
                )
        else:
            model_pred, _ = self.transformer(
                x=noisy_latents_4d,
                t=timesteps,
                cap_feats=prompt_embeds,
                cap_mask=attention_mask,
            )

        # Remove frame dimension: [B, C, 1, H, W] -> [B, C, H, W]
        model_pred = model_pred.squeeze(2)

        if profile_vram:
            print_vram_usage("[train_step_zimage] After Transformer forward")

        # Get target using unified framework
        target = get_target_unified(
            noise_process=noise_process,
            prediction_target=prediction_target,
            noise_scheduler=self.noise_scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )

        # Calculate MSE loss (always in fp32)
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Flow Matching doesn't use Min-SNR weighting (uniform timestep distribution)
        mse_loss = loss_per_sample.mean()

        # Add SNR and/or Energy regularization if enabled (can use both simultaneously)
        regularization_loss = torch.tensor(0.0, device=self.device)

        # Compute predicted latent once (used by both regularization losses)
        predicted_latent_for_reg = None
        if self.snr_regularization_loss is not None or self.energy_regularization_loss is not None:
            # Compute predicted latent using unified framework
            predicted_latent_for_reg = predict_original_latent_unified(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noise_scheduler=self.noise_scheduler,
                noisy_latents=noisy_latents,
                model_pred=model_pred,
                timesteps=timesteps,
            )

        # SNR regularization (周波数領域の過剰デノイズ抑制)
        if self.snr_regularization_loss is not None:
            # timesteps are already [0, 1] for flow matching
            snr_reg_loss = self.snr_regularization_loss(
                predicted_latent_for_reg,
                latents,
                timesteps
            )
            regularization_loss = regularization_loss + snr_reg_loss

        # Energy regularization (空間領域のエネルギー保存)
        if self.energy_regularization_loss is not None:
            energy_reg_loss = self.energy_regularization_loss(
                predicted_latent_for_reg,
                latents,
                timesteps
            )
            regularization_loss = regularization_loss + energy_reg_loss

        # Total loss
        loss = mse_loss + regularization_loss

        # Calculate reconstruction loss
        with torch.no_grad():
            # Reuse predicted_latent_for_reg if already computed, otherwise compute it
            if predicted_latent_for_reg is not None:
                predicted_latent_for_recon = predicted_latent_for_reg.detach()
            else:
                predicted_latent_for_recon = predict_original_latent_unified(
                    noise_process=noise_process,
                    prediction_target=prediction_target,
                    noise_scheduler=self.noise_scheduler,
                    noisy_latents=noisy_latents,
                    model_pred=model_pred,
                    timesteps=timesteps,
                )

            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
            recon_loss = recon_loss_per_sample.mean()

        if profile_vram:
            print_vram_usage("[train_step_zimage] After loss calculation")

        # Debug save if requested
        if debug_save_path is not None:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            timestep_value = timesteps[0].item()

            with torch.no_grad():
                predicted_latent = predict_original_latent_unified(
                    noise_process=noise_process,
                    prediction_target=prediction_target,
                    noise_scheduler=self.noise_scheduler,
                    noisy_latents=noisy_latents,
                    model_pred=model_pred,
                    timesteps=timesteps,
                )

            debug_data = {
                'latents': latents[0:1].detach().cpu(),
                'noisy_latents': noisy_latents[0:1].detach().cpu(),
                'predicted_velocity': model_pred[0:1].detach().cpu(),
                'actual_velocity': target[0:1].detach().cpu(),
                'predicted_latent': predicted_latent[0:1].detach().cpu(),
                'timestep': timestep_value,
                'loss': loss_per_sample[0].item(),
                'loss_batch_mean': loss.item(),
                'recon_loss': recon_loss_per_sample[0].item(),
                'recon_loss_batch_mean': recon_loss.item(),
                'batch_size': batch_size,
                'scheduler_type': 'FlowMatching',
            }

            if debug_captions is not None and len(debug_captions) > 0:
                debug_data['caption'] = debug_captions[0]
                debug_data['all_captions'] = debug_captions

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:.4f}.pt")
            del predicted_latent

        # Return loss tensor (with gradient) and reconstruction loss value
        # IMPORTANT: Do NOT call .item() on loss here - it breaks the computation graph!
        # The training loop will call .backward() on the loss tensor.
        recon_loss_value = recon_loss.item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        # But keep 'loss' tensor for backward pass
        del noise, noisy_latents, noisy_latents_4d, model_pred, target
        del loss_per_element, loss_per_sample, recon_loss_per_element, recon_loss_per_sample, recon_loss

        return loss, recon_loss_value

    # ============================================================
    # Sample Generation (to be continued in next section)
    # ============================================================

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
    ) -> "Image.Image":
        """
        Generate sample image during training (SD/SDXL).
        Uses custom_sampling_loop() - EXACTLY the same method as normal txt2img generation.

        Args:
            prompt: Text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed (-1 for random)
            current_step: Current training step (for logging)
            schedule_type: Timestep schedule type (uniform, karras, exponential)

        Returns:
            PIL Image
        """
        from PIL import Image
        from core.inference.custom_sampling import custom_sampling_loop
        from core.inference.schedulers import get_scheduler
        import random

        print(f"{self.log_prefix} Generating sample: {prompt[:50]}...")

        # Set models to eval mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()

        # Debug: Check if LoRA is applied to U-Net
        lora_layers_found = 0
        for name, module in self.unet.named_modules():
            if hasattr(module, 'lora_down') or 'LoRA' in type(module).__name__:
                lora_layers_found += 1
        print(f"{self.log_prefix} [Sample] U-Net has {lora_layers_found} LoRA layers")

        try:
            # ========================================
            # STEP 1: Create Temporary Pipeline Object
            # ========================================
            # custom_sampling_loop() requires a pipeline object with scheduler, unet, vae, etc.
            # Create a minimal pipeline-like object with necessary components

            if self.is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                # Create a minimal pipeline object
                class TempPipeline:
                    def __init__(self, unet, vae, text_encoder, text_encoder_2, scheduler, tokenizer, tokenizer_2):
                        self.unet = unet
                        self.vae = vae
                        self.text_encoder = text_encoder
                        self.text_encoder_2 = text_encoder_2
                        self.scheduler = scheduler
                        self.tokenizer = tokenizer
                        self.tokenizer_2 = tokenizer_2
                        # Set default config
                        self.vae_scale_factor = 8
                        self.image_processor = None  # Not needed for custom_sampling_loop

                # Map schedule_type (sgm_uniform -> uniform)
                schedule_type_mapped = schedule_type
                if schedule_type == "sgm_uniform":
                    schedule_type_mapped = "uniform"

                # Create scheduler using get_scheduler()
                class SchedulerContainer:
                    def __init__(self, scheduler):
                        self.scheduler = scheduler

                scheduler_container = SchedulerContainer(self.original_scheduler)
                scheduler = get_scheduler(
                    pipeline=scheduler_container,
                    sampler="euler",
                    schedule_type=schedule_type_mapped
                )

                # Create temporary pipeline
                pipeline = TempPipeline(
                    unet=self.unet,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    text_encoder_2=self.text_encoder_2,
                    scheduler=scheduler,
                    tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2
                )
            else:
                from diffusers import StableDiffusionPipeline
                # Create a minimal pipeline object for SD1.5
                class TempPipeline:
                    def __init__(self, unet, vae, text_encoder, scheduler, tokenizer):
                        self.unet = unet
                        self.vae = vae
                        self.text_encoder = text_encoder
                        self.scheduler = scheduler
                        self.tokenizer = tokenizer
                        # Set default config
                        self.vae_scale_factor = 8
                        self.image_processor = None  # Not needed for custom_sampling_loop

                # Map schedule_type (sgm_uniform -> uniform)
                schedule_type_mapped = schedule_type
                if schedule_type == "sgm_uniform":
                    schedule_type_mapped = "uniform"

                # Create scheduler using get_scheduler()
                class SchedulerContainer:
                    def __init__(self, scheduler):
                        self.scheduler = scheduler

                scheduler_container = SchedulerContainer(self.original_scheduler)
                scheduler = get_scheduler(
                    pipeline=scheduler_container,
                    sampler="euler",
                    schedule_type=schedule_type_mapped
                )

                # Create temporary pipeline
                pipeline = TempPipeline(
                    unet=self.unet,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    scheduler=scheduler,
                    tokenizer=self.tokenizer
                )

            # ========================================
            # STEP 2: Text Encoding
            # ========================================
            self.move_text_encoder_to_gpu()

            # Encode prompt
            if self.is_sdxl:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt("", requires_grad=False)
            else:
                prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds = self.encode_prompt("", requires_grad=False)
                pooled_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            # Pad negative embeddings to match positive embeddings sequence length (for prompt chunking)
            if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                # Positive prompt has more tokens (chunking applied)
                # Pad negative embeddings with zeros to match
                seq_len_diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
                padding = torch.zeros(
                    (negative_prompt_embeds.shape[0], seq_len_diff, negative_prompt_embeds.shape[2]),
                    dtype=negative_prompt_embeds.dtype,
                    device=negative_prompt_embeds.device
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)
                print(f"{self.log_prefix} [Sample] Padded negative embeddings: {negative_prompt_embeds.shape[1] - seq_len_diff} -> {negative_prompt_embeds.shape[1]} tokens")

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
            # STEP 4: Call custom_sampling_loop (SAME as pipeline.generate_txt2img)
            # ========================================
            self.move_main_model_to_gpu()
            self.move_vae_to_gpu()

            # Detect v-prediction and apply guidance_rescale if needed
            is_v_prediction = pipeline.scheduler.config.get("prediction_type") == "v_prediction"
            guidance_rescale = 0.7 if is_v_prediction else 0.0

            print(f"{self.log_prefix} [Sample] Using custom_sampling_loop()")
            print(f"{self.log_prefix} [Sample] Scheduler: {type(pipeline.scheduler).__name__}")
            print(f"{self.log_prefix} [Sample] V-prediction: {is_v_prediction}, guidance_rescale: {guidance_rescale}")

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
                ancestral_generator=None,  # Not needed for training samples
                latents=None,
                prompt_embeds_callback=None,  # No prompt editing for training samples
                progress_callback=None,
                step_callback=None,
                developer_mode=False,
                cfg_schedule_type="constant",  # Simple constant CFG for training samples
                cfg_schedule_min=1.0,
                cfg_schedule_max=None,
                cfg_schedule_power=2.0,
                cfg_rescale_snr_alpha=0.0,
                dynamic_threshold_percentile=0.0,
                dynamic_threshold_mimic_scale=1.0,
                nag_enable=False,  # No NAG for training samples
                nag_scale=5.0,
                nag_tau=3.5,
                nag_alpha=0.25,
                nag_sigma_end=0.0,
                nag_negative_prompt_embeds=None,
                nag_negative_pooled_prompt_embeds=None,
                attention_type="normal",  # Normal attention for training samples
            )

            # Move models back to CPU
            self.move_main_model_to_cpu()
            self.move_vae_to_cpu()
            torch.cuda.empty_cache()

            print(f"{self.log_prefix} Sample generated successfully (seed: {actual_seed})")
            return image

        finally:
            # Restore training mode
            self.unet.train()
            self.vae.train()
            self.text_encoder.train()
            if self.text_encoder_2 is not None:
                self.text_encoder_2.train()

            # Ensure U-Net is back on GPU for training continuation
            self.move_main_model_to_gpu()

    def _generate_sample_zimage(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
    ) -> Image.Image:
        """
        Generate sample image during training (Z-Image).

        Args:
            prompt: Text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed (-1 for random)

        Returns:
            PIL Image
        """
        print(f"{self.log_prefix} Generating Z-Image sample: {prompt[:50]}...")

        # Set models to eval mode for inference (same as lora_trainer.py.backup:2481-2484)
        self.transformer.eval()
        self.transformer_original.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Store original devices for restoration
        text_encoder_device = next(self.text_encoder.parameters()).device
        vae_device = next(self.vae.parameters()).device
        transformer_device = next(self.transformer_original.parameters()).device

        try:
            # ============================================================
            # Stage 0: Offload Transformer AND Optimizer State to CPU
            # ============================================================
            print(f"{self.log_prefix} [Sample] Offloading Transformer and Optimizer state to CPU")

            # Move Transformer to CPU
            self.transformer_original.to("cpu")

            # CRITICAL: Move Optimizer state (gradients, momentum) to CPU
            # Optimizer state (exp_avg, exp_avg_sq) stays on GPU even after model.to(cpu)
            # This can consume 2x model size in VRAM (for AdamW: exp_avg + exp_avg_sq)
            optimizer_state_dict = self.optimizer.state_dict()
            for param_id, state in optimizer_state_dict['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                        state[key] = value.cpu()
            self.optimizer.load_state_dict(optimizer_state_dict)

            torch.cuda.empty_cache()
            print(f"{self.log_prefix} [Sample] Transformer and Optimizer state offloaded to CPU")

            # ============================================================
            # Stage 1: Text Encoding (Sequential Offloading Pattern)
            # ============================================================
            # Move Text Encoder to GPU for encoding
            if text_encoder_device != self.device:
                print(f"{self.log_prefix} [Sample] Moving Text Encoder to GPU for encoding")
                self.text_encoder.to(self.device)

            # Encode prompt
            prompt_embeds, attention_mask = self.encode_prompt_zimage(prompt)

            # Encode unconditional prompt only if CFG is enabled
            if guidance_scale > 1.0:
                uncond_embeds, uncond_mask = self.encode_prompt_zimage("")
            else:
                uncond_embeds, uncond_mask = None, None

            # Move Text Encoder back to CPU to free VRAM
            if text_encoder_device != self.device:
                print(f"{self.log_prefix} [Sample] Moving Text Encoder back to CPU")
                self.text_encoder.to(text_encoder_device)
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Move Transformer back to GPU for denoising
            # ============================================================
            print(f"{self.log_prefix} [Sample] Moving Transformer to GPU for denoising")
            self.transformer_original.to(transformer_device)
            torch.cuda.empty_cache()

            # Add batch dimension
            prompt_embeds = prompt_embeds.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if uncond_embeds is not None:
                uncond_embeds = uncond_embeds.unsqueeze(0)
                uncond_mask = uncond_mask.unsqueeze(0)

            # ============================================================
            # Stage 2: Denoising Loop (Transformer already on GPU from training)
            # ============================================================
            print(f"{self.log_prefix} [Sample] Running denoising loop (Transformer on GPU)")

            # Prepare latents with seed
            latent_height = height // 8
            latent_width = width // 8
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            # Use FP32 for latents initialization (same as pipeline.py for numerical stability)
            latents = torch.randn(
                (1, self.vae.config.latent_channels, latent_height, latent_width),
                device=self.device,
                dtype=torch.float32,
                generator=generator,
            )

            # Setup scheduler (create new instance with same config)
            # Note: We cannot use from_config() because Z-Image scheduler.config is not a standard ConfigMixin
            inference_scheduler = type(self.scheduler)(
                num_train_timesteps=self.scheduler.config.get("num_train_timesteps", 1000),
                shift=self.scheduler.config.get("shift", 1.0),
                use_dynamic_shifting=self.scheduler.config.get("use_dynamic_shifting", False),
            )

            # Calculate dynamic shift for flow matching (same as pipeline.py:964-981)
            from core.zimage_utils import calculate_shift
            image_seq_len = (latent_height // 2) * (latent_width // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

            # Set scheduler parameters (same as pipeline.py:977-981)
            inference_scheduler.sigma_min = 0.0
            inference_scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)

            # Denoising loop
            latents = self._run_zimage_denoising_loop(
                latents=latents,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
                guidance_scale=guidance_scale,
                scheduler=inference_scheduler,
            )

            # Free prompt embeddings
            del prompt_embeds, attention_mask
            if uncond_embeds is not None:
                del uncond_embeds, uncond_mask

            # ============================================================
            # Stage 3: Offload Transformer to CPU, move VAE to GPU
            # ============================================================
            # Move Transformer to CPU to free VRAM for VAE decode
            print(f"{self.log_prefix} [Sample] Moving Transformer to CPU to free VRAM")
            self.transformer_original.to("cpu")
            torch.cuda.empty_cache()

            # Move VAE to GPU for decoding
            if vae_device != self.device:
                print(f"{self.log_prefix} [Sample] Moving VAE to GPU for decoding")
                self.vae.to(device=self.device, dtype=self.vae_dtype)

            # Decode latents
            image = self._decode_zimage_latents(latents)

            # Move VAE back to CPU
            if vae_device != self.device:
                print(f"{self.log_prefix} [Sample] Moving VAE back to CPU")
                self.vae.to(device=vae_device, dtype=self.vae_dtype)

            # Free latents
            del latents
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 4: Restore Transformer and Optimizer State to GPU
            # ============================================================
            print(f"{self.log_prefix} [Sample] Restoring Transformer and Optimizer state to GPU")

            # Move Transformer back to GPU
            self.transformer_original.to(transformer_device)

            # CRITICAL: Move Optimizer state back to GPU (skip for Ring Buffer optimizers)
            # AdamW8bit_RingBuffer and Lion8bit_RingBuffer keep states on CPU intentionally
            from .optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer
            from .optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer
            if not isinstance(self.optimizer, (AdamW8bit_RingBuffer, Lion8bit_RingBuffer)):
                # Optimizer state must be on the same device as model parameters for training
                optimizer_state_dict = self.optimizer.state_dict()
                for param_id, state in optimizer_state_dict['state'].items():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                            state[key] = value.to(transformer_device)
                self.optimizer.load_state_dict(optimizer_state_dict)
                print(f"{self.log_prefix} [Sample] Optimizer state restored to GPU")
            else:
                print(f"{self.log_prefix} [Sample] Optimizer state kept on CPU (Ring Buffer)")

            torch.cuda.empty_cache()
            print(f"{self.log_prefix} [Sample] Transformer restored to GPU")

            return image

        finally:
            # Ensure all models are back to their original devices (safety fallback)
            # Text Encoder and VAE should already be on CPU from sequential offloading
            # Transformer should already be on GPU from restoration
            # But we check anyway in case of exceptions during sample generation

            # Restore models to train mode (same as lora_trainer.py.backup:2638-2639)
            self.transformer.train()
            self.transformer_original.train()

    def _run_zimage_denoising_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_mask: torch.Tensor,
        guidance_scale: float,
        scheduler,
    ) -> torch.Tensor:
        """Run Z-Image denoising loop for sample generation.

        Note: Uses transformer_original (not batched wrapper) for single-image inference.
        """
        with torch.no_grad():
            for i, t in enumerate(tqdm(scheduler.timesteps, desc="Generating")):
                # Check for stop flag during sample generation (allow graceful shutdown)
                stop_flag_file = self.output_dir / ".stop_training"
                if stop_flag_file.exists():
                    print(f"\n{self.log_prefix} [Sample] Stop flag detected during sample generation, aborting...")
                    raise KeyboardInterrupt("Training stopped by user during sample generation")

                # Skip last step if t=0 (flow matching termination, same as pipeline.py:1001-1004)
                if t == 0 and i == len(scheduler.timesteps) - 1:
                    continue

                # Prepare input
                if guidance_scale > 1.0:
                    latent_input = torch.cat([latents] * 2)
                    embeds_input = torch.cat([uncond_embeds, prompt_embeds])
                    mask_input = torch.cat([uncond_mask, attention_mask])
                else:
                    latent_input = latents
                    embeds_input = prompt_embeds
                    mask_input = attention_mask

                # Predict noise (use original transformer for single-image inference)
                # Use inference interface: List[Tensor] format, positional args only (no cap_mask)

                # Prepare timestep (expand to batch size, same as inference pipeline)
                timestep = t.to(self.device).expand(latent_input.shape[0])

                # Normalize timestep to [0, 1] (Z-Image expects normalized timesteps)
                timestep = (1000 - timestep) / 1000

                # Convert latents to transformer dtype (same as inference pipeline:1037-1046)
                transformer_dtype = next(self.transformer_original.parameters()).dtype
                latent_input = latent_input.to(transformer_dtype)

                # Add channel dimension and convert to list (same as inference pipeline)
                latent_input_5d = latent_input.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
                latent_input_list = list(latent_input_5d.unbind(dim=0))  # List of [C, 1, H, W]

                # Convert embeddings to list (each item: [seq_len, 2560])
                embeds_input_list = list(embeds_input.unbind(dim=0))

                # Call transformer (inference interface: positional args, List format)
                model_out_list = self.transformer_original(
                    latent_input_list,
                    timestep,
                    embeds_input_list,
                )[0]

                # Apply CFG if enabled (same as stable lora_trainer.py:2474-2492)
                batch_size = latents.shape[0]
                if guidance_scale > 1.0:
                    # CFG output order matches input: [negative, positive]
                    neg_out = model_out_list[:batch_size]  # negative (uncond)
                    pos_out = model_out_list[batch_size:]  # positive (cond)
                    noise_pred = []
                    for j in range(batch_size):
                        neg = neg_out[j].float()
                        pos = pos_out[j].float()
                        # Standard CFG formula (consistent with stable version)
                        # pred = uncond + guidance_scale * (cond - uncond)
                        pred = neg + guidance_scale * (pos - neg)
                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

                # Remove frames dimension for scheduler (5D → 4D) and negate (same as stable version)
                noise_pred = -noise_pred.squeeze(2)

                # Denoise step
                latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

        return latents

    def _decode_zimage_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode Z-Image latents to image."""
        # Unscale latents
        shift_factor = self.vae.config.shift_factor if self.vae.config.shift_factor is not None else 0.0
        latents = (latents / self.vae.config.scaling_factor) + shift_factor

        # Decode (convert to VAE dtype to match decoder weights)
        with torch.no_grad():
            latents = latents.to(self.vae.dtype)
            if self.vae.post_quant_conv is not None:
                latents = self.vae.post_quant_conv(latents)
            image = self.vae.decoder(latents)

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).astype(np.uint8)[0]

        return Image.fromarray(image)

    # ============================================================
    # Latent Cache Management (to be added in continuation)
    # ============================================================

    def _setup_latent_caches(self, datasets: List[Any]) -> Dict[str, Any]:
        """
        Setup per-dataset latent caches.

        Args:
            datasets: List of dataset objects

        Returns:
            Dictionary mapping dataset_unique_id to LatentCache instance
        """
        from core.training.latent_cache import LatentCache, get_cache_base_dir

        latent_caches = {}
        # Use global cache directory (shared across all training runs)
        # This allows cache reuse when training the same dataset multiple times
        base_cache_dir = get_cache_base_dir()
        print(f"{self.log_prefix} Using global latent cache directory: {base_cache_dir}")

        for dataset in datasets:
            latent_caches[dataset.unique_id] = LatentCache(
                dataset_unique_id=dataset.unique_id,
                base_cache_dir=str(base_cache_dir),
            )
            cache_dir = Path(base_cache_dir) / dataset.unique_id
            print(f"{self.log_prefix} Setup latent cache for dataset '{dataset.unique_id}': {cache_dir}")

        return latent_caches

    def _validate_and_generate_latent_caches(
        self,
        datasets: List[Any],
        latent_caches: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        force_recache: bool = False,
    ):
        """
        Check latent caches and generate missing ones.

        Args:
            datasets: List of dataset objects
            latent_caches: Dictionary of latent caches
            progress_callback: Progress callback function
            force_recache: Force regenerate all caches even if they exist
        """
        if force_recache:
            print(f"{self.log_prefix} Force recache enabled: regenerating all latent caches...")
        else:
            print(f"{self.log_prefix} Checking and generating missing latent caches...")

        # Generate missing latents (this will skip already cached items unless force_recache=True)
        self._generate_missing_latents_with_model_offload(
            datasets=datasets,
            latent_caches=latent_caches,
            progress_callback=progress_callback,
            force_recache=force_recache,
        )

    def _generate_missing_latents_with_model_offload(
        self,
        datasets: List[Any],
        latent_caches: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        force_recache: bool = False,
    ):
        """
        Generate missing latents with model offloading for memory efficiency.

        Args:
            datasets: List of dataset objects
            latent_caches: Dictionary of latent caches
            progress_callback: Progress callback function
        """
        print(f"{self.log_prefix} Generating latent cache with model offloading...")

        # Count total items
        total_items = sum(len(dataset.items) for dataset in datasets)
        processed_items = 0

        # Move VAE to GPU (only if not already there)
        vae_current_device = next(self.vae.parameters()).device
        if vae_current_device != self.device:
            print(f"{self.log_prefix} Moving VAE from {vae_current_device} to {self.device}...")
            self.vae.to(device=self.device, dtype=self.vae_dtype)
        else:
            print(f"{self.log_prefix} VAE already on {self.device}, skipping move")

        iteration_count = 0
        for dataset in datasets:
            cache = latent_caches[dataset.unique_id]

            for item in tqdm(dataset.items, desc=f"Caching {dataset.unique_id}"):
                # Check if already cached (skip if force_recache is False)
                image_path = item["image_path"]
                width = item["width"]
                height = item["height"]

                if not force_recache and cache.has_latent(image_path, width, height):
                    processed_items += 1
                    continue

                # Load and encode image
                try:
                    image = Image.open(image_path)

                    latent = self.encode_image(
                        image=image,
                        target_width=width,
                        target_height=height,
                    )

                    # Save to cache
                    cache.save_latent(
                        image_path=image_path,
                        width=width,
                        height=height,
                        latents=latent,
                    )

                    iteration_count += 1

                except Exception as e:
                    # Use repr() to avoid UnicodeEncodeError on Windows (cp932)
                    safe_path = os.path.basename(image_path)
                    try:
                        print(f"{self.log_prefix} ERROR encoding {safe_path}: {e}")
                    except UnicodeEncodeError:
                        # Fallback: encode-safe output
                        print(f"{self.log_prefix} ERROR encoding image (path contains non-ASCII chars): {e}")
                finally:
                    # Clean up to prevent VRAM accumulation
                    if 'image' in locals():
                        image.close()
                        del image
                    if 'latent' in locals():
                        del latent
                    # Clear CUDA cache periodically (every 50 images)
                    if iteration_count % 50 == 0:
                        torch.cuda.empty_cache()

                processed_items += 1

                # Progress callback
                if progress_callback:
                    progress_callback(
                        phase="latent_cache",
                        step=processed_items,
                        total=total_items,
                    )

        # VAE stays on CPU (already there)
        print(f"{self.log_prefix} Latent cache generation complete ({iteration_count} images encoded)")

    def _regenerate_single_latent(
        self,
        image_path: str,
        width: int,
        height: int,
        cache: Any,
        latent_caches: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Regenerate a single latent on-the-fly during training with model offloading.

        This is called when latent cache is corrupted or has shape mismatch.
        Offloads training components temporarily and loads VAE to GPU.

        Args:
            image_path: Path to source image
            width: Target width
            height: Target height
            cache: LatentCache object
            latent_caches: Dictionary of all latent caches (unused, for future use)

        Returns:
            Regenerated latent tensor
        """
        print(f"{self.log_prefix} [Latent Regeneration] Offloading models...")

        # Save current device states
        if self.is_zimage:
            transformer_device = next(self.transformer.parameters()).device
            text_encoder_device = next(self.text_encoder.parameters()).device
        else:
            unet_device = next(self.unet.parameters()).device
            if self.text_encoder:
                text_encoder_device = next(self.text_encoder.parameters()).device
            if self.is_sdxl and self.text_encoder_2:
                text_encoder_2_device = next(self.text_encoder_2.parameters()).device

        vae_device = next(self.vae.parameters()).device

        try:
            # Offload training components to CPU
            if self.is_zimage:
                if transformer_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Transformer to CPU...")
                    self.transformer.to('cpu')
                if text_encoder_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Text Encoder to CPU...")
                    self.text_encoder.to('cpu')
            else:
                if unet_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving U-Net to CPU...")
                    self.unet.to('cpu')
                if self.text_encoder and text_encoder_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Text Encoder to CPU...")
                    self.text_encoder.to('cpu')
                if self.is_sdxl and self.text_encoder_2 and text_encoder_2_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Text Encoder 2 to CPU...")
                    self.text_encoder_2.to('cpu')

            torch.cuda.empty_cache()

            # Move VAE to GPU
            if vae_device != self.device:
                print(f"{self.log_prefix} [Latent Regeneration] Moving VAE to GPU...")
                self.vae.to(device=self.device, dtype=self.vae_dtype)

            # Load and encode image
            print(f"{self.log_prefix} [Latent Regeneration] Encoding image: {image_path}")
            image = Image.open(image_path)
            latent = self.encode_image(
                image=image,
                target_width=width,
                target_height=height,
            )
            image.close()

            # Save to cache
            cache.save_latent(
                image_path=image_path,
                width=width,
                height=height,
                latents=latent,
            )
            print(f"{self.log_prefix} [Latent Regeneration] Latent regenerated and saved to cache")

        finally:
            # Restore original device states
            print(f"{self.log_prefix} [Latent Regeneration] Restoring models...")
            if self.is_zimage:
                if transformer_device != torch.device('cpu'):
                    self.transformer.to(transformer_device)
                if text_encoder_device != torch.device('cpu'):
                    self.text_encoder.to(text_encoder_device)
            else:
                if unet_device != torch.device('cpu'):
                    self.unet.to(unet_device)
                if self.text_encoder and text_encoder_device != torch.device('cpu'):
                    self.text_encoder.to(text_encoder_device)
                if self.is_sdxl and self.text_encoder_2 and text_encoder_2_device != torch.device('cpu'):
                    self.text_encoder_2.to(text_encoder_2_device)

            if vae_device != self.device:
                self.vae.to(device=vae_device, dtype=self.vae_dtype)

            torch.cuda.empty_cache()
            print(f"{self.log_prefix} [Latent Regeneration] Models restored")

        return latent

    def _setup_text_encoder_caches(self, datasets: List[Any]) -> Dict[str, Path]:
        """
        Setup per-dataset text encoder cache directories for all architectures.
        Similar to _setup_latent_caches(), this only creates directories.

        Args:
            datasets: List of dataset objects

        Returns:
            Dictionary mapping dataset_unique_id to cache directory path
        """
        from pathlib import Path
        from core.training.latent_cache import get_cache_base_dir

        base_dir = Path(get_cache_base_dir())
        text_encoder_caches = {}

        arch_name = "Z-Image" if self.is_zimage else ("SDXL" if self.is_sdxl else "SD1.5")
        print(f"{self.log_prefix} Setting up text encoder cache directories ({arch_name})...")
        print(f"{self.log_prefix} Using global cache directory: {base_dir}")

        for dataset in datasets:
            cache_dir = base_dir / dataset.unique_id / "text_embeddings"
            cache_dir.mkdir(parents=True, exist_ok=True)
            text_encoder_caches[dataset.unique_id] = cache_dir
            print(f"{self.log_prefix} Setup text encoder cache for dataset '{dataset.unique_id}': {cache_dir}")

        return text_encoder_caches

    def _validate_and_generate_text_encoder_caches(
        self,
        datasets: List[Any],
        text_encoder_caches: Dict[str, Path],
        progress_callback: Optional[Callable] = None,
        epoch_num: Optional[int] = None,
    ):
        """
        Check text encoder caches and encode missing captions.
        Similar to _validate_and_generate_latent_caches(), this generates missing embeddings.

        Args:
            datasets: List of dataset objects
            text_encoder_caches: Dictionary mapping dataset_unique_id to cache directory
            progress_callback: Progress callback function
            epoch_num: Current epoch number (for logging)
        """
        import hashlib

        arch_name = "Z-Image" if self.is_zimage else ("SDXL" if self.is_sdxl else "SD1.5")
        epoch_info = f" (Epoch {epoch_num + 1})" if epoch_num is not None else ""
        print(f"{self.log_prefix} Validating and generating text encoder caches ({arch_name}){epoch_info}...")

        # Collect captions per dataset
        dataset_captions = {}
        total_captions = 0

        for dataset in datasets:
            unique_captions = set()
            caption_samples = []
            for item in dataset.items:
                caption = item.get("caption", "")
                if caption:
                    unique_captions.add(caption)
                    if len(caption_samples) < 3:
                        caption_samples.append(caption)
            dataset_captions[dataset.unique_id] = unique_captions
            total_captions += len(unique_captions)
            print(f"{self.log_prefix} Dataset '{dataset.unique_id}': {len(unique_captions)} unique captions")
            if caption_samples and epoch_num is not None:
                print(f"{self.log_prefix}   Sample captions (epoch {epoch_num + 1}):")
                for i, sample in enumerate(caption_samples[:3], 1):
                    print(f"{self.log_prefix}     [{i}] {sample[:80]}...")

        print(f"{self.log_prefix} Total unique captions across all datasets: {total_captions}")

        # Encode missing captions for each dataset
        total_encoded = 0
        total_cached = 0

        # Move text encoder(s) to GPU for encoding
        self.move_text_encoder_to_gpu()

        try:
            for dataset in datasets:
                cache_dir = text_encoder_caches[dataset.unique_id]
                captions = dataset_captions[dataset.unique_id]

                # Check which captions are missing
                captions_to_encode = []
                for caption in captions:
                    caption_hash = hashlib.md5(caption.encode()).hexdigest()
                    embeds_path = cache_dir / f"{caption_hash}_embeds.pt"

                    # Check auxiliary data file (architecture-specific)
                    if self.is_zimage:
                        auxiliary_path = cache_dir / f"{caption_hash}_mask.pt"
                    elif self.is_sdxl:
                        auxiliary_path = cache_dir / f"{caption_hash}_pooled.pt"
                    else:
                        auxiliary_path = None  # SD1.5 has no auxiliary data

                    # Check if all required files exist
                    if auxiliary_path is not None:
                        if not (embeds_path.exists() and auxiliary_path.exists()):
                            captions_to_encode.append(caption)
                        else:
                            total_cached += 1
                    else:
                        if not embeds_path.exists():
                            captions_to_encode.append(caption)
                        else:
                            total_cached += 1

                if len(captions_to_encode) == 0:
                    print(f"{self.log_prefix} Dataset '{dataset.unique_id}': All {len(captions)} captions already cached")
                else:
                    print(f"{self.log_prefix} Dataset '{dataset.unique_id}': Encoding {len(captions_to_encode)}/{len(captions)} captions...")

                    for idx, caption in enumerate(tqdm(captions_to_encode, desc=f"Encoding captions [{dataset.unique_id}]")):
                        # Encode caption (unified method)
                        embeddings, auxiliary_data = self.encode_caption(caption, requires_grad=False)
                        embeds_cpu = embeddings.cpu()
                        auxiliary_cpu = auxiliary_data.cpu() if auxiliary_data is not None else None

                        # Save immediately to disk to avoid memory accumulation
                        caption_hash = hashlib.md5(caption.encode()).hexdigest()
                        embeds_path = cache_dir / f"{caption_hash}_embeds.pt"

                        try:
                            # Save main embeddings
                            torch.save(embeds_cpu, embeds_path)

                            # Save auxiliary data (architecture-specific)
                            if self.is_zimage and auxiliary_cpu is not None:
                                mask_path = cache_dir / f"{caption_hash}_mask.pt"
                                torch.save(auxiliary_cpu, mask_path)
                            elif self.is_sdxl and auxiliary_cpu is not None:
                                pooled_path = cache_dir / f"{caption_hash}_pooled.pt"
                                torch.save(auxiliary_cpu, pooled_path)
                            # SD1.5: no auxiliary data to save

                            total_encoded += 1

                            # Free memory immediately after saving
                            del embeds_cpu, embeddings
                            if auxiliary_cpu is not None:
                                del auxiliary_cpu, auxiliary_data
                        except Exception as e:
                            print(f"{self.log_prefix} WARNING: Failed to save cache for caption '{caption[:30]}...': {e}")

                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                phase="text_encoder_cache",
                                step=total_cached + total_encoded,
                                total=total_captions,
                            )

        finally:
            # Move text encoder(s) back to CPU
            self.move_text_encoder_to_cpu()

        print(f"{self.log_prefix} Text encoder cache validation complete:")
        print(f"{self.log_prefix}   - Cached: {total_cached}")
        print(f"{self.log_prefix}   - Newly encoded: {total_encoded}")
        print(f"{self.log_prefix}   - Total: {total_cached + total_encoded}")

    def _load_caption_embedding_from_disk(
        self,
        caption: str,
        dataset_unique_id: str,
        text_encoder_caches: Dict[str, Path]
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Load caption embedding from disk cache for all architectures.

        Args:
            caption: Caption text
            dataset_unique_id: Dataset unique ID
            text_encoder_caches: Dictionary mapping dataset_unique_id to cache directory

        Returns:
            Tuple of (embeddings, auxiliary_data) if cached, None otherwise:
            - Z-Image: (prompt_embeds, attention_mask)
            - SD1.5: (text_embeddings, None)
            - SDXL: (text_embeddings, pooled_embeddings)
        """
        import hashlib

        cache_dir = text_encoder_caches.get(dataset_unique_id)
        if cache_dir is None:
            return None

        caption_hash = hashlib.md5(caption.encode()).hexdigest()
        embeds_path = cache_dir / f"{caption_hash}_embeds.pt"

        # Check architecture-specific auxiliary file
        if self.is_zimage:
            auxiliary_path = cache_dir / f"{caption_hash}_mask.pt"
        elif self.is_sdxl:
            auxiliary_path = cache_dir / f"{caption_hash}_pooled.pt"
        else:
            auxiliary_path = None  # SD1.5 has no auxiliary data

        # Check if required files exist
        if auxiliary_path is not None:
            if not (embeds_path.exists() and auxiliary_path.exists()):
                return None
        else:
            if not embeds_path.exists():
                return None

        # Load embeddings
        try:
            embeddings = torch.load(embeds_path, map_location='cpu')
            if auxiliary_path is not None:
                auxiliary_data = torch.load(auxiliary_path, map_location='cpu')
            else:
                auxiliary_data = None
            return (embeddings, auxiliary_data)
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to load cached embedding for caption '{caption[:30]}...': {e}")
            return None

    # ============================================================
    # Training Loop Infrastructure
    # ============================================================

    def train(
        self,
        datasets: List[Any],
        num_epochs: int = 10,
        total_steps: Optional[int] = None,  # If specified, overrides num_epochs
        batch_size: int = 1,
        save_every_n_steps: int = 500,
        sample_every_n_steps: int = 500,
        sample_prompt: str = "a beautiful landscape",
        sample_guidance_scale: float = 3.5,
        sample_steps: int = 28,
        sample_width: int = 1024,
        sample_height: int = 1024,
        sample_seed: int = -1,
        sample_schedule_type: str = "uniform",
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "constant",
        enable_bucketing: bool = True,
        base_resolutions: Optional[List[int]] = None,
        bucket_strategy: str = "resize",
        multi_resolution_mode: str = "max",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        multi_noise_timesteps: int = 1,
        multi_noise_mode: str = "independent",  # Unused (MNT disabled), kept for compatibility
        trajectory_blend_alpha: float = 0.7,  # Unused (MNT disabled), kept for compatibility
        timestep_sampling_config: Optional[Dict[str, Any]] = None,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
        progress_callback: Optional[Callable] = None,
        run_id: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        force_recache: bool = False,
        max_step_saves_to_keep: int = 3,
        text_encoding_mode: str = "swap_onthefly",
        text_encoding_swap_interval: int = 256,
        latent_encoding_mode: str = "swap_onthefly",
        latent_encoding_swap_interval: int = 256,
    ):
        """
        Main training loop.

        Args:
            datasets: List of dataset objects
            num_epochs: Number of training epochs
            batch_size: Batch size per step
            save_every_n_steps: Save checkpoint every N steps
            sample_every_n_steps: Generate sample every N steps
            sample_prompt: Prompt for sample generation
            optimizer_type: Optimizer type
            lr_scheduler_type: LR scheduler type
            enable_bucketing: Enable resolution bucketing
            base_resolutions: List of base resolutions (e.g., [512, 768, 1024])
            bucket_strategy: Bucketing strategy ("resize", "crop", "random_crop")
            multi_resolution_mode: Multi-resolution mode ("max", "random")
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            debug_latents: Enable debug latent saving
            debug_latents_every: Save debug latents every N steps
            progress_callback: Progress callback function
            text_encoding_mode: Text encoding mode for Z-Image
                - "swap_onthefly": Swap TE/Transformer, encode on-the-fly (recommended for large datasets)
                - "pre_encoded_cache": Use pre-encoded disk cache (NOT recommended for large datasets)
                - "onthefly_gpu": Encode on-the-fly on GPU without cache (NOT recommended for Z-Image)
            text_encoding_swap_interval: Swap interval for swap_onthefly mode (default: 256 steps)
        """
        print(f"{self.log_prefix} Starting training...")
        print(f"{self.log_prefix} Datasets: {len(datasets)}")
        print(f"{self.log_prefix} Epochs: {num_epochs}")
        print(f"{self.log_prefix} Batch size: {batch_size}")
        print(f"{self.log_prefix} Gradient accumulation: {gradient_accumulation_steps}")
        print(f"{self.log_prefix} Debug latents: {debug_latents} (every {debug_latents_every} steps)")

        # Validate text_encoding_mode when Text Encoder is trainable
        # Check if any Text Encoder has trainable parameters (works for both LoRA and full fine-tune)
        text_encoder_trainable = False
        te1_trainable_params = 0
        te2_trainable_params = 0

        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            te1_trainable_params = sum(1 for p in self.text_encoder.parameters() if p.requires_grad)
            text_encoder_trainable = te1_trainable_params > 0

        if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
            te2_trainable_params = sum(1 for p in self.text_encoder_2.parameters() if p.requires_grad)
            text_encoder_trainable = text_encoder_trainable or (te2_trainable_params > 0)

        # Log trainable parameter counts
        if text_encoder_trainable:
            print(f"{self.log_prefix} Text Encoder trainable parameters detected:")
            if te1_trainable_params > 0:
                print(f"{self.log_prefix}   Text Encoder 1: {te1_trainable_params} trainable params")
            if te2_trainable_params > 0:
                print(f"{self.log_prefix}   Text Encoder 2: {te2_trainable_params} trainable params")

        # If Text Encoder is trainable, embeddings must be recomputed each step
        if text_encoder_trainable and text_encoding_mode in ['swap_onthefly', 'pre_encoded_cache']:
            print(f"{self.log_prefix} WARNING: Text Encoder is trainable but text_encoding_mode='{text_encoding_mode}'")
            print(f"{self.log_prefix} Text embeddings would be cached and NOT updated during training!")
            print(f"{self.log_prefix} Overriding to 'onthefly_gpu' - embeddings must be recomputed each step")
            text_encoding_mode = 'onthefly_gpu'

        # Log final text encoding mode
        print(f"{self.log_prefix} Text encoding mode: {text_encoding_mode}")

        # Setup debug directory
        debug_dir = None
        if debug_latents:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            print(f"{self.log_prefix} Debug latents will be saved to: {debug_dir}")

        # Setup bucketing
        if enable_bucketing:
            from core.training.bucketing import BucketManager

            # Default to [1024] if not specified
            if base_resolutions is None:
                base_resolutions = [1024]

            bucket_manager = BucketManager(
                base_resolutions=base_resolutions,
                divisibility=8,
                strategy=bucket_strategy,
                multi_resolution_mode=multi_resolution_mode
            )
            print(f"{self.log_prefix} Bucketing enabled: base_resolutions={base_resolutions}, strategy={bucket_strategy}, mode={multi_resolution_mode}")
        else:
            bucket_manager = None
            print(f"{self.log_prefix} Bucketing disabled")

        # Validate MNT parameters
        if multi_noise_timesteps < 1:
            raise ValueError(f"multi_noise_timesteps must be >= 1, got {multi_noise_timesteps}")

        # Setup timestep sampler
        from .timestep_sampler import TimestepSampler

        if timestep_sampling_config is None:
            # Default: uniform [0, 1]
            timestep_sampling_config = {
                "distribution": "uniform",
                "min_timestep": 0.0,
                "max_timestep": 1.0,
            }

        timestep_sampler = TimestepSampler.from_config(timestep_sampling_config)
        print(f"{self.log_prefix} Timestep sampler: {timestep_sampler.__class__.__name__}")
        print(f"{self.log_prefix} Timestep range: [{timestep_sampler.min_timestep:.3f}, {timestep_sampler.max_timestep:.3f}]")
        print(f"{self.log_prefix} Multi Noise-Timesteps (MNT): {multi_noise_timesteps}")

        if multi_noise_timesteps > 1:
            print(f"{self.log_prefix} MNT enabled: Each batch will be processed {multi_noise_timesteps} times with different timesteps")

        # Calculate effective gradient accumulation (MNT acts as additional accumulation)
        effective_gradient_accumulation = gradient_accumulation_steps * multi_noise_timesteps
        print(f"{self.log_prefix} Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"{self.log_prefix} Effective gradient accumulation (with MNT): {effective_gradient_accumulation}")

        # Calculate total steps and epochs
        total_items = sum(len(dataset.items) for dataset in datasets)
        batches_per_epoch = (total_items + batch_size - 1) // batch_size
        steps_per_epoch = batches_per_epoch * multi_noise_timesteps  # MNT multiplier

        # If total_steps is specified, calculate num_epochs; otherwise use num_epochs
        if total_steps is not None:
            # Step-based training: calculate epochs needed
            num_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
            actual_total_steps = total_steps
            print(f"{self.log_prefix} Training mode: Step-based ({total_steps} steps)")
            print(f"{self.log_prefix} Calculated epochs needed: {num_epochs}")
        else:
            # Epoch-based training: calculate total steps
            actual_total_steps = steps_per_epoch * num_epochs
            print(f"{self.log_prefix} Training mode: Epoch-based ({num_epochs} epochs)")

        print(f"{self.log_prefix} Total items: {total_items}")
        print(f"{self.log_prefix} Batches per epoch: {batches_per_epoch}")
        print(f"{self.log_prefix} Steps per epoch (with MNT): {steps_per_epoch}")
        print(f"{self.log_prefix} Total training steps: {actual_total_steps}")

        # Setup optimizer
        self.setup_optimizer(
            optimizer_type=optimizer_type,
            lr_scheduler_type=lr_scheduler_type,
            total_steps=actual_total_steps,
        )

        # Apply bucketing to datasets
        if bucket_manager:
            print(f"{self.log_prefix} Assigning images to buckets...")
            for dataset in datasets:
                for item in dataset.items:
                    width = item.get("width", 1024)
                    height = item.get("height", 1024)
                    bucket, image_info = bucket_manager.assign_image_to_bucket(
                        image_path=item["image_path"],
                        width=width,
                        height=height,
                        caption=item.get("caption", ""),
                        dataset_unique_id=dataset.unique_id
                    )
                    # Update item with bucket dimensions
                    item["width"] = image_info["bucket_width"]
                    item["height"] = image_info["bucket_height"]

            # Print bucket statistics
            bucket_counts = bucket_manager.get_bucket_counts()
            print(f"{self.log_prefix} Bucket distribution:")
            for bucket_size, count in sorted(bucket_counts.items()):
                print(f"  {bucket_size}: {count} images")

        # Setup latent caches (mode-dependent)
        latent_caches = None
        print(f"{self.log_prefix} Latent encoding mode: {latent_encoding_mode}")
        if latent_encoding_mode == "swap_onthefly":
            print(f"{self.log_prefix} Latent swap interval: {latent_encoding_swap_interval} steps")
            print(f"{self.log_prefix} VAE will swap with main model during training")
            # No cache setup needed for swap mode
        elif latent_encoding_mode == "pre_encoded_cache":
            print(f"{self.log_prefix} Using pre-encoded latent disk cache mode")
            latent_caches = self._setup_latent_caches(datasets)
            self._validate_and_generate_latent_caches(datasets, latent_caches, progress_callback, force_recache=force_recache)
        elif latent_encoding_mode == "onthefly_gpu":
            print(f"{self.log_prefix} Using on-the-fly GPU latent encoding (no cache)")
            # No cache setup needed
        else:
            raise ValueError(f"Invalid latent_encoding_mode: {latent_encoding_mode}")

        # Setup text encoder caches (all architectures)
        text_encoder_caches = None
        print(f"{self.log_prefix} Text encoding mode: {text_encoding_mode}")
        if text_encoding_mode == "swap_onthefly":
            print(f"{self.log_prefix} Swap interval: {text_encoding_swap_interval} steps")
            if self.is_zimage:
                print(f"{self.log_prefix} Text encoder will swap with transformer during training")
            else:
                print(f"{self.log_prefix} Text encoder will swap with U-Net during training")
            # No cache setup needed for swap mode
        elif text_encoding_mode == "pre_encoded_cache":
            print(f"{self.log_prefix} Using pre-encoded disk cache mode")
            text_encoder_caches = self._setup_text_encoder_caches(datasets)
        elif text_encoding_mode == "onthefly_gpu":
            print(f"{self.log_prefix} Using on-the-fly GPU encoding (no cache)")
            # No cache setup needed
        else:
            raise ValueError(f"Invalid text_encoding_mode: {text_encoding_mode}")

        # Clean up stop flag from previous run (if any)
        stop_flag_file = self.output_dir / ".stop_training"
        if stop_flag_file.exists():
            print(f"{self.log_prefix} Removing stale stop flag from previous run")
            stop_flag_file.unlink()

        # Training loop
        global_step = 0
        start_epoch = 0
        resume_batch_idx = 0  # Batch index to resume from within epoch
        resume_training_state = None  # Training state for mid-epoch resume

        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            if resume_from_checkpoint.lower() == "latest":
                # Auto-detect latest checkpoint
                checkpoint_result = self.find_latest_checkpoint()
                if checkpoint_result is not None:
                    checkpoint_path, checkpoint_step = checkpoint_result
                    print(f"{self.log_prefix} Resuming from latest checkpoint: {checkpoint_path}")
                    loaded_step = self.load_checkpoint(checkpoint_path)
                    global_step = loaded_step

                    # Try to load training state for mid-epoch resume
                    resume_training_state = self.load_training_state(loaded_step)
                    if resume_training_state:
                        start_epoch = resume_training_state['epoch']
                        resume_batch_idx = resume_training_state['batch_idx']

                        # Use global_step from state.json (most accurate, saved at same time as batch_idx)
                        if 'global_step' in resume_training_state:
                            global_step = resume_training_state['global_step']
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}, global_step={global_step}")
                        else:
                            # Fallback: use global_step from checkpoint filename
                            print(f"{self.log_prefix} WARNING: No global_step in training state, using checkpoint filename: {global_step}")
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}")

                        print(f"{self.log_prefix} Mid-epoch resume: epoch {start_epoch + 1}, batch {resume_batch_idx}, step {global_step}")
                    else:
                        # No training state file, fall back to epoch-level resume
                        start_epoch = global_step // steps_per_epoch
                        print(f"{self.log_prefix} Resuming from step {global_step}, epoch {start_epoch + 1}")

                    # Fast-forward lr_scheduler to match the checkpoint
                    for _ in range(global_step):
                        self.lr_scheduler.step()

                    # IMPORTANT: Update optimizer learning rate from YAML config
                    # (Necessary when user modifies LR in YAML before resume)
                    if hasattr(self, 'optimizer') and self.optimizer is not None:
                        for param_group in self.optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] = self.learning_rate
                            if old_lr != self.learning_rate:
                                print(f"{self.log_prefix} Updated optimizer LR: {old_lr:.2e} -> {self.learning_rate:.2e}")

                    # IMPORTANT: Also update LR Scheduler's base_lrs to prevent it from resetting LR
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        if hasattr(self.lr_scheduler, 'base_lrs'):
                            for i in range(len(self.lr_scheduler.base_lrs)):
                                old_base_lr = self.lr_scheduler.base_lrs[i]
                                self.lr_scheduler.base_lrs[i] = self.learning_rate
                                if old_base_lr != self.learning_rate:
                                    print(f"{self.log_prefix} Updated LR Scheduler base_lrs[{i}]: {old_base_lr:.2e} -> {self.learning_rate:.2e}")
                else:
                    print(f"{self.log_prefix} No checkpoint found for auto-resume, starting from scratch")
            else:
                # User specified a specific checkpoint file
                checkpoint_path = self.output_dir / resume_from_checkpoint
                if checkpoint_path.exists():
                    print(f"{self.log_prefix} Resuming from specified checkpoint: {checkpoint_path}")
                    loaded_step = self.load_checkpoint(str(checkpoint_path))
                    global_step = loaded_step

                    # Try to load training state for mid-epoch resume
                    resume_training_state = self.load_training_state(loaded_step)
                    if resume_training_state:
                        start_epoch = resume_training_state['epoch']
                        resume_batch_idx = resume_training_state['batch_idx']

                        # Use global_step from state.json (most accurate, saved at same time as batch_idx)
                        if 'global_step' in resume_training_state:
                            global_step = resume_training_state['global_step']
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}, global_step={global_step}")
                        else:
                            # Fallback: use global_step from checkpoint filename
                            print(f"{self.log_prefix} WARNING: No global_step in training state, using checkpoint filename: {global_step}")
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}")

                        print(f"{self.log_prefix} Mid-epoch resume: epoch {start_epoch + 1}, batch {resume_batch_idx}, step {global_step}")
                    else:
                        # No training state file, fall back to epoch-level resume
                        start_epoch = global_step // steps_per_epoch
                        print(f"{self.log_prefix} Resuming from step {global_step}, epoch {start_epoch + 1}")

                    # Fast-forward lr_scheduler to match the checkpoint
                    for _ in range(global_step):
                        self.lr_scheduler.step()

                    # IMPORTANT: Update optimizer learning rate from YAML config
                    # (Necessary when user modifies LR in YAML before resume)
                    if hasattr(self, 'optimizer') and self.optimizer is not None:
                        for param_group in self.optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] = self.learning_rate
                            if old_lr != self.learning_rate:
                                print(f"{self.log_prefix} Updated optimizer LR: {old_lr:.2e} -> {self.learning_rate:.2e}")

                    # IMPORTANT: Also update LR Scheduler's base_lrs to prevent it from resetting LR
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        if hasattr(self.lr_scheduler, 'base_lrs'):
                            for i in range(len(self.lr_scheduler.base_lrs)):
                                old_base_lr = self.lr_scheduler.base_lrs[i]
                                self.lr_scheduler.base_lrs[i] = self.learning_rate
                                if old_base_lr != self.learning_rate:
                                    print(f"{self.log_prefix} Updated LR Scheduler base_lrs[{i}]: {old_base_lr:.2e} -> {self.learning_rate:.2e}")
                else:
                    print(f"{self.log_prefix} WARNING: Checkpoint not found: {checkpoint_path}")
                    print(f"{self.log_prefix} Starting from scratch")

        # Generate step 0 sample to verify base model output
        if sample_every_n_steps > 0 and global_step == 0:
            print(f"{self.log_prefix} [Step 0] Generating sample to verify base model...")
            print(f"{self.log_prefix} [Step 0] Sample params: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, steps={sample_steps}, seed={sample_seed}")
            if self.is_zimage:
                sample = self._generate_sample_zimage(
                    prompt=sample_prompt,
                    width=sample_width,
                    height=sample_height,
                    num_inference_steps=sample_steps,
                    guidance_scale=sample_guidance_scale,
                    seed=sample_seed
                )
            else:
                sample = self.generate_sample(
                    prompt=sample_prompt,
                    width=sample_width,
                    height=sample_height,
                    num_inference_steps=sample_steps,
                    guidance_scale=sample_guidance_scale,
                    seed=sample_seed,
                    current_step=0,
                    schedule_type=sample_schedule_type
                )

            # Save step 0 sample
            sample_path = self.output_dir / "samples" / f"step_{0:06d}_sample_0.png"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample.save(sample_path)
            print(f"{self.log_prefix} [Step 0] Saved sample to {sample_path.relative_to(self.output_dir)}")

        try:
            for epoch in range(start_epoch, num_epochs):
                print(f"\n{self.log_prefix} Epoch {epoch + 1}/{num_epochs}")

                # Reload datasets for per-epoch shuffle/dropout
                # (This regenerates captions with different shuffle/dropout based on epoch_num)
                for dataset in datasets:
                    if hasattr(dataset, 'reload_for_epoch'):
                        dataset.items = dataset.reload_for_epoch(epoch_num=epoch, run_id=run_id)
                        print(f"{self.log_prefix} Reloaded dataset {dataset.unique_id} for epoch {epoch + 1} ({len(dataset.items)} items)")

                # Validate and generate text encoder cache for new captions (all architectures)
                # Only for pre_encoded_cache mode
                if text_encoding_mode == "pre_encoded_cache":
                    self._validate_and_generate_text_encoder_caches(datasets, text_encoder_caches, progress_callback, epoch_num=epoch)

                # Create all_items list (needed for swap buffer and batching)
                all_items = []
                for dataset in datasets:
                    all_items.extend([(item, dataset) for item in dataset.items])

                # Mid-epoch resume: restore random state BEFORE building batches
                # This ensures batches are shuffled in the same order as the interrupted run
                if epoch == start_epoch and resume_training_state is not None:
                    import random
                    print(f"{self.log_prefix} Restoring random state for mid-epoch resume...")
                    random.setstate(resume_training_state['random_state'])

                # Create batches
                if bucket_manager:
                    # BucketManager only manages items, we need to pair with datasets
                    # Build mapping from image_path to dataset
                    path_to_dataset = {}
                    for dataset in datasets:
                        for item in dataset.items:
                            path_to_dataset[item["image_path"]] = dataset

                    # Get batches from bucket manager
                    item_batches = bucket_manager.build_batch_indices(batch_size)

                    # Convert to (item, dataset) tuples
                    batches = []
                    for item_batch in item_batches:
                        batch_with_dataset = [
                            (item, path_to_dataset[item["image_path"]])
                            for item in item_batch
                        ]
                        batches.append(batch_with_dataset)
                else:
                    # Simple sequential batching
                    batches = [all_items[i:i+batch_size] for i in range(0, len(all_items), batch_size)]

                # Mid-epoch resume: skip completed batches
                # (random state was already restored before batch building)
                if epoch == start_epoch and resume_training_state is not None:
                    print(f"{self.log_prefix} Skipping {resume_batch_idx} completed batches...")
                    batches = batches[resume_batch_idx:]

                    # Clear resume state so we don't skip batches in subsequent epochs
                    resume_training_state = None

                # Initialize swap mode buffer if needed (all architectures)
                swap_buffer = [] if text_encoding_mode == "swap_onthefly" else None
                swap_buffer_idx = 0
                next_swap_at_step = 0 if swap_buffer is not None else -1

                # Pre-fill swap buffer for first interval
                if swap_buffer is not None:
                    print(f"{self.log_prefix} Pre-filling swap buffer for first {text_encoding_swap_interval} steps...")
                    if progress_callback:
                        progress_callback(
                            phase="text_encoder_cache",
                            step=0,
                            total=text_encoding_swap_interval
                        )

                    # Move Text Encoder to GPU for encoding
                    self.move_text_encoder_to_gpu()
                    # Move main model to CPU to free VRAM
                    self.move_main_model_to_cpu()

                    # Encode captions for first interval
                    # Use batches (which have bucket info) instead of all_items
                    buffer_items = []
                    for batch in batches[:text_encoding_swap_interval]:
                        buffer_items.extend(batch)
                    for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding captions")):
                        caption = item.get("caption", "")
                        image_path = item["image_path"]
                        embeddings, auxiliary_data = self.encode_caption(caption, requires_grad=False)
                        # Store on CPU to save GPU VRAM
                        # auxiliary_data: attention_mask (Z-Image), pooled_embeddings (SDXL), None (SD1.5)
                        # IMPORTANT: Also store caption and image_path to ensure correct pairing during training
                        swap_buffer.append((
                            embeddings.cpu(),
                            auxiliary_data.cpu() if auxiliary_data is not None else None,
                            caption,  # String (CPU memory, minimal overhead)
                            image_path  # String (CPU memory, minimal overhead)
                        ))

                        # Send progress update
                        if progress_callback and idx % 10 == 0:
                            progress_callback(
                                phase="text_encoder_cache",
                                step=idx,
                                total=len(buffer_items)
                            )

                    # Move Text Encoder back to CPU
                    self.move_text_encoder_to_cpu()
                    # Move main model to GPU for training
                    self.move_main_model_to_gpu()

                    next_swap_at_step = text_encoding_swap_interval
                    print(f"{self.log_prefix} Buffer pre-filled with {len(swap_buffer)} embeddings")

                # Initialize latent swap mode buffer if needed
                latent_swap_buffer = [] if latent_encoding_mode == "swap_onthefly" else None
                latent_swap_buffer_idx = 0
                next_latent_swap_at_step = 0 if latent_swap_buffer is not None else -1

                # Pre-fill latent swap buffer for first interval
                if latent_swap_buffer is not None:
                    print(f"{self.log_prefix} Pre-filling latent swap buffer for first {latent_encoding_swap_interval} steps...")
                    if progress_callback:
                        progress_callback(
                            phase="latent_cache",
                            step=0,
                            total=latent_encoding_swap_interval
                        )

                    # Move VAE to GPU for encoding
                    self.move_vae_to_gpu()
                    # Move main model to CPU to free VRAM
                    self.move_main_model_to_cpu()

                    # Encode images for first interval
                    # Use batches (which have bucket info) instead of all_items
                    buffer_items = []
                    for batch in batches[:latent_encoding_swap_interval]:
                        buffer_items.extend(batch)
                    for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding latents")):
                        image_path = item["image_path"]
                        caption = item.get("caption", "")
                        width = item.get("width") or item.get("bucket_width")
                        height = item.get("height") or item.get("bucket_height")

                        # Load and encode image
                        image = Image.open(image_path)
                        latent = self.encode_image(
                            image=image,
                            target_width=width,
                            target_height=height,
                            bucket_strategy=bucket_strategy
                        )
                        # Store on CPU to save GPU VRAM
                        # IMPORTANT: Also store caption and image_path to ensure correct pairing during training
                        latent_swap_buffer.append((
                            latent.cpu(),
                            caption,  # String (CPU memory, minimal overhead)
                            image_path  # String (CPU memory, minimal overhead)
                        ))

                        # Send progress update
                        if progress_callback and idx % 10 == 0:
                            progress_callback(
                                phase="latent_cache",
                                step=idx,
                                total=len(buffer_items)
                            )

                    # Move VAE back to CPU
                    self.move_vae_to_cpu()
                    # Move main model to GPU for training
                    self.move_main_model_to_gpu()

                    next_latent_swap_at_step = latent_encoding_swap_interval
                    print(f"{self.log_prefix} Latent buffer pre-filled with {len(latent_swap_buffer)} latents")

                # Training loop
                for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}")):
                    # Reset fused optimizer groups counters (start of each step)
                    if self.fused_optimizer_groups is not None:
                        self.fused_optimizer_groups.reset_counters()

                    # Check for stop flag (user-requested stop from frontend)
                    stop_flag_file = self.output_dir / ".stop_training"
                    if stop_flag_file.exists():
                        print(f"\n{self.log_prefix} Stop flag detected, stopping training...")
                        stop_flag_file.unlink()  # Clean up flag file
                        raise KeyboardInterrupt("Training stopped by user")

                    # Check if we need to refill swap buffer
                    if swap_buffer is not None and batch_idx >= next_swap_at_step:
                        # Calculate next batch range
                        start_idx = next_swap_at_step
                        end_idx = min(start_idx + text_encoding_swap_interval, len(batches))
                        # Use batches (which have bucket info) instead of all_items
                        buffer_items = []
                        for batch in batches[start_idx:end_idx]:
                            buffer_items.extend(batch)

                        print(f"\n{self.log_prefix} Refilling swap buffer (steps {start_idx}-{end_idx})...")
                        if progress_callback:
                            progress_callback(
                                phase="text_encoder_cache",
                                step=0,
                                total=len(buffer_items)
                            )

                        # Move Text Encoder to GPU
                        self.move_text_encoder_to_gpu()
                        # Move main model to CPU
                        self.move_main_model_to_cpu()

                        # Clear old buffer and encode new captions
                        swap_buffer.clear()
                        swap_buffer_idx = 0
                        for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding captions", leave=False)):
                            caption = item.get("caption", "")
                            image_path = item["image_path"]
                            embeddings, auxiliary_data = self.encode_caption(caption, requires_grad=False)
                            # Store on CPU to save GPU VRAM
                            # IMPORTANT: Also store caption and image_path to ensure correct pairing during training
                            swap_buffer.append((
                                embeddings.cpu(),
                                auxiliary_data.cpu() if auxiliary_data is not None else None,
                                caption,  # String (CPU memory, minimal overhead)
                                image_path  # String (CPU memory, minimal overhead)
                            ))

                            # Send progress update
                            if progress_callback and idx % 10 == 0:
                                progress_callback(
                                    phase="text_encoder_cache",
                                    step=idx,
                                    total=len(buffer_items)
                                )

                        # Move Text Encoder back to CPU
                        self.move_text_encoder_to_cpu()
                        # Move main model to GPU
                        self.move_main_model_to_gpu()

                        # Clear CUDA cache after model movement to free fragmented memory
                        torch.cuda.empty_cache()

                        next_swap_at_step += text_encoding_swap_interval
                        print(f"{self.log_prefix} Buffer refilled with {len(swap_buffer)} embeddings")

                    # Check if we need to refill latent swap buffer
                    if latent_swap_buffer is not None and batch_idx >= next_latent_swap_at_step:
                        # Calculate next batch range
                        start_idx = next_latent_swap_at_step
                        end_idx = min(start_idx + latent_encoding_swap_interval, len(batches))
                        # Use batches (which have bucket info) instead of all_items
                        buffer_items = []
                        for batch in batches[start_idx:end_idx]:
                            buffer_items.extend(batch)

                        print(f"\n{self.log_prefix} Refilling latent swap buffer (steps {start_idx}-{end_idx})...")
                        if progress_callback:
                            progress_callback(
                                phase="latent_cache",
                                step=0,
                                total=len(buffer_items)
                            )

                        # Move VAE to GPU
                        self.move_vae_to_gpu()
                        # Move main model to CPU
                        self.move_main_model_to_cpu()

                        # Clear old buffer and encode new latents
                        latent_swap_buffer.clear()
                        latent_swap_buffer_idx = 0
                        for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding latents", leave=False)):
                            image_path = item["image_path"]
                            caption = item.get("caption", "")
                            width = item.get("width") or item.get("bucket_width")
                            height = item.get("height") or item.get("bucket_height")

                            # Load and encode image
                            image = Image.open(image_path)
                            latent = self.encode_image(
                                image=image,
                                target_width=width,
                                target_height=height,
                                bucket_strategy=bucket_strategy
                            )
                            # Store on CPU to save GPU VRAM
                            # IMPORTANT: Also store caption and image_path to ensure correct pairing during training
                            latent_swap_buffer.append((
                                latent.cpu(),
                                caption,  # String (CPU memory, minimal overhead)
                                image_path  # String (CPU memory, minimal overhead)
                            ))

                            # Send progress update
                            if progress_callback and idx % 10 == 0:
                                progress_callback(
                                    phase="latent_cache",
                                    step=idx,
                                    total=len(buffer_items)
                                )

                        # Move VAE back to CPU
                        self.move_vae_to_cpu()
                        # Move main model to GPU
                        self.move_main_model_to_gpu()

                        # Clear CUDA cache after model movement to free fragmented memory
                        torch.cuda.empty_cache()

                        next_latent_swap_at_step += latent_encoding_swap_interval
                        print(f"{self.log_prefix} Latent buffer refilled with {len(latent_swap_buffer)} latents")

                    # MNT loop: Process same batch with different noise-timesteps
                    # Save swap buffer indices for this batch (restore after MNT iterations)
                    swap_buffer_idx_batch_start = swap_buffer_idx
                    latent_swap_buffer_idx_batch_start = latent_swap_buffer_idx

                    for mnt_idx in range(multi_noise_timesteps):
                        # Restore swap buffer indices for each MNT iteration (reuse same embeddings/latents)
                        swap_buffer_idx = swap_buffer_idx_batch_start
                        latent_swap_buffer_idx = latent_swap_buffer_idx_batch_start

                        # Prepare batch data
                        latents_list = []
                        text_embeddings_list = []
                        auxiliary_data_list = []  # Unified: attention_mask (Z-Image), pooled_embeddings (SDXL), or None (SD1.5)

                        for item, dataset in batch:
                            # BucketManager stores bucket_width/bucket_height, not width/height
                            width = item.get("width") or item.get("bucket_width")
                            height = item.get("height") or item.get("bucket_height")

                            # Load latent (mode-specific)
                            if latent_encoding_mode == "swap_onthefly":
                                # Get from swap buffer (now 3-tuple: latent, caption, image_path)
                                if latent_swap_buffer_idx < len(latent_swap_buffer):
                                    latent_cpu, buffer_caption, buffer_image_path = latent_swap_buffer[latent_swap_buffer_idx]
                                    # Transfer to GPU
                                    latent = latent_cpu.to(self.device, non_blocking=True)
                                    latents_list.append(latent)
                                    # Store caption/image_path for later (will be used by text encoding or debug)
                                    # Note: caption will be overridden again if text_encoding_mode == "swap_onthefly"
                                    item["caption"] = buffer_caption
                                    item["image_path"] = buffer_image_path
                                    latent_swap_buffer_idx += 1
                                else:
                                    # Fallback to on-the-fly encoding
                                    print(f"{self.log_prefix} WARNING: Latent swap buffer exhausted, encoding on-the-fly")
                                    image = Image.open(item["image_path"])
                                    latent = self.encode_image(
                                        image=image,
                                        target_width=width,
                                        target_height=height,
                                        bucket_strategy=bucket_strategy
                                    )
                                    latents_list.append(latent)

                            elif latent_encoding_mode == "pre_encoded_cache":
                                # Load from disk cache
                                cache = latent_caches[dataset.unique_id]
                                latent = cache.load_latent(item["image_path"], width, height)

                                # On-the-fly regeneration if cache is corrupted or incompatible
                                if latent is None:
                                    print(f"{self.log_prefix} WARNING: Latent cache miss or corrupted for {item['image_path']}, regenerating...")
                                    latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)

                                # Validate latent shape
                                expected_latent_height = height // 8
                                expected_latent_width = width // 8
                                if latent.shape[2] != expected_latent_height or latent.shape[3] != expected_latent_width:
                                    print(f"{self.log_prefix} WARNING: Latent shape mismatch for {item['image_path']}")
                                    print(f"{self.log_prefix}   Expected: [1, {self.vae_latent_channels}, {expected_latent_height}, {expected_latent_width}]")
                                    print(f"{self.log_prefix}   Got: {list(latent.shape)}")
                                    print(f"{self.log_prefix}   Regenerating latent...")
                                    latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)

                                latents_list.append(latent)

                            elif latent_encoding_mode == "onthefly_gpu":
                                # Encode on GPU without cache
                                image = Image.open(item["image_path"])
                                latent = self.encode_image(
                                    image=image,
                                    target_width=width,
                                    target_height=height,
                                    bucket_strategy=bucket_strategy
                                )
                                latents_list.append(latent)

                            # Encode caption (mode-specific, architecture-unified)
                            caption = item.get("caption", "")

                            if text_encoding_mode == "swap_onthefly":
                                # Get from swap buffer (now 4-tuple: embeddings, auxiliary, caption, image_path)
                                if swap_buffer_idx < len(swap_buffer):
                                    embeddings_cpu, auxiliary_cpu, buffer_caption, buffer_image_path = swap_buffer[swap_buffer_idx]
                                    # Transfer to GPU
                                    embeddings = embeddings_cpu.to(self.device, non_blocking=True)
                                    auxiliary = auxiliary_cpu.to(self.device, non_blocking=True) if auxiliary_cpu is not None else None
                                    text_embeddings_list.append(embeddings)
                                    auxiliary_data_list.append(auxiliary)
                                    # Override caption from buffer (correct pairing)
                                    caption = buffer_caption
                                    swap_buffer_idx += 1
                                else:
                                    # Shouldn't happen, but fallback to on-the-fly encoding
                                    print(f"{self.log_prefix} WARNING: Swap buffer exhausted, encoding on-the-fly")
                                    embeddings, auxiliary = self.encode_caption(caption, requires_grad=True)
                                    text_embeddings_list.append(embeddings)
                                    auxiliary_data_list.append(auxiliary)

                            elif text_encoding_mode == "pre_encoded_cache":
                                # Load from disk cache (per-dataset)
                                cached_result = self._load_caption_embedding_from_disk(
                                    caption=caption,
                                    dataset_unique_id=dataset.unique_id,
                                    text_encoder_caches=text_encoder_caches
                                )
                                if cached_result is not None:
                                    embeddings_cpu, auxiliary_cpu = cached_result
                                    embeddings = embeddings_cpu.to(self.device, non_blocking=True)
                                    auxiliary = auxiliary_cpu.to(self.device, non_blocking=True) if auxiliary_cpu is not None else None
                                    text_embeddings_list.append(embeddings)
                                    auxiliary_data_list.append(auxiliary)
                                else:
                                    # Not in cache, encode on-the-fly (shouldn't happen if cache setup worked)
                                    print(f"{self.log_prefix} WARNING: Caption not in cache, encoding on-the-fly: '{caption[:30]}...'")
                                    embeddings, auxiliary = self.encode_caption(caption, requires_grad=True)
                                    text_embeddings_list.append(embeddings)
                                    auxiliary_data_list.append(auxiliary)

                            elif text_encoding_mode == "onthefly_gpu":
                                # Encode on GPU without cache
                                embeddings, auxiliary = self.encode_caption(caption, requires_grad=True)
                                text_embeddings_list.append(embeddings)
                                auxiliary_data_list.append(auxiliary)

                        # Stack batch
                        latents = torch.cat(latents_list, dim=0)

                        # Text embeddings are [1, seq_len, dim], use cat to get [batch_size, seq_len, dim]
                        # IMPORTANT: Pad embeddings to same sequence length if chunking is used
                        if text_embeddings_list:
                            # Check if all embeddings have same sequence length
                            seq_lengths = [emb.shape[1] for emb in text_embeddings_list]
                            max_seq_len = max(seq_lengths)

                            if len(set(seq_lengths)) > 1:
                                # Different sequence lengths - need padding
                                padded_embeddings = []
                                for emb in text_embeddings_list:
                                    if emb.shape[1] < max_seq_len:
                                        # Pad to max_seq_len with zeros
                                        pad_length = max_seq_len - emb.shape[1]
                                        padding = torch.zeros(
                                            (emb.shape[0], pad_length, emb.shape[2]),
                                            dtype=emb.dtype,
                                            device=emb.device
                                        )
                                        emb = torch.cat([emb, padding], dim=1)
                                    padded_embeddings.append(emb)
                                text_embeddings = torch.cat(padded_embeddings, dim=0)
                            else:
                                # All same length - direct concatenation
                                text_embeddings = torch.cat(text_embeddings_list, dim=0)
                        else:
                            text_embeddings = None

                        # Sample timesteps for this MNT iteration
                        batch_size = latents.shape[0]
                        timesteps = timestep_sampler.sample(batch_size, self.device)

                        # Determine if we should save debug latents (only on first MNT iteration)
                        debug_save_path = None
                        if mnt_idx == 0 and debug_dir is not None and global_step % debug_latents_every == 0:
                            debug_save_path = debug_dir / f"step_{global_step:06d}"

                        # Collect batch captions only when needed for debug (prevents DRAM accumulation)
                        # NOTE: item["caption"] has been overridden from swap buffer (if applicable), ensuring correct pairing
                        batch_captions = None
                        if debug_save_path is not None:
                            batch_captions = [item.get("caption", "") for item, dataset in batch]

                        # Training step (architecture-specific calling convention)
                        if self.is_zimage:
                            # auxiliary_data_list contains attention_mask for Z-Image
                            attention_mask = torch.stack([aux for aux in auxiliary_data_list if aux is not None], dim=0)
                            loss, recon_loss = self.train_step_zimage(
                                latents=latents,
                                prompt_embeds=text_embeddings,
                                attention_mask=attention_mask,
                                timesteps=timesteps,  # Pass sampled timesteps
                                debug_save_path=debug_save_path,
                                debug_captions=batch_captions,
                                profile_vram=self.debug_vram,
                            )
                        else:
                            # auxiliary_data_list contains pooled_embeddings for SDXL, None for SD1.5
                            pooled_embeddings = None
                            if self.is_sdxl and any(aux is not None for aux in auxiliary_data_list):
                                # Pooled embeddings are [1, dim], use cat to get [batch_size, dim]
                                pooled_embeddings = torch.cat([aux for aux in auxiliary_data_list if aux is not None], dim=0)
                            loss, recon_loss = self.train_step(
                                latents=latents,
                                text_embeddings=text_embeddings,
                                pooled_embeddings=pooled_embeddings,
                                timesteps=timesteps,  # Pass sampled timesteps
                                debug_save_path=debug_save_path,
                                debug_captions=batch_captions,
                                profile_vram=self.debug_vram,
                            )

                        # Backward pass
                        # loss is already a tensor with computation graph from train_step/train_step_zimage
                        loss.backward()

                        # Verify gradient flow on first step (LoRA training only)
                        if global_step == 1 and hasattr(self, 'print_gradient_flow_summary'):
                            self.print_gradient_flow_summary()

                        # Clear saved activations immediately after backward to prevent VRAM leaks
                        if hasattr(self, 'layer_offload_conductor') and self.layer_offload_conductor is not None:
                            self.layer_offload_conductor.clear_activations()

                        # Free batch tensors immediately after backward to prevent VRAM accumulation
                        del latents, text_embeddings
                        if self.is_zimage:
                            del attention_mask
                        if self.is_sdxl and pooled_embeddings is not None:
                            del pooled_embeddings
                        del latents_list, text_embeddings_list, auxiliary_data_list

                        # Increment global step for each MNT iteration
                        global_step += 1

                    # Gradient accumulation check (after all MNT iterations)
                    # effective_gradient_accumulation = gradient_accumulation_steps * multi_noise_timesteps
                    if global_step % effective_gradient_accumulation == 0:
                        if not self.use_fused_backward and self.fused_optimizer_groups is None:
                            # Normal flow: optimizer.step() and zero_grad() here
                            # Gradient clipping
                            if max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_grad_norm)

                            # Optimizer step
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        # else: Fused backward/groups flow - step() and zero_grad() already called by hooks

                        # LR scheduler step
                        if self.fused_optimizer_groups is not None:
                            # Step all schedulers for optimizer groups
                            for lr_scheduler in self.lr_schedulers:
                                lr_scheduler.step()
                        else:
                            # Single scheduler
                            self.lr_scheduler.step()

                        # Logging (convert loss tensor to float for logging)
                        loss_value = loss.item()
                        recon_loss_value = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
                        current_lr = self.lr_scheduler.get_last_lr()[0]

                        # TensorBoard logging (for external tools, backward compatibility)
                        self.writer.add_scalar("train/loss", loss_value, global_step)
                        self.writer.add_scalar("train/recon_loss", recon_loss_value, global_step)
                        self.writer.add_scalar("train/lr", current_lr, global_step)

                        # Database logging (for fast frontend queries, UPSERT on duplicate step)
                        if self.run_id is not None:
                            self._log_metrics_to_db(
                                step=global_step,
                                loss=loss_value,
                                recon_loss=recon_loss_value,
                                learning_rate=current_lr
                            )

                        # Flush TensorBoard writer periodically to prevent DRAM accumulation
                        # (TensorBoard buffers events internally, can accumulate GBs over long training)
                        if global_step % 100 == 0:
                            self.writer.flush()
                            # Also clear CUDA cache to prevent fragmented memory accumulation
                            torch.cuda.empty_cache()

                        # Free loss tensor after logging
                        del loss, recon_loss

                        # Save checkpoint
                        if global_step % save_every_n_steps == 0:
                            self.save_checkpoint(step=global_step, epoch=epoch)
                            # Save training state (epoch progress) for mid-epoch resume
                            self.save_training_state(step=global_step, epoch=epoch, batch_idx=batch_idx + 1)
                            # Cleanup old checkpoints (LoRA uses 3-arg version, Full FT uses 1-arg version)
                            if hasattr(self, '_cleanup_old_checkpoints'):
                                import inspect
                                sig = inspect.signature(self._cleanup_old_checkpoints)
                                if len(sig.parameters) == 3:
                                    # LoRATrainer version: (current_step, max_to_keep, save_every)
                                    self._cleanup_old_checkpoints(global_step, max_step_saves_to_keep, save_every_n_steps)
                                else:
                                    # BaseTrainer/FullParameterTrainer version: (max_step_saves_to_keep)
                                    self._cleanup_old_checkpoints(max_step_saves_to_keep)
                            # Clear CUDA cache after checkpoint save to free temporary buffers
                            torch.cuda.empty_cache()

                        # Generate sample
                        # Also generate at step 0 to verify base model output
                        should_generate_sample = (
                            (global_step == 0 and sample_every_n_steps > 0) or
                            (global_step > 0 and global_step % sample_every_n_steps == 0)
                        )
                        if should_generate_sample:
                            print(f"{self.log_prefix} Generating sample with width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, steps={sample_steps}, seed={sample_seed}")
                            if self.is_zimage:
                                sample = self._generate_sample_zimage(
                                    prompt=sample_prompt,
                                    width=sample_width,
                                    height=sample_height,
                                    num_inference_steps=sample_steps,
                                    guidance_scale=sample_guidance_scale,
                                    seed=sample_seed
                                )
                            else:
                                sample = self.generate_sample(
                                    prompt=sample_prompt,
                                    width=sample_width,
                                    height=sample_height,
                                    num_inference_steps=sample_steps,
                                    guidance_scale=sample_guidance_scale,
                                    seed=sample_seed,
                                    current_step=global_step,
                                    schedule_type=sample_schedule_type
                                )

                            # Save sample with format matching API expectations: step_{step:06d}_sample_{i}.png
                            sample_path = self.output_dir / "samples" / f"step_{global_step:06d}_sample_0.png"
                            sample_path.parent.mkdir(parents=True, exist_ok=True)
                            sample.save(sample_path)
                            print(f"{self.log_prefix} Saved sample to {sample_path}")

                            # Log to TensorBoard (same as stable version)
                            import torchvision
                            image_tensor = torchvision.transforms.ToTensor()(sample)
                            self.writer.add_image("samples/sample_0", image_tensor, global_step=global_step)

                            # Free sample-related tensors and clear VRAM cache
                            del sample, image_tensor
                            torch.cuda.empty_cache()

                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                phase="training",
                                step=global_step,
                                total=actual_total_steps,
                                epoch=epoch,
                                loss=loss_value,
                            )

                        # Check if total_steps reached (step-based training)
                        if total_steps is not None and global_step >= total_steps:
                            print(f"\n{self.log_prefix} Reached target steps ({total_steps}), stopping training")
                            return  # Exit training loop
                    else:
                        # Gradient accumulation: Free loss tensor but don't do optimizer step yet
                        del loss, recon_loss

        except KeyboardInterrupt:
            print(f"\n{self.log_prefix} Training interrupted by user")
            print(f"{self.log_prefix} Saving checkpoint at step {global_step}, epoch {epoch}...")

            # Try to save checkpoint (even if it fails, continue to save state)
            checkpoint_saved = False
            try:
                self.save_checkpoint(step=global_step, epoch=epoch)
                checkpoint_saved = True
                print(f"{self.log_prefix} Checkpoint saved successfully")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to save checkpoint: {e}")
                import traceback
                traceback.print_exc()

            # Try to save training state (independent of checkpoint save)
            # Note: If stopped mid-MNT, skip the current batch and resume from next batch
            # This is acceptable as MNT iterations are gradient accumulation (can skip partial progress)
            state_saved = False
            try:
                self.save_training_state(step=global_step, epoch=epoch, batch_idx=batch_idx + 1)
                state_saved = True
                print(f"{self.log_prefix} Training state saved successfully")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to save training state: {e}")
                import traceback
                traceback.print_exc()

            # Try to cleanup old checkpoints (even if above failed)
            try:
                self._cleanup_old_checkpoints(max_step_saves_to_keep)
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to cleanup old checkpoints: {e}")
                import traceback
                traceback.print_exc()

            if checkpoint_saved and state_saved:
                print(f"{self.log_prefix} Checkpoint and state saved successfully, exiting...")
            elif checkpoint_saved:
                print(f"{self.log_prefix} Checkpoint saved (but state save failed), exiting...")
            elif state_saved:
                print(f"{self.log_prefix} State saved (but checkpoint save failed), exiting...")
            else:
                print(f"{self.log_prefix} WARNING: Both checkpoint and state save failed, exiting...")

            self.writer.close()
            raise

        print(f"{self.log_prefix} Training complete!")

        # Cleanup resources
        self.cleanup()

    def _log_metrics_to_db(self, step: int, loss: float, recon_loss: float, learning_rate: float):
        """
        Log training metrics to database (dual logging: TensorBoard + DB).

        Features:
        - UPSERT behavior: Same (run_id, step) will overwrite existing values
        - Allows training restart from checkpoint without duplicating metrics
        - Fast queries: indexed by (run_id, step) for incremental fetching

        Args:
            step: Global training step
            loss: Total loss value
            recon_loss: Reconstruction loss value
            learning_rate: Current learning rate
        """
        try:
            from database.models import TrainingMetrics
            from database import get_training_db

            # Get database session
            db = next(get_training_db())

            # UPSERT: Check if metric exists for this (run_id, step)
            existing = db.query(TrainingMetrics).filter(
                TrainingMetrics.run_id == self.run_id,
                TrainingMetrics.step == step
            ).first()

            if existing:
                # Update existing metric (training restarted from checkpoint)
                existing.loss = loss
                existing.recon_loss = recon_loss
                existing.learning_rate = learning_rate
                existing.timestamp = datetime.now()
            else:
                # Insert new metric
                metric = TrainingMetrics(
                    run_id=self.run_id,
                    step=step,
                    loss=loss,
                    recon_loss=recon_loss,
                    learning_rate=learning_rate
                )
                db.add(metric)

            # Commit every step (same as TensorBoard logging)
            db.commit()
            db.close()

        except Exception as e:
            # Non-critical: Continue training even if DB logging fails
            print(f"{self.log_prefix} WARNING: Failed to log metrics to DB: {e}")

    def cleanup(self):
        """
        Cleanup training resources.

        - Remove Layer Offload Conductor hooks
        - Restore layers to GPU
        - Close TensorBoard writer
        """
        print(f"{self.log_prefix} Cleaning up training resources...")

        # Cleanup Layer Offload Conductor
        if self.layer_offload_conductor is not None:
            print(f"{self.log_prefix} Cleaning up LayerOffloadConductor...")
            self.layer_offload_conductor.cleanup()
            self.layer_offload_conductor = None

        # Close TensorBoard writer
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            print(f"{self.log_prefix} TensorBoard writer closed")

        print(f"{self.log_prefix} Cleanup complete")
