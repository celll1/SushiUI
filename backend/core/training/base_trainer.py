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
from PIL import Image, PngImagePlugin
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
# Training Logger Helper
# ============================================================

def log_verbose(message: str):
    """
    Log verbose messages only to file (not to console).
    Uses global logger from train_runner.py if available.

    Args:
        message: Message to log
    """
    # Import logger from train_runner (circular import avoided by late import)
    try:
        from core.training.train_runner import logger
        if logger is not None:
            logger.log_only(message)
        # If logger not initialized, silently ignore (avoid spamming console during tests)
    except (ImportError, AttributeError):
        # Logger not available (e.g., during unit tests), silently ignore
        pass


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


def format_param_count(n: int) -> str:
    """Format parameter count as B (>=1B) or M (>=1M) or K."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    elif n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


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


def compute_snr(noise_scheduler, timesteps, alphas_cumprod_cached=None):
    """
    Computes SNR (Signal-to-Noise Ratio) from diffusion timesteps.

    SNR = alpha_bar / (1 - alpha_bar)

    Args:
        noise_scheduler: DDPMScheduler instance
        timesteps: Tensor of timesteps [batch_size]
        alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (optional, for performance)

    Returns:
        SNR values [batch_size]
    """
    # Get alpha_bar for each timestep
    # Use cached version if available (avoids repeated .to(device) calls)
    if alphas_cumprod_cached is not None:
        alphas_cumprod = alphas_cumprod_cached
    else:
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
    alpha_bar = alphas_cumprod[timesteps].float()

    # SNR = alpha / (1 - alpha)
    snr = alpha_bar / (1.0 - alpha_bar)

    return snr


def apply_snr_weight(loss, timesteps, noise_scheduler, min_snr_gamma=5.0, return_weights=False, alphas_cumprod_cached=None):
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
        return_weights: If True, also return the weight values [batch_size]
        alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (optional, for performance)

    Returns:
        Weighted loss (same shape as input)
        If return_weights=True: (weighted_loss, weights [batch_size])
    """
    snr = compute_snr(noise_scheduler, timesteps, alphas_cumprod_cached)

    # Min-SNR gamma weighting: min(SNR, gamma) / SNR
    # This clamps the weight for low-noise (high SNR) timesteps
    mse_loss_weights = torch.clamp(snr, max=min_snr_gamma) / snr

    # Keep original 1D weights for return
    weights_1d = mse_loss_weights.clone()

    # Reshape to match loss dimensions [batch_size, 1, 1, 1]
    while mse_loss_weights.dim() < loss.dim():
        mse_loss_weights = mse_loss_weights.unsqueeze(-1)

    # Apply weighting
    weighted_loss = loss * mse_loss_weights

    if return_weights:
        return weighted_loss, weights_1d
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
        # Flow Matching: x_t = (1 - t) * x_0 + t * noise
        # At t=0: x_t = x_0 (clean latents)
        # At t=1: x_t = noise (pure noise)
        # timesteps are continuous [0, 1]
        t = timesteps.float()
        while t.dim() < latents.dim():
            t = t.unsqueeze(-1)

        noisy_latents = (1.0 - t) * latents + t * noise
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
            # Predict velocity: v = noise - x_0 (direction from x_0 to noise)
            # This matches diffusers: target = noise - model_input
            return noise - latents

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
        # Flow Matching: x_t = (1 - t) * x_0 + t * noise
        # At t=0: x_t = x_0, At t=1: x_t = noise
        t = timesteps.float()
        while t.dim() < noisy_latents.dim():
            t = t.unsqueeze(-1)

        if prediction_target == "epsilon":
            # model_pred = noise, solve for x_0: x_0 = (x_t - t * noise) / (1 - t)
            # Avoid division by zero at t=1
            epsilon = 1e-8
            predicted_latent = (noisy_latents - t * model_pred) / (1.0 - t + epsilon)
        elif prediction_target == "velocity":
            # model_pred = v = noise - x_0
            # From diffusers: x_0 = x_t - t * v (line 459: x0 = sample - current_sigma * model_output)
            predicted_latent = noisy_latents - t * model_pred
        elif prediction_target == "sample":
            # model_pred = x_0 directly
            predicted_latent = model_pred
        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    else:
        raise ValueError(f"Unknown noise_process: {noise_process}")

    return predicted_latent


# ============================================================
# Parameter Change Tracker
# ============================================================

class ParameterChangeTracker:
    """
    Tracks per-component parameter changes during training.

    Computes two metrics every `interval` optimizer steps:
      B - Update norm:       ||θ_t - θ_{t-K}||_F  (how much changed in last K steps)
      C - Cumulative drift:  ||θ_t - θ_0||_F / ||θ_0||_F  (relative change from start)

    All computation and storage happens on CPU (fp16) → zero VRAM overhead.
    CPU RAM usage: ~2 × sum(component_param_bytes / 2) for full FT SDXL ≈ 14 GB total.
    """

    def __init__(self, components: Dict[str, torch.nn.Module], interval: int = 100):
        """
        Args:
            components: {name: module} for each trainable component
                        Keys: 'unet', 'te1', 'te2', 've'
            interval:   Compute metrics every N optimizer steps
        """
        self.components = {k: v for k, v in components.items() if v is not None}
        self.interval = interval

        # Reference snapshot for C (set once at init, never updated)
        self._reference: Dict[str, List[torch.Tensor]] = {}
        self._reference_norms: Dict[str, float] = {}

        # Previous snapshot for B (updated every `interval` steps)
        self._prev: Dict[str, List[torch.Tensor]] = {}

        self._initialize()

    def _snapshot(self, module: torch.nn.Module) -> List[torch.Tensor]:
        """Copy all trainable parameters to CPU as fp16 tensors."""
        return [p.detach().cpu().to(torch.float16)
                for p in module.parameters() if p.requires_grad]

    @staticmethod
    def _norm_sq(tensors: List[torch.Tensor]) -> float:
        """Compute sum of squared L2 norms (returns ||tensors||_F^2)."""
        total = 0.0
        for t in tensors:
            total += t.float().norm(2).item() ** 2
        return total

    @staticmethod
    def _delta_norm_sq(curr: List[torch.Tensor], ref: List[torch.Tensor]) -> float:
        """Compute ||curr - ref||_F^2 parameter-by-parameter to avoid large allocations."""
        total = 0.0
        for c, r in zip(curr, ref):
            delta = c.float() - r.float()
            total += delta.norm(2).item() ** 2
        return total

    def _initialize(self):
        total_params = 0
        total_bytes = 0
        for name, module in self.components.items():
            snap = self._snapshot(module)
            self._reference[name] = snap
            self._reference_norms[name] = self._norm_sq(snap) ** 0.5
            # Deep copy for prev (independent list of cloned tensors)
            self._prev[name] = [t.clone() for t in snap]
            n = sum(t.numel() for t in snap)
            total_params += n
            total_bytes += n * 2  # fp16 = 2 bytes per element
            print(f"[ParamTracker]   {name}: {n / 1e6:.1f}M params snapshot stored")
        print(f"[ParamTracker] Initialized. "
              f"Total tracked: {total_params / 1e6:.1f}M params, "
              f"~{total_bytes * 2 / 1e9:.1f} GB CPU RAM (ref + prev snapshots)")

    def compute(self, step: int) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute B and C metrics if `step` is a multiple of `interval`.

        Returns:
            {'update_norm': {name: float}, 'cumulative_drift': {name: float}}
            or None if not at interval boundary.
        """
        if step % self.interval != 0 or step == 0:
            return None

        update_norms: Dict[str, float] = {}
        cumulative_drifts: Dict[str, float] = {}

        for name, module in self.components.items():
            curr = self._snapshot(module)

            # B: update norm since last checkpoint
            update_norms[name] = self._delta_norm_sq(curr, self._prev[name]) ** 0.5

            # C: normalized cumulative drift from reference
            drift = self._delta_norm_sq(curr, self._reference[name]) ** 0.5
            ref_norm = self._reference_norms[name]
            cumulative_drifts[name] = drift / ref_norm if ref_norm > 0 else 0.0

            # Update prev for next B computation
            self._prev[name] = curr

        return {'update_norm': update_norms, 'cumulative_drift': cumulative_drifts}


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
        reconstruction_loss_weight: float = 0.0,
        # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
        prompt_chunking_mode: str = "a1111",  # "a1111", "sd_scripts", "nobos"
        max_prompt_chunks: int = 0,  # 0 = unlimited
        # Component-specific learning rates
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        image_encoder_lr: Optional[float] = None,  # Image Encoder (future T2I support)
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
        # Resume training
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Initialize base trainer.

        Args:
            model_path: Path to base Stable Diffusion model (or checkpoint for resume)
            resume_from_checkpoint: "latest" to auto-detect, or path to specific checkpoint
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

        # Metrics buffering for DB performance optimization
        # Batch DB commits every N steps instead of every step
        self._metrics_buffer = []
        self._metrics_flush_interval = 10  # Flush every 10 steps (configurable)

        # Async DB logging with ThreadPoolExecutor
        # DB writes happen in background thread, not blocking training loop
        from concurrent.futures import ThreadPoolExecutor
        self._db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_logger")
        self._db_futures = []  # Track pending futures for cleanup

        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Component-specific learning rates
        self.unet_lr = unet_lr if unet_lr is not None else learning_rate
        self.text_encoder_lr = text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_1_lr = text_encoder_1_lr if text_encoder_1_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_2_lr = text_encoder_2_lr if text_encoder_2_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.image_encoder_lr = image_encoder_lr if image_encoder_lr is not None else learning_rate

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

        # Resume training
        self.resume_from_checkpoint = resume_from_checkpoint
        self._loaded_checkpoint_path = None  # Actual checkpoint path loaded (may differ from requested if fallback occurred)

        # Convert dtype strings to torch.dtype
        self.weight_dtype = get_torch_dtype(weight_dtype)
        self.training_dtype = get_torch_dtype(training_dtype)
        self.output_dtype = get_torch_dtype(output_dtype)
        self.vae_dtype = get_torch_dtype(vae_dtype)
        self.mixed_precision = mixed_precision
        self.debug_vram = debug_vram
        self.use_flash_attention = use_flash_attention
        self.min_snr_gamma = min_snr_gamma
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # Initialize GradScaler for mixed precision training
        # GradScaler is needed when:
        # - training_dtype is fp16 (autocast is used)
        # - This includes cases where LoRA weights (fp32) are autocast to training dtype
        # GradScaler prevents gradient underflow during fp16 backward pass
        #
        # NOTE: BFloat16 does NOT need GradScaler because:
        # - BF16 has the same exponent range as FP32 (8 bits), so it doesn't suffer from
        #   the same overflow/underflow issues as FP16 (5 bit exponent)
        # - PyTorch's _amp_foreach_non_finite_check_and_unscale_cuda is not implemented for BF16
        # - Most modern training (FLUX.2, etc.) uses BF16 without GradScaler
        self.use_grad_scaler = (
            self.mixed_precision and
            self.training_dtype == torch.float16  # Only FP16, not BF16
        )
        if self.use_grad_scaler:
            from torch.cuda.amp import GradScaler

            # Use higher init_scale for FP16 to prevent gradient underflow
            # Problem: Initial gradients in LoRA training are very small (1e-7 ~ 1e-8)
            # FP16 smallest normal: ~6e-5, so gradients < 6e-5 underflow to 0
            # Solution: Use higher init_scale for FP16 (2^20 = 1048576)
            # - 1e-7 × 2^20 = 0.105 (representable in FP16)
            # - 1e-8 × 2^20 = 0.01 (representable in FP16)
            init_scale = 2**20  # 1048576 (higher scale for SD/SDXL)

            self.grad_scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
            print(f"[Trainer] GradScaler enabled for {training_dtype} training")
            print(f"[Trainer]   Init scale: {init_scale} (2^{init_scale.bit_length()-1})")
            print(f"[Trainer]   Weight dtype: {weight_dtype}")
            print(f"[Trainer]   Training dtype: {training_dtype}")
            if hasattr(self, 'lora_dtype'):
                print(f"[Trainer]   LoRA dtype: {self.lora_dtype}")
        else:
            self.grad_scaler = None
            if self.training_dtype == torch.bfloat16:
                print(f"[Trainer] GradScaler disabled (BF16 has FP32-equivalent exponent range, no scaling needed)")
            else:
                print(f"[Trainer] GradScaler disabled (training_dtype={training_dtype})")

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

        # Log component learning rates
        print(f"{self.log_prefix} ===== Component Learning Rates =====")
        print(f"{self.log_prefix}   Base LR: {self.learning_rate}")
        print(f"{self.log_prefix}   U-Net LR: {self.unet_lr}")
        print(f"{self.log_prefix}   Text Encoder LR: {self.text_encoder_lr}")
        if hasattr(self, 'text_encoder_1_lr'):
            print(f"{self.log_prefix}   Text Encoder 1 LR: {self.text_encoder_1_lr}")
        if hasattr(self, 'text_encoder_2_lr'):
            print(f"{self.log_prefix}   Text Encoder 2 LR: {self.text_encoder_2_lr}")
        # Note: Vision Encoder LR is logged in train() when VE is actually loaded
        print(f"{self.log_prefix} ====================================")

        print(f"[Trainer] Precision settings:")
        print(f"  Weight dtype: {weight_dtype} ({self.weight_dtype})")
        print(f"  Training dtype: {training_dtype} ({self.training_dtype})")
        print(f"  Output dtype: {output_dtype} ({self.output_dtype})")
        print(f"  VAE dtype: {vae_dtype} ({self.vae_dtype})")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Loss calculation: Always FP32 for numerical stability")
        print(f"  Min-SNR gamma: {min_snr_gamma} ({'enabled' if min_snr_gamma > 0 else 'disabled'})")

        # Warn about FP32 training VRAM usage
        if self.training_dtype == torch.float32:
            print(f"[Trainer] WARNING: training_dtype=fp32 uses ~2x VRAM compared to fp16/bf16")
            print(f"[Trainer] WARNING: Consider using training_dtype=fp16 or bf16 for large models")

        # Optimize: disable autocast if training_dtype is fp32 (no benefit, only overhead)
        # autocast with dtype=fp32 does nothing but adds context manager overhead
        if self.training_dtype == torch.float32:
            self.mixed_precision = False
            if mixed_precision:
                print(f"[Trainer] Note: mixed_precision disabled (training_dtype=fp32, autocast has no effect)")

        # Initialize tensorboard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = self.output_dir / "tensorboard" / timestamp
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        print(f"{self.log_prefix} Initializing on {self.device}")
        print(f"{self.log_prefix} Tensorboard logs: {tensorboard_dir}")

        # Check if resuming from checkpoint
        checkpoint_to_load = None
        if self.resume_from_checkpoint:
            if self.resume_from_checkpoint.lower() == "latest":
                # Find latest checkpoint in output directory
                checkpoint_files = list(self.output_dir.glob("*_step_*.safetensors"))
                if checkpoint_files:
                    # Get latest checkpoint by step number
                    def get_step(path):
                        try:
                            step_str = path.stem.split("_step_")[-1]
                            return int(step_str)
                        except (ValueError, IndexError):
                            return 0

                    latest_checkpoint = max(checkpoint_files, key=get_step)
                    checkpoint_to_load = str(latest_checkpoint)
                    print(f"{self.log_prefix} Found checkpoint to resume from: {checkpoint_to_load}")
            else:
                # Specific checkpoint path provided (treat as relative to output_dir first, then absolute)
                checkpoint_path_obj = self.output_dir / self.resume_from_checkpoint
                if checkpoint_path_obj.exists():
                    checkpoint_to_load = str(checkpoint_path_obj)
                    print(f"{self.log_prefix} Using specified checkpoint: {checkpoint_to_load}")
                else:
                    # Try as absolute path
                    checkpoint_path_obj = Path(self.resume_from_checkpoint)
                    if checkpoint_path_obj.exists():
                        checkpoint_to_load = str(checkpoint_path_obj)
                        print(f"{self.log_prefix} Using specified checkpoint (absolute path): {checkpoint_to_load}")

        if checkpoint_to_load:
            # Load checkpoint directly as base model (resume training)
            # Use fallback mechanism to handle corrupted checkpoints
            print(f"{self.log_prefix} Loading checkpoint as base model: {checkpoint_to_load}")
            try:
                self._load_checkpoint_as_base(checkpoint_to_load)
                print(f"{self.log_prefix} Successfully loaded checkpoint as base model")
                self._loaded_checkpoint_path = checkpoint_to_load
            except Exception as e:
                error_str = str(e).lower()
                # Check for corruption-related errors
                is_corruption = any(x in error_str for x in [
                    "incomplete metadata",
                    "file not fully covered",
                    "deserializing header",
                    "safetensor",
                    "corrupted",
                    "truncated",
                    "unexpected end",
                    "invalid header",
                ])

                if is_corruption:
                    print(f"{self.log_prefix} WARNING: Checkpoint appears corrupted: {e}")
                    print(f"{self.log_prefix} Attempting to fall back to previous checkpoint...")

                    # Try fallback mechanism
                    success, loaded_path = self._try_load_checkpoint_with_fallback(checkpoint_to_load)

                    if success and loaded_path:
                        print(f"{self.log_prefix} Successfully loaded fallback checkpoint: {loaded_path}")
                        self._loaded_checkpoint_path = loaded_path
                    else:
                        print(f"{self.log_prefix} ERROR: All checkpoints failed to load")
                        print(f"{self.log_prefix} Checkpoint loading failed, but resume_from_checkpoint was specified.")
                        print(f"{self.log_prefix} Aborting training to prevent unintended behavior.")
                        raise RuntimeError(
                            f"Failed to load checkpoint '{checkpoint_to_load}' and all fallback checkpoints. "
                            f"Training aborted to prevent starting from base model when resume was requested. "
                            f"Error: {e}"
                        )
                else:
                    # Non-corruption error, don't fallback
                    print(f"{self.log_prefix} ERROR: Failed to load checkpoint: {e}")
                    print(f"{self.log_prefix} Checkpoint loading failed, but resume_from_checkpoint was specified.")
                    print(f"{self.log_prefix} Aborting training to prevent unintended behavior.")
                    raise RuntimeError(
                        f"Failed to load checkpoint '{checkpoint_to_load}'. "
                        f"Training aborted to prevent starting from base model when resume was requested. "
                        f"Error: {e}"
                    )
        else:
            # Load base model (new training)
            print(f"{self.log_prefix} Loading model from {model_path}")
            self._load_model_components()

    def _load_model_components(self):
        """Load model components (dispatcher for different model types)."""
        # Detect model type
        from core.model_loader import ModelLoader
        model_type = ModelLoader.detect_model_type(self.model_path)
        self.is_zimage = (model_type == "zimage")
        # DEUS support removed - architecture no longer maintained
        self.is_deus = False  # (model_type == "deus")
        self.is_flux2 = (model_type == "flux2")
        self.is_anima = (model_type == "anima")
        self.is_sdxl = False

        if self.is_zimage:
            self._load_zimage_components()
        # DEUS support removed
        # elif self.is_deus:
        #     self._load_deus_components()
        elif self.is_flux2:
            self._load_flux2_components()
        elif self.is_anima:
            self._load_anima_components()
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

    # ============================================================
    # Anima (Cosmos-Predict2 DiT) component loading and training
    # ============================================================

    def _load_anima_components(self):
        """Load Anima model components for training.

        Anima ships as either a split-files HuggingFace layout or a single DiT
        safetensors plus separately-discovered Qwen3 / Qwen-Image VAE files.
        ModelLoader.load_anima_from_files handles both and returns a component
        dict identical to the one used by the inference path.
        """
        print(f"{self.log_prefix} Detected Anima model")
        print(f"{self.log_prefix} Loading Anima components from {self.model_path}")

        from core.model_loader import ModelLoader
        components = ModelLoader.load_anima_from_files(
            path=self.model_path,
            device="cpu",
            torch_dtype=self.weight_dtype,
        )

        # Store components on the trainer in the standard slots.
        self.transformer = components["transformer"]
        self.transformer_original = self.transformer  # No wrapper for Anima.
        self.vae = components["vae"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.t5_tokenizer = components["t5_tokenizer"]
        self.scheduler = components["scheduler"]

        # Anima specific: no dual TE / no U-Net.
        self.text_encoder_2 = None
        self.tokenizer_2 = None
        self.unet = None
        self.noise_scheduler = self.scheduler

        # Cast VAE to the desired dtype.
        self.vae = self.vae.to(dtype=self.vae_dtype)

        # Gradient checkpointing mode for the DiT blocks. Three options:
        #   standard         (default) — activations stay on GPU
        #   cpu_offload      — blocking CPU offload (saves VRAM, slower)
        #   async_cpu_offload — non-blocking CPU offload (saves VRAM, fast)
        # When both flags are True, async wins and we warn.
        cpu_offload_ckpt = bool(self.config.get("cpu_offload_checkpointing", False))
        async_offload_ckpt = bool(self.config.get("async_cpu_offload_checkpointing", False))
        if cpu_offload_ckpt and async_offload_ckpt:
            print(f"{self.log_prefix} WARNING: both cpu_offload_checkpointing and "
                  f"async_cpu_offload_checkpointing are True; using async (faster).")
            cpu_offload_ckpt = False
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing(
                cpu_offload=cpu_offload_ckpt,
                async_offload=async_offload_ckpt,
            )
            ckpt_mode = ("async_cpu_offload" if async_offload_ckpt
                          else "cpu_offload" if cpu_offload_ckpt else "standard")
            print(f"{self.log_prefix} Gradient checkpointing enabled for Anima DiT "
                  f"(mode={ckpt_mode})")
        if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
            self.text_encoder.gradient_checkpointing_enable()
            print(f"{self.log_prefix} Gradient checkpointing enabled for Qwen3 text encoder")

        # Freeze all base weights. Trainable LoRA modules are added later by the
        # adapter via apply_lora_to_unet.
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # Optional: FP8 the base DiT before LoRA wraps anything (LoRA-only).
        # Only safe when the base is frozen — which is true for the LoRA path
        # (Phase C.1 freezes everything before adapter injection). Full FT
        # needs trainable base weights, so silently ignore the flag with a
        # warning. We piggy-back on the Phase B.1-d inference quantiser which
        # patches each Linear's forward to dequantise on-the-fly.
        fp8_base_dtype = self.config.get("fp8_base_dtype") or None
        training_method = self.config.get("training_method", "lora")
        if fp8_base_dtype and training_method == "lora":
            print(f"{self.log_prefix} Quantising frozen Anima DiT base to "
                  f"{fp8_base_dtype} (LoRA-on-FP8-base, ~50% VRAM reduction)")
            from core.vram_optimization import _anima_quantize_fp8
            # deepcopy + patch — replaces self.transformer with the quantised
            # copy so subsequent block-swap and adapter wrap target the new
            # module references.
            self.transformer = _anima_quantize_fp8(
                self.transformer, fp8_base_dtype, "DiT (training base)",
            )
            # transformer_original keeps pointing at the quantised model too,
            # so downstream move_main_model_to_* keeps working.
            self.transformer_original = self.transformer
            self.transformer.requires_grad_(False)
        elif fp8_base_dtype:
            print(f"{self.log_prefix} WARNING: fp8_base_dtype={fp8_base_dtype} is "
                  f"only supported for training_method='lora' "
                  f"(current: {training_method!r}); ignoring.")

        # Plain GPU move. Block-swap init is deferred to setup_anima_block_swap(),
        # which is called by the trainer subclass AFTER any structural changes
        # (LoRA wrap / full-FT requires_grad toggling). The reason: the
        # LayerOffloadConductor snapshots each layer's state_dict at hook-
        # registration time, and a later LoRA wrap inserts new submodule keys
        # (.original_module.weight) that aren't in the snapshot, breaking the
        # CPU<->GPU swap with a KeyError. Setting up after the adapter avoids
        # that. The conductor handle is initialised to None here so callers
        # can rely on attribute presence.
        self.layer_offload_conductor = None
        if self.blocks_to_swap > 0:
            print(f"{self.log_prefix} Block Swap requested ({self.blocks_to_swap} blocks); "
                  f"deferred until adapter setup completes")
        print(f"{self.log_prefix} Moving Anima DiT to {self.device} "
              f"(block swap, if any, will redistribute after adapter setup)")
        self.transformer.to(self.device)

        print(f"{self.log_prefix} Anima model loaded successfully")
        print(f"{self.log_prefix} Scheduler: {self.scheduler.__class__.__name__}, "
              f"latent_channels=16")

    def setup_anima_block_swap(self):
        """Initialise the LayerOffloadConductor for the Anima DiT, AFTER any
        structural model changes (LoRA wrapping / full-FT param toggling).

        Idempotent: no-op when the trainer isn't on Anima, blocks_to_swap is
        0, or a conductor is already attached. The conductor snapshots each
        layer's state_dict at register_hooks() time, which is why this has to
        run after LoRALinearLayer wrappers (if any) have been inserted.
        """
        if not self.is_anima:
            return
        if self.blocks_to_swap <= 0:
            return
        if getattr(self, "layer_offload_conductor", None) is not None:
            return
        if not hasattr(self.transformer, "blocks"):
            raise ValueError("Anima DiT must expose `.blocks` (nn.ModuleList) for block swap")

        print(f"{self.log_prefix} [block-swap] initialising LayerOffloadConductor "
              f"(blocks_to_swap={self.blocks_to_swap}, pinned_memory={self.use_pinned_memory})")
        from core.memory_management import LayerOffloadConductor
        self.layer_offload_conductor = LayerOffloadConductor(
            layers=self.transformer.blocks,
            blocks_to_swap=self.blocks_to_swap,
            device=self.device,
            use_pinned_memory=self.use_pinned_memory,
            cpu_buffer_size_mb=8192,
            activation_buffer_size_mb=4096,
            enable_prefetch=True,
            enable_activation_offload=False,
        )
        self.transformer._layer_offload_conductor = self.layer_offload_conductor
        self.layer_offload_conductor.register_hooks()
        print(f"{self.log_prefix} [block-swap] LayerOffloadConductor hooks registered for Anima")

    # DEUS support removed - architecture no longer maintained
    # def _load_deus_components(self):
    #     """Load DEUS model components.
    #
    #     DEUS architecture:
    #     - SigLIP-2 text encoder (1152d output, variable sequence length)
    #     - U-Net with Transformer2DModel blocks
    #     - SDXL VAE (same scaling factor 0.13025)
    #     - DDPM epsilon prediction
    #
    #     Key differences from SDXL:
    #     - Single text encoder (SigLIP-2) vs dual CLIP
    #     - No pooled_embeddings
    #     - No time_ids / added_cond_kwargs
    #     """
    #     print(f"{self.log_prefix} Detected DEUS model")
    #     print(f"{self.log_prefix} Loading DEUS components from {self.model_path}")
    #
    #     from core.model_loader import ModelLoader
    #     from diffusers import DDPMScheduler
    #
    #     components = ModelLoader.load_deus_from_safetensors(
    #         file_path=self.model_path,
    #         device="cpu",
    #         torch_dtype=self.weight_dtype
    #     )
    #
    #     # Store components
    #     self.unet = components["unet"]
    #     self.vae = components["vae"]
    #     self.text_encoder = components["text_encoder"]
    #     self.tokenizer = components.get("tokenizer")
    #     self.processor = components.get("processor")
    #     self.scheduler = components["scheduler"]
    #     self.pipeline = components.get("pipeline")  # Keep reference for encode_prompt
    #
    #     # DEUS specific: no text_encoder_2, no transformer
    #     self.text_encoder_2 = None
    #     self.tokenizer_2 = None
    #     self.transformer = None
    #     self.transformer_original = None
    #
    #     # Create DDPM scheduler for training
    #     self.noise_scheduler = DDPMScheduler.from_config(self.scheduler.config)
    #
    #     # Save original scheduler for inference (sample generation)
    #     self.original_scheduler = self.scheduler
    #
    #     # Convert VAE to vae_dtype
    #     self.vae = self.vae.to(dtype=self.vae_dtype)
    #
    #     # Enable gradient checkpointing for U-Net (CRITICAL for VRAM reduction)
    #     if hasattr(self.unet, 'enable_gradient_checkpointing'):
    #         self.unet.enable_gradient_checkpointing()
    #         print(f"{self.log_prefix} Gradient checkpointing enabled for DEUS U-Net")
    #     else:
    #         print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for DEUS U-Net")
    #
    #     # Enable gradient checkpointing for Text Encoder
    #     if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
    #         self.text_encoder.gradient_checkpointing_enable()
    #         print(f"{self.log_prefix} Gradient checkpointing enabled for SigLIP-2 Text Encoder")
    #
    #     # Move VAE to device (always frozen during training)
    #     print(f"{self.log_prefix} Moving VAE to {self.device}...")
    #     self.vae.to(self.device)
    #
    #     # Move U-Net to device
    #     print(f"{self.log_prefix} Moving U-Net to {self.device}...")
    #     self.unet.to(self.device)
    #
    #     # Move Text Encoder to device
    #     print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
    #     self.text_encoder.to(self.device)
    #
    #     print(f"{self.log_prefix} DEUS model loaded successfully")
    #     print(f"{self.log_prefix} U-Net: {self.unet.__class__.__name__}")
    #     print(f"{self.log_prefix} Text Encoder: {self.text_encoder.__class__.__name__}")
    #     print(f"{self.log_prefix} Scheduler type: {self.scheduler.__class__.__name__}")
    #
    #     # Debug: Check for inf/nan in U-Net parameters
    #     unet_has_inf = False
    #     unet_has_nan = False
    #     for name, param in self.unet.named_parameters():
    #         if torch.isinf(param).any():
    #             print(f"{self.log_prefix} WARNING: U-Net param '{name}' contains inf!")
    #             unet_has_inf = True
    #         if torch.isnan(param).any():
    #             print(f"{self.log_prefix} WARNING: U-Net param '{name}' contains nan!")
    #             unet_has_nan = True
    #     if not unet_has_inf and not unet_has_nan:
    #         print(f"{self.log_prefix} U-Net parameters: No inf/nan detected")

    def _load_flux2_components(self):
        """Load FLUX.2 Klein model components.

        FLUX.2 Klein architecture:
        - Qwen3 text encoder (Qwen3ForCausalLM)
        - Flux2Transformer2DModel (8 dual stream + 48 single stream blocks)
        - AutoencoderKLFlux2 (32ch latent with BatchNorm)
        - Flow matching with velocity prediction
        - 4D position coordinates for RoPE (T, H, W, L)

        Key differences from FLUX.1:
        - Single stream blocks use parallel attention+MLP (fused projections)
        - VAE uses BatchNorm for latent normalization
        - Text encoder extracts hidden states from layers 9, 18, 27
        """
        print(f"{self.log_prefix} Detected FLUX.2 Klein model")
        print(f"{self.log_prefix} Loading FLUX.2 components from {self.model_path}")

        from core.model_loader import ModelLoader

        components = ModelLoader.load_flux2_from_safetensors(
            file_path=self.model_path,
            device="cpu",
            torch_dtype=self.weight_dtype
        )

        # Store components
        self.transformer = components["transformer"]
        self.transformer_original = self.transformer  # FLUX.2 doesn't need wrapper
        self.vae = components["vae"]
        self.text_encoder = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.scheduler = components["scheduler"]

        # FLUX.2 specific: no text_encoder_2, no unet
        self.text_encoder_2 = None
        self.tokenizer_2 = None
        self.unet = None
        self.noise_scheduler = self.scheduler

        # Save base model info for checkpoint metadata
        config = components.get("config", {})
        self.base_model_repo = config.get("base_model_repo", None)
        self.is_distilled = config.get("is_distilled", False)

        # Convert VAE to vae_dtype
        self.vae = self.vae.to(dtype=self.vae_dtype)

        # Enable gradient checkpointing for Transformer (CRITICAL for VRAM reduction)
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            print(f"{self.log_prefix} Gradient checkpointing enabled for FLUX.2 Transformer")
        else:
            print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for FLUX.2 Transformer")

        # Enable gradient checkpointing for Text Encoder (Qwen3)
        if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable()
            print(f"{self.log_prefix} Gradient checkpointing enabled for Qwen3 Text Encoder")

        # Setup Flash Attention if enabled
        if self.use_flash_attention:
            self._setup_flash_attention_flux2()

        # Freeze all base weights (full parameter training will unfreeze specific layers later)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # Setup Block Swap if enabled (before moving to GPU)
        self.flux2_block_offloader = None  # FLUX.2 specific offloader

        if self.blocks_to_swap > 0:
            print(f"{self.log_prefix} Block Swap enabled for FLUX.2 training: {self.blocks_to_swap} blocks")
            print(f"{self.log_prefix} Using FluxBlockOffloader (dual-list architecture)")
            print(f"{self.log_prefix} Pinned memory: {self.use_pinned_memory}")

            # Import FLUX.2 specific block offloader
            from core.memory_management import create_flux_block_offloader

            # Check if transformer has required attributes
            if not hasattr(self.transformer, 'transformer_blocks') or not hasattr(self.transformer, 'single_transformer_blocks'):
                raise ValueError(
                    f"FLUX.2 Transformer must have 'transformer_blocks' and 'single_transformer_blocks' attributes for Block Swap. "
                    f"Found: {type(self.transformer)}"
                )

            # Initialize FLUX.2 Block Offloader
            self.flux2_block_offloader = create_flux_block_offloader(
                transformer=self.transformer,
                blocks_to_swap=self.blocks_to_swap,
                device=self.device,
                target_dtype=self.training_dtype,
                use_pinned_memory=self.use_pinned_memory,
                supports_backward=True  # Training mode
            )

            # Prepare block devices (keep some on GPU, offload rest to CPU)
            self.flux2_block_offloader.prepare_block_devices_before_forward()

            num_dual = len(self.transformer.transformer_blocks)
            num_single = len(self.transformer.single_transformer_blocks)
            print(f"{self.log_prefix} FLUX.2 Block Swap initialized:")
            print(f"{self.log_prefix}   Dual stream blocks: {num_dual}")
            print(f"{self.log_prefix}   Single stream blocks: {num_single}")
            print(f"{self.log_prefix}   Total blocks: {num_dual + num_single}")
            print(f"{self.log_prefix}   Blocks to swap: {self.blocks_to_swap}")

            # Move VAE and Text Encoder to device (Transformer managed by block offloader)
            print(f"{self.log_prefix} Moving VAE to {self.device}...")
            self.vae.to(self.device)
            print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
            self.text_encoder.to(self.device)
        else:
            # No Block Swap: move everything to GPU
            print(f"{self.log_prefix} Moving VAE to {self.device}...")
            self.vae.to(self.device)

            print(f"{self.log_prefix} Moving Transformer to {self.device}...")
            self.transformer.to(self.device)

            print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
            self.text_encoder.to(self.device)

        print(f"{self.log_prefix} FLUX.2 model loaded successfully")
        print(f"{self.log_prefix} Transformer: {self.transformer.__class__.__name__}")
        print(f"{self.log_prefix} Text Encoder: {self.text_encoder.__class__.__name__}")
        print(f"{self.log_prefix} Scheduler type: {self.scheduler.__class__.__name__}")

        # Debug: Check for inf/nan in Transformer parameters
        transformer_has_inf = False
        transformer_has_nan = False
        for name, param in self.transformer.named_parameters():
            if torch.isinf(param).any():
                print(f"{self.log_prefix} WARNING: Transformer param '{name}' contains inf!")
                transformer_has_inf = True
            if torch.isnan(param).any():
                print(f"{self.log_prefix} WARNING: Transformer param '{name}' contains nan!")
                transformer_has_nan = True
        if not transformer_has_inf and not transformer_has_nan:
            print(f"{self.log_prefix} Transformer parameters: No inf/nan detected")

    def _load_checkpoint_as_base(self, checkpoint_path: str):
        """
        Load checkpoint directly as base model (for resume training).

        Uses same VRAM-optimized loading pattern as _load_model_components():
        - Load to CPU first
        - Move to GPU in controlled manner
        - Enable gradient checkpointing

        This avoids loading base model + checkpoint (VRAM duplication).

        Args:
            checkpoint_path: Path to checkpoint file (.safetensors)
        """
        from core.model_loader import ModelLoader
        from diffusers import DDPMScheduler, EulerAncestralDiscreteScheduler

        # Detect model type from checkpoint
        model_type = ModelLoader.detect_model_type(checkpoint_path)
        self.is_zimage = (model_type == "zimage")
        # DEUS support removed - architecture no longer maintained
        self.is_deus = False  # (model_type == "deus")
        self.is_flux2 = (model_type == "flux2")
        self.is_anima = (model_type == "anima")
        self.is_sdxl = False

        # DEUS support removed
        # if self.is_deus:
        #     print(f"{self.log_prefix} Loading DEUS checkpoint as base model")
        #     ...
        #     return

        if self.is_flux2:
            print(f"{self.log_prefix} Loading FLUX.2 checkpoint as base model")

            # FLUX.2 checkpoints from training are loaded via ModelLoader
            from core.model_loader import ModelLoader

            components = ModelLoader.load_flux2_from_safetensors(
                file_path=checkpoint_path,
                device="cpu",
                torch_dtype=self.weight_dtype
            )

            # Store components
            self.transformer = components["transformer"]
            self.transformer_original = self.transformer  # FLUX.2 doesn't need wrapper
            self.vae = components["vae"]
            self.text_encoder = components["text_encoder"]
            self.tokenizer = components["tokenizer"]
            self.scheduler = components["scheduler"]

            # FLUX.2 specific: no text_encoder_2, no unet
            self.text_encoder_2 = None
            self.tokenizer_2 = None
            self.unet = None
            self.noise_scheduler = self.scheduler

            # Convert VAE to vae_dtype
            self.vae = self.vae.to(dtype=self.vae_dtype)

            # Enable gradient checkpointing for Transformer (CRITICAL for VRAM reduction)
            if hasattr(self.transformer, 'enable_gradient_checkpointing'):
                self.transformer.enable_gradient_checkpointing()
                print(f"{self.log_prefix} Gradient checkpointing enabled for FLUX.2 Transformer")
            else:
                print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for FLUX.2 Transformer")

            # Enable gradient checkpointing for Text Encoder (Qwen3)
            if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Qwen3 Text Encoder")

            # Setup Flash Attention if enabled (FLUX.2 checkpoint resume)
            if self.use_flash_attention:
                self._setup_flash_attention_flux2()

            # Freeze all base weights (full parameter training will unfreeze specific layers later)
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.transformer.requires_grad_(False)

            # Setup Block Swap if enabled (before moving to GPU)
            self.flux2_block_offloader = None  # FLUX.2 specific offloader

            if self.blocks_to_swap > 0:
                print(f"{self.log_prefix} Block Swap enabled for FLUX.2 training: {self.blocks_to_swap} blocks")
                print(f"{self.log_prefix} Using FluxBlockOffloader (dual-list architecture)")
                print(f"{self.log_prefix} Pinned memory: {self.use_pinned_memory}")

                # Import FLUX.2 specific block offloader
                from core.memory_management import create_flux_block_offloader

                # Check if transformer has required attributes
                if not hasattr(self.transformer, 'transformer_blocks') or not hasattr(self.transformer, 'single_transformer_blocks'):
                    raise ValueError(
                        f"FLUX.2 Transformer must have 'transformer_blocks' and 'single_transformer_blocks' attributes for Block Swap. "
                        f"Found: {type(self.transformer)}"
                    )

                # Initialize FLUX.2 Block Offloader
                self.flux2_block_offloader = create_flux_block_offloader(
                    transformer=self.transformer,
                    blocks_to_swap=self.blocks_to_swap,
                    device=self.device,
                    target_dtype=self.training_dtype,
                    use_pinned_memory=self.use_pinned_memory,
                    supports_backward=True  # Training mode
                )

                # Prepare block devices (keep some on GPU, offload rest to CPU)
                self.flux2_block_offloader.prepare_block_devices_before_forward()

                num_dual = len(self.transformer.transformer_blocks)
                num_single = len(self.transformer.single_transformer_blocks)
                print(f"{self.log_prefix}   FLUX.2 Block Swap initialized:")
                print(f"{self.log_prefix}   Dual stream blocks: {num_dual}")
                print(f"{self.log_prefix}   Single stream blocks: {num_single}")
                print(f"{self.log_prefix}   Total blocks: {num_dual + num_single}")
                print(f"{self.log_prefix}   Blocks to swap: {self.blocks_to_swap}")

                # Move VAE and Text Encoder to device (Transformer managed by block offloader)
                print(f"{self.log_prefix} Moving VAE to {self.device}...")
                self.vae.to(self.device)
                print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
                self.text_encoder.to(self.device)
            else:
                # No Block Swap: move everything to GPU
                print(f"{self.log_prefix} Moving VAE to {self.device}...")
                self.vae.to(self.device)

                print(f"{self.log_prefix} Moving Transformer to {self.device}...")
                self.transformer.to(self.device)

                print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
                self.text_encoder.to(self.device)

            print(f"{self.log_prefix} FLUX.2 checkpoint loaded successfully as base model")
            return

        elif self.is_zimage:
            print(f"{self.log_prefix} Loading Z-Image checkpoint as base model")

            # Z-Image checkpoints from training are saved with all components
            # We can load them as a complete model checkpoint
            from core.model_loader import ModelLoader

            # Detect format (ComfyUI or diffusers)
            from safetensors import safe_open
            with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
                keys = list(f.keys())
                # ComfyUI format has keys like "model.diffusion_model.x_embedder.proj.weight"
                # Diffusers format has keys like "transformer.x_embedder.proj.weight"
                is_comfy_format = any(k.startswith("model.diffusion_model.") for k in keys)

            if is_comfy_format:
                # ComfyUI format checkpoint
                print(f"{self.log_prefix} Detected ComfyUI format Z-Image checkpoint")
                components = ModelLoader.load_zimage_from_comfy_safetensors(
                    file_path=checkpoint_path,
                    device="cpu",
                    torch_dtype=self.weight_dtype,
                    base_model_repo="Tongyi-MAI/Z-Image-Turbo"
                )
            else:
                # Diffusers format checkpoint (training checkpoint)
                # Extract checkpoint directory (assumes checkpoint is in training output dir with other components)
                checkpoint_dir = Path(checkpoint_path).parent

                # Check if other components exist in the same directory
                # Training saves: model_step_xxx.safetensors, vae/, text_encoder/, tokenizer/, scheduler/
                if (checkpoint_dir / "vae").exists():
                    # Load from directory structure
                    print(f"{self.log_prefix} Loading Z-Image from checkpoint directory: {checkpoint_dir}")
                    components = ModelLoader.load_zimage_from_diffusers(
                        model_path=str(checkpoint_dir),
                        device="cpu",
                        torch_dtype=self.weight_dtype
                    )

                    # Load transformer weights from checkpoint file
                    from safetensors.torch import load_file
                    print(f"{self.log_prefix} Loading transformer weights from: {checkpoint_path}")
                    transformer_state_dict = load_file(checkpoint_path, device="cpu")
                    components["transformer"].load_state_dict(transformer_state_dict, strict=False)
                else:
                    # Single-file checkpoint with all components (full model save)
                    # This requires special handling - for now, raise error
                    raise RuntimeError(
                        f"Z-Image checkpoint resume from single-file format not yet supported. "
                        f"Please ensure checkpoint directory contains vae/, text_encoder/, tokenizer/, scheduler/ subdirectories. "
                        f"Checkpoint: {checkpoint_path}"
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

            # Save original scheduler for inference (sample generation)
            self.original_scheduler = self.scheduler

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

            # Move models to GPU (controlled, VRAM-optimized)
            # Text Encoder first (smallest)
            print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
            self.text_encoder.to(self.device)

            # Transformer second (Block Swap will be set up separately if enabled)
            if self.blocks_to_swap > 0:
                print(f"{self.log_prefix} Block Swap enabled: Transformer will stay on CPU during setup")
                # Block Swap setup will happen later in training setup
            else:
                print(f"{self.log_prefix} Moving Transformer to {self.device}...")
                self.transformer_original.to(self.device)

            # VAE stays on CPU (moved to GPU only during sample generation)
            print(f"{self.log_prefix} VAE remains on CPU (will move to GPU during sample generation)")

            print(f"{self.log_prefix} Z-Image checkpoint loaded successfully as base model")

        else:
            # SD/SDXL checkpoint resume
            print(f"{self.log_prefix} Loading SD/SDXL checkpoint as base model")

            from safetensors import safe_open

            # Peek at keys only (reads header, not tensors) to detect model type
            with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
                checkpoint_keys = list(f.keys())

            # Detect if SDXL or SD1.5 based on state dict keys
            # SDXL has text_encoder_2 keys
            is_sdxl_model = any("text_model_2" in k or "conditioner.embedders.1" in k for k in checkpoint_keys)

            # Load components using diffusers from_single_file
            # This properly reconstructs the model from checkpoint state dict
            if is_sdxl_model:
                print(f"{self.log_prefix} Detected SDXL checkpoint")
                from diffusers import StableDiffusionXLPipeline

                temp_pipeline = StableDiffusionXLPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=self.weight_dtype,
                    use_safetensors=True,
                    device_map=None,  # Load to CPU first
                )
            else:
                print(f"{self.log_prefix} Detected SD1.5 checkpoint")
                from diffusers import StableDiffusionPipeline

                temp_pipeline = StableDiffusionPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=self.weight_dtype,
                    use_safetensors=True,
                    device_map=None,  # Load to CPU first
                )

            # Extract components
            self.vae = temp_pipeline.vae
            self.text_encoder = temp_pipeline.text_encoder
            self.tokenizer = temp_pipeline.tokenizer
            self.unet = temp_pipeline.unet

            # Save original scheduler for inference (sample generation)
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

            # Store SDXL flag
            self.is_sdxl = is_sdxl_model

            # No transformer for SD/SDXL
            self.transformer = None
            self.transformer_original = None

            # Clean up temporary pipeline
            del temp_pipeline
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Convert VAE to vae_dtype
            self.vae = self.vae.to(dtype=self.vae_dtype)

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

            # Freeze VAE
            self.vae.requires_grad_(False)

            # Move models to GPU (controlled, VRAM-optimized)
            # Text Encoder 1 first (smallest)
            print(f"{self.log_prefix} Moving Text Encoder 1 to {self.device}...")
            self.text_encoder.to(self.device)

            # Text Encoder 2 (if SDXL)
            if self.text_encoder_2 is not None:
                print(f"{self.log_prefix} Moving Text Encoder 2 to {self.device}...")
                self.text_encoder_2.to(self.device)

            # U-Net second
            print(f"{self.log_prefix} Moving U-Net to {self.device}...")
            self.unet.to(self.device)

            # VAE stays on CPU (moved to GPU only during sample generation)
            print(f"{self.log_prefix} VAE remains on CPU (will move to GPU during sample generation)")

            print(f"{self.log_prefix} {'SDXL' if is_sdxl_model else 'SD1.5'} checkpoint loaded successfully as base model")

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
        """Setup Flash Attention for SD/SDXL models.

        Uses diffusers' set_attention_backend('flash') which requires flash-attn package.
        This is the same modern API used by FLUX.2 and other DiT models.

        Available backends: 'flash', 'sage', 'native', 'xformers', etc.
        See: diffusers.models.attention_dispatch.AttentionBackendName
        """
        if self.unet is None:
            print(f"{self.log_prefix} WARNING: UNet not loaded, skipping Flash Attention setup")
            return

        try:
            # Use set_attention_backend('flash') - modern diffusers API (same as FLUX.2)
            print(f"{self.log_prefix} Setting Flash Attention backend for SD/SDXL UNet...")
            self.unet.set_attention_backend("flash")
            print(f"{self.log_prefix} [OK] Flash Attention enabled via set_attention_backend('flash')")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to enable Flash Attention: {e}")
            print(f"{self.log_prefix} Ensure flash-attn is installed: pip install flash-attn")

    def _setup_flash_attention_flux2(self):
        """Setup Flash Attention for FLUX.2 models.

        Uses diffusers' set_attention_backend('flash') which requires flash-attn package.
        Same modern API as SD/SDXL.

        Available backends: 'flash', 'sage', 'native', 'xformers', etc.
        See: diffusers.models.attention_dispatch.AttentionBackendName
        """
        if self.transformer is None:
            print(f"{self.log_prefix} WARNING: Transformer not loaded, skipping Flash Attention setup")
            return

        try:
            # Use set_attention_backend('flash') - modern diffusers API
            print(f"{self.log_prefix} Setting Flash Attention backend for FLUX.2 Transformer...")
            self.transformer.set_attention_backend("flash")
            print(f"{self.log_prefix} [OK] Flash Attention enabled via set_attention_backend('flash')")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to enable Flash Attention: {e}")
            print(f"{self.log_prefix} Ensure flash-attn is installed: pip install flash-attn")

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

    def _save_vision_encoder_checkpoint(self, step: int, epoch: int):
        """
        Save Vision Encoder checkpoint as a separate safetensors file (if loaded and trained).

        The VE checkpoint is saved alongside the main checkpoint with the suffix
        '_vision_encoder_step_XXXXXX.safetensors', independent of the main model format.
        """
        ve_obj = getattr(self, 'vision_encoder', None)
        if ve_obj is None:
            return

        try:
            from safetensors.torch import save_file
            ve_path = self.output_dir / f"{self.run_name}_vision_encoder_step_{step:06d}.safetensors"
            ve_sd = ve_obj.state_dict_for_save()
            metadata = {
                "step": str(step),
                "epoch": str(epoch),
                "model_type": "siglip2_vision_encoder",
            }
            save_file(ve_sd, ve_path, metadata=metadata)
            print(f"{self.log_prefix} Saved Vision Encoder checkpoint: {ve_path}")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to save Vision Encoder checkpoint: {e}")

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

    def _compute_dataset_fingerprint(self, datasets: List[Any]) -> dict:
        """
        Compute a fingerprint of the dataset structure for change detection on resume.

        This fingerprint is used to detect if the dataset has changed between training sessions.
        If the dataset changes, the saved random_state (shuffle order) becomes invalid.

        IMPORTANT: Only image file information is included in the fingerprint.
        Caption changes do NOT invalidate the shuffle state because:
        - Captions don't affect the order of images in batches
        - Users may want to edit captions without losing training progress

        Args:
            datasets: List of dataset objects

        Returns:
            Dict containing:
                - dataset_ids: List of dataset unique_ids
                - total_item_count: Total number of items across all datasets
                - image_paths_hash: Hash of sorted image paths (to detect additions/removals)
        """
        import hashlib

        dataset_ids = []
        all_image_paths = []

        for dataset in datasets:
            dataset_ids.append(dataset.unique_id)
            for item in dataset.items:
                # Only include image_path - captions are intentionally excluded
                all_image_paths.append(item.get("image_path", ""))

        # Sort paths for consistent hashing (order within dataset matters, but we hash sorted for detection)
        # Actually, we want to detect if the SET of images changed, not their order
        sorted_paths = sorted(all_image_paths)
        paths_str = "\n".join(sorted_paths)
        paths_hash = hashlib.md5(paths_str.encode('utf-8')).hexdigest()

        return {
            "dataset_ids": dataset_ids,
            "total_item_count": len(all_image_paths),
            "image_paths_hash": paths_hash,
        }

    def _check_dataset_fingerprint_changed(self, saved_fingerprint: Optional[dict], current_fingerprint: dict) -> bool:
        """
        Check if the dataset fingerprint has changed since the checkpoint was saved.

        Args:
            saved_fingerprint: Fingerprint from saved training state (may be None for old checkpoints)
            current_fingerprint: Current dataset fingerprint

        Returns:
            True if dataset has changed (shuffle state should be invalidated)
        """
        if saved_fingerprint is None:
            # Old checkpoint without fingerprint - assume unchanged for backward compatibility
            print(f"{self.log_prefix} No dataset fingerprint in saved state (old checkpoint format)")
            return False

        # Check if any key component changed
        if saved_fingerprint.get("total_item_count") != current_fingerprint.get("total_item_count"):
            print(f"{self.log_prefix} Dataset item count changed: {saved_fingerprint.get('total_item_count')} -> {current_fingerprint.get('total_item_count')}")
            return True

        if saved_fingerprint.get("image_paths_hash") != current_fingerprint.get("image_paths_hash"):
            print(f"{self.log_prefix} Dataset image paths changed (hash mismatch)")
            return True

        if saved_fingerprint.get("dataset_ids") != current_fingerprint.get("dataset_ids"):
            print(f"{self.log_prefix} Dataset IDs changed: {saved_fingerprint.get('dataset_ids')} -> {current_fingerprint.get('dataset_ids')}")
            return True

        return False

    def save_training_state(self, step: int, epoch: int, batch_idx: int, multi_noise_timesteps: int = 1):
        """
        Save training state (epoch progress, batch index, random state) to JSON file.

        This is saved separately from the model checkpoint to keep checkpoint files lightweight.
        Enables mid-epoch resume without re-processing already trained batches.

        Args:
            step: Current global step
            epoch: Current epoch (0-indexed)
            batch_idx: Current batch index within epoch (next batch to process)
            multi_noise_timesteps: MNT value at checkpoint time (for MNT-change detection on resume)
        """
        import json
        import random

        # Use full run_name with zero-padded step (consistent with model checkpoint naming)
        state_file = self.output_dir / f"{self.run_name}_step_{step:06d}_state.json"

        state = {
            "global_step": step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "multi_noise_timesteps": multi_noise_timesteps,  # Save MNT for resume calculation
            "random_state": random.getstate(),  # Save Python random state for batch shuffle reproducibility
            # Dataset fingerprint for change detection on resume
            "dataset_fingerprint": getattr(self, '_dataset_fingerprint', None),
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
            Dict with keys: global_step, epoch, batch_idx, random_state, dataset_fingerprint
            None if state file not found
        """
        import json
        import random
        import re

        # Try new naming format first (consistent with model checkpoint)
        state_file = self.output_dir / f"{self.run_name}_step_{step:06d}_state.json"

        # Fallback to old naming format (short name, no leading zeros) for backward compatibility
        if not state_file.exists():
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)
                state_file_legacy = self.output_dir / f"{short_name}_step_{step}_state.json"
                if state_file_legacy.exists():
                    state_file = state_file_legacy
                    print(f"{self.log_prefix} Using legacy training state file: {state_file.name}")

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

    def save_optimizer_state(self, step: int):
        """
        Save optimizer state dict to .pt file.

        Args:
            step: Current global step
        """
        if self.optimizer is None:
            return

        # Use full run_name with zero-padded step (consistent with model checkpoint naming)
        optimizer_file = self.output_dir / f"{self.run_name}_step_{step:06d}_optimizer.pt"

        # Save optimizer state dict
        torch.save(self.optimizer.state_dict(), optimizer_file)
        print(f"{self.log_prefix} Saved optimizer state to {optimizer_file.name}")

    def load_optimizer_state(self, step: int) -> bool:
        """
        Load optimizer state dict from .pt file.

        Args:
            step: Step number to load optimizer state for

        Returns:
            True if successfully loaded, False otherwise
        """
        import re

        if self.optimizer is None:
            print(f"{self.log_prefix} WARNING: Cannot load optimizer state (optimizer not initialized)")
            return False

        # Try new naming format first (consistent with model checkpoint)
        optimizer_file = self.output_dir / f"{self.run_name}_step_{step:06d}_optimizer.pt"

        # Fallback to old naming format (short name, no leading zeros) for backward compatibility
        if not optimizer_file.exists():
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)
                optimizer_file_legacy = self.output_dir / f"{short_name}_step_{step}_optimizer.pt"
                if optimizer_file_legacy.exists():
                    optimizer_file = optimizer_file_legacy
                    print(f"{self.log_prefix} Using legacy optimizer state file: {optimizer_file.name}")

        if not optimizer_file.exists():
            print(f"{self.log_prefix} No optimizer state file found: {optimizer_file.name}")
            print(f"{self.log_prefix} Starting with fresh optimizer state")
            return False

        def move_tensors_to_device(obj, device):
            """Recursively move all tensors in nested dict/list to target device."""
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: move_tensors_to_device(v, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_tensors_to_device(v, device) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(move_tensors_to_device(v, device) for v in obj)
            else:
                return obj

        try:
            # Load optimizer state dict
            optimizer_state = torch.load(optimizer_file, map_location='cpu')

            # Recursively move all optimizer state tensors to GPU
            # This is necessary for 8-bit optimizers that have CUDA-only buffers
            # (absmax_z, absmax1, absmax2, etc.) which must be on CUDA device
            print(f"{self.log_prefix} Moving optimizer state tensors to {self.device}...")
            optimizer_state = move_tensors_to_device(optimizer_state, self.device)

            # Attempt to load state dict with error handling
            try:
                self.optimizer.load_state_dict(optimizer_state)

                # IMPORTANT: After load_state_dict(), move all tensors in optimizer.state to GPU
                # load_state_dict() may create new tensor references, so we need to move again
                moved_count = 0
                for param_state in self.optimizer.state.values():
                    for key, value in param_state.items():
                        if isinstance(value, torch.Tensor) and not value.is_cuda:
                            param_state[key] = value.to(self.device)
                            moved_count += 1
                if moved_count > 0:
                    print(f"{self.log_prefix} Moved {moved_count} optimizer state tensors to {self.device}")

                print(f"{self.log_prefix} Successfully loaded optimizer state from {optimizer_file.name}")
                return True
            except Exception as e:
                # Fallback: Optimizer configuration changed (e.g., different optimizer type, LR, etc.)
                print(f"{self.log_prefix} WARNING: Failed to load optimizer state: {e}")
                print(f"{self.log_prefix} This can happen if:")
                print(f"{self.log_prefix}   - Optimizer type was changed")
                print(f"{self.log_prefix}   - Model architecture was changed")
                print(f"{self.log_prefix}   - Number of trainable parameters changed")
                print(f"{self.log_prefix} Continuing with fresh optimizer state (momentum/variance will be reset)")
                return False
        except Exception as e:
            print(f"{self.log_prefix} ERROR: Failed to load optimizer file: {e}")
            print(f"{self.log_prefix} Continuing with fresh optimizer state")
            return False

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

    def _get_sorted_checkpoints(self) -> List[Tuple[Path, int]]:
        """
        Get all checkpoints sorted by step number (descending, newest first).

        Returns:
            List of (checkpoint_path, step_number) tuples, sorted newest first.
            Empty list if no checkpoints exist.
        """
        checkpoint_files = list(self.output_dir.glob("*_step_*.safetensors"))

        if not checkpoint_files:
            return []

        def get_step(path):
            try:
                step_str = path.stem.split("_step_")[-1]
                return int(step_str)
            except (ValueError, IndexError):
                return 0

        # Sort by step number descending (newest first)
        sorted_checkpoints = sorted(checkpoint_files, key=get_step, reverse=True)
        return [(ckpt, get_step(ckpt)) for ckpt in sorted_checkpoints]

    def _try_load_checkpoint_with_fallback(self, checkpoint_path: str) -> Tuple[bool, Optional[str]]:
        """
        Try to load a checkpoint, with fallback to previous checkpoints if corrupted.

        Args:
            checkpoint_path: Path to the checkpoint to load (or "latest" for auto-detection)

        Returns:
            Tuple of (success, loaded_checkpoint_path).
            If success is False, loaded_checkpoint_path is None.
        """
        # Get sorted list of all checkpoints
        sorted_checkpoints = self._get_sorted_checkpoints()

        if not sorted_checkpoints:
            print(f"{self.log_prefix} No checkpoints found for fallback")
            return (False, None)

        # If specific checkpoint was requested, find its index
        if checkpoint_path and checkpoint_path.lower() != "latest":
            checkpoint_path_obj = Path(checkpoint_path)
            start_idx = 0
            for i, (ckpt, step) in enumerate(sorted_checkpoints):
                if ckpt.name == checkpoint_path_obj.name or str(ckpt) == checkpoint_path:
                    start_idx = i
                    break
        else:
            # Start from the newest checkpoint
            start_idx = 0

        # Try loading checkpoints starting from the requested one
        for i in range(start_idx, len(sorted_checkpoints)):
            ckpt_path, ckpt_step = sorted_checkpoints[i]
            ckpt_path_str = str(ckpt_path)

            if i > start_idx:
                print(f"{self.log_prefix} Attempting fallback to previous checkpoint: {ckpt_path.name} (step {ckpt_step})")

            try:
                self._load_checkpoint_as_base(ckpt_path_str)
                if i > start_idx:
                    print(f"{self.log_prefix} Successfully loaded fallback checkpoint: {ckpt_path.name}")
                return (True, ckpt_path_str)
            except Exception as e:
                error_str = str(e).lower()
                # Check for corruption-related errors
                is_corruption = any(x in error_str for x in [
                    "incomplete metadata",
                    "file not fully covered",
                    "deserializing header",
                    "safetensor",
                    "corrupted",
                    "truncated",
                    "unexpected end",
                    "invalid header",
                ])

                if is_corruption:
                    print(f"{self.log_prefix} WARNING: Checkpoint corrupted: {ckpt_path.name}")
                    print(f"{self.log_prefix}   Error: {e}")
                    if i + 1 < len(sorted_checkpoints):
                        print(f"{self.log_prefix}   Will try previous checkpoint...")
                        continue
                    else:
                        print(f"{self.log_prefix} ERROR: No more checkpoints to try")
                        return (False, None)
                else:
                    # Non-corruption error, don't fallback
                    print(f"{self.log_prefix} ERROR: Failed to load checkpoint (non-corruption): {e}")
                    raise

        print(f"{self.log_prefix} ERROR: All checkpoints failed to load")
        return (False, None)

    def _cleanup_old_checkpoints(self, max_step_saves_to_keep: int):
        """
        Delete old checkpoints, keeping only the most recent N checkpoints.

        Args:
            max_step_saves_to_keep: Maximum number of checkpoints to keep (0 = keep all)
        """
        if max_step_saves_to_keep <= 0:
            return

        # Find main checkpoint files only (exclude VE checkpoints to avoid double-counting)
        checkpoint_files = [
            f for f in self.output_dir.glob("*_step_*.safetensors")
            if "vision_encoder" not in f.name
        ]
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
            step_num = get_step(checkpoint_path)
            # Also delete associated _optimizer.pt file and _state.json file
            # Pattern: {short_name}_step_{step}.safetensors
            #          {short_name}_step_{step}_optimizer.pt
            #          {short_name}_step_{step}_state.json
            optimizer_pt_path = checkpoint_path.parent / f"{checkpoint_path.stem}_optimizer.pt"
            state_json_path = checkpoint_path.parent / f"{checkpoint_path.stem}_state.json"

            print(f"{self.log_prefix} Deleting old checkpoint: {checkpoint_path.name}")
            checkpoint_path.unlink()

            if optimizer_pt_path.exists():
                print(f"{self.log_prefix} Deleting old optimizer state: {optimizer_pt_path.name}")
                optimizer_pt_path.unlink()

            if state_json_path.exists():
                print(f"{self.log_prefix} Deleting old training state: {state_json_path.name}")
                state_json_path.unlink()

            # Also delete VE checkpoint for this step if it exists
            ve_pattern = f"*_vision_encoder_step_{step_num:06d}.safetensors"
            for ve_file in checkpoint_path.parent.glob(ve_pattern):
                print(f"{self.log_prefix} Deleting old VE checkpoint: {ve_file.name}")
                ve_file.unlink()

    # ============================================================
    # Optimizer Setup
    # ============================================================

    def _build_component_lr_list(self):
        """
        Build a (component_lrs, component_names) pair matching the actual optimizer
        param group order created by setup_trainable_parameters() + VE append.

        Group ordering:
          - UNet (if train_unet)
          - TE1 (if train_text_encoder and text_encoder is not None)
          - TE2 (if train_text_encoder and is_sdxl and text_encoder_2 is not None)
          - VE  (if _train_vision_encoder and vision_encoder is not None)

        Returns:
            Tuple[List[float], List[str]]: (lrs, names) matching optimizer group indices
        """
        lrs = []
        names = []

        if getattr(self, 'train_unet', True) and getattr(self, 'unet', None) is not None:
            lrs.append(getattr(self, 'unet_lr', self.learning_rate))
            names.append("U-Net")

        if getattr(self, 'train_text_encoder', False):
            if getattr(self, 'text_encoder', None) is not None:
                lrs.append(getattr(self, 'text_encoder_1_lr',
                                   getattr(self, 'text_encoder_lr', self.learning_rate)))
                names.append("TE1")
            if getattr(self, 'is_sdxl', False) and getattr(self, 'text_encoder_2', None) is not None:
                lrs.append(getattr(self, 'text_encoder_2_lr', self.learning_rate))
                names.append("TE2")

        if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
            ve_lr = getattr(self, '_vision_encoder_lr', None) or getattr(self, 'text_encoder_lr', self.learning_rate)
            lrs.append(ve_lr)
            names.append("VisionEncoder")

        return lrs, names

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

        # Add Vision Encoder parameters if training is enabled
        if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
            ve_lr = getattr(self, '_vision_encoder_lr', None) or self.text_encoder_lr
            ve_params = list(self.vision_encoder.parameters())
            if ve_params:
                param_groups.append({"params": ve_params, "lr": ve_lr})
                ve_total = sum(p.numel() for p in ve_params)
                print(f"{self.log_prefix} Vision Encoder: Added {len(ve_params)} param tensors ({ve_total/1e6:.1f}M params, lr={ve_lr}) to optimizer")
                # Set requires_grad on VE model
                for p in ve_params:
                    p.requires_grad_(True)

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

        # Set optimizer to train mode (required for RingBuffer optimizers)
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
            print(f"{self.log_prefix} Optimizer set to train mode")

        # Log actual LR values for each parameter group
        print(f"{self.log_prefix} ===== Optimizer Parameter Group LRs =====")
        for i, group in enumerate(self.optimizer.param_groups):
            group_lr = group.get('lr', 'N/A')
            num_tensors = len(group['params'])
            num_scalars = sum(p.numel() for p in group['params'])
            print(f"{self.log_prefix}   Group {i}: lr={group_lr}, tensors={num_tensors}, params={format_param_count(num_scalars)}")
        print(f"{self.log_prefix} ==========================================")

        # Setup LR scheduler
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        self.lr_scheduler = get_diffusers_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.optimizer_warmup_steps,
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

        # Set all optimizers to train mode (required for RingBuffer optimizers)
        for optimizer in optimizers:
            if hasattr(optimizer, 'train'):
                optimizer.train()
        print(f"{self.log_prefix} All {len(optimizers)} optimizers set to train mode")

        # Create LR schedulers for all optimizers
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        lr_schedulers = []
        for optimizer in optimizers:
            lr_scheduler = get_diffusers_scheduler(
                lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.optimizer_warmup_steps,
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

    def _has_fp8_text_encoder(self) -> bool:
        """
        Check if text encoder has FP8 quantized weights.

        Returns:
            True if any text encoder has FP8 weights
        """
        # Check text_encoder
        if self.text_encoder is not None:
            for module in self.text_encoder.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        return True

        # Check text_encoder_2 (SDXL)
        if self.text_encoder_2 is not None:
            for module in self.text_encoder_2.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        return True

        return False

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
        # DEUS support removed
        # if self.is_deus:
        #     return self._encode_prompt_deus(prompt, requires_grad)

        # Check prompt length - use tokenizer_2 for SDXL as it determines chunking
        tokenizer = self.tokenizer_2 if self.is_sdxl else self.tokenizer
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]

        # If prompt is short (<=75 tokens), use simple encoding
        if len(tokens) <= 75:
            return self._encode_prompt_simple(prompt, requires_grad)

        # Long prompt - use chunking
        return self._encode_prompt_chunked(prompt, requires_grad)

    # DEUS support removed - architecture no longer maintained
    # def _encode_prompt_deus(self, prompt: str, requires_grad: bool = False):
    #     """
    #     Encode prompt using DEUS's SigLIP-2 text encoder.
    #     ...
    #     """
    #     pass

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

            # Check if text encoders have FP8 weights (requires autocast)
            has_fp8_weights = self._has_fp8_text_encoder()

            with context_manager:
                # For FP8 quantized text encoders, use autocast for mixed precision
                # This prevents "ufunc_add_CUDA not implemented for Float8_e4m3fn" errors
                if has_fp8_weights:
                    with torch.autocast(device_type='cuda', dtype=self.training_dtype):
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
                else:
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

            # Check if text encoder has FP8 weights (requires autocast)
            has_fp8_weights = self._has_fp8_text_encoder()

            with context_manager:
                # For FP8 quantized text encoder, use autocast for mixed precision
                if has_fp8_weights:
                    with torch.autocast(device_type='cuda', dtype=self.training_dtype):
                        text_embeddings = self.text_encoder(
                            text_inputs.input_ids.to(self.device),
                        )[0]
                else:
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
        # Check if text encoder has FP8 weights (requires autocast)
        has_fp8_weights = self._has_fp8_text_encoder()

        with torch.no_grad():
            # For FP8 quantized text encoder, use autocast for mixed precision
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=self.training_dtype):
                    encoder_output = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    prompt_embeds = encoder_output.hidden_states[-2]
            else:
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

    def encode_prompt_anima(self, prompt: str, qwen3_max_length: int = 512,
                             t5_max_length: int = 512):
        """Encode prompt for Anima using the Phase A/B inference pipeline.

        Returns the same dict-style auxiliary payload that the Anima DiT
        forward expects: prompt_embeds (Qwen3 hidden states), source_mask
        (Qwen3 attention mask), t5_input_ids, t5_attn_mask. Caching is
        handled upstream — this method always re-encodes.
        """
        from core.models.anima.anima_pipeline_ops import encode_prompt as _encode

        # Reuse the inference encode_prompt — it already handles Qwen3 hidden-state
        # extraction, T5 tokenisation for the LLM Adapter, and zero-masking.
        # Phase B.1-e added A1111-style emphasis support there which is
        # intentionally NOT applied during training (captions go through raw).
        encoded = _encode(
            text_encoder=self.text_encoder,
            qwen3_tokenizer=self.tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            prompt=prompt,
            device=str(self.device),
            dtype=self.training_dtype,
            qwen3_max_length=qwen3_max_length,
            t5_max_length=t5_max_length,
        )
        # encode_prompt returns batched tensors of shape [1, L, ...]; drop the
        # batch dim so caches accumulate per-sample. Detach for storage.
        return {
            "prompt_embeds": encoded["prompt_embeds"][0].detach(),
            "source_mask": encoded["source_mask"][0].detach(),
            "t5_input_ids": encoded["t5_input_ids"][0].detach(),
            "t5_attn_mask": encoded["t5_attn_mask"][0].detach(),
        }

    def encode_caption(self, caption: str, requires_grad: bool = False):
        """
        Unified caption encoding for all architectures.

        Returns:
            Tuple of (embeddings, auxiliary_data):
            - Z-Image: (prompt_embeds, attention_mask)
            - SD1.5: (text_embeddings, None)
            - SDXL: (text_embeddings, pooled_embeddings)
            - FLUX.2: (prompt_embeds, None) - text_ids computed in train_step
            - Anima: (prompt_embeds, anima_aux_dict) where aux dict has
              {source_mask, t5_input_ids, t5_attn_mask}
        """
        if self.is_zimage:
            return self.encode_prompt_zimage(caption)
        elif self.is_anima:
            payload = self.encode_prompt_anima(caption)
            # Return the Qwen3 hidden states as the primary embedding plus the
            # rest as a dict so callers can hand them to train_step_anima as
            # a single bundle.
            return payload["prompt_embeds"], {
                "source_mask": payload["source_mask"],
                "t5_input_ids": payload["t5_input_ids"],
                "t5_attn_mask": payload["t5_attn_mask"],
            }
        elif self.is_flux2:
            # FLUX.2: Use Qwen3 text encoder with hidden state extraction
            # Note: text_ids are generated dynamically in train_step_flux2, not cached
            prompt_embeds, _ = self._flux2_encode_prompt(caption)
            return prompt_embeds, None  # text_ids are computed in train_step
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
            # SD/SDXL: CLIPTextModel
            self.text_encoder.to(self.device)
            # Ensure embedding layer stays on GPU (critical for gradient checkpointing)
            if hasattr(self.text_encoder, 'text_model') and hasattr(self.text_encoder.text_model, 'embeddings'):
                self.text_encoder.text_model.embeddings.to(self.device)
        if self.is_sdxl and self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.device)
            # Ensure embedding layer stays on GPU (critical for gradient checkpointing)
            if hasattr(self.text_encoder_2, 'text_model') and hasattr(self.text_encoder_2.text_model, 'embeddings'):
                self.text_encoder_2.text_model.embeddings.to(self.device)

    def move_text_encoder_to_cpu(self):
        """Move Text Encoder(s) to CPU to free VRAM."""
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")
        if self.is_sdxl and self.text_encoder_2 is not None:
            self.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    def move_main_model_to_gpu(self):
        """Move main model (U-Net or Transformer) to GPU for training."""
        if self.is_zimage or self.is_anima:
            if self.transformer_original is not None:
                self.transformer_original.to(self.device)
        else:
            if self.unet is not None:
                self.unet.to(self.device)

    def move_main_model_to_cpu(self):
        """Move main model (U-Net or Transformer) to CPU to free VRAM."""
        if self.is_zimage or self.is_anima:
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
        # Match VAE dtype to prevent type mismatch errors
        image_tensor = image_tensor.to(device=vae_device, dtype=self.vae_dtype)

        # DEBUG: Log preprocessing
        debug_preprocessing = False  # Set to True to debug latent encoding
        if debug_preprocessing:
            print(f"[encode_image DEBUG] Image tensor before VAE:")
            print(f"  Shape: {image_tensor.shape}, dtype: {image_tensor.dtype}, device: {image_tensor.device}")
            print(f"  Mean: {image_tensor.mean():.6f}, Std: {image_tensor.std():.6f}")
            print(f"  Min: {image_tensor.min():.6f}, Max: {image_tensor.max():.6f}")

        # Encode to latents
        with torch.no_grad():
            if self.is_flux2:
                # FLUX.2 VAE encoding with BatchNorm normalization
                latent_dist = self.vae.encode(image_tensor).latent_dist
                latents = latent_dist.sample()

                # DEBUG: Log raw latents
                if debug_preprocessing:
                    print(f"[encode_image DEBUG] FLUX.2 raw latents:")
                    print(f"  Shape: {latents.shape}")
                    print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")

                # Patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)
                latents = self._flux2_patchify_latents_for_training(latents)

                # Apply BatchNorm normalization (like pipeline.py)
                latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                    latents.device, latents.dtype
                )
                latents = (latents - latents_bn_mean) / latents_bn_std

                # DEBUG: Log normalized latents
                if debug_preprocessing:
                    print(f"[encode_image DEBUG] FLUX.2 normalized latents:")
                    print(f"  Shape: {latents.shape}")
                    print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")

                del latent_dist

            elif self.is_zimage:
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
            elif self.is_anima:
                # Anima uses the Qwen-Image VAE (Wan VAE 2.1 latent space, 16ch).
                # Encode -> sample posterior -> apply latents_mean / latents_std
                # normalisation (same as anima_pipeline_ops.vae_encode_image).
                # AutoencoderKLQwenImage expects [B, C, T, H, W] (T=1 for images).
                image_tensor_5d = image_tensor.unsqueeze(2)
                latent_dist = self.vae.encode(image_tensor_5d).latent_dist
                latents_5d = latent_dist.sample()  # [B, 16, 1, H/8, W/8]
                from core.models.anima.anima_pipeline_ops import _get_qwen_vae_normalization
                mean_t, std_t = _get_qwen_vae_normalization(self.vae, latents_5d.device, latents_5d.dtype)
                latents_5d = (latents_5d - mean_t) / std_t
                # Drop the temporal dim for storage; train_step_anima re-adds it.
                latents = latents_5d.squeeze(2)
                del image_tensor_5d, latent_dist, latents_5d
            else:
                # SD/SDXL VAE - 統一された処理フロー
                from core.models.sdxl_vae_wrapper import SDXLVAEWrapper

                if isinstance(self.vae, SDXLVAEWrapper):
                    # SDXLVAEWrapperの場合、内部のAutoencoderKLにアクセス
                    vae_model = self.vae.vae
                else:
                    # 標準のAutoencoderKL
                    vae_model = self.vae

                # 統一されたエンコード処理
                encoder_output = vae_model.encode(image_tensor)
                latents = encoder_output.latent_dist.sample()

                # DEBUG: Log raw latents before scaling
                if debug_preprocessing:
                    print(f"[encode_image DEBUG] Raw latents (before scaling):")
                    print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")
                    print(f"  Min: {latents.min():.6f}, Max: {latents.max():.6f}")
                    print(f"  scaling_factor: {vae_model.config.scaling_factor}")

                latents = latents * vae_model.config.scaling_factor

                # DEBUG: Log scaled latents
                if debug_preprocessing:
                    print(f"[encode_image DEBUG] Scaled latents (after * scaling_factor):")
                    print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")
                    print(f"  Min: {latents.min():.6f}, Max: {latents.max():.6f}")

                # Clean up intermediate tensors
                del encoder_output

        # Clean up image_tensor before moving latents to CPU
        del image_tensor

        # Convert to training dtype and move to CPU immediately to free VRAM
        latents = latents.to(dtype=self.training_dtype, device='cpu')

        # DEBUG: Log final latents after dtype conversion
        if debug_preprocessing:
            print(f"[encode_image DEBUG] Final latents (after dtype={self.training_dtype}, device=cpu):")
            print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")
            print(f"  Min: {latents.min():.6f}, Max: {latents.max():.6f}")

        return latents

    # ============================================================
    # OOM Recovery: Batch Splitting
    # ============================================================

    def _forward_backward_with_oom_recovery(
        self,
        mnt_latents: torch.Tensor,
        mnt_text_embeddings: torch.Tensor,
        mnt_attention_mask: Optional[torch.Tensor],
        mnt_pooled_embeddings: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        debug_save_path: Optional[Path],
        batch_captions: Optional[List[str]],
        batch_reference_paths: Optional[List[Optional[str]]],
        alphas_cumprod_cached: Optional[torch.Tensor],
        use_condition_images: bool,
        condition_images_batch: Optional[torch.Tensor],
        reference_latents_nested: Optional[list],
        min_split_batch_size: int = 1,
    ) -> Tuple[float, float, float, bool]:
        """
        Execute forward + backward pass with OOM recovery via batch splitting.

        When OOM occurs, the batch is split in half and processed sequentially.
        Gradients are accumulated across splits, achieving the same result as
        processing the full batch (except for BatchNorm, which this model doesn't use).

        Args:
            mnt_latents: Latents for this MNT iteration [B, C, H, W]
            mnt_text_embeddings: Text embeddings [B, seq_len, dim]
            mnt_attention_mask: Attention mask (Z-Image only)
            mnt_pooled_embeddings: Pooled embeddings (SDXL only)
            timesteps: Timesteps for diffusion [B]
            debug_save_path: Path to save debug latents
            batch_captions: Captions for debug output
            alphas_cumprod_cached: Cached alphas_cumprod tensor
            use_condition_images: Whether ControlNet conditioning is used
            condition_images_batch: ControlNet condition images
            reference_latents_nested: Reference latents for FLUX.2
            min_split_batch_size: Minimum batch size (stop splitting below this)

        Returns:
            Tuple of (loss_value, pred_loss_value, recon_loss_value, cuda_error_skip) as Python floats
            cuda_error_skip is True if batch was skipped due to unrecoverable CUDA error
        """
        batch_size = mnt_latents.shape[0]

        try:
            # Attempt full batch forward + backward
            loss, pred_loss, recon_loss = self._execute_forward_backward(
                mnt_latents=mnt_latents,
                mnt_text_embeddings=mnt_text_embeddings,
                mnt_attention_mask=mnt_attention_mask,
                mnt_pooled_embeddings=mnt_pooled_embeddings,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                batch_captions=batch_captions,
                batch_reference_paths=batch_reference_paths,
                alphas_cumprod_cached=alphas_cumprod_cached,
                use_condition_images=use_condition_images,
                condition_images_batch=condition_images_batch,
                reference_latents_nested=reference_latents_nested,
            )
            return loss, pred_loss, recon_loss, False  # cuda_error_skip=False (success)

        except RuntimeError as e:
            error_str = str(e).lower()
            # Check for various CUDA memory-related errors
            is_recoverable_cuda_error = (
                "out of memory" in error_str or
                "cuda error" in error_str or
                "cublas" in error_str or
                "cudnn" in error_str or
                "cusparse" in error_str or
                "cufft" in error_str
            )
            if not is_recoverable_cuda_error:
                raise  # Re-raise non-CUDA errors

            # ============================================================
            # CUDA Error Recovery: Clean up VRAM before retry
            # ============================================================
            # Critical: Must release all tensors from failed forward/backward pass
            # before attempting batch split. Otherwise VRAM stays full.
            print(f"{self.log_prefix} [CUDA Recovery] Error detected, cleaning up VRAM...")

            # Step 1: Zero gradients to release gradient tensors from failed backward
            # This is critical - partial backward may have accumulated invalid gradients
            try:
                self.optimizer.zero_grad(set_to_none=True)
                print(f"{self.log_prefix} [CUDA Recovery] Gradients cleared (set_to_none=True)")
            except Exception as grad_error:
                print(f"{self.log_prefix} [CUDA Recovery] zero_grad() failed: {grad_error}")

            # Step 2: Clear gradient checkpointing saved activations if using layer offloading
            if hasattr(self, 'layer_offload_conductor') and self.layer_offload_conductor is not None:
                try:
                    self.layer_offload_conductor.clear_activations()
                    print(f"{self.log_prefix} [CUDA Recovery] Layer offload activations cleared")
                except Exception:
                    pass

            # Step 3: Clear FLUX.2 block swap activations if applicable
            if hasattr(self, 'flux2_block_offloader') and self.flux2_block_offloader is not None:
                try:
                    self.flux2_block_offloader.clear_activations()
                    print(f"{self.log_prefix} [CUDA Recovery] FLUX.2 block swap activations cleared")
                except Exception:
                    pass

            # Step 4: Force Python garbage collection to release orphaned tensors
            gc.collect()

            # Step 5: Synchronize CUDA to ensure all pending operations complete/fail
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # May fail if CUDA is in bad state, that's okay

            # Step 6: Clear CUDA cache (releases unreferenced GPU memory)
            empty_cache_failed = False
            try:
                torch.cuda.empty_cache()
                print(f"{self.log_prefix} [CUDA Recovery] CUDA cache cleared")
            except Exception as cache_error:
                print(f"{self.log_prefix} [CUDA Recovery] empty_cache() failed: {cache_error}")
                empty_cache_failed = True
                # If empty_cache itself fails, CUDA context is severely corrupted
                # Try to reset CUDA context by forcing synchronization
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                # Try ipc_collect as last resort
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            # Step 7: Reset CUDA error state by attempting a small allocation
            try:
                _test = torch.zeros(1, device=self.device)
                del _test
                print(f"{self.log_prefix} [CUDA Recovery] CUDA state verified OK")
            except Exception as cuda_state_error:
                print(f"{self.log_prefix} [CUDA Recovery] CUDA still in bad state: {cuda_state_error}")
                # If CUDA is still broken after empty_cache failed, this is unrecoverable
                # Signal that emergency checkpoint should be saved and process should restart
                if empty_cache_failed:
                    print(f"{self.log_prefix} [CUDA Recovery] CUDA context is severely corrupted (empty_cache failed)")
                    print(f"{self.log_prefix} [CUDA Recovery] Raising exception to trigger emergency checkpoint save")
                    raise RuntimeError(f"CUDA context unrecoverable: empty_cache() failed. Original error: {str(e)[:200]}")
                # If CUDA is still broken, we may need to skip this batch
                if batch_size <= min_split_batch_size:
                    print(f"{self.log_prefix} [CUDA Recovery] Cannot recover, SKIPPING BATCH")
                    return 0.0, 0.0, 0.0, True  # cuda_error_skip=True

            # CUDA error occurred - attempt batch splitting
            if batch_size <= min_split_batch_size:
                # Cannot split further - SKIP this batch instead of crashing
                print(f"{self.log_prefix} [CUDA Error] Cannot split further (batch_size={batch_size}), SKIPPING BATCH")
                print(f"{self.log_prefix} [CUDA Error] Original error: {str(e)[:200]}")
                # Return zero loss - this batch contributes nothing but training continues
                return 0.0, 0.0, 0.0, True  # cuda_error_skip=True

            split_size = batch_size // 2
            print(f"{self.log_prefix} [CUDA Recovery] Splitting batch {batch_size} -> {split_size} + {batch_size - split_size} (error: {str(e)[:100]})")

            # Process first half with error handling
            # If sub-batch fails, skip it and continue with the other half
            try:
                loss1, pred1, recon1, skip1 = self._forward_backward_with_oom_recovery(
                    mnt_latents=mnt_latents[:split_size],
                    mnt_text_embeddings=mnt_text_embeddings[:split_size],
                    mnt_attention_mask=mnt_attention_mask[:split_size] if mnt_attention_mask is not None else None,
                    mnt_pooled_embeddings=mnt_pooled_embeddings[:split_size] if mnt_pooled_embeddings is not None else None,
                    timesteps=timesteps[:split_size],
                    debug_save_path=debug_save_path,
                    batch_captions=batch_captions[:split_size] if batch_captions else None,
                    batch_reference_paths=batch_reference_paths[:split_size] if batch_reference_paths else None,
                    alphas_cumprod_cached=alphas_cumprod_cached,
                    use_condition_images=use_condition_images,
                    condition_images_batch=condition_images_batch[:split_size] if condition_images_batch is not None else None,
                    reference_latents_nested=reference_latents_nested[:split_size] if reference_latents_nested is not None else None,
                    min_split_batch_size=min_split_batch_size,
                )
                first_half_success = not skip1  # skip1=True means this half was skipped
            except Exception as split1_error:
                print(f"{self.log_prefix} [CUDA Recovery] First half failed: {str(split1_error)[:100]}")
                loss1, pred1, recon1, skip1 = 0.0, 0.0, 0.0, True
                first_half_success = False
                # Clean up after failure
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Process second half with error handling
            try:
                loss2, pred2, recon2, skip2 = self._forward_backward_with_oom_recovery(
                    mnt_latents=mnt_latents[split_size:],
                    mnt_text_embeddings=mnt_text_embeddings[split_size:],
                    mnt_attention_mask=mnt_attention_mask[split_size:] if mnt_attention_mask is not None else None,
                    mnt_pooled_embeddings=mnt_pooled_embeddings[split_size:] if mnt_pooled_embeddings is not None else None,
                    timesteps=timesteps[split_size:],
                    debug_save_path=None,  # Only save debug from first split
                    batch_captions=batch_captions[split_size:] if batch_captions else None,
                    batch_reference_paths=batch_reference_paths[split_size:] if batch_reference_paths else None,
                    alphas_cumprod_cached=alphas_cumprod_cached,
                    use_condition_images=use_condition_images,
                    condition_images_batch=condition_images_batch[split_size:] if condition_images_batch is not None else None,
                    reference_latents_nested=reference_latents_nested[split_size:] if reference_latents_nested is not None else None,
                    min_split_batch_size=min_split_batch_size,
                )
                second_half_success = not skip2  # skip2=True means this half was skipped
            except Exception as split2_error:
                print(f"{self.log_prefix} [CUDA Recovery] Second half failed: {str(split2_error)[:100]}")
                loss2, pred2, recon2, skip2 = 0.0, 0.0, 0.0, True
                second_half_success = False
                # Clean up after failure
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # If both halves failed, return zero (batch skipped)
            if not first_half_success and not second_half_success:
                print(f"{self.log_prefix} [CUDA Recovery] Both halves failed, SKIPPING BATCH")
                return 0.0, 0.0, 0.0, True  # cuda_error_skip=True

            # Average losses (weighted by split sizes for correctness)
            # Only count successful halves in the average
            if first_half_success and second_half_success:
                w1, w2 = split_size, batch_size - split_size
                total = w1 + w2
                avg_loss = (loss1 * w1 + loss2 * w2) / total
                avg_pred = (pred1 * w1 + pred2 * w2) / total
                avg_recon = (recon1 * w1 + recon2 * w2) / total
            elif first_half_success:
                # Only first half succeeded
                avg_loss, avg_pred, avg_recon = loss1, pred1, recon1
            else:
                # Only second half succeeded
                avg_loss, avg_pred, avg_recon = loss2, pred2, recon2

            # At least one half succeeded, so we have valid gradients
            return avg_loss, avg_pred, avg_recon, False  # cuda_error_skip=False

    def _execute_forward_backward(
        self,
        mnt_latents: torch.Tensor,
        mnt_text_embeddings: torch.Tensor,
        mnt_attention_mask: Optional[torch.Tensor],
        mnt_pooled_embeddings: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        debug_save_path: Optional[Path],
        batch_captions: Optional[List[str]],
        batch_reference_paths: Optional[List[Optional[str]]],
        alphas_cumprod_cached: Optional[torch.Tensor],
        use_condition_images: bool,
        condition_images_batch: Optional[torch.Tensor],
        reference_latents_nested: Optional[list],
    ) -> Tuple[float, float, float]:
        """
        Execute forward pass (train_step_xxx) and backward pass for a batch.

        Returns loss values as Python floats (not tensors).
        Gradients are accumulated in model parameters.
        """
        # Forward pass (architecture-specific)
        if self.is_zimage:
            loss, pred_loss, recon_loss = self.train_step_zimage(
                latents=mnt_latents,
                prompt_embeds=mnt_text_embeddings,
                attention_mask=mnt_attention_mask,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
        elif self.is_anima:
            # Anima carries the LLM-Adapter side payload (source_mask, t5 ids)
            # in mnt_attention_mask, which here holds a dict produced by
            # encode_caption() rather than a single tensor.
            anima_aux = mnt_attention_mask if isinstance(mnt_attention_mask, dict) else {}
            loss, pred_loss, recon_loss = self.train_step_anima(
                latents=mnt_latents,
                prompt_embeds=mnt_text_embeddings,
                anima_aux=anima_aux,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
        elif self.is_flux2:
            # FLUX.2 training with position IDs
            img_ids = self._flux2_prepare_latent_ids(mnt_latents).to(self.device)
            packed_latents = self._flux2_pack_latents(mnt_latents)
            txt_ids = self._flux2_prepare_text_ids(mnt_text_embeddings).to(self.device)

            # Prepare reference latents
            mnt_reference_latents_nested = None
            if reference_latents_nested is not None:
                mnt_reference_latents_nested = [
                    [lat.detach() for lat in item_lats]
                    for item_lats in reference_latents_nested
                ]

            loss, pred_loss, recon_loss = self.train_step_flux2(
                latents=packed_latents,
                prompt_embeds=mnt_text_embeddings,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=None,
                reference_latents_nested=mnt_reference_latents_nested,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
        elif use_condition_images and condition_images_batch is not None:
            # ControlNet training
            mnt_condition_images = condition_images_batch.detach()
            loss, pred_loss, recon_loss = self.train_step_controlnet(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                condition_images=mnt_condition_images,
                pooled_embeddings=mnt_pooled_embeddings,
                timesteps=timesteps,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
        else:
            # SD1.5/SDXL
            loss, pred_loss, recon_loss = self.train_step(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                pooled_embeddings=mnt_pooled_embeddings,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )

        # Backward pass
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # Extract values before deleting tensors
        loss_value = loss.item()
        pred_loss_value = pred_loss.item() if isinstance(pred_loss, torch.Tensor) else pred_loss
        recon_loss_value = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss

        # Free computation graph
        del loss, pred_loss, recon_loss

        return loss_value, pred_loss_value, recon_loss_value

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
        debug_reference_image_paths: Optional[List[str]] = None,
        profile_vram: bool = False,
        alphas_cumprod_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform single training step (SD1.5/SDXL).

        Args:
            latents: Image latents [B, C, H, W]
            text_embeddings: Text prompt embeddings
            pooled_embeddings: Pooled text embeddings (SDXL only)
            timesteps: Optional timesteps tensor. If None, sample uniformly from [0, num_train_timesteps)
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output
            profile_vram: If True, print VRAM usage
            alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (for SNR weight computation)

        Returns:
            (loss_tensor, loss_value) - Loss tensor with grad and scalar value
        """
        if profile_vram:
            print_vram_usage("[train_step] Start")

        # Move latents to GPU with correct dtype
        # Latents come from cache (CPU, training_dtype) and must be moved to GPU before training
        latents = latents.to(device=self.device, dtype=self.training_dtype, non_blocking=True)

        # Sample noise (now on GPU)
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
                    # IMPORTANT: DDPM convention is REVERSED from Flow Matching
                    # DDPM: t=999 (noisy) → t=0 (clean)
                    # Flow: t=0 (noisy) → t=1 (clean)
                    # So we need to flip: YAML [0,1] → DDPM [999,0]
                    # Example: YAML min=0, max=0.2 (want noisy) → DDPM [999, 800] (noisy)
                    timesteps_continuous = self.timestep_sampler.sample(batch_size, self.device)
                    timesteps = ((1.0 - timesteps_continuous) * self.noise_scheduler.config.num_train_timesteps).long()
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
                # IMPORTANT: DDPM convention is REVERSED from Flow Matching
                # DDPM: t=999 (noisy) → t=0 (clean)
                # Flow: t=0 (noisy) → t=1 (clean)
                # So we need to flip: YAML [0,1] → DDPM [999,0]
                timesteps = ((1.0 - timesteps) * self.noise_scheduler.config.num_train_timesteps).long()
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

        # DEUS debug check removed (architecture no longer maintained)

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
        # TEMPORARY: .float() is redundant since everything is FP32, but kept for safety
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Apply Min-SNR gamma weighting (only for epsilon prediction)
        # Min-SNR was designed for epsilon prediction; applying it to v-prediction is theoretically unsound
        # When dual loss is enabled (reconstruction_loss_weight > 0), also return weights
        # to compensate for lost prediction weight by boosting reconstruction weight
        min_snr_weights = None
        if self.min_snr_gamma > 0 and prediction_target == "epsilon":
            if self.reconstruction_loss_weight > 0:
                # Return weights for dual loss compensation
                loss_per_sample_weighted, min_snr_weights = apply_snr_weight(
                    loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma,
                    return_weights=True, alphas_cumprod_cached=alphas_cumprod_cached
                )
            else:
                loss_per_sample_weighted = apply_snr_weight(
                    loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma,
                    alphas_cumprod_cached=alphas_cumprod_cached
                )
        else:
            loss_per_sample_weighted = loss_per_sample

        mse_loss = loss_per_sample_weighted.mean()

        # Add SNR and/or Energy regularization if enabled (can use both simultaneously)
        regularization_loss = torch.tensor(0.0, device=self.device)

        # Compute predicted latent once (used by both regularization losses and debug save)
        predicted_latent_for_reg = None
        predicted_latent_for_recon = None  # Will be set in reconstruction loss path
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

        # Calculate reconstruction loss (for monitoring or dual loss training)
        # If reconstruction_loss_weight > 0, compute with gradients for backprop
        # Otherwise, compute without gradients (monitoring only)
        if self.reconstruction_loss_weight > 0:
            # Dual loss training: compute reconstruction loss with gradients
            # Reuse predicted_latent_for_reg if already computed (has gradients)
            if predicted_latent_for_reg is not None:
                predicted_latent_for_recon = predicted_latent_for_reg
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

            # Dual loss with min-SNR weight compensation
            # When min_snr_gamma > 0, the prediction loss is reduced by min_snr_weights for clean timesteps.
            # We compensate for this "lost" weight by boosting the reconstruction loss weight.
            #
            # Original dual loss: alpha * pred_loss + beta * recon_loss (alpha + beta = 1.0)
            # With min-SNR: pred_loss is already weighted by min_snr_weights
            #
            # Compensation formula (per-sample):
            #   lost_weight = (1 - min_snr_weight) * alpha  (weight originally for pred_loss that was reduced)
            #   effective_beta = beta + lost_weight        (boost recon_loss by lost amount)
            #   combined_loss = pred_loss_weighted + effective_beta * recon_loss
            #
            # Note: pred_loss already has min_snr_weight applied, so we use it directly without alpha multiplier

            alpha = 1.0 - self.reconstruction_loss_weight
            beta = self.reconstruction_loss_weight

            if min_snr_weights is not None:
                # Per-sample compensation: boost recon_loss weight based on how much pred_loss was reduced
                # lost_weight[i] = (1 - min_snr_weights[i]) * alpha
                # effective_beta[i] = beta + lost_weight[i]
                lost_weight = (1.0 - min_snr_weights) * alpha  # [batch_size]
                effective_beta = beta + lost_weight  # [batch_size]

                # Per-sample combined loss
                # loss_per_sample_weighted already has min_snr weighting applied
                combined_loss_per_sample = loss_per_sample_weighted + effective_beta * recon_loss_per_sample
                combined_loss = combined_loss_per_sample.mean()
            else:
                # No min-SNR: standard dual loss
                recon_loss = recon_loss_per_sample.mean()
                combined_loss = alpha * mse_loss + beta * recon_loss

            # For return value
            recon_loss = recon_loss_per_sample.mean()

            # Total loss with regularization
            loss = combined_loss + regularization_loss
        else:
            # Standard training: prediction loss only
            # Calculate reconstruction loss for monitoring (no gradients)
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

            # Total loss (prediction loss + regularization)
            loss = mse_loss + regularization_loss

        if profile_vram:
            print_vram_usage("[train_step] After loss calculation")

        # Debug save if requested
        if debug_save_path is not None:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            timestep_value = timesteps[0].item()

            # Reuse predicted_latent from reconstruction loss calculation if available
            # This avoids redundant computation (predict_original_latent_unified is expensive)
            if predicted_latent_for_recon is not None:
                predicted_latent_for_debug = predicted_latent_for_recon.detach()
            elif predicted_latent_for_reg is not None:
                predicted_latent_for_debug = predicted_latent_for_reg.detach()
            else:
                # Fallback: compute predicted_latent if not available
                with torch.no_grad():
                    predicted_latent_for_debug = predict_original_latent_unified(
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
                'predicted_latent': predicted_latent_for_debug[0:1].detach().cpu(),
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

            if debug_reference_image_paths is not None and len(debug_reference_image_paths) > 0:
                first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
                if first_ref:
                    debug_data['reference_image_path'] = first_ref

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:04d}.pt")
            del predicted_latent_for_debug

        # Return loss tensor (with gradient), pred_loss value, and recon_loss value
        # IMPORTANT: Do NOT call .item() on loss here - it breaks the computation graph!
        # The training loop will call .backward() on the loss tensor.
        pred_loss_value = mse_loss.item()
        recon_loss_value = recon_loss.item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        # But keep 'loss' tensor for backward pass
        del noise, noisy_latents, model_pred, target, recon_loss
        if self.is_sdxl and added_cond_kwargs is not None:
            del added_cond_kwargs

        return loss, pred_loss_value, recon_loss_value

    def train_step_controlnet(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        condition_images: torch.Tensor,
        pooled_embeddings: torch.Tensor = None,
        timesteps: Optional[torch.Tensor] = None,
        profile_vram: bool = False,
        alphas_cumprod_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Perform single ControlNet training step (SD1.5/SDXL).

        Standard ControlNet:
        1. ControlNet forward: condition_images + noisy_latents -> residuals
        2. UNet forward with residuals injected -> model_pred
        3. Loss = MSE(model_pred, target)

        UNet is frozen but runs with gradients enabled so that gradient
        flows back through the residual additions to the ControlNet.

        Args:
            latents: Image latents [B, C, H, W]
            text_embeddings: Text prompt embeddings
            condition_images: Condition image tensor [B, 3, H, W] in [0, 1] range
            pooled_embeddings: Pooled text embeddings (SDXL only)
            timesteps: Optional timesteps tensor
            profile_vram: If True, print VRAM usage
            alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU

        Returns:
            (loss_tensor, pred_loss_value, recon_loss_value)
        """
        if profile_vram:
            print_vram_usage("[train_step_controlnet] Start")

        # Move tensors to GPU
        latents = latents.to(device=self.device, dtype=self.training_dtype, non_blocking=True)
        condition_images = condition_images.to(device=self.device, dtype=self.training_dtype, non_blocking=True)

        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample timesteps (DDPM)
        noise_process = getattr(self, 'noise_process', 'ddpm')

        if timesteps is None:
            if noise_process == "ddpm":
                if self.timestep_sampler is not None:
                    timesteps_continuous = self.timestep_sampler.sample(batch_size, self.device)
                    timesteps = ((1.0 - timesteps_continuous) * self.noise_scheduler.config.num_train_timesteps).long()
                    timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
                else:
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (batch_size,), device=self.device,
                    ).long()
            elif noise_process == "flow":
                if self.timestep_sampler is not None:
                    timesteps = self.timestep_sampler.sample(batch_size, self.device)
                else:
                    timesteps = torch.rand((batch_size,), device=self.device)
        else:
            if noise_process == "ddpm":
                timesteps = ((1.0 - timesteps) * self.noise_scheduler.config.num_train_timesteps).long()
                timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)

        # Add noise to latents
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

            add_time_ids = torch.tensor([[
                image_height, image_width, 0, 0, image_height, image_width
            ]], dtype=pooled_embeddings.dtype, device=self.device)
            add_time_ids = add_time_ids.repeat(batch_size, 1)

            added_cond_kwargs = {
                "text_embeds": pooled_embeddings,
                "time_ids": add_time_ids,
            }

        if profile_vram:
            print_vram_usage("[train_step_controlnet] Before ControlNet forward")

        # Enable gradients for gradient checkpointing (ControlNet needs grad flow)
        noisy_latents.requires_grad_(True)
        text_embeddings.requires_grad_(True)
        if pooled_embeddings is not None:
            pooled_embeddings.requires_grad_(True)

        # ControlNet forward pass (trainable)
        # Get adapter from ControlNetTrainer
        controlnet_adapter = self.adapter
        controlnet_module = self.controlnet
        is_lllite = getattr(self, 'controlnet_type', 'standard') == 'lllite'

        if is_lllite:
            # LLLite mode: apply patches to UNet attention layers before forward
            controlnet_module.apply_patches(self.unet, condition_images)
            controlnet_output = None
        else:
            # Standard ControlNet: get residuals from ControlNet forward
            if self.mixed_precision:
                with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                    controlnet_output = controlnet_adapter.controlnet_forward(
                        controlnet=controlnet_module,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        text_embeddings=text_embeddings,
                        condition_images=condition_images,
                        added_cond_kwargs=added_cond_kwargs,
                    )
            else:
                controlnet_output = controlnet_adapter.controlnet_forward(
                    controlnet=controlnet_module,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    text_embeddings=text_embeddings,
                    condition_images=condition_images,
                    added_cond_kwargs=added_cond_kwargs,
                )

        if profile_vram:
            print_vram_usage("[train_step_controlnet] After ControlNet forward")

        # UNet forward pass
        try:
            if controlnet_output is not None:
                # Standard ControlNet: inject residuals into UNet
                down_block_res_samples, mid_block_res_sample = controlnet_output

                # UNet is frozen but we need gradients to flow through residual additions
                if self.mixed_precision:
                    with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                        if self.is_sdxl and added_cond_kwargs is not None:
                            model_pred = self.unet(
                                noisy_latents,
                                timesteps,
                                text_embeddings,
                                added_cond_kwargs=added_cond_kwargs,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            ).sample
                        else:
                            model_pred = self.unet(
                                noisy_latents,
                                timesteps,
                                text_embeddings,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            ).sample
                else:
                    if self.is_sdxl and added_cond_kwargs is not None:
                        model_pred = self.unet(
                            noisy_latents,
                            timesteps,
                            text_embeddings,
                            added_cond_kwargs=added_cond_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample
                    else:
                        model_pred = self.unet(
                            noisy_latents,
                            timesteps,
                            text_embeddings,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample
            else:
                # LLLite mode: patches already applied, normal UNet forward
                if self.mixed_precision:
                    with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                        if self.is_sdxl and added_cond_kwargs is not None:
                            model_pred = self.unet(
                                noisy_latents, timesteps, text_embeddings,
                                added_cond_kwargs=added_cond_kwargs,
                            ).sample
                        else:
                            model_pred = self.unet(
                                noisy_latents, timesteps, text_embeddings,
                            ).sample
                else:
                    if self.is_sdxl and added_cond_kwargs is not None:
                        model_pred = self.unet(
                            noisy_latents, timesteps, text_embeddings,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                    else:
                        model_pred = self.unet(
                            noisy_latents, timesteps, text_embeddings,
                        ).sample
        finally:
            # Remove LLLite patches after UNet forward (must always run)
            if is_lllite:
                controlnet_module.remove_patches(self.unet)

        if profile_vram:
            print_vram_usage("[train_step_controlnet] After UNet forward")

        # Get prediction target
        prediction_target = getattr(self, 'prediction_target', 'epsilon')
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
        if self.min_snr_gamma > 0 and prediction_target == "epsilon":
            loss_per_sample_weighted = apply_snr_weight(
                loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma,
                alphas_cumprod_cached=alphas_cumprod_cached
            )
        else:
            loss_per_sample_weighted = loss_per_sample

        mse_loss = loss_per_sample_weighted.mean()
        loss = mse_loss

        # Reconstruction loss (monitoring only, no gradients for ControlNet training)
        with torch.no_grad():
            predicted_latent = predict_original_latent_unified(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noise_scheduler=self.noise_scheduler,
                noisy_latents=noisy_latents,
                model_pred=model_pred,
                timesteps=timesteps,
            )
            recon_loss_per_element = F.mse_loss(predicted_latent.float(), latents.float(), reduction="none")
            recon_loss = recon_loss_per_element.mean()

        if profile_vram:
            print_vram_usage("[train_step_controlnet] After loss calculation")

        pred_loss_value = mse_loss.item()
        recon_loss_value = recon_loss.item()

        # Cleanup
        del noise, noisy_latents, model_pred, target, recon_loss, predicted_latent
        if controlnet_output is not None:
            del down_block_res_samples, mid_block_res_sample
        if added_cond_kwargs is not None:
            del added_cond_kwargs

        return loss, pred_loss_value, recon_loss_value

    def train_step_zimage(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        debug_reference_image_paths: Optional[List[str]] = None,
        profile_vram: bool = False,
        alphas_cumprod_cached: Optional[torch.Tensor] = None,
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
            alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (unused for Z-Image, included for API consistency)

        Returns:
            Tuple of (loss tensor, reconstruction loss value)
        """
        if profile_vram:
            print_vram_usage("[train_step_zimage] Start")

        # Z-Image uses Flow Matching with velocity prediction
        noise_process = getattr(self, 'noise_process', 'flow')  # Z-Image default: flow
        prediction_target = getattr(self, 'prediction_target', 'velocity')  # Z-Image default: velocity

        # Move latents to GPU with correct dtype
        # Latents come from cache (CPU, training_dtype) and must be moved to GPU before training
        latents = latents.to(device=self.device, dtype=self.training_dtype, non_blocking=True)

        # Sample random timesteps from [0, 1] if not provided
        batch_size = latents.shape[0]
        if timesteps is None:
            if self.timestep_sampler is not None:
                # Use timestep sampler (returns [0, 1] for flow matching)
                timesteps = self.timestep_sampler.sample(batch_size, self.device)
            else:
                # Legacy behavior: uniform sampling from [0, 1]
                timesteps = torch.rand(batch_size, device=self.device)

        # Sample noise (standard normal distribution, now on GPU)
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

        # Z-Image uses INVERTED velocity convention: v = latents - noise
        # This is opposite from standard Flow Matching (v = noise - latents)
        # diffusers Z-Image pipeline inverts the sign during inference: noise_pred = -model_output
        # So we train with target = latents - noise to match this convention
        target = latents - noise

        # Calculate MSE loss (always in fp32)
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Flow Matching doesn't use Min-SNR weighting (uniform timestep distribution)
        mse_loss = loss_per_sample.mean()

        # Add SNR and/or Energy regularization if enabled (can use both simultaneously)
        regularization_loss = torch.tensor(0.0, device=self.device)

        # Compute predicted latent once (used by regularization losses and dual loss)
        # Z-Image inverse velocity: v = latents - noise, so x_0 = x_t + t * v
        predicted_latent_for_reg = None
        if self.snr_regularization_loss is not None or self.energy_regularization_loss is not None or self.reconstruction_loss_weight > 0:
            # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
            t = timesteps.float()
            while t.dim() < noisy_latents.dim():
                t = t.unsqueeze(-1)
            predicted_latent_for_reg = noisy_latents + t * model_pred

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

        # Calculate reconstruction loss (for monitoring or dual loss training)
        # If reconstruction_loss_weight > 0, compute with gradients for backprop
        # Otherwise, compute without gradients (monitoring only)
        if self.reconstruction_loss_weight > 0:
            # Dual loss training: compute reconstruction loss with gradients
            # Reuse predicted_latent_for_reg if already computed (has gradients)
            if predicted_latent_for_reg is not None:
                predicted_latent_for_recon = predicted_latent_for_reg
            else:
                # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
                t = timesteps.float()
                while t.dim() < noisy_latents.dim():
                    t = t.unsqueeze(-1)
                predicted_latent_for_recon = noisy_latents + t * model_pred

            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
            recon_loss = recon_loss_per_sample.mean()

            # Normalized dual loss: alpha * pred_loss + beta * recon_loss (alpha + beta = 1.0)
            alpha = 1.0 - self.reconstruction_loss_weight
            beta = self.reconstruction_loss_weight
            combined_loss = alpha * mse_loss + beta * recon_loss

            # Total loss with regularization
            loss = combined_loss + regularization_loss
        else:
            # Standard training: prediction loss only
            # Calculate reconstruction loss for monitoring (no gradients)
            with torch.no_grad():
                # Reuse predicted_latent_for_reg if already computed, otherwise compute it
                if predicted_latent_for_reg is not None:
                    predicted_latent_for_recon = predicted_latent_for_reg.detach()
                else:
                    # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
                    t = timesteps.float()
                    while t.dim() < noisy_latents.dim():
                        t = t.unsqueeze(-1)
                    predicted_latent_for_recon = noisy_latents + t * model_pred

                recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
                recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
                recon_loss = recon_loss_per_sample.mean()

            # Total loss (prediction loss + regularization)
            loss = mse_loss + regularization_loss

        if profile_vram:
            print_vram_usage("[train_step_zimage] After loss calculation")

        # Debug save if requested
        if debug_save_path is not None:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            timestep_value = timesteps[0].item()

            with torch.no_grad():
                # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
                t = timesteps.float()
                while t.dim() < noisy_latents.dim():
                    t = t.unsqueeze(-1)
                predicted_latent = noisy_latents + t * model_pred

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

            if debug_reference_image_paths is not None and len(debug_reference_image_paths) > 0:
                first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
                if first_ref:
                    debug_data['reference_image_path'] = first_ref

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:.4f}.pt")
            del predicted_latent

        # Return loss tensor (with gradient), pred_loss value, and recon_loss value
        # IMPORTANT: Do NOT call .item() on loss here - it breaks the computation graph!
        # The training loop will call .backward() on the loss tensor.
        pred_loss_value = mse_loss.item()
        recon_loss_value = recon_loss.item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        # But keep 'loss' tensor for backward pass
        del noise, noisy_latents, noisy_latents_4d, model_pred, target
        del loss_per_element, loss_per_sample, recon_loss_per_element, recon_loss_per_sample, recon_loss

        return loss, pred_loss_value, recon_loss_value

    def train_step_anima(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        anima_aux: Dict[str, torch.Tensor],
        timesteps: Optional[torch.Tensor] = None,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        debug_reference_image_paths: Optional[List[str]] = None,
        profile_vram: bool = False,
        alphas_cumprod_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """Single Anima training step (rectified flow / velocity prediction).

        Args:
            latents:   Normalised Qwen-Image latents [B, 16, H/8, W/8] (the
                       Anima DiT requires a singleton temporal dim, added below).
            prompt_embeds: Qwen3 hidden states [B, L_qwen, 1024], zero-masked.
            anima_aux: dict with {source_mask, t5_input_ids, t5_attn_mask}
                       as produced by encode_prompt_anima.
            timesteps: Optional pre-sampled sigma values in [0, 1]; otherwise
                       sampled via self.timestep_sampler or uniform random.

        Returns:
            (loss tensor, prediction loss value, reconstruction loss value)
        """
        if profile_vram:
            print_vram_usage("[train_step_anima] Start")

        latents = latents.to(device=self.device, dtype=self.training_dtype, non_blocking=True)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.training_dtype, non_blocking=True)
        source_mask = anima_aux["source_mask"].to(device=self.device, non_blocking=True)
        t5_input_ids = anima_aux["t5_input_ids"].to(device=self.device, non_blocking=True)
        t5_attn_mask = anima_aux["t5_attn_mask"].to(device=self.device, non_blocking=True)

        batch_size = latents.shape[0]
        if timesteps is None:
            if self.timestep_sampler is not None:
                timesteps = self.timestep_sampler.sample(batch_size, self.device)
            else:
                timesteps = torch.rand(batch_size, device=self.device)
        timesteps = timesteps.to(self.training_dtype)

        noise = torch.randn_like(latents)

        # Flow-matching forward: x_t = (1 - sigma) * x_0 + sigma * noise
        sigma_view = timesteps.view(-1, *([1] * (latents.dim() - 1))).to(latents.dtype)
        noisy_latents = (1.0 - sigma_view) * latents + sigma_view * noise

        # Anima DiT requires a singleton temporal dim: [B, C, H, W] -> [B, C, 1, H, W].
        noisy_latents_5d = noisy_latents.unsqueeze(2)

        # Padding mask matches latent spatial resolution; all-valid (zeros).
        latent_h = noisy_latents.shape[-2]
        latent_w = noisy_latents.shape[-1]
        padding_mask = torch.zeros(
            (batch_size, 1, latent_h, latent_w),
            device=self.device, dtype=self.training_dtype,
        )

        if profile_vram:
            print_vram_usage("[train_step_anima] Before DiT forward")

        # The DiT forward returns velocity in 5D ([B, 16, 1, H, W]).
        if self.mixed_precision:
            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                model_pred = self.transformer(
                    x=noisy_latents_5d,
                    timesteps=timesteps,
                    context=prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=source_mask,
                )
        else:
            model_pred = self.transformer(
                x=noisy_latents_5d,
                timesteps=timesteps,
                context=prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=source_mask,
            )

        # Drop the temporal dim back: [B, 16, 1, H, W] -> [B, 16, H, W].
        if model_pred.dim() == 5:
            model_pred = model_pred.squeeze(2)

        if profile_vram:
            print_vram_usage("[train_step_anima] After DiT forward")

        # Rectified flow target: v = noise - x_0  (sd-scripts anima convention,
        # matches our inference scheduler which integrates `latents + dt * v`
        # with dt = sigma_next - sigma < 0).
        target = noise - latents

        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_element.mean([1, 2, 3])
        mse_loss = loss_per_sample.mean()

        loss = mse_loss

        # Reconstruction loss (predicted x0 vs ground-truth x0) — optional.
        recon_loss_value = 0.0
        if self.reconstruction_loss_weight > 0:
            with torch.no_grad():
                pred_x0 = noisy_latents - sigma_view * model_pred  # x_0 = x_t - sigma * v
                recon_loss = F.mse_loss(pred_x0.float(), latents.float())
                recon_loss_value = recon_loss.item()
            loss = loss + self.reconstruction_loss_weight * recon_loss

        pred_loss_value = mse_loss.item()

        del noise, noisy_latents, noisy_latents_5d, model_pred, target
        del loss_per_element, loss_per_sample
        return loss, pred_loss_value, recon_loss_value

    def train_step_flux2(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        reference_latents_nested: Optional[List[List[torch.Tensor]]] = None,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        debug_reference_image_paths: Optional[List[str]] = None,
        profile_vram: bool = False,
        alphas_cumprod_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Perform single training step (FLUX.2 Klein).

        Args:
            latents: Image latents [B, seq_len, C] (already patchified)
            prompt_embeds: Prompt embeddings [B, text_seq_len, dim]
            img_ids: Image position IDs [B, seq_len, 4] for RoPE
            txt_ids: Text position IDs [B, text_seq_len, 4] for RoPE
            timesteps: Timesteps for this batch [B]. If None, sampled uniformly from [0, 1]
            guidance: Guidance values [B]. If None, uses default 3.5
            reference_latents_nested: Reference image latents for conditioning.
                List of length B, where each element is a list of latents [C, H, W] for that batch item.
                Example: [[ref1_lat, ref2_lat], [ref1_lat], [ref1_lat, ref2_lat, ref3_lat]]
                T coordinates applied: 10, 20, 30... per reference image within each batch item.
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output
            profile_vram: If True, print VRAM usage
            alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (unused for FLUX.2, included for API consistency)

        Returns:
            Tuple of (loss tensor, prediction loss value, reconstruction loss value)
        """
        if profile_vram:
            print_vram_usage("[train_step_flux2] Start")

        # FLUX.2 uses Flow Matching with velocity prediction
        noise_process = getattr(self, 'noise_process', 'flow')  # FLUX.2 default: flow
        prediction_target = getattr(self, 'prediction_target', 'velocity')  # FLUX.2 default: velocity

        # Move latents to GPU with correct dtype
        latents = latents.to(device=self.device, dtype=self.training_dtype, non_blocking=True)
        img_ids = img_ids.to(device=self.device, non_blocking=True)
        txt_ids = txt_ids.to(device=self.device, non_blocking=True)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.training_dtype, non_blocking=True)

        # Sample random timesteps from [0, 1] if not provided
        batch_size = latents.shape[0]
        if timesteps is None:
            if self.timestep_sampler is not None:
                timesteps = self.timestep_sampler.sample(batch_size, self.device)
            else:
                timesteps = torch.rand(batch_size, device=self.device)

        # Set default guidance if not provided
        if guidance is None:
            guidance = torch.full((batch_size,), 3.5, device=self.device, dtype=self.training_dtype)

        # Sample noise (standard normal distribution)
        noise = torch.randn_like(latents)

        # Add noise using flow matching: noisy = (1 - t) * latents + t * noise
        noisy_latents = add_noise_unified(
            noise_process=noise_process,
            noise_scheduler=self.noise_scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )

        # ============================================================
        # Reference Image Conditioning (Latent Concatenation)
        # ============================================================
        # If reference latents are provided, pack them and concatenate with noisy latents
        # This allows the model to condition on reference images during training
        #
        # Multiple reference images per batch item:
        # - reference_latents_nested is List[List[Tensor]] where each inner list contains
        #   reference latents for one batch item
        # - Each reference image gets T coordinate offset: 10, 20, 30, ...
        # - All reference latents are packed and concatenated per batch item
        #
        # Shape: noisy_latents [B, seq_len, C] + ref_latents [B, ref_seq_len, C]
        #        -> concatenated [B, seq_len + ref_seq_len, C]
        # img_ids are also extended with reference position IDs
        packed_reference_latents = None
        if reference_latents_nested is not None and len(reference_latents_nested) > 0:
            # Process each batch item's reference images
            all_packed_refs = []
            all_ref_ids = []

            for batch_idx, item_ref_latents in enumerate(reference_latents_nested):
                item_packed_refs = []
                item_ref_ids = []

                for ref_idx, ref_latent in enumerate(item_ref_latents):
                    # ref_latent shape: [1, C, H, W] (single reference image)
                    # Pack: (1, C, H, W) -> (1, H*W, C)
                    packed_ref = self._flux2_pack_latents(ref_latent)
                    packed_ref = packed_ref.to(device=self.device, dtype=self.training_dtype, non_blocking=True)
                    item_packed_refs.append(packed_ref)

                    # Prepare position IDs for this reference image
                    ref_img_id = self._flux2_prepare_latent_ids(ref_latent).to(self.device)
                    # Apply T coordinate offset: T = scale + scale * ref_idx (scale=10)
                    # ref_idx 0 -> T=10, ref_idx 1 -> T=20, ref_idx 2 -> T=30, etc.
                    t_offset = 10 + 10 * ref_idx
                    ref_img_id[..., 0] = ref_img_id[..., 0] + t_offset
                    item_ref_ids.append(ref_img_id)

                # Concatenate all reference latents for this batch item
                # Shape: (1, total_ref_seq_len, C)
                item_packed_concat = torch.cat(item_packed_refs, dim=1)
                item_ids_concat = torch.cat(item_ref_ids, dim=1)

                all_packed_refs.append(item_packed_concat)
                all_ref_ids.append(item_ids_concat)

            # Stack across batch dimension
            # All batch items must have same total reference sequence length
            # (This is guaranteed if all items have same number of reference images with same dimensions)
            # If dimensions vary, we need padding - for now, assume consistent structure
            try:
                packed_reference_latents = torch.cat(all_packed_refs, dim=0)  # [B, ref_seq_len, C]
                ref_img_ids = torch.cat(all_ref_ids, dim=0)  # [B, ref_seq_len, 4]

                # Concatenate reference latents with noisy latents along sequence dimension
                noisy_latents = torch.cat([noisy_latents, packed_reference_latents], dim=1)

                # Concatenate reference position IDs with image position IDs
                img_ids = torch.cat([img_ids, ref_img_ids], dim=1)
            except RuntimeError as e:
                # Handle dimension mismatch (different reference image counts/sizes per batch item)
                print(f"{self.log_prefix} WARNING: Reference latent dimension mismatch in batch, skipping reference conditioning: {e}")
                packed_reference_latents = None

        if profile_vram:
            print_vram_usage("[train_step_flux2] Before Transformer forward")

        # Predict velocity using FLUX.2 Transformer
        if self.mixed_precision:
            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                output = self.transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )
                model_pred = output[0]
        else:
            output = self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )
            model_pred = output[0]

        if profile_vram:
            print_vram_usage("[train_step_flux2] After Transformer forward")

        # ============================================================
        # Slice output to remove reference latent predictions
        # ============================================================
        # If we concatenated reference latents, the model output contains predictions
        # for both target + reference. We only want predictions for the target latents.
        original_seq_len = latents.shape[1]  # Original target latent sequence length
        if packed_reference_latents is not None:
            # Slice to keep only predictions for target latents
            model_pred = model_pred[:, :original_seq_len, :]
            # Also slice noisy_latents for consistency in loss computation
            noisy_latents = noisy_latents[:, :original_seq_len, :]

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
        loss_per_sample = loss_per_element.mean([1, 2])  # Mean over seq_len and channels

        # Flow Matching doesn't use Min-SNR weighting (uniform timestep distribution)
        mse_loss = loss_per_sample.mean()

        # Add regularization if enabled
        regularization_loss = torch.tensor(0.0, device=self.device)

        # Compute predicted latent once (used by regularization losses and dual loss)
        predicted_latent_for_reg = None
        if self.snr_regularization_loss is not None or self.energy_regularization_loss is not None or self.reconstruction_loss_weight > 0:
            predicted_latent_for_reg = predict_original_latent_unified(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noise_scheduler=self.noise_scheduler,
                noisy_latents=noisy_latents,
                model_pred=model_pred,
                timesteps=timesteps,
            )

        # SNR regularization
        if self.snr_regularization_loss is not None:
            snr_reg_loss = self.snr_regularization_loss(
                predicted_latent_for_reg,
                latents,
                timesteps
            )
            regularization_loss = regularization_loss + snr_reg_loss

        # Energy regularization
        if self.energy_regularization_loss is not None:
            energy_reg_loss = self.energy_regularization_loss(
                predicted_latent_for_reg,
                latents,
                timesteps
            )
            regularization_loss = regularization_loss + energy_reg_loss

        # Calculate reconstruction loss
        if self.reconstruction_loss_weight > 0:
            if predicted_latent_for_reg is not None:
                predicted_latent_for_recon = predicted_latent_for_reg
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
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2])
            recon_loss = recon_loss_per_sample.mean()

            alpha = 1.0 - self.reconstruction_loss_weight
            beta = self.reconstruction_loss_weight
            combined_loss = alpha * mse_loss + beta * recon_loss

            loss = combined_loss + regularization_loss
        else:
            with torch.no_grad():
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
                recon_loss_per_sample = recon_loss_per_element.mean([1, 2])
                recon_loss = recon_loss_per_sample.mean()

            loss = mse_loss + regularization_loss

        if profile_vram:
            print_vram_usage("[train_step_flux2] After loss calculation")

        # Debug save if requested
        if debug_save_path is not None:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            timestep_value = timesteps[0].item()

            with torch.no_grad():
                # FLUX.2 uses standard Flow Matching: x_0 = x_t - t * v
                t = timesteps.float()
                while t.dim() < noisy_latents.dim():
                    t = t.unsqueeze(-1)
                predicted_latent = noisy_latents - t * model_pred

                # Convert packed latents (B, seq_len, C) to (B, C, H, W) for visualization
                # This makes debug output consistent with other models (SD/SDXL/Z-Image)
                latents_4d = self._flux2_unpack_latents_with_ids(latents[0:1], img_ids[0:1])
                noisy_latents_4d = self._flux2_unpack_latents_with_ids(noisy_latents[0:1], img_ids[0:1])
                predicted_velocity_4d = self._flux2_unpack_latents_with_ids(model_pred[0:1], img_ids[0:1])
                actual_velocity_4d = self._flux2_unpack_latents_with_ids(target[0:1], img_ids[0:1])
                predicted_latent_4d = self._flux2_unpack_latents_with_ids(predicted_latent[0:1], img_ids[0:1])

            debug_data = {
                'latents': latents_4d.detach().cpu(),
                'noisy_latents': noisy_latents_4d.detach().cpu(),
                'predicted_velocity': predicted_velocity_4d.detach().cpu(),
                'actual_velocity': actual_velocity_4d.detach().cpu(),
                'predicted_latent': predicted_latent_4d.detach().cpu(),
                'timestep': timestep_value,
                'loss': loss_per_sample[0].item(),
                'loss_batch_mean': loss.item(),
                'recon_loss': recon_loss_per_sample[0].item(),
                'recon_loss_batch_mean': recon_loss.item(),
                'batch_size': batch_size,
                'scheduler_type': 'FlowMatching',
                'model_type': 'flux2',
                'img_ids_shape': list(img_ids.shape),
                'txt_ids_shape': list(txt_ids.shape),
                'latent_shape_4d': list(latents_4d.shape),  # Store 4D shape for reference
            }

            if debug_captions is not None and len(debug_captions) > 0:
                debug_data['caption'] = debug_captions[0]
                debug_data['all_captions'] = debug_captions

            if debug_reference_image_paths is not None and len(debug_reference_image_paths) > 0:
                first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
                if first_ref:
                    debug_data['reference_image_path'] = first_ref

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:.4f}.pt")
            del predicted_latent, latents_4d, noisy_latents_4d, predicted_velocity_4d, actual_velocity_4d, predicted_latent_4d

        # Return loss tensor and loss values
        pred_loss_value = mse_loss.item()
        recon_loss_value = recon_loss.item()

        # Free intermediate tensors
        del noise, noisy_latents, model_pred, target
        del loss_per_element, loss_per_sample, recon_loss_per_element, recon_loss_per_sample, recon_loss

        return loss, pred_loss_value, recon_loss_value

    # ============================================================
    # FLUX.2 Position ID Helpers
    # ============================================================

    def _flux2_prepare_text_ids(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Prepare 4D position IDs for FLUX.2 text embeddings.

        FLUX.2 uses 4D position coordinates: (T, H, W, L)
        - T: Time coordinate (0 for text)
        - H: Height coordinate (0 for text - dummy dimension)
        - W: Width coordinate (0 for text - dummy dimension)
        - L: Sequence position (0 to seq_len-1)

        Args:
            prompt_embeds: Text embeddings [B, seq_len, hidden_dim]

        Returns:
            text_ids: Position IDs [B, seq_len, 4]
        """
        batch_size, seq_len, _ = prompt_embeds.shape
        out_ids = []

        for _ in range(batch_size):
            t = torch.arange(1)  # Time: 0
            h = torch.arange(1)  # Height: 0 (dummy)
            w = torch.arange(1)  # Width: 0 (dummy)
            l = torch.arange(seq_len)  # Sequence position
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def _flux2_prepare_latent_ids(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Prepare 4D position IDs for FLUX.2 image latents.

        FLUX.2 uses 4D position coordinates: (T, H, W, L)
        - T: Time coordinate (0 for single image)
        - H: Height coordinate (0 to height-1)
        - W: Width coordinate (0 to width-1)
        - L: Channel/patch coordinate (0 for unpatchified)

        Args:
            latents: Image latents [B, C, H, W]

        Returns:
            img_ids: Position IDs [B, H*W, 4]
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # Time: 0
        h = torch.arange(height)  # Height positions
        w = torch.arange(width)  # Width positions
        l = torch.arange(1)  # Patch/channel: 0

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    def _flux2_pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents from (B, C, H, W) to (B, H*W, C) for FLUX.2 transformer.

        Args:
            latents: Image latents [B, C, H, W]

        Returns:
            packed_latents: [B, H*W, C]
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    def _flux2_encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt for FLUX.2 using Qwen3 text encoder.

        FLUX.2 Klein uses Qwen3 with hidden states from layers 9, 18, 27.
        Output is concatenated: (B, seq_len, 3 * hidden_dim)

        IMPORTANT: This must match pipeline.py _flux2_encode_prompt() exactly,
        including chat template application, attention_mask, and use_cache settings.

        Args:
            prompt: Text prompt
            max_sequence_length: Maximum sequence length

        Returns:
            Tuple of (prompt_embeds, text_ids)
        """
        # Apply chat template (must match inference exactly)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        # Forward through text encoder (must match inference exactly)
        with torch.no_grad():
            output = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract hidden states from specified layers (9, 18, 27 for Klein 4B)
        # FLUX.2 Klein uses layers 9, 18, 27 (1-indexed), which are indices 9, 18, 27 in hidden_states array
        # This must match inference code in pipeline.py:_flux2_encode_prompt()
        hidden_states_layers = (9, 18, 27)  # Same as inference

        # Stack hidden states
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=self.training_dtype, device=self.device)

        # Reshape: (B, num_layers, seq_len, hidden_dim) -> (B, seq_len, num_layers * hidden_dim)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        # Prepare text IDs
        text_ids = self._flux2_prepare_text_ids(prompt_embeds).to(self.device)

        return prompt_embeds, text_ids

    # ============================================================
    # Sample Generation
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
        condition_image_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
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
        import random

        print(f"{self.log_prefix} Generating sample: {prompt[:50]}...")

        # SD/SDXL: Use custom_sampling_loop
        from core.inference.custom_sampling import custom_sampling_loop
        from core.inference.schedulers import get_scheduler

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
        log_verbose(f"{self.log_prefix} [Sample] U-Net has {lora_layers_found} LoRA layers")

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
                log_verbose(f"{self.log_prefix} [Sample] Padded negative embeddings: {negative_prompt_embeds.shape[1] - seq_len_diff} -> {negative_prompt_embeds.shape[1]} tokens")

            self.move_text_encoder_to_cpu()
            torch.cuda.empty_cache()

            # ========================================
            # STEP 2.5: Vision Encoder conditioning (if reference image + VE loaded)
            # ========================================
            ve_obj = getattr(self, 'vision_encoder', None)
            if reference_image_path and ve_obj is not None:
                try:
                    from PIL import Image as PILImage
                    ref_img = PILImage.open(reference_image_path).convert("RGB")
                    target_dim = prompt_embeds.shape[-1]
                    train_ve = getattr(self, '_train_vision_encoder', False)
                    if not train_ve:
                        print(f"{self.log_prefix} [Sample] Moving Vision Encoder to GPU for sample conditioning")
                        ve_obj.to(self.device)
                    ve_obj.eval()
                    with torch.no_grad():
                        ve_pos, _ = ve_obj.encode([ref_img], target_dim=target_dim, dtype=prompt_embeds.dtype)
                    ve_pos = ve_pos.to(self.device)
                    ve_neg = torch.zeros_like(ve_pos)
                    prompt_embeds = torch.cat([prompt_embeds, ve_pos], dim=1)
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds, ve_neg], dim=1)
                    if not train_ve:
                        ve_obj.to("cpu")
                        torch.cuda.empty_cache()
                        print(f"{self.log_prefix} [Sample] Vision Encoder moved back to CPU")
                    print(f"{self.log_prefix} [Sample] VE conditioning applied: embeds shape {prompt_embeds.shape}")
                except Exception as ve_err:
                    print(f"{self.log_prefix} [Sample] WARNING: VE conditioning failed: {ve_err}, skipping")

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

            log_verbose(f"{self.log_prefix} [Sample] Using custom_sampling_loop()")
            log_verbose(f"{self.log_prefix} [Sample] Scheduler: {type(pipeline.scheduler).__name__}")
            log_verbose(f"{self.log_prefix} [Sample] V-prediction: {is_v_prediction}, guidance_rescale: {guidance_rescale}")

            # Use autocast for sample generation (ensures LoRA dtype compatibility)
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

                log_verbose(f"{self.log_prefix} Sample generated successfully (seed: {actual_seed})")
                return image

        except Exception as e:
            print(f"{self.log_prefix} [Sample] ERROR: {type(e).__name__}: {str(e)}")
            print(f"{self.log_prefix} [Sample] Sample generation failed - this is expected for early training steps")
            print(f"{self.log_prefix} [Sample] Training will continue normally")

            # Return a placeholder image (blank white image)
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
            log_verbose(f"{self.log_prefix} [Sample] Offloading Transformer and Optimizer state to CPU")

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
            log_verbose(f"{self.log_prefix} [Sample] Transformer and Optimizer state offloaded to CPU")

            # ============================================================
            # Stage 1: Text Encoding (Sequential Offloading Pattern)
            # ============================================================
            # Move Text Encoder to GPU for encoding
            if text_encoder_device != self.device:
                log_verbose(f"{self.log_prefix} [Sample] Moving Text Encoder to GPU for encoding")
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
                log_verbose(f"{self.log_prefix} [Sample] Moving Text Encoder back to CPU")
                self.text_encoder.to(text_encoder_device)
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Move Transformer back to GPU for denoising
            # ============================================================
            log_verbose(f"{self.log_prefix} [Sample] Moving Transformer to GPU for denoising")
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
            log_verbose(f"{self.log_prefix} [Sample] Running denoising loop (Transformer on GPU)")

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
    # FLUX.2 Sample Generation
    # ============================================================

    def _generate_sample_flux2(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        seed: int = -1,
        reference_image_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate sample image during training (FLUX.2 Klein).

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
        import random
        import numpy as np

        print(f"{self.log_prefix} Generating FLUX.2 sample: {prompt[:50]}...")

        # Set models to eval mode for inference
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Store original devices for restoration
        text_encoder_device = next(self.text_encoder.parameters()).device
        vae_device = next(self.vae.parameters()).device
        transformer_device = next(self.transformer.parameters()).device

        try:
            # ============================================================
            # Stage 0: Offload Transformer AND Optimizer State to CPU
            # ============================================================
            log_verbose(f"{self.log_prefix} [Sample] Offloading Transformer and Optimizer state to CPU")

            # Move Transformer to CPU
            self.transformer.to("cpu")

            # CRITICAL: Move Optimizer state (gradients, momentum) to CPU
            optimizer_state_dict = self.optimizer.state_dict()
            for param_id, state in optimizer_state_dict['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                        state[key] = value.cpu()
            self.optimizer.load_state_dict(optimizer_state_dict)

            torch.cuda.empty_cache()
            log_verbose(f"{self.log_prefix} [Sample] Transformer and Optimizer state offloaded to CPU")

            # ============================================================
            # Stage 1: Text Encoding (Qwen3)
            # ============================================================
            if text_encoder_device != self.device:
                log_verbose(f"{self.log_prefix} [Sample] Moving Text Encoder to GPU for encoding")
                self.text_encoder.to(self.device)

            # Encode prompt using FLUX.2's Qwen3 text encoder
            prompt_embeds, text_ids = self._flux2_encode_prompt_for_sample(prompt)

            # Encode unconditional prompt only if CFG is enabled
            if guidance_scale > 1.0:
                negative_prompt_embeds, negative_text_ids = self._flux2_encode_prompt_for_sample("")
            else:
                negative_prompt_embeds, negative_text_ids = None, None

            # Move Text Encoder back to CPU to free VRAM
            if text_encoder_device != self.device:
                log_verbose(f"{self.log_prefix} [Sample] Moving Text Encoder back to CPU")
                self.text_encoder.to(text_encoder_device)
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Move Transformer back to GPU for denoising
            # ============================================================
            log_verbose(f"{self.log_prefix} [Sample] Moving Transformer to GPU for denoising")
            self.transformer.to(transformer_device)
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.6: Reference Image VAE encoding (FLUX.2 latent concat)
            # ============================================================
            packed_reference_latents = None
            ref_img_ids = None
            if reference_image_path:
                try:
                    from PIL import Image as PILImage
                    ref_img = PILImage.open(reference_image_path).convert("RGB")
                    ref_img = ref_img.resize((width, height), PILImage.LANCZOS)
                    print(f"{self.log_prefix} [Sample] Moving VAE to GPU for reference image encoding")
                    self.vae.to(self.device)
                    with torch.no_grad():
                        ref_tensor = torch.from_numpy(
                            np.array(ref_img).astype(np.float32) / 127.5 - 1.0
                        ).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.vae.dtype)
                        ref_latent = self.vae.encode(ref_tensor).latent_dist.sample()
                        ref_latent = ref_latent * self.vae.config.scaling_factor
                    self.vae.to("cpu")
                    torch.cuda.empty_cache()
                    packed_reference_latents = self._flux2_pack_latents_for_sample(ref_latent)
                    packed_reference_latents = packed_reference_latents.to(
                        device=self.device, dtype=prompt_embeds.dtype)
                    ref_ids = self._flux2_prepare_latent_ids_for_sample(ref_latent).to(self.device)
                    ref_ids[..., 0] = ref_ids[..., 0] + 10  # T coordinate offset
                    ref_img_ids = ref_ids
                    print(f"{self.log_prefix} [Sample] Reference image encoded: {packed_reference_latents.shape}")
                except Exception as ref_err:
                    print(f"{self.log_prefix} [Sample] WARNING: Reference image encoding failed: {ref_err}, skipping")
                    packed_reference_latents = None
                    ref_img_ids = None

            # ============================================================
            # Stage 2: Prepare Latents
            # ============================================================
            vae_scale_factor = 8
            patch_size = 2

            # Ensure height/width divisible by vae_scale_factor * patch_size
            latent_height = 2 * (int(height) // (vae_scale_factor * patch_size))
            latent_width = 2 * (int(width) // (vae_scale_factor * patch_size))

            # FLUX.2 has 32 latent channels, but patchified to 128
            num_channels_latents = self.transformer.config.in_channels // 4  # 32

            # Create random latents with seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

            latent_shape = (1, num_channels_latents * 4, latent_height // 2, latent_width // 2)
            latents = torch.randn(latent_shape, generator=generator, device=self.device, dtype=prompt_embeds.dtype)

            # Prepare latent position IDs
            latent_ids = self._flux2_prepare_latent_ids_for_sample(latents).to(self.device)

            # Pack latents: (B, C, H, W) -> (B, H*W, C)
            latents = self._flux2_pack_latents_for_sample(latents)

            # Concatenate reference latents with noise latents (if provided)
            if packed_reference_latents is not None and ref_img_ids is not None:
                latents = torch.cat([latents, packed_reference_latents], dim=1)
                latent_ids = torch.cat([latent_ids, ref_img_ids], dim=1)
                print(f"{self.log_prefix} [Sample] Latents after reference concat: {latents.shape}")

            # ============================================================
            # Stage 3: Denoising Loop
            # ============================================================
            log_verbose(f"{self.log_prefix} [Sample] Running denoising loop")

            # Prepare timesteps
            image_seq_len = latents.shape[1]
            mu = self._flux2_compute_empirical_mu_for_sample(image_seq_len, num_inference_steps)

            # Set timesteps with sigmas
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
            timesteps = self.scheduler.timesteps
            self.scheduler.set_begin_index(0)

            # Check if distilled model (no CFG)
            is_distilled = getattr(self.transformer.config, "is_distilled", False)
            do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

            with torch.no_grad():
                for i, t in enumerate(tqdm(timesteps, desc="Generating")):
                    # Expand timestep
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    latent_model_input = latents.to(self.transformer.dtype)

                    # Batch CFG: Concatenate unconditional and conditional for single forward pass
                    if do_classifier_free_guidance:
                        # Double the batch: [uncond, cond]
                        latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                        timestep_doubled = torch.cat([timestep, timestep], dim=0)
                        prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                        latent_ids_doubled = torch.cat([latent_ids, latent_ids], dim=0)

                        # Single forward pass for both unconditional and conditional
                        noise_pred_combined = self.transformer(
                            hidden_states=latent_model_input_doubled,
                            timestep=timestep_doubled / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds_combined,
                            txt_ids=text_ids_combined,
                            img_ids=latent_ids_doubled,
                            return_dict=False,
                        )[0]

                        # Split and apply CFG formula
                        noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        # Distilled model: Use guidance vector (not CFG)
                        guidance_vec = torch.full(
                            (latent_model_input.shape[0],),
                            guidance_scale,
                            device=latent_model_input.device,
                            dtype=latent_model_input.dtype
                        )
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance_vec,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_ids,
                            return_dict=False,
                        )[0]

                    # Scheduler step
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    if latents.dtype != latents_dtype:
                        latents = latents.to(latents_dtype)

            # Free prompt embeddings
            del prompt_embeds, text_ids
            if negative_prompt_embeds is not None:
                del negative_prompt_embeds, negative_text_ids

            # ============================================================
            # Stage 4: Offload Transformer to CPU, move VAE to GPU
            # ============================================================
            print(f"{self.log_prefix} [Sample] Moving Transformer to CPU to free VRAM")
            self.transformer.to("cpu")
            torch.cuda.empty_cache()

            # Move VAE to GPU for decoding
            if vae_device != self.device:
                print(f"{self.log_prefix} [Sample] Moving VAE to GPU for decoding")
                self.vae.to(device=self.device, dtype=self.vae_dtype)

            # Decode latents
            image = self._decode_flux2_latents(latents, latent_ids, latent_height, latent_width)

            # Move VAE back to CPU
            if vae_device != self.device:
                print(f"{self.log_prefix} [Sample] Moving VAE back to CPU")
                self.vae.to(device=vae_device, dtype=self.vae_dtype)

            # Free latents
            del latents, latent_ids
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 5: Restore Transformer and Optimizer State to GPU
            # ============================================================
            print(f"{self.log_prefix} [Sample] Restoring Transformer and Optimizer state to GPU")

            # Move Transformer back to GPU
            self.transformer.to(transformer_device)

            # CRITICAL: Move Optimizer state back to GPU
            from .optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer
            from .optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer
            if not isinstance(self.optimizer, (AdamW8bit_RingBuffer, Lion8bit_RingBuffer)):
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
            # Restore models to train mode
            self.transformer.train()

    def _flux2_encode_prompt_for_sample(self, prompt: str):
        """Encode prompt using Qwen3 text encoder for FLUX.2 sample generation."""
        max_sequence_length = 512
        hidden_states_layers = (9, 18, 27)

        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        # Apply chat template
        # IMPORTANT: Must match pipeline.py _flux2_encode_prompt() exactly
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Forward pass with hidden states
        # IMPORTANT: Must match pipeline.py _flux2_encode_prompt() exactly
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract and stack hidden states from specified layers
        # IMPORTANT: Must match pipeline.py _flux2_encode_prompt() exactly
        # Use stack + permute + reshape (NOT simple cat) for correct tensor structure
        out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        # Reshape: (B, num_layers, seq_len, hidden_dim) -> (B, seq_len, num_layers * hidden_dim)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        # Generate text IDs for RoPE
        batch_size, seq_len = prompt_embeds.shape[:2]
        text_ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=torch.long)
        text_ids[..., 0] = 0  # T dimension
        text_ids[..., 3] = torch.arange(seq_len, device=device)  # L dimension

        return prompt_embeds, text_ids

    def _flux2_prepare_latent_ids_for_sample(self, latents: torch.Tensor) -> torch.Tensor:
        """Prepare latent position IDs for FLUX.2 sample generation."""
        batch_size, channels, height, width = latents.shape

        # Create position IDs for each latent position
        latent_ids = torch.zeros(batch_size, height * width, 4, device=latents.device)

        # T=0, H, W, L coordinates
        h_coords = torch.arange(height, device=latents.device).repeat_interleave(width)
        w_coords = torch.arange(width, device=latents.device).repeat(height)
        l_coords = torch.arange(height * width, device=latents.device)

        latent_ids[:, :, 0] = 1  # T dimension (different from text)
        latent_ids[:, :, 1] = h_coords
        latent_ids[:, :, 2] = w_coords
        latent_ids[:, :, 3] = l_coords

        return latent_ids

    def _flux2_pack_latents_for_sample(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents from (B, C, H, W) to (B, H*W, C) for FLUX.2."""
        batch_size, channels, height, width = latents.shape
        latents = latents.permute(0, 2, 3, 1)  # (B, H, W, C)
        latents = latents.reshape(batch_size, height * width, channels)  # (B, H*W, C)
        return latents

    def _flux2_compute_empirical_mu_for_sample(self, image_seq_len: int, num_steps: int) -> float:
        """Compute empirical mu for FLUX.2 timestep scheduling."""
        # From diffusers FLUX implementation
        return 0.5 * (math.log(1 + image_seq_len) - math.log(num_steps))

    def _decode_flux2_latents(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        latent_height: int,
        latent_width: int
    ) -> Image.Image:
        """Decode FLUX.2 latents to PIL image."""
        import numpy as np

        # Step 1: Unpack latents using position IDs: (B, H*W, C) -> (B, C, H, W)
        latents = self._flux2_unpack_latents_with_ids(latents, latent_ids)

        # Step 2: Apply BatchNorm scaling (FLUX.2-specific)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean

        # Step 3: Unpatchify: (B, 128, H/2, W/2) -> (B, 32, H, W)
        latents = self._flux2_unpatchify_latents(latents)

        # Convert latents to VAE dtype (bfloat16 -> float32)
        latents = latents.to(dtype=self.vae.dtype)

        # Decode
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).astype(np.uint8)

        return Image.fromarray(image)

    def _flux2_unpack_latents_with_ids(self, x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """Unpack latents using position IDs: (B, H*W, C) -> (B, C, H, W)"""
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def _flux2_patchify_latents_for_training(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify latents for 2x2 patches: (B, 32, H, W) -> (B, 128, H/2, W/2)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    def _flux2_unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Unpatchify latents from 2x2 patches: (B, 128, H/2, W/2) -> (B, 32, H, W)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

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
        log_verbose(f"[Latent Cache] Generating latent cache with model offloading...")

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

            # Log to file only (no console spam)
            log_verbose(f"[Latent Cache] Caching dataset {dataset.unique_id} ({len(dataset.items)} items)...")

            for item in tqdm(dataset.items, desc=f"Caching {dataset.unique_id}", disable=True):
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
        log_verbose(f"[Latent Cache] Generation complete ({iteration_count} images encoded)")

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
        sample_prompts: Optional[List[Dict[str, str]]] = None,
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
        update_total_steps_callback: Optional[Callable[[int], None]] = None,
        run_id: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        force_recache: bool = False,
        max_step_saves_to_keep: int = 3,
        text_encoding_mode: str = "swap_onthefly",
        text_encoding_swap_interval: int = 256,
        latent_encoding_mode: str = "swap_onthefly",
        latent_encoding_swap_interval: int = 256,
        use_reference_images: bool = False,
        train_vision_encoder: bool = False,
        vision_encoder_path: Optional[str] = None,
        vision_encoder_lr: Optional[float] = None,
        gradient_routing_ve: bool = False,
        param_tracking: bool = False,
        param_tracking_interval: int = 100,
        priority_training: Optional[Dict] = None,
    ):
        """
        Main training loop.

        Args:
            datasets: List of dataset objects
            num_epochs: Number of training epochs
            batch_size: Batch size per step
            save_every_n_steps: Save checkpoint every N steps
            sample_every_n_steps: Generate sample every N steps
            sample_prompts: List of sample prompt dicts [{positive, negative, condition_image_path?}, ...]
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
            use_reference_images: Enable reference image conditioning during training (FLUX.2 only)
        """
        # Store references for subclass access
        self._training_datasets = datasets
        self._sample_prompts = sample_prompts or [{"positive": "a beautiful landscape", "negative": ""}]

        print(f"{self.log_prefix} Starting training...")
        print(f"{self.log_prefix} Datasets: {len(datasets)}")
        print(f"{self.log_prefix} Epochs: {num_epochs}")
        print(f"{self.log_prefix} Batch size: {batch_size}")
        print(f"{self.log_prefix} Gradient accumulation: {gradient_accumulation_steps}")
        print(f"{self.log_prefix} Debug latents: {debug_latents} (every {debug_latents_every} steps)")

        # Compute dataset fingerprint for change detection on resume
        # This is stored in training state and compared when resuming
        # IMPORTANT: Only image paths are included - caption changes do NOT invalidate shuffle state
        self._dataset_fingerprint = self._compute_dataset_fingerprint(datasets)
        print(f"{self.log_prefix} Dataset fingerprint: {self._dataset_fingerprint['total_item_count']} items, hash={self._dataset_fingerprint['image_paths_hash'][:8]}...")

        if use_reference_images:
            print(f"{self.log_prefix} Reference images: ENABLED (conditioning will be applied)")
            if not self.is_flux2:
                print(f"{self.log_prefix} WARNING: use_reference_images is only supported for FLUX.2, will be ignored")

        # Load Vision Encoder if specified (SigLIP2 for SDXL/SD1.5)
        if vision_encoder_path:
            print(f"{self.log_prefix} Vision Encoder: Loading from {vision_encoder_path}")
            try:
                from core.vision_encoder import SigLIP2VisionEncoderWrapper
                self.vision_encoder = SigLIP2VisionEncoderWrapper(vision_encoder_path, device="cpu")
                self._train_vision_encoder = train_vision_encoder
                self._gradient_routing_ve = gradient_routing_ve
                self._vision_encoder_lr = vision_encoder_lr
                if train_vision_encoder:
                    # Move to GPU immediately and keep it there for the duration of training.
                    # Per-batch CPU offloading is skipped when training VE (92.9M params ≈ 186MB
                    # is negligible vs UNet, and PCIe round-trips per batch hurt throughput).
                    self.vision_encoder.to(self.device)
                    print(f"{self.log_prefix} Vision Encoder: Will be trained (lr={vision_encoder_lr or 'inherit'}), kept on GPU")
                else:
                    print(f"{self.log_prefix} Vision Encoder: Frozen (inference only, CPU offloaded between batches)")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to load Vision Encoder: {e}")
                self.vision_encoder = None
                self._train_vision_encoder = False
                self._vision_encoder_lr = None
        else:
            if not hasattr(self, 'vision_encoder'):
                self.vision_encoder = None
            self._train_vision_encoder = False
            self._vision_encoder_lr = None

        # Validate text_encoding_mode when Text Encoder is trainable
        # Check if any Text Encoder has trainable parameters (works for both LoRA and full fine-tune)
        text_encoder_trainable = False
        te1_trainable_tensors = 0
        te1_trainable_scalars = 0
        te2_trainable_tensors = 0
        te2_trainable_scalars = 0

        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            te1_trainable_tensors = sum(1 for p in self.text_encoder.parameters() if p.requires_grad)
            te1_trainable_scalars = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
            text_encoder_trainable = te1_trainable_tensors > 0

        if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
            te2_trainable_tensors = sum(1 for p in self.text_encoder_2.parameters() if p.requires_grad)
            te2_trainable_scalars = sum(p.numel() for p in self.text_encoder_2.parameters() if p.requires_grad)
            text_encoder_trainable = text_encoder_trainable or (te2_trainable_tensors > 0)

        # Log trainable parameter counts (U-Net + Text Encoders)
        unet_obj = getattr(self, 'unet', None) or getattr(self, 'transformer', None)
        if unet_obj is not None:
            unet_trainable_tensors = sum(1 for p in unet_obj.parameters() if p.requires_grad)
            unet_trainable_scalars = sum(p.numel() for p in unet_obj.parameters() if p.requires_grad)
            print(f"{self.log_prefix} Trainable parameters:")
            print(f"{self.log_prefix}   U-Net/Transformer: tensors={unet_trainable_tensors}, params={format_param_count(unet_trainable_scalars)}")
        else:
            print(f"{self.log_prefix} Trainable parameters:")
        if text_encoder_trainable:
            if te1_trainable_tensors > 0:
                print(f"{self.log_prefix}   Text Encoder 1:    tensors={te1_trainable_tensors}, params={format_param_count(te1_trainable_scalars)}")
            if te2_trainable_tensors > 0:
                print(f"{self.log_prefix}   Text Encoder 2:    tensors={te2_trainable_tensors}, params={format_param_count(te2_trainable_scalars)}")
        if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
            ve_trainable_tensors = sum(1 for p in self.vision_encoder.parameters() if p.requires_grad)
            ve_trainable_scalars = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
            print(f"{self.log_prefix}   Vision Encoder:    tensors={ve_trainable_tensors}, params={format_param_count(ve_trainable_scalars)}")

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

            # Enable reference separation when use_reference_images is enabled for FLUX.2
            separate_by_reference = use_reference_images and self.is_flux2

            bucket_manager = BucketManager(
                base_resolutions=base_resolutions,
                divisibility=8,
                strategy=bucket_strategy,
                multi_resolution_mode=multi_resolution_mode,
                separate_by_reference=separate_by_reference
            )
            print(f"{self.log_prefix} Bucketing enabled: base_resolutions={base_resolutions}, strategy={bucket_strategy}, mode={multi_resolution_mode}")
            if separate_by_reference:
                print(f"{self.log_prefix} Reference separation enabled: batches will be separated by reference image availability")
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
        # Log distribution-specific parameters
        if hasattr(timestep_sampler, 'mean') and hasattr(timestep_sampler, 'std'):
            print(f"{self.log_prefix} Timestep params: mean={timestep_sampler.mean:.2f}, std={timestep_sampler.std:.2f}")
        elif hasattr(timestep_sampler, 'alpha') and hasattr(timestep_sampler, 'beta'):
            print(f"{self.log_prefix} Timestep params: alpha={timestep_sampler.alpha:.2f}, beta={timestep_sampler.beta:.2f}")
        print(f"{self.log_prefix} Multi Noise-Timesteps (MNT): {multi_noise_timesteps}")

        # Cache alphas_cumprod on GPU to avoid repeated .to(device) calls in compute_snr()
        # This is called thousands of times during training, so caching saves significant overhead
        # Note: Flow Matching schedulers (FLUX.2) don't have alphas_cumprod
        if hasattr(self.noise_scheduler, 'alphas_cumprod'):
            alphas_cumprod_cached = self.noise_scheduler.alphas_cumprod.to(device=self.device)
            print(f"{self.log_prefix} Cached alphas_cumprod on GPU ({alphas_cumprod_cached.shape[0]} steps)")
        else:
            # FLUX.2 uses Flow Matching (no alphas_cumprod, SNR weighting not applicable)
            alphas_cumprod_cached = None
            print(f"{self.log_prefix} Flow Matching scheduler detected (no alphas_cumprod)")

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

        # Update DB with calculated total_steps (for resume correctness)
        if update_total_steps_callback is not None:
            update_total_steps_callback(actual_total_steps)

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
                    # For ve_reconstruction_mode items: inject reference_images BEFORE bucketing
                    # so bucket_manager records has_reference=True and includes reference_images
                    # in image_info. This must happen here, not in the epoch loop, because
                    # bucket_manager creates new image_info dicts that would lose the flag.
                    if item.get("_ve_reconstruction_mode") and not item.get("reference_images"):
                        item["reference_images"] = [item["image_path"]]

                    width = item.get("width", 1024)
                    height = item.get("height", 1024)
                    # Check if item has reference images
                    reference_images = item.get("reference_images", [])
                    has_reference = len(reference_images) > 0

                    bucket_key, image_info = bucket_manager.assign_image_to_bucket(
                        image_path=item["image_path"],
                        width=width,
                        height=height,
                        caption=item.get("caption", ""),
                        dataset_unique_id=dataset.unique_id,
                        has_reference=has_reference,
                        reference_images=reference_images if reference_images else None,
                    )
                    # Propagate _ve_reconstruction_mode into image_info so training step
                    # can zero text embeddings for these items.
                    if item.get("_ve_reconstruction_mode"):
                        image_info["_ve_reconstruction_mode"] = True
                    # Update item with bucket dimensions
                    item["width"] = image_info["bucket_width"]
                    item["height"] = image_info["bucket_height"]

            # Print bucket statistics
            bucket_counts = bucket_manager.get_bucket_counts()
            print(f"{self.log_prefix} Bucket distribution:")
            for bucket_size, count in sorted(bucket_counts.items()):
                print(f"  {bucket_size}: {count} images")

            # Print reference image statistics if separation is enabled
            if bucket_manager.separate_by_reference:
                ref_stats = bucket_manager.get_reference_statistics()
                print(f"{self.log_prefix} Reference image distribution:")
                print(f"  With reference: {ref_stats['with_reference']} images")
                print(f"  Without reference: {ref_stats['without_reference']} images")

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
        # NOTE: Checkpoint weights were already loaded in __init__() if resume_from_checkpoint was set
        # Here we only need to extract step number and load training state (epoch/batch_idx)
        if resume_from_checkpoint:
            if resume_from_checkpoint.lower() == "latest":
                # Use the checkpoint that was actually loaded in __init__ (may differ from "latest" if fallback occurred)
                if self._loaded_checkpoint_path:
                    checkpoint_path = self._loaded_checkpoint_path
                    # Extract step number from filename
                    import re
                    match = re.search(r'_step_(\d+)', Path(checkpoint_path).stem)
                    if match:
                        checkpoint_step = int(match.group(1))
                    else:
                        print(f"{self.log_prefix} WARNING: Could not extract step number from loaded checkpoint: {checkpoint_path}")
                        checkpoint_step = 0
                    checkpoint_result = (checkpoint_path, checkpoint_step)
                else:
                    # Fallback to find_latest_checkpoint (should not normally happen)
                    checkpoint_result = self.find_latest_checkpoint()

                if checkpoint_result is not None:
                    checkpoint_path, checkpoint_step = checkpoint_result
                    print(f"{self.log_prefix} Resuming from checkpoint (weights already loaded in __init__): {checkpoint_path}")
                    # NOTE: Model weights were already loaded in __init__()
                    # We only need the step number here
                    global_step = checkpoint_step

                    # Try to load training state for mid-epoch resume
                    resume_training_state = self.load_training_state(checkpoint_step)
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

                        # Restore ReLoRA-specific state (merge_count, etc.)
                        if hasattr(self, '_restore_relora_state'):
                            self._restore_relora_state(resume_training_state)
                    else:
                        # No training state file, fall back to epoch-level resume
                        start_epoch = global_step // steps_per_epoch
                        print(f"{self.log_prefix} Resuming from step {global_step}, epoch {start_epoch + 1}")

                    # Fast-forward lr_scheduler to match the checkpoint
                    for _ in range(global_step):
                        self.lr_scheduler.step()

                    # IMPORTANT: Update optimizer learning rate from YAML config
                    # (Necessary when user modifies LR in YAML before resume)
                    # Build component LR list matching actual optimizer group order:
                    #   [UNet, TE1, TE2 (SDXL only), VE (if _train_vision_encoder)]
                    if hasattr(self, 'optimizer') and self.optimizer is not None:
                        component_lrs, component_names = self._build_component_lr_list()

                        for i, param_group in enumerate(self.optimizer.param_groups):
                            old_lr = param_group['lr']
                            new_lr = component_lrs[i] if i < len(component_lrs) else self.learning_rate
                            name = component_names[i] if i < len(component_names) else f"group{i}"
                            param_group['lr'] = new_lr
                            if old_lr != new_lr:
                                print(f"{self.log_prefix} Updated optimizer {name} LR (param_group[{i}]): {old_lr:.2e} -> {new_lr:.2e}")

                    # IMPORTANT: Also update LR Scheduler's base_lrs to prevent it from resetting LR
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        if hasattr(self.lr_scheduler, 'base_lrs'):
                            component_lrs, component_names = self._build_component_lr_list()

                            for i in range(len(self.lr_scheduler.base_lrs)):
                                old_base_lr = self.lr_scheduler.base_lrs[i]
                                new_base_lr = component_lrs[i] if i < len(component_lrs) else self.learning_rate
                                name = component_names[i] if i < len(component_names) else f"group{i}"
                                self.lr_scheduler.base_lrs[i] = new_base_lr
                                if old_base_lr != new_base_lr:
                                    print(f"{self.log_prefix} Updated LR Scheduler {name} base_lr[{i}]: {old_base_lr:.2e} -> {new_base_lr:.2e}")

                    # Load optimizer state (momentum, variance, etc.)
                    self.load_optimizer_state(checkpoint_step)
                else:
                    print(f"{self.log_prefix} No checkpoint found for auto-resume, starting from scratch")
            else:
                # User specified a specific checkpoint file
                checkpoint_path = self.output_dir / resume_from_checkpoint
                if checkpoint_path.exists():
                    print(f"{self.log_prefix} Resuming from specified checkpoint (weights already loaded in __init__): {checkpoint_path}")

                    # NOTE: Model weights were already loaded in __init__()
                    # Extract step number from filename
                    import re
                    match = re.search(r'_step_(\d+)', checkpoint_path.stem)
                    if match:
                        checkpoint_step = int(match.group(1))
                        global_step = checkpoint_step
                    else:
                        print(f"{self.log_prefix} WARNING: Could not extract step number from filename: {checkpoint_path.name}")
                        global_step = 0

                    # Try to load training state for mid-epoch resume
                    resume_training_state = self.load_training_state(checkpoint_step)
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

                        # Restore ReLoRA-specific state (merge_count, etc.)
                        if hasattr(self, '_restore_relora_state'):
                            self._restore_relora_state(resume_training_state)
                    else:
                        # No training state file, fall back to epoch-level resume
                        start_epoch = global_step // steps_per_epoch
                        print(f"{self.log_prefix} Resuming from step {global_step}, epoch {start_epoch + 1}")

                    # Fast-forward lr_scheduler to match the checkpoint
                    for _ in range(global_step):
                        self.lr_scheduler.step()

                    # IMPORTANT: Update optimizer learning rate from YAML config
                    # (Necessary when user modifies LR in YAML before resume)
                    # Build component LR list matching actual optimizer group order:
                    #   [UNet, TE1, TE2 (SDXL only), VE (if _train_vision_encoder)]
                    if hasattr(self, 'optimizer') and self.optimizer is not None:
                        component_lrs, component_names = self._build_component_lr_list()

                        for i, param_group in enumerate(self.optimizer.param_groups):
                            old_lr = param_group['lr']
                            new_lr = component_lrs[i] if i < len(component_lrs) else self.learning_rate
                            name = component_names[i] if i < len(component_names) else f"group{i}"
                            param_group['lr'] = new_lr
                            if old_lr != new_lr:
                                print(f"{self.log_prefix} Updated optimizer {name} LR (param_group[{i}]): {old_lr:.2e} -> {new_lr:.2e}")

                    # IMPORTANT: Also update LR Scheduler's base_lrs to prevent it from resetting LR
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        if hasattr(self.lr_scheduler, 'base_lrs'):
                            component_lrs, component_names = self._build_component_lr_list()

                            for i in range(len(self.lr_scheduler.base_lrs)):
                                old_base_lr = self.lr_scheduler.base_lrs[i]
                                new_base_lr = component_lrs[i] if i < len(component_lrs) else self.learning_rate
                                name = component_names[i] if i < len(component_names) else f"group{i}"
                                self.lr_scheduler.base_lrs[i] = new_base_lr
                                if old_base_lr != new_base_lr:
                                    print(f"{self.log_prefix} Updated LR Scheduler {name} base_lr[{i}]: {old_base_lr:.2e} -> {new_base_lr:.2e}")

                    # Load optimizer state (momentum, variance, etc.)
                    self.load_optimizer_state(checkpoint_step)
                else:
                    print(f"{self.log_prefix} WARNING: Checkpoint not found: {checkpoint_path}")
                    print(f"{self.log_prefix} Starting from scratch")

        # ============================================================
        # MNT Change Detection and total_steps Recalculation
        # ============================================================
        # When MNT changes between runs, we need to recalculate total_steps:
        # - global_step (from checkpoint) = already completed steps
        # - remaining_steps = (remaining batches) * new_mnt
        # - new_total_steps = global_step + remaining_steps
        #
        # This ensures training continues for the correct duration regardless
        # of MNT changes during resume.
        if resume_training_state is not None and global_step > 0:
            checkpoint_mnt = resume_training_state.get('multi_noise_timesteps', 1)

            if checkpoint_mnt != multi_noise_timesteps:
                print(f"{self.log_prefix} MNT changed: {checkpoint_mnt} -> {multi_noise_timesteps}")

                # Calculate remaining batches from current position
                remaining_batches_in_epoch = batches_per_epoch - resume_batch_idx
                remaining_full_epochs = num_epochs - start_epoch - 1
                remaining_full_epoch_batches = remaining_full_epochs * batches_per_epoch
                total_remaining_batches = remaining_batches_in_epoch + remaining_full_epoch_batches

                # Calculate remaining steps with NEW MNT value
                remaining_steps = total_remaining_batches * multi_noise_timesteps

                # New total_steps = already completed + remaining
                new_actual_total_steps = global_step + remaining_steps

                print(f"{self.log_prefix} Recalculating total_steps due to MNT change:")
                print(f"{self.log_prefix}   Completed steps (from checkpoint): {global_step}")
                print(f"{self.log_prefix}   Remaining batches: {total_remaining_batches}")
                print(f"{self.log_prefix}   Remaining steps (with new MNT={multi_noise_timesteps}): {remaining_steps}")
                print(f"{self.log_prefix}   Old total_steps: {actual_total_steps}")
                print(f"{self.log_prefix}   New total_steps: {new_actual_total_steps}")

                actual_total_steps = new_actual_total_steps

                # Update DB with corrected total_steps
                if update_total_steps_callback is not None:
                    update_total_steps_callback(actual_total_steps)

                # Note: LR scheduler was already fast-forwarded to global_step
                # It will continue from there with the remaining steps
                # No need to reinitialize optimizer/scheduler since global_step is preserved
                #
                # Warning: For non-constant LR schedulers (cosine, etc.), the scheduler's
                # total_steps was set to the old value. This may cause incorrect LR decay.
                # For constant scheduler, this is not an issue.
                if lr_scheduler_type.lower() != "constant":
                    print(f"{self.log_prefix} WARNING: MNT change with {lr_scheduler_type} LR scheduler")
                    print(f"{self.log_prefix} WARNING: LR scheduler was initialized with old total_steps")
                    print(f"{self.log_prefix} WARNING: LR decay curve may be affected. Consider using 'constant' scheduler for MNT experiments.")

        # Clean up future steps in database (old data from previous interrupted training)
        # This prevents duplicate metrics when training resumes from an earlier step
        if self.run_id is not None:
            self._cleanup_future_metrics(global_step)

        # ============================================================
        # Parameter Change Tracker initialization
        # ============================================================
        self._param_tracker: Optional[ParameterChangeTracker] = None
        if param_tracking:
            tracked_components: Dict[str, torch.nn.Module] = {}
            if getattr(self, 'unet', None) is not None:
                tracked_components['unet'] = self.unet
            elif getattr(self, 'transformer', None) is not None:
                tracked_components['unet'] = self.transformer  # flux2 / zimage
            if getattr(self, 'text_encoder', None) is not None:
                tracked_components['te1'] = self.text_encoder
            if getattr(self, 'text_encoder_2', None) is not None:
                tracked_components['te2'] = self.text_encoder_2
            if (getattr(self, '_train_vision_encoder', False)
                    and getattr(self, 'vision_encoder', None) is not None):
                tracked_components['ve'] = self.vision_encoder
            if tracked_components:
                print(f"{self.log_prefix} [ParamTracker] Initializing "
                      f"(interval={param_tracking_interval} steps, "
                      f"components={list(tracked_components.keys())})...")
                self._param_tracker = ParameterChangeTracker(
                    tracked_components, interval=param_tracking_interval
                )
            else:
                print(f"{self.log_prefix} [ParamTracker] No trainable components found, disabled")

        # Generate step 0 sample to verify base model output
        if sample_every_n_steps > 0 and global_step == 0:
            step0_prompt = self._sample_prompts[0].get('positive', 'a beautiful landscape') if self._sample_prompts else 'a beautiful landscape'
            print(f"{self.log_prefix} [Step 0] Generating sample to verify base model...")
            print(f"{self.log_prefix} [Step 0] Sample params: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, steps={sample_steps}, seed={sample_seed}")
            if self.is_flux2:
                sample = self._generate_sample_flux2(
                    prompt=step0_prompt,
                    width=sample_width,
                    height=sample_height,
                    num_inference_steps=sample_steps,
                    guidance_scale=sample_guidance_scale,
                    seed=sample_seed
                )
            elif self.is_zimage:
                sample = self._generate_sample_zimage(
                    prompt=step0_prompt,
                    width=sample_width,
                    height=sample_height,
                    num_inference_steps=sample_steps,
                    guidance_scale=sample_guidance_scale,
                    seed=sample_seed
                )
            else:
                sample = self.generate_sample(
                    prompt=step0_prompt,
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
                        new_items = dataset.reload_for_epoch(epoch_num=epoch, run_id=run_id)
                        if new_items is not None:
                            # Dataset was reloaded with new items
                            dataset.items = new_items
                            print(f"{self.log_prefix} Reloaded dataset {dataset.unique_id} for epoch {epoch + 1} ({len(dataset.items)} items)")
                        else:
                            # Dataset reload skipped (same epoch as initial load, items already loaded)
                            print(f"{self.log_prefix} Using pre-loaded dataset {dataset.unique_id} for epoch {epoch + 1} ({len(dataset.items)} items)")

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

                    # Check if dataset has changed since checkpoint was saved
                    # If changed, the saved random_state is invalid and should NOT be restored
                    saved_fingerprint = resume_training_state.get('dataset_fingerprint')
                    dataset_changed = self._check_dataset_fingerprint_changed(saved_fingerprint, self._dataset_fingerprint)

                    if dataset_changed:
                        print(f"{self.log_prefix} WARNING: Dataset has changed since checkpoint was saved!")
                        print(f"{self.log_prefix} Saved shuffle state is invalid - using fresh random state")
                        print(f"{self.log_prefix} Restarting current epoch from batch 0 (global_step={global_step} preserved)")
                        # Do NOT restore random state - let it use current random state.
                        # Also clear resume_training_state so the batch-truncation at
                        # ``batches = batches[resume_batch_idx:]`` below does NOT run —
                        # otherwise we'd skip the first resume_batch_idx batches of an
                        # entirely different sample order, which means arbitrary
                        # samples get skipped rather than the ones already trained on.
                        resume_training_state = None
                        resume_batch_idx     = 0
                    else:
                        print(f"{self.log_prefix} Dataset unchanged - restoring random state for mid-epoch resume...")
                        random.setstate(resume_training_state['random_state'])

                # Load priority training config (if specified)
                priority_config = None
                if priority_training:
                    try:
                        from core.training.priority_training import (
                            PriorityTrainingConfig, classify_items, build_priority_batches
                        )
                        if isinstance(priority_training, dict) and "_legacy_path" in priority_training:
                            priority_config = PriorityTrainingConfig.load(priority_training["_legacy_path"])
                        else:
                            priority_config = PriorityTrainingConfig.from_dict(priority_training)
                    except Exception as e:
                        print(f"{self.log_prefix} WARNING: Failed to load priority training config: {e}")
                        print(f"{self.log_prefix} Continuing with normal training")

                # Create batches
                if bucket_manager:
                    # BucketManager only manages items, we need to pair with datasets
                    # Build mapping from image_path to dataset
                    path_to_dataset = {}
                    for dataset in datasets:
                        for item in dataset.items:
                            path_to_dataset[item["image_path"]] = dataset

                    if priority_config and priority_config.entries:
                        # Priority training: split items, build priority batches first
                        priority_items, normal_items = classify_items(all_items, priority_config)

                        # Build priority batches (sorted by entry index, bucketed by resolution)
                        priority_batches = build_priority_batches(
                            priority_items, batch_size, bucket_manager
                        )

                        # Build normal batches from remaining items using bucket manager
                        # Temporarily replace bucket contents with normal items only
                        from core.training.bucketing import BucketManager
                        normal_bucket_manager = BucketManager(
                            base_resolutions=bucket_manager.base_resolutions,
                            divisibility=8,
                            strategy=bucket_manager.strategy,
                            multi_resolution_mode=bucket_manager.multi_resolution_mode,
                        )
                        for item, dataset in normal_items:
                            normal_bucket_manager.assign_image_to_bucket(
                                image_path=item["image_path"],
                                width=item.get("width", 1024),
                                height=item.get("height", 1024),
                                caption=item.get("caption", ""),
                                dataset_unique_id=getattr(dataset, 'unique_id', None),
                            )
                        normal_item_batches = normal_bucket_manager.build_batch_indices(batch_size)
                        normal_batches = []
                        for item_batch in normal_item_batches:
                            batch_with_dataset = [
                                (item, path_to_dataset[item["image_path"]])
                                for item in item_batch
                            ]
                            normal_batches.append(batch_with_dataset)

                        # Combine: priority x multiplier + normal
                        batches = priority_batches * priority_config.multiplier + normal_batches
                        print(f"{self.log_prefix} [PriorityTraining] Epoch batch structure: "
                              f"{len(priority_batches)} priority batches x {priority_config.multiplier} "
                              f"+ {len(normal_batches)} normal batches = {len(batches)} total")
                    else:
                        # Standard bucketed batching (no priority)
                        item_batches = bucket_manager.build_batch_indices(batch_size)
                        batches = []
                        for item_batch in item_batches:
                            batch_with_dataset = [
                                (item, path_to_dataset[item["image_path"]])
                                for item in item_batch
                            ]
                            batches.append(batch_with_dataset)
                else:
                    # Simple sequential batching
                    if priority_config and priority_config.entries:
                        priority_items, normal_items = classify_items(all_items, priority_config)
                        p_items = [(item, dataset) for item, dataset, _ in priority_items]
                        priority_batches = [p_items[i:i+batch_size] for i in range(0, len(p_items), batch_size)]
                        normal_batches = [normal_items[i:i+batch_size] for i in range(0, len(normal_items), batch_size)]
                        batches = priority_batches * priority_config.multiplier + normal_batches
                        print(f"{self.log_prefix} [PriorityTraining] Epoch batch structure: "
                              f"{len(priority_batches)} priority x {priority_config.multiplier} "
                              f"+ {len(normal_batches)} normal = {len(batches)} total")
                    else:
                        batches = [all_items[i:i+batch_size] for i in range(0, len(all_items), batch_size)]

                # Mid-epoch resume: skip completed batches
                # (random state was already restored before batch building)
                if epoch == start_epoch and resume_training_state is not None:
                    print(f"{self.log_prefix} Skipping {resume_batch_idx} completed batches...")
                    batches = batches[resume_batch_idx:]

                    # Clear resume state so we don't skip batches in subsequent epochs
                    resume_training_state = None

                # Inject reference_images for ve_reconstruction_mode items (use own image as reference).
                # Must happen BEFORE the batch splitting below so these items go into "ref" sub-batches.
                if getattr(self, 'vision_encoder', None) is not None:
                    for _b in batches:
                        for _item, _ in _b:
                            if _item.get("_ve_reconstruction_mode") and not _item.get("reference_images"):
                                _item["reference_images"] = [_item["image_path"]]

                # When VE is configured, split any mixed batch (ref + no-ref) into pure sub-batches.
                # Ref-image batches and no-ref batches have different embedding shapes so they cannot
                # be collated together.
                if getattr(self, 'vision_encoder', None) is not None:
                    import random as _random_ve
                    clean_batches = []
                    for _b in batches:
                        _ref_items   = [(_i, _ds) for _i, _ds in _b if _i.get("reference_images")]
                        _noref_items = [(_i, _ds) for _i, _ds in _b if not _i.get("reference_images")]
                        if _ref_items and _noref_items:
                            # Mixed batch: split into two pure sub-batches
                            clean_batches.append(_ref_items)
                            clean_batches.append(_noref_items)
                            print(f"{self.log_prefix} [VE] Split mixed batch → "
                                  f"{len(_ref_items)} ref + {len(_noref_items)} no-ref sub-batches")
                        else:
                            clean_batches.append(_b)
                    _random_ve.shuffle(clean_batches)
                    batches = clean_batches

                # Initialize swap mode buffer if needed (all architectures)
                # Use dict keyed by image_path for robust lookup (immune to index misalignment)
                swap_buffer = {} if text_encoding_mode == "swap_onthefly" else None
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
                        # Store on CPU to save GPU VRAM, keyed by image_path
                        # auxiliary_data: attention_mask (Z-Image), pooled_embeddings (SDXL), None (SD1.5)
                        swap_buffer[image_path] = (
                            embeddings.cpu(),
                            auxiliary_data.cpu() if auxiliary_data is not None else None,
                            caption,  # String (CPU memory, minimal overhead)
                        )

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
                # Use dict keyed by image_path for robust lookup (immune to index misalignment)
                latent_swap_buffer = {} if latent_encoding_mode == "swap_onthefly" else None
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
                        # Store on CPU to save GPU VRAM, keyed by image_path
                        # This eliminates index-based lookup issues with variable batch sizes
                        latent_swap_buffer[image_path] = (
                            latent.cpu(),
                            caption,  # String (CPU memory, minimal overhead)
                        )

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

                # onthefly_gpu mode: Ensure text encoders and main model are on GPU for entire epoch
                if text_encoding_mode == "onthefly_gpu":
                    print(f"{self.log_prefix} Moving text encoders to GPU for onthefly_gpu mode...")
                    self.move_text_encoder_to_gpu()
                    # Ensure U-Net is on GPU (critical for mid-epoch resume)
                    self.move_main_model_to_gpu()

                # Training loop
                # Calculate expected steps for this epoch (accounting for MNT and mid-epoch resume)
                epoch_batches = len(batches)  # After mid-epoch resume slicing
                epoch_steps = epoch_batches * multi_noise_timesteps
                epoch_start_step = global_step

                # Update total_steps with actual batch count (first epoch only)
                # This corrects for bucketing overhead (each bucket rounds up batch count)
                # Works for both new training and resumed training
                if epoch == start_epoch:
                    # Calculate actual steps per epoch (before mid-epoch slicing)
                    if bucket_manager:
                        # For bucketing: use the full batch count before resume slicing.
                        # Priority training path may not define `item_batches`, so use
                        # the pre-sliced `batches` list which is always available here.
                        full_batch_count = len(batches)
                    else:
                        # For simple batching: calculate from total items
                        full_batch_count = (len(all_items) + batch_size - 1) // batch_size

                    actual_steps_per_epoch = full_batch_count * multi_noise_timesteps
                    actual_total_steps = actual_steps_per_epoch * num_epochs

                    # Update DB if actual differs from initial estimate
                    if actual_total_steps != steps_per_epoch * num_epochs:
                        print(f"{self.log_prefix} Correcting total_steps: {steps_per_epoch * num_epochs} → {actual_total_steps} (bucketing overhead)")
                        if update_total_steps_callback is not None:
                            update_total_steps_callback(actual_total_steps)

                for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs} ({epoch_steps} steps)")):
                    # Reset fused optimizer groups counters (start of each step)
                    if self.fused_optimizer_groups is not None:
                        self.fused_optimizer_groups.reset_counters()

                    # Check for stop flag (user-requested stop from frontend)
                    stop_flag_file = self.output_dir / ".stop_training"
                    if stop_flag_file.exists():
                        print(f"\n{self.log_prefix} Stop flag detected, stopping training...")
                        stop_flag_file.unlink()  # Clean up flag file
                        raise KeyboardInterrupt("Training stopped by user")

                    # Check for on-demand preview requests from the API
                    # (file-based RPC, see core/training/training_preview_rpc.py).
                    # Each request is processed in-place using the current
                    # in-training model state.  Failures are isolated per
                    # request and reported via the result file — training
                    # never crashes because of a bad request.
                    try:
                        from core.training.training_preview_rpc import (
                            list_pending_requests, read_request, cleanup_stale,
                        )
                        _pending = list_pending_requests(self.output_dir)
                        if _pending:
                            from core.training.training_inference import TrainingPreviewGenerator
                            if not hasattr(self, "_preview_gen"):
                                self._preview_gen = TrainingPreviewGenerator(self)
                            for _req_path in _pending:
                                _req = read_request(_req_path)
                                # Always delete the request file first so a
                                # malformed / re-emitted request isn't picked
                                # up twice.
                                try: _req_path.unlink()
                                except OSError: pass
                                if _req is None:
                                    continue
                                _rid = _req.get("request_id", "?")
                                _params = _req.get("params", {})
                                print(f"\n{self.log_prefix} Preview request {_rid} — processing...")
                                self._preview_gen.process_request(_rid, _params)
                                print(f"{self.log_prefix} Preview request {_rid} — done")
                            cleanup_stale(str(self.output_dir))
                    except Exception as _pe:   # noqa: BLE001
                        # Never let preview handling kill training
                        print(f"{self.log_prefix} WARNING: preview poll failed: {_pe}")

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

                        # Clear old buffer and encode new captions (dict keyed by image_path)
                        swap_buffer.clear()
                        for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding captions", leave=False)):
                            caption = item.get("caption", "")
                            image_path = item["image_path"]
                            embeddings, auxiliary_data = self.encode_caption(caption, requires_grad=False)
                            # Store on CPU to save GPU VRAM, keyed by image_path
                            swap_buffer[image_path] = (
                                embeddings.cpu(),
                                auxiliary_data.cpu() if auxiliary_data is not None else None,
                                caption,  # String (CPU memory, minimal overhead)
                            )

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

                        # Clear old buffer and encode new latents (dict keyed by image_path)
                        latent_swap_buffer.clear()
                        corrupted_images = []  # Track corrupted images for logging
                        for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding latents", leave=False)):
                            image_path = item["image_path"]
                            caption = item.get("caption", "")
                            width = item.get("width") or item.get("bucket_width")
                            height = item.get("height") or item.get("bucket_height")

                            # Load and encode image with corruption handling
                            try:
                                image = Image.open(image_path)
                                # Force load to detect truncated images early
                                image.load()
                                latent = self.encode_image(
                                    image=image,
                                    target_width=width,
                                    target_height=height,
                                    bucket_strategy=bucket_strategy
                                )
                                # Store on CPU to save GPU VRAM, keyed by image_path
                                latent_swap_buffer[image_path] = (
                                    latent.cpu(),
                                    caption,  # String (CPU memory, minimal overhead)
                                )
                            except Exception as img_error:
                                # Log corrupted image and skip it
                                corrupted_images.append(image_path)
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Skipping: {image_path}")
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Error: {str(img_error)[:200]}")
                                continue

                            # Send progress update
                            if progress_callback and idx % 10 == 0:
                                progress_callback(
                                    phase="latent_cache",
                                    step=idx,
                                    total=len(buffer_items)
                                )

                        # Log summary of corrupted images
                        if corrupted_images:
                            print(f"{self.log_prefix} [CORRUPTED IMAGES] Total skipped: {len(corrupted_images)}")
                            for path in corrupted_images:
                                print(f"{self.log_prefix} [CORRUPTED IMAGES]   - {path}")

                        # Move VAE back to CPU
                        self.move_vae_to_cpu()
                        # Move main model to GPU
                        self.move_main_model_to_gpu()

                        # Clear CUDA cache after model movement to free fragmented memory
                        torch.cuda.empty_cache()

                        next_latent_swap_at_step += latent_encoding_swap_interval
                        print(f"{self.log_prefix} Latent buffer refilled with {len(latent_swap_buffer)} latents")

                    # ============================================================
                    # Batch data preparation (ONCE per batch, OUTSIDE MNT loop)
                    # ============================================================
                    # IMPORTANT: Prepare batch tensors once and reuse across MNT iterations
                    # This prevents redundant CPU->GPU transfers and reduces VRAM fragmentation

                    latents_list = []
                    text_embeddings_list = []
                    auxiliary_data_list = []  # Unified: attention_mask (Z-Image), pooled_embeddings (SDXL), or None (SD1.5)
                    reference_latents_list = []  # FLUX.2 reference image conditioning
                    condition_images_list = []  # ControlNet condition images [B, 3, H, W]

                    # Flag to track if batch should be skipped due to corrupted image
                    batch_has_corrupted_image = False
                    corrupted_image_path = None

                    for item, dataset in batch:
                        # BucketManager stores bucket_width/bucket_height, not width/height
                        width = item.get("width") or item.get("bucket_width")
                        height = item.get("height") or item.get("bucket_height")
                        image_path = item["image_path"]

                        # Load latent (mode-specific)
                        if latent_encoding_mode == "swap_onthefly":
                            # Get from swap buffer using image_path as key (dict lookup)
                            # This eliminates index-based alignment issues
                            if image_path in latent_swap_buffer:
                                latent_cpu, buffer_caption = latent_swap_buffer[image_path]
                                # Transfer to GPU
                                latent = latent_cpu.to(self.device, non_blocking=True)
                                latents_list.append(latent)
                                # Update caption from buffer (ensures correct pairing)
                                item["caption"] = buffer_caption
                            else:
                                # Fallback to on-the-fly encoding (image not in buffer)
                                # This happens when buffer hasn't been refilled yet for this batch
                                # or when image was skipped during buffer refill (corrupted)
                                print(f"{self.log_prefix} WARNING: Image not in latent swap buffer, encoding on-the-fly: {image_path}")
                                try:
                                    self.move_vae_to_gpu()
                                    image = Image.open(image_path)
                                    image.load()  # Force load to detect truncated images
                                    latent = self.encode_image(
                                        image=image,
                                        target_width=width,
                                        target_height=height,
                                        bucket_strategy=bucket_strategy
                                    )
                                    # Ensure latent is on training device
                                    latent = latent.to(self.device)
                                    latents_list.append(latent)
                                    self.move_vae_to_cpu()
                                except Exception as img_error:
                                    # Corrupted image - log and skip entire batch
                                    print(f"{self.log_prefix} [CORRUPTED IMAGE] Batch skipped due to: {image_path}")
                                    print(f"{self.log_prefix} [CORRUPTED IMAGE] Error: {str(img_error)[:200]}")
                                    # Set flag to skip this batch
                                    batch_has_corrupted_image = True
                                    corrupted_image_path = image_path
                                    break

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
                            try:
                                image = Image.open(item["image_path"])
                                image.load()  # Force load to detect truncated images
                                latent = self.encode_image(
                                    image=image,
                                    target_width=width,
                                    target_height=height,
                                    bucket_strategy=bucket_strategy
                                )
                                latents_list.append(latent)
                            except Exception as img_error:
                                # Corrupted image - log and skip entire batch
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Batch skipped due to: {item['image_path']}")
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Error: {str(img_error)[:200]}")
                                batch_has_corrupted_image = True
                                corrupted_image_path = item["image_path"]
                                break

                        # Encode caption (mode-specific, architecture-unified)
                        caption = item.get("caption", "")

                        if text_encoding_mode == "swap_onthefly":
                            # Get from swap buffer using image_path as key (dict lookup)
                            # This eliminates index-based alignment issues
                            if image_path in swap_buffer:
                                embeddings_cpu, auxiliary_cpu, buffer_caption = swap_buffer[image_path]
                                # Transfer to GPU
                                embeddings = embeddings_cpu.to(self.device, non_blocking=True)
                                auxiliary = auxiliary_cpu.to(self.device, non_blocking=True) if auxiliary_cpu is not None else None
                                text_embeddings_list.append(embeddings)
                                auxiliary_data_list.append(auxiliary)
                                # Override caption from buffer (correct pairing)
                                caption = buffer_caption
                            else:
                                # Fallback to on-the-fly encoding (image not in buffer)
                                print(f"{self.log_prefix} WARNING: Image not in text swap buffer, encoding on-the-fly: {image_path}")
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

                        # ============================================================
                        # Reference Image Latent Encoding (FLUX.2 only)
                        # ============================================================
                        # Note: Only items WITH reference images are conditioned.
                        # If an item has no reference images, we append None to maintain list alignment.
                        # Later, if ANY item in batch has no reference, we skip conditioning for entire batch.
                        #
                        # Multiple reference images per item:
                        # - Each item can have multiple reference images (up to 10)
                        # - reference_latents_list contains List[List[Tensor]] or List[None]
                        # - Each inner list has latents for that item's reference images
                        # - train_step applies T=10, 20, 30... to each reference image
                        if use_reference_images and self.is_flux2:
                            reference_images = item.get("reference_images", [])
                            if reference_images:
                                # Encode all reference images for this item (max 10)
                                item_ref_latents = []
                                for ref_idx, ref_image_path in enumerate(reference_images[:10]):
                                    try:
                                        ref_image = Image.open(ref_image_path)
                                        # Use same bucket dimensions as target image
                                        ref_latent = self.encode_image(
                                            image=ref_image,
                                            target_width=width,
                                            target_height=height,
                                            bucket_strategy=bucket_strategy
                                        )
                                        item_ref_latents.append(ref_latent.to(self.device))
                                    except Exception as e:
                                        print(f"{self.log_prefix} WARNING: Failed to encode reference image {ref_image_path}: {e}")
                                        # Skip this reference image, continue with others
                                        continue

                                if item_ref_latents:
                                    # Successfully encoded at least one reference image
                                    reference_latents_list.append(item_ref_latents)
                                else:
                                    # All reference images failed - mark as None
                                    reference_latents_list.append(None)
                            else:
                                # No reference images for this item - mark as None
                                reference_latents_list.append(None)

                        # ControlNet: Load condition images from reference_images[0]
                        # Condition images stay in pixel space [0, 1] (not VAE-encoded)
                        use_condition_images = getattr(self, 'use_condition_images', False)
                        if use_condition_images:
                            reference_images = item.get("reference_images", [])
                            if reference_images:
                                try:
                                    # Use first reference image only
                                    cond_image = Image.open(reference_images[0]).convert("RGB")
                                    # Resize to match target dimensions
                                    cond_image = cond_image.resize((width, height), Image.LANCZOS)
                                    # Convert to tensor [0, 1] range: [1, 3, H, W]
                                    import torchvision.transforms.functional as TF
                                    cond_tensor = TF.to_tensor(cond_image).unsqueeze(0)  # [1, 3, H, W]
                                    condition_images_list.append(cond_tensor)
                                except Exception as e:
                                    print(f"{self.log_prefix} WARNING: Failed to load condition image {reference_images[0]}: {e}")
                                    condition_images_list.append(None)
                            else:
                                # No reference image - mark as None (will skip this item)
                                condition_images_list.append(None)

                    # Skip batch if corrupted image was detected
                    if batch_has_corrupted_image:
                        print(f"{self.log_prefix} [CORRUPTED IMAGE] Skipping batch due to corrupted image: {corrupted_image_path}")
                        # Cleanup partial lists
                        del latents_list, text_embeddings_list, auxiliary_data_list
                        if reference_latents_list:
                            del reference_latents_list
                        if condition_images_list:
                            del condition_images_list
                        # Update global_step for skipped batch (to maintain step counting)
                        # Each batch would have processed multi_noise_timesteps steps
                        global_step += multi_noise_timesteps
                        continue

                    # Stack batch with size validation
                    # Filter out latents with mismatched spatial dimensions (rare edge case)
                    if len(latents_list) > 1:
                        # Get expected shape from first latent
                        expected_shape = latents_list[0].shape[2:]  # (H, W)
                        valid_indices = []
                        for idx, lat in enumerate(latents_list):
                            if lat.shape[2:] == expected_shape:
                                valid_indices.append(idx)
                            else:
                                print(f"{self.log_prefix} WARNING: Latent size mismatch in batch - expected {expected_shape}, got {lat.shape[2:]}, skipping item")

                        if len(valid_indices) < len(latents_list):
                            # Filter lists to keep only valid items
                            latents_list = [latents_list[i] for i in valid_indices]
                            text_embeddings_list = [text_embeddings_list[i] for i in valid_indices]
                            auxiliary_data_list = [auxiliary_data_list[i] for i in valid_indices]
                            if reference_latents_list:
                                reference_latents_list = [reference_latents_list[i] for i in valid_indices]
                            if condition_images_list:
                                condition_images_list = [condition_images_list[i] for i in valid_indices]

                    # Skip batch if no valid latents remain
                    if len(latents_list) == 0:
                        print(f"{self.log_prefix} WARNING: No valid latents in batch, skipping")
                        continue

                    # Create batch tensors (ONCE, reused across MNT iterations)
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
                            text_embeddings = torch.cat(padded_embeddings, dim=0)  # [batch, seq_len, dim]
                        else:
                            # All same length - direct concatenation
                            text_embeddings = torch.cat(text_embeddings_list, dim=0)  # [batch, seq_len, dim]
                    else:
                        text_embeddings = None

                    # Prepare auxiliary data (attention_mask for Z-Image, pooled_embeddings for SDXL)
                    # These are also reused across MNT iterations
                    attention_mask = None
                    pooled_embeddings = None
                    if self.is_zimage:
                        attention_mask = torch.stack([aux for aux in auxiliary_data_list if aux is not None], dim=0)
                    elif self.is_sdxl and any(aux is not None for aux in auxiliary_data_list):
                        pooled_embeddings = torch.cat([aux for aux in auxiliary_data_list if aux is not None], dim=0)

                    # Prepare reference latents for FLUX.2 conditioning
                    # Only apply conditioning if ALL items in batch have valid reference latents
                    # reference_latents_list is now List[List[Tensor]] or List[None]
                    # We pass the nested structure to train_step which handles T coordinates
                    reference_latents_nested = None
                    if use_reference_images and self.is_flux2 and reference_latents_list:
                        # Check if any item is missing reference latent (None)
                        if all(lat is not None for lat in reference_latents_list):
                            # Pass nested list structure to train_step
                            # train_step will apply T=10, 20, 30... per reference image
                            reference_latents_nested = reference_latents_list
                        else:
                            # Mixed batch (some with, some without reference) - skip conditioning
                            # This ensures consistent training behavior
                            pass

                    # Prepare condition images batch for ControlNet training
                    condition_images_batch = None
                    use_condition_images = getattr(self, 'use_condition_images', False)
                    if use_condition_images and condition_images_list:
                        # Only use batch if ALL items have valid condition images
                        if all(ci is not None for ci in condition_images_list):
                            condition_images_batch = torch.cat(condition_images_list, dim=0)  # [B, 3, H, W]
                        else:
                            # Mixed batch (some without condition images) - skip this batch
                            print(f"{self.log_prefix} WARNING: Some items in batch missing condition images, skipping batch")
                            del latents_list, text_embeddings_list, auxiliary_data_list
                            if reference_latents_list:
                                del reference_latents_list
                            del condition_images_list
                            continue

                    # Free individual item lists (no longer needed, batch tensors are created)
                    del latents_list, text_embeddings_list, auxiliary_data_list
                    if reference_latents_list:
                        del reference_latents_list
                    if condition_images_list:
                        del condition_images_list

                    # Collect batch captions for debug (done once, outside MNT loop)
                    batch_captions = [item.get("caption", "") for item, dataset in batch]

                    # Collect first reference image path per item for debug visualization
                    _ref_paths = [
                        (item.get("reference_images") or [None])[0]
                        for item, dataset in batch
                    ]
                    batch_reference_paths = _ref_paths if any(p is not None for p in _ref_paths) else None

                    batch_size = latents.shape[0]

                    # ============================================================
                    # MNT loop: Process same batch with different noise-timesteps
                    # ============================================================
                    # Sequential MNT Implementation (VRAM optimized):
                    # Each MNT iteration: forward → backward → optimizer.step() → zero_grad()
                    # This prevents gradient accumulation across MNT iterations, keeping
                    # VRAM usage at MNT=1 level regardless of actual MNT value.
                    #
                    # For gradient accumulation across batches, we track accumulated_steps
                    # and only run optimizer.step() when accumulation is complete.
                    #
                    # IMPORTANT: When Text Encoder is trainable AND MNT > 1, we need to
                    # re-encode text embeddings for each MNT iteration to maintain gradient flow.
                    # Otherwise, detach() would cut the gradient to Text Encoder.
                    need_recompute_text_embeddings = (
                        text_encoder_trainable and
                        multi_noise_timesteps > 1 and
                        text_encoding_mode == "onthefly_gpu"
                    )

                    for mnt_idx in range(multi_noise_timesteps):
                        # Sample timesteps for this MNT iteration
                        timesteps = timestep_sampler.sample(batch_size, self.device)

                        # Determine if we should save debug latents (only on first MNT iteration)
                        # With MNT > 1, global_step increments multiple times per batch.
                        # We check if any step within this batch's MNT range hits the debug interval.
                        # Example: MNT=32, debug_every=200
                        #   - Batch 6: steps 192-223, includes step 200 → save at mnt_idx=0
                        #   - Old logic: mnt_idx=0, global_step=192 → 192 % 200 != 0 → NO save (BUG)
                        #   - New logic: mnt_idx=0, check if 192..223 contains a multiple of 200 → YES → save
                        debug_save_path = None
                        if mnt_idx == 0 and debug_dir is not None:
                            # batch_start_step = global_step (current step before MNT loop increments)
                            # batch_end_step = global_step + multi_noise_timesteps - 1 (inclusive)
                            batch_start_step = global_step
                            batch_end_step = global_step + multi_noise_timesteps - 1
                            # Check if any multiple of debug_latents_every falls within [batch_start, batch_end]
                            # This happens when floor(batch_end / every) > floor((batch_start - 1) / every)
                            # Or more simply: batch_start <= k*every <= batch_end for some integer k
                            next_debug_step = ((batch_start_step // debug_latents_every) + 1) * debug_latents_every
                            if batch_start_step % debug_latents_every == 0 or next_debug_step <= batch_end_step:
                                # Determine which step to use for the filename
                                if batch_start_step % debug_latents_every == 0:
                                    save_step = batch_start_step
                                else:
                                    save_step = next_debug_step
                                debug_save_path = debug_dir / f"step_{save_step:06d}"

                        # Detach latents to create fresh computation graph for this MNT iteration
                        # This is necessary because backward() frees the graph
                        mnt_latents = latents.detach()

                        # Handle text embeddings based on training mode
                        if need_recompute_text_embeddings:
                            # Text Encoder trainable + MNT > 1: Re-encode text for each iteration
                            # This creates a fresh computation graph with gradient flow to Text Encoder
                            mnt_text_embeddings_list = []
                            mnt_auxiliary_data_list = []
                            for caption in batch_captions:
                                embeddings, auxiliary = self.encode_caption(caption, requires_grad=True)
                                mnt_text_embeddings_list.append(embeddings)
                                mnt_auxiliary_data_list.append(auxiliary)

                            # Stack embeddings (handle variable sequence lengths)
                            seq_lengths = [emb.shape[1] for emb in mnt_text_embeddings_list]
                            max_seq_len = max(seq_lengths)
                            if len(set(seq_lengths)) > 1:
                                padded_embeddings = []
                                for emb in mnt_text_embeddings_list:
                                    if emb.shape[1] < max_seq_len:
                                        pad_length = max_seq_len - emb.shape[1]
                                        padding = torch.zeros(
                                            (emb.shape[0], pad_length, emb.shape[2]),
                                            dtype=emb.dtype, device=emb.device
                                        )
                                        emb = torch.cat([emb, padding], dim=1)
                                    padded_embeddings.append(emb)
                                mnt_text_embeddings = torch.cat(padded_embeddings, dim=0)
                            else:
                                mnt_text_embeddings = torch.cat(mnt_text_embeddings_list, dim=0)

                            # Prepare auxiliary data
                            if self.is_zimage:
                                mnt_attention_mask = torch.stack([aux for aux in mnt_auxiliary_data_list if aux is not None], dim=0)
                                mnt_pooled_embeddings = None
                            elif self.is_sdxl and any(aux is not None for aux in mnt_auxiliary_data_list):
                                mnt_pooled_embeddings = torch.cat([aux for aux in mnt_auxiliary_data_list if aux is not None], dim=0)
                                mnt_attention_mask = None
                            else:
                                mnt_attention_mask = None
                                mnt_pooled_embeddings = None

                            del mnt_text_embeddings_list, mnt_auxiliary_data_list
                        else:
                            # MNT == 1 or Text Encoder frozen: Use pre-computed embeddings
                            # Detach to prevent "backward through graph twice" error
                            mnt_text_embeddings = text_embeddings.detach() if text_embeddings is not None else None
                            mnt_attention_mask = attention_mask.detach() if attention_mask is not None else None
                            mnt_pooled_embeddings = pooled_embeddings.detach() if pooled_embeddings is not None else None

                        # === Vision Encoder: per-item encoding (SD1.5/SDXL only) ===
                        # Each batch item is conditioned on its own reference image only.
                        # Batches without any reference images skip VE entirely.
                        # When train_vision_encoder=True, VE is already on GPU (moved at training start)
                        # and stays there for the entire training — no per-batch offloading.
                        # When train_vision_encoder=False, VE is moved to GPU for encoding and back to CPU after.
                        ve_obj = getattr(self, 'vision_encoder', None)
                        if ve_obj is not None and mnt_text_embeddings is not None and not self.is_flux2 and not self.is_zimage:
                            train_ve = getattr(self, '_train_vision_encoder', False)
                            ref_paths = [_item.get("reference_images", [None])[0] for _item, _ in batch]
                            batch_has_ref = any(p is not None for p in ref_paths)
                            # Gradient Routing: block gradient flow to TE when batch has reference images,
                            # allowing U-net cross-attention K,V projections to learn VE's feature subspace.
                            if getattr(self, '_gradient_routing_ve', False) and batch_has_ref:
                                mnt_text_embeddings = mnt_text_embeddings.detach()
                                if mnt_pooled_embeddings is not None:
                                    mnt_pooled_embeddings = mnt_pooled_embeddings.detach()
                            # VE Reconstruction Mode: zero text embeddings for items that use their own
                            # image as reference. Mask broadcasts over sequence dim (handles chunking).
                            _ve_recon_mask = [bool(_item.get("_ve_reconstruction_mode")) for _item, _ in batch]
                            if any(_ve_recon_mask) and mnt_text_embeddings is not None:
                                _mask = torch.tensor(
                                    _ve_recon_mask,
                                    dtype=mnt_text_embeddings.dtype,
                                    device=mnt_text_embeddings.device,
                                ).view(-1, 1, 1)  # [B, 1, 1] broadcasts over [B, seq_len, dim]
                                mnt_text_embeddings = mnt_text_embeddings * (1.0 - _mask)
                                if mnt_pooled_embeddings is not None:
                                    _mask_p = _mask.view(-1, 1)  # [B, 1] for pooled embedding
                                    mnt_pooled_embeddings = mnt_pooled_embeddings * (1.0 - _mask_p)
                            if batch_has_ref:
                                try:
                                    ve_obj.to(self.device)
                                    ve_obj.train(train_ve)
                                    target_dim = mnt_text_embeddings.shape[-1]
                                    ve_pos_list = []
                                    for _ref_path in ref_paths:
                                        if _ref_path is not None:
                                            _pil = Image.open(_ref_path).convert("RGB")
                                            # with_grad=True keeps gradients flowing through VE for training;
                                            # with_grad=False (default) wraps in torch.no_grad() for inference.
                                            _ve_pos_i, _ = ve_obj.encode(
                                                [_pil],
                                                target_dim=target_dim,
                                                dtype=self.training_dtype,
                                                with_grad=train_ve,
                                            )
                                            ve_pos_list.append(_ve_pos_i.to(self.device))  # [1, 257, dim]
                                    if ve_pos_list:
                                        # Stack per-item embeddings: [B, 257, dim]
                                        ve_pos_batch = torch.cat(ve_pos_list, dim=0)
                                        mnt_text_embeddings = torch.cat([mnt_text_embeddings, ve_pos_batch], dim=1)
                                    if not train_ve:
                                        # No gradients needed — offload immediately
                                        ve_obj.to("cpu")
                                        torch.cuda.empty_cache()
                                except Exception as _ve_err:
                                    print(f"{self.log_prefix} WARNING: VE encoding failed: {_ve_err}, skipping VE conditioning")
                                    try:
                                        ve_obj.to("cpu")
                                    except Exception:
                                        pass

                        # Training step with OOM recovery (forward + backward)
                        # If OOM occurs, the batch is automatically split and processed sequentially
                        # Wrap in try-except as final safety net - if all recovery fails, skip batch
                        cuda_error_skip = False  # Flag to skip optimizer step when CUDA is in bad state
                        try:
                            mnt_loss_value, mnt_pred_loss_value, mnt_recon_loss_value, cuda_error_skip = self._forward_backward_with_oom_recovery(
                                mnt_latents=mnt_latents,
                                mnt_text_embeddings=mnt_text_embeddings,
                                mnt_attention_mask=mnt_attention_mask,
                                mnt_pooled_embeddings=mnt_pooled_embeddings,
                                timesteps=timesteps,
                                debug_save_path=debug_save_path,
                                batch_captions=batch_captions,
                                batch_reference_paths=batch_reference_paths,
                                alphas_cumprod_cached=alphas_cumprod_cached,
                                use_condition_images=use_condition_images,
                                condition_images_batch=condition_images_batch,
                                reference_latents_nested=reference_latents_nested,
                                min_split_batch_size=1,
                            )
                        except Exception as batch_error:
                            # Final safety net: if all OOM recovery attempts failed,
                            # skip this batch and continue training
                            error_str = str(batch_error).lower()
                            is_cuda_error = (
                                "out of memory" in error_str or
                                "cuda error" in error_str or
                                "cublas" in error_str or
                                "cudnn" in error_str
                            )
                            if is_cuda_error:
                                print(f"{self.log_prefix} [FATAL CUDA Error] All recovery attempts failed, SKIPPING BATCH")
                                print(f"{self.log_prefix} [FATAL CUDA Error] {str(batch_error)[:200]}")
                                # Set flag to skip optimizer step - CUDA is in bad state
                                cuda_error_skip = True
                                # Aggressive cleanup
                                try:
                                    self.optimizer.zero_grad(set_to_none=True)
                                except Exception as e:
                                    print(f"{self.log_prefix} [FATAL CUDA Error] zero_grad failed: {e}")
                                gc.collect()
                                try:
                                    torch.cuda.synchronize()
                                except Exception:
                                    pass
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                # Skip this batch with zero loss
                                mnt_loss_value, mnt_pred_loss_value, mnt_recon_loss_value = 0.0, 0.0, 0.0
                            else:
                                # Non-CUDA error - re-raise
                                raise


                        # Clear MNT iteration tensors (backward already done in helper)
                        del mnt_latents, mnt_text_embeddings
                        if mnt_attention_mask is not None:
                            del mnt_attention_mask
                        if mnt_pooled_embeddings is not None:
                            del mnt_pooled_embeddings

                        # Clear saved activations immediately after backward to prevent VRAM leaks
                        if hasattr(self, 'layer_offload_conductor') and self.layer_offload_conductor is not None:
                            self.layer_offload_conductor.clear_activations()

                        # FLUX.2: Clear block swap activations
                        if hasattr(self, 'flux2_block_offloader') and self.flux2_block_offloader is not None:
                            self.flux2_block_offloader.clear_activations()

                        # Increment global step for each MNT iteration
                        global_step += 1

                        # ============================================================
                        # Per-MNT-iteration logging (for real-time frontend updates)
                        # ============================================================
                        # Log loss immediately for each MNT iteration so frontend
                        # updates every step, not just every MNT*grad_accum steps.
                        # Grad norm will be updated after optimizer step.
                        # Note: mnt_loss_value, mnt_pred_loss_value, mnt_recon_loss_value
                        # are already extracted as floats by _forward_backward_with_oom_recovery()
                        mnt_current_lr = self.lr_scheduler.get_last_lr()[0]

                        # TensorBoard logging (per-iteration for loss only)
                        self.writer.add_scalar("train/loss", mnt_loss_value, global_step)
                        self.writer.add_scalar("train/pred_loss", mnt_pred_loss_value, global_step)
                        self.writer.add_scalar("train/recon_loss", mnt_recon_loss_value, global_step)
                        self.writer.add_scalar("train/lr", mnt_current_lr, global_step)

                        # Database logging (per-iteration, loss only - grad_norm logged at optimizer step)
                        # Grad norm is only available after optimizer step, so we don't log it here.
                        # This prevents grad_norm=0 from corrupting smoothed grad norm charts.
                        if self.run_id is not None:
                            self._log_metrics_to_db(
                                step=global_step,
                                loss=mnt_pred_loss_value,
                                recon_loss=mnt_recon_loss_value,
                                learning_rate=mnt_current_lr,
                                grad_norm=None,  # Don't set - will be updated after optimizer step
                                grad_norm_text_encoder=None,
                                grad_norm_unet=None
                            )

                        # Progress callback (per-iteration for real-time UI updates)
                        if progress_callback:
                            progress_callback(
                                phase="training",
                                step=global_step,
                                total=actual_total_steps,
                                epoch=epoch,
                                loss=mnt_loss_value,
                            )

                        # ============================================================
                        # Sequential MNT: Optimizer step after each MNT iteration
                        # ============================================================
                        # This prevents gradient accumulation across MNT iterations,
                        # keeping VRAM at MNT=1 level.
                        #
                        # Key insight: Each MNT iteration is treated as an independent
                        # training step. Gradient accumulation (if configured) happens
                        # across these MNT steps, not across batches.
                        #
                        # global_step = (batch_idx * multi_noise_timesteps) + (mnt_idx + 1)
                        # We step optimizer when global_step is divisible by gradient_accumulation_steps
                        #
                        # IMPORTANT: Skip optimizer step if CUDA error occurred and batch was skipped.
                        # When CUDA is in bad state, grad_scaler.unscale_() will fail.
                        should_step_optimizer = (global_step % gradient_accumulation_steps == 0)

                        if cuda_error_skip:
                            # CUDA error occurred - skip optimizer step entirely
                            # The batch was skipped, so there are no valid gradients to step with
                            print(f"{self.log_prefix} [CUDA Recovery] Skipping optimizer step (batch was skipped)")
                            grad_norm_total, grad_norm_te, grad_norm_unet, grad_norm_ve = 0.0, 0.0, 0.0, 0.0
                            # Still step LR scheduler to keep it in sync with global_step
                            if should_step_optimizer:
                                try:
                                    if self.fused_optimizer_groups is not None:
                                        for lr_scheduler in self.lr_schedulers:
                                            lr_scheduler.step()
                                    else:
                                        self.lr_scheduler.step()
                                except Exception as lr_err:
                                    print(f"{self.log_prefix} [CUDA Recovery] LR scheduler step failed: {lr_err}")
                        elif should_step_optimizer:
                            if not self.use_fused_backward and self.fused_optimizer_groups is None:
                                # Normal flow: optimizer.step() and zero_grad() here
                                if self.use_grad_scaler:
                                    # GradScaler flow
                                    self.grad_scaler.unscale_(self.optimizer)
                                    grad_norm_total, grad_norm_te, grad_norm_te1, grad_norm_te2, grad_norm_unet, grad_norm_ve = self._calculate_grad_norms()
                                    if max_grad_norm > 0:
                                        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_grad_norm)
                                    self.grad_scaler.step(self.optimizer)
                                    self.grad_scaler.update()
                                    self.optimizer.zero_grad()
                                else:
                                    # Normal flow without GradScaler
                                    grad_norm_total, grad_norm_te, grad_norm_te1, grad_norm_te2, grad_norm_unet, grad_norm_ve = self._calculate_grad_norms()
                                    if max_grad_norm > 0:
                                        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_grad_norm)
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()
                            else:
                                # Fused backward/groups flow - calculate grad norm (step/zero_grad by hooks)
                                grad_norm_total, grad_norm_te, grad_norm_te1, grad_norm_te2, grad_norm_unet, grad_norm_ve = self._calculate_grad_norms()

                            # LR scheduler step
                            if self.fused_optimizer_groups is not None:
                                for lr_scheduler in self.lr_schedulers:
                                    lr_scheduler.step()
                            else:
                                self.lr_scheduler.step()

                            # Log grad_norm to TensorBoard
                            self.writer.add_scalar("train/grad_norm", grad_norm_total, global_step)
                            if grad_norm_te > 0.0:
                                self.writer.add_scalar("train/grad_norm_text_encoder", grad_norm_te, global_step)
                            self.writer.add_scalar("train/grad_norm_unet", grad_norm_unet, global_step)
                            if grad_norm_te1 > 0.0:
                                self.writer.add_scalar("train/grad_norm_text_encoder_1", grad_norm_te1, global_step)
                            if grad_norm_te2 > 0.0:
                                self.writer.add_scalar("train/grad_norm_text_encoder_2", grad_norm_te2, global_step)
                            if grad_norm_ve > 0.0:
                                self.writer.add_scalar("train/grad_norm_vision_encoder", grad_norm_ve, global_step)

                            # Update grad_norm in database
                            if self.run_id is not None:
                                self._log_metrics_to_db(
                                    step=global_step,
                                    loss=None,
                                    recon_loss=None,
                                    learning_rate=None,
                                    grad_norm=grad_norm_total,
                                    grad_norm_text_encoder=grad_norm_te if grad_norm_te > 0.0 else None,
                                    grad_norm_text_encoder_1=grad_norm_te1 if grad_norm_te1 > 0.0 else None,
                                    grad_norm_text_encoder_2=grad_norm_te2 if grad_norm_te2 > 0.0 else None,
                                    grad_norm_unet=grad_norm_unet,
                                    grad_norm_vision_encoder=grad_norm_ve if grad_norm_ve > 0.0 else None,
                                )

                            # Parameter change tracking (B: update norm, C: cumulative drift)
                            if self._param_tracker is not None:
                                pt = self._param_tracker.compute(global_step)
                                if pt is not None:
                                    un = pt['update_norm']
                                    cd = pt['cumulative_drift']
                                    for name, val in un.items():
                                        self.writer.add_scalar(f"param/update_norm_{name}", val, global_step)
                                    for name, val in cd.items():
                                        self.writer.add_scalar(f"param/cumulative_drift_{name}", val, global_step)
                                    if self.run_id is not None:
                                        self._log_metrics_to_db(
                                            step=global_step,
                                            param_update_norm_unet=un.get('unet'),
                                            param_update_norm_te1=un.get('te1'),
                                            param_update_norm_te2=un.get('te2'),
                                            param_update_norm_ve=un.get('ve'),
                                            param_cumulative_drift_unet=cd.get('unet'),
                                            param_cumulative_drift_te1=cd.get('te1'),
                                            param_cumulative_drift_te2=cd.get('te2'),
                                            param_cumulative_drift_ve=cd.get('ve'),
                                        )

                            # ReLoRA merge-reinit cycle hook
                            # Only active for ReLoRATrainer (has should_merge method)
                            if hasattr(self, 'should_merge'):
                                is_first_batch = (batch_idx == 0 and mnt_idx == 0)
                                if self.should_merge(global_step, epoch, is_first_batch):
                                    self.perform_merge_reinit_cycle(global_step, epoch)

                        # Force CUDA memory cleanup between MNT iterations to prevent
                        # VRAM fragmentation and accumulation. Skip on last iteration
                        # since batch cleanup follows immediately.
                        if multi_noise_timesteps > 1 and mnt_idx < multi_noise_timesteps - 1:
                            torch.cuda.empty_cache()

                    # Free batch tensors AFTER all MNT iterations complete
                    del latents, text_embeddings
                    if attention_mask is not None:
                        del attention_mask
                    if pooled_embeddings is not None:
                        del pooled_embeddings
                    if reference_latents_nested is not None:
                        del reference_latents_nested

                    # ============================================================
                    # Post-batch processing (Sequential MNT: optimizer step done in loop)
                    # ============================================================
                    # With Sequential MNT, optimizer.step() is called inside the MNT loop
                    # after each MNT iteration. Here we only handle:
                    # - TensorBoard flushing
                    # - Checkpoint saving
                    # - Sample generation

                    # Flush TensorBoard writer periodically to prevent DRAM accumulation
                    # (TensorBoard buffers events internally, can accumulate GBs over long training)
                    if global_step % 100 == 0:
                        self.writer.flush()
                        # Also clear CUDA cache to prevent fragmented memory accumulation
                        torch.cuda.empty_cache()

                    # Save checkpoint (check against global_step which increments per MNT iteration)
                    if global_step % save_every_n_steps == 0:
                        # Flush metrics buffer before checkpoint to ensure consistency
                        if self.run_id is not None:
                            self._log_metrics_to_db(step=global_step, force_flush=True)
                        self.save_checkpoint(step=global_step, epoch=epoch)
                        # Save training state (epoch progress) for mid-epoch resume
                        self.save_training_state(step=global_step, epoch=epoch, batch_idx=batch_idx + 1, multi_noise_timesteps=multi_noise_timesteps)
                        # Save optimizer state (momentum, variance, etc.)
                        self.save_optimizer_state(step=global_step)
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
                    # With MNT > 1, check if any step in the batch's MNT range contains a sample interval
                    # batch range: [global_step - multi_noise_timesteps + 1, global_step] (inclusive)
                    should_generate_sample = False
                    sample_step = global_step  # Default: use current global_step for filename

                    if sample_every_n_steps > 0:
                        batch_start_step = global_step - multi_noise_timesteps + 1
                        batch_end_step = global_step

                        # Check step 0 (only if batch starts at 0)
                        if batch_start_step == 0:
                            should_generate_sample = True
                            sample_step = 0
                        else:
                            # Check if any multiple of sample_every_n_steps falls within [batch_start, batch_end]
                            next_sample_step = ((batch_start_step // sample_every_n_steps) + 1) * sample_every_n_steps
                            if batch_start_step % sample_every_n_steps == 0:
                                should_generate_sample = True
                                sample_step = batch_start_step
                            elif next_sample_step <= batch_end_step:
                                should_generate_sample = True
                                sample_step = next_sample_step

                    if should_generate_sample:
                        import torchvision

                        for sample_idx, prompt_config in enumerate(self._sample_prompts):
                            positive = prompt_config.get('positive', 'a beautiful landscape')
                            condition_image_path = prompt_config.get('condition_image_path') or None
                            reference_image_path = prompt_config.get('reference_image_path') or None

                            print(f"{self.log_prefix} Generating sample {sample_idx} with prompt='{positive[:50]}...', width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, steps={sample_steps}, seed={sample_seed}")
                            if self.is_flux2:
                                sample = self._generate_sample_flux2(
                                    prompt=positive,
                                    width=sample_width,
                                    height=sample_height,
                                    num_inference_steps=sample_steps,
                                    guidance_scale=sample_guidance_scale,
                                    seed=sample_seed,
                                    reference_image_path=reference_image_path,
                                )
                            elif self.is_zimage:
                                sample = self._generate_sample_zimage(
                                    prompt=positive,
                                    width=sample_width,
                                    height=sample_height,
                                    num_inference_steps=sample_steps,
                                    guidance_scale=sample_guidance_scale,
                                    seed=sample_seed
                                )
                            else:
                                sample = self.generate_sample(
                                    prompt=positive,
                                    width=sample_width,
                                    height=sample_height,
                                    num_inference_steps=sample_steps,
                                    guidance_scale=sample_guidance_scale,
                                    seed=sample_seed,
                                    current_step=global_step,
                                    schedule_type=sample_schedule_type,
                                    condition_image_path=condition_image_path,
                                    reference_image_path=reference_image_path,
                                )

                            # Save sample with format matching API expectations: step_{step:06d}_sample_{i}.png
                            # Use sample_step (which accounts for MNT batch range) for consistent naming
                            sample_path = self.output_dir / "samples" / f"step_{sample_step:06d}_sample_{sample_idx}.png"
                            sample_path.parent.mkdir(parents=True, exist_ok=True)

                            # Embed generation metadata in PNG for display in Training Monitor
                            png_metadata = PngImagePlugin.PngInfo()
                            png_metadata.add_text("prompt", positive)
                            png_metadata.add_text("negative_prompt", prompt_config.get('negative', ''))
                            png_metadata.add_text("steps", str(sample_steps))
                            png_metadata.add_text("cfg_scale", str(sample_guidance_scale))
                            png_metadata.add_text("seed", str(sample_seed))
                            png_metadata.add_text("width", str(sample_width))
                            png_metadata.add_text("height", str(sample_height))
                            png_metadata.add_text("schedule_type", sample_schedule_type)
                            if condition_image_path:
                                png_metadata.add_text("condition_image_path", condition_image_path)
                            if reference_image_path:
                                png_metadata.add_text("reference_image_path", reference_image_path)
                            sample.save(sample_path, pnginfo=png_metadata)
                            print(f"{self.log_prefix} Saved sample to {sample_path}")

                            # Log to TensorBoard
                            image_tensor = torchvision.transforms.ToTensor()(sample)
                            self.writer.add_image(f"samples/sample_{sample_idx}", image_tensor, global_step=sample_step)

                            # Free sample-related tensors
                            del sample, image_tensor

                        torch.cuda.empty_cache()

                        # onthefly_gpu mode: Restore text encoders to GPU after sample generation
                        if text_encoding_mode == "onthefly_gpu":
                            self.move_text_encoder_to_gpu()

                    # Note: Progress callback is now called per-MNT-iteration (above)
                    # for real-time frontend updates during MNT training.

                    # Check if total_steps reached
                    # Use actual_total_steps (which may be recalculated on MNT change during resume)
                    if global_step >= actual_total_steps:
                        print(f"\n{self.log_prefix} Reached target steps ({actual_total_steps}), stopping training")
                        return  # Exit training loop

                    # Note: With Sequential MNT, optimizer.step() and loss deletion
                    # are handled inside the MNT loop. No else clause needed here.

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
                self.save_training_state(step=global_step, epoch=epoch, batch_idx=batch_idx + 1, multi_noise_timesteps=multi_noise_timesteps)
                state_saved = True
                print(f"{self.log_prefix} Training state saved successfully")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to save training state: {e}")

            # Try to save optimizer state (independent of checkpoint/state save)
            optimizer_saved = False
            try:
                self.save_optimizer_state(step=global_step)
                optimizer_saved = True
                print(f"{self.log_prefix} Optimizer state saved successfully")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to save optimizer state: {e}")
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

        except Exception as e:
            # Emergency checkpoint save on any unhandled exception (CUDA errors, etc.)
            print(f"\n{self.log_prefix} [EMERGENCY] Training failed with error: {type(e).__name__}: {str(e)[:200]}")
            print(f"{self.log_prefix} [EMERGENCY] Attempting to save emergency checkpoint at step {global_step}, epoch {epoch}...")

            # For CUDA errors, first try to move model to CPU to free GPU memory
            try:
                print(f"{self.log_prefix} [EMERGENCY] Moving model to CPU to free GPU memory...")
                self.move_main_model_to_cpu()
                self.move_text_encoder_to_cpu()
                self.move_vae_to_cpu()
            except Exception as move_error:
                print(f"{self.log_prefix} [EMERGENCY] Failed to move model to CPU: {move_error}")

            # Try to clear CUDA cache (may fail if context is corrupted)
            try:
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass  # Ignore - CUDA may be in bad state

            # Try to save checkpoint
            checkpoint_saved = False
            try:
                self.save_checkpoint(step=global_step, epoch=epoch)
                checkpoint_saved = True
                print(f"{self.log_prefix} [EMERGENCY] Checkpoint saved successfully")
            except Exception as save_error:
                print(f"{self.log_prefix} [EMERGENCY] Failed to save checkpoint: {save_error}")
                import traceback
                traceback.print_exc()

            # Try to save training state
            state_saved = False
            try:
                self.save_training_state(step=global_step, epoch=epoch, batch_idx=batch_idx + 1, multi_noise_timesteps=multi_noise_timesteps)
                state_saved = True
                print(f"{self.log_prefix} [EMERGENCY] Training state saved successfully")
            except Exception as state_error:
                print(f"{self.log_prefix} [EMERGENCY] Failed to save training state: {state_error}")

            # Try to save optimizer state
            optimizer_saved = False
            try:
                self.save_optimizer_state(step=global_step)
                optimizer_saved = True
                print(f"{self.log_prefix} [EMERGENCY] Optimizer state saved successfully")
            except Exception as opt_error:
                print(f"{self.log_prefix} [EMERGENCY] Failed to save optimizer state: {opt_error}")

            # Summary
            if checkpoint_saved or state_saved or optimizer_saved:
                saved_items = []
                if checkpoint_saved:
                    saved_items.append("checkpoint")
                if state_saved:
                    saved_items.append("state")
                if optimizer_saved:
                    saved_items.append("optimizer")
                print(f"{self.log_prefix} [EMERGENCY] Saved: {', '.join(saved_items)}")
                print(f"{self.log_prefix} [EMERGENCY] Training can be resumed from step {global_step}")
            else:
                print(f"{self.log_prefix} [EMERGENCY] WARNING: All save attempts failed!")
                print(f"{self.log_prefix} [EMERGENCY] Training progress may be lost")

            self.writer.close()
            raise  # Re-raise the original exception

        print(f"{self.log_prefix} Training complete!")

        # Cleanup resources
        self.cleanup()

    def _calculate_grad_norms(self):
        """
        Calculate gradient norms for different parameter groups.

        Returns:
            Tuple of (total_grad_norm, text_encoder_grad_norm, text_encoder_1_grad_norm,
                      text_encoder_2_grad_norm, unet_grad_norm, vision_encoder_grad_norm)
            text_encoder_1/2 are non-zero only for SDXL (LoRA: te1_/te2_ prefix; Full FT: text_encoder/text_encoder_2).
        """
        total_grad_norm = 0.0
        text_encoder_grad_norm = 0.0
        text_encoder_1_grad_norm = 0.0
        text_encoder_2_grad_norm = 0.0
        unet_grad_norm = 0.0
        vision_encoder_grad_norm = 0.0

        # For LoRA training, iterate through lora_layers dict
        if hasattr(self, 'lora_layers'):
            grad_count = 0
            for lora_name, lora_layer in self.lora_layers.items():
                for param in lora_layer.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                        grad_count += 1

                        # Categorize by LoRA layer name — TE1/TE2 separation for SDXL
                        if 'te1_' in lora_name:
                            text_encoder_grad_norm += param_norm ** 2
                            text_encoder_1_grad_norm += param_norm ** 2
                        elif 'te2_' in lora_name:
                            text_encoder_grad_norm += param_norm ** 2
                            text_encoder_2_grad_norm += param_norm ** 2
                        elif 'text_encoder' in lora_name or 'clip_' in lora_name:
                            # SD1.5 or unknown TE prefix — add to combined only
                            text_encoder_grad_norm += param_norm ** 2
                        elif 'unet' in lora_name or 'transformer' in lora_name or 'dit_' in lora_name:
                            unet_grad_norm += param_norm ** 2

            # Debug: Print first calculation only
            if grad_count > 0 and not hasattr(self, '_grad_norm_debug_printed'):
                print(f"{self.log_prefix} [GradNorm] Calculated from {grad_count} parameters with gradients")
                print(f"{self.log_prefix} [GradNorm] Sample LoRA layer names (first 3):")
                for i, name in enumerate(list(self.lora_layers.keys())[:3]):
                    print(f"{self.log_prefix}   {name}")
                self._grad_norm_debug_printed = True

        # For Full Fine-Tuning, iterate through base model parameters
        else:
            # SD1.5/SDXL: Direct text_encoder access — treat as TE1
            if hasattr(self, 'text_encoder') and self.text_encoder is not None:
                for name, param in self.text_encoder.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                        text_encoder_grad_norm += param_norm ** 2
                        text_encoder_1_grad_norm += param_norm ** 2

            # Iterate through text encoder 2 parameters (if trainable, SDXL) — TE2
            if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
                for name, param in self.text_encoder_2.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                        text_encoder_grad_norm += param_norm ** 2
                        text_encoder_2_grad_norm += param_norm ** 2

            # Iterate through U-Net parameters (if trainable, SD1.5/SDXL)
            if hasattr(self, 'unet') and self.unet is not None:
                for name, param in self.unet.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                        unet_grad_norm += param_norm ** 2

            # Iterate through Transformer parameters (if trainable, Z-Image)
            if hasattr(self, 'transformer_original') and self.transformer_original is not None:
                for name, param in self.transformer_original.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                        unet_grad_norm += param_norm ** 2

            # Iterate through Vision Encoder parameters (if training VE, SD1.5/SDXL only)
            if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
                for param in self.vision_encoder.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                        vision_encoder_grad_norm += param_norm ** 2

        # Take square root to get L2 norm
        total_grad_norm = total_grad_norm ** 0.5
        text_encoder_grad_norm = text_encoder_grad_norm ** 0.5
        text_encoder_1_grad_norm = text_encoder_1_grad_norm ** 0.5
        text_encoder_2_grad_norm = text_encoder_2_grad_norm ** 0.5
        unet_grad_norm = unet_grad_norm ** 0.5
        vision_encoder_grad_norm = vision_encoder_grad_norm ** 0.5

        # Debug: Print values once
        if not hasattr(self, '_grad_norm_values_printed'):
            print(f"{self.log_prefix} [GradNorm] Total: {total_grad_norm:.6f}, TE: {text_encoder_grad_norm:.6f}, TE1: {text_encoder_1_grad_norm:.6f}, TE2: {text_encoder_2_grad_norm:.6f}, UNet: {unet_grad_norm:.6f}, VE: {vision_encoder_grad_norm:.6f}")
            self._grad_norm_values_printed = True

        return total_grad_norm, text_encoder_grad_norm, text_encoder_1_grad_norm, text_encoder_2_grad_norm, unet_grad_norm, vision_encoder_grad_norm

    def _log_metrics_to_db(
        self,
        step: int,
        loss: float = None,
        recon_loss: float = None,
        learning_rate: float = None,
        grad_norm: float = None,
        grad_norm_text_encoder: float = None,
        grad_norm_text_encoder_1: float = None,
        grad_norm_text_encoder_2: float = None,
        grad_norm_unet: float = None,
        grad_norm_vision_encoder: float = None,
        param_update_norm_unet: float = None,
        param_update_norm_te1: float = None,
        param_update_norm_te2: float = None,
        param_update_norm_ve: float = None,
        param_cumulative_drift_unet: float = None,
        param_cumulative_drift_te1: float = None,
        param_cumulative_drift_te2: float = None,
        param_cumulative_drift_ve: float = None,
        force_flush: bool = False
    ):
        """
        Log training metrics to database with buffering (dual logging: TensorBoard + DB).

        OPTIMIZED: Buffers metrics and batch commits every N steps to reduce I/O overhead.
        This reduces DB operations from every step to every _metrics_flush_interval steps.

        Features:
        - UPSERT behavior: Same (run_id, step) will overwrite existing values
        - Allows training restart from checkpoint without duplicating metrics
        - Fast queries: indexed by (run_id, step) for incremental fetching
        - Partial update: If a parameter is None, existing value is preserved
        - Buffered commits: Batches DB writes for performance

        Args:
            step: Global training step
            loss: Prediction loss value (MSE with Min-SNR weighting), None to keep existing
            recon_loss: Reconstruction loss value, None to keep existing
            learning_rate: Current learning rate, None to keep existing
            grad_norm: Total gradient norm, None to keep existing
            grad_norm_text_encoder: Text encoder gradient norm, None to keep existing
            grad_norm_unet: U-Net/Transformer gradient norm, None to keep existing
            grad_norm_vision_encoder: Vision Encoder gradient norm, None to keep existing
            force_flush: If True, flush buffer immediately (for checkpoints, end of training)

        Note:
            The 'loss' parameter stores prediction loss (not combined loss).
            This allows monitoring pred_loss and recon_loss separately in DB.
            Combined loss can be calculated as: (1-β)*loss + β*recon_loss
        """
        # Buffer the metrics (merge if same step already exists in buffer)
        # This handles the case where loss and grad_norm are logged separately for the same step
        existing_entry = None
        for entry in self._metrics_buffer:
            if entry['step'] == step:
                existing_entry = entry
                break

        if existing_entry is not None:
            # Merge: update existing entry with new non-None values
            if loss is not None:
                existing_entry['loss'] = loss
            if recon_loss is not None:
                existing_entry['recon_loss'] = recon_loss
            if learning_rate is not None:
                existing_entry['learning_rate'] = learning_rate
            if grad_norm is not None:
                existing_entry['grad_norm'] = grad_norm
            if grad_norm_text_encoder is not None:
                existing_entry['grad_norm_text_encoder'] = grad_norm_text_encoder
            if grad_norm_text_encoder_1 is not None:
                existing_entry['grad_norm_text_encoder_1'] = grad_norm_text_encoder_1
            if grad_norm_text_encoder_2 is not None:
                existing_entry['grad_norm_text_encoder_2'] = grad_norm_text_encoder_2
            if grad_norm_unet is not None:
                existing_entry['grad_norm_unet'] = grad_norm_unet
            if grad_norm_vision_encoder is not None:
                existing_entry['grad_norm_vision_encoder'] = grad_norm_vision_encoder
            if param_update_norm_unet is not None:
                existing_entry['param_update_norm_unet'] = param_update_norm_unet
            if param_update_norm_te1 is not None:
                existing_entry['param_update_norm_te1'] = param_update_norm_te1
            if param_update_norm_te2 is not None:
                existing_entry['param_update_norm_te2'] = param_update_norm_te2
            if param_update_norm_ve is not None:
                existing_entry['param_update_norm_ve'] = param_update_norm_ve
            if param_cumulative_drift_unet is not None:
                existing_entry['param_cumulative_drift_unet'] = param_cumulative_drift_unet
            if param_cumulative_drift_te1 is not None:
                existing_entry['param_cumulative_drift_te1'] = param_cumulative_drift_te1
            if param_cumulative_drift_te2 is not None:
                existing_entry['param_cumulative_drift_te2'] = param_cumulative_drift_te2
            if param_cumulative_drift_ve is not None:
                existing_entry['param_cumulative_drift_ve'] = param_cumulative_drift_ve
        else:
            # New step: add to buffer
            self._metrics_buffer.append({
                'step': step,
                'loss': loss,
                'recon_loss': recon_loss,
                'learning_rate': learning_rate,
                'grad_norm': grad_norm,
                'grad_norm_text_encoder': grad_norm_text_encoder,
                'grad_norm_text_encoder_1': grad_norm_text_encoder_1,
                'grad_norm_text_encoder_2': grad_norm_text_encoder_2,
                'grad_norm_unet': grad_norm_unet,
                'grad_norm_vision_encoder': grad_norm_vision_encoder,
                'param_update_norm_unet': param_update_norm_unet,
                'param_update_norm_te1': param_update_norm_te1,
                'param_update_norm_te2': param_update_norm_te2,
                'param_update_norm_ve': param_update_norm_ve,
                'param_cumulative_drift_unet': param_cumulative_drift_unet,
                'param_cumulative_drift_te1': param_cumulative_drift_te1,
                'param_cumulative_drift_te2': param_cumulative_drift_te2,
                'param_cumulative_drift_ve': param_cumulative_drift_ve,
            })

        # Only flush when buffer is full or force_flush is requested
        should_flush = force_flush or len(self._metrics_buffer) >= self._metrics_flush_interval
        if not should_flush:
            return

        # Copy buffer and clear immediately (so training can continue adding to new buffer)
        buffer_to_flush = self._metrics_buffer.copy()
        self._metrics_buffer = []

        if force_flush:
            # Synchronous flush for checkpoints/end of training (ensure data is written)
            self._flush_metrics_to_db(buffer_to_flush)
        else:
            # Async flush: submit to background thread, don't block training
            # Clean up completed futures first
            self._db_futures = [f for f in self._db_futures if not f.done()]
            future = self._db_executor.submit(self._flush_metrics_to_db, buffer_to_flush)
            self._db_futures.append(future)

    def _flush_metrics_to_db(self, buffer: list):
        """
        Actually flush metrics buffer to database (runs in background thread).

        Args:
            buffer: List of metrics dicts to flush
        """
        if not buffer:
            return

        try:
            from database.models import TrainingMetrics
            from database import get_training_db

            # Get database session
            db = next(get_training_db())

            for metrics in buffer:
                m_step = metrics['step']
                m_loss = metrics['loss']
                m_recon_loss = metrics['recon_loss']
                m_learning_rate = metrics['learning_rate']
                m_grad_norm = metrics['grad_norm']
                m_grad_norm_te = metrics['grad_norm_text_encoder']
                m_grad_norm_te1 = metrics.get('grad_norm_text_encoder_1')
                m_grad_norm_te2 = metrics.get('grad_norm_text_encoder_2')
                m_grad_norm_unet = metrics['grad_norm_unet']
                m_grad_norm_ve = metrics.get('grad_norm_vision_encoder')
                m_param_upd_unet = metrics.get('param_update_norm_unet')
                m_param_upd_te1  = metrics.get('param_update_norm_te1')
                m_param_upd_te2  = metrics.get('param_update_norm_te2')
                m_param_upd_ve   = metrics.get('param_update_norm_ve')
                m_param_dft_unet = metrics.get('param_cumulative_drift_unet')
                m_param_dft_te1  = metrics.get('param_cumulative_drift_te1')
                m_param_dft_te2  = metrics.get('param_cumulative_drift_te2')
                m_param_dft_ve   = metrics.get('param_cumulative_drift_ve')

                # UPSERT: Check if metric exists for this (run_id, step)
                existing = db.query(TrainingMetrics).filter(
                    TrainingMetrics.run_id == self.run_id,
                    TrainingMetrics.step == m_step
                ).first()

                if existing:
                    # Update existing metric (training restarted from checkpoint)
                    if m_loss is not None:
                        existing.loss = m_loss
                    if m_recon_loss is not None:
                        existing.recon_loss = m_recon_loss
                    if m_learning_rate is not None:
                        existing.learning_rate = m_learning_rate
                    if m_grad_norm is not None:
                        existing.grad_norm = m_grad_norm
                    if m_grad_norm_te is not None:
                        existing.grad_norm_text_encoder = m_grad_norm_te
                    if m_grad_norm_te1 is not None:
                        existing.grad_norm_text_encoder_1 = m_grad_norm_te1
                    if m_grad_norm_te2 is not None:
                        existing.grad_norm_text_encoder_2 = m_grad_norm_te2
                    if m_grad_norm_unet is not None:
                        existing.grad_norm_unet = m_grad_norm_unet
                    if m_grad_norm_ve is not None:
                        existing.grad_norm_vision_encoder = m_grad_norm_ve
                    if m_param_upd_unet is not None:
                        existing.param_update_norm_unet = m_param_upd_unet
                    if m_param_upd_te1 is not None:
                        existing.param_update_norm_te1 = m_param_upd_te1
                    if m_param_upd_te2 is not None:
                        existing.param_update_norm_te2 = m_param_upd_te2
                    if m_param_upd_ve is not None:
                        existing.param_update_norm_ve = m_param_upd_ve
                    if m_param_dft_unet is not None:
                        existing.param_cumulative_drift_unet = m_param_dft_unet
                    if m_param_dft_te1 is not None:
                        existing.param_cumulative_drift_te1 = m_param_dft_te1
                    if m_param_dft_te2 is not None:
                        existing.param_cumulative_drift_te2 = m_param_dft_te2
                    if m_param_dft_ve is not None:
                        existing.param_cumulative_drift_ve = m_param_dft_ve
                    existing.timestamp = datetime.now()
                else:
                    # Insert new metric
                    metric = TrainingMetrics(
                        run_id=self.run_id,
                        step=m_step,
                        loss=m_loss if m_loss is not None else 0.0,
                        recon_loss=m_recon_loss if m_recon_loss is not None else 0.0,
                        learning_rate=m_learning_rate if m_learning_rate is not None else 0.0,
                        grad_norm=m_grad_norm,
                        grad_norm_text_encoder=m_grad_norm_te,
                        grad_norm_text_encoder_1=m_grad_norm_te1,
                        grad_norm_text_encoder_2=m_grad_norm_te2,
                        grad_norm_unet=m_grad_norm_unet,
                        grad_norm_vision_encoder=m_grad_norm_ve,
                        param_update_norm_unet=m_param_upd_unet,
                        param_update_norm_te1=m_param_upd_te1,
                        param_update_norm_te2=m_param_upd_te2,
                        param_update_norm_ve=m_param_upd_ve,
                        param_cumulative_drift_unet=m_param_dft_unet,
                        param_cumulative_drift_te1=m_param_dft_te1,
                        param_cumulative_drift_te2=m_param_dft_te2,
                        param_cumulative_drift_ve=m_param_dft_ve,
                    )
                    db.add(metric)

            # Single commit for entire buffer
            db.commit()
            db.close()

            # Broadcast latest metrics to WebSocket clients
            # Only send the most recent entry to avoid flooding
            if buffer:
                latest = buffer[-1]
                try:
                    from api.websocket import manager as ws_manager
                    ws_manager.send_training_metrics(
                        run_id=self.run_id,
                        step=latest['step'],
                        loss=latest['loss'],
                        recon_loss=latest['recon_loss'],
                        learning_rate=latest['learning_rate'],
                        grad_norm=latest['grad_norm'],
                        grad_norm_text_encoder=latest['grad_norm_text_encoder'],
                        grad_norm_text_encoder_1=latest.get('grad_norm_text_encoder_1'),
                        grad_norm_text_encoder_2=latest.get('grad_norm_text_encoder_2'),
                        grad_norm_unet=latest['grad_norm_unet'],
                        grad_norm_vision_encoder=latest.get('grad_norm_vision_encoder'),
                    )
                except Exception:
                    pass  # Non-critical

        except Exception as e:
            # Non-critical: Continue training even if DB logging fails
            print(f"{self.log_prefix} WARNING: Failed to log metrics to DB: {e}")

    def _shutdown_db_executor(self):
        """Shutdown the DB executor and wait for pending writes to complete."""
        if hasattr(self, '_db_executor') and self._db_executor is not None:
            # Wait for all pending futures
            from concurrent.futures import wait
            if self._db_futures:
                wait(self._db_futures, timeout=30)  # Wait up to 30 seconds
            self._db_executor.shutdown(wait=True)
            self._db_executor = None

    def _cleanup_future_metrics(self, current_step: int):
        """
        Clean up future metrics in database (old data from previous interrupted training).

        When training resumes from an earlier step (e.g., resume from step 100 when previous
        run reached step 500), the UPSERT logic will overwrite steps 1-100, but steps 101-500
        from the old run will remain in the database, causing duplicate/stale data.

        This method removes all metrics with step > current_step to prevent this issue.

        Args:
            current_step: Current global step (resume point)
        """
        try:
            from database.models import TrainingMetrics
            from database import get_training_db

            # Get database session
            db = next(get_training_db())

            # Find future metrics (step > current_step)
            future_metrics = db.query(TrainingMetrics).filter(
                TrainingMetrics.run_id == self.run_id,
                TrainingMetrics.step > current_step
            ).all()

            if future_metrics:
                # Get range for logging
                future_steps = [m.step for m in future_metrics]
                min_future_step = min(future_steps)
                max_future_step = max(future_steps)

                print(f"{self.log_prefix} Found {len(future_metrics)} old metrics (steps {min_future_step}-{max_future_step}) beyond current step {current_step}")
                print(f"{self.log_prefix} Cleaning up old metrics to prevent duplicates...")

                # Delete future metrics
                for metric in future_metrics:
                    db.delete(metric)

                db.commit()
                print(f"{self.log_prefix} Deleted {len(future_metrics)} old metrics")
            else:
                print(f"{self.log_prefix} No old metrics beyond current step {current_step} (clean start)")

            db.close()

        except Exception as e:
            # Non-critical: Log warning but continue training
            print(f"{self.log_prefix} WARNING: Failed to cleanup future metrics: {e}")

    def cleanup(self):
        """
        Cleanup training resources.

        - Flush metrics buffer to database
        - Remove Layer Offload Conductor hooks
        - Restore layers to GPU
        - Close TensorBoard writer
        """
        print(f"{self.log_prefix} Cleaning up training resources...")

        # Flush any remaining metrics to database
        if hasattr(self, '_metrics_buffer') and self._metrics_buffer and self.run_id is not None:
            print(f"{self.log_prefix} Flushing {len(self._metrics_buffer)} remaining metrics to database...")
            self._log_metrics_to_db(step=0, force_flush=True)  # step ignored, force_flush processes buffer

        # Shutdown DB executor (wait for async writes to complete)
        self._shutdown_db_executor()

        # Cleanup Layer Offload Conductor
        if hasattr(self, 'layer_offload_conductor') and self.layer_offload_conductor is not None:
            print(f"{self.log_prefix} Cleaning up LayerOffloadConductor...")
            self.layer_offload_conductor.cleanup()
            self.layer_offload_conductor = None

        # Close TensorBoard writer
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            print(f"{self.log_prefix} TensorBoard writer closed")

        print(f"{self.log_prefix} Cleanup complete")
