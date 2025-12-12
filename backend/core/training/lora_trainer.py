"""
LoRA Training Engine for SushiUI

Custom LoRA training implementation using SushiUI's architecture.
Uses individual components (UNet, VAE, TextEncoder) rather than Pipelines.
Implements training loop from scratch following ai-toolkit patterns.
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
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

    # Optionally show memory summary (detailed breakdown)
    # Uncomment for more details:
    # print(torch.cuda.memory_summary(device=None, abbreviated=True))


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
        print(f"{self.log_prefix} WARNING: Unknown dtype '{dtype_str}', defaulting to fp16")
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
    Get the target tensor based on prediction type.

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


class LoRALinearLayer(torch.nn.Module):
    """LoRA-enhanced linear layer that wraps the original linear layer."""

    def __init__(self, original_module: torch.nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_module.in_features
        out_features = original_module.out_features

        # Freeze original layer (we only train LoRA)
        self.original_module.requires_grad_(False)

        # LoRA matrices
        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)

        # Initialize
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        # Move to same device/dtype as original
        self.lora_down.to(original_module.weight.device, dtype=original_module.weight.dtype)
        self.lora_up.to(original_module.weight.device, dtype=original_module.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer output + LoRA adjustment
        result = self.original_module(x)
        lora_result = self.lora_up(self.lora_down(x)) * self.scaling
        return result + lora_result

    # Delegate attributes to original module
    @property
    def weight(self):
        return self.original_module.weight

    @property
    def bias(self):
        return self.original_module.bias

    @property
    def in_features(self):
        return self.original_module.in_features

    @property
    def out_features(self):
        return self.original_module.out_features


def inject_lora_into_linear(module: torch.nn.Linear, rank: int = 4, alpha: float = 1.0):
    """Inject LoRA into a linear layer by wrapping it."""
    lora_module = LoRALinearLayer(
        original_module=module,
        rank=rank,
        alpha=alpha
    )
    return lora_module


class LoRATrainer:
    """LoRA trainer using SushiUI's component-based architecture."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        run_name: str = None,  # Add run_name parameter
        lora_rank: int = 16,
        lora_alpha: int = 16,
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
        # Component-specific learning rates
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
    ):
        """
        Initialize LoRA trainer.

        Args:
            model_path: Path to base Stable Diffusion model
            output_dir: Directory to save checkpoints
            run_name: Training run name (for checkpoint filename generation)
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor)
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
        self.run_name = run_name or Path(output_dir).name  # Use directory name if not provided
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Component-specific learning rates (fallback to main learning_rate if not specified)
        self.unet_lr = unet_lr if unet_lr is not None else learning_rate
        self.text_encoder_lr = text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_1_lr = text_encoder_1_lr if text_encoder_1_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_2_lr = text_encoder_2_lr if text_encoder_2_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate

        # Convert dtype strings to torch.dtype
        self.weight_dtype = get_torch_dtype(weight_dtype)
        self.training_dtype = get_torch_dtype(training_dtype)
        self.output_dtype = get_torch_dtype(output_dtype)  # For safetensors saving
        self.vae_dtype = get_torch_dtype(vae_dtype)  # VAE-specific dtype (SDXL VAE works with fp16)
        self.mixed_precision = mixed_precision
        self.debug_vram = debug_vram
        self.use_flash_attention = use_flash_attention
        self.min_snr_gamma = min_snr_gamma

        # Legacy dtype for compatibility (defaults to weight_dtype)
        self.dtype = self.weight_dtype

        # Log prefix for subclass override (LoRA vs Full Finetune identification)
        self.log_prefix = "[Trainer]"  # Common logs, subclasses can override
        self.specific_log_prefix = "{self.log_prefix}"  # LoRA-specific logs

        print(f"[Trainer] Precision settings:")
        print(f"  Weight dtype: {weight_dtype} ({self.weight_dtype})")
        print(f"  Training dtype: {training_dtype} ({self.training_dtype})")
        print(f"  Output dtype: {output_dtype} ({self.output_dtype}) - for safetensors saving")
        print(f"  VAE dtype: {vae_dtype} ({self.vae_dtype})")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Loss calculation: Always FP32 for numerical stability")
        print(f"  Min-SNR gamma: {min_snr_gamma} ({'enabled' if min_snr_gamma > 0 else 'disabled'})")

        # Initialize tensorboard writer
        # Create subdirectory with timestamp for each training session (useful for resume)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = self.output_dir / "tensorboard" / timestamp
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        print(f"{self.log_prefix} Initializing on {self.device}")
        print(f"{self.log_prefix} Tensorboard logs: {tensorboard_dir}")
        print(f"{self.log_prefix} Loading model from {model_path}")

        # Detect model type (SD1.5, SDXL, Z-Image)
        from core.model_loader import ModelLoader
        model_type = ModelLoader.detect_model_type(model_path)
        self.is_zimage = (model_type == "zimage")
        self.is_sdxl = False  # Will be set later for SD/SDXL

        # Z-Image model loading
        if self.is_zimage:
            print(f"{self.log_prefix} Detected Z-Image model")
            print(f"{self.log_prefix} Loading Z-Image components from {model_path}")

            # Load Z-Image components using ModelLoader
            components = ModelLoader.load_zimage_from_diffusers(
                model_path=model_path,
                device="cpu",  # Load on CPU first, move to GPU later
                torch_dtype=self.weight_dtype
            )

            # Store original transformer before wrapping
            self.transformer_original = components["transformer"]
            self.vae = components["vae"]
            self.text_encoder = components["text_encoder"]
            self.tokenizer = components["tokenizer"]
            self.scheduler = components["scheduler"]

            # Z-Image specific: no text_encoder_2, no unet
            self.text_encoder_2 = None
            self.tokenizer_2 = None
            self.unet = None
            self.noise_scheduler = self.scheduler  # Alias for compatibility

            # Convert VAE to vae_dtype
            self.vae = self.vae.to(dtype=self.vae_dtype)

            # Wrap transformer with BatchedZImageWrapperOptimized for complete batched processing
            from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
            print(f"{self.log_prefix} Wrapping Z-Image Transformer with BatchedZImageWrapperOptimized")
            self.transformer = BatchedZImageWrapperOptimized(self.transformer_original)
            print(f"{self.log_prefix} Phase 2 optimization: Complete batched processing (NO List[Tensor] operations)")
            print(f"{self.log_prefix} - Batched patchify/unpatchify (no loops)")
            print(f"{self.log_prefix} - Direct batched tensor processing throughout")
            print(f"{self.log_prefix} - Expected VRAM reduction: significant (eliminates all List overhead)")

            print(f"{self.log_prefix} Z-Image model loaded successfully")
            print(f"{self.log_prefix} Scheduler type: {self.scheduler.__class__.__name__}")
            print(f"{self.log_prefix} VAE latent channels: {self.vae.config.latent_channels}")

        # SD/SDXL model loading (existing logic)
        elif not self.is_zimage:
            # Detect if model is safetensors file or diffusers directory
            is_safetensors = model_path.endswith('.safetensors')

            if is_safetensors:
                print(f"{self.log_prefix} Loading from safetensors file")
                # Load pipeline from single safetensors file, then extract components
                # Try SDXL first, fall back to SD1.5
                try:
                    print(f"{self.log_prefix} Trying SDXL pipeline...")
                    temp_pipeline = StableDiffusionXLPipeline.from_single_file(
                        model_path,
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                    )
                    is_sdxl_model = True
                except Exception as e:
                    print(f"{self.log_prefix} Not SDXL, trying SD1.5 pipeline...")
                    temp_pipeline = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                    )
                    is_sdxl_model = False

                # Extract components from pipeline
                self.vae = temp_pipeline.vae
                self.text_encoder = temp_pipeline.text_encoder
                self.tokenizer = temp_pipeline.tokenizer
                self.unet = temp_pipeline.unet

                # IMPORTANT: Always use DDPMScheduler for training (not the inference scheduler from model)
                # The model may contain EulerDiscreteScheduler or other inference schedulers,
                # but training requires DDPMScheduler with specific settings (sd-scripts approach)
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

                # Clean up pipeline reference (we only need components)
                del temp_pipeline

                # Convert VAE to vae_dtype (SDXL VAE works fine with fp16)
                self.vae = self.vae.to(dtype=self.vae_dtype)
            else:
                print(f"{self.log_prefix} Loading from diffusers directory")
                # Load model components from diffusers directory (SushiUI style)
                self.vae = AutoencoderKL.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=self.vae_dtype  # Use vae_dtype for VAE (SDXL VAE works with fp16)
                )

                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_path,
                    subfolder="text_encoder",
                    torch_dtype=self.dtype
                )

                self.tokenizer = CLIPTokenizer.from_pretrained(
                    model_path,
                    subfolder="tokenizer"
                )

                self.unet = UNet2DConditionModel.from_pretrained(
                    model_path,
                    subfolder="unet",
                    torch_dtype=self.dtype
                )

                # Load noise scheduler
                self.noise_scheduler = DDPMScheduler.from_pretrained(
                    model_path,
                    subfolder="scheduler"
                )

                # Try to load SDXL-specific components (text_encoder_2, tokenizer_2)
                try:
                    self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                        model_path,
                        subfolder="text_encoder_2",
                        torch_dtype=self.dtype
                    )
                    self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                        model_path,
                        subfolder="tokenizer_2"
                    )
                    print(f"{self.log_prefix} Loaded SDXL text_encoder_2 and tokenizer_2")
                except Exception:
                    # SD1.5 models don't have these
                    self.text_encoder_2 = None
                    self.tokenizer_2 = None

            # Detect model type (SD1.5 vs SDXL) - only for SD/SDXL
            self.is_sdxl = hasattr(self.unet.config, "addition_embed_type") if hasattr(self, 'unet') and self.unet is not None else False
            print(f"{self.log_prefix} Model type: {'SDXL' if self.is_sdxl else 'SD1.5'}")
            print(f"{self.log_prefix} Prediction type: {self.noise_scheduler.config.prediction_type}")
            print(f"{self.log_prefix} VAE scaling factor: {self.vae.config.scaling_factor}")

        # Z-Image: Different setup from SD/SDXL
        if self.is_zimage:
            # Enable gradient checkpointing for Transformer (Z-Image)
            if hasattr(self.transformer, 'enable_gradient_checkpointing'):
                self.transformer.enable_gradient_checkpointing()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Z-Image Transformer")
            else:
                print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for Z-Image Transformer")

            # Text Encoder (Qwen3) gradient checkpointing
            # NOTE: Text Encoder is frozen, but gradient checkpointing is enabled for potential future use
            if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder (Qwen3)")

            # Freeze all base weights
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)  # Text Encoder is ALWAYS frozen for Z-Image
            self.transformer.requires_grad_(False)

            # Move to device
            # IMPORTANT: VAE stays on CPU (only used for latent cache generation)
            # IMPORTANT: Text Encoder stays on CPU initially (will be moved to GPU only during caption pre-encoding)
            print(f"{self.log_prefix} VAE will stay on CPU (used only for latent cache generation)")
            # self.vae stays on CPU (loaded on CPU by ModelLoader)
            # self.text_encoder stays on CPU (loaded on CPU by ModelLoader)
            print(f"{self.log_prefix} Moving Transformer to {self.device}...")
            self.transformer.to(self.device)

            # Log actual device placement
            def get_device(model):
                """Get device of first parameter"""
                return next(model.parameters()).device

            print(f"\n{self.log_prefix} Component device placement:")
            print(f"  Transformer: {get_device(self.transformer)}")
            print(f"  Text Encoder: {get_device(self.text_encoder)}")
            print(f"  VAE: {get_device(self.vae)}")
            print()

            if self.debug_vram:
                print_vram_usage("After loading Z-Image models to GPU")

            # Set VAE to eval mode (never trained)
            self.vae.eval()

            # Transformer must be in train mode for gradient checkpointing to work
            # Text Encoder remains in eval mode (frozen)
            self.transformer.train()
            self.text_encoder.eval()
            print(f"{self.log_prefix} Z-Image Transformer set to train mode, Text Encoder to eval mode (frozen)")

        # SD/SDXL: Existing setup
        else:
            # Enable Flash Attention BEFORE gradient checkpointing
            # Gradient checkpointing must be enabled after setting attention processors
            if self.use_flash_attention:
                try:
                    from core.inference.attention_processors import FlashAttnProcessor
                    print(f"{self.log_prefix} Setting Flash Attention processors...")
                    processor = FlashAttnProcessor()
                    new_processors = {name: processor for name in self.unet.attn_processors.keys()}
                    self.unet.set_attn_processor(new_processors)
                    num_processors = len(new_processors)
                    print(f"{self.log_prefix} [OK] Flash Attention enabled for {num_processors} attention layers")
                except ImportError:
                    print(f"{self.log_prefix} WARNING: Flash Attention not available, falling back to default")
                except Exception as e:
                    print(f"{self.log_prefix} WARNING: Failed to enable Flash Attention: {e}")

            # Enable gradient checkpointing AFTER Flash Attention setup
            # This must be done before LoRA application to avoid breaking gradients
            if hasattr(self.unet, 'enable_gradient_checkpointing'):
                self.unet.enable_gradient_checkpointing()
                print(f"{self.log_prefix} Gradient checkpointing enabled for U-Net")
            else:
                print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for this U-Net")

            # Enable gradient checkpointing for Text Encoders (sd-scripts/ai-toolkit approach)
            if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder 1")

            if self.text_encoder_2 is not None and hasattr(self.text_encoder_2, 'gradient_checkpointing_enable'):
                self.text_encoder_2.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder 2")

            # Freeze all base weights
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.unet.requires_grad_(False)
            if self.text_encoder_2 is not None:
                self.text_encoder_2.requires_grad_(False)

            # Move to device
            self.vae.to(self.device)
            self.text_encoder.to(self.device)
            self.unet.to(self.device)
            if self.text_encoder_2 is not None:
                self.text_encoder_2.to(self.device)

            if self.debug_vram:
                print_vram_usage("After loading models to GPU")

            # Set VAE to eval mode (never trained)
            self.vae.eval()

            # U-Net and Text Encoders must be in train mode for gradient checkpointing to work (sd-scripts approach)
            # This is required according to Diffusers TI example
            self.unet.train()
            self.text_encoder.train()
            if self.text_encoder_2 is not None:
                self.text_encoder_2.train()
            print(f"{self.log_prefix} U-Net and Text Encoders set to train mode for gradient checkpointing")

        # LoRA layers storage
        self.lora_layers = {}

        # Apply LoRA to UNet
        self._apply_lora()
        if self.debug_vram:
            print_vram_usage("After applying LoRA layers")

        self.optimizer = None
        self.lr_scheduler = None

    def _apply_lora(self):
        """Apply LoRA layers to model modules."""
        print(f"{self.specific_log_prefix} Applying LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})")

        if self.is_zimage:
            # Z-Image: Apply LoRA to Transformer only (Text Encoder is frozen)
            self._apply_lora_zimage()
        else:
            # SD/SDXL: Apply LoRA to U-Net and Text Encoder
            # Apply LoRA to U-Net (Transformer2DModel approach, compatible with sd-scripts)
            unet_lora_count = self._apply_lora_to_unet_transformers()
            print(f"{self.log_prefix} Injected {unet_lora_count} LoRA layers into U-Net")

            # Apply LoRA to Text Encoder 1
            te1_lora_count = self._apply_lora_to_module(
                self.text_encoder,
                prefix="te1",
                target_modules=["mlp.fc1", "mlp.fc2"]  # MLP layers in text encoder
            )
            print(f"{self.log_prefix} Injected {te1_lora_count} LoRA layers into Text Encoder 1")

            # Apply LoRA to Text Encoder 2 (SDXL)
            if self.text_encoder_2 is not None:
                te2_lora_count = self._apply_lora_to_module(
                    self.text_encoder_2,
                    prefix="te2",
                    target_modules=["mlp.fc1", "mlp.fc2"]
                )
                print(f"{self.log_prefix} Injected {te2_lora_count} LoRA layers into Text Encoder 2")

    def _apply_lora_zimage(self):
        """
        Apply LoRA to Z-Image Transformer attention layers.

        Targets ZImageAttention modules: to_q, to_k, to_v, to_out[0] (ModuleList)

        Based on musubi-tuner's lora_zimage.py implementation:
        - ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]
        - Attention layers: qkv_proj, out_proj (musubi splits into to_q/k/v internally)
        """
        lora_count = 0

        print(f"{self.specific_log_prefix} Applying LoRA to Z-Image Transformer (ZImageAttention modules)")

        # Access the original transformer inside the wrapper
        # self.transformer is BatchedZImageWrapper, self.transformer.transformer is the original model
        target_transformer = self.transformer.transformer if hasattr(self.transformer, 'transformer') else self.transformer

        # Find all ZImageAttention modules in the Transformer
        attention_modules = []
        for name, module in target_transformer.named_modules():
            if module.__class__.__name__ == "ZImageAttention":
                attention_modules.append((name, module))

        print(f"{self.log_prefix} Found {len(attention_modules)} ZImageAttention modules")

        # Target layers: to_q, to_k, to_v, to_out[0]
        target_attrs = ["to_q", "to_k", "to_v"]

        for attn_name, attn_module in attention_modules:
            # Handle to_q, to_k, to_v
            for attr_name in target_attrs:
                if hasattr(attn_module, attr_name):
                    original_linear = getattr(attn_module, attr_name)

                    if isinstance(original_linear, torch.nn.Linear):
                        # Create LoRA layer
                        lora_module = inject_lora_into_linear(original_linear, self.lora_rank, self.lora_alpha)

                        # Replace in attention module
                        setattr(attn_module, attr_name, lora_module)

                        # Store reference
                        storage_key = f"transformer.{attn_name}.{attr_name}"
                        self.lora_layers[storage_key] = lora_module
                        lora_count += 1

            # Handle to_out (ModuleList in Z-Image, first element is Linear projection)
            if hasattr(attn_module, "to_out") and isinstance(attn_module.to_out, torch.nn.ModuleList):
                if len(attn_module.to_out) > 0 and isinstance(attn_module.to_out[0], torch.nn.Linear):
                    original_linear = attn_module.to_out[0]

                    # Create LoRA layer
                    lora_module = inject_lora_into_linear(original_linear, self.lora_rank, self.lora_alpha)

                    # Replace in ModuleList
                    attn_module.to_out[0] = lora_module

                    # Store reference
                    storage_key = f"transformer.{attn_name}.to_out.0"
                    self.lora_layers[storage_key] = lora_module
                    lora_count += 1

        print(f"{self.log_prefix} Injected {lora_count} LoRA layers into Z-Image Transformer")
        print(f"{self.log_prefix} Text Encoder (Qwen3) is frozen (no LoRA)")

    def _convert_diffusers_to_sd_key(self, diffusers_name: str) -> str:
        """
        Convert diffusers-format U-Net module name to SD format.

        Based on sd-scripts conversion mapping for SDXL:
        - down_blocks.i.attentions.j → input_blocks.{3*i + j + 1}.1
        - mid_block.attentions.0 → middle_block.1
        - up_blocks.i.attentions.j → output_blocks.{3*i + j}.1

        Args:
            diffusers_name: Full diffusers module name (e.g., "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q")

        Returns:
            SD format name (e.g., "input_blocks_4_1_transformer_blocks_0_attn1_to_q")
        """
        import re

        # Handle down_blocks
        match = re.match(r'down_blocks\.(\d+)\.attentions\.(\d+)\.(.+)', diffusers_name)
        if match:
            i, j, rest = match.groups()
            block_idx = 3 * int(i) + int(j) + 1
            sd_name = f"input_blocks_{block_idx}_1_{rest}"
            return sd_name.replace(".", "_")

        # Handle mid_block
        match = re.match(r'mid_block\.attentions\.0\.(.+)', diffusers_name)
        if match:
            rest = match.group(1)
            sd_name = f"middle_block_1_{rest}"
            return sd_name.replace(".", "_")

        # Handle up_blocks
        match = re.match(r'up_blocks\.(\d+)\.attentions\.(\d+)\.(.+)', diffusers_name)
        if match:
            i, j, rest = match.groups()
            block_idx = 3 * int(i) + int(j)
            sd_name = f"output_blocks_{block_idx}_1_{rest}"
            return sd_name.replace(".", "_")

        # Fallback: just replace dots with underscores
        return diffusers_name.replace(".", "_")

    def _apply_lora_to_unet_transformers(self) -> int:
        """
        Apply LoRA to all Transformer2DModel modules in U-Net.
        This follows the sd-scripts approach of targeting entire transformer blocks.

        For SDXL, this targets 11 transformer blocks:
        - down_blocks.1.attentions.0 → input_blocks.4.1 (IN04)
        - down_blocks.1.attentions.1 → input_blocks.5.1 (IN05)
        - down_blocks.2.attentions.0 → input_blocks.7.1 (IN07)
        - down_blocks.2.attentions.1 → input_blocks.8.1 (IN08)
        - mid_block.attentions.0 → middle_block.1 (MID)
        - up_blocks.0.attentions.0-2 → output_blocks.0-2.1 (OUT00-OUT02)
        - up_blocks.1.attentions.0-2 → output_blocks.3-5.1 (OUT03-OUT05)

        Returns:
            Number of LoRA layers injected
        """
        lora_count = 0

        # Find all Transformer2DModel modules
        transformer_modules = []
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == "Transformer2DModel":
                transformer_modules.append((name, module))

        print(f"{self.log_prefix} Found {len(transformer_modules)} Transformer2DModel modules in U-Net")

        # For each transformer, apply LoRA to all Linear layers inside
        for transformer_name, transformer_module in transformer_modules:
            for child_name, child_module in transformer_module.named_modules():
                if isinstance(child_module, torch.nn.Linear):
                    # Build full diffusers name
                    if child_name:
                        full_diffusers_name = f"{transformer_name}.{child_name}"
                    else:
                        full_diffusers_name = transformer_name

                    # Convert to SD format for storage key
                    sd_key = self._convert_diffusers_to_sd_key(full_diffusers_name)
                    storage_key = f"unet.{sd_key}"

                    # Navigate to parent and replace with LoRA
                    name_parts = full_diffusers_name.split(".")
                    parent = self.unet
                    for part in name_parts[:-1]:
                        parent = getattr(parent, part)

                    child_attr_name = name_parts[-1]

                    # Create LoRA layer
                    lora_module = inject_lora_into_linear(child_module, self.lora_rank, self.lora_alpha)

                    # Replace in parent
                    setattr(parent, child_attr_name, lora_module)

                    # Store reference with SD format key
                    self.lora_layers[storage_key] = lora_module
                    lora_count += 1

        return lora_count

    def _apply_lora_to_module(self, module: torch.nn.Module, prefix: str, target_modules: list) -> int:
        """
        Apply LoRA to target layers in a module.

        Args:
            module: The module to apply LoRA to (unet, text_encoder, etc.)
            prefix: Prefix for LoRA layer names (e.g., "unet", "te1", "te2")
            target_modules: List of target module name patterns (e.g., ["to_q", "to_k"])

        Returns:
            Number of LoRA layers injected
        """
        lora_count = 0

        # Collect modules to replace (can't modify dict while iterating)
        modules_to_replace = []
        for name, submodule in module.named_modules():
            # Check if this is a target module
            if any(target in name for target in target_modules):
                if isinstance(submodule, torch.nn.Linear):
                    modules_to_replace.append((name, submodule))

        # Replace modules
        for full_name, original_module in modules_to_replace:
            # Parse the full name to get parent and child name
            # e.g., "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
            name_parts = full_name.split(".")

            # Navigate to parent module
            parent = module
            for part in name_parts[:-1]:
                parent = getattr(parent, part)

            child_name = name_parts[-1]

            # Create LoRA layer
            lora_module = inject_lora_into_linear(original_module, self.lora_rank, self.lora_alpha)

            # Replace in parent
            setattr(parent, child_name, lora_module)

            # Store reference with prefix
            storage_key = f"{prefix}.{full_name}"
            self.lora_layers[storage_key] = lora_module
            lora_count += 1

        return lora_count

    def setup_optimizer(self, optimizer_type: str = "adamw8bit", lr_scheduler_type: str = "constant", total_steps: int = 1000):
        """
        Setup optimizer and learning rate scheduler using OptimizerFactory.

        See OptimizerFactory.get_available_optimizers() for supported optimizers.
        """
        from core.training.optimizers import OptimizerFactory

        print(f"{self.log_prefix} Setting up optimizer: {optimizer_type}")

        # Group trainable parameters by component
        unet_params = []  # SD/SDXL U-Net
        transformer_params = []  # Z-Image Transformer
        text_encoder_1_params = []
        text_encoder_2_params = []

        for key, lora in self.lora_layers.items():
            # Only add LoRA-specific parameters (lora_down and lora_up)
            lora_params = list(lora.lora_down.parameters()) + list(lora.lora_up.parameters())

            if key.startswith("unet."):
                unet_params.extend(lora_params)
            elif key.startswith("transformer."):
                # Z-Image Transformer
                transformer_params.extend(lora_params)
            elif key.startswith("te2.") or key.startswith("text_encoder_2."):
                text_encoder_2_params.extend(lora_params)
            elif key.startswith("te1.") or key.startswith("text_encoder."):
                text_encoder_1_params.extend(lora_params)

        # Build param_groups with component-specific learning rates
        param_groups = []
        if len(unet_params) > 0:
            param_groups.append({"params": unet_params, "lr": self.unet_lr})
            print(f"{self.log_prefix}   U-Net: {len(unet_params)} params, lr={self.unet_lr}")
        if len(transformer_params) > 0:
            # Z-Image Transformer uses unet_lr (same role as U-Net in SD/SDXL)
            param_groups.append({"params": transformer_params, "lr": self.unet_lr})
            print(f"{self.log_prefix}   Z-Image Transformer: {len(transformer_params)} params, lr={self.unet_lr}")
        if len(text_encoder_1_params) > 0:
            param_groups.append({"params": text_encoder_1_params, "lr": self.text_encoder_1_lr})
            print(f"{self.log_prefix}   Text Encoder 1: {len(text_encoder_1_params)} params, lr={self.text_encoder_1_lr}")
        if len(text_encoder_2_params) > 0:
            param_groups.append({"params": text_encoder_2_params, "lr": self.text_encoder_2_lr})
            print(f"{self.log_prefix}   Text Encoder 2: {len(text_encoder_2_params)} params, lr={self.text_encoder_2_lr}")

        if len(param_groups) == 0:
            raise RuntimeError("No trainable parameters found")

        # Create optimizer using factory
        try:
            self.optimizer = OptimizerFactory.create_optimizer(
                optimizer_type=optimizer_type,
                params=param_groups,
                learning_rate=self.learning_rate,  # This will be overridden by param_groups
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        except (ValueError, ImportError) as e:
            print(f"{self.log_prefix} ERROR: {e}")
            print("{self.log_prefix} Falling back to AdamW")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,  # This will be overridden by param_groups
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

    def encode_prompt(self, prompt: str, requires_grad: bool = False):
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt to encode
            requires_grad: Whether to enable gradient computation for text encoders (for training)

        Returns:
            For SD1.5: text_embeddings tensor
            For SDXL: tuple of (text_embeddings, pooled_embeddings)
        """
        if self.is_sdxl:
            # SDXL: Use two text encoders
            # First text encoder (CLIP ViT-L)
            text_inputs_1 = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            # Second text encoder (OpenCLIP ViT-bigG)
            text_inputs_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            # Enable/disable gradient based on training mode
            context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

            with context_manager:
                # Encode with first text encoder
                text_embeddings_1 = self.text_encoder(
                    text_inputs_1.input_ids.to(self.device),
                    output_hidden_states=False,
                )[0]

                # Encode with second text encoder (get penultimate hidden state and pooled output)
                encoder_output_2 = self.text_encoder_2(
                    text_inputs_2.input_ids.to(self.device),
                    output_hidden_states=True,
                )
                text_embeddings_2 = encoder_output_2.hidden_states[-2]  # Penultimate layer
                pooled_embeddings = encoder_output_2[0]  # Pooled output

                # Concatenate embeddings from both encoders
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

            # Enable/disable gradient based on training mode
            context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

            with context_manager:
                text_embeddings = self.text_encoder(
                    text_inputs.input_ids.to(self.device),
                )[0]

                return text_embeddings

    def encode_prompt_zimage(
        self,
        prompt: str,
        max_sequence_length: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt using Qwen3 text encoder with chat template (Z-Image).

        Args:
            prompt: Text prompt
            max_sequence_length: Maximum sequence length (default: 512)

        Returns:
            Tuple of (prompt_embeds, attention_mask)
            - prompt_embeds: [valid_seq_len, 2560] (variable length)
            - attention_mask: [max_sequence_length] (bool)
        """
        # Format with Qwen chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Qwen-specific feature
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

        # Encode with penultimate layer (similar to SDXL text_encoder_2)
        # Text Encoder is ALWAYS frozen for Z-Image, so always use no_grad
        with torch.no_grad():
            encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            prompt_embeds = encoder_output.hidden_states[-2]  # [1, seq_len, 2560]

        # Return full padded embeddings and mask (for easier batching during training)
        # The mask will be used to attend only to valid tokens
        return prompt_embeds[0], attention_mask[0]  # [max_seq_len, 2560], [max_seq_len]

    def encode_image(
        self,
        image: Image.Image,
        target_size: int = 512,
        target_width: int = None,
        target_height: int = None
    ) -> torch.Tensor:
        """
        Encode image to latents.

        Args:
            image: PIL Image
            target_size: Square target size (deprecated, use target_width/height)
            target_width: Target width (for bucketing)
            target_height: Target height (for bucketing)

        Returns:
            Latent tensor
        """
        image = image.convert("RGB")

        # Determine target dimensions
        if target_width is not None and target_height is not None:
            # Bucketing mode: use specified dimensions
            width, height = target_width, target_height
        else:
            # Legacy mode: square resize
            width, height = target_size, target_size

        # Resize with aspect ratio preservation + center crop
        # This matches ai-toolkit's approach
        img_width, img_height = image.size

        # Debug: Log large image resize
        if img_width * img_height > 5000 * 5000:
            print(f"[encode_image] Resizing large image {img_width}x{img_height} -> {width}x{height}")

        scale = max(width / img_width, height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize with Lanczos resampling
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Center crop to target size
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        image = image.crop((left, top, left + width, top + height))

        # Debug: Verify final size
        if image.size != (width, height):
            print(f"[encode_image] ERROR: Final image size {image.size} != target {(width, height)}")

        # Convert to tensor and normalize to [-1, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = (image_array - 0.5) * 2.0

        # Convert to torch tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

        # Get VAE device (it might be on CPU if latent caching is enabled)
        vae_device = next(self.vae.parameters()).device
        image_tensor = image_tensor.to(device=vae_device, dtype=self.vae.dtype)

        # Debug: Log VRAM before VAE encode for large images
        if vae_device.type == 'cuda' and img_width * img_height > 5000 * 5000:
            vram_before = torch.cuda.memory_allocated(vae_device) / 1024**3
            print(f"[encode_image] VRAM before VAE encode: {vram_before:.2f} GB")

        # Encode to latents
        with torch.no_grad():
            if self.is_zimage:
                # Z-Image VAE: Use encoder directly (no .encode() method)
                h = self.vae.encoder(image_tensor)
                if self.vae.quant_conv is not None:
                    h = self.vae.quant_conv(h)
                # Split to mean and logvar, sample
                mean, logvar = torch.chunk(h, 2, dim=1)
                latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                # Apply scaling and shift (must match ai-toolkit: z = scaling * (raw - shift))
                shift_factor = self.vae.config.shift_factor if self.vae.config.shift_factor is not None else 0.0
                latents = self.vae.config.scaling_factor * (latents - shift_factor)
            else:
                # SD/SDXL VAE: Use standard .encode() method
                latents = self.vae.encode(image_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        # Debug: Log VRAM after VAE encode for large images
        if vae_device.type == 'cuda' and img_width * img_height > 5000 * 5000:
            vram_after = torch.cuda.memory_allocated(vae_device) / 1024**3
            print(f"[encode_image] VRAM after VAE encode: {vram_after:.2f} GB")
            print(f"[encode_image] Latent shape: {latents.shape}")

        return latents

    # Note: unet_forward_with_lora() removed - no longer needed
    # LoRA layers are already integrated into self.unet via _apply_lora()
    # Just call self.unet() directly

    def train_step(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor = None,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        profile_vram: bool = False,
    ) -> float:
        """
        Perform single training step.

        Args:
            latents: Image latents [B, C, H, W]
            text_embeddings: Text prompt embeddings [B, 77, 768]
            pooled_embeddings: Pooled text embeddings (SDXL only)
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output (optional)
            profile_vram: If True, print VRAM usage at each step

        Returns:
            Loss value
        """
        # Keep latents in their original dtype to avoid memory duplication
        # We'll convert to output_dtype only when needed for loss calculation

        if profile_vram:
            print_vram_usage("[train_step] Start")

        # Sample noise
        noise = torch.randn_like(latents)

        if profile_vram:
            print_vram_usage("[train_step] After noise generation")

        # Sample random timestep
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        # Add noise to latents according to noise scheduler
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Prepare added_cond_kwargs for SDXL
        added_cond_kwargs = None
        if self.is_sdxl and pooled_embeddings is not None:
            # Calculate image size from latents (latents are downscaled by VAE factor of 8)
            latent_height, latent_width = latents.shape[2], latents.shape[3]
            image_height, image_width = latent_height * 8, latent_width * 8

            # Prepare time_ids: [original_size, crops_coords_top_left, target_size]
            original_size = (image_height, image_width)
            crops_coords_top_left = (0, 0)
            target_size = (image_height, image_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=pooled_embeddings.dtype, device=self.device)

            # Repeat time_ids for each item in batch
            add_time_ids = add_time_ids.repeat(batch_size, 1)

            added_cond_kwargs = {
                "text_embeds": pooled_embeddings,
                "time_ids": add_time_ids
            }

        if profile_vram:
            print_vram_usage("[train_step] Before UNet forward")

        # Enable gradients on inputs for gradient checkpointing (sd-scripts approach)
        # This is required for gradient checkpointing to recompute activations correctly
        noisy_latents.requires_grad_(True)
        text_embeddings.requires_grad_(True)
        if pooled_embeddings is not None:
            pooled_embeddings.requires_grad_(True)

        # Predict noise using UNet (LoRA is already integrated)
        # Use autocast if mixed precision is enabled
        if self.mixed_precision:
            # Autocast to training_dtype for forward pass (reduces memory and speeds up)
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
            # No mixed precision: use weight_dtype throughout
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

        # Get target based on prediction type
        prediction_type = self.noise_scheduler.config.prediction_type
        target = get_target_from_prediction_type(
            self.noise_scheduler,
            prediction_type,
            latents,
            noise,
            timesteps,
        )

        # Debug: Verify target correctness on first step
        if not hasattr(self, '_debug_logged_target_verification'):
            print(f"{self.log_prefix} Target verification (prediction_type='{prediction_type}'):")
            if prediction_type == "epsilon":
                target_matches_noise = torch.equal(target, noise)
                print(f"  - Target is identical to noise: {target_matches_noise}")
                if not target_matches_noise:
                    print(f"  - WARNING: Target should equal noise for epsilon prediction!")
                    print(f"  - Target mean: {target.mean().item():.6f}, Noise mean: {noise.mean().item():.6f}")
            print(f"  - Target shape: {target.shape}")
            print(f"  - Target dtype: {target.dtype}")
            self._debug_logged_target_verification = True

        # Calculate loss (always in fp32 for numerical stability)
        # Use reduction="none" to apply SNR weighting per-sample
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")

        # Take mean across spatial and channel dimensions first (keep batch dimension)
        # Shape: [B, C, H, W] -> [B]
        # This matches sd-scripts standard implementation
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Apply Min-SNR gamma weighting if enabled (per-sample in batch)
        if self.min_snr_gamma > 0:
            loss_per_sample_weighted = apply_snr_weight(loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma)
        else:
            loss_per_sample_weighted = loss_per_sample

        # Take mean across batch dimension
        # Shape: [B] -> scalar
        loss = loss_per_sample_weighted.mean()

        # Calculate reconstruction loss (for monitoring/visualization)
        # This measures how well the model can reconstruct the target latent from noisy input
        with torch.no_grad():
            # Get alpha_bar for reconstruction
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
            alpha_bar_t = alphas_cumprod[timesteps]
            while alpha_bar_t.dim() < latents.dim():
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

            # Reconstruct x_0 based on prediction type
            if prediction_type == "epsilon":
                # x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
                predicted_latent_for_recon = (noisy_latents - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar
            elif prediction_type == "v_prediction":
                # x_0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v_pred
                predicted_latent_for_recon = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * model_pred
            elif prediction_type == "sample":
                # Direct prediction of x_0
                predicted_latent_for_recon = model_pred
            else:
                # Fallback (should not reach here)
                predicted_latent_for_recon = noisy_latents - model_pred

            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
            recon_loss = recon_loss_per_sample.mean()

        if profile_vram:
            print_vram_usage("[train_step] After loss calculation")

        # Debug: Save latents if requested
        if debug_save_path is not None:
            debug_save_path.mkdir(parents=True, exist_ok=True)

            # Save as .pt files with detailed info (save first item in batch)
            timestep_value = timesteps[0].item()

            # Calculate predicted_latent (denoised latent at t=0)
            # Use exact reconstruction formula based on prediction type
            with torch.no_grad():
                # Reuse alpha_bar from reconstruction loss calculation
                if prediction_type == "epsilon":
                    predicted_latent = (noisy_latents - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar
                elif prediction_type == "v_prediction":
                    predicted_latent = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * model_pred
                elif prediction_type == "sample":
                    predicted_latent = model_pred
                else:
                    predicted_latent = noisy_latents - model_pred

            # Prepare debug data
            debug_data = {
                'latents': latents[0:1].detach().cpu(),  # Save only first item in batch
                'noisy_latents': noisy_latents[0:1].detach().cpu(),
                'predicted_noise': model_pred[0:1].detach().cpu(),
                'actual_noise': noise[0:1].detach().cpu(),
                'predicted_latent': predicted_latent[0:1].detach().cpu(),
                'timestep': timestep_value,
                'loss': loss_per_sample_weighted[0].item(),  # Per-sample loss (weighted) for first sample
                'loss_batch_mean': loss.item(),  # Batch mean loss (for reference)
                'loss_unweighted': loss_per_sample[0].item(),  # Raw MSE loss for first sample (no Min-SNR)
                'recon_loss': recon_loss_per_sample[0].item(),  # Reconstruction loss for first sample
                'recon_loss_batch_mean': recon_loss.item(),  # Batch mean reconstruction loss
                'batch_size': batch_size,
                'min_snr_gamma': self.min_snr_gamma,
            }

            # Add caption if available (save first caption in batch)
            if debug_captions is not None and len(debug_captions) > 0:
                debug_data['caption'] = debug_captions[0]
                # Also save all captions in batch for reference
                debug_data['all_captions'] = debug_captions

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:04d}.pt")

            del predicted_latent  # Free memory after debug save
            caption_info = f" (caption: {debug_captions[0][:50]}...)" if debug_captions and len(debug_captions) > 0 else ""
            print(f"[Debug] Saved latents to {debug_save_path} (timestep={timestep_value}){caption_info}")

        # Backward pass
        if profile_vram:
            print_vram_usage("[train_step] Before backward")

        self.optimizer.zero_grad()
        loss.backward()

        if profile_vram:
            print_vram_usage("[train_step] After backward")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for lora in self.lora_layers.values() for p in lora.parameters()],
            max_norm=1.0
        )

        if profile_vram:
            print_vram_usage("[train_step] After gradient clipping")

        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()

        # Clear gradients to free VRAM (set_to_none=True for memory efficiency)
        self.optimizer.zero_grad(set_to_none=True)

        if profile_vram:
            print_vram_usage("[train_step] After optimizer step")

        # Get loss values before cleanup
        loss_value = loss.detach().item()
        recon_loss_value = recon_loss.detach().item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        del noise, noisy_latents, model_pred, loss, recon_loss
        if self.is_sdxl and added_cond_kwargs is not None:
            del added_cond_kwargs

        return loss_value, recon_loss_value

    def train_step_zimage(
        self,
        latents: torch.Tensor,
        caption_embeds: torch.Tensor,
        caption_mask: torch.Tensor,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        profile_vram: bool = False,
    ) -> float:
        """
        Perform single training step for Z-Image (Flow Matching).

        Args:
            latents: Image latents [B, 16, H, W] (Z-Image uses 16 latent channels)
            caption_embeds: Pre-encoded caption embeddings [B, seq_len, 2560]
            caption_mask: Attention mask [B, seq_len] (bool)
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output (optional)
            profile_vram: If True, print VRAM usage at each step

        Returns:
            Loss value
        """
        if profile_vram:
            print_vram_usage("[train_step_zimage] Start")

        # Flow Matching: Sample random timesteps from [0, 1]
        batch_size = latents.shape[0]
        # Sample timesteps uniformly from [0, 1]
        timesteps = torch.rand(batch_size, device=self.device)

        if profile_vram:
            print_vram_usage("[train_step_zimage] After timestep sampling")

        # Flow Matching: Sample noise (standard normal distribution)
        noise = torch.randn_like(latents)

        # Flow Matching: Interpolate between noise and data
        # x_t = (1 - t) * noise + t * data
        # Reshape timesteps for broadcasting: [B] -> [B, 1, 1, 1]
        t = timesteps[:, None, None, None]
        noisy_latents = (1.0 - t) * noise + t * latents

        if profile_vram:
            print_vram_usage("[train_step_zimage] After noise interpolation")

        # Enable gradients on inputs for gradient checkpointing
        noisy_latents.requires_grad_(True)
        caption_embeds.requires_grad_(True)

        # Predict velocity using Z-Image Transformer (LoRA is already integrated)
        # Velocity target: v = data - noise
        # Note: BatchedZImageWrapper handles conversion to/from List[Tensor] internally
        # Input: [B, C, H, W], Output: [B, C, H, W]
        # Add frame dimension for Z-Image: [B, C, H, W] -> [B, C, 1, H, W]
        noisy_latents_4d = noisy_latents.unsqueeze(2)  # Add F=1 dimension

        # Debug: Verify gradient checkpointing is active (first step only)
        if not hasattr(self, '_debug_logged_gc_status'):
            print(f"[DEBUG] Gradient Checkpointing Status:")
            print(f"  - Transformer.training: {self.transformer.training}")
            print(f"  - Transformer.gradient_checkpointing: {self.transformer.gradient_checkpointing}")
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            print(f"  - VRAM before forward pass: {allocated_before:.2f} GB")
            print(f"[DEBUG] Input data verification:")
            print(f"  - latents dtype: {latents.dtype}, device: {latents.device}")
            print(f"  - latents contains NaN: {torch.isnan(latents).any().item()}")
            print(f"  - noisy_latents dtype: {noisy_latents.dtype}, device: {noisy_latents.device}")
            print(f"  - noisy_latents contains NaN: {torch.isnan(noisy_latents).any().item()}")
            print(f"  - timesteps dtype: {timesteps.dtype}, range: [{timesteps.min().item():.4f}, {timesteps.max().item():.4f}]")
            print(f"  - caption_embeds dtype: {caption_embeds.dtype}, device: {caption_embeds.device}")
            print(f"  - caption_embeds contains NaN: {torch.isnan(caption_embeds).any().item()}")
            # Check Transformer weights for NaN
            has_nan_weights = False
            for name, param in self.transformer.named_parameters():
                if torch.isnan(param).any():
                    print(f"  - WARNING: Transformer weight '{name}' contains NaN!")
                    has_nan_weights = True
                    break
            if not has_nan_weights:
                print(f"  - Transformer weights: OK (no NaN)")
            self._debug_logged_gc_status = True

        if self.mixed_precision:
            # Autocast to training_dtype for forward pass
            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                model_pred, _ = self.transformer(
                    x=noisy_latents_4d,
                    t=timesteps,
                    cap_feats=caption_embeds,
                    cap_mask=caption_mask,
                )
        else:
            # No mixed precision: use weight_dtype throughout
            model_pred, _ = self.transformer(
                x=noisy_latents_4d,
                t=timesteps,
                cap_feats=caption_embeds,
                cap_mask=caption_mask,
            )

        # BatchedZImageWrapper returns batched tensor [B, C, 1, H, W]
        # Remove frame dimension: [B, C, 1, H, W] -> [B, C, H, W]
        model_pred = model_pred.squeeze(2)

        # Debug: Log VRAM after forward pass (first step only)
        if not hasattr(self, '_debug_logged_gc_vram_after'):
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            print(f"  - VRAM after forward pass: {allocated_after:.2f} GB")
            self._debug_logged_gc_vram_after = True

        if profile_vram:
            print_vram_usage("[train_step_zimage] After Transformer forward")

        # Flow Matching target: velocity = data - noise
        target = latents - noise

        # Debug: Verify target correctness on first step
        if not hasattr(self, '_debug_logged_zimage_target'):
            print(f"{self.log_prefix} Z-Image Flow Matching Target verification:")
            print(f"  - Target formula: velocity = data - noise")
            print(f"  - Target shape: {target.shape}")
            print(f"  - Target dtype: {target.dtype}")
            print(f"  - Latents mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")
            print(f"  - Noise mean: {noise.mean().item():.6f}, std: {noise.std().item():.6f}")
            print(f"  - Velocity mean: {target.mean().item():.6f}, std: {target.std().item():.6f}")
            print(f"[DEBUG] Model prediction verification:")
            print(f"  - model_pred shape: {model_pred.shape}")
            print(f"  - model_pred dtype: {model_pred.dtype}")
            print(f"  - model_pred mean: {model_pred.mean().item():.6f}, std: {model_pred.std().item():.6f}")
            print(f"  - model_pred contains NaN: {torch.isnan(model_pred).any().item()}")
            print(f"  - model_pred contains Inf: {torch.isinf(model_pred).any().item()}")
            print(f"  - model_pred min: {model_pred.min().item():.6f}, max: {model_pred.max().item():.6f}")
            self._debug_logged_zimage_target = True

        # Calculate loss (always in fp32 for numerical stability)
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")

        # Take mean across spatial and channel dimensions first (keep batch dimension)
        # Shape: [B, 16, H, W] -> [B]
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Flow Matching does not use Min-SNR weighting (continuous time formulation already handles weighting)
        # Take mean across batch dimension
        loss = loss_per_sample.mean()

        # Calculate reconstruction loss (for monitoring/visualization)
        # This measures how well the model can reconstruct the target latent from noisy input
        with torch.no_grad():
            predicted_latent_for_recon = noisy_latents + (1.0 - t) * model_pred
            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
            recon_loss = recon_loss_per_sample.mean()

        if profile_vram:
            print_vram_usage("[train_step_zimage] After loss calculation")

        # Debug: Save latents if requested
        if debug_save_path is not None:
            debug_save_path.mkdir(parents=True, exist_ok=True)

            # Save as .pt files with detailed info (save first item in batch)
            timestep_value = timesteps[0].item()

            # Calculate predicted_latent (denoised latent at t=0)
            # Flow Matching: x_1 = x_t + (1 - t) * v
            # Derivation: x_t = x_1 - (1-t) * v  →  x_1 = x_t + (1-t) * v
            with torch.no_grad():
                predicted_latent = noisy_latents + (1.0 - t) * model_pred

            # Prepare debug data
            debug_data = {
                'latents': latents[0:1].detach().cpu(),
                'noisy_latents': noisy_latents[0:1].detach().cpu(),
                'predicted_velocity': model_pred[0:1].detach().cpu(),
                'actual_velocity': target[0:1].detach().cpu(),
                'predicted_latent': predicted_latent[0:1].detach().cpu(),
                'timestep': timestep_value,
                'loss': loss_per_sample[0].item(),
                'loss_batch_mean': loss.item(),
                'recon_loss': recon_loss_per_sample[0].item(),  # Reconstruction loss for first sample
                'recon_loss_batch_mean': recon_loss.item(),  # Batch mean reconstruction loss
                'batch_size': batch_size,
                'scheduler_type': 'FlowMatching',
            }

            # Add caption if available
            if debug_captions is not None and len(debug_captions) > 0:
                debug_data['caption'] = debug_captions[0]
                debug_data['all_captions'] = debug_captions

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:.4f}.pt")

            del predicted_latent
            caption_info = f" (caption: {debug_captions[0][:50]}...)" if debug_captions and len(debug_captions) > 0 else ""
            print(f"[Debug] Saved Z-Image latents to {debug_save_path} (timestep={timestep_value:.4f}){caption_info}")

        # Backward pass
        if profile_vram:
            print_vram_usage("[train_step_zimage] Before backward")

        # Debug: Log VRAM before backward pass (first step only)
        if not hasattr(self, '_debug_logged_gc_vram_before_backward'):
            allocated_before_backward = torch.cuda.memory_allocated() / 1024**3
            print(f"  - VRAM before backward pass: {allocated_before_backward:.2f} GB")
            self._debug_logged_gc_vram_before_backward = True

        self.optimizer.zero_grad()
        loss.backward()

        # Debug: Log VRAM after backward pass (first step only)
        if not hasattr(self, '_debug_logged_gc_vram_after_backward'):
            allocated_after_backward = torch.cuda.memory_allocated() / 1024**3
            print(f"  - VRAM after backward pass: {allocated_after_backward:.2f} GB")
            self._debug_logged_gc_vram_after_backward = True

        if profile_vram:
            print_vram_usage("[train_step_zimage] After backward")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for lora in self.lora_layers.values() for p in lora.parameters()],
            max_norm=1.0
        )

        if profile_vram:
            print_vram_usage("[train_step_zimage] After gradient clipping")

        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()

        # Clear gradients to free VRAM (set_to_none=True for memory efficiency)
        self.optimizer.zero_grad(set_to_none=True)

        # Debug: Log VRAM after optimizer step and zero_grad (first step only)
        if not hasattr(self, '_debug_logged_gc_vram_after_optim'):
            allocated_after_optim = torch.cuda.memory_allocated() / 1024**3
            print(f"  - VRAM after optimizer.step() + zero_grad(): {allocated_after_optim:.2f} GB")
            self._debug_logged_gc_vram_after_optim = True

        if profile_vram:
            print_vram_usage("[train_step_zimage] After optimizer step")

        # Get loss values before cleanup
        loss_value = loss.detach().item()
        recon_loss_value = recon_loss.detach().item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        del noise, noisy_latents, model_pred, target, loss, recon_loss

        return loss_value, recon_loss_value

    def find_latest_checkpoint(self) -> Optional[tuple[str, int]]:
        """
        Find the latest valid checkpoint in output directory.

        Strategy:
        1. Find all .safetensors files
        2. Validate each checkpoint (can be loaded)
        3. Extract step number
        4. Return the one with highest step number

        Returns:
            Tuple of (checkpoint_path, step) or None if no valid checkpoint found
        """
        from safetensors.torch import load_file

        # Find all safetensors files
        checkpoint_files = list(self.output_dir.glob("*.safetensors"))

        if not checkpoint_files:
            return None

        # Validate checkpoints and extract step numbers
        valid_checkpoints = []
        for ckpt_path in checkpoint_files:
            try:
                # Try to load safetensors file (validation)
                state_dict = load_file(str(ckpt_path))

                # Extract step from metadata or filename
                step = 0
                if hasattr(state_dict, 'metadata') and 'ss_training_step' in state_dict.metadata():
                    step = int(state_dict.metadata()['ss_training_step'])
                else:
                    # Fallback: extract from filename (any file with "step_{number}")
                    stem = ckpt_path.stem
                    parts = stem.split("_")
                    if "step" in parts:
                        step_idx = parts.index("step")
                        if step_idx + 1 < len(parts):
                            step = int(parts[step_idx + 1])

                # Check if this checkpoint has LoRA weights (basic validation)
                has_lora_weights = any("lora_down" in key or "lora_up" in key for key in state_dict.keys())
                if has_lora_weights:
                    valid_checkpoints.append((str(ckpt_path), step))
                    print(f"{self.log_prefix} Found valid checkpoint: {ckpt_path.name} (step {step})")

            except Exception as e:
                print(f"{self.log_prefix} Skipping invalid checkpoint {ckpt_path.name}: {e}")
                continue

        if not valid_checkpoints:
            return None

        # Sort by step and return latest
        valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
        latest_ckpt, latest_step = valid_checkpoints[0]

        # Check for optimizer state
        optimizer_path = Path(latest_ckpt).with_suffix('.pt')
        if optimizer_path.exists():
            print(f"{self.log_prefix} Latest checkpoint: {latest_ckpt} (step {latest_step}, with optimizer state)")
        else:
            print(f"{self.log_prefix} Latest checkpoint: {latest_ckpt} (step {latest_step}, no optimizer state)")

        return latest_ckpt, latest_step

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load LoRA checkpoint from safetensors file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Step number from checkpoint
        """
        from safetensors.torch import load_file

        print(f"{self.log_prefix} Loading checkpoint from {checkpoint_path}")

        state_dict = load_file(checkpoint_path)

        # Load weights into LoRA layers
        loaded_count = 0
        for name, lora in self.lora_layers.items():
            # Parse prefix and module name (same logic as save_checkpoint)
            if "." in name:
                prefix, module_name = name.split(".", 1)
            else:
                # Fallback for legacy keys without prefix
                prefix = "unet"
                module_name = name

            # Generate key prefix based on module type (same as save_checkpoint)
            # Use diffusers format (compatible with save_checkpoint format)
            if prefix == "unet":
                key_prefix = f"unet.{module_name}"
            elif prefix == "transformer":
                # Z-Image transformer (FlowDiT)
                key_prefix = f"transformer.{module_name}"
            elif prefix == "te1":
                key_prefix = f"text_encoder.{module_name}"
            elif prefix == "te2":
                key_prefix = f"text_encoder_2.{module_name}"
            else:
                # Unknown prefix, use as-is
                key_prefix = f"{prefix}.{module_name}"

            down_key = f"{key_prefix}.lora_down.weight"
            up_key = f"{key_prefix}.lora_up.weight"

            # Try to load with the generated key
            if down_key in state_dict and up_key in state_dict:
                lora.lora_down.weight.data = state_dict[down_key].to(self.device)
                lora.lora_up.weight.data = state_dict[up_key].to(self.device)
                loaded_count += 1
            else:
                print(f"{self.log_prefix} WARNING: Keys not found for {name}: {down_key}")

        print(f"{self.log_prefix} Loaded {loaded_count}/{len(self.lora_layers)} LoRA layers from checkpoint")

        # Extract step from metadata or filename
        step = 0
        if hasattr(state_dict, 'metadata') and 'ss_training_step' in state_dict.metadata():
            step = int(state_dict.metadata()['ss_training_step'])
        else:
            # Fallback: extract from filename
            try:
                step_str = Path(checkpoint_path).stem.split("_")[-1]
                step = int(step_str)
            except (ValueError, IndexError):
                pass

        print(f"{self.log_prefix} Checkpoint loaded (step {step})")

        # Try to load optimizer state if it exists
        optimizer_path = Path(checkpoint_path).with_suffix('.pt')
        if optimizer_path.exists() and hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                print(f"{self.log_prefix} Loading optimizer state from {optimizer_path}")
                optimizer_state = torch.load(optimizer_path, map_location=self.device)
                self.optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
                print(f"{self.log_prefix} Optimizer state loaded successfully")
            except Exception as e:
                print(f"{self.log_prefix} WARNING: Failed to load optimizer state: {e}")
                print(f"{self.log_prefix} Training will continue with fresh optimizer state")
        else:
            if not optimizer_path.exists():
                print(f"{self.log_prefix} No optimizer state found at {optimizer_path}, using fresh optimizer state")

        return step

    def save_checkpoint(self, step: int, save_path: Optional[str] = None, save_optimizer: bool = True, max_to_keep: Optional[int] = None, save_every: int = 100, run_id: Optional[int] = None, epoch: Optional[int] = None):
        """
        Save LoRA checkpoint as safetensors and optimizer state as .pt.

        Args:
            step: Current training step
            save_path: Path to save checkpoint (default: output_dir/{run_name}_step_{step}.safetensors)
            save_optimizer: Whether to save optimizer state (default: True)
            max_to_keep: Maximum number of checkpoints to keep (None = keep all)
            save_every: Save interval (used to calculate which checkpoint to delete)
            run_id: Training run ID (for DB registration)
            epoch: Current epoch (for DB registration)
        """
        if save_path is None:
            # Extract short name from run_name
            # If run_name is in format "YYYYMMDD_HHMMSS_ID", use only ID
            # Otherwise, use full run_name
            import re
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)  # Extract ID part
            else:
                short_name = self.run_name  # Use full name

            save_path = self.output_dir / f"{short_name}_step_{step}.safetensors"
        else:
            save_path = Path(save_path)

        print(f"{self.log_prefix} Saving checkpoint to {save_path}")
        print(f"{self.log_prefix} Converting weights to {self.output_dtype} for saving")

        # Collect all LoRA weights and convert to output_dtype
        state_dict = {}
        for name, lora in self.lora_layers.items():
            # Parse prefix and module name (e.g., "unet.down_blocks.0..." or "te1.text_model.encoder...")
            if "." in name:
                prefix, module_name = name.split(".", 1)
            else:
                # Fallback for legacy keys without prefix
                prefix = "unet"
                module_name = name

            # Generate key in diffusers format (compatible with diffusers library's load_lora_weights)
            # diffusers expects keys like: "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k"
            # Z-Image: "transformer.layers.0.self_attn_qkv.to_q"
            # NOT SD format like: "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k"

            if prefix == "unet":
                key_prefix = f"unet.{module_name}"
            elif prefix == "transformer":
                # Z-Image transformer (FlowDiT)
                key_prefix = f"transformer.{module_name}"
            elif prefix == "te1":
                key_prefix = f"text_encoder.{module_name}"
            elif prefix == "te2":
                key_prefix = f"text_encoder_2.{module_name}"
            else:
                # Unknown prefix, use as-is
                key_prefix = f"{prefix}.{module_name}"

            # Convert to output_dtype for saving (e.g., fp16 to reduce file size)
            state_dict[f"{key_prefix}.lora_down.weight"] = lora.lora_down.weight.detach().cpu().to(dtype=self.output_dtype)
            state_dict[f"{key_prefix}.lora_up.weight"] = lora.lora_up.weight.detach().cpu().to(dtype=self.output_dtype)

            # Add alpha value (LoRA scaling parameter)
            state_dict[f"{key_prefix}.alpha"] = torch.tensor(self.lora_alpha, dtype=self.output_dtype)

        # Add metadata (diffusers-compatible format)
        metadata = {
            "format": "diffusers",  # Indicate this is diffusers format, not SD format
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "base_model": self.model_path,
            "training_step": str(step),
            "output_dtype": str(self.output_dtype),
        }

        # Save as safetensors
        save_file(state_dict, str(save_path), metadata=metadata)
        print(f"{self.log_prefix} Checkpoint saved: {save_path}")

        # Get file size
        file_size = save_path.stat().st_size

        # Save optimizer state separately as .pt
        if save_optimizer and hasattr(self, 'optimizer') and self.optimizer is not None:
            optimizer_path = save_path.with_suffix('.pt')
            optimizer_state = {
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step': step,
            }
            torch.save(optimizer_state, optimizer_path)
            print(f"{self.log_prefix} Optimizer state saved: {optimizer_path}")

        # Register checkpoint in database
        if run_id is not None:
            try:
                from database import get_training_db
                from database.models import TrainingCheckpoint

                db = next(get_training_db())
                try:
                    checkpoint_record = TrainingCheckpoint(
                        run_id=run_id,
                        checkpoint_name=save_path.name,
                        step=step,
                        epoch=epoch,
                        file_path=str(save_path),
                        file_size=file_size,
                        loss=None  # Loss can be added if tracked
                    )
                    db.add(checkpoint_record)
                    db.commit()
                    print(f"{self.log_prefix} Checkpoint registered in DB: run_id={run_id}, step={step}")
                except Exception as e:
                    print(f"{self.log_prefix} WARNING: Failed to register checkpoint in DB: {e}")
                    db.rollback()
                finally:
                    db.close()
            except Exception as e:
                print(f"{self.log_prefix} WARNING: Failed to connect to DB for checkpoint registration: {e}")

        # Remove old checkpoints if max_to_keep is set
        if max_to_keep is not None and max_to_keep > 0:
            self._cleanup_old_checkpoints(step, max_to_keep, save_every)

    def _cleanup_old_checkpoints(self, current_step: int, max_to_keep: int, save_every: int):
        """
        Remove old checkpoints to keep only the latest N checkpoints.

        Args:
            current_step: Current training step
            max_to_keep: Maximum number of checkpoints to keep
            save_every: Save interval (used to calculate which checkpoint to delete)
        """
        # Calculate which step to remove
        # Example: save_every=100, max_to_keep=10
        # At step 1100, keep checkpoints from 1100, 1000, 900, 800, ..., 200
        # Remove checkpoint from step 100
        remove_step = current_step - (save_every * max_to_keep)

        if remove_step < save_every:
            # No checkpoint to remove yet
            return

        # Extract short name from run_name (same logic as save_checkpoint)
        import re
        match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
        if match:
            short_name = match.group(1)
        else:
            short_name = self.run_name

        # Remove checkpoint at remove_step
        checkpoint_path = self.output_dir / f"{short_name}_step_{remove_step}.safetensors"
        optimizer_path = self.output_dir / f"{short_name}_step_{remove_step}.pt"

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print(f"{self.log_prefix} Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"{self.log_prefix} WARNING: Failed to remove old checkpoint {checkpoint_path}: {e}")

        if optimizer_path.exists():
            try:
                optimizer_path.unlink()
                print(f"{self.log_prefix} Removed old optimizer state: {optimizer_path}")
            except Exception as e:
                print(f"{self.log_prefix} WARNING: Failed to remove old optimizer state {optimizer_path}: {e}")

    def generate_sample(self, step: int, sample_prompts: List[Dict[str, str]], config: Dict[str, Any], vae_on_cpu: bool = False):
        """
        Generate sample images during training using custom sampling loop.

        Args:
            step: Current training step
            sample_prompts: List of prompt dicts with 'positive' and 'negative' keys
            config: Generation configuration (width, height, steps, cfg_scale, sampler, etc.)
        """
        if not sample_prompts:
            return

        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        print(f"{self.log_prefix} Generating {len(sample_prompts)} samples at step {step}...")

        # Z-Image uses different sample generation pipeline
        if self.is_zimage:
            return self._generate_sample_zimage(step, sample_prompts, config, vae_on_cpu)

        # Extract config parameters
        width = config.get("width", 1024)
        height = config.get("height", 1024)
        num_steps = config.get("steps", 28)
        cfg_scale = config.get("cfg_scale", 7.0)
        sampler = config.get("sampler", "euler")
        schedule_type = config.get("schedule_type", "sgm_uniform")
        seed = config.get("seed", -1)

        # Set components to eval mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()

        # Create temporary pipeline for prompt encoding and component access
        # Use a dummy inference scheduler (will be replaced by get_scheduler)
        from diffusers import EulerDiscreteScheduler
        dummy_scheduler = EulerDiscreteScheduler.from_config({
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
        })

        if self.is_sdxl:
            from diffusers import StableDiffusionXLPipeline
            temp_pipeline = StableDiffusionXLPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                unet=self.unet,
                scheduler=dummy_scheduler,
            )
        else:
            from diffusers import StableDiffusionPipeline
            temp_pipeline = StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=dummy_scheduler,
            )

        # Replace with user-selected scheduler and schedule_type
        # IMPORTANT: Use inference scheduler for sample generation, not training scheduler (DDPM)
        from core.inference.schedulers import get_scheduler
        temp_pipeline.scheduler = get_scheduler(temp_pipeline, sampler, schedule_type)
        print(f"{self.log_prefix} Using scheduler: {temp_pipeline.scheduler.__class__.__name__} (sampler={sampler}, schedule_type={schedule_type})")

        # VRAM optimization: Use sequential component loading
        print(f"{self.log_prefix} Applying VRAM optimization (sequential component loading)")

        # Import VRAM optimization functions and custom sampling loop
        from core.vram_optimization import (
            move_text_encoders_to_gpu,
            move_text_encoders_to_cpu,
            move_unet_to_gpu,
            move_unet_to_cpu,
            move_vae_to_gpu,
            move_vae_to_cpu,
            log_device_status,
        )
        from core.inference.custom_sampling import custom_sampling_loop

        # Start with components on CPU for VRAM optimization
        move_text_encoders_to_cpu(temp_pipeline)
        move_unet_to_cpu(temp_pipeline)
        move_vae_to_cpu(temp_pipeline)
        torch.cuda.empty_cache()

        log_device_status("Sample generation - Initial state (all on CPU)", temp_pipeline)

        try:
            for i, prompt_pair in enumerate(sample_prompts):
                positive_prompt = prompt_pair.get("positive", "")
                negative_prompt = prompt_pair.get("negative", "")

                # Generate seed
                if seed == -1:
                    gen_seed = torch.randint(0, 2**32 - 1, (1,)).item()
                else:
                    gen_seed = seed + i

                # Create generator (device will be checked in custom_sampling_loop)
                generator = torch.Generator(device=self.device).manual_seed(gen_seed)

                print(f"{self.log_prefix} Encoding prompts for sample {i}...")

                # Move text encoders to GPU for prompt encoding
                move_text_encoders_to_gpu(temp_pipeline)

                # Encode prompts using pipeline's encode_prompt method
                prompt_embeds_tuple = temp_pipeline.encode_prompt(
                    prompt=positive_prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )
                prompt_embeds = prompt_embeds_tuple[0]
                pooled_prompt_embeds = prompt_embeds_tuple[2] if len(prompt_embeds_tuple) > 2 and self.is_sdxl else None

                # Encode negative prompt
                if negative_prompt:
                    neg_embeds_tuple = temp_pipeline.encode_prompt(
                        prompt=negative_prompt,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False
                    )
                    negative_prompt_embeds = neg_embeds_tuple[0]
                    negative_pooled_prompt_embeds = neg_embeds_tuple[2] if len(neg_embeds_tuple) > 2 and self.is_sdxl else None
                else:
                    negative_prompt_embeds = None
                    negative_pooled_prompt_embeds = None

                # Move text encoders back to CPU to free VRAM
                move_text_encoders_to_cpu(temp_pipeline)

                # Move UNet and VAE to GPU for inference
                move_unet_to_gpu(temp_pipeline, quantization=None, use_torch_compile=False)
                move_vae_to_gpu(temp_pipeline)

                log_device_status(f"Sample {i} - Ready for inference", temp_pipeline)

                print(f"{self.log_prefix} Generating sample {i} using custom_sampling_loop...")

                # Generate image using custom sampling loop (same as SushiUI)
                with torch.no_grad():
                    image = custom_sampling_loop(
                        pipeline=temp_pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        num_inference_steps=num_steps,
                        guidance_scale=cfg_scale,
                        guidance_rescale=0.0,
                        width=width,
                        height=height,
                        generator=generator,
                        ancestral_generator=None,
                        latents=None,
                        prompt_embeds_callback=None,
                        progress_callback=None,
                        step_callback=None,
                        developer_mode=False,
                        controlnet_images=None,
                        controlnet_conditioning_scale=None,
                        control_guidance_start=None,
                        control_guidance_end=None,
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
                    )

                # Move components back to CPU after generation
                move_unet_to_cpu(temp_pipeline)
                move_vae_to_cpu(temp_pipeline)
                torch.cuda.empty_cache()

                log_device_status(f"Sample {i} - After generation (moved to CPU)", temp_pipeline)

                # Save image
                sample_filename = f"step_{step:06d}_sample_{i}.png"
                sample_path = samples_dir / sample_filename
                image.save(sample_path)

                # Log to TensorBoard
                import torchvision
                image_tensor = torchvision.transforms.ToTensor()(image)
                self.writer.add_image(f"samples/sample_{i}", image_tensor, global_step=step)

                print(f"{self.log_prefix} Sample {i} saved: {sample_path}")

        except Exception as e:
            print(f"{self.log_prefix} ERROR generating sample: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temporary pipeline
            del temp_pipeline
            torch.cuda.empty_cache()

            # IMPORTANT: Restore all components to GPU for training
            # (sample generation may have moved them to CPU)
            print(f"{self.log_prefix} Restoring components for training")

            # VAE: Keep on CPU if latent caching is enabled, otherwise move to GPU
            if vae_on_cpu:
                print(f"{self.log_prefix} Keeping VAE on CPU (latent caching enabled)")
                self.vae.to('cpu')
            else:
                self.vae.to(self.device)

            # Text encoders and U-Net always go to GPU
            self.text_encoder.to(self.device)
            self.unet.to(self.device)
            if self.text_encoder_2 is not None:
                self.text_encoder_2.to(self.device)
            torch.cuda.empty_cache()

            print(f"{self.log_prefix} Components restored for training")

        # Set back to train mode
        self.unet.train()

    def _generate_sample_zimage(self, step: int, sample_prompts: List[Dict[str, str]], config: Dict[str, Any], vae_on_cpu: bool = False):
        """
        Generate sample images during Z-Image training using custom denoising loop.
        Works for both LoRA and full fine-tuning by temporarily merging LoRA weights into transformer_original.

        Args:
            step: Current training step
            sample_prompts: List of prompt dicts with 'positive' and 'negative' keys
            config: Generation configuration (width, height, steps, cfg_scale, sampler, etc.)
            vae_on_cpu: Whether VAE should stay on CPU after generation
        """
        if not sample_prompts:
            return

        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        print(f"{self.log_prefix} Generating {len(sample_prompts)} Z-Image samples at step {step}...")

        # Extract config parameters
        width = config.get("width", 1024)
        height = config.get("height", 1024)
        num_steps = config.get("steps", 20)
        cfg_scale = config.get("cfg_scale", 1.0)  # Z-Image default: CFG=1
        sampler = config.get("sampler", "euler")
        schedule_type = config.get("schedule_type", "uniform")
        seed = config.get("seed", -1)

        # Set components to eval mode
        self.transformer.eval()
        self.transformer_original.eval()
        self.vae.eval()
        self.text_encoder.eval()

        print(f"{self.log_prefix} Z-Image sample: {width}x{height}, steps={num_steps}, cfg={cfg_scale}")

        # Import Z-Image denoising loop and VRAM functions
        from core.vram_optimization import (
            move_zimage_text_encoder_to_gpu,
            move_zimage_text_encoder_to_cpu,
            move_zimage_transformer_to_gpu,
            move_zimage_transformer_to_cpu,
            move_zimage_vae_to_gpu,
            move_zimage_vae_to_cpu,
        )

        # Start with components on CPU for VRAM optimization
        self.text_encoder.to('cpu')
        self.transformer_original.to('cpu')
        self.vae.to('cpu')
        torch.cuda.empty_cache()

        # Temporary merge LoRA into transformer_original for inference
        # This allows using the unwrapped transformer with LoRA applied
        lora_merged = False
        if self.training_method == "lora":
            print(f"{self.specific_log_prefix} Temporarily merging LoRA weights into transformer for inference...")
            try:
                from peft import get_peft_model
                # Merge LoRA weights from self.transformer (wrapper) into self.transformer_original
                # Access the wrapped transformer's state dict
                wrapped_transformer = self.transformer.transformer if hasattr(self.transformer, 'transformer') else self.transformer
                # Copy LoRA-applied state to transformer_original
                lora_state = {k: v.clone() for k, v in wrapped_transformer.state_dict().items()}
                self.transformer_original.load_state_dict(lora_state, strict=False)
                lora_merged = True
                print(f"{self.specific_log_prefix} LoRA weights merged successfully")
            except Exception as e:
                print(f"{self.specific_log_prefix} WARNING: Could not merge LoRA weights: {e}")
                print(f"{self.log_prefix} Samples will use base model without LoRA")

        try:
            for i, prompt_pair in enumerate(sample_prompts):
                positive_prompt = prompt_pair.get("positive", "")
                negative_prompt = prompt_pair.get("negative", "")

                # Generate seed
                if seed == -1:
                    gen_seed = torch.randint(0, 2**32 - 1, (1,)).item()
                else:
                    gen_seed = seed + i

                generator = torch.Generator(device=self.device).manual_seed(gen_seed)

                print(f"{self.log_prefix} Generating Z-Image sample {i} (seed={gen_seed})...")

                # Move Text Encoder to GPU
                print(f"{self.log_prefix} Encoding prompts...")
                move_zimage_text_encoder_to_gpu(self.text_encoder)

                # Encode prompts
                prompt_embeds_list = []
                negative_prompt_embeds_list = []

                # Encode positive prompt
                embeds, mask = self.encode_prompt_zimage(positive_prompt)
                prompt_embeds_list.append(embeds)

                # Encode negative prompt if CFG > 1
                do_cfg = abs(cfg_scale - 1.0) > 1e-5
                if do_cfg:
                    if negative_prompt:
                        neg_embeds, neg_mask = self.encode_prompt_zimage(negative_prompt)
                    else:
                        neg_embeds, neg_mask = self.encode_prompt_zimage("")
                    negative_prompt_embeds_list.append(neg_embeds)

                # Move Text Encoder to CPU
                move_zimage_text_encoder_to_cpu(self.text_encoder)

                # Move Transformer and VAE to GPU
                print(f"{self.log_prefix} Moving Transformer and VAE to GPU...")
                move_zimage_transformer_to_gpu(self.transformer_original)
                move_zimage_vae_to_gpu(self.vae)

                # Run Z-Image denoising loop (using local zimage_utils, no external dependencies)
                print(f"{self.log_prefix} Running Z-Image denoising loop...")
                with torch.no_grad():
                    latents = self._run_zimage_denoising_loop(
                        transformer=self.transformer_original,
                        scheduler=self.noise_scheduler,
                        prompt_embeds_list=prompt_embeds_list,
                        negative_prompt_embeds_list=negative_prompt_embeds_list if do_cfg else [],
                        height=height,
                        width=width,
                        num_inference_steps=num_steps,
                        guidance_scale=cfg_scale,
                        do_classifier_free_guidance=do_cfg,
                        generator=generator,
                    )

                    # Decode latents to image
                    print(f"{self.log_prefix} Decoding latents to image...")
                    image = self._decode_zimage_latents(latents, self.vae)

                # Move components to CPU
                move_zimage_transformer_to_cpu(self.transformer_original)
                move_zimage_vae_to_cpu(self.vae)
                torch.cuda.empty_cache()

                # Save image
                sample_filename = f"step_{step:06d}_sample_{i}.png"
                sample_path = samples_dir / sample_filename
                image.save(sample_path)

                # Log to TensorBoard
                import torchvision
                image_tensor = torchvision.transforms.ToTensor()(image)
                self.writer.add_image(f"samples/sample_{i}", image_tensor, global_step=step)

                print(f"{self.log_prefix} Sample {i} saved: {sample_path}")

        except Exception as e:
            print(f"{self.log_prefix} ERROR generating Z-Image sample: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore transformer_original to original state (unmerge LoRA)
            if lora_merged and self.training_method == "lora":
                print(f"{self.log_prefix} Restoring transformer_original to original state...")
                # Reload original weights (base model without LoRA)
                # This is automatically done since we only modified in-memory state

            # Restore components for training
            print(f"{self.log_prefix} Restoring Z-Image components for training")

            # Text Encoder: Keep on CPU (frozen)
            self.text_encoder.to('cpu')

            # Transformer: Move both wrapper and original to GPU
            # IMPORTANT: Move transformer_original first, then wrapper
            # This ensures the wrapper's reference to transformer is also on GPU
            self.transformer_original.to(self.device)
            self.transformer.to(self.device)

            # VAE: Keep on CPU if latent caching is enabled
            if vae_on_cpu:
                print(f"{self.log_prefix} Keeping VAE on CPU (latent caching enabled)")
                self.vae.to('cpu')
            else:
                self.vae.to(self.device)

            torch.cuda.empty_cache()
            print(f"{self.log_prefix} Z-Image components restored for training")

        # Set back to train mode
        self.transformer.train()
        self.transformer_original.train()

    def _run_zimage_denoising_loop(
        self, transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
        height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
        generator
    ):
        """
        Run Z-Image denoising loop (simplified version for training samples).
        Adapted from DiffusionPipelineManager._zimage_denoising_loop()
        Uses local zimage_utils (no external dependencies).
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

        device = torch.device(self.device)

        # Calculate VAE scale factor
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels"):
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            vae_scale_factor = 8
        vae_scale = vae_scale_factor * 2

        # Calculate latent dimensions
        height_latent = 2 * (int(height) // vae_scale)
        width_latent = 2 * (int(width) // vae_scale)
        batch_size = len(prompt_embeds_list)
        shape = (batch_size, transformer.in_channels, height_latent, width_latent)

        # Initialize random latents
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

        # Calculate dynamic shift for flow matching (using local implementation or fallback)
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(image_seq_len)

        # Set timesteps with flow matching (0.0 to 1.0)
        scheduler.set_timesteps(num_inference_steps, mu=mu)
        timesteps = scheduler.timesteps.to(device)

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Normalize timestep to [0, 1] (same as inference pipeline)
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            timestep_model_input = timestep

            # Prepare model input with CFG (same as inference pipeline)
            # Convert to transformer's dtype
            transformer_dtype = next(transformer.parameters()).dtype

            if do_classifier_free_guidance:
                # CFG: duplicate latents and concatenate [negative, positive] (same order as inference)
                latent_model_input = latents.to(transformer_dtype).repeat(2, 1, 1, 1)
                # CFG input order: [negative, positive] (consistent with inference)
                model_input_list = negative_prompt_embeds_list + prompt_embeds_list
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(transformer_dtype)
                model_input_list = prompt_embeds_list
                timestep_model_input = timestep

            # Add frames dimension for Z-Image (same as inference pipeline)
            # Z-Image expects [batch, channels, frames, height, width]
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            # Call transformer (same as inference pipeline)
            with torch.no_grad():
                model_out_list = transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    model_input_list,
                )[0]

            # Apply CFG if enabled (same as inference pipeline)
            if do_classifier_free_guidance:
                # CFG output order matches input: [negative, positive]
                neg_out = model_out_list[:batch_size]  # negative (uncond)
                pos_out = model_out_list[batch_size:]  # positive (cond)
                noise_pred = []
                for j in range(batch_size):
                    neg = neg_out[j].float()
                    pos = pos_out[j].float()
                    # Standard CFG formula (consistent with inference)
                    # pred = uncond + guidance_scale * (cond - uncond)
                    pred = neg + guidance_scale * (pos - neg)
                    noise_pred.append(pred)
                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

            # Remove frames dimension for scheduler (5D → 4D) and negate (same as inference pipeline)
            noise_pred = -noise_pred.squeeze(2)

            # Scheduler step (same as inference pipeline)
            latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

        return latents

    def _decode_zimage_latents(self, latents, vae):
        """Decode Z-Image latents to PIL Image."""
        # Apply VAE scaling and shift (must match inference pipeline)
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        latents = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor

        # VAE decode
        with torch.no_grad():
            image = vae.decode(latents, return_dict=False)[0]

        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")

        from PIL import Image
        return Image.fromarray(image)

    def train(
        self,
        dataset_items: List[Dict[str, Any]],
        num_epochs: int = 1,
        target_steps: Optional[int] = None,
        batch_size: int = 1,
        save_every: int = 100,
        save_every_unit: str = "steps",
        sample_every: int = 100,
        sample_prompts: List[str] = None,
        sample_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
        update_total_steps_callback: Optional[Callable[[int], None]] = None,  # Called once when total_steps is determined
        reload_dataset_callback: Optional[Callable[[int], List[Dict[str, Any]]]] = None,  # Called each epoch to reload dataset with new captions
        resume_from_checkpoint: Optional[str] = None,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
        # Bucketing parameters
        enable_bucketing: bool = False,
        base_resolutions: List[int] = None,
        bucket_strategy: str = "resize",
        multi_resolution_mode: str = "max",
        # Latent caching
        cache_latents_to_disk: bool = True,
        cache_text_embeds_to_disk: bool = False,  # Reserved for future use, not exposed in UI
        dataset_unique_ids: Optional[List[str]] = None,  # List of dataset unique IDs for cache management
        force_recache: bool = False,  # Force regenerate cache even if valid cache exists
        # Checkpoint management
        max_step_saves_to_keep: Optional[int] = None,  # Maximum number of checkpoints to keep (None = keep all)
        run_id: Optional[int] = None,  # Training run ID for DB registration
    ):
        """
        Train LoRA on dataset.

        Args:
            dataset_items: List of dataset items (image_path, caption, width, height)
            num_epochs: Number of epochs (if None, will be calculated from target_steps)
            target_steps: Target total steps (overrides num_epochs if provided)
            batch_size: Batch size (currently only supports 1)
            save_every: Save checkpoint every N steps or epochs
            save_every_unit: Unit for save_every ("steps" or "epochs")
            sample_every: Generate sample every N steps
            sample_prompts: List of prompts for sample generation
            sample_config: Configuration for sample generation (width, height, steps, cfg_scale, etc.)
            progress_callback: Callback(step, loss, lr) for progress updates
            resume_from_checkpoint: Checkpoint filename to resume from (e.g., "lora_step_100.safetensors")
            debug_latents: Enable debug latent saving
            debug_latents_every: Save debug latents every N steps
            enable_bucketing: Enable aspect ratio bucketing
            base_resolutions: List of base resolutions for bucketing (e.g., [512, 768, 1024])
            bucket_strategy: Strategy for oversized images ("resize", "crop", "random_crop")
            multi_resolution_mode: How to assign images to resolutions ("max", "random")
        """
        print(f"{self.log_prefix} Starting training")
        print(f"{self.log_prefix} Dataset: {len(dataset_items)} items")
        print(f"{self.log_prefix} Epochs: {num_epochs}")
        print(f"{self.log_prefix} Batch size: {batch_size}")
        print(f"{self.log_prefix} Debug latents: {debug_latents} (every {debug_latents_every} steps)")
        print(f"{self.log_prefix} Bucketing: {enable_bucketing}")

        # Create debug directory if debug mode is enabled
        debug_dir = None
        if debug_latents:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            print(f"{self.log_prefix} Debug latents will be saved to: {debug_dir}")

        # Setup bucketing if enabled
        bucket_manager = None
        if enable_bucketing:
            from core.training.bucketing import BucketManager

            if base_resolutions is None:
                base_resolutions = [1024]  # Default to SDXL base

            bucket_manager = BucketManager(
                base_resolutions=base_resolutions,
                divisibility=8,
                strategy=bucket_strategy,
                multi_resolution_mode=multi_resolution_mode
            )

            print(f"{self.log_prefix} Bucketing enabled with resolutions: {base_resolutions}")
            print(f"{self.log_prefix} Bucket strategy: {bucket_strategy}")
            print(f"{self.log_prefix} Multi-resolution mode: {multi_resolution_mode}")

            # Assign all images to buckets
            print(f"{self.log_prefix} Assigning images to buckets...")
            for idx, item in enumerate(dataset_items):
                width = item.get("width", 1024)
                height = item.get("height", 1024)

                # Debug: Log first 3 images to verify width/height are correct
                if idx < 3:
                    print(f"{self.log_prefix} Image {idx}: {width}x{height} - {item['image_path']}")

                bucket_manager.assign_image_to_bucket(
                    image_path=item["image_path"],
                    width=width,
                    height=height,
                    caption=item.get("caption", ""),
                    dataset_unique_id=item.get("dataset_unique_id")
                )

            # Print bucket statistics
            bucket_counts = bucket_manager.get_bucket_counts()
            print(f"{self.log_prefix} Bucket distribution:")
            for bucket_size, count in sorted(bucket_counts.items()):
                print(f"  {bucket_size}: {count} images")

            # Shuffle buckets and build batches
            bucket_manager.shuffle_buckets()
            batches = bucket_manager.build_batch_indices(batch_size)
            print(f"{self.log_prefix} Created {len(batches)} batches from {len(dataset_items)} images (batch_size={batch_size})")

        else:
            # No bucketing: create simple batches from dataset_items
            batches = []
            for start_idx in range(0, len(dataset_items), batch_size):
                end_idx = min(start_idx + batch_size, len(dataset_items))
                batches.append(dataset_items[start_idx:end_idx])
            print(f"{self.log_prefix} Created {len(batches)} batches (no bucketing, batch_size={batch_size})")

        # Calculate epochs and total steps based on actual batch count
        steps_per_epoch = len(batches)

        if target_steps is not None:
            # Calculate epochs needed to reach target steps
            # Always train in full epoch units (never stop mid-epoch)
            # If target_steps < steps_per_epoch, train for 1 full epoch
            # Otherwise, train for the number of epochs that fit within target_steps
            if target_steps < steps_per_epoch:
                num_epochs = 1
                total_steps = steps_per_epoch
                print(f"{self.log_prefix} Target steps: {target_steps}, but will complete 1 full epoch ({steps_per_epoch} steps)")
            else:
                # Calculate how many full epochs fit within target_steps
                num_epochs = target_steps // steps_per_epoch
                total_steps = steps_per_epoch * num_epochs
                print(f"{self.log_prefix} Target steps: {target_steps}, will train for {num_epochs} full epochs ({total_steps} steps)")
        elif num_epochs is None:
            # Fallback: default to 1 epoch
            num_epochs = 1
            total_steps = steps_per_epoch
            print(f"{self.log_prefix} No target_steps or num_epochs provided, defaulting to 1 epoch ({total_steps} steps)")
        else:
            # num_epochs was explicitly provided
            total_steps = steps_per_epoch * num_epochs

        print(f"{self.log_prefix} Training plan:")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Total steps: {total_steps}")

        # Notify total_steps to caller (for DB update)
        if update_total_steps_callback:
            update_total_steps_callback(total_steps)

        # Generate latent cache if enabled (per-dataset management)
        latent_caches = {}  # Map: dataset_unique_id -> LatentCache instance
        image_to_cache_id = {}  # Map: image_path -> dataset_unique_id

        if cache_latents_to_disk and dataset_unique_ids is not None and len(dataset_unique_ids) > 0:
            from core.training.latent_cache import LatentCache

            print(f"{self.log_prefix} Initializing {len(dataset_unique_ids)} latent cache(s)...")

            # Create LatentCache instance for each dataset
            for unique_id in dataset_unique_ids:
                latent_caches[unique_id] = LatentCache(dataset_unique_id=unique_id)
                print(f"{self.log_prefix}   Dataset {unique_id[:8]}...: {latent_caches[unique_id].cache_dir}")

            # Build image_path -> dataset_unique_id mapping from dataset_items
            print(f"{self.log_prefix} Building image-to-cache mapping...")
            for batch in batches:
                for item in batch:
                    image_path = item["image_path"]
                    dataset_unique_id = item.get("dataset_unique_id")
                    if dataset_unique_id:
                        image_to_cache_id[image_path] = dataset_unique_id

            print(f"{self.log_prefix} Mapped {len(image_to_cache_id)} images to {len(latent_caches)} cache(s)")

            # Check cache validity and generate if needed for each dataset
            # Determine model type for cache validation
            if self.is_zimage:
                model_type = "zimage"
            elif self.is_sdxl:
                model_type = "sdxl"
            else:
                model_type = "sd15"
            caches_to_generate = []

            for unique_id, cache in latent_caches.items():
                # Force recache if requested
                if force_recache:
                    print(f"[LatentCache] force_recache=True, invalidating cache for dataset {unique_id[:8]}...")
                    caches_to_generate.append((unique_id, cache))
                    continue

                # Check metadata validity (model path, model type)
                if not cache.is_valid(self.model_path, model_type):
                    print(f"[LatentCache] Cache metadata invalid for dataset {unique_id[:8]}...")
                    caches_to_generate.append((unique_id, cache))
                    continue

                # Validate cache format by random sampling
                # Z-Image uses 16 channels, SD/SDXL use 4 channels
                expected_channels = 16 if self.is_zimage else 4
                if not cache.validate_cache_format(expected_channels=expected_channels, sample_count=5):
                    print(f"[LatentCache] Cache format validation failed for dataset {unique_id[:8]}...")
                    caches_to_generate.append((unique_id, cache))
                    continue

            if len(caches_to_generate) > 0:
                print(f"\n{'='*80}")
                print(f"[LatentCache] Cache validation failed for {len(caches_to_generate)} dataset(s)")
                print(f"[LatentCache] Regenerating cache (this will take some time but significantly reduces VRAM during training)")
                print(f"[LatentCache] Model: {self.model_path}")
                print(f"[LatentCache] Model type: {model_type}")
                print(f"{'='*80}\n")

                # Clear old caches
                for unique_id, cache in caches_to_generate:
                    print(f"[LatentCache] Clearing old cache for dataset {unique_id[:8]}...")
                    cache.clear()

                # Move VAE to GPU for encoding
                print(f"{self.log_prefix} Moving VAE to GPU for cache generation...")
                self.vae.to(self.device)

                # Generate cache for all images
                total_images = sum(len(batch) for batch in batches)
                total_new_cached = 0
                total_existing_cached = 0

                # Progress tracking (print every 10%)
                import sys
                processed_images = 0
                last_progress_percent = 0

                print(f"[LatentCache] Encoding {total_images} images...")

                for batch in batches:
                    for item in batch:
                        image_path = item["image_path"]
                        dataset_unique_id = item.get("dataset_unique_id")

                        if not dataset_unique_id or dataset_unique_id not in latent_caches:
                            print(f"[LatentCache] WARNING: No cache for image {image_path}")
                            processed_images += 1
                            continue

                        # Get target dimensions
                        if "bucket_width" in item and "bucket_height" in item:
                            target_width = item["bucket_width"]
                            target_height = item["bucket_height"]
                        else:
                            target_width = item.get("width", 1024)
                            target_height = item.get("height", 1024)

                        # Get cache for this dataset
                        cache = latent_caches[dataset_unique_id]

                        # Check if already cached
                        cached_latent = cache.load_latent(
                            image_path, target_width, target_height, device=self.device
                        )

                        if cached_latent is None:
                            if not os.path.exists(image_path):
                                print(f"[LatentCache] WARNING: Image not found: {image_path}")
                                processed_images += 1
                                continue

                            try:
                                image = Image.open(image_path)
                                image.verify()
                                image = Image.open(image_path)

                                latents = self.encode_image(
                                    image, target_width=target_width, target_height=target_height
                                )
                                cache.save_latent(
                                    image_path, target_width, target_height, latents
                                )
                                total_new_cached += 1
                            except Exception as e:
                                print(f"[LatentCache] ERROR: Failed to encode {image_path}: {e}")
                        else:
                            total_existing_cached += 1

                        processed_images += 1

                        # Print progress every 10%
                        current_progress_percent = (processed_images * 100) // total_images
                        if current_progress_percent >= last_progress_percent + 10:
                            print(f"[LatentCache] Progress: {current_progress_percent}% ({processed_images}/{total_images} images)")
                            sys.stdout.flush()
                            last_progress_percent = current_progress_percent

                # Print 100% completion
                if last_progress_percent < 100:
                    print(f"[LatentCache] Progress: 100% ({processed_images}/{total_images} images)")
                    sys.stdout.flush()
                print(f"[LatentCache] Cache generation complete:")
                print(f"[LatentCache]   Existing cache used: {total_existing_cached} images")
                print(f"[LatentCache]   Newly cached: {total_new_cached} images")
                print(f"[LatentCache]   Total: {total_existing_cached + total_new_cached} images")

                # Save cache metadata for each dataset
                for unique_id, cache in latent_caches.items():
                    # Count images for this dataset
                    dataset_image_count = sum(
                        1 for batch in batches for item in batch
                        if item.get("dataset_unique_id") == unique_id
                    )
                    cache.save_cache_info(
                        model_path=self.model_path,
                        model_type=model_type,
                        item_count=dataset_image_count
                    )

                # Move VAE back to CPU to free VRAM
                print(f"{self.log_prefix} Moving VAE to CPU (will stay on CPU during training)")
                self.vae.to('cpu')
                torch.cuda.empty_cache()
                if self.debug_vram:
                    print_vram_usage("After moving VAE to CPU")
            else:
                print(f"{self.log_prefix} Using existing latent cache(s)")

                # Count cache statistics per dataset and collect missing images
                total_images = sum(len(batch) for batch in batches)
                total_existing_cached = 0
                total_missing = 0
                cache_stats = {uid: {"cached": 0, "missing": 0} for uid in latent_caches.keys()}
                missing_images = []  # Collect images that need to be cached

                for batch in batches:
                    for item in batch:
                        image_path = item["image_path"]
                        dataset_unique_id = item.get("dataset_unique_id")

                        if not dataset_unique_id or dataset_unique_id not in latent_caches:
                            total_missing += 1
                            continue

                        # Get target dimensions
                        if "bucket_width" in item and "bucket_height" in item:
                            target_width = item["bucket_width"]
                            target_height = item["bucket_height"]
                        else:
                            target_width = item.get("width", 1024)
                            target_height = item.get("height", 1024)

                        # Check if cached
                        cache = latent_caches[dataset_unique_id]
                        cached_latent = cache.load_latent(
                            image_path, target_width, target_height, device='cpu'
                        )

                        if cached_latent is not None:
                            total_existing_cached += 1
                            cache_stats[dataset_unique_id]["cached"] += 1
                        else:
                            total_missing += 1
                            cache_stats[dataset_unique_id]["missing"] += 1
                            missing_images.append(item)

                print(f"[LatentCache] Cache statistics:")
                for unique_id, stats in cache_stats.items():
                    print(f"[LatentCache]   Dataset {unique_id[:8]}...: {stats['cached']} cached, {stats['missing']} missing")
                print(f"[LatentCache]   Total: {total_existing_cached} cached, {total_missing} missing ({total_images} images)")

                # Generate cache for missing images
                if len(missing_images) > 0:
                    print(f"{self.log_prefix} Generating cache for {len(missing_images)} missing image(s)...")

                    # Log device status before VAE move
                    from core.vram_optimization import log_device_status
                    log_device_status(
                        "Before moving VAE to GPU for cache generation",
                        pipeline=None,
                        show_details=False,
                        zimage_components={
                            "text_encoder": self.text_encoder,
                            "transformer": self.transformer if self.is_zimage else self.unet,
                            "vae": self.vae
                        } if self.is_zimage else None
                    )

                    # Move Transformer/UNet to CPU to free VRAM for VAE
                    if self.is_zimage:
                        print(f"{self.log_prefix} Moving Transformer to CPU temporarily for VAE cache generation...")
                        self.transformer.to('cpu')
                    else:
                        if hasattr(self, 'unet') and self.unet is not None:
                            print(f"{self.log_prefix} Moving U-Net to CPU temporarily for VAE cache generation...")
                            self.unet.to('cpu')
                    torch.cuda.empty_cache()

                    # Check if VAE is already on GPU (avoid redundant .to() call)
                    vae_device = next(self.vae.parameters()).device
                    if vae_device.type != 'cuda':
                        print(f"{self.log_prefix} Moving VAE from {vae_device} to GPU for cache generation...")
                        self.vae.to(self.device)
                        torch.cuda.empty_cache()
                    else:
                        print(f"{self.log_prefix} VAE already on GPU, skipping move")

                    log_device_status(
                        "After moving VAE to GPU for cache generation",
                        pipeline=None,
                        show_details=False,
                        zimage_components={
                            "text_encoder": self.text_encoder,
                            "transformer": self.transformer,
                            "vae": self.vae
                        } if self.is_zimage else None
                    )

                    import sys
                    cache_pbar = tqdm(
                        total=len(missing_images),
                        desc="[LatentCache] Encoding missing images",
                        unit="img",
                        ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                        file=sys.stdout,
                        dynamic_ncols=False,
                        mininterval=0.1
                    )
                    sys.stdout.flush()

                    newly_cached = 0
                    for item in missing_images:
                        image_path = item["image_path"]
                        dataset_unique_id = item.get("dataset_unique_id")

                        if not dataset_unique_id or dataset_unique_id not in latent_caches:
                            cache_pbar.update(1)
                            continue

                        # Get target dimensions
                        if "bucket_width" in item and "bucket_height" in item:
                            target_width = item["bucket_width"]
                            target_height = item["bucket_height"]
                        else:
                            target_width = item.get("width", 1024)
                            target_height = item.get("height", 1024)

                        cache = latent_caches[dataset_unique_id]

                        if not os.path.exists(image_path):
                            cache_pbar.write(f"[LatentCache] WARNING: Image not found: {image_path}")
                            cache_pbar.update(1)
                            continue

                        try:
                            image = Image.open(image_path)
                            image.verify()
                            image = Image.open(image_path)

                            # Debug: Log large images
                            if image.size[0] * image.size[1] > 5000 * 5000:
                                cache_pbar.write(f"[LatentCache] WARNING: Large image {image.size[0]}x{image.size[1]} -> bucketed to {target_width}x{target_height}: {os.path.basename(image_path)}")

                            latents = self.encode_image(
                                image, target_width=target_width, target_height=target_height
                            )

                            # Debug: Verify latent size
                            if latents.shape[2] * 8 != target_height or latents.shape[3] * 8 != target_width:
                                cache_pbar.write(f"[LatentCache] ERROR: Latent size mismatch! Expected {target_height//8}x{target_width//8}, got {latents.shape[2]}x{latents.shape[3]}")

                            cache.save_latent(
                                image_path, target_width, target_height, latents
                            )
                            newly_cached += 1
                        except Exception as e:
                            cache_pbar.write(f"[LatentCache] ERROR: Failed to encode {image_path}: {e}")

                        cache_pbar.update(1)
                        sys.stdout.flush()

                    cache_pbar.close()
                    sys.stdout.flush()
                    print(f"[LatentCache] Generated cache for {newly_cached} new image(s)")

                print(f"{self.log_prefix} Moving VAE to CPU (will stay on CPU during training)")
                self.vae.to('cpu')
                torch.cuda.empty_cache()

                # Move Transformer/UNet back to GPU after VAE cache generation
                if self.is_zimage:
                    print(f"{self.log_prefix} Moving Transformer back to GPU...")
                    self.transformer.to(self.device)
                else:
                    if hasattr(self, 'unet') and self.unet is not None:
                        print(f"{self.log_prefix} Moving U-Net back to GPU...")
                        self.unet.to(self.device)
                torch.cuda.empty_cache()

                if self.debug_vram:
                    print_vram_usage("After moving Transformer/UNet back to GPU")

        # Z-Image: Caption pre-encoding (MANDATORY since Text Encoder is frozen)
        if self.is_zimage:
            print(f"\n{'='*80}")
            print(f"{self.log_prefix} Z-Image detected: Pre-encoding captions...")
            print(f"{self.log_prefix} is_zimage flag: {self.is_zimage}")
            print(f"{'='*80}\n")

            # Move Text Encoder to GPU for encoding
            print(f"{self.log_prefix} Moving Text Encoder (Qwen3) to GPU for caption encoding...")
            self.text_encoder.to(self.device)
            if self.debug_vram:
                print_vram_usage("After moving Text Encoder to GPU")
            torch.cuda.empty_cache()

            # Log VRAM status after moving Text Encoder to GPU
            from core.vram_optimization import log_device_status
            log_device_status(
                "After moving Text Encoder to GPU (before caption encoding)",
                pipeline=None,
                show_details=False,
                zimage_components={
                    "text_encoder": self.text_encoder,
                    "transformer": self.transformer,
                    "vae": self.vae
                }
            )

            # Collect unique captions from all datasets
            unique_captions = set()
            for batch in batches:
                for item in batch:
                    caption = item.get("caption", "")
                    if caption:
                        unique_captions.add(caption)

            print(f"{self.log_prefix} Found {len(unique_captions)} unique caption(s)")

            # Pre-encode all captions
            caption_cache = {}  # Map: caption_text -> {"embeddings": Tensor, "mask": Tensor}


            # Try to load from disk cache (Z-Image caption embeddings)
            import hashlib
            from pathlib import Path
            from core.training.latent_cache import get_cache_base_dir
            caption_cache_loaded = 0
            if dataset_unique_ids and len(dataset_unique_ids) > 0:
                # Use first dataset_unique_id for caption cache directory
                cache_base_dir = Path(get_cache_base_dir()) / dataset_unique_ids[0] / "text_embeddings"
                if cache_base_dir.exists():
                    print(f"[CaptionCache] Loading cached caption embeddings from {cache_base_dir}...")
                    for caption in unique_captions:
                        caption_hash = hashlib.md5(caption.encode()).hexdigest()
                        embeds_path = cache_base_dir / f"{caption_hash}_embeds.pt"
                        mask_path = cache_base_dir / f"{caption_hash}_mask.pt"
                        if embeds_path.exists() and mask_path.exists():
                            try:
                                embeds = torch.load(embeds_path, map_location="cpu")
                                mask = torch.load(mask_path, map_location="cpu")
                                caption_cache[caption] = {"embeddings": embeds, "mask": mask}
                                caption_cache_loaded += 1
                            except Exception as e:
                                print(f"[CaptionCache] WARNING: Failed to load cache for caption '{caption[:30]}...': {e}")
                    if caption_cache_loaded > 0:
                        print(f"[CaptionCache] Loaded {caption_cache_loaded}/{len(unique_captions)} cached caption embeddings from disk")

            # Encode captions that are not cached
            captions_to_encode = [c for c in unique_captions if c not in caption_cache]
            if len(captions_to_encode) == 0:
                print(f"[CaptionCache] All {len(unique_captions)} captions already cached, skipping encoding")
            else:
                print(f"[CaptionCache] Encoding {len(captions_to_encode)}/{len(unique_captions)} captions...")
            if len(captions_to_encode) > 0:
                import sys
                caption_pbar = tqdm(
                    total=len(captions_to_encode),
                    desc="[CaptionCache] Encoding captions",
                    unit="caption",
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    file=sys.stdout,
                    dynamic_ncols=False,
                    mininterval=0.1
                )
                sys.stdout.flush()

                for caption in captions_to_encode:
                    try:
                        embeds, mask = self.encode_prompt_zimage(caption)
                        # Store on CPU to save VRAM
                        caption_cache[caption] = {
                            "embeddings": embeds.cpu(),
                            "mask": mask.cpu(),
                        }
                    except Exception as e:
                        caption_pbar.write(f"[CaptionCache] ERROR: Failed to encode caption '{caption[:50]}...': {e}")
                        import traceback
                        caption_pbar.write(traceback.format_exc())
                        # Store empty embeddings as fallback
                        caption_cache[caption] = {
                            "embeddings": torch.zeros((1, 2560), dtype=self.weight_dtype),
                            "mask": torch.zeros(512, dtype=torch.bool),
                        }

                    caption_pbar.update(1)
                    sys.stdout.flush()

                caption_pbar.close()
                sys.stdout.flush()
                print(f"[CaptionCache] Caption encoding complete: {len(caption_cache)} caption(s)")


            # Save newly encoded captions to disk
            if len(captions_to_encode) > 0 and dataset_unique_ids and len(dataset_unique_ids) > 0:
                cache_base_dir = Path(get_cache_base_dir()) / dataset_unique_ids[0] / "text_embeddings"
                cache_base_dir.mkdir(parents=True, exist_ok=True)
                print(f"[CaptionCache] Saving {len(captions_to_encode)} newly encoded caption embeddings to {cache_base_dir}...")
                saved_count = 0
                for caption in captions_to_encode:
                    if caption in caption_cache:
                        caption_hash = hashlib.md5(caption.encode()).hexdigest()
                        embeds_path = cache_base_dir / f"{caption_hash}_embeds.pt"
                        mask_path = cache_base_dir / f"{caption_hash}_mask.pt"
                        try:
                            torch.save(caption_cache[caption]["embeddings"], embeds_path)
                            torch.save(caption_cache[caption]["mask"], mask_path)
                            saved_count += 1
                        except Exception as e:
                            print(f"[CaptionCache] WARNING: Failed to save cache for caption '{caption[:30]}...': {e}")
                print(f"[CaptionCache] Saved {saved_count} caption embeddings to disk")
            # Attach cached embeddings to dataset items
            # IMPORTANT: Create individual copies for each item to avoid reference sharing
            for batch in batches:
                for item in batch:
                    caption = item.get("caption", "")
                    if caption and caption in caption_cache:
                        # Clone tensors to create independent copies (避免共享引用)
                        item["cached_caption_embeds"] = caption_cache[caption]["embeddings"].clone()
                        item["cached_caption_mask"] = caption_cache[caption]["mask"].clone()

            # Save caption_cache for reuse in subsequent epochs
            # (Needed when dataset is reloaded with new captions)
            self.caption_cache = caption_cache
            print(f"[CaptionCache] Saved {len(self.caption_cache)} caption embeddings for reuse in subsequent epochs")
            # Move Text Encoder to CPU to free VRAM (it's frozen, won't be used during training)
            print(f"{self.log_prefix} Moving Text Encoder (Qwen3) to CPU (frozen, no longer needed)")
            self.text_encoder.to('cpu')
            torch.cuda.empty_cache()

            # Log device placement and VRAM usage after caption encoding
            from core.vram_optimization import log_device_status
            log_device_status(
                "After caption encoding (Text Encoder moved to CPU)",
                pipeline=None,
                show_details=False,
                zimage_components={
                    "text_encoder": self.text_encoder,
                    "transformer": self.transformer,
                    "vae": self.vae
                }
            )

            if self.debug_vram:
                print_vram_usage("After moving Text Encoder to CPU")

        if self.optimizer is None:
            self.setup_optimizer(total_steps=total_steps)
            if self.debug_vram:
                print_vram_usage("After optimizer setup")

        # Try to resume from checkpoint
        global_step = 0
        start_epoch = 0

        if resume_from_checkpoint and resume_from_checkpoint.lower() != "latest":
            # User specified a specific checkpoint file to resume from
            checkpoint_path = self.output_dir / resume_from_checkpoint
            if checkpoint_path.exists():
                print(f"{self.log_prefix} Resuming from specified checkpoint: {checkpoint_path}")
                loaded_step = self.load_checkpoint(str(checkpoint_path))
                global_step = loaded_step

                # Calculate which epoch to start from
                start_epoch = global_step // len(batches)
                batches_in_epoch = global_step % len(batches)

                print(f"{self.log_prefix} Resuming from epoch {start_epoch + 1}, batch {batches_in_epoch}")

                # Fast-forward lr_scheduler to match the checkpoint
                for _ in range(global_step):
                    self.lr_scheduler.step()
            else:
                print(f"{self.log_prefix} WARNING: Checkpoint not found: {checkpoint_path}")
                print(f"{self.log_prefix} Starting from scratch")
        elif resume_from_checkpoint and resume_from_checkpoint.lower() == "latest":
            # User explicitly requested "latest" - auto-detect latest checkpoint
            checkpoint_result = self.find_latest_checkpoint()
            if checkpoint_result is not None:
                checkpoint_path, checkpoint_step = checkpoint_result
                print(f"{self.log_prefix} Resuming from latest checkpoint: {checkpoint_path}")
                loaded_step = self.load_checkpoint(checkpoint_path)
                global_step = loaded_step

                # Calculate which epoch to start from
                start_epoch = global_step // len(batches)
                batches_in_epoch = global_step % len(batches)

                print(f"{self.log_prefix} Resuming from epoch {start_epoch + 1}, batch {batches_in_epoch}")

                # Fast-forward lr_scheduler to match the checkpoint
                for _ in range(global_step):
                    self.lr_scheduler.step()
            else:
                print(f"{self.log_prefix} No checkpoint found for auto-resume, starting from scratch")
        else:
            # User selected "Start from Beginning" (resume_from_checkpoint is None or empty)
            print(f"{self.log_prefix} Starting training from beginning")

        # Training loop
        try:
            for epoch in range(start_epoch, num_epochs):
                # Reload dataset for this epoch if callback is provided
                # This allows caption processing (shuffle/dropout) to vary per epoch
                if reload_dataset_callback is not None and epoch > 0:
                    print(f"{self.log_prefix} Reloading dataset for epoch {epoch + 1} (caption processing may change)...")
                    try:
                        dataset_items = reload_dataset_callback(epoch)
                        print(f"{self.log_prefix} Reloaded {len(dataset_items)} items")

                        # Debug: Check first reloaded item
                        if len(dataset_items) > 0:
                            first_item = dataset_items[0]
                            print(f"{self.log_prefix} First reloaded item keys: {list(first_item.keys())}")
                            print(f"{self.log_prefix} First reloaded item dataset_unique_id: {first_item.get('dataset_unique_id', 'MISSING')}")

                        # Rebuild buckets/batches with new captions
                        if enable_bucketing:
                            print(f"{self.log_prefix} Rebuilding buckets for epoch {epoch + 1}...")
                            bucket_manager = BucketManager(base_resolutions=base_resolutions)
                            for idx, item in enumerate(dataset_items):
                                width = item.get("width", 1024)
                                height = item.get("height", 1024)
                                bucket_manager.assign_image_to_bucket(
                                    image_path=item["image_path"],
                                    width=width,
                                    height=height,
                                    caption=item.get("caption", ""),
                                    dataset_unique_id=item.get("dataset_unique_id")
                                )
                            bucket_manager.shuffle_buckets()
                            batches = bucket_manager.build_batch_indices(batch_size)
                            print(f"{self.log_prefix} Rebuilt {len(batches)} batches")

                            # Debug: Check first rebuilt batch
                            if len(batches) > 0 and len(batches[0]) > 0:
                                first_batch_item = batches[0][0]
                                print(f"{self.log_prefix} First batch item keys after rebuild: {list(first_batch_item.keys())}")
                                print(f"{self.log_prefix} First batch item dataset_unique_id after rebuild: {first_batch_item.get('dataset_unique_id', 'MISSING')}")
                        else:
                            batches = []
                            for start_idx in range(0, len(dataset_items), batch_size):
                                end_idx = min(start_idx + batch_size, len(dataset_items))
                                batches.append(dataset_items[start_idx:end_idx])

                        # Rebuild caption cache for Z-Image (handle shuffle/dropout per epoch)
                        if self.is_zimage and hasattr(self, "caption_cache"):
                            print(f"[CaptionCache] Checking for new captions in epoch {epoch + 1}...")

                            # Collect unique captions from reloaded dataset
                            new_unique_captions = set()
                            for batch in batches:
                                for item in batch:
                                    caption = item.get("caption", "")
                                    if caption:
                                        new_unique_captions.add(caption)

                            # Find captions that need encoding (not in cache)
                            captions_to_encode = [c for c in new_unique_captions if c not in self.caption_cache]

                            if len(captions_to_encode) > 0:
                                print(f"[CaptionCache] Found {len(captions_to_encode)} new caption(s), encoding...")

                                # Move Text Encoder to GPU temporarily
                                print(f"[CaptionCache] Moving Text Encoder to GPU for encoding...")
                                self.text_encoder.to(self.device)
                                torch.cuda.empty_cache()

                                # Encode new captions
                                import sys
                                caption_pbar = tqdm(
                                    total=len(captions_to_encode),
                                    desc=f"[CaptionCache] Encoding new captions (Epoch {epoch + 1})",
                                    unit="caption",
                                    ncols=100,
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                                    file=sys.stdout,
                                    dynamic_ncols=False,
                                    mininterval=0.1
                                )
                                sys.stdout.flush()

                                for caption in captions_to_encode:
                                    try:
                                        embeds, mask = self.encode_prompt_zimage(caption)
                                        # Store on CPU to save VRAM
                                        self.caption_cache[caption] = {
                                            "embeddings": embeds.cpu(),
                                            "mask": mask.cpu(),
                                        }
                                    except Exception as e:
                                        caption_pbar.write(f"[CaptionCache] ERROR: Failed to encode caption '{caption[:50]}...': {e}")
                                        import traceback
                                        caption_pbar.write(traceback.format_exc())
                                        # Store empty embeddings as fallback
                                        self.caption_cache[caption] = {
                                            "embeddings": torch.zeros((1, 2560), dtype=self.weight_dtype),
                                            "mask": torch.zeros(512, dtype=torch.bool),
                                        }

                                    caption_pbar.update(1)
                                    sys.stdout.flush()

                                caption_pbar.close()
                                sys.stdout.flush()

                                # Save newly encoded captions to disk
                                if dataset_unique_ids and len(dataset_unique_ids) > 0:
                                    import hashlib
                                    from pathlib import Path
                                    from core.training.latent_cache import get_cache_base_dir
                                    cache_base_dir = Path(get_cache_base_dir()) / dataset_unique_ids[0] / "text_embeddings"
                                    cache_base_dir.mkdir(parents=True, exist_ok=True)
                                    print(f"[CaptionCache] Saving {len(captions_to_encode)} newly encoded captions to {cache_base_dir}...")
                                    saved_count = 0
                                    for caption in captions_to_encode:
                                        if caption in self.caption_cache:
                                            caption_hash = hashlib.md5(caption.encode()).hexdigest()
                                            embeds_path = cache_base_dir / f"{caption_hash}_embeds.pt"
                                            mask_path = cache_base_dir / f"{caption_hash}_mask.pt"
                                            try:
                                                torch.save(self.caption_cache[caption]["embeddings"], embeds_path)
                                                torch.save(self.caption_cache[caption]["mask"], mask_path)
                                                saved_count += 1
                                            except Exception as e:
                                                print(f"[CaptionCache] WARNING: Failed to save cache for caption '{caption[:30]}...': {e}")
                                    print(f"[CaptionCache] Saved {saved_count} caption embeddings to disk")

                                # Move Text Encoder back to CPU
                                print(f"[CaptionCache] Moving Text Encoder back to CPU...")
                                self.text_encoder.to('cpu')
                                torch.cuda.empty_cache()

                                print(f"[CaptionCache] Caption cache updated: {len(self.caption_cache)} total captions")
                            else:
                                print(f"[CaptionCache] All {len(new_unique_captions)} captions already cached, no encoding needed")

                            # Attach cached embeddings to reloaded items
                            print(f"[CaptionCache] Attaching cached embeddings to {len(new_unique_captions)} unique caption(s)...")
                            reattached_count = 0
                            for batch in batches:
                                for item in batch:
                                    caption = item.get("caption", "")
                                    if caption and caption in self.caption_cache:
                                        item["cached_caption_embeds"] = self.caption_cache[caption]["embeddings"].clone()
                                        item["cached_caption_mask"] = self.caption_cache[caption]["mask"].clone()
                                        reattached_count += 1
                            print(f"[CaptionCache] Attached embeddings to {reattached_count} items")
                    except Exception as reload_err:
                        print(f"{self.log_prefix} ERROR: Failed to reload dataset for epoch {epoch + 1}: {reload_err}")
                        import traceback
                        traceback.print_exc()
                        raise

                # Calculate starting batch index for this epoch (for resume)
                start_batch_idx = 0
                if epoch == start_epoch:
                    start_batch_idx = global_step % len(batches)

                # Print epoch header
                print(f"{self.log_prefix} === Epoch {epoch + 1}/{num_epochs} ===")
                total_batches = len(batches)
                print(f"{self.log_prefix} Total batches: {total_batches}")

                # Debug: Check if batches are valid
                if total_batches == 0:
                    print(f"{self.log_prefix} ERROR: No batches generated for epoch {epoch + 1}")
                    raise RuntimeError(f"No batches generated for epoch {epoch + 1}")

                # Calculate progress log interval (10% of total batches, minimum 1)
                progress_interval = max(1, total_batches // 10)

                print(f"{self.log_prefix} Starting batch loop from batch {start_batch_idx}...")

                for batch_idx, batch in enumerate(batches[start_batch_idx:], start=start_batch_idx):
                    # Check for stop flag (file-based, works on Windows)
                    stop_flag_file = self.output_dir / ".stop_training"
                    if stop_flag_file.exists():
                        print(f"\n{self.log_prefix} Stop flag detected at step {global_step}, initiating graceful shutdown...")
                        stop_flag_file.unlink()  # Remove flag file
                        raise KeyboardInterrupt("User requested stop via stop flag")

                    try:
                        # Debug: Log batch entry (first batch only to avoid spam)
                        if batch_idx == start_batch_idx:
                            # Log VRAM status before starting training loop
                            from core.vram_optimization import log_device_status
                            log_device_status(
                                "Before training loop (ready to train)",
                                pipeline=None,
                                show_details=False,
                                zimage_components={
                                    "text_encoder": self.text_encoder,
                                    "transformer": self.transformer,
                                    "vae": self.vae
                                } if self.is_zimage else None
                            )

                            print(f"{self.log_prefix} Processing first batch {batch_idx}, batch size: {len(batch)}")
                            if len(batch) > 0:
                                first_item = batch[0]
                                print(f"{self.log_prefix} First batch item keys: {list(first_item.keys())}")
                                print(f"{self.log_prefix} First batch item dataset_unique_id: {first_item.get('dataset_unique_id', 'MISSING')}")

                        # VRAM profiling for first batch only (to avoid spam)
                        profile_vram = self.debug_vram and (global_step == 0)

                        if profile_vram:
                            print_vram_usage("Start of first batch")

                        # Process entire batch
                        batch_latents = []
                        batch_text_embeddings = []
                        batch_pooled_embeddings = [] if self.is_sdxl else None
                        batch_caption_embeds = [] if self.is_zimage else None  # Z-Image: pre-encoded caption embeddings
                        batch_caption_masks = [] if self.is_zimage else None  # Z-Image: attention masks
                        batch_captions = []  # Store captions for debug output

                        # Get bucket dimensions (all items in batch have same resolution)
                        first_item = batch[0]
                        if enable_bucketing and "bucket_width" in first_item and "bucket_height" in first_item:
                            target_width = first_item["bucket_width"]
                            target_height = first_item["bucket_height"]
                        else:
                            target_width = None
                            target_height = None

                        # Load and encode all images in batch
                        for item_idx, item in enumerate(batch):
                            try:
                                image_path = item["image_path"]
                                if not os.path.exists(image_path):
                                    print(f"{self.log_prefix} WARNING: Image not found: {image_path}")
                                    continue

                                # Try to load from cache first
                                latents = None
                                if len(latent_caches) > 0:
                                    dataset_unique_id = item.get("dataset_unique_id")
                                    if dataset_unique_id and dataset_unique_id in latent_caches:
                                        cache = latent_caches[dataset_unique_id]
                                        if target_width is not None and target_height is not None:
                                            try:
                                                latents = cache.load_latent(
                                                    image_path, target_width, target_height, device=self.device
                                                )
                                            except Exception as cache_err:
                                                print(f"{self.log_prefix} WARNING: Cache load failed for {image_path}: {cache_err}")
                                                latents = None
                                    else:
                                        # Debug: Log missing dataset_unique_id (first batch only)
                                        if batch_idx == start_batch_idx and item_idx == 0:
                                            if not dataset_unique_id:
                                                print(f"{self.log_prefix} WARNING: No dataset_unique_id in item")
                                            elif dataset_unique_id not in latent_caches:
                                                print(f"{self.log_prefix} WARNING: dataset_unique_id {dataset_unique_id[:8]}... not in latent_caches")
                                                print(f"{self.log_prefix} Available cache IDs: {[uid[:8] + '...' for uid in latent_caches.keys()]}")

                                # If not cached, encode normally
                                if latents is None:
                                    try:
                                        image = Image.open(image_path)
                                        image.verify()  # Verify image integrity
                                        image = Image.open(image_path)  # Reopen after verify

                                        # Encode image
                                        if target_width is not None and target_height is not None:
                                            latents = self.encode_image(image, target_width=target_width, target_height=target_height)
                                        else:
                                            latents = self.encode_image(image)
                                    except Exception as img_err:
                                        print(f"{self.log_prefix} ERROR: Corrupted or invalid image {image_path}: {img_err}")
                                        continue

                                batch_latents.append(latents)

                                # Encode caption
                                caption = item.get("caption", "")
                                batch_captions.append(caption)  # Store for debug output

                                if self.is_zimage:
                                    # Z-Image: Use pre-encoded caption embeddings (Text Encoder is frozen)
                                    if "cached_caption_embeds" in item and "cached_caption_mask" in item:
                                        caption_embeds = item["cached_caption_embeds"].to(self.device)
                                        caption_mask = item["cached_caption_mask"].to(self.device)
                                    else:
                                        # Fallback: encode on-the-fly (should not happen if pre-encoding worked)
                                        print(f"{self.log_prefix} WARNING: No cached caption embeddings for item {item_idx}, encoding on-the-fly")
                                        print(f"{self.log_prefix} Item keys: {list(item.keys())}")
                                        print(f"{self.log_prefix} Caption: {caption[:50]}...")

                                        # Move Text Encoder to GPU temporarily
                                        self.text_encoder.to(self.device)
                                        caption_embeds, caption_mask = self.encode_prompt_zimage(caption)
                                        caption_embeds = caption_embeds.to(self.device)
                                        caption_mask = caption_mask.to(self.device)
                                        # Move back to CPU
                                        self.text_encoder.to('cpu')

                                    batch_caption_embeds.append(caption_embeds)
                                    batch_caption_masks.append(caption_mask)
                                else:
                                    # SD/SDXL: Encode caption with gradient for text encoder training
                                    prompt_output = self.encode_prompt(caption, requires_grad=True)

                                    if self.is_sdxl:
                                        text_emb, pooled_emb = prompt_output
                                        batch_text_embeddings.append(text_emb)
                                        batch_pooled_embeddings.append(pooled_emb)
                                    else:
                                        batch_text_embeddings.append(prompt_output)
                            except Exception as item_err:
                                print(f"{self.log_prefix} ERROR: Failed to process item {item_idx} in batch {batch_idx}: {item_err}")
                                import traceback
                                traceback.print_exc()
                                continue

                        # Skip if batch is empty (all items failed)
                        if len(batch_latents) == 0:
                            continue

                        # Stack into batched tensors
                        if profile_vram:
                            print_vram_usage("After loading batch data (before concat)")

                        batched_latents = torch.cat(batch_latents, dim=0)
                        del batch_latents  # Free memory immediately after concat

                        if self.is_zimage:
                            # Z-Image: Pad caption embeddings to max length in batch for batching
                            # Find max sequence length in batch
                            max_seq_len = max(emb.shape[0] for emb in batch_caption_embeds)

                            # Pad embeddings and masks
                            padded_embeds = []
                            padded_masks = []
                            for emb, mask in zip(batch_caption_embeds, batch_caption_masks):
                                seq_len = emb.shape[0]
                                if seq_len < max_seq_len:
                                    # Pad embeddings with zeros
                                    pad_size = max_seq_len - seq_len
                                    padded_emb = torch.cat([
                                        emb,
                                        torch.zeros((pad_size, emb.shape[1]), dtype=emb.dtype, device=emb.device)
                                    ], dim=0)
                                    # Pad mask with False
                                    padded_mask = torch.cat([
                                        mask,
                                        torch.zeros(pad_size, dtype=torch.bool, device=mask.device)
                                    ], dim=0)
                                else:
                                    padded_emb = emb
                                    padded_mask = mask

                                padded_embeds.append(padded_emb.unsqueeze(0))  # Add batch dimension
                                padded_masks.append(padded_mask.unsqueeze(0))  # Add batch dimension

                            batched_caption_embeds = torch.cat(padded_embeds, dim=0)  # [B, max_seq_len, 2560]
                            batched_caption_masks = torch.cat(padded_masks, dim=0)  # [B, max_seq_len]
                            # Free intermediate tensors
                            del batch_caption_embeds, batch_caption_masks, padded_embeds, padded_masks
                        else:
                            batched_text_embeddings = torch.cat(batch_text_embeddings, dim=0)
                            del batch_text_embeddings  # Free memory immediately after concat

                            if self.is_sdxl:
                                batched_pooled_embeddings = torch.cat(batch_pooled_embeddings, dim=0)
                                del batch_pooled_embeddings  # Free memory immediately after concat

                        if profile_vram:
                            print_vram_usage("After concat (before train_step)")

                        # Determine if we should save debug latents for this step
                        debug_save_path = None
                        debug_captions = None
                        if debug_dir is not None and global_step % debug_latents_every == 0:
                            debug_save_path = debug_dir / f"step_{global_step:06d}"
                            debug_captions = batch_captions  # Pass captions for debug output

                        # Call appropriate train_step based on model type
                        if self.is_zimage:
                            loss, recon_loss = self.train_step_zimage(
                                batched_latents,
                                batched_caption_embeds,
                                batched_caption_masks,
                                debug_save_path=debug_save_path,
                                debug_captions=debug_captions,
                                profile_vram=profile_vram
                            )
                        elif self.is_sdxl:
                            loss, recon_loss = self.train_step(
                                batched_latents,
                                batched_text_embeddings,
                                batched_pooled_embeddings,
                                debug_save_path=debug_save_path,
                                debug_captions=debug_captions,
                                profile_vram=profile_vram
                            )
                        else:
                            loss, recon_loss = self.train_step(
                                batched_latents,
                                batched_text_embeddings,
                                debug_save_path=debug_save_path,
                                debug_captions=debug_captions,
                                profile_vram=profile_vram
                            )

                        # Free batch tensors after training step to reduce VRAM usage
                        del batched_latents
                        if self.is_zimage:
                            del batched_caption_embeds, batched_caption_masks
                        else:
                            del batched_text_embeddings
                            if self.is_sdxl:
                                del batched_pooled_embeddings

                        # Clear CUDA cache to free GPU memory
                        torch.cuda.empty_cache()

                        if profile_vram:
                            print_vram_usage("After train_step and cleanup")

                        global_step += 1

                        # Log to tensorboard
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        self.writer.add_scalar('train/loss', loss, global_step)
                        self.writer.add_scalar('train/recon_loss', recon_loss, global_step)
                        self.writer.add_scalar('train/learning_rate', current_lr, global_step)
                        self.writer.add_scalar('train/epoch', epoch + 1, global_step)

                        # Progress callback
                        if progress_callback:
                            progress_callback(global_step, loss, current_lr)

                        # Print progress every 10% of epoch
                        current_batch = batch_idx - start_batch_idx + 1
                        if current_batch % progress_interval == 0 or current_batch == total_batches:
                            progress_pct = (current_batch / total_batches) * 100
                            print(f"{self.log_prefix} Epoch {epoch + 1}/{num_epochs} - Batch {current_batch}/{total_batches} ({progress_pct:.0f}%) - Loss: {loss:.4f}, LR: {current_lr:.2e}, Step: {global_step}")

                        # Save checkpoint (step-based)
                        if save_every_unit == "steps" and global_step % save_every == 0:
                            print(f"{self.log_prefix} Checkpoint saved at step {global_step}")
                            try:
                                self.save_checkpoint(global_step, save_optimizer=False, max_to_keep=max_step_saves_to_keep, save_every=save_every, run_id=run_id, epoch=epoch + 1)
                            except Exception as save_error:
                                print(f"{self.log_prefix} WARNING: Checkpoint save failed: {save_error}")
                                print(f"{self.log_prefix} Training will continue.")

                        # Sample generation
                        # Generate samples at step 0 (initial) or every sample_every steps
                        should_generate_sample = (global_step == 0 or global_step % sample_every == 0)
                        if should_generate_sample:
                            if sample_prompts and sample_config:
                                print(f"{self.log_prefix} Generating samples at step {global_step}")
                                vae_on_cpu = len(latent_caches) > 0
                                self.generate_sample(global_step, sample_prompts, sample_config, vae_on_cpu=vae_on_cpu)
                            else:
                                print(f"{self.log_prefix} Skipping sample generation at step {global_step}: sample_prompts={sample_prompts is not None and len(sample_prompts) > 0}, sample_config={sample_config is not None}")

                    except Exception as e:
                        print(f"{self.log_prefix} ERROR processing batch: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                # Save checkpoint (epoch-based)
                if save_every_unit == "epochs" and (epoch + 1) % save_every == 0:
                    print(f"{self.log_prefix} Checkpoint saved at epoch {epoch + 1}")
                    try:
                        self.save_checkpoint(global_step, save_optimizer=False, max_to_keep=max_step_saves_to_keep, save_every=save_every, run_id=run_id, epoch=epoch + 1)
                    except Exception as save_error:
                        print(f"{self.log_prefix} WARNING: Checkpoint save failed: {save_error}")
                        print(f"{self.log_prefix} Training will continue.")

            print(f"\n{self.log_prefix} Training completed! Total steps: {global_step}")

            # Save final checkpoint (with optimizer state for potential resume)
            # Use run_name format: extract short name if auto-generated
            import re
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)  # Extract ID part
            else:
                short_name = self.run_name  # Use full name

            final_path = self.output_dir / f"{short_name}_final.safetensors"
            try:
                self.save_checkpoint(global_step, final_path, save_optimizer=True, run_id=run_id, epoch=num_epochs)
            except Exception as save_error:
                print(f"{self.log_prefix} WARNING: Final checkpoint save failed: {save_error}")
                print(f"{self.log_prefix} Training is complete but final checkpoint could not be saved.")

        except KeyboardInterrupt:
            print(f"\n{self.log_prefix} Training interrupted by user at step {global_step}")
            print(f"{self.log_prefix} Saving checkpoint at interruption point...")

            # Use run_name format
            import re
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)
            else:
                short_name = self.run_name

            interrupted_path = self.output_dir / f"{short_name}_step_{global_step}_interrupted.safetensors"
            try:
                self.save_checkpoint(global_step, interrupted_path, save_optimizer=True)
                print(f"{self.log_prefix} Checkpoint saved. You can resume training from this point.")
            except Exception as save_error:
                print(f"{self.log_prefix} WARNING: Failed to save checkpoint at interruption point: {save_error}")
            raise  # Re-raise to propagate the interruption

        except Exception as e:
            print(f"\n{self.log_prefix} Training failed with error at step {global_step}: {e}")
            print(f"{self.log_prefix} Saving checkpoint at failure point...")

            # Use run_name format
            import re
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)
            else:
                short_name = self.run_name

            failed_path = self.output_dir / f"{short_name}_step_{global_step}_failed.safetensors"
            try:
                self.save_checkpoint(global_step, failed_path, save_optimizer=True)
                print(f"{self.log_prefix} Checkpoint saved. You can resume training from this point.")
            except Exception as save_error:
                print(f"{self.log_prefix} WARNING: Failed to save checkpoint at failure point: {save_error}")
                print(f"{self.log_prefix} Original error: {e}")
            raise  # Re-raise the original exception

        finally:
            # Close tensorboard writer
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.close()
