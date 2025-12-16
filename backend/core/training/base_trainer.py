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
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Component-specific learning rates
        self.unet_lr = unet_lr if unet_lr is not None else learning_rate
        self.text_encoder_lr = text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_1_lr = text_encoder_1_lr if text_encoder_1_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_2_lr = text_encoder_2_lr if text_encoder_2_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate

        # Convert dtype strings to torch.dtype
        self.weight_dtype = get_torch_dtype(weight_dtype)
        self.training_dtype = get_torch_dtype(training_dtype)
        self.output_dtype = get_torch_dtype(output_dtype)
        self.vae_dtype = get_torch_dtype(vae_dtype)
        self.mixed_precision = mixed_precision
        self.debug_vram = debug_vram
        self.use_flash_attention = use_flash_attention
        self.min_snr_gamma = min_snr_gamma

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

        # Move Transformer to GPU
        print(f"{self.log_prefix} Moving Transformer to {self.device}...")
        self.transformer_original.to(self.device)
        self.transformer.to(self.device)

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

        print(f"{self.log_prefix} {'SDXL' if self.is_sdxl else 'SD1.5'} model loaded successfully")

    def _setup_flash_attention_zimage(self):
        """Setup Flash Attention for Z-Image models."""
        import sys
        if 'zimage.transformer' in sys.modules:
            zimage_transformer_module = sys.modules['zimage.transformer']
            ZImageAttention = zimage_transformer_module.ZImageAttention
            print(f"{self.log_prefix} Setting Flash Attention backend for Z-Image...")
            ZImageAttention._attention_backend = "flash"
            print(f"{self.log_prefix} [OK] Flash Attention enabled: {ZImageAttention._attention_backend}")
        else:
            from core.models.zimage_transformer import ZImageAttention
            print(f"{self.log_prefix} Setting Flash Attention backend for Z-Image...")
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
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint state (step, epoch, etc.)
        """
        pass

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint in output directory.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        # Default implementation (can be overridden)
        checkpoint_files = list(self.output_dir.glob("checkpoint-*.safetensors"))
        if not checkpoint_files:
            return None

        # Sort by step number
        def get_step(path):
            try:
                return int(path.stem.split("-")[-1])
            except:
                return 0

        latest = max(checkpoint_files, key=get_step)
        return str(latest)

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
        from core.training.optimizers import OptimizerFactory
        try:
            self.optimizer = OptimizerFactory.create_optimizer(
                optimizer_type=optimizer_type,
                params=param_groups,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8,
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

    # ============================================================
    # Prompt Encoding
    # ============================================================

    def encode_prompt(self, prompt: str, requires_grad: bool = False):
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt to encode
            requires_grad: Whether to enable gradient computation for text encoders

        Returns:
            For SD1.5: text_embeddings tensor
            For SDXL: tuple of (text_embeddings, pooled_embeddings)
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
                text_embeddings_1 = self.text_encoder(
                    text_inputs_1.input_ids.to(self.device),
                    output_hidden_states=False,
                )[0]

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

        return prompt_embeds[0], attention_mask[0]

    # ============================================================
    # Image Encoding
    # ============================================================

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
            width, height = target_width, target_height
        else:
            width, height = target_size, target_size

        # Resize with aspect ratio preservation + center crop
        img_width, img_height = image.size

        if img_width * img_height > 5000 * 5000:
            print(f"[encode_image] Resizing large image {img_width}x{img_height} -> {width}x{height}")

        scale = max(width / img_width, height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Center crop
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        image = image.crop((left, top, left + width, top + height))

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
                latents = self.vae.encode(image_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

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
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        profile_vram: bool = False,
    ) -> float:
        """
        Perform single training step (SD/SDXL).

        Args:
            latents: Image latents [B, C, H, W]
            text_embeddings: Text prompt embeddings
            pooled_embeddings: Pooled text embeddings (SDXL only)
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output
            profile_vram: If True, print VRAM usage

        Returns:
            Loss value
        """
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

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

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

        # Get target based on prediction type
        prediction_type = self.noise_scheduler.config.prediction_type
        target = get_target_from_prediction_type(
            self.noise_scheduler,
            prediction_type,
            latents,
            noise,
            timesteps,
        )

        # Calculate loss (always in fp32)
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Apply Min-SNR gamma weighting
        if self.min_snr_gamma > 0:
            loss_per_sample_weighted = apply_snr_weight(loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma)
        else:
            loss_per_sample_weighted = loss_per_sample

        loss = loss_per_sample_weighted.mean()

        # Calculate reconstruction loss for monitoring
        with torch.no_grad():
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
            alpha_bar_t = alphas_cumprod[timesteps]
            while alpha_bar_t.dim() < latents.dim():
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

            if prediction_type == "epsilon":
                predicted_latent_for_recon = (noisy_latents - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar
            elif prediction_type == "v_prediction":
                predicted_latent_for_recon = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * model_pred
            elif prediction_type == "sample":
                predicted_latent_for_recon = model_pred
            else:
                predicted_latent_for_recon = noisy_latents - model_pred

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
                if prediction_type == "epsilon":
                    predicted_latent = (noisy_latents - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar
                elif prediction_type == "v_prediction":
                    predicted_latent = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * model_pred
                elif prediction_type == "sample":
                    predicted_latent = model_pred
                else:
                    predicted_latent = noisy_latents - model_pred

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

        # Return both prediction loss and reconstruction loss
        loss_value = loss.item()
        recon_loss_value = recon_loss.item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        del noise, noisy_latents, model_pred, loss, recon_loss
        if self.is_sdxl and added_cond_kwargs is not None:
            del added_cond_kwargs

        return loss_value, recon_loss_value

    def train_step_zimage(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        profile_vram: bool = False,
    ) -> float:
        """
        Perform single training step (Z-Image).

        Args:
            latents: Image latents [B, C, H, W]
            prompt_embeds: Prompt embeddings [B, seq_len, 2560]
            attention_mask: Attention mask [B, seq_len]
            debug_save_path: If provided, save latents for debugging
            debug_captions: Captions for debug output
            profile_vram: If True, print VRAM usage

        Returns:
            Loss value
        """
        if profile_vram:
            print_vram_usage("[train_step_zimage] Start")

        # Flow Matching: Sample random timesteps from [0, 1]
        batch_size = latents.shape[0]
        timesteps = torch.rand(batch_size, device=self.device)

        # Flow Matching: Sample noise (standard normal distribution)
        noise = torch.randn_like(latents)

        # Flow Matching: Interpolate between noise and data
        # x_t = (1 - t) * noise + t * data
        # Reshape timesteps for broadcasting: [B] -> [B, 1, 1, 1]
        t = timesteps[:, None, None, None]
        noisy_latents = (1.0 - t) * noise + t * latents

        if profile_vram:
            print_vram_usage("[train_step_zimage] Before Transformer forward")

        # Enable gradients for gradient checkpointing
        noisy_latents.requires_grad_(True)
        prompt_embeds.requires_grad_(True)
        # Note: attention_mask is bool type, does not need gradients

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

        # Flow Matching target: velocity = data - noise
        target = latents - noise

        # Calculate loss (always in fp32)
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_element.mean([1, 2, 3])

        # Flow Matching doesn't use Min-SNR weighting (uniform timestep distribution)
        loss = loss_per_sample.mean()

        # Calculate reconstruction loss
        # For Flow Matching, reconstruct using: x_0 = x_t + (1-t) * v_pred
        with torch.no_grad():
            predicted_latent_for_recon = noisy_latents + (1.0 - t) * model_pred
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
                predicted_latent = noisy_latents + (1.0 - t) * model_pred

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

        # Return both prediction loss and reconstruction loss
        loss_value = loss.item()
        recon_loss_value = recon_loss.item()

        # Free intermediate tensors explicitly to reduce VRAM usage
        del noise, noisy_latents, noisy_latents_4d, model_pred, target, loss, recon_loss

        return loss_value, recon_loss_value

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
    ) -> Image.Image:
        """
        Generate sample image during training (SD/SDXL).

        Args:
            prompt: Text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale

        Returns:
            PIL Image
        """
        print(f"{self.log_prefix} Generating sample: {prompt[:50]}...")

        # Encode prompt
        if self.is_sdxl:
            text_embeddings, pooled_embeddings = self.encode_prompt(prompt, requires_grad=False)
            # Generate unconditional embeddings for CFG
            uncond_embeddings, uncond_pooled = self.encode_prompt("", requires_grad=False)
        else:
            text_embeddings = self.encode_prompt(prompt, requires_grad=False)
            uncond_embeddings = self.encode_prompt("", requires_grad=False)
            pooled_embeddings = None
            uncond_pooled = None

        # Prepare latents
        latent_height = height // 8
        latent_width = width // 8
        latents = torch.randn(
            (1, self.unet.config.in_channels, latent_height, latent_width),
            device=self.device,
            dtype=self.training_dtype,
        )

        # Setup scheduler for inference
        from diffusers import EulerDiscreteScheduler
        inference_scheduler = EulerDiscreteScheduler.from_config(self.noise_scheduler.config)
        inference_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        with torch.no_grad():
            for t in tqdm(inference_scheduler.timesteps, desc="Generating"):
                # Prepare latent input
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

                # Prepare text embeddings
                if guidance_scale > 1.0:
                    text_input = torch.cat([uncond_embeddings, text_embeddings])
                else:
                    text_input = text_embeddings

                # Prepare added_cond_kwargs for SDXL
                added_cond_kwargs = None
                if self.is_sdxl:
                    if guidance_scale > 1.0:
                        pooled_input = torch.cat([uncond_pooled, pooled_embeddings])
                    else:
                        pooled_input = pooled_embeddings

                    time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self.device, dtype=self.training_dtype)
                    if guidance_scale > 1.0:
                        time_ids = time_ids.repeat(2, 1)

                    added_cond_kwargs = {
                        "text_embeds": pooled_input,
                        "time_ids": time_ids
                    }

                # Predict noise
                timestep = t.to(self.device)
                if self.is_sdxl and added_cond_kwargs is not None:
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        text_input,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        text_input
                    ).sample

                # CFG
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Denoise step
                latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.vae.dtype)).sample

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).astype(np.uint8)[0]

        return Image.fromarray(image)

    def _generate_sample_zimage(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
    ) -> Image.Image:
        """
        Generate sample image during training (Z-Image).

        Args:
            prompt: Text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale

        Returns:
            PIL Image
        """
        print(f"{self.log_prefix} Generating Z-Image sample: {prompt[:50]}...")

        # Move text encoder and VAE to GPU for inference
        text_encoder_device = next(self.text_encoder.parameters()).device
        vae_device = next(self.vae.parameters()).device

        if text_encoder_device != self.device:
            self.text_encoder.to(self.device)
        if vae_device != self.device:
            self.vae.to(self.device)

        try:
            # Encode prompt
            prompt_embeds, attention_mask = self.encode_prompt_zimage(prompt)

            # Encode unconditional prompt only if CFG is enabled
            if guidance_scale > 1.0:
                uncond_embeds, uncond_mask = self.encode_prompt_zimage("")
            else:
                uncond_embeds, uncond_mask = None, None

            # Add batch dimension
            prompt_embeds = prompt_embeds.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if uncond_embeds is not None:
                uncond_embeds = uncond_embeds.unsqueeze(0)
                uncond_mask = uncond_mask.unsqueeze(0)

            # Prepare latents
            latent_height = height // 8
            latent_width = width // 8
            latents = torch.randn(
                (1, self.vae.config.latent_channels, latent_height, latent_width),
                device=self.device,
                dtype=self.training_dtype,
            )

            # Setup scheduler (create new instance with same config)
            # Note: We cannot use from_config() because Z-Image scheduler.config is not a standard ConfigMixin
            inference_scheduler = type(self.scheduler)(
                num_train_timesteps=self.scheduler.config.get("num_train_timesteps", 1000),
                shift=self.scheduler.config.get("shift", 1.0),
                use_dynamic_shifting=self.scheduler.config.get("use_dynamic_shifting", False),
            )
            inference_scheduler.set_timesteps(num_inference_steps)

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

            # Decode latents
            image = self._decode_zimage_latents(latents)

            return image

        finally:
            # Move text encoder and VAE back to original devices
            if text_encoder_device != self.device:
                self.text_encoder.to(text_encoder_device)
            if vae_device != self.device:
                self.vae.to(vae_device)

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
            for t in tqdm(scheduler.timesteps, desc="Generating"):
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

                # Stack back to tensor and remove channel dimension
                model_pred = torch.stack(model_out_list, dim=0)  # [B, C, 1, H, W]
                noise_pred = model_pred.squeeze(2)  # [B, C, H, W]

                # CFG
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Denoise step
                latents = scheduler.step(noise_pred, t, latents).prev_sample

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
    ):
        """
        Check latent caches and generate missing ones.

        Args:
            datasets: List of dataset objects
            latent_caches: Dictionary of latent caches
            progress_callback: Progress callback function
        """
        print(f"{self.log_prefix} Checking and generating missing latent caches...")

        # Generate missing latents (this will skip already cached items)
        self._generate_missing_latents_with_model_offload(
            datasets=datasets,
            latent_caches=latent_caches,
            progress_callback=progress_callback,
        )

    def _generate_missing_latents_with_model_offload(
        self,
        datasets: List[Any],
        latent_caches: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
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
            self.vae.to(self.device)
        else:
            print(f"{self.log_prefix} VAE already on {self.device}, skipping move")

        iteration_count = 0
        for dataset in datasets:
            cache = latent_caches[dataset.unique_id]

            for item in tqdm(dataset.items, desc=f"Caching {dataset.unique_id}"):
                # Check if already cached
                image_path = item["image_path"]
                width = item["width"]
                height = item["height"]

                if cache.has_latent(image_path, width, height):
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

    def _setup_text_encoder_cache(
        self,
        datasets: List[Any],
        progress_callback: Optional[Callable] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Setup text encoder cache for Z-Image (caption pre-encoding).

        Args:
            datasets: List of dataset objects
            progress_callback: Progress callback function

        Returns:
            Dictionary mapping caption to (prompt_embeds, attention_mask)
        """
        if not self.is_zimage:
            return None

        print(f"{self.log_prefix} Setting up text encoder cache (Z-Image)...")

        # Collect unique captions
        unique_captions = set()
        for dataset in datasets:
            for item in dataset.items:
                caption = item.get("caption", "")
                if caption:
                    unique_captions.add(caption)

        print(f"{self.log_prefix} Found {len(unique_captions)} unique captions")

        # Encode all captions
        text_encoder_cache = {}
        self.text_encoder.to(self.device)

        unique_captions_list = list(unique_captions)
        total_captions = len(unique_captions_list)

        for idx, caption in enumerate(tqdm(unique_captions_list, desc="Encoding captions")):
            prompt_embeds, attention_mask = self.encode_prompt_zimage(caption)
            text_encoder_cache[caption] = (prompt_embeds.cpu(), attention_mask.cpu())

            # Progress callback
            if progress_callback:
                progress_callback(
                    phase="text_encoder_cache",
                    step=idx + 1,
                    total=total_captions,
                )

        # Move text encoder back to CPU
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        print(f"{self.log_prefix} Text encoder cache setup complete")
        return text_encoder_cache

    # ============================================================
    # Training Loop Infrastructure
    # ============================================================

    def train(
        self,
        datasets: List[Any],
        num_epochs: int = 10,
        batch_size: int = 1,
        save_every_n_steps: int = 500,
        sample_every_n_steps: int = 500,
        sample_prompt: str = "a beautiful landscape",
        sample_guidance_scale: float = 3.5,
        sample_steps: int = 28,
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "constant",
        enable_bucketing: bool = True,
        base_resolutions: Optional[List[int]] = None,
        bucket_strategy: str = "resize",
        multi_resolution_mode: str = "max",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
        progress_callback: Optional[Callable] = None,
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
        """
        print(f"{self.log_prefix} Starting training...")
        print(f"{self.log_prefix} Datasets: {len(datasets)}")
        print(f"{self.log_prefix} Epochs: {num_epochs}")
        print(f"{self.log_prefix} Batch size: {batch_size}")
        print(f"{self.log_prefix} Gradient accumulation: {gradient_accumulation_steps}")
        print(f"{self.log_prefix} Debug latents: {debug_latents} (every {debug_latents_every} steps)")

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

        # Calculate total steps
        total_items = sum(len(dataset.items) for dataset in datasets)
        steps_per_epoch = (total_items + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * num_epochs

        print(f"{self.log_prefix} Total items: {total_items}")
        print(f"{self.log_prefix} Steps per epoch: {steps_per_epoch}")
        print(f"{self.log_prefix} Total steps: {total_steps}")

        # Setup optimizer
        self.setup_optimizer(
            optimizer_type=optimizer_type,
            lr_scheduler_type=lr_scheduler_type,
            total_steps=total_steps,
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

        # Setup latent caches
        latent_caches = self._setup_latent_caches(datasets)
        self._validate_and_generate_latent_caches(datasets, latent_caches, progress_callback)

        # Setup text encoder cache (Z-Image only)
        text_encoder_cache = self._setup_text_encoder_cache(datasets, progress_callback) if self.is_zimage else None

        # Training loop
        global_step = 0

        for epoch in range(num_epochs):
            print(f"\n{self.log_prefix} Epoch {epoch + 1}/{num_epochs}")

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
                all_items = []
                for dataset in datasets:
                    all_items.extend([(item, dataset) for item in dataset.items])
                batches = [all_items[i:i+batch_size] for i in range(0, len(all_items), batch_size)]

            # Training loop
            for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}")):
                # Prepare batch data
                latents_list = []
                text_embeddings_list = []
                pooled_embeddings_list = [] if self.is_sdxl else None
                attention_masks_list = [] if self.is_zimage else None

                for item, dataset in batch:
                    # Load latent from cache
                    cache = latent_caches[dataset.unique_id]
                    # BucketManager stores bucket_width/bucket_height, not width/height
                    width = item.get("width") or item.get("bucket_width")
                    height = item.get("height") or item.get("bucket_height")
                    latent = cache.load_latent(item["image_path"], width, height)
                    latents_list.append(latent)

                    # Encode caption
                    caption = item.get("caption", "")
                    if self.is_zimage:
                        if text_encoder_cache and caption in text_encoder_cache:
                            prompt_embeds, attention_mask = text_encoder_cache[caption]
                            text_embeddings_list.append(prompt_embeds.to(self.device))
                            attention_masks_list.append(attention_mask.to(self.device))
                        else:
                            prompt_embeds, attention_mask = self.encode_prompt_zimage(caption)
                            text_embeddings_list.append(prompt_embeds)
                            attention_masks_list.append(attention_mask)
                    elif self.is_sdxl:
                        text_emb, pooled_emb = self.encode_prompt(caption, requires_grad=True)
                        text_embeddings_list.append(text_emb)
                        pooled_embeddings_list.append(pooled_emb)
                    else:
                        text_emb = self.encode_prompt(caption, requires_grad=True)
                        text_embeddings_list.append(text_emb)

                # Stack batch
                latents = torch.cat(latents_list, dim=0)
                text_embeddings = torch.stack(text_embeddings_list, dim=0) if text_embeddings_list else None

                # Collect batch captions for debug output
                batch_captions = [item.get("caption", "") for item, dataset in batch]

                # Determine if we should save debug latents
                debug_save_path = None
                if debug_dir is not None and global_step % debug_latents_every == 0:
                    debug_save_path = debug_dir / f"step_{global_step:06d}"

                # Training step
                if self.is_zimage:
                    attention_mask = torch.stack(attention_masks_list, dim=0)
                    loss, recon_loss = self.train_step_zimage(
                        latents=latents,
                        prompt_embeds=text_embeddings,
                        attention_mask=attention_mask,
                        debug_save_path=debug_save_path,
                        debug_captions=batch_captions,
                        profile_vram=self.debug_vram,
                    )
                else:
                    pooled_embeddings = torch.stack(pooled_embeddings_list, dim=0) if pooled_embeddings_list else None
                    loss, recon_loss = self.train_step(
                        latents=latents,
                        text_embeddings=text_embeddings,
                        pooled_embeddings=pooled_embeddings,
                        debug_save_path=debug_save_path,
                        debug_captions=batch_captions,
                        profile_vram=self.debug_vram,
                    )

                # Backward pass
                loss_tensor = torch.tensor(loss, requires_grad=True)
                loss_tensor.backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_grad_norm)

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                    # Logging
                    self.writer.add_scalar("train/loss", loss, global_step)
                    self.writer.add_scalar("train/recon_loss", recon_loss, global_step)
                    self.writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], global_step)

                    # Save checkpoint
                    if global_step % save_every_n_steps == 0:
                        self.save_checkpoint(step=global_step, epoch=epoch)

                    # Generate sample
                    if global_step % sample_every_n_steps == 0:
                        if self.is_zimage:
                            sample = self._generate_sample_zimage(
                                prompt=sample_prompt,
                                num_inference_steps=sample_steps,
                                guidance_scale=sample_guidance_scale
                            )
                        else:
                            sample = self.generate_sample(
                                prompt=sample_prompt,
                                num_inference_steps=sample_steps,
                                guidance_scale=sample_guidance_scale
                            )

                        sample_path = self.output_dir / "samples" / f"step_{global_step}.png"
                        sample_path.parent.mkdir(parents=True, exist_ok=True)
                        sample.save(sample_path)
                        print(f"{self.log_prefix} Saved sample to {sample_path}")

                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            phase="training",
                            step=global_step,
                            total=total_steps,
                            epoch=epoch,
                            loss=loss,
                        )

        print(f"{self.log_prefix} Training complete!")
        self.writer.close()
