"""
Base Trainer for Stable Diffusion Models

Provides common functionality for LoRA and Full Parameter training.
"""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype"""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    return dtype_map.get(dtype_str, torch.float16)


class BaseTrainer:
    """
    Base trainer for Stable Diffusion models.

    Provides common functionality:
    - Model loading (U-Net, VAE, Text Encoders)
    - Precision settings
    - Prompt encoding
    - Image encoding
    - Training step (noise prediction, loss calculation)
    - Sample generation
    - Latent caching
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
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
    ):
        """
        Initialize base trainer.

        Args:
            model_path: Path to base Stable Diffusion model
            output_dir: Directory to save checkpoints
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
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

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

        # Optimizer and scheduler (to be set by subclass)
        self.optimizer = None
        self.lr_scheduler = None

        # Print precision settings
        print(f"[BaseTrainer] Precision settings:")
        print(f"  Weight dtype: {weight_dtype} ({self.weight_dtype})")
        print(f"  Training dtype: {training_dtype} ({self.training_dtype})")
        print(f"  Output dtype: {output_dtype} ({self.output_dtype}) - for safetensors saving")
        print(f"  VAE dtype: {vae_dtype} ({self.vae_dtype})")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Loss calculation: Always FP32 for numerical stability")
        print(f"  Min-SNR gamma: {min_snr_gamma} ({'enabled' if min_snr_gamma > 0 else 'disabled'})")

        # Initialize tensorboard writer
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = self.output_dir / "tensorboard" / timestamp
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        print(f"[BaseTrainer] Initializing on {self.device}")
        print(f"[BaseTrainer] Tensorboard logs: {tensorboard_dir}")
        print(f"[BaseTrainer] Loading model from {model_path}")

        # Load model components
        self.load_model(model_path)

        # Enable gradient checkpointing (saves VRAM during training)
        if hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()
            print("[BaseTrainer] Gradient checkpointing enabled for U-Net")

    def load_model(self, model_path: str):
        """
        Load model components from safetensors or diffusers directory.

        Sets: self.vae, self.text_encoder, self.tokenizer, self.unet, self.noise_scheduler
              self.text_encoder_2, self.tokenizer_2 (SDXL only)
              self.is_sdxl (bool)
        """
        is_safetensors = model_path.endswith('.safetensors')

        if is_safetensors:
            print(f"[BaseTrainer] Loading from safetensors file")
            # Try SDXL first, fall back to SD1.5
            try:
                print(f"[BaseTrainer] Trying SDXL pipeline...")
                temp_pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                )
                is_sdxl_model = True
            except Exception as e:
                print(f"[BaseTrainer] Not SDXL, trying SD1.5 pipeline...")
                temp_pipeline = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                )
                is_sdxl_model = False

            # Extract components
            self.vae = temp_pipeline.vae
            self.text_encoder = temp_pipeline.text_encoder
            self.tokenizer = temp_pipeline.tokenizer
            self.unet = temp_pipeline.unet
            self.noise_scheduler = temp_pipeline.scheduler

            # SDXL-specific components
            if is_sdxl_model:
                self.text_encoder_2 = temp_pipeline.text_encoder_2
                self.tokenizer_2 = temp_pipeline.tokenizer_2
            else:
                self.text_encoder_2 = None
                self.tokenizer_2 = None

            del temp_pipeline

            # Convert VAE to vae_dtype
            self.vae = self.vae.to(dtype=self.vae_dtype)
        else:
            print(f"[BaseTrainer] Loading from diffusers directory")
            # Load components from diffusers directory
            self.vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", torch_dtype=self.vae_dtype
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=self.dtype
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer"
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                model_path, subfolder="unet", torch_dtype=self.dtype
            )

            # Try to load SDXL components
            try:
                self.text_encoder_2 = CLIPTextModel.from_pretrained(
                    model_path, subfolder="text_encoder_2", torch_dtype=self.dtype
                )
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                    model_path, subfolder="tokenizer_2"
                )
                is_sdxl_model = True
            except:
                self.text_encoder_2 = None
                self.tokenizer_2 = None
                is_sdxl_model = False

            # Load scheduler
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )

        # Store model type
        self.is_sdxl = is_sdxl_model

        # Log model info
        print(f"[BaseTrainer] Model type: {'SDXL' if self.is_sdxl else 'SD1.5'}")
        print(f"[BaseTrainer] Prediction type: {self.noise_scheduler.config.prediction_type}")
        print(f"[BaseTrainer] VAE scaling factor: {self.vae.config.scaling_factor}")

    def setup_flash_attention(self):
        """
        Setup Flash Attention processors for U-Net.
        Should be called after model loading and before training.
        """
        if not self.use_flash_attention:
            return

        print("[BaseTrainer] Setting Flash Attention processors...")

        try:
            from core.inference.attention_processors import FlashAttnProcessor
            processor = FlashAttnProcessor()

            # Count attention layers
            attn_count = 0
            for name, module in self.unet.named_modules():
                if hasattr(module, 'set_processor'):
                    module.set_processor(processor)
                    attn_count += 1

            print(f"[BaseTrainer] [OK] Flash Attention enabled for {attn_count} attention layers")
        except Exception as e:
            print(f"[BaseTrainer] Failed to enable Flash Attention: {e}")
            print("[BaseTrainer] Continuing with default attention")

    def setup_optimizer(self, optimizer_type: str, lr_scheduler_type: str, total_steps: int = 1000):
        """
        Setup optimizer and learning rate scheduler.
        Must be implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement setup_optimizer()")

    def train(self, dataset_items: list, **kwargs):
        """
        Main training loop.
        Must be implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement train()")
