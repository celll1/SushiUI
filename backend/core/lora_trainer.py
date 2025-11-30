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
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file
import json
from datetime import datetime
import numpy as np


class LoRALinearLayer(torch.nn.Module):
    """Simple LoRA linear layer implementation."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)

        # Initialize
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.lora_down(x)) * self.scaling


def inject_lora_into_linear(module: torch.nn.Linear, rank: int = 4, alpha: float = 1.0):
    """Inject LoRA into a linear layer."""
    lora = LoRALinearLayer(
        module.in_features,
        module.out_features,
        rank=rank,
        alpha=alpha
    )
    lora.to(module.weight.device, dtype=module.weight.dtype)
    return lora


class LoRATrainer:
    """LoRA trainer using SushiUI's component-based architecture."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        """
        Initialize LoRA trainer.

        Args:
            model_path: Path to base Stable Diffusion model
            output_dir: Directory to save checkpoints
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor)
            learning_rate: Learning rate
            device: Device to use (cuda/cpu)
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(f"[LoRATrainer] Initializing on {self.device}")
        print(f"[LoRATrainer] Loading model from {model_path}")

        # Detect if model is safetensors file or diffusers directory
        is_safetensors = model_path.endswith('.safetensors')

        if is_safetensors:
            print(f"[LoRATrainer] Loading from safetensors file")
            # Load pipeline from single safetensors file, then extract components
            # This is the same approach SushiUI uses in model_loader.py
            temp_pipeline = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )

            # Extract components from pipeline
            self.vae = temp_pipeline.vae
            self.text_encoder = temp_pipeline.text_encoder
            self.tokenizer = temp_pipeline.tokenizer
            self.unet = temp_pipeline.unet
            self.noise_scheduler = temp_pipeline.scheduler

            # Clean up pipeline reference (we only need components)
            del temp_pipeline
        else:
            print(f"[LoRATrainer] Loading from diffusers directory")
            # Load model components from diffusers directory (SushiUI style)
            self.vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=self.dtype
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

        # Freeze all base weights
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Move to device
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)

        # Set to eval mode (except UNet which will have LoRA)
        self.vae.eval()
        self.text_encoder.eval()

        # LoRA layers storage
        self.lora_layers = {}

        # Apply LoRA to UNet
        self._apply_lora()

        self.optimizer = None
        self.lr_scheduler = None

    def _apply_lora(self):
        """Apply LoRA layers to UNet attention modules."""
        print(f"[LoRATrainer] Applying LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})")

        # Target attention projection layers
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

        lora_count = 0

        # Recursively find and inject LoRA into target modules
        for name, module in self.unet.named_modules():
            # Check if this is a target module
            if any(target in name for target in target_modules):
                if isinstance(module, torch.nn.Linear):
                    # Create LoRA layer
                    lora = inject_lora_into_linear(module, self.lora_rank, self.lora_alpha)
                    self.lora_layers[name] = lora
                    lora_count += 1

        print(f"[LoRATrainer] Injected {lora_count} LoRA layers")

        # Count trainable parameters
        trainable_params = sum(p.numel() for lora in self.lora_layers.values() for p in lora.parameters())
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"[LoRATrainer] Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def setup_optimizer(self, optimizer_type: str = "adamw8bit", lr_scheduler_type: str = "constant", total_steps: int = 1000):
        """Setup optimizer and learning rate scheduler."""
        print(f"[LoRATrainer] Setting up optimizer: {optimizer_type}")

        # Get trainable parameters (LoRA weights only)
        trainable_params = []
        for lora in self.lora_layers.values():
            trainable_params.extend(lora.parameters())

        if optimizer_type == "adamw8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    eps=1e-8,
                )
                print("[LoRATrainer] Using AdamW8bit optimizer")
            except ImportError:
                print("[LoRATrainer] bitsandbytes not available, falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    eps=1e-8,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
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

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]

        return text_embeddings

    def encode_image(self, image: Image.Image, target_size: int = 512) -> torch.Tensor:
        """Encode image to latents."""
        # Resize to target size
        image = image.convert("RGB")
        image = image.resize((target_size, target_size), Image.LANCZOS)

        # Convert to tensor and normalize to [-1, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = (image_array - 0.5) * 2.0

        # Convert to torch tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device=self.device, dtype=self.vae.dtype)

        # Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        return latents

    def unet_forward_with_lora(self, sample, timestep, encoder_hidden_states):
        """Forward pass through UNet with LoRA injected."""
        # Store original forward methods
        original_forwards = {}

        # Inject LoRA into forward pass
        for name, module in self.unet.named_modules():
            if name in self.lora_layers and isinstance(module, torch.nn.Linear):
                lora = self.lora_layers[name]

                # Save original forward
                original_forwards[name] = module.forward

                # Create new forward that includes LoRA
                def make_forward_with_lora(original_module, lora_layer):
                    def forward_with_lora(x):
                        return original_module._original_forward(x) + lora_layer(x)
                    return forward_with_lora

                # Temporarily replace forward
                module._original_forward = module.forward
                module.forward = make_forward_with_lora(module, lora)

        # Forward pass
        output = self.unet(sample, timestep, encoder_hidden_states)

        # Restore original forwards
        for name, module in self.unet.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
                if hasattr(module, '_original_forward'):
                    delattr(module, '_original_forward')

        return output

    def train_step(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> float:
        """
        Perform single training step.

        Args:
            latents: Image latents [B, C, H, W]
            text_embeddings: Text prompt embeddings [B, 77, 768]

        Returns:
            Loss value
        """
        # Sample noise
        noise = torch.randn_like(latents)

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

        # Predict noise using UNet with LoRA
        model_pred = self.unet_forward_with_lora(
            noisy_latents,
            timesteps,
            text_embeddings
        ).sample

        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for lora in self.lora_layers.values() for p in lora.parameters()],
            max_norm=1.0
        )

        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.detach().item()

    def save_checkpoint(self, step: int, save_path: Optional[str] = None):
        """Save LoRA checkpoint as safetensors."""
        if save_path is None:
            save_path = self.output_dir / f"lora_step_{step}.safetensors"
        else:
            save_path = Path(save_path)

        print(f"[LoRATrainer] Saving checkpoint to {save_path}")

        # Collect all LoRA weights
        state_dict = {}
        for name, lora in self.lora_layers.items():
            # Save with proper naming convention
            key_prefix = f"lora_unet_{name.replace('.', '_')}"
            state_dict[f"{key_prefix}.lora_down.weight"] = lora.lora_down.weight.detach().cpu()
            state_dict[f"{key_prefix}.lora_up.weight"] = lora.lora_up.weight.detach().cpu()

        # Add metadata
        metadata = {
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(self.lora_rank),
            "ss_network_alpha": str(self.lora_alpha),
            "ss_base_model": self.model_path,
            "ss_training_step": str(step),
        }

        # Save as safetensors
        save_file(state_dict, str(save_path), metadata=metadata)

        print(f"[LoRATrainer] Checkpoint saved: {save_path}")

    def train(
        self,
        dataset_items: List[Dict[str, Any]],
        num_epochs: int = 1,
        batch_size: int = 1,
        save_every: int = 100,
        sample_every: int = 100,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
    ):
        """
        Train LoRA on dataset.

        Args:
            dataset_items: List of dataset items (image_path, caption)
            num_epochs: Number of epochs
            batch_size: Batch size (currently only supports 1)
            save_every: Save checkpoint every N steps
            sample_every: Generate sample every N steps
            progress_callback: Callback(step, loss, lr) for progress updates
        """
        print(f"[LoRATrainer] Starting training")
        print(f"[LoRATrainer] Dataset: {len(dataset_items)} items")
        print(f"[LoRATrainer] Epochs: {num_epochs}")
        print(f"[LoRATrainer] Batch size: {batch_size}")

        total_steps = len(dataset_items) * num_epochs

        if self.optimizer is None:
            self.setup_optimizer(total_steps=total_steps)

        # Training loop
        global_step = 0

        for epoch in range(num_epochs):
            print(f"\n[LoRATrainer] === Epoch {epoch + 1}/{num_epochs} ===")

            for item in tqdm(dataset_items, desc=f"Epoch {epoch + 1}"):
                try:
                    # Load and encode image
                    image_path = item["image_path"]
                    if not os.path.exists(image_path):
                        print(f"[LoRATrainer] WARNING: Image not found: {image_path}")
                        continue

                    image = Image.open(image_path)
                    latents = self.encode_image(image)

                    # Encode caption
                    caption = item.get("caption", "")
                    text_embeddings = self.encode_prompt(caption)

                    # Training step
                    loss = self.train_step(latents, text_embeddings)

                    global_step += 1

                    # Progress callback
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    if progress_callback:
                        progress_callback(global_step, loss, current_lr)

                    # Save checkpoint
                    if global_step % save_every == 0:
                        self.save_checkpoint(global_step)

                    # TODO: Sample generation
                    # if global_step % sample_every == 0:
                    #     self.generate_sample(...)

                except Exception as e:
                    print(f"[LoRATrainer] ERROR processing {item.get('image_path', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"\n[LoRATrainer] Training completed! Total steps: {global_step}")

        # Save final checkpoint
        self.save_checkpoint(global_step, self.output_dir / "lora_final.safetensors")
