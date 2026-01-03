"""
LoRA Training Engine for SushiUI

Custom LoRA training implementation using SushiUI's architecture.
Inherits from BaseTrainer and adds LoRA-specific functionality.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List
from safetensors.torch import save_file, load_file
import numpy as np
import re

from .base_trainer import BaseTrainer, get_torch_dtype


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


class LoRATrainer(BaseTrainer):
    """LoRA trainer using SushiUI's component-based architecture."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        run_name: str = None,
        run_id: Optional[int] = None,  # Database run ID for metrics logging
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
        # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
        prompt_chunking_mode: str = "a1111",
        max_prompt_chunks: int = 0,
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
        # Store LoRA-specific parameters BEFORE calling parent __init__
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_layers = {}  # Storage for LoRA layers

        # Component-specific learning rates (for optimizer setup)
        self.unet_lr = unet_lr or learning_rate
        self.text_encoder_1_lr = text_encoder_1_lr or text_encoder_lr or (learning_rate * 0.5)
        self.text_encoder_2_lr = text_encoder_2_lr or text_encoder_lr or (learning_rate * 0.5)

        # Debug: Log LR values
        print(f"[LoRATrainer.__init__] LR parameters received:")
        print(f"  learning_rate={learning_rate}")
        print(f"  unet_lr={unet_lr} → self.unet_lr={self.unet_lr}")
        print(f"  text_encoder_lr={text_encoder_lr}")
        print(f"  text_encoder_1_lr={text_encoder_1_lr} → self.text_encoder_1_lr={self.text_encoder_1_lr}")
        print(f"  text_encoder_2_lr={text_encoder_2_lr} → self.text_encoder_2_lr={self.text_encoder_2_lr}")

        # Call parent __init__ (loads model components)
        super().__init__(
            model_path=model_path,
            output_dir=output_dir,
            run_name=run_name,
            run_id=run_id,  # Pass run_id for DB metrics logging
            learning_rate=learning_rate,
            device=device,
            weight_dtype=weight_dtype,
            training_dtype=training_dtype,
            output_dtype=output_dtype,
            vae_dtype=vae_dtype,
            mixed_precision=mixed_precision,
            debug_vram=debug_vram,
            use_flash_attention=use_flash_attention,
            min_snr_gamma=min_snr_gamma,
            # Component-specific learning rates (pass to BaseTrainer)
            unet_lr=unet_lr,
            text_encoder_lr=text_encoder_lr,
            text_encoder_1_lr=text_encoder_1_lr,
            text_encoder_2_lr=text_encoder_2_lr,
            # Optimizer options
            optimizer_is_paged=optimizer_is_paged,
            optimizer_cautious=optimizer_cautious,
            optimizer_beta1=optimizer_beta1,
            optimizer_beta2=optimizer_beta2,
            optimizer_epsilon=optimizer_epsilon,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_schedule_free=optimizer_schedule_free,
            optimizer_warmup_steps=optimizer_warmup_steps,
            optimizer_schedule_free_r=optimizer_schedule_free_r,
            optimizer_schedule_free_weight_lr_power=optimizer_schedule_free_weight_lr_power,
            optimizer_use_radam=optimizer_use_radam,
            # Prompt chunking
            prompt_chunking_mode=prompt_chunking_mode,
            max_prompt_chunks=max_prompt_chunks,
        )

        # Override log prefix
        self.specific_log_prefix = "[LoRATrainer]"

        # Apply LoRA layers to loaded model components
        # This must be called AFTER parent __init__ (model components loaded)
        # and BEFORE train_runner calls setup_optimizer()
        self.setup_trainable_parameters()

    def setup_trainable_parameters(self):
        """
        Implement BaseTrainer abstract method - Apply LoRA layers.

        This is called by BaseTrainer.__init__() after model components are loaded.
        """
        print(f"{self.specific_log_prefix} Applying LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})")

        if self.is_zimage:
            # Z-Image: Apply LoRA to Transformer only (Text Encoder is frozen)
            self._apply_lora_zimage()

            # Set VAE to eval mode (never trained)
            self.vae.eval()

            # Transformer must be in train mode for gradient checkpointing to work
            # Text Encoder remains in eval mode (frozen)
            self.transformer.train()
            self.text_encoder.eval()
            print(f"{self.log_prefix} Z-Image Transformer set to train mode, Text Encoder to eval mode (frozen)")

            # Note: Gradient checkpointing re-enable DISABLED
            # Reason: gradient_checkpointing_enable() causes device placement issues
            # Testing shows gradient checkpointing continues to work after LoRA injection
            # without re-enabling (hooks are preserved on parent modules)

            # TEMPORARY: Commented out to test if gradient checkpointing works without re-enable
            # if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            #     self.transformer.enable_gradient_checkpointing()
            #     print(f"{self.log_prefix} Gradient checkpointing re-enabled for Z-Image Transformer after LoRA injection")
        else:
            # SD/SDXL: Apply LoRA to U-Net and Text Encoder
            self._apply_lora()

            # Set VAE to eval mode (never trained)
            self.vae.eval()

            # U-Net and Text Encoders must be in train mode for gradient checkpointing to work (sd-scripts approach)
            # This is required according to Diffusers TI example
            self.unet.train()
            self.text_encoder.train()
            if self.text_encoder_2 is not None:
                self.text_encoder_2.train()
            print(f"{self.log_prefix} U-Net and Text Encoders set to train mode for gradient checkpointing")

            # CRITICAL: Set embedding layer requires_grad=True for gradient checkpointing (sd-scripts approach)
            # This is required for gradients to flow through embedding layers during checkpointing
            # (embeddings are leaf tensors in the computation graph)
            # Also ensure embedding layer is on the correct device
            if hasattr(self.text_encoder, 'text_model') and hasattr(self.text_encoder.text_model, 'embeddings'):
                self.text_encoder.text_model.embeddings.requires_grad_(True)
                # Ensure embedding layer is on GPU (may be on CPU after requires_grad_ call)
                self.text_encoder.text_model.embeddings.to(self.device, dtype=self.weight_dtype)
                print(f"{self.log_prefix} Text Encoder 1 embedding layer set to requires_grad=True and moved to {self.device}")

            if self.text_encoder_2 is not None:
                if hasattr(self.text_encoder_2, 'text_model') and hasattr(self.text_encoder_2.text_model, 'embeddings'):
                    self.text_encoder_2.text_model.embeddings.requires_grad_(True)
                    # Ensure embedding layer is on GPU
                    self.text_encoder_2.text_model.embeddings.to(self.device, dtype=self.weight_dtype)
                    print(f"{self.log_prefix} Text Encoder 2 embedding layer set to requires_grad=True and moved to {self.device}")

            # Note: Gradient checkpointing re-enable DISABLED
            # Reason: gradient_checkpointing_enable() causes device placement issues
            # with Transformers library (Embedding layers stay on CPU when model moved to GPU)
            # Testing shows gradient checkpointing continues to work after LoRA injection
            # without re-enabling (hooks are preserved on parent modules)

            # TEMPORARY: Commented out to test if gradient checkpointing works without re-enable
            # if hasattr(self.unet, 'enable_gradient_checkpointing'):
            #     self.unet.enable_gradient_checkpointing()
            #     print(f"{self.log_prefix} Gradient checkpointing re-enabled for U-Net after LoRA injection")

            # if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            #     self.text_encoder.gradient_checkpointing_enable()
            #     print(f"{self.log_prefix} Gradient checkpointing re-enabled for Text Encoder 1 after LoRA injection")

            # if self.text_encoder_2 is not None:
            #     if hasattr(self.text_encoder_2, 'gradient_checkpointing_enable'):
            #         self.text_encoder_2.gradient_checkpointing_enable()
            #         print(f"{self.log_prefix} Gradient checkpointing re-enabled for Text Encoder 2 after LoRA injection")

    def _apply_lora(self):
        """Apply LoRA layers to SD/SDXL model modules."""
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

    def setup_optimizer(
        self,
        optimizer_type: str = "adamw8bit",
        lr_scheduler_type: str = "constant",
        total_steps: int = 1000,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Setup optimizer and learning rate scheduler for LoRA parameters.

        Args:
            optimizer_type: Optimizer type (see OptimizerFactory.get_available_optimizers())
            lr_scheduler_type: LR scheduler type
            total_steps: Total training steps
            weight_decay: Weight decay coefficient
            betas: Adam beta parameters
            eps: Adam epsilon
        """
        from .optimizer_factory import OptimizerFactory

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
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
        except (ValueError, ImportError) as e:
            print(f"{self.log_prefix} ERROR: {e}")
            print(f"{self.log_prefix} Falling back to AdamW")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,  # This will be overridden by param_groups
                betas=betas,
                weight_decay=weight_decay,
                eps=eps,
            )

        # Setup LR scheduler
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        self.lr_scheduler = get_diffusers_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

    def verify_gradient_flow(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Verify that gradients are flowing to all LoRA layers.

        Args:
            verbose: If True, print detailed gradient statistics for each layer

        Returns:
            Dictionary with gradient flow statistics:
            {
                "total_lora_params": int,
                "params_with_grad": int,
                "params_without_grad": int,
                "layers_with_grad": List[str],
                "layers_without_grad": List[str],
                "grad_stats": Dict[str, Dict] (if verbose=True)
            }
        """
        total_params = 0
        params_with_grad = 0
        params_without_grad = 0
        layers_with_grad = []
        layers_without_grad = []
        grad_stats = {}

        # Check all LoRA layers
        for lora_name, lora_module in self.lora_layers.items():
            # Check lora_up and lora_down parameters
            for param_name in ['lora_up', 'lora_down']:
                if hasattr(lora_module, param_name):
                    param_module = getattr(lora_module, param_name)
                    for param in param_module.parameters():
                        total_params += 1
                        full_name = f"{lora_name}.{param_name}"

                        if param.grad is not None:
                            params_with_grad += 1
                            if full_name not in layers_with_grad:
                                layers_with_grad.append(full_name)

                            if verbose:
                                grad_stats[full_name] = {
                                    "grad_mean": param.grad.mean().item(),
                                    "grad_std": param.grad.std().item(),
                                    "grad_max": param.grad.abs().max().item(),
                                    "grad_min": param.grad.abs().min().item(),
                                    "param_shape": list(param.shape),
                                }
                        else:
                            params_without_grad += 1
                            if full_name not in layers_without_grad:
                                layers_without_grad.append(full_name)

        result = {
            "total_lora_params": total_params,
            "params_with_grad": params_with_grad,
            "params_without_grad": params_without_grad,
            "layers_with_grad": layers_with_grad,
            "layers_without_grad": layers_without_grad,
        }

        if verbose:
            result["grad_stats"] = grad_stats

        return result

    def print_gradient_flow_summary(self):
        """Print a summary of gradient flow to all LoRA layers."""
        stats = self.verify_gradient_flow(verbose=False)

        print(f"\n{'='*60}")
        print(f"[LoRA Trainer] Gradient Flow Verification")
        print(f"{'='*60}")
        print(f"Total LoRA parameters: {stats['total_lora_params']}")
        print(f"Parameters WITH gradients: {stats['params_with_grad']} ({stats['params_with_grad']/stats['total_lora_params']*100:.1f}%)")
        print(f"Parameters WITHOUT gradients: {stats['params_without_grad']} ({stats['params_without_grad']/stats['total_lora_params']*100:.1f}%)")

        if stats['params_without_grad'] > 0:
            print(f"\n⚠️  WARNING: {stats['params_without_grad']} parameters have NO gradients!")
            print(f"Layers without gradients:")
            for layer_name in stats['layers_without_grad'][:10]:  # Show first 10
                print(f"  - {layer_name}")
            if len(stats['layers_without_grad']) > 10:
                print(f"  ... and {len(stats['layers_without_grad']) - 10} more")
        else:
            print(f"\n✓ All LoRA parameters have gradients!")

        print(f"{'='*60}\n")

        return stats

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        save_path: Optional[str] = None,
        save_optimizer: bool = True,
        max_to_keep: Optional[int] = None,
        save_every: int = 100,
        run_id: Optional[int] = None,
    ):
        """
        Save LoRA checkpoint as safetensors and optimizer state as .pt.

        Args:
            step: Current training step
            epoch: Current epoch
            save_path: Path to save checkpoint (default: output_dir/{run_name}_step_{step}.safetensors)
            save_optimizer: Whether to save optimizer state (default: True)
            max_to_keep: Maximum number of checkpoints to keep (None = keep all)
            save_every: Save interval (used to calculate which checkpoint to delete)
            run_id: Training run ID (for DB registration)
        """
        if save_path is None:
            # Extract short name from run_name
            # If run_name is in format "YYYYMMDD_HHMMSS_ID", use only ID
            # Otherwise, use full run_name
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
        print(f"{self.log_prefix} [Debug] Total LoRA layers in self.lora_layers: {len(self.lora_layers)}")

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

        # Add metadata (diffusers-compatible format + ModelSpec)
        metadata = {
            "format": "diffusers",  # Indicate this is diffusers format, not SD format
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "base_model": self.model_path,
            "training_step": str(step),
            "output_dtype": str(self.output_dtype),
        }

        # Add ModelSpec 1.0.0 metadata for prediction configuration
        # Note: LoRA inherits prediction config from base model, so we save it for reference
        if hasattr(self, 'noise_process') and hasattr(self, 'prediction_target'):
            # ModelSpec standard keys
            metadata["modelspec.architecture"] = "lora"
            metadata["modelspec.implementation"] = "https://github.com/huggingface/diffusers"
            metadata["modelspec.title"] = f"LoRA trained on {Path(self.model_path).stem}"

            # Prediction configuration (unified training framework)
            metadata["modelspec.noise_process"] = self.noise_process  # "ddpm" or "flow"
            metadata["modelspec.prediction_type"] = self.prediction_target  # "epsilon", "velocity", "sample"

            # Legacy compatibility: add v_pred marker for v-prediction models
            if self.prediction_target == "velocity" and self.noise_process == "ddpm":
                # Add empty v_pred tensor as marker (NoobAI-XL-Vpred style)
                state_dict["v_pred"] = torch.tensor([], dtype=self.output_dtype)

        # Save as safetensors
        print(f"{self.log_prefix} [Debug] state_dict keys: {len(state_dict)}")
        if len(state_dict) > 0:
            print(f"{self.log_prefix} [Debug] Sample keys: {list(state_dict.keys())[:5]}")

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
        match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
        if match:
            short_name = match.group(1)
        else:
            short_name = self.run_name

        # Build checkpoint path to remove
        checkpoint_to_remove = self.output_dir / f"{short_name}_step_{remove_step}.safetensors"
        optimizer_to_remove = self.output_dir / f"{short_name}_step_{remove_step}.pt"

        # Remove checkpoint if it exists
        if checkpoint_to_remove.exists():
            checkpoint_to_remove.unlink()
            print(f"{self.log_prefix} Removed old checkpoint: {checkpoint_to_remove.name}")

        # Remove optimizer state if it exists
        if optimizer_to_remove.exists():
            optimizer_to_remove.unlink()
            print(f"{self.log_prefix} Removed old optimizer state: {optimizer_to_remove.name}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load LoRA checkpoint from safetensors file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with 'step' and 'epoch' keys
        """
        print(f"{self.log_prefix} Loading checkpoint from {checkpoint_path}")

        state_dict = load_file(checkpoint_path)

        # Detect checkpoint type
        is_lora_checkpoint = any("lora_down" in key or "lora_up" in key for key in state_dict.keys())

        if is_lora_checkpoint:
            # Load LoRA weights
            print(f"{self.log_prefix} Detected LoRA checkpoint")
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
        else:
            # Load Full Finetune weights (entire model)
            print(f"{self.log_prefix} Detected Full Finetune checkpoint")

            # Load weights directly into transformer
            if self.is_zimage and hasattr(self, 'transformer'):
                # Load state dict into transformer (incompatible keys are expected)
                missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, strict=False)
                print(f"{self.log_prefix} Loaded Full Finetune weights into Transformer")
                if missing_keys:
                    print(f"{self.log_prefix} WARNING: Missing keys: {len(missing_keys)} (expected for partial checkpoint)")
                if unexpected_keys:
                    print(f"{self.log_prefix} WARNING: Unexpected keys: {len(unexpected_keys)}")
            else:
                # SDXL/SD1.5 U-Net
                if hasattr(self, 'unet'):
                    missing_keys, unexpected_keys = self.unet.load_state_dict(state_dict, strict=False)
                    print(f"{self.log_prefix} Loaded Full Finetune weights into U-Net")
                    if missing_keys:
                        print(f"{self.log_prefix} WARNING: Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"{self.log_prefix} WARNING: Unexpected keys: {len(unexpected_keys)}")
                else:
                    print(f"{self.log_prefix} WARNING: No model to load weights into (unet/transformer not found)")

        # Extract step from metadata or filename
        step = 0
        epoch = 0

        # Try metadata first (safetensors format)
        try:
            # safetensors.torch.load_file doesn't expose metadata directly
            # We need to use safetensors.safe_open to read metadata
            from safetensors import safe_open
            with safe_open(checkpoint_path, framework="pt") as f:
                metadata = f.metadata()
                if metadata:
                    if 'training_step' in metadata:
                        step = int(metadata['training_step'])
                    if 'epoch' in metadata:
                        epoch = int(metadata['epoch'])
        except Exception:
            pass

        # Fallback: extract from filename
        if step == 0:
            try:
                step_str = Path(checkpoint_path).stem.split("_")[-1]
                step = int(step_str)
            except (ValueError, IndexError):
                pass

        print(f"{self.log_prefix} Checkpoint loaded (step {step}, epoch {epoch})")

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

        return {"step": step, "epoch": epoch}

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest valid LoRA checkpoint in output directory.

        Strategy:
        1. Find all .safetensors files
        2. Validate each checkpoint (can be loaded)
        3. Extract step number
        4. Return the one with highest step number

        Returns:
            Checkpoint path or None if no valid checkpoint found
        """
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

                # Try metadata first
                try:
                    from safetensors import safe_open
                    with safe_open(str(ckpt_path), framework="pt") as f:
                        metadata = f.metadata()
                        if metadata and 'training_step' in metadata:
                            step = int(metadata['training_step'])
                except Exception:
                    pass

                # Fallback: extract from filename (any file with "step_{number}")
                if step == 0:
                    stem = ckpt_path.stem
                    parts = stem.split("_")
                    if "step" in parts:
                        step_idx = parts.index("step")
                        if step_idx + 1 < len(parts):
                            step = int(parts[step_idx + 1])

                # Check if this checkpoint has weights (validation)
                # LoRA: check for lora_down/lora_up keys
                # Full Finetune: check for any model weights
                has_lora_weights = any("lora_down" in key or "lora_up" in key for key in state_dict.keys())
                has_model_weights = len(state_dict.keys()) > 0  # Any weights indicate valid checkpoint

                if has_lora_weights or has_model_weights:
                    valid_checkpoints.append((str(ckpt_path), step))
                    checkpoint_type = "LoRA" if has_lora_weights else "Full Finetune"
                    print(f"{self.log_prefix} Found valid checkpoint: {ckpt_path.name} (step {step}, {checkpoint_type})")

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

        return latest_ckpt
