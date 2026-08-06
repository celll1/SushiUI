"""
Full Parameter Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- SD15FullParameterAdapter: SD1.5 models
- SDXLFullParameterAdapter: SDXL models
- ZImageFullParameterAdapter: Z-Image models
- FLUX2FullParameterAdapter: FLUX.2 Klein models

Key improvements:
- Model-specific logic separated into adapters
- Supports SD1.5, SDXL, Z-Image, and FLUX.2
- Clean separation of concerns
- Easy to extend with new model types

References:
- sd-scripts (Apache-2 license) by kohya-ss
- ai-toolkit (MIT license) by ostris
- musubi-tuner (Apache-2 license) by kohya-ss (Z-Image support)

Author: Claude (2026-01-04)
"""

from pathlib import Path
from typing import Dict, List
import torch

from .base_trainer import BaseTrainer
from .adapters import (
    SD15FullParameterAdapter,
    SDXLFullParameterAdapter,
    ZImageFullParameterAdapter,
    # DEUSFullParameterAdapter,  # DEUS support removed
    FLUX2FullParameterAdapter,
    AnimaFullParameterAdapter,
    LensFullParameterAdapter,
    MiniT2IFullParameterAdapter,
    Krea2FullParameterAdapter,
    Ltx2FullParameterAdapter,
    AceStepFullParameterAdapter,
)


class FullParameterTrainer(BaseTrainer):
    """
    Full Parameter Trainer for SD/SDXL/Z-Image models.

    Uses model-specific adapters for parameter preparation, collection,
    and checkpoint saving.
    """

    def __init__(
        self,
        train_unet: bool = True,
        train_text_encoder: bool = False,
        train_image_encoder: bool = False,  # Image Encoder (future support)
        **kwargs
    ):
        """
        Initialize Full Parameter Trainer.

        Args:
            train_unet: Whether to train U-Net/Transformer
            train_text_encoder: Whether to train Text Encoder(s)
            train_image_encoder: Whether to train Image Encoder (future support)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # Full fine-tune settings (set before super().__init__)
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder
        self.train_image_encoder = train_image_encoder

        # Architectures that refuse full fine-tuning are refused HERE, before
        # super().__init__() -- which loads the model. `_create_adapter()` runs
        # far below that, so a refusal expressed only there is paid for with the
        # entire load first (MiniMax-H3: a 21 GB DiT plus a 48 GiB memory-mapped
        # text encoder, minutes of work and a documented RAM cliff, to reach a
        # message that was knowable from the checkpoint header alone).
        self._refuse_unsupported_full_finetune(kwargs.get("model_path"))

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[Full Parameter Trainer]"

        # Create model-specific adapter
        self._create_adapter()

        # Prepare models for training using adapter
        self._prepare_models()

        # Anima block swap is deferred until after adapter sets requires_grad
        # / freezes-the-right-things. For Full FT this isn't strictly required
        # (no LoRA wrap to break the snapshot) but the post-adapter ordering
        # keeps the contract uniform with LoRATrainer. No-op for other archs.
        if hasattr(self, "setup_anima_block_swap"):
            self.setup_anima_block_swap()
        if hasattr(self, "setup_lens_block_swap"):
            self.setup_lens_block_swap()
        if hasattr(self, "setup_ideogram4_block_swap"):
            self.setup_ideogram4_block_swap()
        if hasattr(self, "setup_minit2i_block_swap"):
            self.setup_minit2i_block_swap()
        if hasattr(self, "setup_krea2_block_swap"):
            self.setup_krea2_block_swap()
        if hasattr(self, "setup_ltx2_wrapper"):
            self.setup_ltx2_wrapper()
        if hasattr(self, "setup_ltx2_block_swap"):
            self.setup_ltx2_block_swap()
        if hasattr(self, "setup_acestep_block_swap"):
            self.setup_acestep_block_swap()

        print(f"{self.log_prefix} Initialized")
        # Note: Vision Encoder training status is determined in train() after VE is loaded
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}, Image Encoder: {self.train_image_encoder}")

    @staticmethod
    def _refuse_unsupported_full_finetune(model_path):
        """Raise for an architecture that does not offer full fine-tuning, from
        the CHECKPOINT alone -- before anything is loaded.

        Reads the same table the API serves
        (`api.arch_capabilities.TRAINING_UNSUPPORTED`, which a client filters its
        method dropdown from) so the refusal, the dropdown and the documented
        reason can never disagree. Detection failures are non-fatal: an
        unreadable path is the loader's error to report, not this guard's.
        """
        if not model_path:
            return
        try:
            from core.model_loader import ModelLoader
            arch = ModelLoader.detect_model_type(model_path)
        except Exception:
            return
        from api.arch_capabilities import TRAINING_UNSUPPORTED
        reason = (TRAINING_UNSUPPORTED.get(arch) or {}).get("full_finetune")
        if reason:
            raise ValueError(
                f"Full fine-tuning is not supported for architecture '{arch}': {reason} "
                f"Use training_method='lora'.")

    def _create_adapter(self):
        """Create model-specific Full Parameter adapter based on detected model type."""
        if self.is_zimage:
            self.adapter = ZImageFullParameterAdapter(self)
            print(f"{self.log_prefix} Using ZImageFullParameterAdapter")
        # DEUS support removed - architecture no longer maintained
        # elif self.is_deus:
        #     self.adapter = DEUSFullParameterAdapter(self)
        #     print(f"{self.log_prefix} Using DEUSFullParameterAdapter")
        elif self.is_flux2:
            self.adapter = FLUX2FullParameterAdapter(self)
            print(f"{self.log_prefix} Using FLUX2FullParameterAdapter")
        elif self.is_anima:
            self.adapter = AnimaFullParameterAdapter(self)
            print(f"{self.log_prefix} Using AnimaFullParameterAdapter")
            # FP8 base + Full FT is incompatible (base needs gradients).
            # _load_anima_components already runs (before _create_adapter)
            # and base_trainer's fp8_base_dtype branch only triggers for
            # training_method='lora', so a Full FT request with the flag
            # set produces a warning there and is skipped. Nothing to do.
        elif self.is_lens:
            self.adapter = LensFullParameterAdapter(self)
            print(f"{self.log_prefix} Using LensFullParameterAdapter")
        elif self.is_minit2i:
            self.adapter = MiniT2IFullParameterAdapter(self)
            print(f"{self.log_prefix} Using MiniT2IFullParameterAdapter")
        elif self.is_krea2:
            self.adapter = Krea2FullParameterAdapter(self)
            print(f"{self.log_prefix} Using Krea2FullParameterAdapter")
        elif self.is_ltx2:
            self.adapter = Ltx2FullParameterAdapter(self)
            print(f"{self.log_prefix} Using Ltx2FullParameterAdapter")
        elif self.is_minimax_h3:
            # LAYER 3 of the three-layer full-FT refusal for this architecture
            # (design section 7). Layer 1 is
            # `api.arch_capabilities.TRAINING_UNSUPPORTED["minimax_h3"]
            # ["full_finetune"]`, which the UI filters its method dropdown from;
            # layer 2 is the deliberate ABSENCE of a
            # MiniMaxH3FullParameterAdapter class in
            # `adapters/minimax_h3_adapter.py`.
            #
            # The refusal a queued run actually hits is
            # `_refuse_unsupported_full_finetune`, raised BEFORE super().__init__()
            # loads anything. This branch is the backstop for a caller that
            # constructs the trainer some other way (a direct subclass, a test), so
            # the absence of an adapter can never be reached as a silent fallthrough
            # to the SD1.5 one.
            raise ValueError(
                "Full fine-tuning is not supported for MiniMax-H3. Its DiT is a 33 B dense "
                "transformer: parameters, gradients and optimizer state do not fit the "
                "single-GPU 48 GB envelope this integration targets, and the released base is "
                "weight-only FP8 (a full fine-tune would have to dequantize it first). Use "
                "training_method='lora' -- LoRA is measured at 22.45 GB peak on this box.")
        elif self.is_acestep:
            self.adapter = AceStepFullParameterAdapter(self)
            print(f"{self.log_prefix} Using AceStepFullParameterAdapter")
        elif self.is_sdxl:
            self.adapter = SDXLFullParameterAdapter(self)
            print(f"{self.log_prefix} Using SDXLFullParameterAdapter")
        else:
            self.adapter = SD15FullParameterAdapter(self)
            print(f"{self.log_prefix} Using SD15FullParameterAdapter")

    def _prepare_models(self):
        """Prepare models for full parameter training using adapter."""
        self.adapter.prepare_models_for_training()

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters with per-component learning rates.

        Uses adapter to handle model-specific parameter grouping.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters()

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save full parameter checkpoint.

        Uses adapter to handle model-specific checkpoint format.

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}"
        self.adapter.save_checkpoint(step, epoch, checkpoint_path)
        # Save Vision Encoder checkpoint separately (if loaded)
        self._save_vision_encoder_checkpoint(step, epoch)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load full parameter checkpoint for resuming training.

        Args:
            checkpoint_path: Path to checkpoint file (.safetensors) or directory (diffusers format)

        Returns:
            Step number from checkpoint
        """
        import json
        import re
        from safetensors.torch import load_file

        print(f"{self.log_prefix} Loading checkpoint: {checkpoint_path}")

        checkpoint_path_obj = Path(checkpoint_path)

        # Detect checkpoint format: safetensors file vs diffusers directory
        if checkpoint_path_obj.is_file() and checkpoint_path_obj.suffix == ".safetensors":
            # Single safetensors file format (Z-Image training)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            # Extract step number from filename: *_step_NNNNNN.safetensors
            step = 0
            match = re.search(r'_step_(\d+)\.safetensors$', checkpoint_path_obj.name)
            if match:
                step = int(match.group(1))

            # Load checkpoint using checkpoint_utils
            from core.models.checkpoint_utils import load_unified_checkpoint
            loaded_components = load_unified_checkpoint(str(checkpoint_path_obj), device='cpu')

            # Delete old models to free VRAM before loading checkpoint
            import gc
            if self.train_unet and loaded_components.get('unet') is not None:
                if self.unet is not None:
                    del self.unet
                    gc.collect()
                    torch.cuda.empty_cache()
                self.unet = loaded_components['unet']
                # Move to GPU (same as new training initialization)
                self.unet.to(self.device)
                print(f"{self.log_prefix} Loaded U-Net from checkpoint")

            # Load Text Encoders
            if self.train_text_encoder:
                if loaded_components.get('text_encoder') is not None:
                    if self.text_encoder is not None:
                        del self.text_encoder
                        gc.collect()
                        torch.cuda.empty_cache()
                    self.text_encoder = loaded_components['text_encoder']
                    print(f"{self.log_prefix} Loaded Text Encoder from checkpoint")

                if self.is_sdxl and loaded_components.get('text_encoder_2') is not None:
                    if self.text_encoder_2 is not None:
                        del self.text_encoder_2
                        gc.collect()
                        torch.cuda.empty_cache()
                    self.text_encoder_2 = loaded_components['text_encoder_2']
                    print(f"{self.log_prefix} Loaded Text Encoder 2 from checkpoint")

            # Load Image Encoder (if present)
            if loaded_components.get('image_encoder') is not None:
                if hasattr(self, 'image_encoder') and self.image_encoder is not None:
                    del self.image_encoder
                    gc.collect()
                    torch.cuda.empty_cache()
                self.image_encoder = loaded_components['image_encoder']
                print(f"{self.log_prefix} Loaded Image Encoder from checkpoint")

            print(f"{self.log_prefix} Loaded checkpoint from step {step}")
            return step

        else:
            # Diffusers directory format (legacy, for backward compatibility)
            checkpoint_dir = checkpoint_path_obj
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

            # Load metadata to get step number
            metadata_path = checkpoint_dir / "metadata.json"
            step = 0
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    step = metadata.get('step', 0)

            # Load U-Net
            if self.train_unet:
                unet_path = checkpoint_dir / "unet"
                if unet_path.exists() and self.unet is not None:
                    from diffusers import UNet2DConditionModel
                    loaded_unet = UNet2DConditionModel.from_pretrained(unet_path)
                    self.unet.load_state_dict(loaded_unet.state_dict())
                    print(f"{self.log_prefix} Loaded U-Net from {unet_path}")

            # Load Text Encoders
            if self.train_text_encoder:
                te1_path = checkpoint_dir / "text_encoder"
                if te1_path.exists() and self.text_encoder is not None:
                    from transformers import CLIPTextModel
                    loaded_te1 = CLIPTextModel.from_pretrained(te1_path)
                    self.text_encoder.load_state_dict(loaded_te1.state_dict())
                    print(f"{self.log_prefix} Loaded Text Encoder 1 from {te1_path}")

                if self.is_sdxl:
                    te2_path = checkpoint_dir / "text_encoder_2"
                    if te2_path.exists() and self.text_encoder_2 is not None:
                        from transformers import CLIPTextModelWithProjection
                        loaded_te2 = CLIPTextModelWithProjection.from_pretrained(te2_path)
                        self.text_encoder_2.load_state_dict(loaded_te2.state_dict())
                        print(f"{self.log_prefix} Loaded Text Encoder 2 from {te2_path}")

            print(f"{self.log_prefix} Loaded checkpoint from step {step}")
            return step
