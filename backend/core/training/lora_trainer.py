"""
LoRA (Low-Rank Adaptation) Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- SD15LoRAAdapter: SD1.5 models
- SDXLLoRAAdapter: SDXL models
- ZImageLoRAAdapter: Z-Image models
- FLUX2LoRAAdapter: FLUX.2 Klein models

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
import torch.nn as nn

from .base_trainer import BaseTrainer
from .adapters import (
    SD15LoRAAdapter,
    SDXLLoRAAdapter,
    ZImageLoRAAdapter,
    # DEUSLoRAAdapter,  # DEUS support removed
    FLUX2LoRAAdapter,
    AnimaLoRAAdapter,
    LensLoRAAdapter,
    Ideogram4LoRAAdapter,
    MiniT2ILoRAAdapter,
    Krea2LoRAAdapter,
    Ltx2LoRAAdapter,
    MiniMaxH3LoRAAdapter,
    AceStepLoRAAdapter,
    SenseNovaLoRAAdapter,
)


class LoRATrainer(BaseTrainer):
    """
    LoRA Trainer for SD/SDXL/Z-Image models.

    Uses model-specific adapters for LoRA injection, parameter collection,
    and checkpoint saving.
    """

    def __init__(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dtype: str = 'fp32',
        train_unet: bool = True,
        train_text_encoder: bool = False,
        train_image_encoder: bool = False,  # Image Encoder (future support)
        **kwargs
    ):
        """
        Initialize LoRA Trainer.

        Args:
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor = alpha / rank)
            lora_dtype: Data type for LoRA weights ('fp32', 'fp16', 'bf16')
            train_unet: Whether to train U-Net/Transformer
            train_text_encoder: Whether to train Text Encoder(s)
            train_image_encoder: Whether to train Image Encoder (future support)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # LoRA-specific settings (set before super().__init__)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_rank
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder
        self.train_image_encoder = train_image_encoder

        # LoRA modules storage
        self.lora_layers: Dict[str, nn.Module] = {}

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Convert lora_dtype string to torch.dtype (after super().__init__ to have access to get_torch_dtype)
        from .base_trainer import get_torch_dtype
        self.lora_dtype = get_torch_dtype(lora_dtype)

        # Override log prefix
        self.log_prefix = "[LoRA Trainer]"

        # Create model-specific adapter
        self._create_adapter()

        # Apply LoRA using adapter
        self._apply_lora()

        self._setup_sensenova_phase_eviction()

        # Block swap deferred until after LoRA wraps Linear modules — the
        # LayerOffloadConductor snapshots layer state_dicts at registration
        # time and post-wrap key changes would break the swap.
        # No-op when blocks_to_swap == 0.
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
        if getattr(self, "is_minimax_h3", False):
            # Block swap is NOT required for this arch (measured: 22.45 GB peak
            # at 384x640x22 and 25.63 GB at the largest registered cell, both
            # unswapped) but the knob exists, and the ordering contract is the
            # same as every other arch's: after the LoRA wrap, never before.
            self.arch.setup_block_swap(self)

        print(f"{self.log_prefix} Initialized (rank={self.lora_rank}, alpha={self.lora_alpha})")
        ve_status = getattr(self, '_train_vision_encoder', False)
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}, Image Encoder: {self.train_image_encoder}, Vision Encoder: {ve_status}")

    def _create_adapter(self):
        """Create model-specific LoRA adapter based on detected model type."""
        if getattr(self, "is_sensenova", False):
            self.adapter = SenseNovaLoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype
            )
            print(f"{self.log_prefix} Using SenseNovaLoRAAdapter")
        elif self.is_zimage:
            self.adapter = ZImageLoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using ZImageLoRAAdapter")
        # DEUS support removed - architecture no longer maintained
        # elif self.is_deus:
        #     self.adapter = DEUSLoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
        #     print(f"{self.log_prefix} Using DEUSLoRAAdapter")
        elif self.is_flux2:
            self.adapter = FLUX2LoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using FLUX2LoRAAdapter")
        elif self.is_lens:
            from core.models.lens.lens_lora import parse_scope_csv
            scope_csv = (getattr(self, "lens_lora_scope", "")
                          or self.config.get("lens_lora_scope", "")
                          or "img_attn,txt_attn,img_mlp,txt_mlp")
            # parse_scope_csv builds from an all-false scope, so unticking a
            # group in the panel actually removes it. The previous inline parse
            # started from DEFAULT_SCOPE and only set True, so a narrowing
            # selection was silently ignored. Same shape as every sibling arch.
            scope = parse_scope_csv(scope_csv)
            self.adapter = LensLoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope,
            )
            print(f"{self.log_prefix} Using LensLoRAAdapter (scope={scope})")
        elif self.is_ideogram4:
            from core.models.ideogram4.ideogram4_lora import parse_scope_csv
            scope_csv = (getattr(self, "ideogram4_lora_scope", "")
                          or self.config.get("ideogram4_lora_scope", "")
                          or "attn,mlp")
            scope = parse_scope_csv(scope_csv)
            self.adapter = Ideogram4LoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope,
            )
            print(f"{self.log_prefix} Using Ideogram4LoRAAdapter (scope={scope})")
        elif self.is_minit2i:
            from core.models.minit2i.minit2i_lora import parse_scope_csv, parse_te_scope_csv
            scope_csv = (getattr(self, "minit2i_lora_scope", "")
                          or self.config.get("minit2i_lora_scope", "")
                          or "attn,mlp,txt_embed")
            scope = parse_scope_csv(scope_csv)
            te_scope_csv = (getattr(self, "minit2i_te_lora_scope", "")
                             or self.config.get("minit2i_te_lora_scope", "")
                             or "attn,ff")
            te_scope = parse_te_scope_csv(te_scope_csv)
            self.adapter = MiniT2ILoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope, te_scope=te_scope,
            )
            print(f"{self.log_prefix} Using MiniT2ILoRAAdapter (scope={scope}, te_scope={te_scope})")
        elif self.is_krea2:
            from core.models.krea2.krea2_lora import parse_scope_csv
            scope_csv = (getattr(self, "krea2_lora_scope", "")
                          or self.config.get("krea2_lora_scope", "")
                          or "attn,mlp")
            scope = parse_scope_csv(scope_csv)
            self.adapter = Krea2LoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope,
            )
            print(f"{self.log_prefix} Using Krea2LoRAAdapter (scope={scope})")
        elif self.is_anima:
            # Parse scope from config; default to DEFAULT_TRAINING_SCOPE
            # (attention + mlp + llm_adapter, no AdaLN modulation).
            scope_csv = (getattr(self, "anima_lora_scope", "")
                          or self.config.get("anima_lora_scope", "")
                          or "attention,mlp,llm_adapter")
            wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
            # Allow train_llm_adapter to override the llm_adapter scope flag.
            if hasattr(self, "train_llm_adapter") or "train_llm_adapter" in self.config:
                wanted["llm_adapter"] = bool(
                    getattr(self, "train_llm_adapter",
                            self.config.get("train_llm_adapter", True))
                )
            scope = {
                "attention": wanted.get("attention", True),
                "mlp": wanted.get("mlp", True),
                "mod": wanted.get("mod", False),
                "llm_adapter": wanted.get("llm_adapter", True),
            }
            self.adapter = AnimaLoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope,
            )
            print(f"{self.log_prefix} Using AnimaLoRAAdapter (scope={scope})")
        elif self.is_ltx2:
            # Parse scope from config; default to attention-only (video LoRA).
            scope_csv = (getattr(self, "ltx2_lora_scope", "")
                          or self.config.get("ltx2_lora_scope", "")
                          or "attention")
            wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
            scope = {
                "attention": wanted.get("attention", True),
                "ff": wanted.get("ff", False),
                "audio": wanted.get("audio", False),
                "av_cross": wanted.get("av_cross", False),
            }
            self.adapter = Ltx2LoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope,
            )
            print(f"{self.log_prefix} Using Ltx2LoRAAdapter (scope={scope})")
        elif self.is_minimax_h3:
            # Scope from config; default attention+ff, which IS the design's
            # target set (300 modules / 83.1 M params at rank 16 across all 50
            # blocks). The I/O heads, the token refiner and AdaLN are excluded
            # permanently and are not reachable from any scope string -- see
            # adapters/minimax_h3_adapter.py for the reason per exclusion.
            from core.training.adapters.minimax_h3_adapter import parse_scope_csv
            scope_csv = (getattr(self, "minimax_h3_lora_scope", "")
                          or self.config.get("minimax_h3_lora_scope", "")
                          or "attention,ff")
            scope = parse_scope_csv(scope_csv)
            self.adapter = MiniMaxH3LoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope,
            )
            print(f"{self.log_prefix} Using MiniMaxH3LoRAAdapter (scope={scope})")
        elif self.is_acestep:
            # Parse scope from config; default to attention-only (audio LoRA).
            scope_csv = (getattr(self, "acestep_lora_scope", "")
                          or self.config.get("acestep_lora_scope", "")
                          or "attention")
            wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
            scope = {
                "attention": wanted.get("attention", True),
                "mlp": wanted.get("mlp", False),
            }
            self.adapter = AceStepLoRAAdapter(
                self, self.lora_rank, self.lora_alpha, self.lora_dtype, scope=scope,
            )
            print(f"{self.log_prefix} Using AceStepLoRAAdapter (scope={scope})")
        elif self.is_sdxl:
            self.adapter = SDXLLoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using SDXLLoRAAdapter")
        else:
            self.adapter = SD15LoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using SD15LoRAAdapter")

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            evictor = getattr(self, "sensenova_phase_evictor", None)
            if evictor is not None:
                try:
                    evictor.teardown()
                except Exception as exc:
                    print(f"{self.log_prefix} WARNING: SenseNova eviction teardown failed: {exc}")
                finally:
                    self.sensenova_phase_evictor = None

    def _setup_sensenova_phase_eviction(self) -> None:
        if not (self.is_sensenova and self.sensenova_mot_phase_eviction):
            return
        from .sensenova_phase_eviction import install_training_phase_eviction

        install_training_phase_eviction(self)

    def _apply_lora(self):
        """Apply LoRA to U-Net/Transformer and Text Encoders using adapter."""
        print(f"{self.log_prefix} Applying LoRA layers...")

        # Apply LoRA to U-Net/Transformer
        if self.train_unet:
            unet_count = self.adapter.apply_lora_to_unet(self.lora_layers)
            print(f"{self.log_prefix} Injected {unet_count} LoRA layers into U-Net/Transformer")
        else:
            print(f"{self.log_prefix} U-Net/Transformer LoRA skipped (train_unet=False)")

        # Apply LoRA to Text Encoder(s)
        if self.train_text_encoder:
            te_count = self.adapter.apply_lora_to_text_encoders(self.lora_layers)
            print(f"{self.log_prefix} Injected {te_count} LoRA layers into Text Encoder(s)")
        else:
            print(f"{self.log_prefix} Text Encoder LoRA skipped (train_text_encoder=False)")

        print(f"{self.log_prefix} Total LoRA layers: {len(self.lora_layers)}")

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters with per-component learning rates.

        Uses adapter to handle model-specific parameter grouping.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters(self.lora_layers)

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save LoRA checkpoint.

        Uses adapter to handle model-specific checkpoint format.

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}.safetensors"
        self.adapter.save_checkpoint(self.lora_layers, step, epoch, checkpoint_path)
        # Save Vision Encoder checkpoint separately (if loaded)
        self._save_vision_encoder_checkpoint(step, epoch)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load LoRA checkpoint for resuming training.

        Args:
            checkpoint_path: Path to LoRA checkpoint (.safetensors)

        Returns:
            Step number from checkpoint
        """
        from safetensors import safe_open
        from safetensors.torch import load_file
        import torch

        print(f"{self.log_prefix} Loading LoRA checkpoint: {checkpoint_path}")

        # Extract metadata using safe_open
        step = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata and 'step' in metadata:
                step = int(metadata['step'])

        # Load checkpoint weights
        checkpoint = load_file(checkpoint_path)

        # Load LoRA weights into existing layers
        for lora_name, lora_layer in self.lora_layers.items():
            # Load lora_down weight
            down_key = f"{lora_name}.lora_down.weight"
            if down_key in checkpoint:
                lora_layer.lora_down.weight.data.copy_(checkpoint[down_key])

            # Load lora_up weight
            up_key = f"{lora_name}.lora_up.weight"
            if up_key in checkpoint:
                lora_layer.lora_up.weight.data.copy_(checkpoint[up_key])

        print(f"{self.log_prefix} Loaded LoRA checkpoint from step {step}")
        return step
