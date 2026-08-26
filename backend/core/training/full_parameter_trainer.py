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

from typing import Dict, List

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
    SenseNovaFullParameterAdapter,
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
        # Full fine-tune settings (set before super().__init__), so the gates
        # that run during component loading can read them.
        # trains_base_weights is the channel ops/training_method resolves on: an
        # attribute costs no import (a real one would cycle back through
        # base_trainer) and survives a rename of this class.
        self.trains_base_weights = True
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

        self._setup_sensenova_phase_eviction()

        print(f"{self.log_prefix} Initialized")
        # Note: Vision Encoder training status is determined in train() after VE is loaded
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}, Image Encoder: {self.train_image_encoder}")

    def _setup_sensenova_phase_eviction(self) -> None:
        """Install the MoT evictor and, when armed, the four-phase graph cut.

        Built here rather than earlier for the same reason LoRATrainer builds it
        after injection: the selector reads the LIVE module tree, and the adapter
        has just replaced the trained half's ``Int8Linear``s with materialized
        ``nn.Linear``s.
        """
        if not self.is_sensenova:
            return
        from core.training.ops import sensenova_ops

        # BEFORE the eviction gate below: the shared window is legal only on top
        # of the split, and the configuration this refuses would otherwise fall
        # straight through that gate and leave the flag doing nothing at all.
        sensenova_ops.assert_shared_prefix_contract(self)
        if not self.sensenova_mot_phase_eviction:
            return
        from .sensenova_phase_eviction import install_training_phase_eviction

        install_training_phase_eviction(self)
        if self.sensenova_four_phase_eviction:
            from .sensenova_four_phase import install_four_phase_backward

            sensenova_ops.assert_four_phase_contract(self)
            context = install_four_phase_backward(self)
            print(f"{self.log_prefix} SenseNova four-phase eviction ENABLED"
                  + (f" (shared MNT prefix, {context.reduction} boundary gradient)"
                     if context.shared_window else ""))

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            four_phase = getattr(self, "sensenova_four_phase", None)
            if four_phase is not None:
                four_phase.discard()
                self.sensenova_four_phase = None
            evictor = getattr(self, "sensenova_phase_evictor", None)
            if evictor is not None:
                try:
                    evictor.teardown()
                except Exception as exc:
                    print(f"{self.log_prefix} WARNING: SenseNova eviction teardown failed: {exc}")
                finally:
                    self.sensenova_phase_evictor = None

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
            # FP8 base + Full FT is incompatible (a trained base needs
            # gradients); anima_ops.load_components has already skipped the flag
            # with a warning when train_unet is set. Nothing to do.
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
        elif self.is_sensenova:
            # Above the SD1.5 fallthrough because that fallthrough is not a
            # crash here, it is a silent zero: SenseNova keeps both MoT halves
            # inside self.transformer and sets self.unet and self.text_encoder
            # to None, and every group the SD1.5 adapter builds is gated on one
            # of those two being present. The run would collect no parameter at
            # all, after the loader had already dequantized a half for it.
            self.adapter = SenseNovaFullParameterAdapter(self)
            print(f"{self.log_prefix} Using SenseNovaFullParameterAdapter")
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
        """Not the resume path for full fine-tuning, on any architecture. Refuses.

        ``BaseTrainer.load_checkpoint`` is abstract, so this class must define
        something; what it defined was unreachable code with an import of a
        module that does not exist. The dead branches are gone and the refusal
        is loud, because the alternative -- leaving them -- is a resume that
        fails with ``ModuleNotFoundError: core.models.checkpoint_utils`` if
        anything ever calls it.

        WHAT ACTUALLY RESUMES A FULL FINE-TUNE: ``resume_from_checkpoint``, which
        ``BaseTrainer.__init__`` handles by reloading the checkpoint AS THE BASE
        MODEL (``_load_checkpoint_as_base``, with
        ``_try_load_checkpoint_with_fallback`` behind it) before the components
        are built. Every architecture goes through that path and none goes
        through this method: the checkpoint a full FT writes is a single
        arch-specific safetensors file, read by the same loader that reads any
        other base checkpoint. ControlNet and VAE training are the trainers that
        do call their own ``load_checkpoint``; those are different classes with
        their own implementations.

        The two branches removed here were:

        * a ``.safetensors`` branch importing
          ``core.models.checkpoint_utils.load_unified_checkpoint`` -- there is no
          ``checkpoint_utils`` module anywhere in this repository, so the branch
          could only ever have raised;
        * a diffusers-DIRECTORY branch (``UNet2DConditionModel.from_pretrained``
          on ``<dir>/unet``) for a layout no full-parameter adapter in this repo
          writes -- every one of them saves a single safetensors file.

        Reimplementing this would mean inventing a reader for eleven
        architectures' full-FT save formats with no caller and no consumer, so
        it refuses instead of guessing.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.load_checkpoint() is not the resume path for "
            f"full fine-tuning and is not implemented (it was dead code importing "
            f"a module, core.models.checkpoint_utils, that does not exist). To "
            f"resume a full fine-tune, set resume_from_checkpoint (a path, or "
            f"'latest'): BaseTrainer.__init__ loads that checkpoint as the base "
            f"model before building the components, which is how every "
            f"architecture's full-FT resume works. Requested: {checkpoint_path}"
        )
