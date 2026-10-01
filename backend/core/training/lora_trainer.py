"""
LoRA (Low-Rank Adaptation) Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters. Which adapter
serves an architecture is declared on that architecture's ArchHandler
(``core/training/arch/``), not decided here.

Key improvements:
- Model-specific logic separated into adapters
- Clean separation of concerns
- Adding an architecture is one registry entry, not a branch here

References:
- sd-scripts (Apache-2 license) by kohya-ss
- ai-toolkit (MIT license) by ostris
- musubi-tuner (Apache-2 license) by kohya-ss (Z-Image support)

Author: Claude (2026-01-04)
"""

from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer


def require_trainable_algebra(trainer, arch) -> None:
    """The backstop for ``_create_adapter``: ``train_runner`` refuses the same
    thing from the config, before the model loads. A function, not a method, so
    it runs for anything that drives ``_create_adapter``."""
    from core.training.adapters.base_adapter import (
        refuse_untrainable_algebra, resolve_training_adapter_spec)

    config = getattr(trainer, "config", None) or {}
    refuse_untrainable_algebra(resolve_training_adapter_spec(trainer),
                               arch.adapter_capability,
                               getattr(trainer, "blocks_to_swap", 0) or 0,
                               config.get("fp8_base_dtype") or None)


def resolve_lora_resume_checkpoint(trainer, requested):
    """Resolve the same adapter artifact for load-time preflight and resume."""
    if not requested:
        return None
    if str(requested).lower() == "latest":
        candidates = list(trainer.output_dir.glob(f"{trainer.run_name}_step_*.safetensors"))
        from .base_trainer import _checkpoint_step_from_name
        candidates = [path for path in candidates
                      if path.name.removeprefix(f"{trainer.run_name}_step_")
                      .removesuffix(".safetensors").isdigit()
                      and _checkpoint_step_from_name(path.name) is not None]
        checkpoint = max(candidates, key=lambda path: _checkpoint_step_from_name(path.name)) if candidates else None
        if checkpoint is None:
            if list(trainer.output_dir.glob(f"{trainer.run_name}_step_*_state.json")):
                raise RuntimeError("LoRA resume state exists but no weight checkpoint remains")
            return None
    else:
        checkpoint = trainer.output_dir / requested
        if not checkpoint.is_file():
            checkpoint = Path(requested)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"LoRA resume checkpoint does not exist: {requested}")
    return checkpoint


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
        adapter_algorithm: str = 'lora',
        weight_decompose: bool = False,
        adapter_config: Optional[Dict] = None,
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
            adapter_algorithm: Adapter algebra ('lora', 'loha', 'lokr')
            weight_decompose: Weight decomposition (DoRA/DoHa/DoKr);
                enabled per architecture by TRAINABLE_ADAPTER_PAIRS
            adapter_config: Algebra-specific options (LoKr factor / decompose_both)
            train_unet: Whether to train U-Net/Transformer
            train_text_encoder: Whether to train Text Encoder(s)
            train_image_encoder: Whether to train Image Encoder (future support)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # LoRA-specific settings (set before super().__init__)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_rank
        self.adapter_algorithm = adapter_algorithm
        self.weight_decompose = weight_decompose
        self.adapter_config = dict(adapter_config or {})
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder
        self.train_image_encoder = train_image_encoder

        # LoRA modules storage
        self.lora_layers: Dict[str, nn.Module] = {}

        # Adapter checkpoints cannot be loaded as base-model checkpoints.
        self._manages_own_resume = True
        resume_request = kwargs.get("resume_from_checkpoint") or (
            kwargs.get("train_config") or {}).get("resume_from_checkpoint")
        kwargs["resume_from_checkpoint"] = resume_request

        super().__init__(**kwargs)

        # Convert lora_dtype string to torch.dtype (after super().__init__ to have access to get_torch_dtype)
        from .base_trainer import get_torch_dtype
        self.lora_dtype = get_torch_dtype(lora_dtype)

        # Override log prefix
        self.log_prefix = "[LoRA Trainer]"

        self._create_adapter()

        self._apply_lora()
        if not getattr(self, "is_yue2", False):
            self._load_resume_adapter(resume_request)

        # Block swap deferred until after LoRA wraps Linear modules — the
        # LayerOffloadConductor snapshots layer state_dicts at registration
        # time and post-wrap key changes would break the swap.
        # No-op when blocks_to_swap == 0.
        if hasattr(self, "setup_ltx2_wrapper"):
            self.setup_ltx2_wrapper()
        self.arch.setup_block_swap(self)
        self._setup_sensenova_phase_eviction()

        print(f"{self.log_prefix} Initialized (rank={self.lora_rank}, alpha={self.lora_alpha})")
        ve_status = getattr(self, '_train_vision_encoder', False)
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}, Image Encoder: {self.train_image_encoder}, Vision Encoder: {ve_status}")

    def _create_adapter(self):
        """Create the LoRA adapter the arch registry declares for this model.

        Which adapter class an architecture uses, and the scope arguments it
        takes beyond (trainer, rank, alpha, dtype), are declared on its
        ArchHandler (``lora_adapter_class`` / ``lora_adapter_kwargs``). Reading
        them off ``self.arch`` rather than re-testing ``is_<arch>`` here also
        means the adapter and the training ops can never resolve to different
        architectures.
        """
        from core.training.arch import get_arch_handler

        # BaseTrainer.__init__ binds self.arch once every is_<arch> flag is
        # final; resolving from the flags otherwise keeps _create_adapter
        # callable on a bare trainer. Both go through the same registry.
        arch = getattr(self, "arch", None) or get_arch_handler(self)
        require_trainable_algebra(self, arch)
        plan = arch.lora_adapter_plan(self)
        self.adapter = plan.build(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
        print(f"{self.log_prefix} Using {plan.adapter_cls.__name__}"
              f"{plan.log_detail} [{self.adapter.adapter_spec.algorithm}]")

    def train(self, *args, **kwargs):
        try:
            if getattr(self, "is_yue2", False):
                from core.training.ops.yue2_ops import train_abc_ar_loop
                try:
                    return train_abc_ar_loop(self, *args, **kwargs)
                finally:
                    writer = getattr(self, "writer", None)
                    if writer is not None:
                        writer.close()
            requested = kwargs.get("resume_from_checkpoint") or self.resume_from_checkpoint
            self._load_resume_adapter(requested)
            if requested and "resume_from_checkpoint" not in kwargs:
                kwargs["resume_from_checkpoint"] = requested
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

    def _load_resume_adapter(self, requested: Optional[str]) -> None:
        if not requested:
            return
        if self._loaded_checkpoint_path:
            return
        checkpoint = resolve_lora_resume_checkpoint(self, requested)
        if checkpoint is None:
            print(f"{self.log_prefix} Resume: no LoRA checkpoint found; starting fresh")
            return
        step = self.load_checkpoint(str(checkpoint))
        from .base_trainer import _checkpoint_step_from_name
        expected = _checkpoint_step_from_name(checkpoint.name)
        if expected is None or step != expected:
            raise ValueError(f"LoRA checkpoint step mismatch: filename={expected}, metadata={step}")
        self._loaded_checkpoint_path = str(checkpoint)
        print(f"{self.log_prefix} Resume: LoRA weights loaded from {checkpoint.name}")

    def _setup_sensenova_phase_eviction(self) -> None:
        if not self.is_sensenova:
            return
        from core.training.ops import sensenova_ops

        # Same reason as the full-FT trainer: refuse the shared window without
        # its split HERE, or the flag falls through the eviction gate below and
        # silently does nothing. (The split itself is full-fine-tune only, which
        # is why LoRA installs no four-phase context.)
        sensenova_ops.assert_shared_prefix_contract(self)
        if not self.sensenova_mot_phase_eviction:
            return
        from .sensenova_phase_eviction import install_training_phase_eviction

        install_training_phase_eviction(self)

    def _apply_lora(self):
        """Apply LoRA to U-Net/Transformer and Text Encoders using adapter."""
        print(f"{self.log_prefix} Applying LoRA layers...")

        adapter_only = (self.config.get("train_adapter") is True
                        and getattr(getattr(self, "arch", None), "name", None)
                        in {"qwen_image_21", "anima", "ming_image_01"})
        if self.train_unet or adapter_only:
            unet_count = self.adapter.apply_lora_to_unet(self.lora_layers)
            print(f"{self.log_prefix} Injected {unet_count} LoRA layers into U-Net/Transformer")
        else:
            print(f"{self.log_prefix} U-Net/Transformer LoRA skipped (train_unet=False)")

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

        print(f"{self.log_prefix} Loading LoRA checkpoint: {checkpoint_path}")

        # Extract metadata using safe_open
        step = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if 'step' in metadata:
                step = int(metadata['step'])

        if getattr(self, "is_yue2", False):
            from core.training.ops.yue2_ops import YUE2_TRAINING_PROTOCOL_VERSION
            expected = {
                "model_type": "yue2",
                "yue2_objective": self.adapter.objective,
                "yue2_apply_stages": ",".join(self.adapter.apply_stages),
                "yue2_lora_half": self.adapter.half,
                "yue2_training_protocol": YUE2_TRAINING_PROTOCOL_VERSION,
            }
            identity = getattr(self, "yue2_model_identity", {}) or {}
            if identity.get("checkpoint"):
                expected["yue2_base_checkpoint"] = str(identity["checkpoint"])
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise ValueError(
                        f"YuE2 checkpoint metadata {field}={metadata.get(field)!r} "
                        f"does not match this run's {value!r}"
                    )

        if getattr(self, "is_ming_image_01", False):
            from core.models.ming_image_01.convrot_training import validate_resume_metadata
            validate_resume_metadata(self, metadata)
            expected = {"model_type": "ming_image_01",
                "ming_image_01_profile": self.ming_image_01_components.artifact.profile,
                "lora_rank": str(self.lora_rank), "lora_alpha": str(self.lora_alpha)}
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise ValueError(f"Ming LoRA resume {field} does not match this run")

        configure_topology = None
        if getattr(self, "is_qwen_image_21", False):
            from core.models.qwen_image_21.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_ming_image_01", False):
            from core.models.ming_image_01.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_krea2", False):
            from core.models.krea2.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_anima", False):
            from core.models.anima.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_zimage", False):
            from core.models.zimage_residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_flux2", False):
            from core.models.flux2.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_lens", False):
            from core.models.lens.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_minit2i", False):
            from core.models.minit2i.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_ideogram4", False):
            from functools import partial
            from core.models.ideogram4.residual_unet import configure_training_topology
            if "train_uncond" in metadata and metadata["train_uncond"] != str(bool(getattr(self,"ideogram4_train_uncond",False))):
                raise ValueError("Ideogram 4 resume train_uncond does not match this run")
            if getattr(self,"ideogram4_train_uncond",False) and getattr(self,"transformer_uncond",None) is None:
                raise ValueError("Ideogram 4 train_uncond requires a loaded unconditional transformer")
            configure_topology = partial(configure_training_topology,
                unconditional_transformer=getattr(self,"transformer_uncond",None))
        elif getattr(self, "is_sensenova", False):
            from core.models.sensenova.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_acestep", False):
            from core.models.acestep.residual_unet import configure_training_topology
            configure_topology = configure_training_topology
        elif getattr(self, "is_ltx2", False):
            from functools import partial
            from core.models.ltx2.residual_unet import configure_training_topology
            configure_topology = partial(configure_training_topology, trainer=self)
        elif getattr(self, "is_minimax_h3", False):
            from functools import partial
            from core.models.minimax_h3.residual_unet import configure_training_topology
            configure_topology = partial(configure_training_topology, trainer=self)
        if configure_topology is not None:
            from core.rewiring.spec import read_checkpoint_metadata
            resume_topology = read_checkpoint_metadata(metadata)
            configure_topology(self.transformer, self.config, resuming=True,
                               resume_spec=resume_topology, install=False)
        if getattr(self, "is_qwen_image_21", False):
            expected_forward = (
                "convrot_int8_bf16_backward_v1"
                if getattr(self, "qwen_convrot_training_forward", "dequant")
                in {"transient_bf16", "cached_bf16", "prefetch_bf16"}
                else ""
            )
            actual_forward = str(metadata.get("qwen_base_forward") or "")
            if expected_forward and actual_forward != expected_forward:
                raise ValueError(
                    "This Qwen-Image 2.1 run uses the ConvRot INT8 base forward, "
                    f"but the resume checkpoint declares {actual_forward or 'no base-forward contract'!r}. "
                    "Resume it with qwen_convrot_training_forward=dequant, or start "
                    "a new ConvRot-forward run."
                )
            if actual_forward and actual_forward != expected_forward:
                raise ValueError(
                    "The Qwen-Image 2.1 resume checkpoint requires the ConvRot "
                    "INT8 base forward, but this run is configured for dequant."
                )

        checkpoint = load_file(checkpoint_path)

        if getattr(self, "is_qwen_image_21", False):
            global_name = "qwen_partition_global_adapter"
            saved_global = (
                bool(metadata.get(global_name))
                or any(key.startswith(f"{global_name}.") for key in checkpoint)
            )
            current_global = self.lora_layers.get(global_name)
            if saved_global and current_global is None:
                raise ValueError(
                    "Qwen-Image 2.1 checkpoint contains partition global adapter "
                    "weights, but this run has the global adapter disabled"
                )
            if current_global is not None:
                for field, expected in (
                    ("qwen_partition_global_rank", current_global.rank),
                    ("qwen_partition_global_tokens", current_global.summary_tokens),
                ):
                    if field in metadata and metadata[field] != str(expected):
                        raise ValueError(
                            f"Qwen-Image 2.1 checkpoint {field}={metadata[field]!r} "
                            f"conflicts with this run's {expected}"
                        )

        for field in ("lora_rank", "lora_alpha"):
            expected = getattr(self, field, None)
            if field in metadata and expected is not None:
                try:
                    actual = float(metadata[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"LoRA checkpoint metadata {field!r} is not numeric"
                    ) from exc
                if actual != float(expected):
                    raise ValueError(
                        f"LoRA checkpoint metadata {field}={actual:g} conflicts "
                        f"with the current run's {float(expected):g}"
                    )

        if getattr(self, "is_ming_image_01", False):
            import json
            from core.models.ming_image_01.partition_global import GLOBAL_NAME
            current=getattr(self.transformer,GLOBAL_NAME,None)
            actual=metadata.get(GLOBAL_NAME)
            if (current is None) != (actual is None):
                raise ValueError("Ming global adapter checkpoint ownership does not match this run")
            if current is not None:
                decoded=json.loads(actual);expected=current.metadata()
                if (not isinstance(decoded,dict) or set(decoded)!=set(expected)
                        or any(type(v) is not type(expected[k]) for k,v in decoded.items()) or decoded!=expected):
                    raise ValueError("Ming global adapter checkpoint metadata conflicts with this run")
            expected_keys = {f"{name}.{tensor}" for name, layer in self.lora_layers.items()
                             for tensor in (*layer.branch_tensors(), "alpha")}
            if set(checkpoint) != expected_keys:
                raise ValueError("Ming LoRA resume target scope does not match this run")

        prepared = []
        for lora_name, lora_layer in self.lora_layers.items():
            targets = lora_layer.branch_tensors()
            slice_ = {}
            for tensor_name, target in targets.items():
                key = f"{lora_name}.{tensor_name}"
                if key not in checkpoint:
                    raise ValueError(f"LoRA checkpoint is missing required tensor {key!r}")
                value = checkpoint[key]
                if value.shape != target.shape:
                    raise ValueError(
                        f"LoRA checkpoint tensor {key!r} has shape "
                        f"{tuple(value.shape)}, expected {tuple(target.shape)}"
                    )
                if not torch.can_cast(value.dtype, target.dtype):
                    raise ValueError(
                        f"LoRA checkpoint tensor {key!r} cannot convert from "
                        f"{value.dtype} to {target.dtype}"
                    )
                try:
                    slice_[tensor_name] = value.to(
                        device=target.device, dtype=target.dtype)
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"LoRA checkpoint tensor {key!r} cannot be prepared for "
                        f"{target.device}/{target.dtype}"
                    ) from exc
            prepared.append((lora_layer, targets, slice_))

        snapshots = [
            {name: tensor.detach().clone() for name, tensor in targets.items()}
            for _layer, targets, _slice in prepared
        ]
        try:
            for lora_layer, _targets, slice_ in prepared:
                lora_layer.load_tensors(slice_)
        except Exception:
            with torch.no_grad():
                for (layer, targets, _slice), snapshot in zip(prepared, snapshots):
                    # A spec constant (LoHa/LoKr alpha) is a freshly built value,
                    # not a live buffer: copying into it restores a throwaway and
                    # leaves the layer's scale where the failed load put it.
                    constants = layer.spec_constants()
                    for name, tensor in targets.items():
                        if name in constants:
                            layer.load_spec_constant(name, snapshot[name])
                        else:
                            tensor.copy_(snapshot[name])
            raise

        if configure_topology is not None:
            configure_topology(self.transformer, self.config, resuming=True,
                               resume_spec=resume_topology)
        print(f"{self.log_prefix} Loaded LoRA checkpoint from step {step}")
        return step
