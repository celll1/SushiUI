"""
Base adapter classes for model-specific training logic.

Author: Claude (2026-01-04)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn


def count_quantized_linears(module: Optional[nn.Module]) -> int:
    """Number of ``Int8Linear`` / ``Fp8Linear`` modules under ``module``.

    Called from EVERY full-parameter adapter's ``prepare_models_for_training``
    / ``setup_trainable_parameters``, not just the three architectures whose
    loaders can currently produce these classes (Anima, Ideogram 4, Krea 2).
    ``Int8Linear`` / ``Fp8Linear`` hold ``weight`` (and ``weight_scale``) as
    buffers, not ``nn.Parameter``s, so they are invisible to both
    ``requires_grad_(True)`` and ``named_parameters()``. Detecting them is
    the first half of ``reject_quantized_base`` below.

    For an architecture whose loader never swaps in these classes, this is a
    guaranteed no-op (returns 0, ``reject_quantized_base`` returns without
    raising) -- it costs one cheap module scan and exists so the same silent
    failure cannot reappear unnoticed if that architecture later gains a
    weight-only quantized load path, the way Anima/Krea2/Ideogram4 already
    have.
    """
    if module is None:
        return 0
    try:
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    except Exception as e:
        print(f"[quantized-base-guard] weight-only quant classes unavailable "
              f"({e}); assuming an unquantized base")
        return 0
    return sum(1 for m in module.modules() if isinstance(m, (Int8Linear, Fp8Linear)))


def is_lora_wrappable_linear(module: Optional[nn.Module]) -> bool:
    """True for a module a LoRA can wrap: ``nn.Linear`` or EITHER weight-only
    quantized Linear (``Int8Linear`` / ``Fp8Linear``).

    THE reason this exists: ``Int8Linear`` and ``Fp8Linear`` are ``nn.Module``s,
    NOT ``nn.Linear`` subclasses. Every ``isinstance(x, nn.Linear)`` site that
    selects LoRA targets therefore skips every quantized layer SILENTLY -- no
    error, just a smaller ``applied`` count that looks like a LoRA which happens
    to touch fewer modules. Measured on Anima, where the naive predicate dropped
    75% of the intended targets.

    Deliberately does NOT include ``LoRALinearLayer``: the call sites this
    replaces use the predicate to decide whether to WRAP, and an already-wrapped
    module must not be wrapped twice. A caller that wants "wrappable or already
    wrapped" (re-application, target enumeration) tests for that class itself --
    ``core.models.krea2.krea2_lora._is_target`` is the example.
    """
    if module is None:
        return False
    if isinstance(module, nn.Linear):
        return True
    try:
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    except Exception:
        return False
    return isinstance(module, (Int8Linear, Fp8Linear))


def lora_branch_dtype(module: nn.Module,
                      default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """The dtype a LoRA branch attached to ``module`` should compute in.

    The base weight's own dtype when that is a real float, else ``default``. Both
    quantized bases take the default branch -- e4m3 by the "float8" test, int8
    because an integer dtype is not floating point at all -- which is also the
    dtype their own forward produces from a bf16 activation. Without this a
    caller that copies the LoRA weights with ``dtype=base.weight.dtype`` would
    cast them to int8 and quantize the adapter to 8 uniform levels, or to e4m3
    and lose most of its precision.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        return default
    dtype = weight.dtype
    if dtype.is_floating_point and "float8" not in str(dtype):
        return dtype
    return default


def reject_quantized_base(transformer: Optional[nn.Module], *, model_label: str) -> None:
    """Refuse full fine-tuning on a weight-only quantized DiT base.

    ``Int8Linear`` / ``Fp8Linear`` hold ``weight`` (and ``weight_scale``) as
    BUFFERS, not ``nn.Parameter``s, precisely so an inference path cannot
    accidentally build an optimizer state for a non-differentiable int8/fp8
    tensor. The consequence for training is that ``requires_grad_(True)`` is
    a no-op on them and ``named_parameters()`` never yields them, so a full
    fine-tune of a quantized checkpoint would silently train only the layers
    the quantized conversion skipped. Loss still falls, nothing errors, and
    the saved checkpoint reloads: the failure is invisible from the outside,
    which is why this is a hard refusal rather than a warning.

    LoRA is unaffected and deliberately still allowed: ``LoRALinearLayer``
    wraps the quantized module rather than differentiating through its
    weight, and only the adapter's own float parameters are trained.

    Conditional on the base actually being quantized: an unquantized bf16
    checkpoint of the same architecture trains fine and must not be rejected.

    Call this from BOTH ``prepare_models_for_training`` and
    ``setup_trainable_parameters`` — a caller that builds the optimizer
    without going through ``prepare_models_for_training`` first would
    otherwise still get the silently-truncated parameter list this guard
    exists to prevent.
    """
    n = count_quantized_linears(transformer)
    if not n:
        return
    raise NotImplementedError(
        f"{model_label} full fine-tuning requires a bf16 base transformer: this checkpoint is "
        f"weight-only quantized ({n} Int8Linear/Fp8Linear layer(s)), and those layers "
        f"store their weights as BUFFERS. They cannot receive gradients, so a full "
        f"fine-tune would silently train only the layers the quantization skipped "
        f"while reporting a normal, falling loss. Use LoRA on this checkpoint (that "
        f"path works: the adapter wraps the quantized Linears), or select an "
        f"unquantized bf16 {model_label} checkpoint for full fine-tuning."
    )


class BaseLoRAAdapter(ABC):
    """
    Abstract base class for model-specific LoRA adapters.

    Each model architecture (SD1.5, SDXL, Z-Image) implements this interface
    to provide model-specific LoRA injection, parameter collection, and
    checkpoint saving logic.
    """

    def __init__(self, trainer, lora_rank: int, lora_alpha: int, lora_dtype: torch.dtype = torch.float32):
        """
        Initialize adapter.

        Args:
            trainer: Parent trainer instance (BaseTrainer subclass)
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor = alpha / rank)
            lora_dtype: dtype for LoRA weights (can differ from main model dtype)
        """
        self.trainer = trainer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_rank
        self.lora_dtype = lora_dtype

    @abstractmethod
    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to U-Net/Transformer.

        Args:
            lora_layers: Dictionary to store LoRA layer references (key: name, value: LoRA module)

        Returns:
            Number of LoRA layers injected
        """
        pass

    @abstractmethod
    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to text encoder(s).

        Args:
            lora_layers: Dictionary to store LoRA layer references (key: name, value: LoRA module)

        Returns:
            Number of LoRA layers injected
        """
        pass

    @abstractmethod
    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters with per-component learning rates.

        Args:
            lora_layers: Dictionary of LoRA layers (key: name, value: LoRA module)

        Returns:
            List of parameter groups for optimizer (format: [{"params": [...], "lr": ...}, ...])
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        lora_layers: Dict[str, nn.Module],
        step: int,
        epoch: int,
        output_path: Path
    ):
        """
        Save LoRA checkpoint in model-specific format.

        Args:
            lora_layers: Dictionary of LoRA layers (key: name, value: LoRA module)
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint
        """
        pass


class BaseFullParameterAdapter(ABC):
    """
    Abstract base class for model-specific full parameter training adapters.

    Each model architecture (SD1.5, SDXL, Z-Image) implements this interface
    to provide model-specific parameter preparation, collection, and
    checkpoint saving logic.
    """

    def __init__(self, trainer):
        """
        Initialize adapter.

        Args:
            trainer: Parent trainer instance (BaseTrainer subclass)
        """
        self.trainer = trainer

    @abstractmethod
    def prepare_models_for_training(self):
        """
        Prepare models for full parameter training.

        This includes:
        - Setting requires_grad=True for trainable components
        - Freezing non-trainable components
        - Enabling gradient checkpointing
        """
        pass

    @abstractmethod
    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters with per-component learning rates.

        Returns:
            List of parameter groups for optimizer (format: [{"params": [...], "lr": ...}, ...])
        """
        pass

    @abstractmethod
    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """
        Save full parameter checkpoint in model-specific format.

        Args:
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint
        """
        pass
