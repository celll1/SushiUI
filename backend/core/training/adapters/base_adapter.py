"""
Base adapter classes for model-specific training logic.

Author: Claude (2026-01-04)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn


LORA_COMPONENT_UNET = "unet"
LORA_COMPONENT_TEXT_ENCODER = "text_encoder"
LORA_COMPONENT_TEXT_ENCODER_1 = "text_encoder_1"
LORA_COMPONENT_TEXT_ENCODER_2 = "text_encoder_2"
LORA_COMPONENT_VISION_ENCODER = "vision_encoder"

LORA_COMPONENTS = frozenset({
    LORA_COMPONENT_UNET,
    LORA_COMPONENT_TEXT_ENCODER,
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_TEXT_ENCODER_2,
    LORA_COMPONENT_VISION_ENCODER,
})


def resolve_component_lr(trainer, *attr_names: str, label: str = "component") -> float:
    """The first configured LR among ``attr_names``, else the run's ``learning_rate``.

    "Configured" means *not None*, so an explicit ``0.0`` is a rate and not
    "unset". Refuses rather than inventing one when nothing is resolvable.
    See ``adapters/MODEL_ADAPTER_DESIGN.md`` for the convention.
    """
    for name in attr_names:
        value = getattr(trainer, name, None)
        if value is not None:
            return float(value)
    base = getattr(trainer, "learning_rate", None)
    if base is None:
        raise ValueError(
            f"Cannot resolve a learning rate for {label}: none of "
            f"({', '.join(attr_names) or 'no component keys'}) is set on the trainer "
            f"and it has no learning_rate either."
        )
    return float(base)


def count_quantized_linears(module: Optional[nn.Module]) -> int:
    """Number of weight-only quantized Linear modules under ``module``.

    Called from EVERY full-parameter adapter's ``prepare_models_for_training``
    / ``setup_trainable_parameters``, not just the three architectures whose
    loaders can currently produce these classes (Anima, Ideogram 4, Krea 2).
    Quantized Linears hold ``weight`` and scale sidecars as
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
        from core.models.common.w4a8_linear import W4A8Linear
    except Exception as e:
        print(f"[quantized-base-guard] weight-only quant classes unavailable "
              f"({e}); assuming an unquantized base")
        return 0
    return sum(1 for m in module.modules() if isinstance(m, (Int8Linear, Fp8Linear, W4A8Linear)))


def is_lora_wrappable_linear(module: Optional[nn.Module]) -> bool:
    """True for a module a LoRA can wrap: ``nn.Linear`` or EITHER weight-only
    quantized Linear (``Int8Linear`` / ``Fp8Linear`` / ``W4A8Linear``).

    THE reason this exists: the quantized Linear classes are ``nn.Module``s,
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
        from core.models.common.w4a8_linear import W4A8Linear
    except Exception:
        return False
    return isinstance(module, (Int8Linear, Fp8Linear, W4A8Linear))


def lora_branch_dtype(module: nn.Module,
                      default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """The dtype a LoRA branch attached to ``module`` should compute in.

    The base weight's own dtype when that is a real float, else ``default``.
    Weight-only quantized bases take the default branch -- e4m3 by the "float8" test, int8
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

    Quantized Linears hold ``weight`` and their scale sidecars as
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
        f"weight-only quantized ({n} quantized Linear layer(s)), and those layers "
        f"store their weights as BUFFERS. They cannot receive gradients, so a full "
        f"fine-tune would silently train only the layers the quantization skipped "
        f"while reporting a normal, falling loss. Use LoRA on this checkpoint (that "
        f"path works: the adapter wraps the quantized Linears), or select an "
        f"unquantized bf16 {model_label} checkpoint for full fine-tuning."
    )


def warn_quantized_base_without_checkpointing(
    transformer: Optional[nn.Module],
    *,
    gradient_checkpointing: bool,
    log_prefix: str = "[Trainer]",
) -> Optional[str]:
    """Report the memory cost of a quantized base with checkpointing disabled.

    THE CONDITION. Weight-only quantized Linears materialise or decode a compute
    weight for their product. Autograd saves the operand for backward because
    ``grad_input = grad_output @ w``. For a bf16 ``nn.Linear`` that saved tensor
    is an ALIAS of the resident parameter and costs nothing; for a quantized
    Linear it is a fresh ``(out, in)`` allocation in the compute dtype, on top of
    the packed codes. This can retain a compute-dtype weight in addition to the
    quantized storage, unlike an unquantized parameter whose saved tensor aliases it.

    With gradient checkpointing ON, one unit is live at a time and the quantized
    base still uses less memory overall. With it OFF, every layer's temporary is
    live simultaneously, so the whole model materialises in the compute dtype on
    top of the codes and the quantized base uses MORE memory than the bf16 one it
    replaced.

    Numbers in the message are measured (synthetic, `torch.cuda.max_memory_allocated`)
    and derived (safetensors headers), recorded in
    ``core/training/INT8_W8A8_TRAINING_GATE.md (G4)`` together with the
    autograd fix that was built for this, measured, and REFUSED by that gate's
    pre-registered step-time ceiling.

    Returns the message (also printed to the training log, which is the channel a
    training run's output reaches the user through) or None when the condition
    does not hold. Also offered to ``api.generation_status.add_warning``
    best-effort, the same way ``int8_linear._report_int_mm_fallback`` does; that
    channel is per-process and a training subprocess has its own, so the printed
    line is the one that is guaranteed to arrive.
    """
    if gradient_checkpointing:
        return None
    n = count_quantized_linears(transformer)
    if not n:
        return None
    # Bytes per retained element = the compute dtype the quantized modules were
    # built with, read off a real module rather than assumed to be bf16.
    retained_bytes = 2
    for m in transformer.modules():
        dtype = getattr(m, "compute_dtype", None)
        if isinstance(dtype, torch.dtype):
            retained_bytes = torch.empty(0, dtype=dtype).element_size()
            break
    message = (
        f"gradient_checkpointing is disabled and the base transformer is weight-only "
        f"quantized ({n} quantized Linear layer(s)). Each quantized Linear hands its "
        f"decoded weight to autograd, which can retain it until backward: quantized "
        f"storage plus up to {retained_bytes} bytes per "
        f"element retained, against {retained_bytes} bytes plus 0 for an unquantized base. "
        f"With checkpointing disabled all {n} are retained at once. Measured on a 28-layer "
        f"2048x2048 int8 synthetic (the e4m3 arm's memory was not measured): peak 426.4 MiB "
        f"quantized vs 322.2 MiB bf16. Derived for a "
        f"Krea 2 transformer: 11.94 GiB of codes plus 23.88 GiB of retained weights, "
        f"against a 23.88 GiB bf16 base. Enabling gradient_checkpointing keeps one block's "
        f"temporaries live at a time."
    )
    print(f"{log_prefix} WARNING: {message}")
    try:
        from api.generation_status import add_warning

        add_warning(message, code="quantized_base_no_checkpointing")
    except Exception:
        pass
    return message


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
        self._lora_components: Dict[str, str] = {}

    @property
    def lora_components(self) -> Dict[str, str]:
        """``lora_layers`` key -> component bucket, recorded at injection time.

        Grad-norm reporting used to infer the component from substrings of the
        LoRA key ('unet'/'transformer'/'te1_'...). Any architecture whose keys
        are plain module paths (SenseNova) or use another prefix (FLUX.2 /
        MiniT2I text encoders) silently fell through every branch and was
        counted only in the total. The injecting adapter knows the component, so
        it records it here instead.
        """
        comps = self.__dict__.get("_lora_components")
        if comps is None:
            comps = {}
            self.__dict__["_lora_components"] = comps
        return comps

    def register_lora_layer(
        self,
        lora_layers: Dict[str, nn.Module],
        name: str,
        layer: nn.Module,
        component: str,
    ) -> None:
        """Insert a LoRA layer into ``lora_layers`` and record its component."""
        if component not in LORA_COMPONENTS:
            raise ValueError(
                f"Unknown LoRA component {component!r} (expected one of {sorted(LORA_COMPONENTS)})"
            )
        lora_layers[name] = layer
        self.lora_components[name] = component

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

    def grad_norm_components(self) -> Dict[int, str]:
        """``id(param)`` -> ``LORA_COMPONENT_*`` for this adapter's parameters.

        The full-FT counterpart of ``BaseLoRAAdapter.lora_components``, and it
        exists for the same reason: the adapter knows which component a
        parameter belongs to, and every other way of deciding is a name test.
        ``_calculate_grad_norms`` buckets full-FT parameters by the MODULE it
        found them on (``unet`` / ``text_encoder`` / ``transformer_original``),
        which is right for every architecture that keeps one component per
        module and wrong for one that does not: SenseNova's two MoT halves are
        both inside ``transformer_original``, so the understanding half was
        reported as U-Net.

        Empty by default -- an adapter that does not override this keeps the
        module-derived bucketing exactly as it was. Called once per run and
        cached by the trainer.
        """
        return {}
