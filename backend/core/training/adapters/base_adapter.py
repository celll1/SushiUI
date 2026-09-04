"""
Base adapter classes for model-specific training logic.

Author: Claude (2026-01-04)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Any, Mapping, Optional
import torch
import torch.nn as nn
from safetensors.torch import save_file

from core.adapters import LoRALinearLayer, count_quantized_linears
from core.adapters.capability import AXIS_TRAINING
from core.adapters.layers import (new_adapter_branch,
                                  validate_adapter_options)
from core.adapters.spec import (ALGORITHM_LORA, ALGORITHMS, FORMAT_SUSHIUI,
                                METADATA_ALGORITHM, METADATA_FORMAT,
                                METADATA_OPTIONS, METADATA_SCHEMA_VERSION,
                                METADATA_WEIGHT_DECOMPOSE,
                                ADAPTER_SCHEMA_VERSION)


LORA_COMPONENT_UNET = "unet"
LORA_COMPONENT_TEXT_ENCODER = "text_encoder"
LORA_COMPONENT_TEXT_ENCODER_1 = "text_encoder_1"
LORA_COMPONENT_TEXT_ENCODER_2 = "text_encoder_2"
LORA_COMPONENT_VISION_ENCODER = "vision_encoder"

# Param-group order. Every architecture that groups by component already emits
# its groups in this order, and the order is part of a resume's contract:
# `_configured_group_lrs` is written back index-for-index.
LORA_COMPONENT_ORDER = (
    LORA_COMPONENT_UNET,
    LORA_COMPONENT_TEXT_ENCODER,
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_TEXT_ENCODER_2,
    LORA_COMPONENT_VISION_ENCODER,
)

LORA_COMPONENTS = frozenset(LORA_COMPONENT_ORDER)


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


@dataclass(frozen=True)
class TrainingAdapterSpec:
    """WHICH algebra this run trains, normalized from the run's ``network`` block.

    A missing field is ordinary LoRA without weight decomposition, which is what
    every YAML written before this existed means.
    """

    algorithm: str = ALGORITHM_LORA
    weight_decompose: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        algorithm = str(self.algorithm or ALGORITHM_LORA).strip().lower()
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"adapter_algorithm={self.algorithm!r} is not one of "
                f"{list(ALGORITHMS)}")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "weight_decompose", bool(self.weight_decompose))
        object.__setattr__(self, "options",
                           validate_adapter_options(algorithm, self.options))
        if self.weight_decompose:
            raise ValueError(
                "weight_decompose is accepted but not implemented: DoRA/DoHa/"
                "DoKr are Phase 3 (docs/guides/LYCORIS_ADAPTER_DESIGN.md)")

    @property
    def is_ordinary_lora(self) -> bool:
        return self.algorithm == ALGORITHM_LORA and not self.weight_decompose

    def metadata(self) -> Dict[str, str]:
        """The ``sushi.adapter.*`` block, EMPTY for ordinary LoRA.

        Empty on purpose: an ordinary LoRA checkpoint stays byte-identical to
        the ones every architecture already writes and every reader already
        detects from its keys.
        """
        if self.is_ordinary_lora:
            return {}
        meta = {
            METADATA_SCHEMA_VERSION: str(ADAPTER_SCHEMA_VERSION),
            METADATA_ALGORITHM: self.algorithm,
            METADATA_WEIGHT_DECOMPOSE: "true" if self.weight_decompose else "false",
            METADATA_FORMAT: FORMAT_SUSHIUI,
        }
        if self.options:
            import json

            meta[METADATA_OPTIONS] = json.dumps(dict(self.options), sort_keys=True)
        return meta


def refuse_untrainable_algebra(spec: TrainingAdapterSpec, capability,
                               blocks_to_swap: int = 0) -> None:
    """Refuse an algebra this architecture cannot train, and block swap with it.

    ONE implementation, two entry points: ``train_runner``'s config preflight
    (before the model loads) and ``LoRATrainer._create_adapter`` (the backstop
    for a caller that skipped the preflight).
    """
    if spec.is_ordinary_lora:
        return
    capability.require(spec.algorithm, spec.weight_decompose, AXIS_TRAINING)
    if blocks_to_swap:
        # No offloader moves a bare parameter -- they select modules whose class
        # name ends in "Linear" -- and what that costs a training step is
        # unmeasured.
        raise ValueError(
            f"blocks_to_swap={blocks_to_swap} is not supported with "
            f"adapter_algorithm={spec.algorithm}: the block offloader moves "
            f"modules whose class name ends in 'Linear' and a {spec.algorithm} "
            f"branch owns bare parameters instead, so its factors are invisible "
            f"to the swap. Set blocks_to_swap=0, or train an ordinary LoRA.")


def resolve_training_adapter_spec(trainer) -> TrainingAdapterSpec:
    """The run's algebra, off the trainer's own attributes.

    Deliberately NOT falling back to ``trainer.config``, unlike
    ``resolve_scope_csv``: these arrive from the YAML's ``network`` block, which
    ``train_runner`` preflights before the model loads, and a second source in
    the ``train`` block would reach the layer without that check.
    """
    def pick(key, default):
        value = getattr(trainer, key, None)
        return default if value is None else value

    return TrainingAdapterSpec(
        algorithm=pick("adapter_algorithm", ALGORITHM_LORA),
        weight_decompose=pick("weight_decompose", False),
        options=pick("adapter_config", {}) or {},
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
        self._lora_components: Dict[str, str] = {}
        self.adapter_spec = resolve_training_adapter_spec(trainer)
        # A trainer that knows its architecture refuses an algebra that
        # architecture cannot round-trip; LoRATrainer._create_adapter asks the
        # same question earlier, where the message reaches the run's log.
        handler = getattr(trainer, "arch", None)
        capability = getattr(handler, "adapter_capability", None)
        if capability is not None and not self.adapter_spec.is_ordinary_lora:
            capability.require(self.adapter_spec.algorithm,
                               self.adapter_spec.weight_decompose,
                               AXIS_TRAINING)

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

    #: This architecture's ordinary-LoRA branch class. MiniMax-H3 overrides it
    #: (its forward runs without autocast, so the branch casts per call); the
    #: LyCORIS algebras have no per-architecture variant.
    LORA_LAYER_CLS = LoRALinearLayer

    def build_branch(self, original_module: nn.Module, lora_name: str,
                     dtype: Optional[torch.dtype] = None) -> nn.Module:
        """THE construction seam: every architecture wraps a Linear through here.

        What comes back is the algebra the run asked for, at the run's rank,
        alpha and dtype. Constructing a layer class directly instead pins the
        run to ordinary LoRA no matter what its config says.
        """
        return new_adapter_branch(
            self.adapter_spec.algorithm, original_module,
            rank=self.lora_rank, alpha=self.lora_alpha, name=lora_name,
            dtype=self.lora_dtype if dtype is None else dtype,
            weight_decompose=self.adapter_spec.weight_decompose,
            options=self.adapter_spec.options,
            lora_cls=self.LORA_LAYER_CLS)

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
        base = getattr(layer, "original_module", None)
        if isinstance(base, nn.Module):
            base.requires_grad_(False)
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

    def component_param_groups(
        self,
        lora_layers: Dict[str, nn.Module],
        lr_by_component: Dict[str, Callable[[], float]],
    ) -> List[Dict[str, Any]]:
        """One optimizer group per non-empty component, in ``LORA_COMPONENT_ORDER``.

        Buckets by the component recorded at injection, not by a prefix test on
        the layer name: the two agree for every architecture today, and the
        recorded component is the one that cannot drift from the injection.

        Each rate is a THUNK and is called only for a component that actually
        has parameters, so an architecture never has to resolve a rate for a
        component it did not inject. What the thunk does -- read an attribute,
        go through ``resolve_component_lr``, apply an architecture's LR factor
        -- stays with the architecture, because that genuinely differs.
        """
        buckets: Dict[str, List[nn.Parameter]] = {}
        for name, layer in lora_layers.items():
            component = self.lora_components.get(name, LORA_COMPONENT_UNET)
            buckets.setdefault(component, []).extend(layer.trainable_parameters())

        groups: List[Dict[str, Any]] = []
        for component in LORA_COMPONENT_ORDER:
            params = buckets.pop(component, None)
            if not params:
                continue
            resolve = lr_by_component.get(component)
            if resolve is None:
                raise KeyError(
                    f"{type(self).__name__} injected {len(params)} {component!r} LoRA "
                    f"parameter(s) but declares no learning rate for that component"
                )
            groups.append({"params": params, "lr": resolve()})
        leftover = sorted(c for c, p in buckets.items() if p)
        if leftover:
            raise ValueError(
                f"{type(self).__name__} recorded LoRA component(s) {leftover} that are "
                f"missing from LORA_COMPONENT_ORDER, so their parameters would not train"
            )
        return groups

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

    # Z-Image is the exception: its file carries alpha in safetensors metadata
    # only, and its generation loader reads it from there.
    CHECKPOINT_WRITES_ALPHA = True

    CHECKPOINT_LOG_FORMAT = (
        "[{adapter}] Saved LoRA checkpoint ({layers} layers) -> {path}"
    )

    @abstractmethod
    def checkpoint_metadata(
        self, lora_layers: Dict[str, nn.Module], step: int, epoch: int
    ) -> Dict[str, str]:
        """The safetensors ``__metadata__`` block for this architecture.

        Called before anything is written, so an architecture that refuses to
        save some layer set (SenseNova refuses an understanding-only LoRA)
        raises from here.
        """
        pass

    def export_state_dict(self, lora_layers: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
        """``<stem>.<branch tensor name>`` for every layer, plus per-layer alpha.

        The stem is the ``lora_layers`` key verbatim. It is not just a file key:
        for Z-Image it is also the trainer's in-memory layer identity and its
        resume key, so nothing here may reshape it.

        Alpha is a fresh tensor per key on purpose -- one shared tensor object
        under several keys is what safetensors rejects as shared memory.
        """
        state_dict: Dict[str, torch.Tensor] = {}
        for stem, layer in lora_layers.items():
            for name, tensor in layer.export_tensors().items():
                state_dict[f"{stem}.{name}"] = tensor
            if self.CHECKPOINT_WRITES_ALPHA:
                state_dict[f"{stem}.alpha"] = torch.tensor(
                    float(self.lora_alpha), dtype=torch.float32
                )
        return state_dict

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
        metadata = self.checkpoint_metadata(lora_layers, step, epoch)
        # The architecture's keys win a clash: they carry its own spelling of
        # model_type/lora_alpha, and the spec block adds only sushi.adapter.*.
        metadata = {**self.adapter_spec.metadata(), **metadata}
        save_file(self.export_state_dict(lora_layers), str(output_path), metadata=metadata)
        # Metadata fields are available to the log format (SenseNova names its
        # branch from lora_targets); the three below always win a name clash.
        fields = dict(metadata)
        fields.update(adapter=type(self).__name__, layers=len(lora_layers), path=output_path)
        print(self.CHECKPOINT_LOG_FORMAT.format(**fields))


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
