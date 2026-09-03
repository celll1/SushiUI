"""Adapter leaf layers: the wrappers that carry the trainable branch.

Two algebras live here already, and the eventual ``AdapterLayer`` protocol has
to accommodate both: the stock layer relies on an ambient ``torch.autocast`` to
reconcile its fp32 masters with a bf16 activation, the MiniMax-H3 subclass
casts per call because that architecture's forward runs without autocast. Both
are usable as branches of ``CompositeAdapterLayer``, which is what lets two
adapters share one base module.

Moved verbatim from ``core.training.adapters.{sd15,minimax_h3}_adapter``;
those modules now import these classes from here like everyone else.
"""

import math
from typing import Dict, Iterator, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinearLayer(nn.Module):
    """
    LoRA layer for Linear modules.

    Formula: output = original_output + (lora_up(lora_down(x))) * scale
    """

    def __init__(
        self,
        original_module: nn.Linear,
        rank: int,
        alpha: float,
        lora_name: str,
        lora_dtype: torch.dtype = torch.float32,
    ):
        """Initialize LoRA layer."""
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_name = lora_name
        self.lora_dtype = lora_dtype

        in_features = original_module.in_features
        out_features = original_module.out_features

        # Freeze original weights
        self.original_module.requires_grad_(False)

        # LoRA matrices (no bias)
        # Use lora_dtype for LoRA weights (can be different from main model dtype)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize: Kaiming uniform for down, zeros for up
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Move to same device as original, but use lora_dtype
        device = original_module.weight.device
        self.lora_down.to(device=device, dtype=lora_dtype)
        self.lora_up.to(device=device, dtype=lora_dtype)

    @property
    def weight(self):
        """Expose the wrapped Linear's weight so callers that introspect
        `.weight` (e.g. T5's DenseGatedActDense dtype check) keep working when a
        Linear is wrapped. Read-only delegate; not a trained parameter here."""
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Uses autocast to automatically handle mixed precision:
        - LoRA weights (fp32) are automatically converted to training dtype during forward
        - Gradients flow back to fp32 master weights correctly
        - GradScaler handles gradient scaling for fp16/bf16 training
        """
        org_out = self.original_module(x)

        # LoRA computation (autocast will handle dtype conversion automatically)
        # If we're in an autocast context (training_dtype), this will run in that dtype
        # Gradients will still flow back to fp32 master weights correctly
        lora_out = self.lora_up(self.lora_down(x))

        return org_out + lora_out * self.scale

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        """The branch contribution alone, so this layer can be a composite branch.

        Must stay bit-identical to the second term of ``forward``;
        ``adapter_composite_layer_cheap_test`` asserts that with ``atol=0.0``.
        """
        return self.lora_up(self.lora_down(x)) * self.scale

    def set_adapter_strength(self, strength: float) -> None:
        """Refold a request strength into the scale, exactly as the generation
        loaders do (``(alpha / rank) * strength``), so restrengthening an
        installed branch is not a rebuild."""
        self.scale = self.alpha / self.rank * strength

    # -- tensor protocol ---------------------------------------------------
    # The four methods below are what let a caller save, resume and optimise a
    # branch without naming ``lora_down``/``lora_up``. ``branch_tensors`` is the
    # single extension point: an algebra with a different tensor set overrides
    # that one and inherits the rest.

    def branch_tensors(self) -> Dict[str, torch.Tensor]:
        """Stem-relative name -> the LIVE weight, in checkpoint order.

        ``alpha`` is deliberately absent: it is a spec constant the saving
        adapter owns (Z-Image writes none at all), not a tensor this branch
        trains.
        """
        return {
            "lora_down.weight": self.lora_down.weight,
            "lora_up.weight": self.lora_up.weight,
        }

    def tensor_names(self) -> Tuple[str, ...]:
        """The names ``export_tensors`` produces and ``load_tensors`` consumes."""
        return tuple(self.branch_tensors())

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """The branch's own parameters. The wrapped base is frozen and excluded."""
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()

    def export_tensors(self) -> Dict[str, torch.Tensor]:
        """Detached CPU copies, ready to hand to ``save_file``."""
        return {name: weight.detach().cpu()
                for name, weight in self.branch_tensors().items()}

    def load_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Copy a stem-relative slice back in place. Names absent from the
        slice are left alone, which is the tolerance training resume has always
        had for a checkpoint that carries only some of a branch's tensors."""
        for name, weight in self.branch_tensors().items():
            value = tensors.get(name)
            if value is not None:
                weight.data.copy_(value)


class MiniMaxH3LoRALinearLayer(LoRALinearLayer):
    """``LoRALinearLayer`` with the LoRA branch cast to the ACTIVATION dtype.

    MiniMax-H3's training forward runs WITHOUT ``torch.autocast``: the vendored
    transformer owns its own mixed-precision policy (fp32 I/O heads and AdaLN
    projections, bf16 block stack, each activation aligned to its projection's
    parameter dtype), and an autocast context would override those casts and make
    training a different function from generation.

    The stock layer relies on autocast to reconcile its fp32 master weights with
    a bf16 activation. Without autocast that is not a style difference, it is a
    ``RuntimeError`` on the first ``F.linear`` -- and, if the branch happened to
    be built in the activation dtype instead, a silent loss of the fp32 master.
    So the masters stay fp32 and are cast per call; the gradient flows back
    through the cast to the fp32 parameters unchanged. This is exactly the LoRA
    shape Phase 0T measured (bitwise save->reload, 600/600 tensors receiving
    finite gradients).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_out = self.original_module(x)
        down = F.linear(x, self.lora_down.weight.to(x.dtype))
        up = F.linear(down, self.lora_up.weight.to(x.dtype))
        return org_out + up * self.scale

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        down = F.linear(x, self.lora_down.weight.to(x.dtype))
        up = F.linear(down, self.lora_up.weight.to(x.dtype))
        return up * self.scale


def get_module_slot(parent: nn.Module, slot: Union[str, int]) -> nn.Module:
    """Read a child module from an attribute name or a container index.

    Integer slots are not a corner case: Anima's adaln_modulation_* and
    llm_adapter targets, and several other architectures' MLPs, sit inside an
    ``nn.Sequential``, where the only address a target has is its index.
    """
    if isinstance(slot, int):
        return parent[slot]
    return getattr(parent, slot)


def set_module_slot(parent: nn.Module, slot: Union[str, int], module: nn.Module) -> None:
    """Write a child module to an attribute name or a container index."""
    if isinstance(slot, int):
        parent[slot] = module
    else:
        setattr(parent, slot, module)


class CompositeAdapterLayer(nn.Module):
    """One wrapper per base module, holding an ordered set of NAMED branches.

    THE DEFECT THIS FIXES. ``LoRALinearLayer.__init__`` reads
    ``original_module.in_features`` / ``out_features`` into locals and never
    exposes them, so it cannot wrap a wrapper: a second adapter over the same
    module raises ``AttributeError`` at construction, which is why every
    architecture is first-wins or an honest refusal today. This class owns the
    base ONCE and puts branches beside each other, so adding, removing,
    restrengthening or deactivating one rewraps nothing.

    THE NAME ENDS IN ``Layer``, NOT ``Linear``, deliberately. Every offloader in
    ``core.memory_management.block_offloading`` selects modules to move or swap
    by ``__class__.__name__.endswith("Linear")`` plus a non-None ``.weight``
    (``weighs_to_device``, ``_linear_weight_modules``, ``_build_weight_swap_jobs``).
    The ``.weight`` delegate below would then enrol the base weight a SECOND
    time -- once at this module's path, once at ``<path>.original_module`` --
    and a paired staging swap applied twice restores the outgoing block's
    weights with no error.

    BRANCH PROTOCOL. A branch is any ``nn.Module`` with
    ``forward_delta(x) -> Tensor`` returning its contribution ALONE, already
    scaled; ``set_adapter_strength(strength)`` is required only of a branch
    whose strength is changed after installation. ``LoRALinearLayer`` and
    ``MiniMaxH3LoRALinearLayer`` differ in their forward (ambient autocast
    versus a per-call activation cast) and both satisfy it, so the composite
    never tests a branch's class. Saving, resuming and optimising a branch go
    through the tensor protocol on the branch itself (``branch_tensors`` and
    friends); the composite does not aggregate those, because naming tensors
    across several branches in one file is the codec's job, not this class's.

    NUMERICS. With one active branch the output is ``base(x) + delta`` -- the
    same two operations in the same order as ``LoRALinearLayer.forward``, hence
    bit-identical. With several, the deltas are summed first and the base added
    once, so two branches are order-independent EXACTLY (fp addition commutes)
    and three or more only up to associativity.
    """

    def __init__(self, base_module: nn.Module):
        super().__init__()
        self.original_module = base_module
        self.branches = nn.ModuleList()
        self._names: List[str] = []
        self._active: Dict[str, bool] = {}
        self._strengths: Dict[str, Optional[float]] = {}

    @classmethod
    def attach(cls, parent: nn.Module, slot: Union[str, int]) -> "CompositeAdapterLayer":
        """The composite covering ``parent[slot]``, installing one if absent.

        Idempotent by design: a second adapter over an already-wrapped slot gets
        the SAME composite back and adds a branch to it.
        """
        existing = get_module_slot(parent, slot)
        if isinstance(existing, cls):
            return existing
        composite = cls(existing)
        set_module_slot(parent, slot, composite)
        return composite

    def detach(self, parent: nn.Module, slot: Union[str, int]) -> nn.Module:
        """Put the original base module back in its slot and return it."""
        current = get_module_slot(parent, slot)
        if current is not self:
            raise ValueError(
                f"slot {slot!r} holds {type(current).__name__}, not this composite; "
                f"refusing to overwrite it")
        base = self.original_module
        set_module_slot(parent, slot, base)
        return base

    @property
    def base_module(self) -> nn.Module:
        return self.original_module

    @property
    def in_features(self):
        return self.original_module.in_features

    @property
    def out_features(self):
        return self.original_module.out_features

    @property
    def weight(self):
        """Read-only delegate, same contract as ``LoRALinearLayer.weight``."""
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    @property
    def branch_names(self) -> Tuple[str, ...]:
        return tuple(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def has_branch(self, name: str) -> bool:
        return name in self._active

    def get_branch(self, name: str) -> nn.Module:
        return self.branches[self._index(name)]

    def _index(self, name: str) -> int:
        try:
            return self._names.index(name)
        except ValueError:
            raise KeyError(f"no branch named {name!r} on {type(self).__name__}") from None

    def add_branch(self, name: str, branch: nn.Module, *,
                   strength: Optional[float] = None,
                   active: bool = True) -> nn.Module:
        """Install ``branch`` under ``name``.

        ``strength=None`` keeps whatever scale the branch was built with, since
        every generation loader already folds the request strength into it.
        """
        if name in self._active:
            raise ValueError(f"branch {name!r} is already installed")
        if not callable(getattr(branch, "forward_delta", None)):
            raise TypeError(
                f"{type(branch).__name__} is not an adapter branch: it must define "
                f"forward_delta(x) returning its contribution alone")
        owned = getattr(branch, "original_module", None)
        if owned is not None and owned is not self.original_module:
            # A branch built against a different base is the stale-module splice
            # this engine exists to make impossible, not a merge to attempt.
            raise ValueError(
                f"branch {name!r} was built against a different base module "
                f"({type(owned).__name__}); rebuild it against this composite's base")
        self.branches.append(branch)
        self._names.append(name)
        self._active[name] = bool(active)
        self._strengths[name] = None
        if strength is not None:
            self.set_strength(name, strength)
        return branch

    def remove_branch(self, name: str) -> nn.Module:
        index = self._index(name)
        branch = self.branches[index]
        del self.branches[index]
        del self._names[index]
        del self._active[name]
        del self._strengths[name]
        return branch

    def clear_branches(self) -> None:
        for name in list(self._names):
            self.remove_branch(name)

    def set_strength(self, name: str, strength: float) -> None:
        branch = self.get_branch(name)
        setter = getattr(branch, "set_adapter_strength", None)
        if not callable(setter):
            raise TypeError(
                f"branch {name!r} ({type(branch).__name__}) cannot be restrengthened: "
                f"it defines no set_adapter_strength(strength). Applying the strength "
                f"outside the branch would change the arithmetic, not just the scale")
        setter(float(strength))
        self._strengths[name] = float(strength)

    def get_strength(self, name: str) -> Optional[float]:
        self._index(name)
        return self._strengths[name]

    def set_active(self, name: str, active: bool) -> None:
        self._index(name)
        self._active[name] = bool(active)

    def is_active(self, name: str) -> bool:
        self._index(name)
        return self._active[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_module(x)
        delta = None
        for name, branch in zip(self._names, self.branches):
            if not self._active[name]:
                continue
            contribution = branch.forward_delta(x)
            delta = contribution if delta is None else delta + contribution
        # No active branch returns the base output UNCHANGED rather than
        # base + 0, which would flip the sign of a -0.0 output.
        return out if delta is None else out + delta


_ADAPTER_WRAPPER_CLASS_NAMES = frozenset({
    "LoRALinearLayer",
    "CompositeAdapterLayer",
})


def is_adapter_wrapper(module: nn.Module) -> bool:
    """True for a module that HIDES a base Linear behind an adapter.

    Matched by class name so a duplicated class object (the same file reached by
    two import paths) cannot make the test silently false.

    ``MiniMaxH3LoRALinearLayer`` is deliberately absent: the INT8 gates that read
    this have never counted it, and adding it would create a refusal that
    architecture does not have today. When MiniMax-H3 adopts the composite its
    wrapper root becomes a ``CompositeAdapterLayer`` and is covered then.
    """
    return type(module).__name__ in _ADAPTER_WRAPPER_CLASS_NAMES


def count_adapter_wrapper_roots(model: nn.Module) -> int:
    """Number of base modules hidden behind an adapter wrapper under ``model``.

    Counts ROOTS: a composite's branches belong to one wrapped slot, not to
    several, so the walk does not descend into a match. For a tree of plain
    ``LoRALinearLayer`` wrappers -- which cannot nest, that being the defect the
    composite fixes -- this returns exactly what a flat class-name count did.
    """
    if is_adapter_wrapper(model):
        return 1
    return sum(count_adapter_wrapper_roots(child) for child in model.children())
