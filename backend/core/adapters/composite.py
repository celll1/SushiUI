"""Where an adapter wrapper sits in the module tree.

The composite that lets several branches share one base, the slot accessors
that install and restore it, and the predicates every INT8/FP8 gate and
offloader asks about a wrapped tree. Separate from ``layers.py``, which owns
the algebras: this file is about the tree, not the arithmetic.
"""

from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from .layers import (DoRALinearLayer, LoHaLinearLayer, LoKrLinearLayer,
                     LoRALinearLayer)

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
    """One wrapper per base module, holding an ordered set of named branches.

    ``LoRALinearLayer`` reads its base's ``in_features``/``out_features`` into
    locals, so it cannot wrap a wrapper -- which is why stacking used to be
    first-wins or a refusal. This owns the base once and puts branches beside
    each other, so adding or restrengthening one rewraps nothing.

    The name ends in ``Layer``, not ``Linear``: the offloaders in
    ``core.memory_management.block_offloading`` select by
    ``__class__.__name__.endswith("Linear")`` plus a non-None ``.weight``, and
    the ``.weight`` delegate below would enrol the base weight twice -- once
    here, once at ``<path>.original_module`` -- so a paired staging swap would
    restore the outgoing block's weights, silently.

    A branch is any module with ``forward_delta(x)`` returning its already
    scaled contribution, plus ``set_adapter_strength`` if its strength changes
    after installation; the composite never tests a branch's class. Saving and
    resuming go through the branch's own tensor protocol.

    One branch gives ``base(x) + delta`` in the same order as
    ``LoRALinearLayer.forward``, so bit-identical. Several sum the deltas
    first: two are order-independent exactly, three only up to associativity.
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
    "LoHaLinearLayer",
    "LoKrLinearLayer",
    "DoRALinearLayer",
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


def is_adapter_covered(module: Optional[nn.Module]) -> bool:
    """True for a slot an adapter already covers -- either wrapper class.

    The predicate the INJECTION sites want: "leave this alone / do not wrap it a
    second time / do not descend into it". ``is_adapter_wrapper`` above answers a
    different question for the INT8 gates -- it matches by class NAME and so
    deliberately excludes ``MiniMaxH3LoRALinearLayer``, whose slots those gates
    have never counted. Here the subclass must match, because MiniMax-H3's
    injector skips a slot it has already wrapped.
    """
    return isinstance(module, (LoRALinearLayer, LoHaLinearLayer, LoKrLinearLayer, DoRALinearLayer, CompositeAdapterLayer))


def named_modules_outside_adapters(
    root: nn.Module,
    prefix: str = "",
    memo: Optional[Set[nn.Module]] = None,
) -> Iterator[Tuple[str, nn.Module]]:
    """``named_modules()`` that yields an adapter-covered slot but not its inside.

    A wrapper's branches are ``nn.Linear`` children and its base is another, so a
    walk that selects Linears by class would otherwise offer the adapter's own
    ``lora_down``/``lora_up`` -- and the hidden base -- as fresh targets. Same
    order and same duplicate suppression as ``nn.Module.named_modules``, so on a
    tree that holds no adapter the two are indistinguishable.
    """
    if memo is None:
        memo = set()
    if root in memo:
        return
    memo.add(root)
    yield prefix, root
    if is_adapter_covered(root):
        return
    for name, child in root.named_children():
        yield from named_modules_outside_adapters(
            child, f"{prefix}.{name}" if prefix else name, memo)


def branch_survives_block_swap(branch: nn.Module) -> bool:
    """Whether a block offloader's name-based walk reaches every tensor ``branch``
    owns.

    The offloaders move ``module.weight`` for classes whose name ends in
    "Linear", so a LoRA branch's factors ride with their block while a
    LoHa/LoKr layer's bare parameters are left behind. Asked of the object, so
    a later algebra needs no table entry and a checkpoint whose metadata
    mislabels its algebra cannot bypass it.

    The wrapped base is excluded (its weight is the block's own), and a Linear
    ``bias`` is not moved by that walk either -- requiring it would refuse
    every LoRA over a biased base.
    """
    base = getattr(branch, "original_module", None)
    skip = {id(p) for p in base.parameters()} if isinstance(base, nn.Module) else set()
    carried = {id(m.weight) for m in branch.modules()
               if m.__class__.__name__.endswith("Linear")
               and getattr(m, "weight", None) is not None}
    return all(id(p) in carried for p in branch.parameters() if id(p) not in skip)


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
