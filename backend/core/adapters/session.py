"""Atomic runtime adapter session: one lifetime, implemented once.

Eleven generation backends hand-wrote the same lifetime -- resolve, parse,
validate, wrap, account, refuse, restore -- and Phase 0 measured what that
costs. The weakref-keyed original-module bookkeeping was written eleven times
and the model-reload splice it exists to prevent was found on eight
architectures, plus three more that only the checked-in gates caught. This
module owns that lifetime; an architecture supplies target enumeration and
branch construction and nothing else.

WHAT IS ATOMIC, AND WHY IT IS NOT MERELY TIDY. Every file is resolved, parsed
and PLANNED against the live module tree before a single slot is mutated, so a
request whose second file is missing, corrupt or foreign leaves the model
exactly as it found it instead of running with the first file half-applied.
Installation itself rolls back to that state on any exception, so the invariant
holds even for a failure only the install phase can produce.

HOW IT REPORTS WITHOUT IMPORTING ``api``. Nothing under ``core.adapters`` may
import ``api`` (``backend/tests/adapter_layering_test.py`` runs a subprocess
probe and fails otherwise), yet a refusal has to reach the response and the PNG
metadata chunk. So the session takes a ``warn(message, code)`` CALLBACK that the
backend supplies -- the backend already owns the lazy ``api.generation_status``
import -- and every refusal is an ``AdapterRefusal`` carrying its ``code`` AS
DATA. The callback is the push half and the exception attribute is the pull
half; a caller that wants the code on a 400 response reads ``exc.code`` and
needs no warning channel at all.

MESSAGES CARRY A BASENAME AND AN EXCEPTION TYPE, NEVER A PATH. A warning is
written into a PNG text chunk and returned raw in the response's ``warnings[]``;
``PermissionError.__str__`` carries the absolute path the basename was there to
remove. The resolved path stays in ``AdapterFile.path`` for the console log.
"""

from __future__ import annotations

import contextlib
import os
import weakref
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple, Union)

import torch
import torch.nn as nn

from .layers import CompositeAdapterLayer, get_module_slot, set_module_slot

Slot = Union[str, int]

__all__ = [
    "SHAPE_MISMATCH",
    "AdapterComponent",
    "AdapterFile",
    "AdapterFileMissing",
    "AdapterIncompatible",
    "AdapterLoadFailed",
    "AdapterLoadResult",
    "AdapterRefusal",
    "AdapterSession",
    "ApplyCounts",
    "BranchRequest",
    "PreparedBranch",
]


class AdapterRefusal(Exception):
    """A request that must not reach denoising, with a machine-readable code.

    The concrete subclasses ALSO inherit the builtin exception each backend
    raised before this engine existed (``FileNotFoundError`` / ``RuntimeError``),
    so adopting the session changes no caller's ``except`` clause and no test's
    ``pytest.raises``.
    """

    code = "lora_incompatible"

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class AdapterFileMissing(AdapterRefusal, FileNotFoundError):
    code = "lora_not_found"


class AdapterLoadFailed(AdapterRefusal, RuntimeError):
    code = "lora_load_failed"


class AdapterIncompatible(AdapterRefusal, RuntimeError):
    code = "lora_incompatible"


class _Sentinel:
    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return self._name


SHAPE_MISMATCH = _Sentinel("SHAPE_MISMATCH")
"""``build_branch`` result for a target whose tensors are the wrong shape.

Distinct from ``None`` (the file carries nothing for this target): a mismatch is
counted, reported as ``lora_partial``, and leaves the slot bare -- assigning the
branch anyway only fails later, inside the denoise loop.
"""


@dataclass
class AdapterFile:
    """One parsed request item. ``name`` is the only spelling a message may use."""

    index: int
    name: str
    path: str
    strength: float
    config: Mapping[str, Any]
    tensors: Dict[str, torch.Tensor]
    metadata: Dict[str, str]
    branch_name: str
    declared_branches: int


@dataclass
class BranchRequest:
    """One (file, target) pair handed to the architecture's branch builder.

    ``base`` is already unwrapped: for a slot an earlier file of the same request
    will occupy, it is the module the composite owns, so a branch built here
    passes ``CompositeAdapterLayer.add_branch``'s stale-splice guard.
    """

    file: AdapterFile
    component: str
    parent: nn.Module
    slot: Slot
    module_path: str
    base: nn.Module
    current: nn.Module


class PreparedBranch(NamedTuple):
    """A built, not-yet-installed branch and the strength to install it at.

    ``strength=None`` keeps whatever scale the branch was built with. An
    architecture whose scale is not the branch's own ``alpha/rank``
    (MiniMax-H3's fused qkv stem) writes that pair onto the branch here and
    still passes a strength, because ``add_branch(strength=)`` is the only
    folding that stays bit-identical to the pre-session arithmetic.
    """

    branch: nn.Module
    strength: Optional[float] = None


@dataclass
class ApplyCounts:
    """Per-file accounting, split per component.

    The split is not decoration: FLUX.2 needs to know a file matched the text
    encoder and nothing else, because whether that is a refusal depends on which
    components the request enabled.
    """

    applied: int = 0
    mismatched: int = 0
    per_component: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def record(self, component: str, applied: int, mismatched: int) -> None:
        self.applied += applied
        self.mismatched += mismatched
        self.per_component[component] = (applied, mismatched)


@dataclass
class AdapterLoadResult:
    applied: int = 0
    files: List[Tuple[AdapterFile, ApplyCounts]] = field(default_factory=list)


@dataclass
class AdapterComponent:
    """One model object with its own adapter lifetime.

    Its own lifetime is the point: FLUX.2 tears its text encoder's composites
    down in every generation's ``finally`` while the transformer's survive, so
    one shared wrapped-set cannot express "unload that one, keep this one".
    ``module is None`` means not loaded -- the session drops that component's
    bookkeeping so a later load cannot inherit the previous model's modules.
    """

    name: str
    module: Optional[nn.Module]
    iter_targets: Callable[[nn.Module], Iterable[Tuple[nn.Module, Slot, str]]]
    build_branch: Callable[[BranchRequest], Any]
    is_candidate: Optional[Callable[[nn.Module], bool]] = None
    enabled: bool = True


class _ComponentState:
    """Weakref-keyed bookkeeping for one component.

    ``id()`` is unsafe here: a freed model's id is REUSABLE, and a reload that
    allocates at the dead model's address is exactly the case the key must
    survive.
    """

    __slots__ = ("ref", "originals", "wrapped")

    def __init__(self) -> None:
        self.ref: Optional[weakref.ref] = None
        self.originals: Dict[str, nn.Module] = {}
        self.wrapped: set = set()

    def reset(self, module: Optional[nn.Module]) -> None:
        self.originals.clear()
        self.wrapped.clear()
        self.ref = weakref.ref(module) if module is not None else None


class _PlannedInstall(NamedTuple):
    file: AdapterFile
    component: AdapterComponent
    parent: nn.Module
    slot: Slot
    module_path: str
    base: nn.Module
    branch: nn.Module
    strength: Optional[float]


def _default_declared_branches(tensors: Mapping[str, torch.Tensor]) -> int:
    return sum(1 for key in tensors if key.endswith(".lora_down.weight"))


class AdapterSession:
    """Resolve, parse, validate and install a whole request, or none of it.

    One instance per backend object, reused across requests: it holds the
    per-component bookkeeping that must survive between them and reset when the
    model behind a component is replaced.
    """

    def __init__(
        self,
        *,
        resolve_path: Callable[[Any], Optional[str]],
        warn: Optional[Callable[[str, str], None]] = None,
        label: str = "LoRA",
        log: Callable[[str], None] = print,
        count_declared_branches: Callable[[Mapping[str, torch.Tensor]], int] =
        _default_declared_branches,
        describe_zero_targets: Optional[Callable[["AdapterFile", "ApplyCounts"], str]] = None,
    ):
        self._resolve_path = resolve_path
        self._warn_callback = warn
        self._label = label
        self._log = log
        self._count_declared_branches = count_declared_branches
        self._describe_zero_targets = describe_zero_targets
        self._states: Dict[str, _ComponentState] = {}

    # -- bookkeeping -------------------------------------------------------

    def state(self, name: str) -> _ComponentState:
        """The stored state for ``name``, WITHOUT the reload check.

        For a caller that must observe the maps as they are: a stale set is what
        makes the model-reload gate a test at all.
        """
        state = self._states.get(name)
        if state is None:
            state = self._states[name] = _ComponentState()
        return state

    def bind(self, component: AdapterComponent) -> _ComponentState:
        """The state for ``component``, reset if its module was replaced.

        Called on BOTH the load and the unload path, and BEFORE any empty-config
        early exit: the maps hold the OLD model's Linears, and restoring them
        splices them into the new one.
        """
        state = self.state(component.name)
        module = component.module
        current = state.ref() if state.ref is not None else None
        if module is None:
            if state.ref is not None or state.originals or state.wrapped:
                state.reset(None)
        elif current is not module:
            state.reset(module)
        return state

    # -- reporting ---------------------------------------------------------

    def warn(self, message: str, code: str) -> None:
        """Push a user-visible warning through the backend's callback.

        Guarded: a reporting failure must not replace the refusal that caused it.
        """
        if self._warn_callback is None:
            return
        try:
            self._warn_callback(message, code)
        except Exception as e:
            self._log(f"[{self._label}] warning channel failed ({type(e).__name__})")

    def _refuse(self, error: AdapterRefusal) -> AdapterRefusal:
        self.warn(error.message, error.code)
        return error

    # -- parsing -----------------------------------------------------------

    def _parse(self, index: int, config: Mapping[str, Any]) -> AdapterFile:
        raw_path = config.get("path", "")
        name = os.path.basename(str(raw_path))
        strength = float(config.get("strength", 1.0))

        resolved = self._resolve_path(raw_path)
        if resolved is None:
            message = (
                f"LoRA '{name}' was requested but no such file exists in the "
                f"registered LoRA directories -- refusing to generate without it."
            )
            self._log(f"[{self._label}] ERROR: {message}")
            raise self._refuse(AdapterFileMissing(message))

        self._log(f"[{self._label}] Loading LoRA {index + 1}: {raw_path} "
                  f"(strength={strength})")
        try:
            from safetensors import safe_open

            with safe_open(str(resolved), framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
                tensors = {key: f.get_tensor(key) for key in f.keys()}
        except Exception as e:
            raise self._load_failed(name, e) from e

        declared = self._count_declared_branches(tensors)
        self._log(f"[{self._label}] Loaded {len(tensors)} tensors "
                  f"({declared} down/up pairs) from {raw_path}")
        return AdapterFile(
            index=index,
            name=name,
            path=str(resolved),
            strength=strength,
            config=config,
            tensors=tensors,
            metadata=dict(metadata),
            # Unique within the request, so selecting the SAME file twice is two
            # branches rather than a duplicate-name refusal.
            branch_name=f"{index}:{name}",
            declared_branches=declared,
        )

    def _load_failed(self, name: str, error: Exception) -> AdapterLoadFailed:
        self._log(f"[{self._label}] ERROR: could not apply {name}: {error}")
        import traceback

        traceback.print_exc()
        message = (f"{self._label} '{name}' could not be applied "
                   f"({type(error).__name__}); see the server log for details")
        return self._refuse(AdapterLoadFailed(message))

    # -- planning ----------------------------------------------------------

    def _plan_file(self, file: AdapterFile,
                   components: Sequence[AdapterComponent]):
        planned: List[_PlannedInstall] = []
        counts = ApplyCounts()
        for component in components:
            applied = mismatched = 0
            # Materialised: the walk reads named_modules(), and the install phase
            # replaces slots underneath it.
            for parent, slot, module_path in list(component.iter_targets(component.module)):
                current = get_module_slot(parent, slot)
                if component.is_candidate is not None and not component.is_candidate(current):
                    continue
                base = (current.original_module
                        if isinstance(current, CompositeAdapterLayer) else current)
                outcome = component.build_branch(BranchRequest(
                    file=file, component=component.name, parent=parent, slot=slot,
                    module_path=module_path, base=base, current=current,
                ))
                if outcome is None:
                    continue
                if outcome is SHAPE_MISMATCH:
                    mismatched += 1
                    continue
                if isinstance(outcome, PreparedBranch):
                    branch, strength = outcome.branch, outcome.strength
                else:
                    branch, strength = outcome, file.strength
                planned.append(_PlannedInstall(
                    file=file, component=component, parent=parent, slot=slot,
                    module_path=module_path, base=base, branch=branch,
                    strength=strength))
                applied += 1
            counts.record(component.name, applied, mismatched)
        return planned, counts

    def _account(self, file: AdapterFile, counts: ApplyCounts) -> None:
        """The refusal and warning decisions, taken BEFORE anything is mutated."""
        self._log(f"[{self._label}] Applied LoRA to {counts.applied} modules")
        if counts.applied == 0:
            if self._describe_zero_targets is not None:
                message = self._describe_zero_targets(file, counts)
            else:
                message = (
                    f"LoRA '{file.name}': 0 of {file.declared_branches} down/up pairs "
                    f"applied to the loaded model ({counts.mismatched} skipped on shape "
                    f"mismatch) -- unrecognized key format or a different model."
                )
            self._log(f"[{self._label}] ERROR: {message}")
            raise self._refuse(AdapterIncompatible(message))

        if counts.mismatched or counts.applied < file.declared_branches:
            self.warn(
                f"LoRA '{file.name}': applied {counts.applied} of "
                f"{file.declared_branches} down/up pairs "
                f"({counts.mismatched} skipped on shape mismatch).",
                "lora_partial",
            )

    # -- installation ------------------------------------------------------

    def _install(self, plan: Sequence[_PlannedInstall]) -> None:
        """Install every planned branch, or leave the tree exactly as it was.

        The rollback is not tidiness: ``add_branch`` refuses a duplicate name and
        a branch built against a foreign base, and an architecture's builder may
        hand back something that is not a branch at all. Any of those part way
        through a 136-target plan would otherwise leave a model wrapped by half a
        request, with no record of which half.
        """
        done = []
        try:
            for item in plan:
                existing = get_module_slot(item.parent, item.slot)
                created = not isinstance(existing, CompositeAdapterLayer)
                composite = CompositeAdapterLayer.attach(item.parent, item.slot)
                try:
                    composite.add_branch(item.file.branch_name, item.branch,
                                         strength=item.strength)
                except Exception:
                    # attach() already mutated the slot; this pair has to undo
                    # itself, because it is not yet in ``done`` for the rollback
                    # below to reach.
                    if created and len(composite) == 0:
                        composite.detach(item.parent, item.slot)
                    raise
                state = self.state(item.component.name)
                new_original = item.module_path not in state.originals
                if new_original:
                    state.originals[item.module_path] = item.base
                new_wrapped = item.module_path not in state.wrapped
                state.wrapped.add(item.module_path)
                done.append((item, composite, created, new_original, new_wrapped))
        except Exception as e:
            failed = plan[len(done)].file.name if len(done) < len(plan) else self._label
            self._rollback(done)
            raise self._load_failed(failed, e) from e

    def _rollback(self, done) -> None:
        for item, composite, created, new_original, new_wrapped in reversed(done):
            state = self.state(item.component.name)
            try:
                composite.remove_branch(item.file.branch_name)
                if created and len(composite) == 0:
                    composite.detach(item.parent, item.slot)
            except Exception as e:
                self._log(f"[{self._label}] rollback failed at {item.module_path} "
                          f"({type(e).__name__})")
            if new_original:
                state.originals.pop(item.module_path, None)
            if new_wrapped:
                state.wrapped.discard(item.module_path)

    # -- public lifetime ---------------------------------------------------

    def load(self, configs: Optional[Sequence[Mapping[str, Any]]],
             components: Sequence[AdapterComponent]) -> AdapterLoadResult:
        """Apply every selected adapter to every enabled component, atomically."""
        components = list(components)
        # BEFORE the empty-config exit: a request that selects nothing is exactly
        # when a model swap goes unnoticed, and the next unload would then splice
        # the previous model's Linears into the new tree.
        for component in components:
            self.bind(component)

        result = AdapterLoadResult()
        if not configs:
            return result

        live = [c for c in components if c.enabled and c.module is not None]
        plan: List[_PlannedInstall] = []
        for index, config in enumerate(configs):
            file = self._parse(index, config)
            try:
                planned, counts = self._plan_file(file, live)
            except AdapterRefusal:
                raise
            except Exception as e:
                raise self._load_failed(file.name, e) from e
            self._account(file, counts)
            plan.extend(planned)
            result.files.append((file, counts))
            result.applied += counts.applied

        self._install(plan)
        return result

    def unload(self, components: Sequence[AdapterComponent]) -> int:
        """Restore what is ACTUALLY installed, not what the map remembers.

        Driving restore from map membership overcounts on an architecture whose
        originals map outlives the unload, and cannot see a wrapper installed
        through a path the map never recorded.
        """
        restored = 0
        for component in components:
            state = self.bind(component)
            if component.module is None or not state.wrapped:
                continue
            self._log(f"[{self._label}] Unloading {component.name} "
                      f"({len(state.wrapped)} wrapped module(s))...")
            for parent, slot, module_path in list(component.iter_targets(component.module)):
                current = get_module_slot(parent, slot)
                if not isinstance(current, CompositeAdapterLayer):
                    continue
                # pop, not get: a restore that raises part way must leave only
                # what it still owes.
                set_module_slot(parent, slot,
                                state.originals.pop(module_path, current.original_module))
                restored += 1
            state.wrapped.clear()
            self._log(f"[{self._label}] Unloaded {restored} module(s)")
        return restored

    @contextlib.contextmanager
    def activate(self, configs: Optional[Sequence[Mapping[str, Any]]],
                 components: Sequence[AdapterComponent]):
        """Generation-scoped adapters: installed on entry, restored in ``finally``.

        For a component whose wrappers must not outlive the request (FLUX.2's
        text encoder). Z-Image deliberately does NOT use this: its wrappers are
        persistent state that the NEXT request's gate clears, and the ordering
        contract of its INT8 conversion hook depends on that.
        """
        components = list(components)
        result = self.load(configs, components)
        try:
            yield result
        finally:
            self.unload(components)
