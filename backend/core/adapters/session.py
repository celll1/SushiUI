"""Atomic runtime adapter session: one adapter lifetime, implemented once.

Eleven backends hand-wrote resolve/parse/validate/wrap/account/refuse/restore.
The weakref-keyed original-module bookkeeping was written eleven times and the
model-reload splice it prevents was found on eight of them. An architecture
supplies target enumeration and branch construction; the rest is here.

Atomic: every file is parsed and planned against the live tree before a slot is
mutated, and install rolls back, so a request whose second file is bad leaves
the model as it found it rather than half-applied.

Reporting without importing ``api`` (``adapter_layering_test`` gates that): the
backend passes a ``warn(message, code)`` callback, and every refusal is an
``AdapterRefusal`` carrying its ``code`` as data, so a 400 can read
``exc.code`` without the warning channel.

Per-architecture hooks: how a missing file is refused, what one file's keys
mean (``prepare_file``, before accounting), how many branches it declares to
this pass, and what zero targets means. ``parse()`` is split from install for
an architecture whose install is split in time (MiniT2I).

Messages carry a basename, never a path: a warning is written into a PNG text
chunk, and ``PermissionError.__str__`` would put the path back.
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

from .capability import (ADAPTER_PAIRS, BLOCK_SWAP_REFUSAL_CODE,
                         BLOCK_SWAP_WARNING_CODE, ORDINARY_LORA,
                         adapter_refusal_reason, block_swap_refusal_reason,
                         block_swap_strands_branches, block_swap_warning_text)
from .codec import CodecRegistry, CodecSpec
from .layers import (CompositeAdapterLayer, DoRALinearLayer,
                     branch_survives_block_swap, get_module_slot,
                     set_module_slot, weight_decompose_refusal)
from .spec import (ALGORITHM_UNKNOWN, FORMAT_PEFT, FORMAT_UNKNOWN,
                   AdapterSpec)

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


UNPREPARED = _Sentinel("UNPREPARED")
"""``AdapterFile.prepared`` before the architecture's ``prepare_file`` has run.

Distinct from ``None``, which is what a session with no such hook prepares.
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
    apply_to_unet: bool = True
    apply_to_text_encoder: bool = True
    unet_layer_weights: Dict[str, float] = field(default_factory=dict)
    step_range: Optional[Sequence[int]] = None
    codec: Optional[CodecSpec] = None
    # Whatever ``prepare_file`` returned, memoised HERE rather than in the
    # session: a file handed to two passes must be parsed once.
    prepared: Any = UNPREPARED


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

    @property
    def prepared(self) -> Any:
        """This file's ``prepare_file`` result, already computed."""
        return self.file.prepared


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
    kind: Optional[str] = None
    #: "is a block offloader ALREADY splitting this module's blocks?" -- the
    #: half of the block-swap policy only the backend can answer. Declared only
    #: by an architecture whose offloader is built before adapters are
    #: installed; see ``BLOCK_SWAP_ADAPTER_ORDER``.
    block_swap_active: Optional[Callable[[], bool]] = None

    @property
    def component_kind(self) -> str:
        if self.kind is not None:
            return self.kind
        name = self.name.lower()
        if any(token in name for token in ("text_encoder", "te", "clip", "qwen", "prompt")):
            return "text_encoder"
        return "unet"


@dataclass
class _OwnedBranch:
    parent: weakref.ref
    composite: weakref.ref
    slot: Slot
    module_path: str
    branch_name: str
    file: AdapterFile


class _ComponentState:
    """Weakref-keyed bookkeeping for one component.

    ``id()`` is unsafe here: a freed model's id is REUSABLE, and a reload that
    allocates at the dead model's address is exactly the case the key must
    survive.
    """

    __slots__ = ("ref", "originals", "wrapped", "owned")

    def __init__(self) -> None:
        self.ref: Optional[weakref.ref] = None
        self.originals: Dict[str, nn.Module] = {}
        self.wrapped: set = set()
        self.owned: List[_OwnedBranch] = []

    def reset(self, module: Optional[nn.Module]) -> None:
        self.originals.clear()
        self.wrapped.clear()
        self.owned.clear()
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


def _default_declared_branches(tensors: Mapping[str, torch.Tensor],
                               components: Tuple[str, ...]) -> int:
    """Complete factor GROUPS, not ``.lora_down.weight`` keys: a LoHa/LoKr file
    has none of the latter, would declare zero, and ``_account``'s
    partial-application refusal would be inert for it.

    Complete ones ONLY, unlike every architecture's own counter: with no
    ``stem_of`` this cannot tell a foreign half key from a truncated one of its
    own, and over-declaring refuses a file that applied in full. An
    architecture that wants a truncated file refused passes its own counter
    over ``declared_groups`` -- all eleven do.
    """
    from .groups import group_adapter_tensors  # groups imports this module

    return len(group_adapter_tensors(tensors).groups)


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
        architecture: Optional[str] = None,
        label: str = "LoRA",
        message_label: Optional[str] = None,
        log: Callable[[str], None] = print,
        count_declared_branches: Callable[
            [Mapping[str, torch.Tensor], Tuple[str, ...]], int] =
        _default_declared_branches,
        missing_file: Optional[
            Callable[[str, Any], Optional[BaseException]]] = None,
        prepare_file: Optional[Callable[["AdapterFile"], Any]] = None,
        describe_zero_targets: Optional[
            Callable[["AdapterFile", "ApplyCounts"],
                     Union[str, BaseException, None]]] = None,
        canonicalize_foreign_keys: bool = False,
    ):
        """``label`` prefixes the console, ``message_label`` names the adapter
        to the user; separate because one architecture spells itself
        differently and a gate pins that text.

        ``missing_file(name, raw_path)`` returns the refusal for an
        unresolvable path, or ``None`` to skip it (Anima reports every miss
        before refusing). ``prepare_file(file)`` runs once per file, before
        planning and so before any refusal. ``count_declared_branches`` is
        asked per load, so a one-component pass declares only its own pairs.
        ``describe_zero_targets`` returns refusal text, an exception, or
        ``None`` when this pass simply covers none of this file.

        ``architecture`` is the ``ARCH_REGISTRY`` name; it keys
        ``core.adapters.capability`` and unset enables nothing.
        ``canonicalize_foreign_keys`` rewrites Diffusers/PEFT spellings before
        this architecture parses -- off by default, see ``_canonicalize``.
        """
        self._resolve_path = resolve_path
        self._warn_callback = warn
        self._architecture = architecture
        self._label = label
        self._message_label = message_label or label
        self._log = log
        self._count_declared_branches = count_declared_branches
        self._missing_file = missing_file
        self._prepare_file = prepare_file
        self._describe_zero_targets = describe_zero_targets
        self._canonicalize_foreign_keys = canonicalize_foreign_keys
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
                self._restore_state(state)
                state.reset(None)
        elif current is not module:
            self._restore_state(state)
            state.reset(module)
        return state

    def _restore_state(self, state: _ComponentState) -> int:
        """Remove only branches installed by this session."""
        restored = 0
        for owned in reversed(state.owned):
            parent = owned.parent()
            composite = owned.composite()
            if parent is None or composite is None:
                continue
            try:
                current = get_module_slot(parent, owned.slot)
                if current is not composite or not composite.has_branch(owned.branch_name):
                    continue
                composite.remove_branch(owned.branch_name)
                if len(composite) == 0:
                    composite.detach(parent, owned.slot)
                    restored += 1
            except Exception as e:
                self._log(f"[{self._label}] unload failed at {owned.module_path} "
                          f"({type(e).__name__})")
        state.owned.clear()
        state.originals.clear()
        state.wrapped.clear()
        return restored

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

    @staticmethod
    def _refusal_text(error: BaseException) -> str:
        """An architecture's refusal may be any exception: ``AdapterRefusal``,
        a builtin tagged by ``api.error_handlers.with_error_code``, or an
        ``APIError`` whose status the route must keep."""
        return getattr(error, "message", None) or str(error)

    def _refuse(self, error: BaseException,
                default_code: str = AdapterIncompatible.code) -> BaseException:
        self.warn(self._refusal_text(error),
                  getattr(error, "code", None) or default_code)
        return error

    # -- parsing -----------------------------------------------------------

    def _canonicalize(self, tensors: Mapping[str, torch.Tensor],
                      metadata: Mapping[str, str]
                      ) -> Tuple[Dict[str, torch.Tensor], CodecSpec]:
        """``(the keys this architecture will see, the detected codec)``.

        The REWRITE is opt-in: most architectures parse ``lora_A``/``lora_B``
        themselves, and rewriting the suffix under them leaves their PEFT branch
        parsing nothing, so a valid file is refused. See
        ``docs/guides/LYCORIS_ADAPTER_DESIGN.md``.
        """
        try:
            codec = CodecRegistry.detect(tensors, metadata)
        except Exception as e:
            # Detection indexes shapes it has not validated -- a 1-D
            # `.lora_A.bias` from a `lora_bias=True` PEFT export raises
            # IndexError. A failed sniff must not turn the architecture's clean
            # refusal into an unhandled 500; `unknown` is not gated on, so the
            # file still reaches the architecture's own parser.
            self._log(f"[{self._label}] codec detection failed "
                      f"({type(e).__name__}); treating the file as unknown")
            return dict(tensors), CodecSpec(algorithm=ALGORITHM_UNKNOWN,
                                            weight_decompose=False,
                                            format=FORMAT_UNKNOWN,
                                            metadata=dict(metadata or {}))
        if self._canonicalize_foreign_keys and codec.format == FORMAT_PEFT:
            # `normalize_keys` maps onto whatever key it computes without a
            # collision guard, so a file carrying both `base_model.model.X` and
            # a bare `X` silently loses one tensor here.
            return CodecRegistry.normalize_keys(tensors, codec), codec
        return dict(tensors), codec

    def _parse(self, index: int,
               config: Mapping[str, Any]) -> Optional[AdapterFile]:
        """One file, read and described. ``None`` when the architecture's
        ``missing_file`` hook skipped it."""
        raw_path = config.get("path", "")
        name = os.path.basename(str(raw_path))
        strength = float(config.get("strength", 1.0))

        resolved = self._resolve_path(raw_path)
        if resolved is None:
            if self._missing_file is not None:
                error = self._missing_file(name, raw_path)
                if error is None:
                    return None
            else:
                error = AdapterFileMissing(
                    f"LoRA '{name}' was requested but no such file exists in the "
                    f"registered LoRA directories -- refusing to generate without it."
                )
            self._log(f"[{self._label}] ERROR: {self._refusal_text(error)}")
            raise self._refuse(error, AdapterFileMissing.code)

        self._log(f"[{self._label}] Loading LoRA {index + 1}: {raw_path} "
                  f"(strength={strength})")
        try:
            from safetensors import safe_open

            with safe_open(str(resolved), framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
                tensors = {key: f.get_tensor(key) for key in f.keys()}
        except Exception as e:
            raise self._load_failed(name, e) from e

        apply_to_unet = bool(config.get("apply_to_unet", True))
        apply_to_text_encoder = bool(config.get("apply_to_text_encoder", True))
        unet_layer_weights = dict(config.get("unet_layer_weights") or {})
        raw_range = config.get("step_range")
        step_range = tuple(int(x) for x in raw_range) if raw_range is not None else None

        tensors, codec = self._canonicalize(tensors, metadata)
        self._refuse_unsupported_algebra(name, codec)

        self._log(f"[{self._label}] Loaded {len(tensors)} tensors from {raw_path}")
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
            declared_branches=0,
            apply_to_unet=apply_to_unet,
            apply_to_text_encoder=apply_to_text_encoder,
            unet_layer_weights=unet_layer_weights,
            step_range=step_range,
            codec=codec,
        )

    def _refuse_unsupported_algebra(self, name: str, codec: CodecSpec) -> None:
        """Refuse a LoHa/LoKr/DoRA file HERE, before any slot is mutated.

        Two files are deliberately NOT judged. Ordinary LoRA is not validated
        at all: ``validate()`` refuses an algorithm whose rank it cannot see and
        the codec sniffs a rank from three key spellings only, so switching it
        on today would refuse working files. An UNRECOGNIZED algebra is left to
        the architecture's own zero-target refusal for the same reason -- a
        valid ``lora_bias=True`` PEFT export sniffs as ``unknown``, as does any
        file whose detection raised.
        """
        pair = (codec.algorithm, bool(codec.weight_decompose))
        if pair == ORDINARY_LORA or pair not in ADAPTER_PAIRS:
            return
        spec = AdapterSpec.from_codec(codec, architecture=self._architecture)
        try:
            spec.validate()
        except AdapterRefusal as error:
            self._log(f"[{self._label}] ERROR: {self._refusal_text(error)}")
            raise self._refuse(error)
        # The architecture the model IS, not the one the file claims to be.
        reason = adapter_refusal_reason(self._architecture, spec.algorithm,
                                        spec.weight_decompose)
        if reason is None:
            return
        error = AdapterIncompatible(f"{self._message_label} '{name}': {reason}")
        self._log(f"[{self._label}] ERROR: {error.message}")
        raise self._refuse(error)

    def _refuse_decomposed_over_quantized_base(
            self, file: AdapterFile,
            planned: Sequence["_PlannedInstall"]) -> None:
        """Refuse a weight-decomposed branch over a weight-only quantized base.

        The magnitude epilogue divides by ``||W_base + delta||`` and subtracts
        ``W_base``, so an int8/fp8 base would have to be dequantized every
        forward and the fused base GEMM abandoned -- a separate design with its
        own measurement (design doc, phase 3).

        Asked of the BUILT branch and of the base it actually holds, for the
        reason ``_refuse_stranded_branches`` is: detection gives metadata
        priority over keys, so a file of ``dora_scale`` tensors labelled
        ``networks.lora`` passes a label test and then installs a decomposed
        branch anyway. Planning mutates nothing, so this still precedes every
        install.
        """
        for item in planned:
            if not isinstance(item.branch, DoRALinearLayer):
                continue
            refusal = weight_decompose_refusal(item.branch.original_module)
            if refusal is None:
                continue
            error = AdapterIncompatible(
                f"{self._message_label} '{file.name}': {self._architecture or 'this build'} "
                f"cannot apply a weight-decomposed adapter at {item.module_path} -- "
                f"{refusal}")
            self._log(f"[{self._label}] ERROR: {error.message}")
            raise self._refuse(error)

    def _refuse_stranded_branches(self, file: AdapterFile,
                                  planned: Sequence["_PlannedInstall"]) -> None:
        """Refuse a branch a live block offloader could not carry.

        Asked of the BUILT branch, never of the file's detected algorithm:
        detection gives metadata priority over keys, so a ``hada_*`` file
        labelled ``networks.lora`` passes a label test. Planning mutates
        nothing, so this still precedes every install.
        """
        if not block_swap_strands_branches(self._architecture):
            return
        stranded = [item for item in planned
                    if not branch_survives_block_swap(item.branch)]
        if not stranded:
            return
        live = {item.component.name for item in stranded
                if item.component.block_swap_active is not None
                and item.component.block_swap_active()}
        if not live:
            return
        # "LoHaLinearLayer" -> "LoHa": the family the user recognises, taken
        # from the class actually built rather than from the file's label.
        reason = block_swap_refusal_reason(
            self._architecture,
            type(stranded[0].branch).__name__.replace("LinearLayer", "") or "adapter")
        error = AdapterIncompatible(f"{self._message_label} '{file.name}': {reason}",
                                    code=BLOCK_SWAP_REFUSAL_CODE)
        self._log(f"[{self._label}] ERROR: {error.message}")
        raise self._refuse(error, BLOCK_SWAP_REFUSAL_CODE)

    def warn_unoffloaded_branches(self, *component_names: str) -> int:
        """Advise that installed branches will not be offloaded with their block.

        The other ordering: an architecture whose offloader is built AFTER the
        adapters are installed sweeps their factors to the device and never
        returns them, so the numbers are right and only the saving is smaller.
        Called from the offloader build site, the first moment that is knowable
        there. Same object-level predicate as the refusal.
        """
        stranded = set()
        for name in (component_names or tuple(self._states)):
            for owned in self.state(name).owned:
                composite = owned.composite()
                if composite is None:
                    continue
                try:
                    branch = composite.get_branch(owned.branch_name)
                except (KeyError, ValueError):
                    continue
                if not branch_survives_block_swap(branch):
                    stranded.add(owned.module_path)
        if stranded:
            self.warn(block_swap_warning_text(self._architecture, len(stranded)),
                      BLOCK_SWAP_WARNING_CODE)
        return len(stranded)

    def _load_failed(self, name: str, error: Exception) -> AdapterLoadFailed:
        self._log(f"[{self._label}] ERROR: could not apply {name}: {error}")
        import traceback

        traceback.print_exc()
        message = (f"{self._message_label} '{name}' could not be applied "
                   f"({type(error).__name__}); see the server log for details")
        return self._refuse(AdapterLoadFailed(message))

    # -- planning ----------------------------------------------------------

    def _plan_file(self, file: AdapterFile,
                   components: Sequence[AdapterComponent]):
        planned: List[_PlannedInstall] = []
        counts = ApplyCounts()
        for component in components:
            applied = mismatched = 0
            if component.module is None:
                # Accounted anyway: "this pass covered that component and
                # matched nothing" is what tells a zero-target hook which pass
                # it is being asked about.
                counts.record(component.name, 0, 0)
                continue
            kind = component.component_kind
            if kind == "unet" and not file.apply_to_unet:
                counts.record(component.name, 0, 0)
                continue
            if kind == "text_encoder" and not file.apply_to_text_encoder:
                counts.record(component.name, 0, 0)
                continue
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

    def prepare(self, file: AdapterFile) -> Any:
        """The architecture's per-file parse, run once and before accounting.

        Memoised on the file rather than in the session, so the two passes of a
        split install share one result. Public because a split install needs the
        result before its first pass, to report and to take a verdict no single
        pass can take.
        """
        if file.prepared is UNPREPARED:
            file.prepared = (None if self._prepare_file is None
                             else self._prepare_file(file))
        return file.prepared

    def _zero_target_refusal(self, file: AdapterFile,
                             counts: ApplyCounts) -> Optional[BaseException]:
        """What a file that covered nothing means, or ``None`` for "not here".

        ``None`` is how a pass covering one component declines to judge a file
        that binds another: the verdict across components is not this pass's to
        take, and taking it refuses a file that is about to apply in full.
        """
        outcome: Union[str, BaseException, None] = None
        if self._describe_zero_targets is not None:
            outcome = self._describe_zero_targets(file, counts)
            if outcome is None:
                return None
        if isinstance(outcome, BaseException):
            return outcome
        return AdapterIncompatible(outcome or (
            f"LoRA '{file.name}': 0 of {file.declared_branches} down/up pairs "
            f"applied to the loaded model ({counts.mismatched} skipped on shape "
            f"mismatch) -- unrecognized key format or a different model."
        ))

    def _account(self, file: AdapterFile, counts: ApplyCounts) -> None:
        """The refusal and warning decisions, taken BEFORE anything is mutated."""
        self._log(f"[{self._label}] Applied LoRA to {counts.applied} modules")
        if counts.applied == 0:
            if not file.apply_to_unet and not file.apply_to_text_encoder:
                self.warn(f"LoRA '{file.name}' was disabled for both UNet and text encoder",
                          "lora_no_targets")
                return
            error = self._zero_target_refusal(file, counts)
            if error is None:
                return
            self._log(f"[{self._label}] ERROR: {self._refusal_text(error)}")
            raise self._refuse(error, AdapterIncompatible.code)

        if counts.mismatched or counts.applied < file.declared_branches:
            error = AdapterIncompatible(
                f"LoRA '{file.name}': only {counts.applied} of "
                f"{file.declared_branches} declared branches matched the loaded "
                f"model ({counts.mismatched} shape mismatch); refusing a partial "
                f"application.",
                code="lora_partial",
            )
            self._log(f"[{self._label}] ERROR: {error.message}")
            raise self._refuse(error)

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
                owned = _OwnedBranch(
                    parent=weakref.ref(item.parent),
                    composite=weakref.ref(composite),
                    slot=item.slot,
                    module_path=item.module_path,
                    branch_name=item.file.branch_name,
                    file=item.file,
                )
                state.owned.append(owned)
                done.append((item, composite, created, new_original, new_wrapped,
                             owned))
        except Exception as e:
            failed = plan[len(done)].file.name if len(done) < len(plan) else self._label
            self._rollback(done)
            raise self._load_failed(failed, e) from e

    def _rollback(self, done) -> None:
        for item, composite, created, new_original, new_wrapped, owned in reversed(done):
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
            try:
                state.owned.remove(owned)
            except ValueError:
                pass

    # -- public lifetime ---------------------------------------------------

    def parse(self, configs: Optional[Sequence[Mapping[str, Any]]]
              ) -> List[AdapterFile]:
        """Resolve and read every selected file, touching no model.

        For an architecture that installs in more than one pass, or that must see
        every file resolved before any is planned. ``load`` takes the result, so
        a file handed to two passes is read once and keeps one branch name.
        """
        return [file for index, config in enumerate(configs or ())
                if (file := self._parse(index, config)) is not None]

    def load(self,
             configs: Optional[Sequence[Union[Mapping[str, Any], AdapterFile]]],
             components: Sequence[AdapterComponent]) -> AdapterLoadResult:
        """Apply every selected adapter to every enabled component, atomically.

        ``configs`` items are request dicts, or the ``AdapterFile``s ``parse``
        already read.
        """
        components = list(components)
        # BEFORE the empty-config exit: a request that selects nothing is exactly
        # when a model swap goes unnoticed, and the next unload would then splice
        # the previous model's Linears into the new tree.
        for component in components:
            self.bind(component)

        result = AdapterLoadResult()
        if not configs:
            return result

        # Enabled, not merely loaded: a pass over a component that is not
        # loaded is still that pass, and must be accounted as covering it.
        enabled = [c for c in components if c.enabled]
        names = tuple(c.name for c in enabled)
        plan: List[_PlannedInstall] = []
        for index, config in enumerate(configs):
            if isinstance(config, AdapterFile):
                file = config
            elif (file := self._parse(index, config)) is None:
                continue
            file_enabled = [
                c for c in enabled
                if (c.component_kind != "unet" or file.apply_to_unet)
                and (c.component_kind != "text_encoder" or file.apply_to_text_encoder)
            ]
            file.declared_branches = self._count_declared_branches(
                file.tensors, tuple(c.name for c in file_enabled)
            )
            try:
                self.prepare(file)
                planned, counts = self._plan_file(file, enabled)
            except AdapterRefusal:
                raise
            except Exception as e:
                raise self._load_failed(file.name, e) from e
            self._refuse_decomposed_over_quantized_base(file, planned)
            self._refuse_stranded_branches(file, planned)
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
            if component.module is None or not state.owned:
                continue
            self._log(f"[{self._label}] Unloading {component.name} "
                      f"({len(state.wrapped)} wrapped module(s))...")
            restored += self._restore_state(state)
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

    @property
    def has_step_range(self) -> bool:
        """Whether any currently owned branch has a non-default step_range."""
        for state in self._states.values():
            for owned in state.owned:
                sr = owned.file.step_range
                if sr is not None and tuple(sr) != (0, 1000):
                    return True
        return False

    def set_step(self, current_step: int, total_steps: int) -> None:
        """Dynamically activate/deactivate branches based on step_range [0, 1000]."""
        if total_steps <= 0:
            return
        for state in self._states.values():
            for owned in state.owned:
                composite = owned.composite()
                if composite is None:
                    continue
                step_range = owned.file.step_range
                if step_range is not None and len(step_range) == 2:
                    start_step = int((step_range[0] / 1000) * total_steps)
                    end_step = int((step_range[1] / 1000) * total_steps)
                    is_active = (start_step <= current_step <= end_step)
                    if composite.has_branch(owned.branch_name):
                        composite.set_active(owned.branch_name, is_active)
