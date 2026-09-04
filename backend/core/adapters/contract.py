"""What an architecture and the session exchange: the refusals, the two
sentinels, and the records describing a file, a branch and a component.

Separated from ``session.py`` so the lifetime and the vocabulary it speaks
are not one 1000-line file. Nothing here knows how a session runs.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .codec import CodecSpec

Slot = Union[str, int]


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
