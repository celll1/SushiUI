"""Normalized, versioned adapter description shared by training and generation.

``AdapterSpec`` keeps the TWO-AXIS form: ``algorithm`` (``lora``/``loha``/
``lokr``) and ``weight_decompose``, so DoRA stays a weight-decomposition
epilogue on three algorithms instead of becoming a fourth mutually exclusive
one. See ``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` sections 1 and 4.

The ``sushi.adapter.*`` metadata key names are defined HERE and nowhere else;
``codec.py`` imports them rather than repeating the literals, which is why the
dependency runs codec -> spec and ``from_codec`` duck-types its argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (TYPE_CHECKING, Any, Collection, Dict, Mapping, Optional,
                    Tuple)

if TYPE_CHECKING:  # typing only -- see the module docstring
    from .codec import CodecSpec

ADAPTER_SCHEMA_VERSION = 1

ALGORITHM_LORA = "lora"
ALGORITHM_LOHA = "loha"
ALGORITHM_LOKR = "lokr"
ALGORITHM_UNKNOWN = "unknown"
ALGORITHMS: Tuple[str, ...] = (ALGORITHM_LORA, ALGORITHM_LOHA, ALGORITHM_LOKR)

FORMAT_SUSHIUI = "sushiui_canonical"
FORMAT_LYCORIS = "lycoris_kohya"
FORMAT_PEFT = "diffusers_peft"
FORMAT_UNKNOWN = "unknown"
FORMATS: Tuple[str, ...] = (FORMAT_SUSHIUI, FORMAT_LYCORIS, FORMAT_PEFT)

METADATA_SCHEMA_VERSION = "sushi.adapter.schema_version"
METADATA_ALGORITHM = "sushi.adapter.algorithm"
METADATA_WEIGHT_DECOMPOSE = "sushi.adapter.weight_decompose"
METADATA_FORMAT = "sushi.adapter.format"
METADATA_OPTIONS = "sushi.adapter.options"
METADATA_ARCHITECTURE = "model_type"
METADATA_TARGET_SCOPE = "target_scope"
METADATA_RANK = "lora_rank"
METADATA_ALPHA = "lora_alpha"

OPTION_FACTOR = "factor"           # LoKr Kronecker factorization; -1 = auto
OPTION_USE_TUCKER = "use_tucker"   # LoHa hada_t1/hada_t2 present

#: Algorithms whose scale is ``alpha / rank``, so a rank is mandatory. LoKr's
#: full (unfactored) form is rank 0, which is why it is absent here.
RANK_REQUIRED: Tuple[str, ...] = (ALGORITHM_LORA, ALGORITHM_LOHA)

#: Mirrors ``core.training.arch.ARCH_REGISTRY``, which this package may not
#: import (``backend/tests/adapter_layering_test.py``). Same mirrored-and-pinned
#: pattern as ``api.arch_capabilities.CFG_NULL_STAGE_BY_ARCH``;
#: ``adapter_spec_targets_cheap_test.py`` asserts the two sets are equal.
KNOWN_ARCHITECTURES = frozenset({
    "sd15", "sdxl", "zimage", "anima", "lens", "ideogram4", "minit2i",
    "krea2", "flux2", "ltx2", "minimax_h3", "acestep", "sensenova",
})

_METADATA_TRUE = ("true", "1")

#: Kohya/LyCORIS write their algorithm under ``ss_*``, which this module does
#: not read -- see ``AdapterSpec.from_metadata``.
_FOREIGN_KEY_PREFIX = "ss_"

#: Internal display names; no UI surface presents these today.
FAMILY_NAMES: Mapping[Tuple[str, bool], str] = MappingProxyType({
    (ALGORITHM_LORA, False): "lora",
    (ALGORITHM_LOHA, False): "loha",
    (ALGORITHM_LOKR, False): "lokr",
    (ALGORITHM_LORA, True): "dora",
    (ALGORITHM_LOHA, True): "doha",
    (ALGORITHM_LOKR, True): "dokr",
})

__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "ALGORITHMS",
    "ALGORITHM_LOHA",
    "ALGORITHM_LOKR",
    "ALGORITHM_LORA",
    "ALGORITHM_UNKNOWN",
    "AdapterSpec",
    "FAMILY_NAMES",
    "FORMATS",
    "FORMAT_LYCORIS",
    "FORMAT_PEFT",
    "FORMAT_SUSHIUI",
    "FORMAT_UNKNOWN",
    "KNOWN_ARCHITECTURES",
    "METADATA_ALGORITHM",
    "METADATA_ALPHA",
    "METADATA_ARCHITECTURE",
    "METADATA_FORMAT",
    "METADATA_OPTIONS",
    "METADATA_RANK",
    "METADATA_SCHEMA_VERSION",
    "METADATA_TARGET_SCOPE",
    "METADATA_WEIGHT_DECOMPOSE",
    "OPTION_FACTOR",
    "OPTION_USE_TUCKER",
    "RANK_REQUIRED",
    "parse_metadata_bool",
]


def parse_metadata_bool(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _METADATA_TRUE


def _format_metadata_bool(value: bool) -> str:
    return "true" if value else "false"


def _refuse(message: str) -> None:
    # Deferred import: ``session`` -> ``codec`` -> this module would be a cycle.
    from .session import AdapterIncompatible

    raise AdapterIncompatible(message)


@dataclass(frozen=True)
class AdapterSpec:
    """One adapter's normalized description: algebra, geometry, scope, format.

    ``options`` carries the algorithm-specific settings that have no field of
    their own; the known keys have explicit accessors below so a caller never
    spells them inline.
    """

    algorithm: str = ALGORITHM_LORA
    weight_decompose: bool = False
    rank: Optional[int] = None
    alpha: Optional[float] = None
    architecture: Optional[str] = None
    components: Tuple[str, ...] = ()
    format: str = FORMAT_SUSHIUI
    schema_version: int = ADAPTER_SCHEMA_VERSION
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components or ()))
        object.__setattr__(self, "options",
                           MappingProxyType(dict(self.options or {})))

    # -- accessors ---------------------------------------------------------

    @property
    def family(self) -> str:
        """UI-facing name of the (algorithm, weight_decompose) pair."""
        return FAMILY_NAMES.get((self.algorithm, self.weight_decompose),
                                ALGORITHM_UNKNOWN)

    @property
    def lokr_factor(self) -> Optional[int]:
        value = self.options.get(OPTION_FACTOR)
        return None if value is None else int(value)

    @property
    def loha_use_tucker(self) -> bool:
        return bool(self.options.get(OPTION_USE_TUCKER, False))

    @property
    def scale(self) -> Optional[float]:
        if self.alpha is None or not self.rank:
            return None
        return self.alpha / self.rank

    # -- construction ------------------------------------------------------

    @classmethod
    def from_codec(
        cls,
        spec: "CodecSpec",
        *,
        architecture: Optional[str] = None,
        components: Tuple[str, ...] = (),
        options: Optional[Mapping[str, Any]] = None,
    ) -> "AdapterSpec":
        """Normalize a detection result. Detection itself stays in ``codec``."""
        metadata = dict(getattr(spec, "metadata", None) or {})
        merged: Dict[str, Any] = _options_from_metadata(metadata)
        merged.update(dict(options or {}))
        return cls(
            algorithm=spec.algorithm,
            weight_decompose=bool(spec.weight_decompose),
            rank=spec.rank,
            alpha=spec.alpha,
            architecture=architecture or metadata.get(METADATA_ARCHITECTURE) or None,
            components=components or _scope_from_metadata(metadata),
            format=spec.format,
            schema_version=_schema_version_from_metadata(metadata),
            options=merged,
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> "AdapterSpec":
        """Read a SushiUI metadata block; no ``sushi.adapter.*`` key means the
        legacy ordinary-LoRA block every checkpoint on disk carries.

        NOT the entry point for an arbitrary file. A Kohya/LyCORIS block states
        its algorithm in ``ss_*`` keys this does not read, so one is refused
        here rather than silently defaulted to LoRA; read such a file with
        ``AdapterSpec.from_codec(detect_adapter_codec(tensors, metadata))``.
        """
        meta = dict(metadata or {})
        if METADATA_ALGORITHM not in meta:
            foreign = sorted(k for k in meta if k.startswith(_FOREIGN_KEY_PREFIX))
            if foreign:
                _refuse(f"Adapter metadata carries foreign keys {foreign} and no "
                        f"{METADATA_ALGORITHM}: read it with "
                        f"detect_adapter_codec() rather than assuming LoRA.")
        return cls(
            algorithm=(meta.get(METADATA_ALGORITHM) or ALGORITHM_LORA).strip().lower(),
            weight_decompose=parse_metadata_bool(meta.get(METADATA_WEIGHT_DECOMPOSE)),
            rank=_int_or_none(meta.get(METADATA_RANK)),
            alpha=_float_or_none(meta.get(METADATA_ALPHA)),
            architecture=meta.get(METADATA_ARCHITECTURE) or None,
            components=_scope_from_metadata(meta),
            format=(meta.get(METADATA_FORMAT) or FORMAT_SUSHIUI).strip().lower(),
            schema_version=_schema_version_from_metadata(meta),
            options=_options_from_metadata(meta),
        )

    def to_metadata(self) -> Dict[str, str]:
        """The spec-owned half of a checkpoint's metadata block.

        ``step``/``epoch`` and any architecture-specific keys belong to the
        saving adapter; it merges this in rather than this reaching for them.
        """
        meta = {
            METADATA_SCHEMA_VERSION: str(self.schema_version),
            METADATA_ALGORITHM: self.algorithm,
            METADATA_WEIGHT_DECOMPOSE: _format_metadata_bool(self.weight_decompose),
            METADATA_FORMAT: self.format,
        }
        if self.architecture:
            meta[METADATA_ARCHITECTURE] = self.architecture
        if self.components:
            meta[METADATA_TARGET_SCOPE] = ",".join(self.components)
        if self.rank is not None:
            meta[METADATA_RANK] = str(self.rank)
        if self.alpha is not None:
            meta[METADATA_ALPHA] = str(float(self.alpha))
        if self.options:
            meta[METADATA_OPTIONS] = json.dumps(dict(self.options), sort_keys=True)
        return meta

    # -- validation --------------------------------------------------------

    def validate(
        self,
        known_architectures: Optional[Collection[str]] = None,
    ) -> "AdapterSpec":
        """Refuse an inconsistent spec; return ``self`` so callers can chain.

        This is the "refused on application" gate: ``unknown`` is a legitimate
        DETECTION result and an illegitimate thing to apply.
        """
        architectures = (KNOWN_ARCHITECTURES if known_architectures is None
                         else known_architectures)

        if self.algorithm not in ALGORITHMS:
            _refuse(f"Adapter algorithm {self.algorithm!r} is not one of "
                    f"{', '.join(ALGORITHMS)} -- refusing to apply it.")
        if self.format not in FORMATS:
            _refuse(f"Adapter checkpoint format {self.format!r} is not one of "
                    f"{', '.join(FORMATS)} -- refusing to apply it.")
        if self.schema_version > ADAPTER_SCHEMA_VERSION:
            _refuse(f"Adapter schema version {self.schema_version} is newer "
                    f"than this build understands ({ADAPTER_SCHEMA_VERSION}).")
        if self.rank is not None and self.rank < 0:
            _refuse(f"Adapter rank {self.rank} is negative.")
        if self.algorithm in RANK_REQUIRED and not self.rank:
            _refuse(f"{self.algorithm} scales by alpha/rank, so rank "
                    f"{self.rank!r} is unusable.")
        if self.alpha is not None and self.rank is None:
            _refuse(f"Adapter declares alpha {self.alpha} with no rank, so its "
                    f"scale is undefined.")
        if self.architecture is not None and self.architecture not in architectures:
            _refuse(f"Adapter declares architecture {self.architecture!r}, "
                    f"which is not a known training architecture.")
        factor = self.options.get(OPTION_FACTOR)
        if factor is not None and (not isinstance(factor, int)
                                   or isinstance(factor, bool)
                                   or factor == 0 or factor < -1):
            _refuse(f"LoKr factor {factor!r} must be a positive integer or -1 "
                    f"(auto).")
        return self


def _int_or_none(value: Optional[str]) -> Optional[int]:
    try:
        return int(str(value)) if value not in (None, "") else None
    except ValueError:
        return None


def _float_or_none(value: Optional[str]) -> Optional[float]:
    try:
        return float(str(value)) if value not in (None, "") else None
    except ValueError:
        return None


def _schema_version_from_metadata(metadata: Mapping[str, str]) -> int:
    version = _int_or_none(metadata.get(METADATA_SCHEMA_VERSION))
    return ADAPTER_SCHEMA_VERSION if version is None else version


def _scope_from_metadata(metadata: Mapping[str, str]) -> Tuple[str, ...]:
    raw = metadata.get(METADATA_TARGET_SCOPE) or ""
    return tuple(part for part in (p.strip() for p in raw.split(",")) if part)


def _options_from_metadata(metadata: Mapping[str, str]) -> Dict[str, Any]:
    raw = metadata.get(METADATA_OPTIONS)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}
