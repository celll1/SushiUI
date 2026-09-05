"""The latent space an adapter was trained in, and whether it matches a model.

Design ``docs/guides/VAE_SWAP_MIGRATION_DESIGN.md`` D10 / §9.4. This exists
because a VAE swap is INVISIBLE to every shape-based check on SD1.5/SDXL: the
only modules a swap resizes are ``conv_in``/``conv_out``, which are ``Conv2d``
and therefore unreachable by the adapter target scan, so a 4-channel LoRA
applies to a 16-channel checkpoint at 100% with zero mismatches and silently
contributes deltas learned in another latent space.

Layering: pure data. Nothing here imports ``core.training``, ``api``, a
pipeline or torch, so both read paths (``core.extensions.lora_manager`` for the
two diffusers architectures, ``AdapterSession`` for the other eleven) and the
write path share ONE table of verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Optional

METADATA_BASE_LATENT_CHANNELS = "sushi.base.latent_channels"
METADATA_BASE_VAE_TYPE = "sushi.base.vae_type"
METADATA_BASE_VAE_HASH = "sushi.base.vae_hash"
METADATA_BASE_VAE_STRUCT_NATIVE = "sushi.base.vae_struct_native"
METADATA_BASE_VAE_IDENTITY_NATIVE = "sushi.base.vae_identity_native"

#: Every key this module writes; a reader tests membership to decide whether a
#: checkpoint declares an identity at all.
BASE_LATENT_METADATA_KEYS = (
    METADATA_BASE_LATENT_CHANNELS,
    METADATA_BASE_VAE_TYPE,
    METADATA_BASE_VAE_HASH,
    METADATA_BASE_VAE_STRUCT_NATIVE,
    METADATA_BASE_VAE_IDENTITY_NATIVE,
)

#: ``AdapterSpec.options`` key carrying the same block (design §9.4).
OPTION_BASE_LATENT = "base_latent"

REFUSAL_CODE = "lora_incompatible"
WARNING_CODE_MISMATCH = "lora_base_vae_mismatch"
WARNING_CODE_UNKNOWN = "lora_base_vae_unknown"

__all__ = [
    "BASE_LATENT_METADATA_KEYS",
    "METADATA_BASE_LATENT_CHANNELS",
    "METADATA_BASE_VAE_HASH",
    "METADATA_BASE_VAE_IDENTITY_NATIVE",
    "METADATA_BASE_VAE_STRUCT_NATIVE",
    "METADATA_BASE_VAE_TYPE",
    "OPTION_BASE_LATENT",
    "REFUSAL_CODE",
    "WARNING_CODE_MISMATCH",
    "WARNING_CODE_UNKNOWN",
    "BaseLatentIdentity",
    "BaseLatentVerdict",
    "check_base_latent",
    "report_fields",
]


def _flag(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def _channels(value: Any) -> Optional[int]:
    """A positive channel count, or None. Zero is a pixel-space architecture's
    wiring constant and means "this side states nothing", never "0 channels"."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


@dataclass(frozen=True)
class BaseLatentIdentity:
    """The latent space one side of the comparison lives in.

    ``struct_native``/``identity_native`` keep the two questions apart exactly
    as ``component.vae.*`` does (design §5.2, invariant §13-7): structure
    (channel layout) drives hard refusals, identity (which VAE's weights) drives
    warnings.
    """

    latent_channels: Optional[int] = None
    vae_type: Optional[str] = None
    vae_hash: Optional[str] = None
    struct_native: bool = True
    identity_native: bool = True

    @classmethod
    def from_facts(cls, facts: Optional[Mapping[str, Any]]
                   ) -> Optional["BaseLatentIdentity"]:
        """From a ``ResolvedVAE.facts()`` block or a ``current_model_info``
        row. Accepts both spellings of the two flags and both of the hash."""
        if not facts:
            return None
        return cls(
            latent_channels=_channels(facts.get("latent_channels")),
            vae_type=facts.get("family") or facts.get("vae_type") or None,
            vae_hash=facts.get("content_hash") or facts.get("vae_hash") or None,
            struct_native=_flag(facts.get("struct_native",
                                          facts.get("vae_struct_native"))),
            identity_native=_flag(facts.get("identity_native",
                                            facts.get("vae_identity_native"))),
        )

    @classmethod
    def from_metadata(cls, metadata: Optional[Mapping[str, str]]
                      ) -> Optional["BaseLatentIdentity"]:
        """The identity an adapter checkpoint declares, or None when it
        declares none -- which is a distinct D10 row, not a default."""
        meta = metadata or {}
        if not any(key in meta for key in BASE_LATENT_METADATA_KEYS):
            return None
        return cls(
            latent_channels=_channels(meta.get(METADATA_BASE_LATENT_CHANNELS)),
            vae_type=meta.get(METADATA_BASE_VAE_TYPE) or None,
            vae_hash=meta.get(METADATA_BASE_VAE_HASH) or None,
            struct_native=_flag(meta.get(METADATA_BASE_VAE_STRUCT_NATIVE)),
            identity_native=_flag(meta.get(METADATA_BASE_VAE_IDENTITY_NATIVE)),
        )

    def to_metadata(self) -> dict:
        meta = {
            METADATA_BASE_VAE_STRUCT_NATIVE: "1" if self.struct_native else "0",
            METADATA_BASE_VAE_IDENTITY_NATIVE: ("1" if self.identity_native
                                                else "0"),
        }
        if self.latent_channels:
            meta[METADATA_BASE_LATENT_CHANNELS] = str(int(self.latent_channels))
        if self.vae_type:
            meta[METADATA_BASE_VAE_TYPE] = str(self.vae_type)
        if self.vae_hash:
            meta[METADATA_BASE_VAE_HASH] = str(self.vae_hash)
        return meta

    def to_options(self) -> dict:
        return {
            "latent_channels": self.latent_channels,
            "vae_type": self.vae_type,
            "vae_hash": self.vae_hash,
            "struct_native": self.struct_native,
            "identity_native": self.identity_native,
        }

    @classmethod
    def from_options(cls, options: Optional[Mapping[str, Any]]
                     ) -> Optional["BaseLatentIdentity"]:
        if not options:
            return None
        return cls(
            latent_channels=_channels(options.get("latent_channels")),
            vae_type=options.get("vae_type") or None,
            vae_hash=options.get("vae_hash") or None,
            struct_native=_flag(options.get("struct_native")),
            identity_native=_flag(options.get("identity_native")),
        )


class BaseLatentVerdict(NamedTuple):
    """``refuse`` is the hard/soft boundary of D10; ``code`` is what the API
    reports either way. ``message`` is None only when nothing is to be said."""

    refuse: bool
    code: Optional[str]
    message: Optional[str]

    @property
    def ok(self) -> bool:
        return self.code is None


_OK = BaseLatentVerdict(False, None, None)


def check_base_latent(adapter: Optional[BaseLatentIdentity],
                      model: Optional[BaseLatentIdentity],
                      *, name: str = "this adapter") -> BaseLatentVerdict:
    """The D10 table, and only it -- four outcomes, no fifth.

    ``adapter`` is what the checkpoint declares (None = declares nothing),
    ``model`` what the loaded base is (None = the caller could not resolve one,
    which is treated as the architecture's own latent space).
    """
    model = model or BaseLatentIdentity()

    if adapter is None:
        if not model.struct_native:
            return BaseLatentVerdict(
                True, REFUSAL_CODE,
                f"LoRA '{name}' declares no base latent identity, and this "
                f"model runs on a swapped VAE whose channel layout "
                f"({model.latent_channels or 'unknown'} channels) is not the "
                f"architecture's own. Every adapter SushiUI trains against such "
                f"a base records its latent space, so a file without one was "
                f"trained elsewhere; applying it would fit every target shape "
                f"and contribute deltas from a different latent space. Train "
                f"the adapter against this base, or load the architecture's "
                f"stock checkpoint.")
        if not model.identity_native:
            return BaseLatentVerdict(
                False, WARNING_CODE_UNKNOWN,
                f"LoRA '{name}' declares no base latent identity and this model "
                f"runs on a replacement VAE with the architecture's own channel "
                f"layout. The adapter applies, but nothing states which latent "
                f"space it was trained in.")
        return _OK

    if (adapter.latent_channels and model.latent_channels
            and adapter.latent_channels != model.latent_channels):
        return BaseLatentVerdict(
            True, REFUSAL_CODE,
            f"LoRA '{name}' was trained against a {adapter.latent_channels}"
            f"-channel latent space and this model has "
            f"{model.latent_channels}. Its targets are the same Linear layers "
            f"either way, so it would apply in full and steer the model with "
            f"deltas learned elsewhere.")

    if (adapter.vae_hash and model.vae_hash
            and adapter.vae_hash != model.vae_hash):
        return BaseLatentVerdict(
            False, WARNING_CODE_MISMATCH,
            f"LoRA '{name}' was trained against a different VAE "
            f"({adapter.vae_type or 'unknown family'} {adapter.vae_hash}) than "
            f"this model declares ({model.vae_type or 'unknown family'} "
            f"{model.vae_hash}). Same channel layout, different latent "
            f"distribution.")

    if adapter.identity_native != model.identity_native:
        trained, loaded = (("the architecture's own VAE", "a replacement VAE")
                           if adapter.identity_native
                           else ("a replacement VAE", "the architecture's own VAE"))
        return BaseLatentVerdict(
            False, WARNING_CODE_MISMATCH,
            f"LoRA '{name}' was trained against {trained} and this model runs "
            f"on {loaded}. Same channel layout, different latent distribution.")

    return _OK


def report_fields(identity: Optional[BaseLatentIdentity]) -> dict:
    """The ``GET /loras`` half of an adapter's identity. All-null when the file
    declares none -- the listing has no model to judge against."""
    if identity is None:
        return {
            "base_latent_channels": None,
            "base_vae_type": None,
            "base_vae_hash": None,
            "base_vae_struct_native": None,
            "base_vae_identity_native": None,
        }
    return {
        "base_latent_channels": identity.latent_channels,
        "base_vae_type": identity.vae_type,
        "base_vae_hash": identity.vae_hash,
        "base_vae_struct_native": bool(identity.struct_native),
        "base_vae_identity_native": bool(identity.identity_native),
    }
