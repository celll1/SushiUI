"""Resolution and refusal for the CFG null-alignment training parameters.

One user-facing concept, ``cfg_uncond_drop_rate``: the per-sample Bernoulli
probability of training an item against the SAME null condition the
architecture's inference CFG uncond branch builds. ``minit2i_label_drop_rate``
is the deprecated MiniT2I-only spelling that predates it.

Both keys are ``Optional`` with a ``None`` default, because the feature needs
three states where a plain float has two: "disable" (0.0), "use the
architecture's default" (omitted) and a rate. Resolving them therefore needs to
know what the caller actually SENT, which ``BaseModel.model_dump()`` cannot say
-- it materialises every default as a value. The route passes
``request.model_fields_set`` through as ``params["_explicit_fields"]``, the same
channel ``generate_vae_config`` already uses.

Nothing here consumes the resolved rate; it is the parameter surface only.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from api.error_handlers import ValidationError
from api.param_defaults import CFG_UNCOND_DROP_DEFAULTS_BY_ARCH

CFG_KEY = "cfg_uncond_drop_rate"
LEGACY_KEY = "minit2i_label_drop_rate"

#: The architecture the deprecated key was ever wired for.
LEGACY_KEY_ARCH = "minit2i"


class CfgUncondDropResolution:
    """The resolved rate plus how it was arrived at.

    ``rate`` is ``None`` only when the architecture has no per-architecture
    default AND nothing was supplied, i.e. the mechanism is simply not in play
    for this run.
    """

    __slots__ = ("rate", "source", "arch", "stage", "warnings")

    def __init__(self, rate: Optional[float], source: str, arch: Optional[str],
                 stage: Optional[str], warnings: List[str]):
        self.rate = rate
        self.source = source
        self.arch = arch
        self.stage = stage
        self.warnings = warnings

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CfgUncondDropResolution(rate={self.rate!r}, "
                f"source={self.source!r}, arch={self.arch!r}, "
                f"stage={self.stage!r}, warnings={self.warnings!r})")


def _explicit_field_set(params: Dict[str, Any]) -> Optional[set]:
    """``params["_explicit_fields"]`` as a set, or ``None`` when the caller
    supplied no field-set information at all.

    ``None`` is NOT "nothing was sent": a direct/legacy caller (train_runner
    reading a hand-authored YAML, a test) has no Pydantic request behind it, and
    for those the presence of a non-``None`` value in the dict is the only
    signal available. Treating an absent ``_explicit_fields`` as "nothing is
    explicit" would make a hand-authored ``cfg_uncond_drop_rate: 0.0`` silently
    resolve back to MiniT2I's 0.1.
    """
    explicit = params.get("_explicit_fields")
    if explicit is None:
        return None
    return set(explicit)


def _is_explicit(params: Dict[str, Any], key: str,
                 explicit_fields: Optional[set]) -> bool:
    """Whether ``key`` was EXPLICITLY supplied.

    A ``None`` value never counts, even when the key rides in
    ``model_fields_set``. The training form sends its whole parameter block on
    every submit, so an untouched optional control arrives as an explicit
    ``null``; reading that as intent would refuse every run on an architecture
    with no null stage.
    """
    if params.get(key) is None:
        return False
    if explicit_fields is None:
        return True
    return key in explicit_fields


def _validate_rate(key: str, value: Any) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{key} must be a number",
            detail=f"got {value!r} ({type(value).__name__})",
        )
    if rate != rate or rate in (float("inf"), float("-inf")):
        raise ValidationError(
            f"{key} must be finite",
            detail=f"got {value!r}",
        )
    if not (0.0 <= rate <= 1.0):
        raise ValidationError(
            f"{key} must be between 0 and 1",
            detail=f"got {rate}; it is a per-sample probability",
        )
    return rate


def cfg_null_stage(arch: Optional[str]) -> Optional[str]:
    """The declared null-construction stage of ``arch``, or ``None``.

    Read through ``api.arch_capabilities`` (which mirrors
    ``ArchHandler.cfg_null_stage`` for the API process) rather than importing
    the arch registry, which pulls the whole trainer stack into this process.
    """
    from api.arch_capabilities import CFG_NULL_STAGE_BY_ARCH

    if not arch:
        return None
    return CFG_NULL_STAGE_BY_ARCH.get(arch)


def resolve_cfg_uncond_drop_rate(params: Dict[str, Any], *,
                                 arch: Optional[str]) -> CfgUncondDropResolution:
    """Resolve the run's aligned-null drop rate, or raise ``ValidationError``.

    Rules, in the order applied:

    1. BOTH keys explicitly supplied -> REFUSE. Not a precedence choice: the two
       keys mean the same thing and a caller that sent both disagreeing values
       has no answer that is safe to guess.
    2. ``cfg_uncond_drop_rate`` supplied -> use it exactly, ``0.0`` included,
       which is what disables MiniT2I's legacy default. Refused when the
       architecture declares no ``cfg_null_stage``.
    3. only ``minit2i_label_drop_rate`` supplied, on MiniT2I -> use it, and warn
       that the key is deprecated. On any other architecture the key is not
       this architecture's parameter at all and is left where it is
       (historically accepted and ignored there), so it does not resolve a rate.
    4. neither supplied -> ``CFG_UNCOND_DROP_DEFAULTS_BY_ARCH``, or ``None``
       when the architecture is absent from it.
    5. any explicitly supplied value is validated finite and in [0, 1] first, so
       the refusal happens before the model loads.
    """
    explicit_fields = _explicit_field_set(params)
    cfg_explicit = _is_explicit(params, CFG_KEY, explicit_fields)
    legacy_explicit = _is_explicit(params, LEGACY_KEY, explicit_fields)
    stage = cfg_null_stage(arch)
    warnings: List[str] = []

    if cfg_explicit and legacy_explicit:
        raise ValidationError(
            f"Supply either {CFG_KEY} or {LEGACY_KEY}, not both",
            detail=(
                f"{CFG_KEY}={params[CFG_KEY]!r}, {LEGACY_KEY}="
                f"{params[LEGACY_KEY]!r}. They set the same rate; "
                f"{LEGACY_KEY} is the deprecated MiniT2I-only spelling. "
                f"Remove it and keep {CFG_KEY}."
            ),
        )

    if cfg_explicit:
        rate = _validate_rate(CFG_KEY, params[CFG_KEY])
        if stage is None:
            raise ValidationError(
                f"{CFG_KEY} is not supported for architecture "
                f"'{arch or 'unknown'}'",
                detail=(
                    f"This architecture's training path cannot build the null "
                    f"condition its inference CFG uncond branch uses, so the "
                    f"rate -- including {CFG_KEY}=0.0 -- would have no defined "
                    f"meaning here. Remove the field. "
                    f"GET /schema/arch-capabilities -> "
                    f"training_feature_unsupported['{arch or 'unknown'}']"
                    f"['cfg_uncond_drop'] carries the same statement."
                ),
            )
        return CfgUncondDropResolution(rate, CFG_KEY, arch, stage, warnings)

    if legacy_explicit:
        rate = _validate_rate(LEGACY_KEY, params[LEGACY_KEY])
        if arch == LEGACY_KEY_ARCH:
            warnings.append(
                f"{LEGACY_KEY} is deprecated; it is now the MiniT2I-only "
                f"spelling of {CFG_KEY}, which every architecture with an "
                f"aligned null condition shares. The run uses {rate}."
            )
            return CfgUncondDropResolution(rate, LEGACY_KEY, arch, stage,
                                           warnings)
        # Other architectures have always accepted and ignored this key; it
        # resolves nothing for them, and refusing it now would break configs
        # that merely carry it.
        return CfgUncondDropResolution(
            CFG_UNCOND_DROP_DEFAULTS_BY_ARCH.get(arch or ""),
            "arch_default", arch, stage, warnings)

    return CfgUncondDropResolution(
        CFG_UNCOND_DROP_DEFAULTS_BY_ARCH.get(arch or ""),
        "arch_default", arch, stage, warnings)


# ---------------------------------------------------------------------------
# Conflict with caption augmentation (strategy §4)
# ---------------------------------------------------------------------------

DANBOORU_CAPTION_DROPOUT_KEY = "danbooru_aug_caption_dropout_rate"
DANBOORU_ENABLE_KEY = "danbooru_aug_enable"
DATASET_CAPTION_DROPOUT_KEY = "caption_dropout_rate"
REFERENCE_IMAGES_KEY = "use_reference_images"

#: Architectures whose aligned null is text-only, so that a reference-conditioned
#: item has an inference baseline the null does not represent. Declared as a set
#: rather than derived from the stage: whether references interact with the null
#: is a property of the architecture's CFG shape, not of where it builds the
#: condition. An architecture absent from here either has no reference
#: conditioning in training at all, or its references are already outside the
#: text condition; either way there is nothing for this refusal to say, and
#: saying it would misattribute a refusal that train_runner's own reference
#: contract owns.
CFG_NULL_REFERENCE_UNSUPPORTED = frozenset({"sensenova"})

#: Private train-section key carrying ``[(dataset label, caption_processing), ...]``.
#:
#: A dataset's caption processing lives in the datasets DB, never in the YAML
#: ("caption_processing settings are NOT saved to YAML", training_config.py), so
#: a trainer holding only the train section cannot see it. train_runner reads it
#: from the DB before any scan or model load and parks it here, on the same dict
#: it hands the trainer, so both refuse on the same inputs.
DATASET_CAPTION_CONFIGS_KEY = "_dataset_caption_configs"


def _nonzero(value: Any) -> bool:
    try:
        return float(value or 0.0) != 0.0
    except (TypeError, ValueError):
        return False


def find_caption_dropout_conflicts(
    params: Dict[str, Any],
    dataset_caption_configs: Optional[Sequence[Tuple[str, Any]]] = None,
) -> List[str]:
    """Every whole-caption dropout source that is nonzero, named by its key.

    ``dataset_caption_configs`` is ``(dataset label, caption_processing dict)``
    pairs -- the caption config is a DATASET property read from the database at
    training time, so the route has to hand it in.
    """
    conflicts: List[str] = []
    # The rate is stored unconditionally and read only under the enable flag
    # (base_trainer.py: the whole augmentation block is inside it), so a stale
    # nonzero rate on a run with the augmentation OFF drops no caption and is
    # not a conflict.
    if (params.get(DANBOORU_ENABLE_KEY)
            and _nonzero(params.get(DANBOORU_CAPTION_DROPOUT_KEY))):
        conflicts.append(
            f"{DANBOORU_CAPTION_DROPOUT_KEY}="
            f"{params[DANBOORU_CAPTION_DROPOUT_KEY]}"
        )
    for label, caption_config in (dataset_caption_configs or ()):
        if not isinstance(caption_config, dict):
            continue
        value = caption_config.get(DATASET_CAPTION_DROPOUT_KEY)
        if _nonzero(value):
            conflicts.append(
                f"dataset '{label}': {DATASET_CAPTION_DROPOUT_KEY}={value}"
            )
    return conflicts


def check_caption_dropout_conflict(
    resolution: CfgUncondDropResolution,
    params: Dict[str, Any],
    dataset_caption_configs: Optional[Sequence[Tuple[str, Any]]] = None,
) -> List[str]:
    """Refuse, or warn, when whole-caption dropout runs alongside the aligned null.

    A caption dropped to "" is still a real conditional forward on the empty
    string, which for these architectures is NOT the null condition. Running
    both puts a second, differently-represented empty-condition rate into the
    objective that nothing declares.

    EXPLICIT ``cfg_uncond_drop_rate`` -> refuse, naming the conflicting key.
    Legacy MiniT2I runs (the key omitted, or the deprecated spelling) keep
    working and get a warning instead, so saved configurations are not
    invalidated by this change.

    Returns the warnings to surface; raises ``ValidationError`` on refusal.
    """
    conflicts = find_caption_dropout_conflicts(params, dataset_caption_configs)
    if not conflicts:
        return []
    if not _nonzero(resolution.rate):
        return []

    joined = "; ".join(conflicts)
    if resolution.source == CFG_KEY:
        raise ValidationError(
            f"{CFG_KEY}={resolution.rate} cannot be combined with "
            f"whole-caption dropout",
            detail=(
                f"Conflicting setting(s): {joined}. A dropped caption is "
                f"encoded as an empty string, which is a conditional forward, "
                f"not the null condition {CFG_KEY} trains against. With both "
                f"on, the objective carries a second empty-condition rate that "
                f"no parameter states. Set the listed key(s) to 0, or remove "
                f"{CFG_KEY}."
            ),
        )
    return [
        f"{joined} runs alongside a nonzero {resolution.source} "
        f"({resolution.rate}). The two drop different things: the caption "
        f"dropout encodes an empty string (a conditional forward), while "
        f"{resolution.source} builds the inference null condition. "
        f"Supply {CFG_KEY} explicitly to have this combination refused."
    ]


def check_reference_conditioning_conflict(
    resolution: CfgUncondDropResolution, params: Dict[str, Any],
) -> None:
    """Refuse an aligned null on a run that conditions items on reference images.

    The null this release builds is the TEXT-ONLY uncond prefix: an empty prompt
    with no reference tokens. That is the branch inference blends against only
    when no references are in play. With references, the branch generation uses
    as its CFG baseline at the shipped ``img_cfg_scale=1`` is ``img_cond``,
    which KEEPS the reference tokens and drops only the text
    (``core/models/sensenova/sensenova_pipeline_ops.py``: ``needs_uncond =
    needs_cfg and img_cfg_scale != 1``). One Bernoulli label cannot supervise
    both marginals, and mapping a reference item onto the text-only null would
    train it against a condition inference never uses at the served scale.

    Refused rather than warned regardless of where the rate came from: no
    architecture has ever trained an aligned null beside reference conditioning,
    so there is no prior behaviour to preserve here.
    """
    if not _nonzero(resolution.rate):
        return
    if resolution.arch not in CFG_NULL_REFERENCE_UNSUPPORTED:
        return
    if not params.get(REFERENCE_IMAGES_KEY):
        return
    raise ValidationError(
        f"{CFG_KEY}={resolution.rate} cannot be combined with "
        f"reference-conditioned items",
        detail=(
            f"{REFERENCE_IMAGES_KEY}={params[REFERENCE_IMAGES_KEY]!r}. The null "
            f"this trains against is the text-only uncond prefix -- an empty "
            f"prompt with no reference tokens. At the default img_cfg_scale=1 a "
            f"reference-conditioned generation blends against img_cond, which "
            f"keeps the references, so the two are different conditions and one "
            f"per-sample label cannot supervise both. Set "
            f"{REFERENCE_IMAGES_KEY} to false, or remove {CFG_KEY}."
        ),
    )


def resolve_and_check(
    params: Dict[str, Any], *, arch: Optional[str],
    dataset_caption_configs: Optional[Sequence[Tuple[str, Any]]] = None,
) -> CfgUncondDropResolution:
    """``resolve_cfg_uncond_drop_rate`` plus the §4/§6.3 refusals, one call.

    This is what the route, train_runner and the trainer all invoke: every part
    must run before the model loads, and all of them raise the same
    ``ValidationError``.

    ``dataset_caption_configs=None`` falls back to
    ``params[DATASET_CAPTION_CONFIGS_KEY]``, which is how a caller that has only
    the train section (the trainer) still sees the dataset half of the check.
    """
    if dataset_caption_configs is None:
        dataset_caption_configs = params.get(DATASET_CAPTION_CONFIGS_KEY)
    resolution = resolve_cfg_uncond_drop_rate(params, arch=arch)
    check_reference_conditioning_conflict(resolution, params)
    resolution.warnings.extend(
        check_caption_dropout_conflict(resolution, params,
                                       dataset_caption_configs)
    )
    return resolution
