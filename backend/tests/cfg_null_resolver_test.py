"""The CFG null-alignment parameter surface: resolution, refusal, capability.

Delivery item 2 of local/strategy/cfg_null_alignment/IMPLEMENTATION_STRATEGY.md
-- the parameters exist, resolve and refuse correctly, and NOTHING consumes the
resolved rate yet. Every case here runs on synthetic config dicts; no model is
loaded.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_resolver_test.py -v
"""

import re
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from api import arch_capabilities  # noqa: E402
from api.arch_capabilities import (  # noqa: E402
    CFG_NULL_STAGE_BY_ARCH,
    TRAINING_DECLARED_ARCHS,
    TRAINING_FEATURE_PARAMS,
    training_feature_unsupported_reason,
)
from api.cfg_null_resolver import (  # noqa: E402
    CFG_KEY,
    LEGACY_KEY,
    check_caption_dropout_conflict,
    find_caption_dropout_conflicts,
    resolve_and_check,
    resolve_cfg_uncond_drop_rate,
)
from api.error_handlers import ValidationError  # noqa: E402
from api.param_defaults import (  # noqa: E402
    CFG_UNCOND_DROP_DEFAULTS_BY_ARCH,
    TRAINING_DEFAULTS,
)


def _params(explicit=None, **kwargs):
    """A model_dump()-shaped dict: EVERY key materialised, `_explicit_fields`
    naming only what the caller actually sent."""
    base = {
        CFG_KEY: None,
        LEGACY_KEY: None,
        "danbooru_aug_enable": False,
        "danbooru_aug_caption_dropout_rate": 0.0,
    }
    base.update(kwargs)
    if explicit is not None:
        base["_explicit_fields"] = sorted(explicit)
    return base


@pytest.fixture
def collated_arch(monkeypatch):
    """An architecture that DOES declare a stage.

    Every shipped handler declares None at this delivery item, so the rules that
    only apply to a supported architecture would otherwise be unreachable. The
    stage is injected into the same mirror the resolver reads, not faked inside
    it.
    """
    monkeypatch.setitem(arch_capabilities.CFG_NULL_STAGE_BY_ARCH,
                        "minit2i", "collated")
    return "minit2i"


# ---------------------------------------------------------------------------
# The resolver matrix (strategy §3 rules 1-5)
# ---------------------------------------------------------------------------

def test_omitted_resolves_the_per_architecture_default():
    for arch, expected in CFG_UNCOND_DROP_DEFAULTS_BY_ARCH.items():
        resolution = resolve_cfg_uncond_drop_rate(_params(explicit=[]), arch=arch)
        assert resolution.rate == expected
        assert resolution.source == "arch_default"
        assert resolution.warnings == []


def test_omitted_on_an_arch_with_no_default_resolves_nothing():
    resolution = resolve_cfg_uncond_drop_rate(_params(explicit=[]), arch="sdxl")
    assert resolution.rate is None
    assert resolution.warnings == []


def test_a_null_value_in_model_fields_set_is_not_explicit():
    """The training form sends its whole parameter block on every submit. An
    untouched optional control arrives as an explicit null, and reading that as
    intent would refuse every run on an architecture with no stage."""
    params = _params(explicit=[CFG_KEY, LEGACY_KEY])
    resolution = resolve_cfg_uncond_drop_rate(params, arch="sdxl")
    assert resolution.rate is None
    assert resolution.source == "arch_default"


def test_explicit_zero_disables_the_legacy_default(collated_arch):
    """Rule 2: 0.0 is a value, not an omission. This is the whole reason both
    keys are Optional -- a float default of 0.1 could not express it."""
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: 0.0})
    resolution = resolve_cfg_uncond_drop_rate(params, arch=collated_arch)
    assert resolution.rate == 0.0
    assert resolution.source == CFG_KEY
    assert CFG_UNCOND_DROP_DEFAULTS_BY_ARCH[collated_arch] == 0.1


def test_explicit_rate_is_used_exactly(collated_arch):
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: 0.25})
    assert resolve_cfg_uncond_drop_rate(params, arch=collated_arch).rate == 0.25


def test_legacy_only_on_minit2i_is_used_and_warns(collated_arch):
    params = _params(explicit=[LEGACY_KEY], **{LEGACY_KEY: 0.3})
    resolution = resolve_cfg_uncond_drop_rate(params, arch=collated_arch)
    assert resolution.rate == 0.3
    assert resolution.source == LEGACY_KEY
    assert len(resolution.warnings) == 1
    assert "deprecated" in resolution.warnings[0]


def test_legacy_only_on_another_arch_resolves_that_archs_default():
    """The key was only ever wired for MiniT2I. Elsewhere it has always been
    accepted and ignored; refusing it now would break configs that carry it."""
    params = _params(explicit=[LEGACY_KEY], **{LEGACY_KEY: 0.3})
    resolution = resolve_cfg_uncond_drop_rate(params, arch="sdxl")
    assert resolution.rate is None
    assert resolution.source == "arch_default"


def test_both_keys_explicit_is_refused(collated_arch):
    """Rule 1: a REFUSAL, not a precedence choice."""
    params = _params(explicit=[CFG_KEY, LEGACY_KEY],
                     **{CFG_KEY: 0.2, LEGACY_KEY: 0.3})
    with pytest.raises(ValidationError) as exc:
        resolve_cfg_uncond_drop_rate(params, arch=collated_arch)
    assert "not both" in exc.value.message


def test_both_keys_explicit_is_refused_even_when_they_agree(collated_arch):
    params = _params(explicit=[CFG_KEY, LEGACY_KEY],
                     **{CFG_KEY: 0.1, LEGACY_KEY: 0.1})
    with pytest.raises(ValidationError):
        resolve_cfg_uncond_drop_rate(params, arch=collated_arch)


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), float("inf"),
                                   "0.2abc"])
def test_out_of_range_or_non_finite_is_refused(collated_arch, value):
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: value})
    with pytest.raises(ValidationError):
        resolve_cfg_uncond_drop_rate(params, arch=collated_arch)


def test_the_legacy_key_is_validated_too(collated_arch):
    params = _params(explicit=[LEGACY_KEY], **{LEGACY_KEY: 2.0})
    with pytest.raises(ValidationError) as exc:
        resolve_cfg_uncond_drop_rate(params, arch=collated_arch)
    assert LEGACY_KEY in exc.value.message


def test_unsupported_architecture_refuses_an_explicit_rate():
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: 0.1})
    with pytest.raises(ValidationError) as exc:
        resolve_cfg_uncond_drop_rate(params, arch="sdxl")
    assert "not supported" in exc.value.message


def test_unsupported_architecture_refuses_an_explicit_zero():
    """Explicitly required by the strategy: 0.0 must not be silently ignored."""
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: 0.0})
    with pytest.raises(ValidationError):
        resolve_cfg_uncond_drop_rate(params, arch="sdxl")


def test_unknown_architecture_refuses_an_explicit_rate():
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: 0.1})
    for arch in (None, "", "unknown", "brand_new_arch"):
        with pytest.raises(ValidationError):
            resolve_cfg_uncond_drop_rate(params, arch=arch)


def test_absent_explicit_fields_treats_a_present_value_as_explicit(collated_arch):
    """A hand-authored YAML / direct caller has no Pydantic request behind it.
    Reading that as 'nothing is explicit' would turn its 0.0 back into 0.1."""
    params = _params(**{CFG_KEY: 0.0})
    params.pop("_explicit_fields", None)
    assert resolve_cfg_uncond_drop_rate(params, arch=collated_arch).rate == 0.0


# ---------------------------------------------------------------------------
# Conflict with caption augmentation (strategy §4)
# ---------------------------------------------------------------------------

def test_explicit_rate_refuses_dataset_caption_dropout(collated_arch):
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: 0.1})
    with pytest.raises(ValidationError) as exc:
        resolve_and_check(params, arch=collated_arch,
                          dataset_caption_configs=[
                              ("portraits", {"caption_dropout_rate": 0.05})])
    assert "portraits" in exc.value.detail
    assert "caption_dropout_rate" in exc.value.detail


def test_explicit_rate_refuses_danbooru_caption_dropout(collated_arch):
    params = _params(explicit=[CFG_KEY],
                     **{CFG_KEY: 0.1, "danbooru_aug_enable": True,
                        "danbooru_aug_caption_dropout_rate": 0.2})
    with pytest.raises(ValidationError) as exc:
        resolve_and_check(params, arch=collated_arch)
    assert "danbooru_aug_caption_dropout_rate" in exc.value.detail


def test_a_stale_danbooru_rate_with_the_augmentation_off_is_not_a_conflict(
        collated_arch):
    """The rate is written unconditionally and read only under
    `danbooru_aug_enable` (training_config.py says so in as many words), so a
    value left behind by a disabled augmentation drops no caption. Refusing on
    it would 400 a legal run."""
    params = _params(explicit=[CFG_KEY],
                     **{CFG_KEY: 0.1, "danbooru_aug_enable": False,
                        "danbooru_aug_caption_dropout_rate": 0.2})
    assert resolve_and_check(params, arch=collated_arch).rate == 0.1


def test_explicit_zero_does_not_refuse_caption_dropout(collated_arch):
    """Nothing is being trained against the null, so there is no second
    empty-condition rate to conflict with."""
    params = _params(explicit=[CFG_KEY], **{CFG_KEY: 0.0})
    resolution = resolve_and_check(
        params, arch=collated_arch,
        dataset_caption_configs=[("portraits", {"caption_dropout_rate": 0.05})])
    assert resolution.rate == 0.0
    assert resolution.warnings == []


def test_legacy_minit2i_run_warns_instead_of_being_refused(collated_arch):
    """Saved configurations must keep working: the omitted-key path warns."""
    params = _params(explicit=[])
    resolution = resolve_and_check(
        params, arch=collated_arch,
        dataset_caption_configs=[("portraits", {"caption_dropout_rate": 0.05})])
    assert resolution.rate == 0.1
    assert len(resolution.warnings) == 1
    assert "portraits" in resolution.warnings[0]


def test_conflicts_name_every_source():
    params = _params(**{"danbooru_aug_enable": True,
                        "danbooru_aug_caption_dropout_rate": 0.2})
    conflicts = find_caption_dropout_conflicts(
        params,
        [("a", {"caption_dropout_rate": 0.1}),
         ("b", {"caption_dropout_rate": 0.0}),
         ("c", None)],
    )
    assert len(conflicts) == 2
    assert any("danbooru_aug_caption_dropout_rate" in c for c in conflicts)
    assert any("dataset 'a'" in c for c in conflicts)


def test_no_conflict_means_no_warning(collated_arch):
    resolution = resolve_and_check(_params(explicit=[]), arch=collated_arch)
    assert resolution.warnings == []
    assert check_caption_dropout_conflict(resolution, _params(), []) == []


# ---------------------------------------------------------------------------
# Capability declaration + parameter checklist
# ---------------------------------------------------------------------------

def test_the_stage_mirror_matches_the_arch_handlers():
    """CFG_NULL_STAGE_BY_ARCH restates ArchHandler.cfg_null_stage because
    arch_capabilities cannot import the trainer package; this keeps it honest."""
    from core.training.arch import ARCH_REGISTRY

    actual = {arch: handler.cfg_null_stage
              for arch, handler in ARCH_REGISTRY.items()}
    assert actual == CFG_NULL_STAGE_BY_ARCH


def test_only_the_delivered_architectures_declare_a_stage():
    """Items 3 and 4 route MiniT2I and Lens through the resolver; SenseNova
    (item 5) is still undelivered and must not read as enabled."""
    declared = {arch: stage for arch, stage in CFG_NULL_STAGE_BY_ARCH.items()
                if stage is not None}
    assert declared == {"minit2i": "collated", "lens": "collated"}


def test_every_stageless_arch_declares_the_feature_unsupported():
    for arch in TRAINING_DECLARED_ARCHS:
        if CFG_NULL_STAGE_BY_ARCH[arch] is None:
            assert training_feature_unsupported_reason(arch, "cfg_uncond_drop")


def test_the_feature_is_armed_by_the_new_key_only():
    """The deprecated key must not arm the capability gate: an architecture that
    has always accepted and ignored it would newly lose an unrelated control."""
    assert TRAINING_FEATURE_PARAMS["cfg_uncond_drop"] == [CFG_KEY]


def test_the_base_handler_hooks_reject():
    """A handler only implements the hook its declared stage names; every other
    combination refuses instead of silently doing nothing."""
    from core.training.arch import ARCH_REGISTRY

    for arch, handler_cls in ARCH_REGISTRY.items():
        handler = handler_cls.__new__(handler_cls)
        if handler_cls.cfg_null_stage != "collated":
            with pytest.raises(NotImplementedError):
                handler.apply_cfg_null_collated(None, None, None, None)
        if handler_cls.cfg_null_stage != "encode":
            with pytest.raises(NotImplementedError):
                handler.encode_prompt_cfg_null(None, "a prompt")


def test_a_declared_stage_is_backed_by_an_override():
    """The other half of the test above: skipping the refusal check for a
    stage-declaring handler must not let one declare a stage while inheriting
    the base hook, which would refuse on the production path instead."""
    from core.training.arch import ARCH_REGISTRY
    from core.training.arch.base_arch import ArchHandler

    hooks = {"collated": "apply_cfg_null_collated",
             "encode": "encode_prompt_cfg_null"}
    for arch, handler_cls in ARCH_REGISTRY.items():
        stage = handler_cls.cfg_null_stage
        if stage is None:
            continue
        assert stage in hooks, f"{arch} declares unknown stage {stage!r}"
        hook = hooks[stage]
        assert getattr(handler_cls, hook) is not getattr(ArchHandler, hook), (
            f"{arch} declares cfg_null_stage={stage!r} without overriding {hook}")


def test_defaults_are_the_single_source_of_truth():
    assert TRAINING_DEFAULTS[CFG_KEY] is None
    assert TRAINING_DEFAULTS[LEGACY_KEY] is None
    assert CFG_UNCOND_DROP_DEFAULTS_BY_ARCH == {
        "minit2i": 0.1, "lens": 0.0, "sensenova": 0.0}


def test_no_literal_default_survives_outside_param_defaults():
    """The bug this parameter shape exists to prevent is a downstream
    `get(key, 0.1)` turning an explicit 0.0 back into 0.1."""
    for relative in ("core/training/training_config.py",
                     "core/training/ops/minit2i_ops.py"):
        source = (BACKEND / relative).read_text(encoding="utf-8")
        for key in (CFG_KEY, LEGACY_KEY):
            assert not re.search(rf"{key}[\"']?\s*,\s*0\.1", source), (
                f"{relative} reintroduces a literal 0.1 fallback for {key}")


def test_the_request_model_declares_both_keys_optional():
    source = (BACKEND / "api" / "routes.py").read_text(encoding="utf-8")
    for key in (CFG_KEY, LEGACY_KEY):
        assert re.search(rf"^    {key}: Optional\[float\] = TRAINING_DEFAULTS",
                         source, re.M), f"{key} is not Optional in the request model"


def _train_section(arch, **params):
    from core.training.training_config import _build_train_section

    params.setdefault("_explicit_fields", [])
    return _build_train_section(params, total_steps=10, epochs=None,
                                train_unet=True, train_text_encoder=False,
                                arch=arch)


def test_the_yaml_carries_the_supplied_value_not_the_resolved_one(collated_arch):
    """Writing the resolved 0.1 into a MiniT2I config would come back through
    GET /params as a value the edit form re-sends as an EXPLICIT 0.1 -- refused
    on an architecture with no cfg_null_stage. The resolved rate is a function
    of this value and the architecture, both already in the config."""
    assert _train_section(collated_arch)[CFG_KEY] is None
    section = _train_section(collated_arch, **{CFG_KEY: 0.0,
                                               "_explicit_fields": [CFG_KEY]})
    assert section[CFG_KEY] == 0.0


def test_the_yaml_builder_refuses_what_the_route_refuses():
    with pytest.raises(ValidationError):
        _train_section("sdxl", **{CFG_KEY: 0.2, "_explicit_fields": [CFG_KEY]})


def test_the_yaml_carries_the_supplied_legacy_value_not_the_resolved_one():
    """Item 3's landmine. The generator used to materialise MiniT2I's 0.1 into
    every config. Once MiniT2I declares a stage, such a config carries BOTH keys
    with real values and the "supply either, not both" rule fires on a key the
    caller never sent -- at training time, and on any GET /params -> PUT client.
    The supplied value, null included, is what round-trips."""
    assert _train_section("minit2i")[LEGACY_KEY] is None
    assert _train_section("sdxl")[LEGACY_KEY] is None
    section = _train_section("minit2i", minit2i_label_drop_rate=0.4,
                             _explicit_fields=[LEGACY_KEY])
    assert section[LEGACY_KEY] == 0.4


def test_a_generated_minit2i_config_never_carries_both_keys():
    section = _train_section("minit2i", **{CFG_KEY: 0.2,
                                           "_explicit_fields": [CFG_KEY]})
    assert section[CFG_KEY] == 0.2
    assert section[LEGACY_KEY] is None
    # The generated train section IS what the trainer reads, with no field-set
    # information behind it. Re-resolving it must not refuse.
    assert resolve_cfg_uncond_drop_rate(section, arch="minit2i").rate == 0.2


def _generated_minit2i_config(monkeypatch, **params):
    import yaml

    import core.training.training_config as tc

    monkeypatch.setattr(tc, "_detect_arch", lambda _path: "minit2i")
    params.setdefault("learning_rate", 1e-4)
    params.setdefault("batch_size", 1)
    params.setdefault("total_steps", 10)
    config = tc.TrainingConfigGenerator.generate_lora_config(
        params,
        run_name="cfg_null_round_trip",
        base_model_path="/models/minit2i.safetensors",
        output_dir="/tmp/cfg_null_round_trip",
        dataset_configs=[{"dataset_id": 1, "path": "/data/ds"}],
        sample_prompts=[],
    )
    return yaml.safe_load(config)["config"]["process"][0]


def test_the_config_edit_channel_round_trips_without_manufacturing_a_conflict(
        monkeypatch):
    """GET /params -> PUT. The extractor reads both keys off the train section,
    so whatever the generator materialised comes back as a value the client then
    re-sends as explicit."""
    from api.routes import _extract_request_params_from_yaml

    process = _generated_minit2i_config(
        monkeypatch, **{CFG_KEY: 0.2, "_explicit_fields": [CFG_KEY]})
    params = _extract_request_params_from_yaml(process, "lora")
    assert params[CFG_KEY] == 0.2
    assert params[LEGACY_KEY] is None

    # A client that re-sends every field it was handed a value for.
    params["_explicit_fields"] = sorted(
        k for k, v in params.items() if v is not None)
    assert resolve_cfg_uncond_drop_rate(params, arch="minit2i").rate == 0.2


def test_the_old_materialised_legacy_value_is_what_would_have_refused():
    """Negative control for the fix above: this is the exact dict the auditor
    reproduced, and it is no longer reachable from the generator."""
    params = _params(explicit=[CFG_KEY, LEGACY_KEY],
                     **{CFG_KEY: 0.2, LEGACY_KEY: 0.1})
    with pytest.raises(ValidationError):
        resolve_cfg_uncond_drop_rate(params, arch="minit2i")


def test_a_hand_authored_yaml_with_only_the_legacy_key_still_trains(monkeypatch):
    process = _generated_minit2i_config(
        monkeypatch, **{LEGACY_KEY: 0.3, "_explicit_fields": [LEGACY_KEY]})
    train = process["train"]
    assert train[LEGACY_KEY] == 0.3
    assert train[CFG_KEY] is None
    resolution = resolve_cfg_uncond_drop_rate(train, arch="minit2i")
    assert resolution.rate == 0.3
    assert resolution.warnings and "deprecated" in resolution.warnings[0]


def test_the_route_serves_the_capability_keys():
    source = (BACKEND / "api" / "routes.py").read_text(encoding="utf-8")
    for key in ("cfg_null_stage", "cfg_uncond_drop_defaults"):
        assert f'"{key}"' in source


def test_openapi_and_the_frontend_carry_the_parameter():
    spec = (REPO / "openapi.yaml").read_text(encoding="utf-8")
    assert re.search(rf"^        {CFG_KEY}:$", spec, re.M)
    for key in ("cfg_null_stage", "cfg_uncond_drop_defaults"):
        assert re.search(rf"^        {key}:$", spec, re.M)

    api_ts = (REPO / "frontend" / "src" / "utils" / "api.ts").read_text(
        encoding="utf-8")
    assert f"{CFG_KEY}?:" in api_ts
    assert "cfg_uncond_drop_defaults?:" in api_ts

    panel = (REPO / "frontend" / "src" / "components" / "training"
             / "TrainingConfig.tsx").read_text(encoding="utf-8")
    assert CFG_KEY in panel
    assert "cfgUncondDropUnsupported" in panel
