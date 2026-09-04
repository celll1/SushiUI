"""The three-way contradiction in SenseNova's `text_encoder_training` entry, and
the fifth capability axis that replaced it.

WHAT SHIPPED (reproduced below as negative controls): the capability table
declared `text_encoder_training` UNSUPPORTED for SenseNova full fine-tuning, so
the training form forced `train_text_encoder` off and
`GET /schema/arch-capabilities` reported the mechanism absent -- while the REST
API accepted `train_text_encoder: true` and the trainer ran it, on both branches,
measured end to end (SENSENOVA_TRAINING_DESIGN.md 13.4 U-2-5). Three answers to
one question, and "unsupported" stopped meaning "the mechanism is not there".

THE FIX: `TRAINING_FEATURE_ADVISORY`, a fifth axis that says a feature IS
implemented and carries what it costs, with an import-time assert that no
(arch, feature) pair is in both it and `TRAINING_FEATURE_UNSUPPORTED` -- the same
shape as the fourth axis's own no-double-ownership assert.

The reason string's numbers are checked here too. The one that shipped read
"94.5% of a 48 GB card"; 94.5% is the ratio against the PROBE's self-imposed
`set_per_process_memory_fraction(0.72)` gate, and against the card the same peak
is 68%.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_capability_advisory_test.py -v
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import (  # noqa: E402
    TRAINING_ADVISORY_LEVELS,
    TRAINING_DECLARED_ARCHS,
    TRAINING_FEATURE_ADVISORY,
    TRAINING_FEATURE_LABELS,
    TRAINING_FEATURE_PARAMS,
    TRAINING_FEATURE_UNSUPPORTED,
    TRAINING_METHODS,
    training_feature_advisories,
    training_feature_unsupported_reason,
)
from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.model_loader import ModelLoader  # noqa: E402
from core.training.train_runner import (  # noqa: E402
    _apply_sensenova_training_contract,
)

CAPABILITIES_PY = BACKEND / "api" / "arch_capabilities.py"
API_TS = REPO / "frontend" / "src" / "utils" / "api.ts"
# The capability readers live beside the client; both files are the surface.
CAPS_TS = REPO / "frontend" / "src" / "utils" / "trainingCapabilities.ts"
TRAINING_CONFIG_TSX = REPO / "frontend" / "src" / "components" / "training" / "TrainingConfig.tsx"

# The commit this work started from -- the state the audit reviewed.
SHIPPED_COMMIT = "ce713b58"

# The U-2-5 both-branch measurement, from SENSENOVA_TRAINING_DESIGN.md 13.4.
PEAK_GIB = 32.6606
GATE_GIB = 34.551            # set_per_process_memory_fraction(0.72)
CARD_GIB = GATE_GIB / 0.72   # what the 48 GB card reports to torch


def _sensenova():
    return patch.object(ModelLoader, "detect_model_type", return_value="sensenova")


def _config(**overrides):
    keys = ("batch_size", "optimizer", "gradient_accumulation_steps", "use_ema",
            "num_optimizer_groups", "blocks_to_swap", "train_unet",
            "train_text_encoder", "sensenova_mot_phase_eviction",
            "sensenova_four_phase_eviction")
    config = {key: TRAINING_DEFAULTS[key] for key in keys}
    # The full-fine-tune contract's own pins, so a test that is about one flag
    # is not refused for an unrelated one.
    config.update({"optimizer": "adafactor", "batch_size": 1,
                   "gradient_accumulation_steps": 1, "use_ema": False})
    config.update(overrides)
    return config


def _git_show(path: str, commit: str = SHIPPED_COMMIT) -> str:
    try:
        result = subprocess.run(["git", "show", f"{commit}:{path}"],
                                cwd=REPO, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if result.returncode != 0:  # pragma: no cover
        pytest.skip(f"{commit} not in this clone")
    return result.stdout.decode("utf-8")


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS: the contradiction as it shipped
# ---------------------------------------------------------------------------

def test_negative_control_the_shipped_table_called_the_feature_unsupported():
    source = _git_show("backend/api/arch_capabilities.py")
    declaration = source[source.index('_add_training_feature_unsupported(\n    "sensenova", "text_encoder_training"'):]
    declaration = declaration[:declaration.index("\n\n")]
    assert 'methods=["full_finetune"]' in declaration
    # ... and the API side ran it anyway. Both halves of the contradiction, in
    # one assertion pair.
    assert "It is not enforced trainer-side: the API path" in source


def test_negative_control_the_api_accepted_the_parameter_the_table_denied():
    """Unchanged by this work, and that is the point: the REST contract always
    accepted `train_text_encoder` for a SenseNova full fine-tune. The table was
    the side that was wrong."""
    for source in (_git_show("backend/core/training/train_runner.py"),
                   (BACKEND / "core/training/train_runner.py").read_text(encoding="utf-8")):
        # Read as a live value, never refused: the flag selects a branch.
        assert "train_understanding = _normalize_sensenova_bool(" in source
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune", _config(train_text_encoder=True), {"sample": {}})


def test_negative_control_the_shipped_form_forced_the_flag_off():
    tsx = _git_show("frontend/src/components/training/TrainingConfig.tsx")
    assert "if (textEncoderTrainingUnsupported && params.train_text_encoder) {" in tsx
    assert 'disabled={!!textEncoderTrainingUnsupported}' in tsx


def test_negative_control_the_shipped_reason_named_the_wrong_denominator():
    source = _git_show("backend/api/arch_capabilities.py")
    assert "94.5% of a 48 GB card" in source
    # The arithmetic that makes it wrong: 94.5% is the gate, 68% is the card.
    assert round(PEAK_GIB / GATE_GIB * 100, 1) == 94.5
    assert round(PEAK_GIB / CARD_GIB * 100) == 68


# ---------------------------------------------------------------------------
# The fix: one answer, in all three places
# ---------------------------------------------------------------------------

def test_the_feature_is_no_longer_declared_unsupported():
    assert training_feature_unsupported_reason(
        "sensenova", "text_encoder_training", "full_finetune") is None
    assert "text_encoder_training" not in TRAINING_FEATURE_UNSUPPORTED.get("sensenova", {})


def test_the_advisory_answers_where_the_refusal_used_to():
    entry = training_feature_advisories("sensenova", "full_finetune")["text_encoder_training"]
    assert entry["level"] == "high_memory"
    # LoRA has always trained this branch and carries no memory advisory.
    assert "text_encoder_training" not in training_feature_advisories("sensenova", "lora")


@pytest.mark.parametrize("train_text_encoder", [True, False])
def test_the_capability_answer_matches_what_the_runner_does(train_text_encoder):
    """The invariant the contradiction broke: nothing may declare a feature
    absent while the pre-load contract accepts its arming parameter."""
    refused = training_feature_unsupported_reason(
        "sensenova", "text_encoder_training", "full_finetune") is not None
    with _sensenova():
        _apply_sensenova_training_contract(
            "model", "full_finetune",
            _config(train_text_encoder=train_text_encoder), {"sample": {}})
    assert not refused


def test_every_number_in_the_reason_names_its_denominator():
    reason = TRAINING_FEATURE_ADVISORY["sensenova"]["text_encoder_training"]["reason"]
    assert "94.5% of a 48 GB card" not in reason
    assert "set_per_process_memory_fraction(0.72)" in reason
    assert "34.551 GiB" in reason
    assert "32.66 GiB" in reason
    assert "68%" in reason
    # The conditions the peak was measured under, not just the peak.
    for condition in ("64px", "adafactor", "batch 1", "bf16"):
        assert condition in reason
    assert "U-2-5" in reason


def test_the_reason_matches_the_design_doc_measurement():
    doc = (REPO / "docs/guides/SENSENOVA_TRAINING_DESIGN.md").read_text(encoding="utf-8")
    assert "32.6606" in doc and "34.551 GiB" in doc
    assert "51.965" in doc and "61.67" in doc     # the two host RSS peaks
    reason = TRAINING_FEATURE_ADVISORY["sensenova"]["text_encoder_training"]["reason"]
    assert "51.97-61.67 GiB host RSS" in reason


# ---------------------------------------------------------------------------
# The axis itself, and its partition assert
# ---------------------------------------------------------------------------

def test_the_table_names_known_archs_features_methods_and_levels():
    assert set(TRAINING_FEATURE_ADVISORY) <= TRAINING_DECLARED_ARCHS
    for arch, features in TRAINING_FEATURE_ADVISORY.items():
        for feature, entry in features.items():
            assert feature in TRAINING_FEATURE_PARAMS, (arch, feature)
            assert feature in TRAINING_FEATURE_LABELS, (arch, feature)
            assert entry["level"] in TRAINING_ADVISORY_LEVELS
            assert entry["reason"].strip()
            assert set(entry.get("methods", TRAINING_METHODS)) <= set(TRAINING_METHODS)


def test_no_pair_is_both_unsupported_and_advisory():
    for arch, features in TRAINING_FEATURE_ADVISORY.items():
        for feature in features:
            assert feature not in TRAINING_FEATURE_UNSUPPORTED.get(arch, {}), (arch, feature)


def test_advisories_are_empty_for_an_unknown_arch():
    assert training_feature_advisories("brand_new_arch", "lora") == {}
    assert training_feature_advisories(None) == {}


def _exec_patched_module(extra_declaration: str):
    """Re-execute arch_capabilities with one extra declaration spliced in.

    The asserts are module-level, so this is the only way to test that they
    fire: the anchor is the comment that opens the invariant block, and the
    declaration goes immediately before it.
    """
    source = CAPABILITIES_PY.read_text(encoding="utf-8")
    anchor = "\n# Coverage invariants"
    assert anchor in source
    patched = source.replace(anchor, f"\n{extra_declaration}{anchor}", 1)
    exec(compile(patched, "<patched arch_capabilities>", "exec"),
         {"__name__": "patched_arch_capabilities"})


def test_the_patch_harness_itself_is_sound():
    """A no-op splice must still import, or every assertion below would pass for
    the wrong reason."""
    _exec_patched_module("_UNUSED_HARNESS_PROBE = 1\n")


@pytest.mark.parametrize("declaration,match", [
    # The partition: the same pair in both tables.
    ('_add_training_feature_advisory("sensenova", "block_swap", "experimental", "x")\n',
     "both unsupported and advisory"),
    # A level outside the vocabulary.
    ('_add_training_feature_advisory("sensenova", "vae", "probably_fine", "x")\n',
     "unknown level"),
    # A feature name that is not a feature.
    ('_add_training_feature_advisory("sensenova", "no_such_feature", "experimental", "x")\n',
     "unknown feature"),
    # An architecture the registry does not have.
    ('_add_training_feature_advisory("brand_new_arch", "vae", "experimental", "x")\n',
     "undeclared archs"),
    # A method scope outside TRAINING_METHODS.
    ('_add_training_feature_advisory("sensenova", "vae", "experimental", "x", methods=["finetune"])\n',
     "scopes unknown"),
    # The fourth axis's mirror: a parameter cannot be pinned AND presented as a
    # choice with advice attached.
    ('_add_training_feature_advisory("sensenova", "training_samples", "experimental", "x")\n'
     '_add_training_required_value("sensenova", "sample_every", 0, "x")\n',
     "presents as a choice"),
])
def test_the_partition_assert_fires(declaration, match):
    with pytest.raises(AssertionError, match=match):
        _exec_patched_module(declaration)


# ---------------------------------------------------------------------------
# The axis reaches the client
# ---------------------------------------------------------------------------

def test_the_capability_endpoint_serves_the_table_verbatim():
    import asyncio

    from api.routes import get_arch_capabilities

    payload = asyncio.run(get_arch_capabilities())
    assert payload["training_feature_advisory"] is TRAINING_FEATURE_ADVISORY


def test_the_openapi_spec_documents_the_served_key():
    import yaml

    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["ArchCapabilities"]["properties"]
    entry = schema["training_feature_advisory"]["additionalProperties"]["additionalProperties"]
    assert set(entry["required"]) == {"level", "reason"}
    assert set(entry["properties"]) == {"level", "reason", "methods"}
    assert set(entry["properties"]["level"]["enum"]) == set(TRAINING_ADVISORY_LEVELS)
    assert "training_feature_advisory" in (
        spec["paths"]["/schema/arch-capabilities"]["get"]["description"])


def test_the_client_reads_the_served_table_and_holds_no_copy():
    ts = (API_TS.read_text(encoding="utf-8")
              + CAPS_TS.read_text(encoding="utf-8"))
    assert "caps?.training_feature_advisory?.[arch]" in ts
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    for entry in TRAINING_FEATURE_ADVISORY["sensenova"].values():
        assert entry["reason"] not in tsx


def test_the_form_shows_the_advisory_instead_of_disabling_the_control():
    """An advisory that disables its control is the contradiction again, with
    the table telling the truth and the form still lying."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    assert "textEncoderTrainingAdvisory" in tsx
    assert "trainingFeatureAdvisory(" in tsx
    # The checkbox's disabled/checked state is keyed on the REFUSAL only.
    assert "disabled={!!textEncoderTrainingUnsupported}" in tsx
    assert "disabled={!!textEncoderTrainingAdvisory" not in tsx
    assert "textEncoderTrainingAdvisory && params.train_text_encoder" not in tsx
