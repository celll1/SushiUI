"""`sensenova_four_phase_eviction` was reachable from REST and YAML and from
nowhere a user could see (`071e602b` says so in its own commit message).

Combined with the `text_encoder_training` capability entry, the shipped product
state was: generation-only full fine-tuning reachable from the UI; the both-half
four-phase route reachable only through the API; and the both-half route itself
hidden by a capability table that called it unsupported.

This file fixes the second half of that and pins the interlock. The three flags
are ONE setting, not three checkboxes, and the shape is asymmetric -- verified
here against `train_runner` and `ops/sensenova_ops` rather than assumed:

  * `train_text_encoder` alone (no eviction): fine, single backward, no split.
  * `train_text_encoder` + `sensenova_mot_phase_eviction`: REFUSED without the
    split -- the three-state evictor moves the trained half to CPU before its
    own backward.
  * the split: refused unless full fine-tuning, `train_text_encoder` and
    `sensenova_mot_phase_eviction` all hold.
  * eviction under full fine-tuning: refused unless `train_unet` AND
    `train_text_encoder` both hold (U-3, SENSENOVA_TRAINING_DESIGN.md 13.7 (5)).
    A single-branch full fine-tune materializes only the half it trains, and
    the evictor's symmetry rule refuses halves of different kinds. Measured in
    both directions on the real checkpoint; LoRA is exempt because it wraps
    rather than materializing, so both halves stay int8.

Together the last two mean full fine-tuning accepts exactly one eviction
shape: both halves plus the split.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_four_phase_ui_exposure_test.py -v
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
    TRAINING_DECLARED_ARCHS,
    TRAINING_FEATURE_PARAMS,
    training_feature_advisories,
    training_feature_unsupported_reason,
)
from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.model_loader import ModelLoader  # noqa: E402
from core.training.train_runner import _apply_sensenova_training_contract  # noqa: E402

API_TS = REPO / "frontend" / "src" / "utils" / "api.ts"
TRAINING_CONFIG_TSX = REPO / "frontend" / "src" / "components" / "training" / "TrainingConfig.tsx"
SHIPPED_COMMIT = "ce713b58"


def _sensenova():
    return patch.object(ModelLoader, "detect_model_type", return_value="sensenova")


def _config(**overrides):
    config = {
        "batch_size": 1,
        "optimizer": "adafactor",
        "gradient_accumulation_steps": 1,
        "use_ema": False,
        "num_optimizer_groups": 0,
        "blocks_to_swap": 0,
        "block_swap_h2d_only": False,
        "train_unet": True,
        "train_text_encoder": False,
        "sensenova_mot_phase_eviction": False,
        "sensenova_four_phase_eviction": False,
    }
    config.update(overrides)
    return config


def _git_show(path: str) -> str:
    try:
        result = subprocess.run(["git", "show", f"{SHIPPED_COMMIT}:{path}"],
                                cwd=REPO, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if result.returncode != 0:  # pragma: no cover
        pytest.skip(f"{SHIPPED_COMMIT} not in this clone")
    return result.stdout.decode("utf-8")


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: API-only, as it shipped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "frontend/src/utils/api.ts",
    "frontend/src/components/training/TrainingConfig.tsx",
])
def test_negative_control_the_flag_had_no_frontend_surface_at_all(path):
    assert "sensenova_four_phase_eviction" not in _git_show(path)


def test_negative_control_the_eviction_section_was_lora_only():
    tsx = _git_show("frontend/src/components/training/TrainingConfig.tsx")
    assert 'isSenseNovaModel(baseModelPath) && trainingMethod === "lora" && (' in tsx
    # While the backend accepted it for a full fine-tune the whole time, and the
    # split cannot be armed without it. Stated on the `both` branch: a
    # single-branch full fine-tune materializes one half and leaves the other
    # int8, which the evictor's symmetry rule refuses (measured on the real
    # checkpoint in both directions, U-3) -- so the shipped acceptance of a
    # gen-only eviction run was itself a load-then-die.
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune",
            _config(sensenova_mot_phase_eviction=True, train_text_encoder=True,
                    sensenova_four_phase_eviction=True),
            {"sample": {}})


# ---------------------------------------------------------------------------
# The interlock, read off the backend
# ---------------------------------------------------------------------------

def test_understanding_training_alone_needs_neither_eviction_nor_the_split():
    """The brief's premise, checked: `und` alone runs a single backward."""
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune",
            _config(train_unet=False, train_text_encoder=True), {"sample": {}})


def test_understanding_training_with_eviction_is_refused_without_the_split():
    with _sensenova():
        with pytest.raises(ValueError, match="cannot be combined with"):
            _apply_sensenova_training_contract(
                "model", "full_finetune",
                _config(train_text_encoder=True,
                        sensenova_mot_phase_eviction=True), {"sample": {}})


@pytest.mark.parametrize("branch", [
    {"train_unet": True, "train_text_encoder": False},
    {"train_unet": False, "train_text_encoder": True},
])
def test_single_branch_full_finetune_cannot_be_evicted(branch):
    """Measured on the real checkpoint (U-3), in both directions.

    ``select_mot_weight_modules(require_exact_symmetry=True)`` compares each
    layer's two halves by dtype and shape, and a single-branch full fine-tune
    materializes only the half it trains. Before this refusal the run paid the
    17.6 GiB load and the materialize and then raised about layer-0 mlp shapes.
    """
    with _sensenova():
        with pytest.raises(ValueError, match="requires both train_unet and"):
            _apply_sensenova_training_contract(
                "model", "full_finetune",
                _config(sensenova_mot_phase_eviction=True,
                        sensenova_four_phase_eviction=branch["train_text_encoder"],
                        **branch),
                {"sample": {}})


def test_lora_keeps_single_branch_eviction():
    """LoRA wraps rather than materializing, so both halves stay int8."""
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora",
            _config(sensenova_mot_phase_eviction=True), {"sample": {}})


@pytest.mark.parametrize("overrides,match", [
    ({"train_text_encoder": False, "sensenova_mot_phase_eviction": True},
     "requires train_text_encoder"),
    ({"train_text_encoder": True, "sensenova_mot_phase_eviction": False},
     "requires sensenova_mot_phase_eviction"),
])
def test_the_split_is_refused_without_each_precondition(overrides, match):
    config = _config(sensenova_four_phase_eviction=True, **overrides)
    with _sensenova():
        with pytest.raises(ValueError, match=match):
            _apply_sensenova_training_contract(
                "model", "full_finetune", config, {"sample": {}})


def test_the_split_is_refused_outside_full_fine_tuning():
    config = _config(sensenova_four_phase_eviction=True, train_text_encoder=True,
                     sensenova_mot_phase_eviction=True)
    with _sensenova():
        with pytest.raises(ValueError, match="requires\\s+.*full_finetune"):
            _apply_sensenova_training_contract("model", "lora", config, {"sample": {}})


def test_the_designed_combination_is_accepted():
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune",
            _config(train_text_encoder=True, sensenova_mot_phase_eviction=True,
                    sensenova_four_phase_eviction=True), {"sample": {}})


# ---------------------------------------------------------------------------
# The same three answers in the capability table
# ---------------------------------------------------------------------------

def test_the_two_flags_are_declared_as_one_feature():
    # Four keys now: the shared-window pair rides the same feature because it is
    # legal only on top of the split, which is legal only on top of eviction.
    assert TRAINING_FEATURE_PARAMS["sensenova_mot_eviction"] == [
        "sensenova_mot_phase_eviction", "sensenova_four_phase_eviction",
        "sensenova_four_phase_shared_prefix",
        "sensenova_four_phase_grad_reduction"]


def test_the_mechanism_is_declared_absent_everywhere_else():
    for arch in sorted(TRAINING_DECLARED_ARCHS - {"sensenova"}):
        assert training_feature_unsupported_reason(arch, "sensenova_mot_eviction"), arch
    assert training_feature_unsupported_reason("sensenova", "sensenova_mot_eviction") is None


@pytest.mark.parametrize("method", ["lora", "full_finetune"])
def test_the_advisory_states_the_interlock_for_both_methods(method):
    entry = training_feature_advisories("sensenova", method)["sensenova_mot_eviction"]
    assert entry["level"] == "experimental"
    for clause in ("train_text_encoder", "sensenova_mot_phase_eviction",
                   "sensenova_four_phase_eviction", "full_finetune"):
        assert clause in entry["reason"]


@pytest.mark.parametrize("method", ["lora", "full_finetune"])
def test_the_advisory_does_not_call_plain_eviction_the_unconstrained_flag(method):
    """A user-visible description that outran the implementation (U-3, H-1).

    It read "sensenova_mot_phase_eviction ... is available under LoRA and full
    fine-tuning", framing the SPLIT as the constrained one. Under full fine
    tuning that is backwards: plain eviction is refused for every branch but
    `both`, which then requires the split anyway.
    """
    reason = training_feature_advisories(
        "sensenova", method)["sensenova_mot_eviction"]["reason"]
    assert "is available under LoRA and full fine-tuning" not in reason
    assert "train_unet" in reason
    # The claim, not the section number it used to cite: that write-up was
    # condensed out of the design doc, and pinning a dead pointer pins nothing.
    assert "measured in both directions on the real checkpoint" in reason


def test_the_openapi_description_no_longer_calls_eviction_lora_only():
    import yaml

    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    properties = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    description = properties["sensenova_mot_phase_eviction"]["description"]
    assert "LoRA training only" not in description
    assert "full fine-tuning" in description
    # And the refusal a caller meets if it combines the two without the split.
    assert "sensenova_four_phase_eviction" in description


# ---------------------------------------------------------------------------
# The UI surface
# ---------------------------------------------------------------------------

def test_the_flag_has_a_frontend_type_and_a_default():
    assert "sensenova_four_phase_eviction?: boolean;" in API_TS.read_text(encoding="utf-8")
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    assert "sensenova_four_phase_eviction: false," in tsx
    assert TRAINING_DEFAULTS["sensenova_four_phase_eviction"] is False


def test_the_flag_is_submitted_saved_and_restored():
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    # Request payload.
    assert "sensenova_four_phase_eviction: params.sensenova_four_phase_eviction," in tsx
    # YAML round trip and preset round trip -- a control that submits but does
    # not persist is a setting the user loses on the next run.
    assert '"sensenova_four_phase_eviction", "sensenova_four_phase_shared_prefix",' in tsx
    assert '"sensenova_four_phase_grad_reduction", "sensenova_full_finetune_save_format",' in tsx
    # The preset payload is derived from getRequestData() minus
    # PRESET_EXCLUDED_KEYS, so the two assertions above cover it too; the flag
    # is not excluded. training_preset_payload_test.py owns that gate.
    assert "sensenova_four_phase_eviction" not in tsx[
        tsx.index("const PRESET_EXCLUDED_KEYS"):tsx.index("const PRESET_RESTORABLE_KEYS")]


def test_the_control_exists_and_states_each_precondition_the_backend_checks():
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    assert 'id="sensenova-four-phase-eviction"' in tsx
    reason = tsx[tsx.index("const fourPhaseBlockedReason"):]
    reason = reason[:reason.index("undefined;")]
    assert 'trainingMethod !== "full_finetune"' in reason
    assert "!trainTextEncoder" in reason
    assert "!params.sensenova_mot_phase_eviction" in reason


def test_the_eviction_section_is_offered_for_both_methods_from_the_table():
    """Not `method === "lora"` alone: that is what kept the control off the
    full-fine-tune form the backend accepts it on. The arch term stays, because
    `trainingFeatureUnsupportedReason` answers "supported" for an unknown arch
    and for the window before the capability matrix loads -- without it a
    SenseNova-only section renders over an SDXL base model."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    assert 'isSenseNovaModel(baseModelPath) && trainingMethod === "lora" && (' not in tsx
    assert ('{isSenseNovaModel(baseModelPath) && !motEvictionUnsupported && '
            '(trainingMethod === "lora" || trainingMethod === "full_finetune") && (') in tsx


def test_the_form_clears_the_split_when_a_precondition_goes_away():
    """Otherwise the form submits a run refused before the model loads."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    assert ("if (fourPhaseBlockedReason && params.sensenova_four_phase_eviction) {"
            in tsx)
    assert 'if (!e.target.checked) updateParam("sensenova_four_phase_eviction", false);' in tsx


def test_the_clearing_effect_converges_on_value_drift_not_identity():
    """The trap `9e84ac19`'s audit caught for the fourth axis, in the same file.

    `fourPhaseBlockedReason` does not move when only the split itself is
    written, so with an identity-keyed dependency list a preset or a
    copy-from-run that carries `sensenova_four_phase_eviction: true` -- and no
    `training_method` -- parks it true inside a control that is not rendered,
    and every submit is refused pre-load with no way back but toggling the
    method radio. A string match on the effect body passes either way; the
    dependency list is the part that fixes it."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    body = tsx[tsx.index("if (motEvictionUnsupported) {"):]
    deps = body[body.index("}, ["):body.index("]);") + 3]
    assert "params.sensenova_four_phase_eviction" in deps
    assert "params.sensenova_mot_phase_eviction" in deps


def test_the_pair_refusal_is_shown_under_lora_too_with_its_own_remedy():
    """`train_runner` refuses train_text_encoder + eviction under BOTH methods.
    Under LoRA the split cannot lift it (it is full-fine-tune only), so the note
    must not tell a LoRA user to enable it."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    note = tsx[tsx.index("const evictionPairRefusal"):]
    note = note[:note.index(": undefined;")]
    # Not gated on the method: computed for both, worded per method.
    assert 'trainingMethod === "full_finetune"' in note
    assert "implemented for full fine-tuning only" in note
    # Rendered outside the full-fine-tune-only block.
    section = tsx[tsx.index('id="sensenova-mot-phase-eviction"'):]
    section = section[:section.index("SenseNova Checkpoint Format")]
    full_ft_block = section[section.index('{trainingMethod === "full_finetune" && ('):]
    assert "{evictionPairRefusal && (" in section
    assert "{evictionPairRefusal && (" not in full_ft_block[:full_ft_block.index("</>")]

    with _sensenova():
        with pytest.raises(ValueError, match="cannot be combined with"):
            _apply_sensenova_training_contract(
                "model", "lora",
                _config(train_text_encoder=True,
                        sensenova_mot_phase_eviction=True), {"sample": {}})


def test_the_form_starts_sensenova_at_the_backends_own_flag_default():
    """A `train_text_encoder: true` carried over from another architecture would
    otherwise make the run the both-half configuration without anyone choosing
    it. Starting value only -- the box stays checkable, with the advisory."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    branch = tsx[tsx.index('if (arch === "sensenova") {'):]
    assert 'updateParam("train_text_encoder", false);' in branch[:branch.index("}")]
    assert TRAINING_DEFAULTS["train_text_encoder"] is False


def test_the_forms_blocked_reasons_name_the_same_clauses_the_runner_refuses_on():
    """The table shape cannot express a conditional interlock, so the form
    restates three refusals that live in `train_runner`. Nothing but this ties
    the two texts together: a clause renamed on one side must fail here."""
    runner = (BACKEND / "core/training/train_runner.py").read_text(encoding="utf-8")
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    form = tsx[tsx.index("const fourPhaseBlockedReason"):]
    form = form[:form.index("const evictionPairRefusal")]
    for clause in ("full_finetune", "train_text_encoder",
                   "sensenova_mot_phase_eviction"):
        assert clause in runner
        # The form spells them as labels, so match the label the user reads.
        assert {
            "full_finetune": "full fine-tuning only",
            "train_text_encoder": "Train Text Encoder",
            "sensenova_mot_phase_eviction": "MoT Phase Eviction",
        }[clause] in form
