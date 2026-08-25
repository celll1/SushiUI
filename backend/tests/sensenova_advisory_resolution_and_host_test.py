"""Three findings of an external audit against `9937bf54`, and what closes them.

1. The capability advisories quoted numbers the resolution campaign (`8aef7a6a`)
   had already superseded. `text_encoder_training` still said nothing above 64px
   had been measured; `sensenova_mot_eviction` quoted "1.09-1.10x" as THE cost of
   the four-phase route, when that ratio is the graph split ALONE -- measured at
   1024px with understanding gradients supplied by a rank-4 LoRA over int8 halves,
   with the prefix recompute isolated. What a both-branch full fine-tune pays is
   the split PLUS the weight round trips it makes possible: 42.67 s -> 80.51 s
   over 12 steps at 512px, 1.89x.

2. The training form's four-phase gate checked three of the backend's four
   preconditions. `9937bf54` restricted full-fine-tune eviction to the both-halves
   branch, so an understanding-only configuration could still tick the box and be
   refused after submit.

3. Host-side requirements were recorded nowhere a user could see them, and at
   48 GB of VRAM they are the binding constraint.

Every figure asserted here is READ OUT OF `SENSENOVA_TRAINING_DESIGN.md` 8.3.3 --
the measurement matrix, the A/B paragraph, the host non-reproduction box -- and
compared with the advisory, so a literal written twice cannot make this pass.
The recommendations derived from those measurements are checked as advice: they
must be marked as such and must not be phrased as measurements.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_advisory_resolution_and_host_test.py -v
"""

import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import TRAINING_FEATURE_ADVISORY  # noqa: E402
from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.model_loader import ModelLoader  # noqa: E402
from core.training.train_runner import (  # noqa: E402
    _apply_sensenova_training_contract,
)

DESIGN_DOC = REPO / "docs/guides/SENSENOVA_TRAINING_DESIGN.md"
TRAINING_CONFIG_TSX = REPO / "frontend" / "src" / "components" / "training" / "TrainingConfig.tsx"
TRAINING_MONITOR_TSX = REPO / "frontend" / "src" / "components" / "training" / "TrainingMonitor.tsx"

# The state the audit reviewed.
SHIPPED_COMMIT = "9937bf54"


def _doc() -> str:
    return DESIGN_DOC.read_text(encoding="utf-8")


def _section_833() -> str:
    doc = _doc()
    start = doc.index("### 8.3.3")
    return doc[start:doc.index("### 8.4", start)]


def _advisory(feature: str) -> str:
    return TRAINING_FEATURE_ADVISORY["sensenova"][feature]["reason"]


def _floats(text: str):
    return [float(m) for m in re.findall(r"\d+\.\d+", text.replace(",", ""))]


def _matrix():
    """arm id -> {res, four_phase, load, step} from 8.3.3's measurement table.

    `step` is the steady-state figure: the float before 定常 where the cell
    carries one ("32.6606 (step 1) -> 18.7607 定常"), otherwise the first float
    (later ones are the cell's own commentary, e.g. "gate の 98.2%").
    """
    rows = {}
    for line in _section_833().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 8 or not re.fullmatch(r"[A-D]\d", cells[0]):
            continue
        cell = cells[5]
        step = _floats(cell.split("定常")[0] if "定常" in cell else cell)
        if "定常" not in cell:
            step = step[:1]
        rows[cells[0]] = {
            "branch": cells[1],
            "res": int(cells[2]),
            "four_phase": "on" in cells[3],
            "load": _floats(cells[4])[0],
            "step": step[-1] if step else None,
            "step_raw": cells[5],
        }
    assert {"C1", "A1", "A2", "B1", "B2", "B3", "B4"} <= set(rows), rows.keys()
    return rows


def _gib(value: float) -> str:
    return f"{value:.2f}"


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
# The doc is the source. If these fail, the section moved and every assertion
# below is asserting against nothing.
# ---------------------------------------------------------------------------

def test_the_measurement_matrix_is_where_the_advisories_say_it_is():
    rows = _matrix()
    assert rows["A1"]["res"] == 512 and rows["A1"]["branch"] == "gen"
    assert rows["B1"]["branch"] == "both" and not rows["B1"]["four_phase"]
    assert rows["B4"]["four_phase"] and rows["B4"]["res"] == 1024
    # B3 has no step figure: it OOMed.
    assert "OOM" in rows["B3"]["step_raw"]


# ---------------------------------------------------------------------------
# FINDING 1a -- resolution
# ---------------------------------------------------------------------------

def test_negative_control_the_advisory_denied_measurements_that_existed():
    """`8aef7a6a` measured 512 and 1024; `9937bf54`'s advisory still said no."""
    shipped = _git_show("backend/api/arch_capabilities.py")
    assert "Nothing above 64px has been measured" in shipped
    for arm in ("A1", "A2", "B1", "B4"):
        assert _gib(_matrix()[arm]["step"] or 0) not in shipped


def test_the_advisory_no_longer_denies_the_campaign():
    assert "Nothing above 64px has been measured" not in _advisory("text_encoder_training")


@pytest.mark.parametrize("arm,resolution", [
    ("A1", 512), ("A2", 1024),           # generation half alone
    ("B1", 512),                          # both halves, split off
    ("B2", 512), ("B4", 1024),            # both halves, split on
])
def test_each_resolution_figure_is_the_docs_own(arm, resolution):
    row = _matrix()[arm]
    assert row["res"] == resolution
    assert _gib(row["step"]) in _advisory("text_encoder_training"), (arm, row)


def test_the_1024_refusal_is_reported_as_the_probes_cap_not_the_card():
    """B3 was refused 192 MiB with 9.95 GiB free on the card. An advisory that
    said "does not fit in 48 GB" would be claiming something unmeasured."""
    reason = _advisory("text_encoder_training")
    section = _section_833()
    assert "192.00 MiB" in section and "9.95 GiB" in section
    assert "192 MiB" in reason and "9.95 GiB" in reason
    assert _gib(_matrix()["B3"]["step"]) in reason   # where it was refused
    assert "34.55" in reason
    assert "exceeds 34.55 GiB" in reason


def test_the_advisory_keeps_the_reserved_caveat_the_doc_insists_on():
    """8.3.3: the split lowers what a STEP needs, not what the process holds."""
    assert "33.9" in _section_833() and "34.4" in _section_833()
    reason = _advisory("text_encoder_training")
    assert "33.9-34.4 GiB" in reason
    assert "not what the process holds" in reason


def test_the_advisory_refuses_to_extrapolate_where_the_doc_does():
    reason = _advisory("text_encoder_training")
    assert "unmeasured" in reason and "superlinear" in reason
    # The und branch at 512/1024 is NOT measured (8.3.3) -- but 64px IS
    # (U-2-5, 26.2571 GiB), so "at any resolution" was the overstatement this
    # assertion used to pin. It must say ABOVE 64px and carry the value.
    assert "understanding half alone above 64px" in reason
    assert "26.26 GiB" in reason
    assert "at any resolution are unmeasured" not in reason


# ---------------------------------------------------------------------------
# FINDING 1b -- the split ratio is not the eviction ratio
# ---------------------------------------------------------------------------

def _wall_clock_ab():
    """The 512px four-phase A/B: (off_seconds, on_seconds, ratio)."""
    para = _section_833()
    match = re.search(r"(\d+\.\d+) s → (\d+\.\d+) s = (\d+\.\d+) 倍", para)
    assert match, "8.3.3's wall-clock A/B sentence moved"
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def test_negative_control_the_shipped_advisory_quoted_only_the_split_ratio():
    shipped = _git_show("backend/api/arch_capabilities.py")
    assert "1.09-1.10x step; it adds no weight transfer" in shipped
    off, on, ratio = _wall_clock_ab()
    for figure in (f"{off:.2f}", f"{on:.2f}", f"{ratio:.2f}x"):
        assert figure not in shipped, figure


def test_both_ratios_are_present_and_named_as_different_things():
    reason = _advisory("sensenova_mot_eviction")
    off, on, ratio = _wall_clock_ab()
    assert "1.09-1.10x" in reason          # the split alone
    assert f"{ratio:.2f}x" in reason       # eviction included
    assert f"{off:.2f} s" in reason and f"{on:.2f} s" in reason
    assert "SPLIT alone" in reason
    assert "eviction included" in reason


def test_the_split_ratios_conditions_are_stated_because_they_are_not_the_users():
    """1.09-1.10x came off a LoRA arm over int8 halves; the route it is quoted
    for is a bf16 both-branch full fine-tune."""
    reason = _advisory("sensenova_mot_eviction")
    assert "rank-4" in reason and "int8 halves" in reason
    assert "467-token prefix" in reason
    assert "n=25, p50" in reason
    # And the transfer volume the full-fine-tune route actually moves.
    doc = _doc()
    assert "15.09" in doc and "7.60 GiB" in doc
    assert "7.60 GiB int8 half" in reason and "15.09 GiB" in reason


def test_what_the_eviction_buys_is_stated_with_the_same_provenance():
    rows = _matrix()
    reason = _advisory("sensenova_mot_eviction")
    assert _gib(rows["B1"]["step"]) in reason      # 512px, split off
    assert _gib(rows["B2"]["step"]) in reason      # 512px, split on
    assert _gib(rows["B4"]["step"]) in reason      # 1024px, split on
    assert "34.551 GiB cap" in reason


# ---------------------------------------------------------------------------
# The two strings stay byte-identical to what openapi advertises
# ---------------------------------------------------------------------------

def _openapi_advisory_example():
    import yaml

    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    return (spec["components"]["schemas"]["ArchCapabilities"]["properties"]
            ["training_feature_advisory"]["example"])


def test_negative_control_nothing_pinned_the_example_to_the_served_string():
    """The brief said a test pins their equality. It did not exist: the parity
    suite pins `unsupported` reasons and the arch lists, and the advisory suite
    pins the schema's shape -- neither compares the advisory EXAMPLE text. So
    both copies drifted together and no test noticed."""
    for name in ("quantized_capability_parity_test.py",
                 "sensenova_capability_advisory_test.py",
                 "sensenova_four_phase_ui_exposure_test.py"):
        shipped = _git_show(f"backend/tests/{name}")
        assert "training_feature_advisory\"]" not in shipped or \
            "[\"example\"]" not in shipped.split("training_feature_advisory")[1][:400]
    # And they were in fact equal while both were wrong.
    shipped_example = _git_show("openapi.yaml")
    assert "Nothing above 64px has been measured" in shipped_example


@pytest.mark.parametrize("feature", ["text_encoder_training", "sensenova_mot_eviction"])
def test_the_openapi_example_is_the_served_string_byte_for_byte(feature):
    entry = _openapi_advisory_example()["sensenova"][feature]
    served = TRAINING_FEATURE_ADVISORY["sensenova"][feature]
    assert entry["reason"] == served["reason"], (
        "openapi's training_feature_advisory example has drifted from "
        "arch_capabilities.py")
    assert entry["level"] == served["level"]
    assert list(entry.get("methods", [])) == list(served.get("methods", []))


def test_the_four_phase_parameter_description_carries_the_eviction_ratio_too():
    import yaml

    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    description = props["sensenova_four_phase_eviction"]["description"]
    off, on, ratio = _wall_clock_ab()
    assert "1.09-1.10x" in description
    assert f"{ratio:.2f}x" in description
    assert f"{off:.2f} s" in description and f"{on:.2f} s" in description


# ---------------------------------------------------------------------------
# FINDING 2 -- the four-phase gate did not check train_unet
# ---------------------------------------------------------------------------

def _blocked_reason_source(tsx: str) -> str:
    body = tsx[tsx.index("const fourPhaseBlockedReason"):]
    return body[:body.index("undefined;")]


def test_negative_control_the_form_offered_a_configuration_the_backend_refuses():
    shipped = _git_show("frontend/src/components/training/TrainingConfig.tsx")
    reason = _blocked_reason_source(shipped)
    assert "trainUnet" not in reason
    # ... while the runner refused exactly that shape, before the load.
    with _sensenova():
        with pytest.raises(ValueError, match="requires both train_unet and"):
            _apply_sensenova_training_contract(
                "model", "full_finetune",
                _config(train_unet=False, train_text_encoder=True,
                        sensenova_mot_phase_eviction=True,
                        sensenova_four_phase_eviction=True), {"sample": {}})


def test_the_gate_now_names_every_precondition_including_the_branch():
    reason = _blocked_reason_source(TRAINING_CONFIG_TSX.read_text(encoding="utf-8"))
    assert 'trainingMethod !== "full_finetune"' in reason
    assert "!trainTextEncoder" in reason
    assert "!params.sensenova_mot_phase_eviction" in reason
    assert "!trainUnet" in reason


def test_the_branch_refusal_is_shown_where_the_backend_raises_it():
    """The eviction flag itself is what the runner refuses on a single branch,
    so the note belongs beside that checkbox and must not disable it (LoRA and
    the both-halves branch are both legitimate)."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    note = tsx[tsx.index("const motEvictionBranchRefusal"):]
    note = note[:note.index(": undefined;")]
    assert 'trainingMethod === "full_finetune"' in note
    assert "trainUnet && trainTextEncoder" in note
    assert "{motEvictionBranchRefusal && (" in tsx
    assert "disabled={!!motEvictionBranchRefusal" not in tsx


def test_the_clearing_effect_converges_on_the_branch_flags_by_value():
    """Same trap the fourth axis hit in this file: a preset writes `train_unet`
    without touching arch or method. `fourPhaseBlockedReason` is a recomputed
    string and so is value-keyed, but the flags are pinned here explicitly, and
    the dependency list -- not the effect body -- is the part that fixes it."""
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    body = tsx[tsx.index("if (motEvictionUnsupported) {"):]
    deps = body[body.index("}, ["):body.index("]);") + 3]
    assert "fourPhaseBlockedReason" in deps
    assert "params.train_unet" in deps
    assert "params.train_text_encoder" in deps


def test_the_designed_shape_is_still_accepted():
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune",
            _config(train_text_encoder=True, sensenova_mot_phase_eviction=True,
                    sensenova_four_phase_eviction=True), {"sample": {}})


# ---------------------------------------------------------------------------
# FINDING 3 -- host requirements
# ---------------------------------------------------------------------------

def _host_commits():
    match = re.search(r"commit が (\d+\.\d+) と (\d+\.\d+) GiB", _section_833())
    assert match, "8.3.3's host non-reproduction box moved"
    return float(match.group(1)), float(match.group(2))


def test_negative_control_the_host_figures_reached_no_user_surface():
    shipped_caps = _git_show("backend/api/arch_capabilities.py")
    low, high = _host_commits()
    for figure in (f"{low:.2f}", f"{high:.2f}", "32.68", "17.59"):
        assert figure not in shipped_caps, figure
    shipped_doc = _git_show("docs/guides/SENSENOVA_TRAINING_DESIGN.md")
    assert "host 側の所要量" not in shipped_doc


def test_the_measured_host_commit_pair_is_quoted_as_a_pair():
    """One run gave 67.95 and another 89.10 for the same work. Quoting either
    alone is quoting a number that did not reproduce."""
    low, high = _host_commits()
    reason = _advisory("text_encoder_training")
    assert f"{low:.2f}" in reason and f"{high:.2f}" in reason
    assert "the larger is the bound" in reason
    assert "tens of GiB" in reason


def test_the_checkpoint_sizes_come_from_the_docs_byte_counts():
    doc = _doc()
    bf16 = re.search(r"35,091,856,594 B = (\d+\.\d+) GiB", doc)
    int8 = re.search(r"（\*\*(18,885,547,920) byte = (\d+\.\d+) GiB\*\*）", doc)
    assert bf16 and int8, "the save-format byte counts moved"
    reason = _advisory("text_encoder_training")
    assert f"{float(bf16.group(1)):.2f} GiB in bf16" in reason
    # The int8 number is still traceable to the same byte count, but it was
    # measured on a GEN-branch save (C1); the advisory must carry the value and
    # label the both-branch equality as the inference it is.
    assert f"the {float(int8.group(2)):.2f} GiB int8 figure" in reason
    assert "GENERATION-branch save" in reason
    assert "not a measurement" in reason


def test_the_recommendations_are_marked_as_advice_not_measurement():
    reason = _advisory("text_encoder_training")
    assert "ADVICE rather than measurement" in reason
    for clause in ("100 GiB", "110-120 GiB", "96 GiB", "150-300 GiB",
                   "no competing GPU process at 1024px"):
        assert clause in reason, clause
    # The recommended numbers must not be presented as measured ones.
    assert "recommends" in reason


def test_the_design_doc_separates_measured_from_recommended():
    section = _section_833()
    assert "#### host 側の所要量" in section
    block = section[section.index("#### host 側の所要量"):]
    assert "**推奨は監査の助言であって実測ではない。**" in block
    assert "**実測" in block and "**推奨" in block
    # And it points at the surface a user actually reads.
    assert "arch_capabilities.py" in block and "text_encoder_training" in block


# ---------------------------------------------------------------------------
# max_grad_norm under the fused route
# ---------------------------------------------------------------------------

def test_the_warning_now_reaches_the_user_so_the_form_is_the_only_gap():
    """`339790b5` gave trainer notices a channel and the monitor renders them,
    so the warning is delivered -- during the run. The form is what still
    presented 1.0 as if it were in effect."""
    protocol = (BACKEND / "api/WS_PROTOCOL.md").read_text(encoding="utf-8")
    assert "fused_grad_clipping_ignored" in protocol
    trainer = (BACKEND / "core/training/base_trainer.py").read_text(encoding="utf-8")
    assert 'code="fused_grad_clipping_ignored"' in trainer
    monitor = TRAINING_MONITOR_TSX.read_text(encoding="utf-8")
    assert "Trainer notices" in monitor


def test_negative_control_the_form_implied_clipping_on_every_route():
    shipped = _git_show("frontend/src/components/training/TrainingConfig.tsx")
    assert "Gradient clipping threshold. 0 disables clipping." in shipped
    assert "gradClippingIgnoredReason" not in shipped
    assert TRAINING_DEFAULTS["max_grad_norm"] == 1.0


def test_the_form_predicts_the_fused_route_the_trainer_takes():
    """Mirrors base_trainer.setup_optimizer: block swap with groups, block swap
    with a fused-capable optimizer, or a SenseNova full fine-tune."""
    from core.training.base_trainer import FUSED_BACKWARD_OPTIMIZERS

    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    body = tsx[tsx.index("const gradClippingIgnoredReason"):]
    body = body[:body.index("})();")]
    for optimizer in FUSED_BACKWARD_OPTIMIZERS:
        assert f'"{optimizer}"' in body, optimizer
    assert "numOptimizerGroups > 0" in body
    assert "blocksToSwap > 0" in body
    assert 'trainingMethod === "full_finetune"' in body
    # max_grad_norm=0 already means no clipping; no note is owed there.
    assert "(params.max_grad_norm ?? 1.0) <= 0" in body
    assert "{gradClippingIgnoredReason && (" in tsx


def test_the_form_does_not_conflate_adafactors_own_update_clipping():
    """Adafactor bounds the RMS of each parameter's update via clip_threshold
    (optimizers/adafactor_fused.py). That is not global gradient-norm clipping
    and must not be offered as a substitute for it."""
    adafactor = (BACKEND / "core/training/optimizers/adafactor_fused.py").read_text(encoding="utf-8")
    assert 'group["clip_threshold"]' in adafactor
    tsx = TRAINING_CONFIG_TSX.read_text(encoding="utf-8")
    body = tsx[tsx.index("const gradClippingIgnoredReason"):]
    body = body[:body.index("})();")]
    assert "clip_threshold" in body
    assert "a different mechanism" in body
    assert "No clipping of any kind happens" in body
