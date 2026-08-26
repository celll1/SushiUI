"""Two user-reachable defects opened by `b2694674`, and their fixes.

1. The training form offered a method whose shipped defaults could not run.
   SenseNova refuses `optimizer != adafactor` under full fine-tuning and
   `batch_size != 1` under every method, from the config and before the model
   loads -- while `TRAINING_DEFAULTS["optimizer"]` is `adamw8bit`. The fix is a
   fourth capability axis, `TRAINING_REQUIRED_VALUES`: what a parameter must BE,
   next to the three tables that say what is missing. It is served by
   `GET /schema/arch-capabilities`; the form pins the control to it and shows
   the backend's reason. The frontend holds no copy of the values.

2. `train_unet` reached `FullParameterTrainer` and not `LoRATrainer`, so one
   checkbox governed the full-FT run and was inert for a LoRA one. Forwarding it
   makes "train nothing" reachable on the LoRA path too, which is refused here
   for every architecture rather than left to the optimizer.

NEGATIVE CONTROLS (what shipped, recorded so the fix is provably a change):
`test_negative_control_*`.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_required_values_test.py -v
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import (  # noqa: E402
    TRAINING_DECLARED_ARCHS,
    TRAINING_METHODS,
    TRAINING_REQUIRED_VALUES,
    training_required_values,
)
from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.model_loader import ModelLoader  # noqa: E402
from core.training.ops.sensenova_ops import (  # noqa: E402
    SENSENOVA_FULL_FINETUNE_OPTIMIZERS,
)
from core.training.train_runner import (  # noqa: E402
    _apply_sensenova_training_contract,
    _assert_training_scope_is_nonempty,
)

RUNNER_PY = BACKEND / "core" / "training" / "train_runner.py"
API_TS = REPO / "frontend" / "src" / "utils" / "api.ts"


def _sensenova():
    return patch.object(ModelLoader, "detect_model_type", return_value="sensenova")


def _shipped_train_config(**overrides):
    """The `train` section a run created from the shipped defaults produces."""
    keys = ("batch_size", "optimizer", "gradient_accumulation_steps", "use_ema",
            "num_optimizer_groups", "blocks_to_swap", "train_unet",
            "train_text_encoder")
    config = {key: TRAINING_DEFAULTS[key] for key in keys}
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# (1) NEGATIVE CONTROLS: the shipped defaults, refused
# ---------------------------------------------------------------------------

def test_negative_control_shipped_defaults_are_refused_for_a_full_finetune():
    """The optimizer default is `adamw8bit` and this route allows only
    Adafactor, so the first full fine-tune a user starts is rejected."""
    with _sensenova():
        with pytest.raises(ValueError, match="does not support optimizer"):
            _apply_sensenova_training_contract(
                "model", "full_finetune", _shipped_train_config(), {"sample": {}}
            )


def test_negative_control_the_stale_frontend_batch_size_is_refused_for_lora_too():
    """`batch_size: 4` was the frontend literal. It is refused under EVERY
    SenseNova method, not only full fine-tuning -- so the same defect reached
    the LoRA form. (The backend SSOT already said 1; the literal was stale.)"""
    assert TRAINING_DEFAULTS["batch_size"] == 1
    for method in ("lora", "full_finetune"):
        with _sensenova():
            with pytest.raises(ValueError, match="requires batch_size=1"):
                _apply_sensenova_training_contract(
                    "model", method, _shipped_train_config(batch_size=4),
                    {"sample": {}}
                )


# ---------------------------------------------------------------------------
# (1) The table, and that it is the contract rather than a second copy of it
# ---------------------------------------------------------------------------

def test_the_required_values_are_what_the_runner_actually_enforces():
    """Applying every declared value makes the contract pass, for both methods
    SenseNova implements."""
    for method in ("lora", "full_finetune"):
        config = _shipped_train_config()
        config.update({param: entry["value"] for param, entry
                       in training_required_values("sensenova", method).items()})
        with _sensenova():
            assert _apply_sensenova_training_contract(
                "model", method, config, {"sample": {}}
            )


@pytest.mark.parametrize("method,param,wrong", [
    ("lora", "batch_size", 2),
    ("lora", "train_unet", False),
    ("full_finetune", "batch_size", 2),
    ("full_finetune", "optimizer", "adamw8bit"),
    ("full_finetune", "gradient_accumulation_steps", 4),
    ("full_finetune", "use_ema", True),
])
def test_every_declared_requirement_is_a_refusal_not_a_recommendation(
    method, param, wrong
):
    assert param in training_required_values("sensenova", method)
    config = _shipped_train_config()
    config.update({p: e["value"] for p, e
                   in training_required_values("sensenova", method).items()})
    config[param] = wrong
    with _sensenova():
        with pytest.raises(ValueError):
            _apply_sensenova_training_contract(
                "model", method, config, {"sample": {}}
            )


_A_DIFFERENT_VALUE = {
    "batch_size": 2,
    "optimizer": "adamw8bit",
    "gradient_accumulation_steps": 4,
    "use_ema": True,
    "train_unet": False,
    "text_encoding_mode": "pre_encoded_cache",
    "latent_encoding_mode": "pre_encoded_cache",
}


@pytest.mark.parametrize("method", ["lora", "full_finetune"])
def test_every_declared_entry_is_enforced_the_way_its_reason_says(method):
    """The axis carries both enforcement shapes, so the `reason` has to name
    which one applies -- an OVERWRITE that reads as a refusal would tell a user
    their value was rejected when it was silently replaced, and vice versa."""
    declared = training_required_values("sensenova", method)
    for param, entry in declared.items():
        config = _shipped_train_config()
        config.update({p: e["value"] for p, e in declared.items()})
        config[param] = _A_DIFFERENT_VALUE[param]
        try:
            with _sensenova():
                _apply_sensenova_training_contract(
                    "model", method, config, {"sample": {}})
        except ValueError:
            refused = True
        else:
            refused = False
            assert config[param] == entry["value"], (
                f"{param} is declared but neither refused nor overwritten")
        assert refused is ("overwritten" not in entry["reason"]), (
            f"{param}: reason says "
            f"{'overwrite' if not refused else 'refusal'} but the runner does "
            f"the other one")


@pytest.mark.parametrize("param", ["text_encoding_mode", "latent_encoding_mode"])
def test_the_two_encoding_modes_are_the_overwritten_pair(param):
    """train_runner rewrites both for EVERY SenseNova run. Undeclared, the form
    kept offering two live selects the run discarded without saying so -- the
    same silent-override failure this axis exists to prevent."""
    entry = training_required_values("sensenova", "lora")[param]
    assert entry["value"] == "onthefly_gpu"
    assert TRAINING_DEFAULTS[param] != "onthefly_gpu"   # so the form really did differ
    config = _shipped_train_config(**{param: TRAINING_DEFAULTS[param]})
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora", config, {"sample": {}})
    assert config[param] == "onthefly_gpu"


def test_the_declared_optimizer_set_is_the_allowlist_the_trainer_holds():
    """The same set, not a second opinion.

    MUTANT: add a name to SENSENOVA_FULL_FINETUNE_OPTIMIZERS without adding it
    to the table and the form offers a shorter list than the run accepts; add it
    to the table only and the form offers a value refused before the load.
    """
    entry = training_required_values("sensenova", "full_finetune")["optimizer"]
    assert entry["values"] == list(SENSENOVA_FULL_FINETUNE_OPTIMIZERS)
    # `value` is the default member of that set, not a fourth opinion.
    assert entry["value"] in SENSENOVA_FULL_FINETUNE_OPTIMIZERS
    # And the condition the two ring-buffer members carry is stated where the
    # form shows it, since selecting one without the flag is refused.
    assert "optimizer_state_host_resident" in entry["reason"]


@pytest.mark.parametrize("optimizer", ["adafactor", "adamw8bit_ringbuffer",
                                       "lion8bit_ringbuffer"])
def test_every_offered_optimizer_is_actually_accepted(optimizer):
    """A value in `values` the run refuses is the failure this axis prevents.

    The ring-buffer pair carries the residency condition its `reason` names;
    with the flag set every offered value passes the pre-load contract.
    """
    config = _shipped_train_config(optimizer=optimizer,
                                   optimizer_state_host_resident=True)
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune", config, {"sample": {}})


def test_method_scoping_narrows_and_an_unknown_arch_is_unconstrained():
    lora = training_required_values("sensenova", "lora")
    full = training_required_values("sensenova", "full_finetune")
    assert "optimizer" not in lora and "optimizer" in full
    assert "train_unet" in lora and "train_unet" not in full
    assert "batch_size" in lora and "batch_size" in full   # unscoped
    # Absent means unconstrained, in both directions.
    assert training_required_values("sdxl", "lora") == {}
    assert training_required_values(None, "lora") == {}
    assert training_required_values("brand_new_arch", "full_finetune") == {}
    # No method given -> the whole entry, so a caller cannot silently drop a
    # scoped requirement by forgetting the argument.
    assert "optimizer" in training_required_values("sensenova")


def test_the_table_names_known_archs_and_known_methods():
    assert set(TRAINING_REQUIRED_VALUES) <= TRAINING_DECLARED_ARCHS
    for arch, params in TRAINING_REQUIRED_VALUES.items():
        for param, entry in params.items():
            assert set(entry.get("methods", TRAINING_METHODS)) <= set(TRAINING_METHODS)
            assert entry["reason"].strip()
            # JSON-serialisable: the whole point is that a client can pin a
            # control to it.
            json.dumps(entry["value"])
            # And it must be a real training parameter, not a spelling of one.
            assert param in TRAINING_DEFAULTS, param


# ---------------------------------------------------------------------------
# (1) The table reaches the client, and only from the backend
# ---------------------------------------------------------------------------

def test_the_capability_endpoint_serves_the_table_verbatim():
    """The frontend has no copy of these values -- if the route stops serving
    the key, every control unpins rather than pinning to a stale literal."""
    import asyncio

    from api.routes import get_arch_capabilities

    payload = asyncio.run(get_arch_capabilities())
    assert payload["training_required_values"] is TRAINING_REQUIRED_VALUES
    assert payload["training_required_values"]["sensenova"]["optimizer"]["value"] \
        == "adafactor"


def test_the_openapi_spec_documents_the_served_key():
    import yaml

    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["ArchCapabilities"]["properties"]
    entry = schema["training_required_values"]["additionalProperties"][
        "additionalProperties"]
    assert set(entry["required"]) == {"value", "reason"}
    assert set(entry["properties"]) == {"value", "reason", "methods", "values"}
    assert "training_required_values" in (
        spec["paths"]["/schema/arch-capabilities"]["get"]["description"])


@pytest.mark.parametrize("path", [
    "frontend/src/utils/api.ts",
    "frontend/src/components/training/TrainingConfig.tsx",
])
def test_the_frontend_holds_no_copy_of_the_values(path):
    """The whole point of the axis: the values live in one place. A copy in
    either file is a second source that goes stale silently."""
    text = (REPO / path).read_text(encoding="utf-8")
    assert "TRAINING_REQUIRED_VALUES_FALLBACK" not in text
    for entry in TRAINING_REQUIRED_VALUES["sensenova"].values():
        assert entry["reason"] not in text


def test_the_client_reads_the_served_table_and_nothing_else():
    text = API_TS.read_text(encoding="utf-8")
    assert "caps?.training_required_values?.[arch]" in text


def test_the_form_pins_every_declared_parameter_from_the_table():
    """Every parameter the backend declares has a pinned control -- a declared
    value with no control is a knob the run still discards."""
    tsx = (REPO / "frontend/src/components/training/TrainingConfig.tsx").read_text(
        encoding="utf-8")
    assert "trainingRequiredValues(" in tsx
    for param in TRAINING_REQUIRED_VALUES["sensenova"]:
        assert f'requiredValue("{param}")' in tsx, param
    # The adjustment is announced, and names what it replaced.
    assert "contractAdjusted" in tsx
    assert "(changed from {contractAdjusted[param]})" in tsx


def test_the_pin_converges_on_value_drift_not_on_arch_method_identity():
    """A preset writes batch_size/optimizer/train_unet through updateParam
    without touching arch or method. An effect keyed on the requirement set's
    IDENTITY never re-runs for that, so the violating value would sit inside a
    control the form has already disabled, submitted verbatim and refused
    pre-load. `params` in the dependency list is what closes it."""
    tsx = (REPO / "frontend/src/components/training/TrainingConfig.tsx").read_text(
        encoding="utf-8")
    effect = tsx[tsx.index("const startsNewContract"):]
    effect = effect[:effect.index("}, [")+len("}, [requiredValues, params]);")]
    assert "}, [requiredValues, params]);" in effect
    # Applied only on mismatch, so the added dependency cannot loop.
    assert "(params as any)[param] !== entry.value" in effect


# ---------------------------------------------------------------------------
# (2) train_unet on the LoRA path
# ---------------------------------------------------------------------------

def _trainer_call_keywords(source: str, name: str) -> dict:
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == name]
    assert len(calls) == 1, f"{len(calls)} construction sites for {name}"
    return {kw.arg: kw.value for kw in calls[0].keywords}


@pytest.mark.parametrize("trainer", ["LoRATrainer", "ReLoRATrainer",
                                     "FullParameterTrainer"])
def test_every_trainer_that_reads_the_flag_is_given_it(trainer):
    """Read off the AST: the keyword is passed, and from something other than a
    literal (matching the variable's spelling would pin the name instead)."""
    keywords = _trainer_call_keywords(
        RUNNER_PY.read_text(encoding="utf-8"), trainer)
    assert "train_unet" in keywords
    assert not isinstance(keywords["train_unet"], ast.Constant)
    assert "train_text_encoder" in keywords


def test_negative_control_the_lora_path_used_to_ignore_train_unet():
    """What shipped in b2694674: the same checkbox took effect for a full
    fine-tune and did nothing for a LoRA run."""
    try:
        before = subprocess.run(
            ["git", "show", "b2694674:backend/core/training/train_runner.py"],
            cwd=REPO, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if before.returncode != 0:  # pragma: no cover
        pytest.skip("b2694674 not in this clone")
    # Decoded here rather than by subprocess: the console codepage is not UTF-8.
    source = before.stdout.decode("utf-8")
    assert "train_unet" in _trainer_call_keywords(source, "FullParameterTrainer")
    assert "train_unet" not in _trainer_call_keywords(source, "LoRATrainer")
    assert "train_unet" not in _trainer_call_keywords(source, "ReLoRATrainer")


def test_the_flag_is_honoured_in_the_trainer_so_every_arch_obeys_it():
    """`LoRATrainer._apply_lora` gates injection, not the adapters -- which is
    why the three adapters that never mention `train_unet` (sensenova,
    ideogram4, minimax_h3) still obey it."""
    from core.training.lora_trainer import LoRATrainer

    calls = []
    stub = SimpleNamespace(
        log_prefix="[test]", lora_layers={},
        adapter=SimpleNamespace(
            apply_lora_to_unet=lambda layers: calls.append("unet") or 0,
            apply_lora_to_text_encoders=lambda layers: calls.append("te") or 0,
        ),
        train_unet=False, train_text_encoder=True,
    )
    LoRATrainer._apply_lora(stub)
    assert calls == ["te"]

    calls.clear()
    stub.train_unet, stub.train_text_encoder = True, False
    LoRATrainer._apply_lora(stub)
    assert calls == ["unet"]


def test_a_lora_run_that_does_not_touch_the_flags_is_unchanged():
    """The defaults still mean "denoiser only", for the trainer and for the
    contract, and the contract does not rewrite anything else."""
    from core.training.lora_trainer import LoRATrainer

    calls = []
    stub = SimpleNamespace(
        log_prefix="[test]", lora_layers={},
        adapter=SimpleNamespace(
            apply_lora_to_unet=lambda layers: calls.append("unet") or 0,
            apply_lora_to_text_encoders=lambda layers: calls.append("te") or 0,
        ),
        train_unet=TRAINING_DEFAULTS["train_unet"],
        train_text_encoder=TRAINING_DEFAULTS["train_text_encoder"],
    )
    LoRATrainer._apply_lora(stub)
    assert calls == ["unet"]

    config = _shipped_train_config()
    _assert_training_scope_is_nonempty("lora", config)
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora", config, {"sample": {}}
        )
    assert config["train_unet"] is True and config["train_text_encoder"] is False


# ---------------------------------------------------------------------------
# (2) "train nothing", now reachable on both paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("network_type", ["lora", "relora", "full_finetune"])
def test_a_run_that_would_train_nothing_is_refused_before_the_load(network_type):
    with pytest.raises(ValueError, match="nothing to train"):
        _assert_training_scope_is_nonempty(
            network_type,
            {"train_unet": False, "train_text_encoder": False,
             "train_image_encoder": False},
        )


@pytest.mark.parametrize("scope", [
    {},                                    # train_unet defaults True
    {"train_unet": False, "train_text_encoder": True},
    {"train_unet": False, "train_image_encoder": True},
    {"train_unet": False, "train_vision_encoder": True,
     "vision_encoder_path": "/models/siglip2"},
])
def test_any_one_trained_component_is_enough(scope):
    _assert_training_scope_is_nonempty("lora", scope)


def test_the_vision_encoder_only_case_needs_its_weights_named():
    """`train_vision_encoder` alone loads nothing (base_trainer only builds the
    encoder when `vision_encoder_path` is given), so it does not count."""
    with pytest.raises(ValueError, match="nothing to train"):
        _assert_training_scope_is_nonempty(
            "lora", {"train_unet": False, "train_vision_encoder": True})


@pytest.mark.parametrize("raw,expected", [
    ("false", False), ("False", False), ("0", False),
    ("true", True), ("1", True), (0, False), (1, True), (True, True),
    (None, True),   # explicit YAML null = unset = this flag's own default
])
def test_a_scope_flag_is_read_as_a_boolean_and_written_back_as_one(raw, expected):
    """A hand-written YAML `train_unet: "false"` is a non-empty string. The API
    types the field as a bool, so only that path reaches this."""
    config = {"train_unet": raw, "train_text_encoder": True}
    _assert_training_scope_is_nonempty("lora", config)
    assert config["train_unet"] is expected


def test_negative_control_plain_truthiness_would_have_kept_the_string_trainable():
    assert bool("false") is True   # what the guard used to do
    config = {"train_unet": "false", "train_text_encoder": False}
    with pytest.raises(ValueError, match="nothing to train"):
        _assert_training_scope_is_nonempty("lora", config)


def test_a_scope_flag_that_is_not_a_boolean_at_all_is_refused_by_name():
    with pytest.raises(ValueError, match="train_unet must be a boolean"):
        _assert_training_scope_is_nonempty("lora", {"train_unet": "yes"})


@pytest.mark.parametrize("network_type", ["controlnet", "vae_decoder", ""])
def test_methods_that_do_not_read_the_flags_are_exempt(network_type):
    """ControlNet trains its own module with `train_unet=False` by
    construction; a blanket check would refuse every ControlNet run."""
    _assert_training_scope_is_nonempty(
        network_type,
        {"train_unet": False, "train_text_encoder": False},
    )


def test_negative_control_what_an_empty_parameter_list_used_to_do():
    """Before the guard, sd15/sdxl/krea2 collected nothing and the failure was
    the optimizer's, minutes later with the checkpoint already resident."""
    import torch

    with pytest.raises(ValueError, match="empty parameter list"):
        torch.optim.AdamW([], lr=1e-4)


def test_sensenova_refuses_an_understanding_only_lora_before_the_load():
    """Its adapter refuses to SAVE a generation-free LoRA (inference applies
    both branches from one file), so such a run would train to its first save
    -- 100 steps at the shipped default -- and die there."""
    with _sensenova():
        with pytest.raises(ValueError, match="requires train_unet=True"):
            _apply_sensenova_training_contract(
                "model", "lora", _shipped_train_config(
                    train_unet=False, train_text_encoder=True),
                {"sample": {}},
            )
    # The full-FT path keeps the branch it named: understanding-only is "und".
    from core.training.ops.sensenova_ops import resolve_full_finetune_branch

    assert resolve_full_finetune_branch(
        SimpleNamespace(train_unet=False, train_text_encoder=True)) == "und"


def test_both_halves_is_still_how_the_understanding_branch_is_trained():
    config = _shipped_train_config(train_unet=True, train_text_encoder=True)
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora", config, {"sample": {}}
        )
