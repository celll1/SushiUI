"""Execute the trainer's resume guard without loading a model or dataset."""

import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import BaseTrainer, resolve_concept_resume_seed


@pytest.fixture
def guard():
    source = inspect.getsource(BaseTrainer.train)
    start = source.index("        from core.training.concept_exposure import ConceptExposure")
    end = source.index("        # Timestep distribution: resolve the EFFECTIVE sampler", start)
    return compile(textwrap.dedent(source[start:end]), "<trainer-resume-guard>", "exec")


def context(tmp_path, *, changed=True, policy="existing", concept=True):
    saved_fp = dict(total_item_count=24, image_paths_hash="smoke", dataset_ids=[42, 43, 44])
    current_fp = dict(total_item_count=17683, image_paths_hash="full", dataset_ids=[38, 37, 30])
    state = dict(dataset_fingerprint=saved_fp, batches_per_epoch=24,
                 multi_noise_timesteps=1, epoch=4, batch_idx=24)
    if concept:
        state["concept_batch_plan"] = dict(version=1, config_hash="same", seed=123)
    owner = SimpleNamespace(output_dir=tmp_path, log_prefix="[Test]", config={"seed": -1},
                            _concept_order_seed=456, run_seed_drawn=True,
                            _resume_plan_signature_data=None, _dataset_fingerprint=current_fp if changed else saved_fp,
                            crop_planner=None, optimizer=object(), _concept_multiplier_active=False)
    owner._check_dataset_fingerprint_changed = BaseTrainer._check_dataset_fingerprint_changed.__get__(owner)
    return dict(self=owner, resume_training_state=state, concept_config=SimpleNamespace(
                    fingerprint=lambda: "same", multiplied=False, plan_version=1) if concept else None,
                priority_training=False, resume_dataset_change_policy=policy,
                resume_from_checkpoint="latest", global_step=116, multi_noise_timesteps=1,
                batches_per_epoch=17683 if changed else 24, actual_total_steps=88415,
                start_epoch=4, resume_batch_idx=24, lr_scheduler_type="constant",
                total_steps=None, num_epochs=5, steps_per_epoch=17683,
                resolve_concept_resume_seed=resolve_concept_resume_seed)


@pytest.mark.parametrize("concept", [False, True])
def test_changed_dataset_restarts_epoch_and_keeps_training_progress(guard, tmp_path, concept, capsys):
    scope = context(tmp_path, concept=concept)
    optimizer = scope["self"].optimizer
    exec(guard, scope)
    assert (scope["start_epoch"], scope["resume_batch_idx"]) == (0, 0)
    assert scope["resume_training_state"] is None
    assert scope["self"]._resume_structure_changed
    assert scope["global_step"] == 116
    assert scope["actual_total_steps"] == 88415
    assert scope["self"].optimizer is optimizer
    if concept:
        assert scope["self"]._concept_order_seed == 123
        assert "previously seen images" in capsys.readouterr().out


def test_same_dataset_keeps_exact_concept_cursor(guard, tmp_path):
    scope = context(tmp_path, changed=False)
    state = scope["resume_training_state"]
    exec(guard, scope)
    assert scope["resume_training_state"] is state
    assert (scope["start_epoch"], scope["resume_batch_idx"]) == (4, 24)
    assert not scope["self"]._resume_structure_changed


def test_batch_count_only_change_restarts_concept_epoch(guard, tmp_path):
    scope = context(tmp_path, changed=False)
    scope["batches_per_epoch"] = 12
    exec(guard, scope)
    assert scope["resume_training_state"] is None
    assert (scope["start_epoch"], scope["resume_batch_idx"]) == (0, 0)


def test_changed_concept_settings_still_refused(guard, tmp_path):
    scope = context(tmp_path)
    scope["resume_training_state"]["concept_batch_plan"]["config_hash"] = "different"
    with pytest.raises(ValueError, match="changing its configuration"):
        exec(guard, scope)


def test_strict_resume_still_refuses_changed_dataset(guard, tmp_path):
    scope = context(tmp_path, policy="strict")
    scope["resume_training_state"].update(resume_order_signature="old", resume_dataset_manifest={})
    scope["self"]._resume_plan_signature = "new"
    scope["self"]._resume_dataset_manifest = {}
    with pytest.raises(ValueError, match="Cannot strictly resume"):
        exec(guard, scope)


def test_remaining_batches_requires_checkpoint_ledger(guard, tmp_path):
    scope = context(tmp_path, policy="rebase_remaining")
    with pytest.raises(ValueError, match="checkpoint has no batch ledger"):
        exec(guard, scope)


def test_existing_mode_cannot_discard_saved_ledger(guard, tmp_path):
    scope = context(tmp_path)
    scope["resume_training_state"]["resume_batch_plan"] = {"version": 1}
    with pytest.raises(ValueError, match="Cannot change resume_dataset_change_policy"):
        exec(guard, scope)


def boundary_context(tmp_path):
    scope = context(tmp_path, changed=False)
    scope.update(num_epochs=10, gradient_accumulation_steps=1,
                 update_total_steps_callback=lambda total: None)
    scope["resume_training_state"].update(gradient_accumulation_steps=1)
    scope["resume_training_state"]["concept_batch_plan"].update(batch_count=24)
    scope["concept_config"].fingerprint = lambda: "new"
    scope["concept_config"].plan_version = 2
    scope["concept_config"].multiplied = True
    scope["_preview_concept_counts"] = lambda seed: [90 + epoch for epoch in range(10)]
    scope["self"]._reanchor_lr_schedule_total = lambda total, step: None
    return scope


def test_completed_epoch_switches_plan_without_restarting_completed_epochs(guard, tmp_path):
    scope = boundary_context(tmp_path)
    optimizer = scope["self"].optimizer
    exec(guard, scope)
    assert (scope["start_epoch"], scope["resume_batch_idx"]) == (5, 0)
    assert scope["global_step"] == 116
    assert scope["self"].optimizer is optimizer
    assert scope["resume_training_state"] is None
    assert scope["actual_total_steps"] == 116 + sum(range(95, 100))
    assert scope["self"]._concept_order_seed == 123


@pytest.mark.parametrize("change", ["cursor", "gas", "mnt", "crop", "strict", "unknown_version"])
def test_plan_switch_requires_a_verified_clean_epoch_boundary(guard, tmp_path, change):
    scope = boundary_context(tmp_path)
    if change == "cursor":
        scope["resume_training_state"]["batch_idx"] -= 1
    elif change == "gas":
        scope["gradient_accumulation_steps"] = 2
    elif change == "mnt":
        scope["multi_noise_timesteps"] = 2
    elif change == "crop":
        scope["self"].crop_planner = object()
    elif change == "strict":
        scope["resume_dataset_change_policy"] = "strict"
    else:
        scope["resume_training_state"]["concept_batch_plan"]["version"] = 99
    with pytest.raises(ValueError, match="unsupported plan version"):
        exec(guard, scope)


def test_resume_after_switch_counts_only_remaining_random_epochs(guard, tmp_path):
    scope = boundary_context(tmp_path)
    state = scope["resume_training_state"]
    state.update(epoch=5, batch_idx=7, batches_per_epoch=95)
    scope.update(start_epoch=5, resume_batch_idx=7)
    state["concept_batch_plan"].update(version=2, config_hash="new", batch_count=95)
    exec(guard, scope)
    assert scope["resume_training_state"] is state
    assert scope["actual_total_steps"] == 116 + sum(range(95, 100)) - 7


def test_completed_epoch_dataset_and_concept_change_follow_existing_policy(guard, tmp_path):
    scope = boundary_context(tmp_path)
    scope["self"]._dataset_fingerprint = {"image_paths_hash": "changed"}
    optimizer = scope["self"].optimizer
    exec(guard, scope)
    assert (scope["start_epoch"], scope["resume_batch_idx"]) == (0, 0)
    assert scope["global_step"] == 116
    assert scope["self"].optimizer is optimizer
    assert scope["self"]._resume_structure_changed
    assert scope["resume_training_state"] is None
    assert scope["actual_total_steps"] == 116 + sum(range(90, 100))


@pytest.mark.parametrize("dataset_changed", [False, True])
def test_boundary_restores_saved_random_state_before_later_local_import(tmp_path, dataset_changed):
    import random
    scope = boundary_context(tmp_path)
    if dataset_changed:
        scope["self"]._dataset_fingerprint = {"image_paths_hash": "changed"}
    saved = random.Random(123).getstate()
    scope["resume_training_state"]["random_state"] = saved
    source = inspect.getsource(BaseTrainer.train)
    start = source.index("        from core.training.concept_exposure import ConceptExposure")
    end = source.index("        # Timestep distribution: resolve the EFFECTIVE sampler", start)
    body = textwrap.indent(textwrap.dedent(source[start:end]), "    ")
    # Preserve train()'s local binding: an exec-only guard misses UnboundLocalError.
    wrapper = f"def restore({', '.join(scope)}):\n{body}\n    import random\n    return locals()\n"
    namespace = {}
    exec(compile(wrapper, "<resume-function-scope>", "exec"), namespace)
    previous = random.getstate()
    try:
        result = namespace["restore"](**scope)
        assert result["resume_training_state"] is None
        assert result["start_epoch"] == (0 if dataset_changed else 5)
        assert result["global_step"] == 116
        assert random.getstate() == saved
    finally:
        random.setstate(previous)
