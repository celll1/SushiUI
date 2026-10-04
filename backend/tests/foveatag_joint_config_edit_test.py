"""Paused config edits retain Adam state and explicitly rebase schedules."""
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException

from api.foveatag_joint_review import edit_joint_config
from api.param_defaults import TAGGER_JOINT_REVIEW_DEFAULTS
from core.tagger.foveatag.direct_joint import _rebase_schedule
from core.tagger.foveatag.supervised_run import _cosine_with_warmup
from core.tagger.foveatag.joint_checkpoint import EDITABLE_JOINT_FIELDS, save_joint_checkpoint, load_joint_checkpoint


def fake_run_db(root, status="stopped"):
    root.mkdir(exist_ok=True)
    run = SimpleNamespace(config={"architecture": "foveatag_joint_review", "joint_review": dict(TAGGER_JOINT_REVIEW_DEFAULTS)},
        status=status, current_epoch=1, current_step=1595, output_dir=str(root), total_epochs=10, total_steps=300000)
    run.to_list_dict = lambda: {"config": run.config}
    db = SimpleNamespace(query=lambda *a: SimpleNamespace(filter_by=lambda **k: SimpleNamespace(first=lambda: run)),
        commit=lambda: None, refresh=lambda run: None, rollback=lambda: None)
    return run, db


def test_edit_records_only_supported_paused_changes(tmp_path):
    run, db = fake_run_db(tmp_path)
    edit_joint_config("synthetic", {"learning_rate": .00005, "candidate_count": 20, "selection_budget": 6}, db)
    assert run.config["joint_review"]["selection_budget"] == 6
    assert run.config["joint_config_history"][0]["saved_step"] == 1595
    assert (tmp_path / "review_plan.json").is_file()
    assert (tmp_path / "config_revisions.json").is_file()
    with pytest.raises(HTTPException) as error:
        edit_joint_config("synthetic", {"selection_budget": 2, "random_count": 2}, db)
    assert error.value.status_code == 422
    with pytest.raises(HTTPException) as error:
        edit_joint_config("synthetic", {"datasets_db": "replacement.db"}, db)
    assert error.value.status_code == 422


def test_running_config_save_refused_without_mutation(tmp_path):
    run, db = fake_run_db(tmp_path, "running")
    before = run.config
    with pytest.raises(HTTPException) as error:
        edit_joint_config("synthetic", {"learning_rate": .00005}, db)
    assert error.value.status_code == 409
    assert run.config is before


@pytest.mark.parametrize("warmup,step", [(0,5),(10,3),(10,10),(10,18)])
def test_rebased_schedule_matches_new_closed_form_and_next_step(warmup, step):
    plan = {"learning_rate": .00005, "warmup_steps": warmup}
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.001)
    reference_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=plan["learning_rate"])
    reference = _cosine_with_warmup(reference_optimizer, total_steps=100, warmup_steps=warmup, learning_rate=plan["learning_rate"])
    for _ in range(step):
        reference_optimizer.step(); reference.step()
    rebased = _rebase_schedule(optimizer, step, plan, 100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(reference_optimizer.param_groups[0]["lr"], abs=1e-12)
    optimizer.step(); rebased.step()
    reference_optimizer.step(); reference.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(reference_optimizer.param_groups[0]["lr"], abs=1e-12)


def test_editable_resume_preserves_optimizer_moments_and_refuses_frozen_change(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, router = torch.nn.Linear(2,2), torch.nn.Linear(2,2)
    optimizer = torch.optim.AdamW([*model.parameters(), *router.parameters()], lr=.001)
    (model(torch.ones(1,2)).sum() + router(torch.ones(1,2)).sum()).backward()
    optimizer.step()
    old = {"learning_rate": .001, "warmup_steps": 0, "datasets_db": "immutable.db"}
    scheduler = _cosine_with_warmup(optimizer, total_steps=100, warmup_steps=0, learning_rate=.001)
    path = save_joint_checkpoint(tmp_path / "latest.pt", model, router, optimizer, scheduler=scheduler,
        epoch=0, sample_offset=8, global_step=1, best_validation_f1=-1, best_threshold=.35, run_contract=old, history=[])
    moments = [state["exp_avg"].clone() for state in optimizer.state.values()]
    new_scheduler = _cosine_with_warmup(optimizer, total_steps=100, warmup_steps=10, learning_rate=.00005)
    new = {**old, "learning_rate": .00005, "warmup_steps": 10}
    saved = load_joint_checkpoint(path, model, router, optimizer, scheduler=new_scheduler, expected_run_contract=new,
        allowed_contract_changes=EDITABLE_JOINT_FIELDS, scheduler_contract_fields=("learning_rate", "warmup_steps"), restore_rng=False)
    assert saved["global_step"] == 1
    assert all(torch.equal(before, after["exp_avg"]) for before, after in zip(moments, optimizer.state.values()))
    with pytest.raises(ValueError, match="run contract"):
        load_joint_checkpoint(path, model, router, optimizer, scheduler=new_scheduler,
            expected_run_contract={**new, "datasets_db": "replacement.db"},
            allowed_contract_changes=EDITABLE_JOINT_FIELDS, restore_rng=False)
