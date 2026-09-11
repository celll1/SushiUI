"""Guard: runtime LR-schedule commands (P2 of docs/guides/LR_SCHEDULER_DESIGN.md).

P1 built the timeline but left it with no way in from outside: the only events
it could get were a resume's ``total_steps`` comparison and the MNT
recomputation. P2 adds the file-RPC that carries "decay now" / "cancel decay"
from the API process into the trainer, and the display file the GET endpoint
reads back.

Covered here:

* the two queues are separate directories-worth of files with separate caps: a
  sample request is not a command and neither claims the other's;
* a queued command is applied at the next poll;
* every §5.3 result code comes back keyed by ``request_id``, and re-delivering
  an id answers what it answered the first time instead of acting twice;
* a command belonging to another run sharing the output directory is left
  alone (``owns``);
* ``.lr_schedule.json`` is refreshed when the state changes WITH TIME -- a
  decay reaching its end becomes FLOOR with no event at all -- so a reader
  never sees a phase the run has left (§17.3);
* the cancel of §5.4: a linear return over ``W`` steps, and a step change back
  to the base curve when ``W = 0``, applied in the same batch.

CPU-only and hermetic: one 4-element parameter, no model, no dataset, no GPU,
no database.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/lr_schedule_control_rpc_test.py -v
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training import training_control_rpc as control_rpc  # noqa: E402
from core.training import training_file_rpc as file_rpc  # noqa: E402
from core.training import training_sample_rpc as sample_rpc  # noqa: E402
from core.training.base_trainer import (  # noqa: E402
    BaseTrainer,
    install_lr_schedule_events,
    lr_decay_state_code,
    lr_known_group_names,
    lr_schedule_status,
    poll_lr_schedule_commands,
    refresh_lr_schedule_status,
)
from core.training.lr_schedules import (  # noqa: E402
    STATE_BASE,
    STATE_DECAYING,
    STATE_FLOOR,
    STATE_RECOVERING,
    WARN_SELECTOR_ON_UNGROUPED_RUN,
    ScheduleTimeline,
    build_lr_scheduler,
    make_lambda,
    resolve_spec,
)
from core.training.training_events import TRAINING_EVENT_SENTINEL  # noqa: E402
from api.param_defaults import LR_PREVIEW_DEFAULTS  # noqa: E402

BASE_LR = 1e-4
RUN_ID = 7
# A plateau whose configured decay sits at the very end, so a COMMAND is the
# only thing that starts one inside the range these tests walk.
LATE_PLATEAU = {"lr_decay_start_ratio": 1.0, "lr_floor_ratio": 0.25}


class FakeTrainer:
    """The attributes the poll and status helpers actually read."""

    def __init__(self, output_dir, name="constant", W=0, T=100, config=None,
                 run_id=RUN_ID):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.log_prefix = "[Test]"
        self.fused_optimizer_groups = None
        param = torch.nn.Parameter(torch.zeros(4))
        self.optimizer = torch.optim.SGD([{"params": [param]}], lr=BASE_LR)
        self.lr_schedule_spec = resolve_spec(
            config or {}, warmup_steps=W, total_steps=T, name=name)
        self.lr_timeline = ScheduleTimeline()
        self.lr_timeline.set_total_steps(self.lr_schedule_spec.total_steps)
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer, self.lr_schedule_spec, self.lr_timeline)

    def seek(self, step: int) -> None:
        """Put the schedule at ``step`` the way a run's own advances would."""
        BaseTrainer._fast_forward_one_lr_scheduler(self.lr_scheduler, step)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def multiplier(self, step: int) -> float:
        return self.lr_timeline.multiplier(self.lr_schedule_spec, step)

    def state(self, step: int) -> int:
        return self.lr_timeline.state_at(self.lr_schedule_spec, step).code


def poll(trainer, global_step: int = 0) -> int:
    """Run the batch-head poll with its console output swallowed."""
    with redirect_stdout(io.StringIO()):
        return poll_lr_schedule_commands(trainer, global_step)


def results_by_id(output_dir):
    return {r["request_id"]: r for r in control_rpc.list_results(output_dir)}



def test_the_two_queues_do_not_see_each_other(tmp_path):
    sample_rpc.queue_request(tmp_path, seed=1, run_id=RUN_ID)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)

    assert len(sample_rpc.list_pending_requests(tmp_path, RUN_ID)) == 1
    assert len(control_rpc.list_pending_requests(tmp_path, RUN_ID)) == 1

    assert len(control_rpc.claim_all(tmp_path, RUN_ID)) == 1
    # Claiming every command left the sample request in place.
    assert len(sample_rpc.list_pending_requests(tmp_path, RUN_ID)) == 1
    assert sample_rpc.claim_next_request(tmp_path, RUN_ID) is not None


def test_the_command_cap_is_its_own(tmp_path):
    # A command is a list append, not a generation: the caps are different
    # numbers for different reasons.
    assert control_rpc.MAX_PENDING_REQUESTS > sample_rpc.MAX_PENDING_REQUESTS
    for _ in range(control_rpc.MAX_PENDING_REQUESTS):
        control_rpc.queue_request(tmp_path, command="cancel_decay", run_id=RUN_ID)
    with pytest.raises(control_rpc.ControlQueueFullError):
        control_rpc.queue_request(tmp_path, command="cancel_decay", run_id=RUN_ID)


def test_an_unknown_command_is_refused_before_it_is_written(tmp_path):
    with pytest.raises(ValueError):
        control_rpc.queue_request(tmp_path, command="reverse_time", run_id=RUN_ID)
    assert control_rpc.list_pending_requests(tmp_path) == []


def test_a_command_is_claimed_delete_first(tmp_path):
    payload = control_rpc.queue_request(tmp_path, command="start_decay",
                                        run_id=RUN_ID)
    path = control_rpc.request_path(tmp_path, payload["request_id"])
    assert path.exists()
    claimed = control_rpc.claim_all(tmp_path, RUN_ID)
    assert [c["request_id"] for c in claimed] == [payload["request_id"]]
    assert not path.exists()
    assert control_rpc.claim_all(tmp_path, RUN_ID) == []


def test_all_pending_commands_are_claimed_oldest_first(tmp_path):
    ids = [control_rpc.queue_request(tmp_path, command="start_decay",
                                     run_id=RUN_ID,
                                     request_id=f"cmd{i:02d}")["request_id"]
           for i in range(4)]
    claimed = [c["request_id"] for c in control_rpc.claim_all(tmp_path, RUN_ID)]
    assert claimed == ids


# ---------------------------------------------------------------------------
# A queued command is applied at the next poll
# ---------------------------------------------------------------------------

def test_a_queued_start_is_applied_at_the_next_poll(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    assert trainer.state(10) == STATE_BASE

    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    # Nothing happens until the trainer reaches a batch boundary.
    assert trainer.state(10) == STATE_BASE

    assert poll(trainer, global_step=10) == 1
    assert trainer.state(10) == STATE_DECAYING
    assert trainer.state(55) == STATE_DECAYING
    # Halfway through a cosine decay from 1.0 to the 0.25 floor.
    assert trainer.multiplier(55) == pytest.approx(0.25 + 0.75 * 0.5)


def test_a_start_takes_effect_from_the_multiplier_it_was_ordered_at(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(40)
    before = trainer.multiplier(40)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=40)
    # Continuous at the seam: the decay begins at the value the curve was at.
    assert trainer.multiplier(40) == pytest.approx(before)


def test_a_cancel_returns_linearly_to_the_base_curve(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=20, T=200,
                          config=LATE_PLATEAU)
    trainer.seek(50)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=50)

    trainer.seek(80)
    at_cancel = trainer.multiplier(80)
    control_rpc.queue_request(tmp_path, command="cancel_decay", run_id=RUN_ID)
    poll(trainer, global_step=80)

    assert trainer.state(80) == STATE_RECOVERING
    assert trainer.multiplier(80) == pytest.approx(at_cancel)
    # R = W = 20, linear from the cancelled value back to 1.0 (base curve).
    assert trainer.multiplier(90) == pytest.approx(at_cancel + (1.0 - at_cancel) * 0.5)
    assert trainer.multiplier(100) == pytest.approx(1.0)
    assert trainer.state(100) == STATE_BASE


def test_a_cancel_with_no_warmup_returns_in_the_same_batch(tmp_path):
    """W = 0 makes the return a step change (§5.4), and the poll writes it into
    the param groups before the forward -- the fused hooks step during the
    backward, so waiting for the next scheduler.step() would be a batch late."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=0, T=200,
                          config=LATE_PLATEAU)
    trainer.seek(50)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=50)
    trainer.seek(120)
    decayed = trainer.lr
    assert decayed < BASE_LR

    control_rpc.queue_request(tmp_path, command="cancel_decay", run_id=RUN_ID)
    poll(trainer, global_step=120)

    assert trainer.state(120) == STATE_BASE
    # No scheduler.step() in between: the poll re-wrote the groups itself.
    assert trainer.lr_scheduler.last_epoch == 120
    assert trainer.lr == pytest.approx(BASE_LR)


def test_decay_cancel_decay_stays_continuous_at_every_seam(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=40, T=1000,
                          config=LATE_PLATEAU)
    seams = []
    for step, command in ((100, "start_decay"), (300, "cancel_decay"),
                          (320, "start_decay")):
        trainer.seek(step)
        before = trainer.multiplier(step)
        control_rpc.queue_request(tmp_path, command=command, run_id=RUN_ID)
        assert poll(trainer, global_step=step) == 1
        assert trainer.multiplier(step) == pytest.approx(before)
        seams.append((step, trainer.state(step)))

    assert [code for _, code in seams] == [STATE_DECAYING, STATE_RECOVERING,
                                           STATE_DECAYING]
    # The third command was ordered 20 steps into a 40-step return, so the new
    # decay starts from the partly-recovered value, not from 1.0.
    at_cancel = trainer.lr_timeline.state_at(trainer.lr_schedule_spec, 300)
    assert at_cancel.start_multiplier < trainer.multiplier(320) < 1.0


def test_a_commanded_decay_does_not_survive_a_rewind_past_it(tmp_path):
    """state.json truncates to the checkpoint's step, so resuming from BEFORE
    the command un-does it -- the same rule that trims later metrics."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=1000,
                          config=LATE_PLATEAU)
    trainer.seek(137)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=137)
    assert trainer.state(137) == STATE_DECAYING

    saved = trainer.lr_timeline.dump(upto_step=100)
    assert [e["kind"] for e in saved] == ["total_steps"]

    resumed = ScheduleTimeline()
    resumed.load(saved, upto_step=100)
    assert resumed.state_at(trainer.lr_schedule_spec, 137).code == STATE_BASE


def test_a_poll_with_nothing_queued_changes_nothing(tmp_path):
    trainer = FakeTrainer(tmp_path)
    trainer.seek(10)
    before = list(trainer.lr_timeline.events)
    assert poll(trainer, global_step=10) == 0
    assert trainer.lr_timeline.events == before



def test_the_result_code_is_recorded_per_request_id(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    first = control_rpc.queue_request(tmp_path, command="start_decay",
                                      run_id=RUN_ID, request_id="a1")
    second = control_rpc.queue_request(tmp_path, command="start_decay",
                                       run_id=RUN_ID, request_id="b2")
    poll(trainer, global_step=10)

    results = results_by_id(tmp_path)
    assert set(results) == {"a1", "b2"}
    # Both were claimed in the same poll; the second one found a decay already
    # running, and says so instead of being silently dropped.
    assert results["a1"]["result"] == "applied"
    assert results["b2"]["result"] == "ignored_already_decaying"
    for rid, payload in ((first["request_id"], results["a1"]),
                         (second["request_id"], results["b2"])):
        assert payload["request_id"] == rid
        assert payload["run_id"] == RUN_ID
        assert payload["at"] == 10
        assert payload["global_step"] == 10
        assert payload["error"] is None


@pytest.mark.parametrize("scheduler,config,setup,command,expected", [
    # BASE, no configured decay -> nothing to cancel.
    ("constant", None, (), "cancel_decay", "ignored_no_active_decay"),
    # BASE, the run's own config declares a decay it has not reached: a cancel
    # disarms it rather than being a no-op (§17.3).
    ("plateau_cosine_floor", {"lr_decay_start_ratio": 0.85,
                              "lr_floor_ratio": 0.25},
     (), "cancel_decay", "disarmed_scheduled_decay"),
    ("plateau_cosine_floor", LATE_PLATEAU, ("start_decay",), "start_decay",
     "ignored_already_decaying"),
    ("plateau_cosine_floor", LATE_PLATEAU, ("start_decay", "cancel_decay"),
     "cancel_decay", "ignored_already_recovering"),
    # A decay ordered mid-recovery restarts from wherever the ramp had got to.
    ("plateau_cosine_floor", LATE_PLATEAU, ("start_decay", "cancel_decay"),
     "start_decay", "applied"),
])
def test_the_state_machine_answers_through_the_poll(tmp_path, scheduler, config,
                                                    setup, command, expected):
    trainer = FakeTrainer(tmp_path, name=scheduler, W=20, T=400, config=config)
    step = 30
    for prior in setup:
        trainer.seek(step)
        control_rpc.queue_request(tmp_path, command=prior, run_id=RUN_ID)
        poll(trainer, global_step=step)
        step += 5

    trainer.seek(step)
    control_rpc.queue_request(tmp_path, command=command, run_id=RUN_ID,
                              request_id="probe")
    poll(trainer, global_step=step)
    assert results_by_id(tmp_path)["probe"]["result"] == expected


def test_a_start_during_warmup_is_refused(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=50, T=400,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID,
                              request_id="early")
    poll(trainer, global_step=10)

    assert results_by_id(tmp_path)["early"]["result"] == "rejected_during_warmup"
    # A refused request can never take effect for any group.
    assert trainer.state(10) == STATE_BASE
    assert trainer.state(200) == STATE_BASE


def test_an_unknown_command_read_off_disk_is_reported_not_applied(tmp_path):
    trainer = FakeTrainer(tmp_path)
    trainer.seek(10)
    # Written past queue_request's validation, the way a stale file from a
    # future build would arrive.
    file_rpc.atomic_write_json(
        control_rpc.request_path(tmp_path, "weird"),
        {"request_id": "weird", "run_id": RUN_ID, "command": "reverse_time"})
    assert poll(trainer, global_step=10) == 0
    assert results_by_id(tmp_path)["weird"]["result"] == "rejected_unknown_command"
    assert trainer.lr_timeline.events == [
        {"kind": "total_steps", "at": 0, "value": 100, "seq": 0}]


def test_re_delivering_a_request_id_answers_the_same_thing(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID,
                              request_id="once")
    poll(trainer, global_step=10)
    events_after_first = [dict(e) for e in trainer.lr_timeline.events]
    assert results_by_id(tmp_path)["once"]["result"] == "applied"

    trainer.seek(30)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID,
                              request_id="once")
    poll(trainer, global_step=30)

    # Without the id check this second delivery would answer
    # "ignored_already_decaying" and record a second event at step 30.
    assert results_by_id(tmp_path)["once"]["result"] == "applied"
    assert trainer.lr_timeline.events == events_after_first


def test_a_command_for_another_run_is_left_alone(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU, run_id=RUN_ID)
    trainer.seek(10)
    other = control_rpc.queue_request(tmp_path, command="start_decay",
                                      run_id=RUN_ID + 1)

    assert poll(trainer, global_step=10) == 0
    assert trainer.state(10) == STATE_BASE
    # Two runs sharing a run_name share an output_dir: the file must survive for
    # the run it was meant for.
    assert control_rpc.request_path(tmp_path, other["request_id"]).exists()
    assert results_by_id(tmp_path) == {}

    assert [c["request_id"] for c in control_rpc.claim_all(tmp_path, RUN_ID + 1)] \
        == [other["request_id"]]


def test_a_command_with_no_run_id_is_claimable(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=None)
    assert poll(trainer, global_step=10) == 1
    assert trainer.state(10) == STATE_DECAYING



def test_the_status_file_is_written_on_the_first_poll(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(5)
    assert control_rpc.read_status(tmp_path) is None
    poll(trainer, global_step=5)

    status = control_rpc.read_status(tmp_path)
    assert status["state"] == "base"
    assert status["state_code"] == STATE_BASE
    assert status["run_id"] == RUN_ID
    assert status["step"] == 5
    assert status["scheduler"] == "plateau_cosine_floor"
    assert status["nominal_total_steps"] == 100
    assert status["effective_total_steps"] == 100


def test_the_status_file_follows_a_time_driven_transition_with_no_event(tmp_path):
    """A decay reaching its end becomes FLOOR with no command behind it. An
    event-only refresh would leave the reader on `decaying` forever (§17.3)."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=10)
    assert control_rpc.read_status(tmp_path)["state"] == "decaying"

    events_before = [dict(e) for e in trainer.lr_timeline.events]
    trainer.seek(100)
    # The batch after the decay ran out. Nothing was queued.
    assert poll(trainer, global_step=100) == 0

    status = control_rpc.read_status(tmp_path)
    assert trainer.lr_timeline.events == events_before
    assert status["state"] == "floor"
    assert status["state_code"] == STATE_FLOOR
    assert status["step"] == 100
    assert status["multiplier"] == pytest.approx(0.25)


def test_the_status_file_follows_a_recovery_ending_with_no_event(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=20, T=400,
                          config=LATE_PLATEAU)
    trainer.seek(50)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=50)
    trainer.seek(80)
    control_rpc.queue_request(tmp_path, command="cancel_decay", run_id=RUN_ID)
    poll(trainer, global_step=80)
    assert control_rpc.read_status(tmp_path)["state"] == "recovering"

    trainer.seek(100)
    poll(trainer, global_step=100)
    assert control_rpc.read_status(tmp_path)["state"] == "base"


def test_the_status_file_is_rewritten_only_when_the_state_moves(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=1000,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    poll(trainer, global_step=10)
    first = control_rpc.status_path(tmp_path).read_bytes()

    for step in (11, 12, 13):
        trainer.seek(step)
        poll(trainer, global_step=step)
    assert control_rpc.status_path(tmp_path).read_bytes() == first

    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=13)
    assert control_rpc.status_path(tmp_path).read_bytes() != first


def test_the_status_carries_per_group_state_as_well_as_the_representative(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    # Two groups, the shape a run training a backbone and a text encoder has.
    trainer.optimizer.add_param_group(
        {"params": [torch.nn.Parameter(torch.zeros(4))], "lr": BASE_LR / 2})
    trainer.lr_scheduler.base_lrs.append(BASE_LR / 2)
    trainer.lr_scheduler.lr_lambdas.append(trainer.lr_scheduler.lr_lambdas[0])

    trainer.seek(10)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=10)
    trainer.seek(55)
    poll(trainer, global_step=55)

    status = control_rpc.read_status(tmp_path)
    assert len(status["groups"]) == 2
    assert [g["index"] for g in status["groups"]] == [0, 1]
    for group in status["groups"]:
        assert group["state"] == status["state"] == "decaying"
        assert group["state_code"] == STATE_DECAYING
        assert group["multiplier"] == pytest.approx(status["multiplier"])
    # Same schedule, different base LRs: the multiplier is shared, the rate is not.
    assert status["groups"][1]["lr"] == pytest.approx(status["groups"][0]["lr"] / 2)


def test_the_status_is_json_serialisable_and_carries_the_event_list(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID,
                              request_id="e1")
    poll(trainer, global_step=10)

    raw = json.loads(control_rpc.status_path(tmp_path).read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in raw["events"]]
    assert kinds == ["total_steps", "decay"]
    assert raw["events"][1]["request_id"] == "e1"
    assert raw["events"][1]["result"] == "applied"


def test_refresh_survives_a_trainer_with_no_schedule(tmp_path):
    class Bare:
        output_dir = tmp_path
        log_prefix = "[Test]"
        run_id = RUN_ID

    bare = Bare()
    assert lr_schedule_status(bare) is None
    assert refresh_lr_schedule_status(bare) is False
    with redirect_stdout(io.StringIO()):
        assert poll_lr_schedule_commands(bare, 0) == 0
    assert lr_decay_state_code(bare) is None


def test_the_decay_state_metric_matches_the_status(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=100,
                          config=LATE_PLATEAU)
    trainer.seek(10)
    assert lr_decay_state_code(trainer) == STATE_BASE
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=10)
    assert lr_decay_state_code(trainer) == STATE_DECAYING
    trainer.seek(100)
    assert lr_decay_state_code(trainer) == STATE_FLOOR


def test_the_metric_is_registered():
    from core.training.metric_registry import EXTRA_METRIC_DEFS
    entry = EXTRA_METRIC_DEFS["lr_decay_state"]
    # A phase code, not a rate: it must not pool with the learning-rate axis.
    assert entry["scale_group"] != "learning_rate"
    assert entry["range"] == {"kind": "fixed", "min": 0, "max": 3}


def test_clear_all_removes_commands_and_results_but_not_the_state(tmp_path):
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    control_rpc.write_result(tmp_path, "done", {"result": "applied"})
    control_rpc.write_status(tmp_path, {"state": "floor"})

    assert control_rpc.clear_all(tmp_path) == 2
    assert control_rpc.list_pending_requests(tmp_path) == []
    assert control_rpc.list_results(tmp_path) == []
    # The last known state is what GET serves for a stopped run.
    assert control_rpc.read_status(tmp_path) == {"state": "floor"}



def test_the_endpoints_are_documented_in_openapi():
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    path = spec["paths"]["/training/runs/{run_id}/lr-schedule"]
    assert set(path) == {"get", "post"}
    assert set(path["post"]["responses"]) == {"202", "400", "404", "409", "429", "500"}
    assert set(path["get"]["responses"]) == {"200", "404"}

    body = path["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"].endswith("/LrScheduleCommandRequest")
    command = spec["components"]["schemas"]["LrScheduleCommandRequest"]["properties"]["command"]
    assert command["enum"] == list(control_rpc.COMMANDS)

    documented = set(spec["components"]["schemas"]["LrScheduleCommandResult"]
                     ["properties"]["result"]["enum"])
    assert {"applied", "disarmed_scheduled_decay", "ignored_already_decaying",
            "ignored_already_recovering", "ignored_no_active_decay",
            "rejected_during_warmup", "rejected_zero_length",
            "rejected_unknown_command"} <= documented


# ---------------------------------------------------------------------------
# R3: the `retarget` command over the same transport (§19.8)
# ---------------------------------------------------------------------------

# A linear decay to zero over 40 global steps, starting one step after the
# retarget takes effect: `lr_decay_start_step = 0` is `wsd`'s "manual", so a
# curve that decays on its own has to name a step.
RETARGET_TO_WSD = {
    "lr_scheduler": "wsd",
    "lr_warmup_steps": 0,
    "lr_floor_ratio": 0.0,
    "lr_decay_start_step": 1,
    "lr_decay_steps": 40,
    "lr_decay_shape": "linear",
    "anchor": "restart",
    "length": 0,
}


class GroupedFakeTrainer(FakeTrainer):
    """A trainer whose optimizer groups carry component names.

    `known_groups` (§19.4 rule 7) is read off the optimizer, not off
    `lr_group_schedules`: a run with no mapping still has components, and a
    selector naming one of them is accepted (§19.3's last row).
    """

    def __init__(self, output_dir, components=("unet", "text_encoder_1"),
                 **kwargs):
        super().__init__(output_dir, **kwargs)
        self.optimizer = torch.optim.SGD(
            [{"params": [torch.nn.Parameter(torch.zeros(4))],
              "component": c, "name": c, "lr": BASE_LR} for c in components],
            lr=BASE_LR)
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer, self.lr_schedule_spec, self.lr_timeline)


def queue_retarget(output_dir, payload=None, run_id=RUN_ID, **overrides):
    body = dict(RETARGET_TO_WSD if payload is None else payload)
    body.update(overrides)
    return control_rpc.queue_request(
        output_dir, command=control_rpc.RETARGET_COMMAND, run_id=run_id,
        extra={"payload": body})


def test_the_retarget_command_shares_the_queue_but_not_the_button_enum():
    # The two-button endpoint's enum must not grow a command that needs a body.
    assert control_rpc.COMMANDS == ("start_decay", "cancel_decay")
    assert control_rpc.RETARGET_COMMAND not in control_rpc.COMMANDS
    assert control_rpc.COMMAND_EVENT_KINDS[control_rpc.RETARGET_COMMAND] == \
        "retarget"


def test_a_retarget_without_a_payload_is_refused_at_the_queue(tmp_path):
    with pytest.raises(ValueError):
        control_rpc.queue_request(tmp_path,
                                  command=control_rpc.RETARGET_COMMAND,
                                  run_id=RUN_ID)
    assert control_rpc.list_pending_requests(tmp_path) == []


def test_a_queued_retarget_reaches_the_timeline_and_switches_the_curve(tmp_path):
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer.seek(20)
    assert trainer.multiplier(20) == pytest.approx(1.0)

    queued = queue_retarget(tmp_path)
    assert poll(trainer, global_step=20) == 1

    event = trainer.lr_timeline.dump(20)[-1]
    assert (event["kind"], event["at"], event["issued"]) == ("retarget", 20, 20)
    assert event["spec"]["name"] == "wsd"
    # Linear over 40 scheduler steps from the multiplier it was at, on the
    # retarget's own axis: the decay starts at 21, not at 1.
    assert trainer.multiplier(21) == pytest.approx(1.0)
    assert trainer.multiplier(41) == pytest.approx(0.5)
    assert trainer.multiplier(61) == pytest.approx(0.0)
    # And it took effect on the live optimizer in the same batch, not the next.
    assert trainer.lr == pytest.approx(BASE_LR)

    result = results_by_id(tmp_path)[queued["request_id"]]
    assert (result["result"], result["command"], result["at"]) == \
        ("applied", "retarget", 20)


def test_a_retarget_result_is_idempotent_by_request_id(tmp_path):
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer.seek(20)
    queued = queue_retarget(tmp_path)
    poll(trainer, global_step=20)
    control_rpc.queue_request(
        tmp_path, command=control_rpc.RETARGET_COMMAND, run_id=RUN_ID,
        request_id=queued["request_id"], extra={"payload": RETARGET_TO_WSD})
    trainer.seek(30)
    poll(trainer, global_step=30)
    assert [e["kind"] for e in trainer.lr_timeline.dump(30)].count("retarget") \
        == 1


def test_the_payload_step_counts_are_global_steps(tmp_path):
    """gas=4: a 40-global-step decay is 10 scheduler steps, and a 20-step blend
    is 5. Converting only some of them would draw a curve the run never
    follows."""
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer._grad_accum_steps = 4
    trainer.seek(20)
    queue_retarget(tmp_path, length=20)
    poll(trainer, global_step=80)

    event = trainer.lr_timeline.dump(20)[-1]
    assert event["length"] == 5
    assert event["spec"]["decay_length"] == 10
    # The 1-global-step start floors into the first accumulation window, so the
    # decay begins at the anchor and reaches zero 10 scheduler steps later.
    assert trainer.multiplier(25) == pytest.approx(0.5)
    assert trainer.multiplier(30) == pytest.approx(0.0)


def test_a_retarget_can_name_a_future_step(tmp_path):
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer.seek(10)
    queue_retarget(tmp_path, at=50)
    assert poll(trainer, global_step=10) == 1

    event = trainer.lr_timeline.dump(10)[-1]
    assert (event["at"], event["issued"]) == (50, 10)
    # Nothing before the reservation moves; the switch happens at 50.
    assert trainer.multiplier(49) == pytest.approx(1.0)
    assert trainer.multiplier(71) == pytest.approx(0.5)


def test_a_retarget_naming_a_step_already_past_is_refused(tmp_path):
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer.seek(60)
    queued = queue_retarget(tmp_path, at=10)
    poll(trainer, global_step=60)

    assert results_by_id(tmp_path)[queued["request_id"]]["result"] == \
        "rejected_backdated"
    assert trainer.multiplier(80) == pytest.approx(1.0)


def test_the_issued_step_is_the_trainers_not_the_requesters(tmp_path):
    """The control path supplies `issued`; a request cannot pre-date its own
    acceptance by claiming one."""
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer.seek(60)
    queue_retarget(tmp_path, at=70)
    poll(trainer, global_step=60)
    assert trainer.lr_timeline.dump(60)[-1]["issued"] == 60


def test_the_component_list_is_what_rule_7_validates_against(tmp_path):
    trainer = GroupedFakeTrainer(tmp_path, name="constant", T=100)
    assert lr_known_group_names(trainer) == ["unet", "text_encoder_1"]
    trainer.seek(20)

    bad = queue_retarget(tmp_path, groups=["vae"])
    poll(trainer, global_step=20)
    assert results_by_id(tmp_path)[bad["request_id"]]["result"] == \
        "rejected_unknown_group"

    good = queue_retarget(tmp_path, groups=["UNet"])   # D35: case-folded
    poll(trainer, global_step=20)
    assert results_by_id(tmp_path)[good["request_id"]]["result"] == "applied"


def test_without_a_component_list_a_group_name_cannot_be_checked(tmp_path):
    """The plain FakeTrainer's one param group carries no component, so there
    is nothing to validate against and the name is accepted (R1's behaviour)."""
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    assert lr_known_group_names(trainer) == []
    trainer.seek(20)
    queued = queue_retarget(tmp_path, groups=["nowhere"])
    poll(trainer, global_step=20)
    assert results_by_id(tmp_path)[queued["request_id"]]["result"] == "applied"


def test_an_omitted_length_becomes_the_runs_warmup(tmp_path):
    """D31, through the real command path: the endpoint sends no length and the
    run's own warmup fills it in -- on the scheduler axis."""
    trainer = FakeTrainer(tmp_path, name="constant", W=8, T=100)
    trainer.seek(20)
    payload = dict(RETARGET_TO_WSD)
    del payload["length"]
    queue_retarget(tmp_path, payload=payload)
    poll(trainer, global_step=20)
    assert trainer.lr_timeline.dump(20)[-1]["length"] == 8


def test_a_retarget_the_trainer_cannot_build_comes_back_as_an_error(tmp_path):
    """resolve_spec raises on a payload the endpoint would have refused; the
    poll records it instead of letting it into the training loop."""
    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer.seek(20)
    queued = queue_retarget(tmp_path, lr_decay_shape="quadratic")
    poll(trainer, global_step=20)
    result = results_by_id(tmp_path)[queued["request_id"]]
    assert result["result"] == "error"
    assert "quadratic" in result["error"]
    assert [e["kind"] for e in trainer.lr_timeline.dump(20)] == ["total_steps"]



def test_the_status_file_carries_the_anchor_and_absolute_positions(tmp_path):
    trainer = FakeTrainer(tmp_path, name="constant", T=200)
    trainer.seek(50)
    queue_retarget(tmp_path, lr_warmup_steps=10, lr_decay_steps=0, length=20)
    poll(trainer, global_step=50)

    status = control_rpc.read_status(tmp_path)
    assert status["anchor_step"] == 50
    # The spec's own warmup is a LENGTH from the anchor; the file also states
    # where it ends, so no reader has to know which of the two it was given.
    assert status["warmup_steps"] == 10
    assert status["warmup_end_step"] == 60
    assert status["blend"] == {"at": 50, "length": 20, "ends_at": 70,
                               "shape": "linear"}
    assert status["step"] == 50
    assert status["groups"][0]["anchor_step"] == 50


def test_the_status_file_anchor_is_zero_without_a_retarget(tmp_path):
    trainer = FakeTrainer(tmp_path, name="constant", W=5, T=200)
    trainer.seek(50)
    refresh_lr_schedule_status(trainer, global_step=50, force=True)
    status = control_rpc.read_status(tmp_path)
    assert (status["anchor_step"], status["warmup_end_step"]) == (0, 5)
    assert status["blend"] is None


def test_a_finished_blend_leaves_the_status_file(tmp_path):
    trainer = FakeTrainer(tmp_path, name="constant", T=200)
    trainer.seek(50)
    queue_retarget(tmp_path, length=20)
    poll(trainer, global_step=50)
    trainer.seek(70)
    refresh_lr_schedule_status(trainer, global_step=70, force=True)
    assert control_rpc.read_status(tmp_path)["blend"] is None


def test_a_reservation_firing_rewrites_the_status_file(tmp_path):
    """A future-dated retarget takes effect with NO new event, so an
    event-count signature would leave GET naming the old schedule forever."""
    trainer = FakeTrainer(tmp_path, name="constant", T=200)
    trainer.seek(10)
    queue_retarget(tmp_path, at=100)
    poll(trainer, global_step=10)
    assert control_rpc.read_status(tmp_path)["scheduler"] == "constant"

    trainer.seek(120)
    assert poll(trainer, global_step=120) == 0
    status = control_rpc.read_status(tmp_path)
    assert (status["scheduler"], status["anchor_step"]) == ("wsd", 100)


# ---------------------------------------------------------------------------
# R3: the real state.json round trip (§19.5, invariant 6)
# ---------------------------------------------------------------------------

class StateHarness(GroupedFakeTrainer):
    """FakeTrainer plus what the real save/load path reads."""

    from core.training.base_trainer import BaseTrainer as _B

    save_training_state = _B.save_training_state
    load_training_state = _B.load_training_state
    del _B

    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir, **kwargs)
        self.run_name = "20260101_000000_deadbeef"
        self._grad_accum_steps = 1
        self._dataset_fingerprint = None
        self._batches_per_epoch = 10
        self._crop_plan_fingerprint = None
        self.lr_group_specs = [self.lr_schedule_spec.for_group(c)
                               for c in ("unet", "text_encoder_1")]
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer, self.lr_schedule_spec, self.lr_timeline,
            group_specs=self.lr_group_specs)


def test_a_reservation_and_a_scoped_retarget_survive_the_state_file(tmp_path):
    saver = StateHarness(tmp_path, name="cosine", T=200)
    saver.seek(100)
    timeline = saver.lr_timeline
    timeline.add("retarget", at=60, issued=60, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=140, name="constant"),
        length=0, groups=["unet"], known_groups=lr_known_group_names(saver))
    # Ordered now, effective later: `at` is in the future, `issued` is not.
    timeline.add("retarget", at=180, issued=100, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=20, name="linear"), length=0)
    before = {c: [timeline.multiplier(s, step) for step in range(0, 201, 7)]
              for c, s in zip(("unet", "text_encoder_1"), saver.lr_group_specs)}

    with redirect_stdout(io.StringIO()):
        saver.save_training_state(step=100, epoch=0, batch_idx=3)
        loader = StateHarness(tmp_path, name="cosine", T=200)
        state = loader.load_training_state(100)
        install_lr_schedule_events(loader, 100)

    kinds = [e["kind"] for e in state["lr_schedule_events"]]
    assert kinds == ["total_steps", "retarget", "retarget"]
    # The reservation is stored with its own `at`, cut by `issued` (invariant 6).
    assert [e["at"] for e in state["lr_schedule_events"]] == [0, 60, 180]
    after = {c: [loader.lr_timeline.multiplier(s, step)
                 for step in range(0, 201, 7)]
             for c, s in zip(("unet", "text_encoder_1"), loader.lr_group_specs)}
    assert after == before
    # The scoped one still knows whose chain it is on after the round trip.
    assert loader.lr_timeline.active_spec(
        loader.lr_group_specs[0], 100).name == "constant"
    assert loader.lr_timeline.active_spec(
        loader.lr_group_specs[1], 100).name == "cosine"


def test_a_checkpoint_before_the_order_drops_the_reservation(tmp_path):
    """The truncation is by `issued`: rewinding to before the command was given
    un-does it, exactly as it does for a decay."""
    saver = StateHarness(tmp_path, name="cosine", T=200)
    saver.seek(100)
    saver.lr_timeline.add("retarget", at=180, issued=100, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=20, name="linear"), length=0)
    assert len(saver.lr_timeline.dump(99)) == 1
    assert len(saver.lr_timeline.dump(100)) == 2



@pytest.fixture(scope="module")
def routes():
    """The API module. Imported lazily -- it pulls in the whole app."""
    import api.routes as module
    return module


class _FakeProc:
    def __init__(self, output_dir, running=True):
        self.output_dir = str(output_dir)
        self.is_running = running


class _FakeDb:
    """`db.query(TrainingRun).filter(...).first()` and nothing else."""

    def __init__(self, row):
        self._row = row

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._row


def call_retarget(routes, tmp_path, body=None, run=object(), proc=True,
                  monkeypatch=None, **overrides):
    import asyncio
    from core.training.training_process import training_process_manager

    payload = dict(body or {"lr_scheduler": "wsd", "lr_decay_steps": 40})
    payload.update(overrides)
    processes = training_process_manager.processes
    if proc:
        processes[RUN_ID] = _FakeProc(tmp_path, running=proc != "stopped")
    else:
        processes.pop(RUN_ID, None)
    try:
        return asyncio.run(routes.queue_lr_schedule_retarget(
            RUN_ID, routes.LrScheduleRetargetRequest(**payload),
            db=_FakeDb(run)))
    finally:
        processes.pop(RUN_ID, None)


def status_of(excinfo):
    return excinfo.value.status_code


def test_the_endpoint_queues_a_retarget_the_trainer_can_apply(routes, tmp_path):
    accepted = call_retarget(routes, tmp_path, body={
        "lr_scheduler": "wsd", "lr_decay_start_step": 1, "lr_decay_steps": 40,
        "lr_decay_shape": "linear", "lr_floor_ratio": 0.0, "length": 0})
    assert accepted["command"] == "retarget"
    assert accepted["pending_count"] == 1
    # D44: the payload carries only what was supplied; omitted keys are
    # inherited by the trainer from the active spec.
    assert accepted["payload"]["lr_scheduler"] == "wsd"
    assert accepted["payload"]["length"] == 0
    assert "anchor" not in accepted["payload"]
    assert "gain" not in accepted["payload"]
    assert "command_decay_length" not in accepted["payload"]
    assert "length" not in call_retarget(routes, tmp_path / "other")["payload"]

    trainer = FakeTrainer(tmp_path, name="constant", T=100)
    trainer.seek(20)
    poll(trainer, global_step=20)
    assert results_by_id(tmp_path)[accepted["request_id"]]["result"] == "applied"
    assert trainer.multiplier(41) == pytest.approx(0.5)


def test_the_endpoint_refuses_the_runs_own_decay_parameters(routes, tmp_path):
    """D42: `command_decay_*` are run-wide `start_decay` settings. Carried in a
    retarget, a scoped one would hand them to groups it never named."""
    for key in ("command_decay_length", "command_decay_shape"):
        with pytest.raises(routes.HTTPException) as e:
            call_retarget(routes, tmp_path, **{key: 321 if "length" in key
                                               else "rex"})
        assert status_of(e) == 400
        assert "lr_decay_steps" in e.value.detail
    assert control_rpc.list_pending_requests(tmp_path) == []


@pytest.mark.parametrize("bad,message", [
    ({"lr_scheduler": "not_a_schedule"}, "lr_scheduler"),
    ({"anchor": "sideways"}, "anchor"),
    ({"shape": "exp"}, "blend shape"),
    ({"gain": 0}, "gain"),
    ({"gain": -1.0}, "gain"),
    ({"length": -5}, "length"),
    ({"groups": []}, "groups"),
    ({"lr_decay_shape": "quadratic"}, "lr_decay_shape"),
    ({"lr_floor_ratio": 1.5}, "lr_floor_ratio"),
    ({"lr_scheduler": "cosine_with_restarts", "lr_cycle_peak_decay": 0.0},
     "lr_cycle_peak_decay"),
    ({"lr_decay_steps": -1}, "lr_decay_steps"),
])
def test_the_endpoint_refuses_what_it_can_decide_without_the_run(
        routes, tmp_path, bad, message):
    with pytest.raises(routes.HTTPException) as e:
        call_retarget(routes, tmp_path, **bad)
    assert status_of(e) == 400
    assert message in e.value.detail
    assert control_rpc.list_pending_requests(tmp_path) == []


def test_the_endpoint_rejects_an_unknown_key_rather_than_ignoring_it(routes):
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        routes.LrScheduleRetargetRequest(lr_scheduler="wsd", lr_decay_stps=40)


def test_a_missing_run_is_a_404(routes, tmp_path):
    with pytest.raises(routes.HTTPException) as e:
        call_retarget(routes, tmp_path, run=None)
    assert status_of(e) == 404


def test_a_run_that_is_not_executing_is_a_409(routes, tmp_path):
    for proc in (False, "stopped"):
        with pytest.raises(routes.HTTPException) as e:
            call_retarget(routes, tmp_path, proc=proc)
        assert status_of(e) == 409
    assert control_rpc.list_pending_requests(tmp_path) == []


def test_a_full_queue_is_a_429(routes, tmp_path):
    for _ in range(control_rpc.MAX_PENDING_REQUESTS):
        control_rpc.queue_request(tmp_path, command="start_decay",
                                  run_id=RUN_ID)
    with pytest.raises(routes.HTTPException) as e:
        call_retarget(routes, tmp_path)
    assert status_of(e) == 429


def test_the_retarget_endpoint_is_documented_in_openapi():
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    path = spec["paths"]["/training/runs/{run_id}/lr-schedule/retarget"]["post"]
    assert set(path["responses"]) == {"202", "400", "404", "409", "429", "500"}
    body = path["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"].endswith("/LrScheduleRetargetRequest")

    schema = spec["components"]["schemas"]["LrScheduleRetargetRequest"]
    from core.training.lr_schedules import (
        BLEND_SHAPE_NAMES, INTERNAL_SCHEDULER_NAMES, LR_SCHEDULER_NAMES,
        RETARGET_ANCHORS, RETARGET_OPS,
    )
    # The internal name is legal HERE and nowhere else: only a run already on
    # it may retarget onto it, which the timeline decides (§19.4 rule 2).
    assert schema["properties"]["lr_scheduler"]["enum"] == (
        list(LR_SCHEDULER_NAMES) + list(INTERNAL_SCHEDULER_NAMES))
    assert schema["properties"]["op"]["enum"] == list(RETARGET_OPS)
    assert schema["properties"]["anchor"]["enum"] == list(RETARGET_ANCHORS)
    assert schema["properties"]["shape"]["enum"] == list(BLEND_SHAPE_NAMES)
    # D42's two keys are refused, so they are not part of the documented body.
    assert not {"command_decay_length", "command_decay_shape"} & set(
        schema["properties"])

    documented = set(spec["components"]["schemas"]["LrScheduleCommandResult"]
                     ["properties"]["result"]["enum"])
    assert {"rejected_backdated", "rejected_unknown_group",
            "rejected_no_remaining_span", "rejected_warmup_exceeds_span",
            "rejected_empty_group_selector"} <= documented


def test_the_documented_defaults_are_the_ones_the_endpoint_uses():
    """Invariant 7: one defaults table, and openapi says what it says."""
    import yaml
    from api.param_defaults import LR_RETARGET_DEFAULTS
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["LrScheduleRetargetRequest"]["properties"]
    for key, value in LR_RETARGET_DEFAULTS.items():
        assert props[key]["default"] == value, key



def publish(trainer, global_step: int = 0):
    """Write `.lr_schedule.json`, which is all the API process gets to read."""
    with redirect_stdout(io.StringIO()):
        refresh_lr_schedule_status(trainer, global_step=global_step, force=True)


def call_preview(routes, tmp_path=None, live_step=None, **body):
    """One preview. ``live_step`` makes the run LOOK live at that global step,
    which is what the endpoint reads instead of the display file's."""
    import asyncio
    from core.training.training_process import training_process_manager

    run = None if tmp_path is None else SimpleNamespace(
        output_dir=str(tmp_path), current_step=live_step or 0)
    processes = training_process_manager.processes
    if live_step is not None:
        processes[RUN_ID] = _FakeProc(tmp_path)
    try:
        return asyncio.run(routes.preview_lr_schedule_events(
            routes.LrSchedulePreviewRequest(**body), db=_FakeDb(run)))
    finally:
        processes.pop(RUN_ID, None)


def test_a_preview_draws_the_curve_the_trainer_would_produce(routes, tmp_path):
    """The anti-duplication guarantee (D20): the preview's numbers ARE the
    lambda's, because both fold the same events with the same code."""
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(20)
    publish(trainer, global_step=20)

    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=12,
                           events=[dict(RETARGET_TO_WSD)])
    assert [r["result"] for r in preview["results"]] == ["applied"]

    queue_retarget(tmp_path)
    assert poll(trainer, global_step=20) == 1
    applied = make_lambda(trainer.lr_schedule_spec, trainer.lr_timeline)
    assert preview["curves"][0]["points"] == [
        [step, applied(step)] for step, _ in preview["curves"][0]["points"]]
    # And the baseline is the curve the run was on before the candidate.
    assert preview["curves"][0]["baseline_points"] != \
        preview["curves"][0]["points"]


def test_a_preview_writes_nothing_and_moves_no_state(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(20)
    publish(trainer, global_step=20)
    status_before = control_rpc.status_path(tmp_path).read_bytes()
    events_before = json.dumps(trainer.lr_timeline.dump(10 ** 9))

    for _ in range(2):
        call_preview(routes, tmp_path, run_id=RUN_ID,
                     events=[dict(RETARGET_TO_WSD), {"op": "hold"}])

    assert control_rpc.status_path(tmp_path).read_bytes() == status_before
    assert json.dumps(trainer.lr_timeline.dump(10 ** 9)) == events_before
    assert control_rpc.list_pending_requests(tmp_path) == []
    assert control_rpc.list_results(tmp_path) == []
    # No file of any kind appeared beside the one the trainer had written.
    assert [p.name for p in tmp_path.iterdir()] == [control_rpc.STATUS_FILENAME]


def test_a_preview_returns_the_selector_warning_instead_of_emitting_it(
        routes, tmp_path, capsys):
    """§19.5.2-8: the API process has no reader for the trainer's stdout
    sentinel, so the notice travels on the event and in the response."""
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    publish(trainer)
    capsys.readouterr()

    preview = call_preview(routes, tmp_path, run_id=RUN_ID, events=[
        {"lr_scheduler": "constant", "length": 0, "groups": ["unet"]}])

    assert preview["results"][0]["warning"] == WARN_SELECTOR_ON_UNGROUPED_RUN
    assert [w["code"] for w in preview["warnings"]] == [
        WARN_SELECTOR_ON_UNGROUPED_RUN]
    assert "param groups all share one LR schedule" in \
        preview["warnings"][0]["message"]
    printed = capsys.readouterr().out
    assert TRAINING_EVENT_SENTINEL not in printed
    assert WARN_SELECTOR_ON_UNGROUPED_RUN not in printed


def test_a_preview_of_a_config_agrees_with_the_get_on_the_same_config(routes):
    import asyncio
    config = {"lr_scheduler": "wsd", "total_steps": 1000,
              "lr_warmup_steps": 100, "lr_decay_start_step": 600,
              "lr_floor_ratio": 0.1}
    posted = call_preview(routes, config=config, n_points=32)
    got = asyncio.run(routes.preview_lr_schedule(n_points=32, **config))
    assert posted["curves"][0]["points"] == got["points"]
    assert posted["description"] == got["description"]
    assert posted["source"] == "config" and posted["run_id"] is None


def test_a_preview_needs_exactly_one_source(routes, tmp_path):
    for body in ({}, {"run_id": RUN_ID, "config": {"lr_scheduler": "cosine"}}):
        with pytest.raises(routes.HTTPException) as e:
            call_preview(routes, tmp_path, **body)
        assert status_of(e) == 400


def test_a_preview_of_a_missing_run_is_a_404(routes):
    with pytest.raises(routes.HTTPException) as e:
        call_preview(routes, None, run_id=RUN_ID)
    assert status_of(e) == 404


def test_a_run_that_has_published_nothing_yet_is_a_409(routes, tmp_path):
    with pytest.raises(routes.HTTPException) as e:
        call_preview(routes, tmp_path, run_id=RUN_ID)
    assert status_of(e) == 409


def test_a_preview_refuses_a_candidate_the_retarget_endpoint_refuses(
        routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    publish(trainer)
    for bad in ({"lr_scheduler": "not_a_schedule"}, {"lr_scheduler": "wsd",
                                                     "groups": []},
                {"op": "hold", "lr_scheduler": "wsd"},
                {"op": "scale"}):
        with pytest.raises(routes.HTTPException) as e:
            call_preview(routes, tmp_path, run_id=RUN_ID, events=[bad])
        assert status_of(e) == 400


def test_a_preview_reports_a_refusal_the_run_decides(routes, tmp_path):
    """Backdating is measured against the timeline, not the vocabulary, so it
    comes back as a result code with the curve unchanged."""
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(50)
    publish(trainer, global_step=50)
    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=8,
                           events=[dict(RETARGET_TO_WSD, at=10)])
    assert preview["results"][0]["result"] == "rejected_backdated"
    assert preview["results"][0]["kind"] == "noop"
    assert preview["curves"][0]["points"] == \
        preview["curves"][0]["baseline_points"]


def test_a_preview_draws_one_curve_per_diverging_group(routes, tmp_path):
    trainer = GroupedFakeTrainer(tmp_path, name="cosine", T=100)
    trainer.lr_group_specs = [trainer.lr_schedule_spec.for_group(c)
                              for c in ("unet", "text_encoder_1")]
    publish(trainer)

    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=8, events=[
        {"lr_scheduler": "constant", "length": 0, "groups": ["unet"]}])
    assert preview["results"][0]["warning"] is None
    curves = {c["group"]: c for c in preview["curves"]}
    assert set(curves) == {"unet", "text_encoder_1"}
    assert curves["unet"]["points"] != curves["unet"]["baseline_points"]
    assert curves["text_encoder_1"]["points"] == \
        curves["text_encoder_1"]["baseline_points"]


def test_the_preview_endpoint_is_documented_in_openapi():
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    path = spec["paths"]["/training/lr-schedule/preview"]
    assert set(path) == {"get", "post"}
    post = path["post"]
    assert set(post["responses"]) == {"200", "400", "404", "409"}
    body = post["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"].endswith("/LrSchedulePreviewRequest")
    assert post["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"].endswith("/LrSchedulePreviewResult")

    schemas = spec["components"]["schemas"]
    request = schemas["LrSchedulePreviewRequest"]["properties"]
    # The candidates are retarget bodies, not a second vocabulary.
    assert request["events"]["items"]["$ref"].endswith(
        "/LrScheduleRetargetRequest")
    assert request["n_points"]["default"] == LR_PREVIEW_DEFAULTS["n_points"]
    assert request["config"]["allOf"][0]["$ref"].endswith(
        "/LrSchedulePreviewConfig")



def test_every_derived_op_queues_as_a_retarget_and_lands_as_one(routes, tmp_path):
    for index, body in enumerate(({"op": "scale", "gain": 0.5},
                                  {"op": "hold"}, {"op": "undo"})):
        directory = tmp_path / f"op{index}"
        trainer = FakeTrainer(directory, name="cosine", T=100)
        trainer.seek(20)
        # `undo` needs something to undo; the others do not care.
        queue_retarget(directory)
        poll(trainer, global_step=20)
        trainer.seek(30)

        accepted = call_retarget(routes, directory, body=dict(body))
        assert accepted["command"] == control_rpc.RETARGET_COMMAND
        assert accepted["payload"]["op"] == body["op"]
        assert poll(trainer, global_step=30) == 1
        assert results_by_id(directory)[accepted["request_id"]]["result"] == \
            "applied"
        # D27/invariant 17: no new event kind, whatever the button said.
        assert {e["kind"] for e in trainer.lr_timeline.dump(10 ** 9)} == \
            {"total_steps", "retarget"}


def test_a_held_run_stays_at_the_multiplier_it_was_holding(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(20)
    held = trainer.multiplier(20)
    call_retarget(routes, tmp_path, body={"op": "hold"})
    assert poll(trainer, global_step=20) == 1
    assert trainer.lr == pytest.approx(BASE_LR * held)
    for step in (20, 21, 60, 100):
        assert trainer.multiplier(step) == pytest.approx(held)


def test_an_undo_appends_and_restores_the_previous_schedule(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(20)
    baseline = [trainer.multiplier(s) for s in (60, 80, 100)]

    queue_retarget(tmp_path)
    poll(trainer, global_step=20)
    assert [trainer.multiplier(s) for s in (60, 80, 100)] != baseline
    before = json.dumps(trainer.lr_timeline.dump(10 ** 9))

    trainer.seek(40)
    call_retarget(routes, tmp_path, body={"op": "undo", "length": 0})
    assert poll(trainer, global_step=40) == 1
    events = trainer.lr_timeline.dump(10 ** 9)
    assert json.dumps(events[:-1]) == before
    assert events[-1]["at"] == 40
    assert [trainer.multiplier(s) for s in (60, 80, 100)] == \
        pytest.approx(baseline)


def test_an_undo_with_nothing_to_undo_is_named_not_an_error(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(20)
    accepted = call_retarget(routes, tmp_path, body={"op": "undo"})
    poll(trainer, global_step=20)
    recorded = results_by_id(tmp_path)[accepted["request_id"]]
    assert recorded["result"] == "rejected_nothing_to_undo"
    assert recorded["error"]
    assert trainer.lr_timeline.dump(10 ** 9) == [
        e for e in trainer.lr_timeline.dump(10 ** 9)
        if e["kind"] == "total_steps"]


@pytest.mark.parametrize("bad,message", [
    ({"op": "sideways"}, "op"),
    ({"op": "hold", "lr_scheduler": "wsd"}, "lr_scheduler"),
    ({"op": "scale", "gain": 0.5, "anchor": "continue"}, "anchor"),
    ({"op": "scale"}, "gain"),
    ({"op": "hold", "gain": 2.0}, "gain"),
    ({"op": "undo", "gain": 2.0}, "gain"),
    ({"lr_scheduler": None}, "lr_scheduler"),
])
def test_the_endpoint_refuses_a_derived_op_it_cannot_honour(
        routes, tmp_path, bad, message):
    with pytest.raises(routes.HTTPException) as e:
        call_retarget(routes, tmp_path, body=dict(bad))
    assert status_of(e) == 400
    assert message in e.value.detail
    assert control_rpc.list_pending_requests(tmp_path) == []



def test_a_preview_reads_the_live_step_when_the_display_file_is_stale(
        routes, tmp_path):
    """`refresh_lr_schedule_status` writes on a STATE change, so a run that has
    had no LR command publishes `step` once, at its first batch. Dating a
    candidate there draws a curve the trainer will not follow -- and answers
    `applied` where the trainer will answer `rejected_backdated`."""
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    publish(trainer, global_step=0)
    trainer.seek(60)
    # No second publish: nothing changed state, so the trainer writes nothing.
    assert control_rpc.read_status(tmp_path)["step"] == 0

    preview = call_preview(routes, tmp_path, live_step=60, run_id=RUN_ID,
                           n_points=8, events=[{"op": "hold"}])
    assert (preview["step"], preview["position_source"]) == (60, "run_row")
    # What the trainer will actually hold at, not the 1.0 of step 0.
    held = trainer.multiplier(60)
    assert held != pytest.approx(1.0)
    assert preview["curves"][0]["points"][-1][1] == pytest.approx(held)

    call_retarget(routes, tmp_path, body={"op": "hold"})
    poll(trainer, global_step=60)
    assert trainer.multiplier(100) == pytest.approx(held)


def test_a_stale_file_no_longer_backdates_a_candidate(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    publish(trainer, global_step=0)
    trainer.seek(60)
    preview = call_preview(routes, tmp_path, live_step=60, run_id=RUN_ID,
                           n_points=8,
                           events=[dict(RETARGET_TO_WSD, at=50)])
    # `issued` is the live step, so a candidate at 50 is behind the run.
    assert preview["results"][0]["result"] == "rejected_backdated"
    queue_retarget(tmp_path, dict(RETARGET_TO_WSD, at=50))
    poll(trainer, global_step=60)
    assert list(results_by_id(tmp_path).values())[0]["result"] == \
        "rejected_backdated"


def test_a_stopped_run_still_previews_from_its_published_step(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(40)
    publish(trainer, global_step=40)
    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=8)
    assert (preview["step"], preview["position_source"]) == (40, "status_file")


def test_the_live_step_is_converted_to_the_scheduler_axis(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer._grad_accum_steps = 4
    publish(trainer, global_step=0)
    preview = call_preview(routes, tmp_path, live_step=240, run_id=RUN_ID,
                           n_points=4)
    assert preview["advance_interval"] == 4
    assert preview["step"] == 60



def test_the_endpoint_accepts_the_runs_own_internal_curve(routes, tmp_path):
    """Gap 1: whether an internal name is legal needs the run, so the endpoint
    queues it and the timeline scores it -- rather than a 400 that left a
    ReLoRA run unable to retarget onto its own curve at all."""
    accepted = call_retarget(routes, tmp_path,
                             body={"lr_scheduler": "relora"})
    assert accepted["payload"]["lr_scheduler"] == "relora"

    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    poll(trainer, global_step=0)
    assert results_by_id(tmp_path)[accepted["request_id"]]["result"] == \
        "rejected_unknown_scheduler"


def test_the_retarget_defaults_have_a_schema_endpoint(routes):
    import asyncio
    import yaml
    from api.param_defaults import LR_RETARGET_DEFAULTS

    payload = asyncio.run(routes.get_lr_retarget_defaults())
    for key, value in LR_RETARGET_DEFAULTS.items():
        assert payload[key] == value, key
    assert payload["n_points"] == LR_PREVIEW_DEFAULTS["n_points"]
    # Defaults only: the vocabularies are mirrored in the client and pinned by
    # lr_schedule_vocabulary_test, which fails a build instead of a select.
    assert set(payload) == set(LR_RETARGET_DEFAULTS) | {"n_points"}

    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    assert set(spec["paths"]["/schema/lr-retarget-defaults"]) == {"get"}


def _gas_trainer(tmp_path, gas=3, sched_total=333, global_total=1000):
    """A trainer whose global total is NOT a multiple of its accumulation, so
    the product the field exists to avoid gives a different answer."""
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer._grad_accum_steps = gas
    trainer.lr_schedule_spec = resolve_spec({}, warmup_steps=0,
                                            total_steps=sched_total,
                                            name="cosine")
    trainer.lr_timeline = ScheduleTimeline()
    trainer.lr_timeline.set_total_steps(sched_total)
    if global_total is not None:
        trainer._lr_global_total_steps = global_total
    return trainer


def test_the_status_file_carries_a_global_axis_total(tmp_path):
    """Gap 3: with gas > 1 the file's totals are all scheduler steps, leaving a
    reader to map the axis itself -- which would be a second definition of it,
    and a lossy one: to_scheduler_axis floors."""
    status = lr_schedule_status(_gas_trainer(tmp_path), global_step=0)
    assert status["advance_interval"] == 3
    assert status["effective_total_steps"] == 333
    assert status["global_total_steps"] == 1000
    # The whole point: the product is NOT the answer.
    assert status["global_total_steps"] != (status["effective_total_steps"]
                                            * status["advance_interval"])


def test_an_unknown_global_total_is_null_and_never_a_product(tmp_path):
    """§19.5.4-16: publishing the floored product would be exactly the wrong
    number, and 0 would render as a run that ends immediately."""
    status = lr_schedule_status(_gas_trainer(tmp_path, global_total=None),
                                global_step=0)
    assert status["global_total_steps"] is None


def test_a_re_anchored_total_drops_the_global_one(tmp_path):
    """The third write site's blind spot: a resume re-anchors on the SCHEDULER
    axis, so the configured global total is no longer this run's end."""
    trainer = _gas_trainer(tmp_path)
    trainer.lr_timeline.add("total_steps", at=100, value=400)
    status = lr_schedule_status(trainer, global_step=0)
    assert status["effective_total_steps"] == 400
    assert status["global_total_steps"] is None


def test_the_documented_result_codes_are_the_timeline_s():
    """Gap 4: the enum is presented as complete, so it has to be."""
    import yaml
    from core.training import lr_schedules
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    documented = set(spec["components"]["schemas"]["LrScheduleCommandResult"]
                     ["properties"]["result"]["enum"])
    assert {"ignored_unchanged", "ignored_duplicate_restart"} <= documented
    assert set(lr_schedules._REFUSED) <= documented
    assert {lr_schedules.RESULT_NOTHING_TO_UNDO,
            lr_schedules.RESULT_AMBIGUOUS_SCOPE,
            lr_schedules.RESULT_MISSING_GAIN} <= documented
    previewable = set(spec["components"]["schemas"]
                      ["LrSchedulePreviewEventResult"]["properties"]
                      ["result"]["enum"])
    assert previewable <= documented
    assert "applied" in previewable and "error" not in previewable
    # Every refusal a retarget can hit is drawable; the decay/cancel ones are
    # not, because only retargets are previewable.
    assert {c for c in lr_schedules._REFUSED
            if c not in ("rejected_during_warmup", "rejected_zero_length")
            } <= previewable


def test_the_op_survives_onto_the_result_and_the_queue_item(routes, tmp_path):
    """Gap 5: all four forms queue as `retarget` and land as one event kind, so
    the op is only recoverable if the result and the pending item carry it."""
    import asyncio
    accepted = call_retarget(routes, tmp_path, body={"op": "hold"})
    pending = asyncio.run(routes.get_lr_schedule_status(
        RUN_ID, db=_FakeDb(SimpleNamespace(output_dir=str(tmp_path)))))
    assert pending["pending"][0]["op"] == "hold"
    assert pending["pending"][0]["command"] == "retarget"

    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.seek(20)
    poll(trainer, global_step=20)
    recorded = results_by_id(tmp_path)[accepted["request_id"]]
    assert (recorded["command"], recorded["op"]) == ("retarget", "hold")
    # The event itself is still an undifferentiated retarget (invariant 17).
    assert {e["kind"] for e in trainer.lr_timeline.dump(10 ** 9)} == \
        {"total_steps", "retarget"}


def test_a_two_button_command_records_no_op(tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100, config=LATE_PLATEAU)
    trainer.seek(20)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    poll(trainer, global_step=20)
    assert list(results_by_id(tmp_path).values())[0]["op"] is None


def test_the_preview_header_is_null_when_the_curves_disagree(routes, tmp_path):
    """§19.5.4-8 one level up: the representative absorbs every scoped
    candidate, so a header taken from it named a schedule NO group is on."""
    trainer = GroupedFakeTrainer(tmp_path, name="cosine", T=100)
    trainer.lr_group_specs = [trainer.lr_schedule_spec.for_group(c)
                              for c in ("unet", "text_encoder_1")]
    publish(trainer)

    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=8, events=[
        {"lr_scheduler": "constant", "length": 0, "groups": ["unet"]}])
    assert preview["lr_scheduler"] is None
    assert preview["description"] is None
    curves = {c["group"]: c for c in preview["curves"]}
    assert curves["unet"]["lr_scheduler"] == "constant"
    assert curves["text_encoder_1"]["lr_scheduler"] == "cosine"
    assert "constant" in curves["unet"]["description"]


def test_the_preview_header_survives_when_the_curves_agree(routes, tmp_path):
    trainer = GroupedFakeTrainer(tmp_path, name="cosine", T=100)
    trainer.lr_group_specs = [trainer.lr_schedule_spec.for_group(c)
                              for c in ("unet", "text_encoder_1")]
    publish(trainer)
    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=8,
                           events=[{"lr_scheduler": "constant", "length": 0}])
    assert preview["lr_scheduler"] == "constant"
    assert preview["warmup_steps"] is not None
    assert {c["lr_scheduler"] for c in preview["curves"]} == {"constant"}


def test_an_ungrouped_preview_still_names_its_one_schedule(routes, tmp_path):
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    publish(trainer)
    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=8)
    assert preview["lr_scheduler"] == "cosine"
    assert preview["curves"][0]["lr_scheduler"] == "cosine"
    assert preview["floor_defaulted"] is not None


def test_the_preview_warning_carries_d40s_reason(routes, tmp_path):
    """D40: "this run has no lr_group_schedules" is false on a ReLoRA run that
    set one, and sends the operator hunting a config bug that does not exist."""
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    trainer.lr_group_specs_ignored_reason = (
        "lr_group_schedules is ignored on a ReLoRA run")
    publish(trainer)
    assert control_rpc.read_status(tmp_path)["ungrouped_reason"]

    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=4, events=[
        {"lr_scheduler": "constant", "length": 0, "groups": ["unet"]}])
    assert preview["results"][0]["warning"] == WARN_SELECTOR_ON_UNGROUPED_RUN
    assert "ignored on a ReLoRA run" in preview["warnings"][0]["message"]


def test_the_preview_result_contract_matches_what_is_returned(routes, tmp_path):
    import yaml
    trainer = FakeTrainer(tmp_path, name="cosine", T=100)
    publish(trainer)
    preview = call_preview(routes, tmp_path, run_id=RUN_ID, n_points=4)
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["LrSchedulePreviewResult"]
    assert set(schema["required"]) <= set(preview)
    assert set(preview) - set(schema["properties"]) == set()
    curve = spec["components"]["schemas"]["LrSchedulePreviewCurve"]["properties"]
    assert set(preview["curves"][0]) - set(curve) == set()
