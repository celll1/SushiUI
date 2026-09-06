"""Guard: runtime LR-schedule commands (P2 of docs/guides/LR_SCHEDULER_DESIGN.md).

P1 built the timeline but left it with no way in from outside: the only events
it could get were a resume's ``total_steps`` comparison and the MNT
recomputation. P2 adds the file-RPC that carries "decay now" / "cancel decay"
from the API process into the trainer, and the display file the GET endpoint
reads back.

Fixed here:

* the transport is HOISTED, not copied -- ``training_sample_rpc`` and
  ``training_control_rpc`` share ``training_file_rpc``'s atomic write, read,
  age sort and ``owns``, and the sample queue's behaviour is unchanged;
* the two queues are separate directories-worth of files with separate caps: a
  sample request is not a command and neither claims the other's;
* a queued command is applied at the NEXT poll, and the poll is at the head of
  the batch rather than in the sampling block, because the fused optimizer
  paths step from a backward hook (§5.6);
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
    lr_decay_state_code,
    lr_schedule_status,
    poll_lr_schedule_commands,
    refresh_lr_schedule_status,
)
from core.training.lr_schedules import (  # noqa: E402
    STATE_BASE,
    STATE_DECAYING,
    STATE_FLOOR,
    STATE_RECOVERING,
    ScheduleTimeline,
    build_lr_scheduler,
    resolve_spec,
)

BASE_TRAINER_SRC = (BACKEND / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8")
SAMPLE_RPC_SRC = (BACKEND / "core" / "training" / "training_sample_rpc.py").read_text(
    encoding="utf-8")
CONTROL_RPC_SRC = (BACKEND / "core" / "training" / "training_control_rpc.py").read_text(
    encoding="utf-8")
PROCESS_SRC = (BACKEND / "core" / "training" / "training_process.py").read_text(
    encoding="utf-8")
ROUTES_SRC = (BACKEND / "api" / "routes.py").read_text(encoding="utf-8")

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


# ---------------------------------------------------------------------------
# The transport is shared, and the sample queue is unchanged
# ---------------------------------------------------------------------------

def test_the_primitives_are_hoisted_not_copied():
    assert sample_rpc.owns is file_rpc.owns
    assert control_rpc.owns is file_rpc.owns
    assert sample_rpc._atomic_write_json is file_rpc.atomic_write_json
    assert sample_rpc._read_json is file_rpc.read_json
    assert sample_rpc.make_request_id is file_rpc.make_request_id
    for definition in ("def _atomic_write_json", "def _read_json",
                       "def _sorted_by_age", "def owns"):
        assert definition not in SAMPLE_RPC_SRC, definition
        assert definition not in CONTROL_RPC_SRC, definition


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


# ---------------------------------------------------------------------------
# Result codes, per request_id
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# .lr_schedule.json: never a state the run has left
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Where the poll sits, and what is cleared before a spawn
# ---------------------------------------------------------------------------

def test_the_poll_is_at_the_batch_head_not_in_the_sampling_block():
    assert BASE_TRAINER_SRC.count("poll_lr_schedule_commands(self, global_step)") == 1
    stop = BASE_TRAINER_SRC.index("stop_flag_file.unlink()  # Clean up flag file")
    poll_at = BASE_TRAINER_SRC.index("poll_lr_schedule_commands(self, global_step)")
    sampling = BASE_TRAINER_SRC.index(
        "on_demand_request = self._claim_on_demand_sample_request()")
    assert stop < poll_at < sampling


def test_commands_are_claimed_even_while_a_stop_is_pending():
    # The sample claim re-checks the stop flag because a generation would delay
    # the stop; a command is a list append and rides out on the same checkpoint.
    body = CONTROL_RPC_SRC + BASE_TRAINER_SRC[
        BASE_TRAINER_SRC.index("def poll_lr_schedule_commands"):
        BASE_TRAINER_SRC.index("def lr_decay_state_code")]
    assert ".stop_training" not in body


def test_stale_commands_are_cleared_before_the_next_run_is_spawned():
    assert "training_control_rpc import clear_all" in PROCESS_SRC
    spawn = PROCESS_SRC.index("asyncio.create_subprocess_exec")
    assert PROCESS_SRC.index("training_control_rpc import clear_all") < spawn


def test_clear_all_removes_commands_and_results_but_not_the_state(tmp_path):
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    control_rpc.write_result(tmp_path, "done", {"result": "applied"})
    control_rpc.write_status(tmp_path, {"state": "floor"})

    assert control_rpc.clear_all(tmp_path) == 2
    assert control_rpc.list_pending_requests(tmp_path) == []
    assert control_rpc.list_results(tmp_path) == []
    # The last known state is what GET serves for a stopped run.
    assert control_rpc.read_status(tmp_path) == {"state": "floor"}


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

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


def test_the_post_is_fire_and_forget_and_refuses_a_dead_run():
    body = ROUTES_SRC[ROUTES_SRC.index("async def queue_lr_schedule_command"):
                      ROUTES_SRC.index("async def get_lr_schedule_status")]
    assert "status_code=202" in ROUTES_SRC[
        ROUTES_SRC.index('@router.post("/training/runs/{run_id}/lr-schedule"'):
        ROUTES_SRC.index("async def queue_lr_schedule_command")]
    assert "status_code=409" in body
    assert "status_code=429" in body
    # Nothing in the handler waits on the trainer; it writes a file and returns.
    assert "await" not in body


def test_the_get_reads_the_files_and_works_for_a_stopped_run():
    body = ROUTES_SRC[ROUTES_SRC.index("async def get_lr_schedule_status"):
                      ROUTES_SRC.index('@router.get("/training/runs/{run_id}/samples")')]
    assert "read_status(output_dir)" in body
    # No live process: fall back to the run row's directory rather than 409.
    assert "run.output_dir" in body
    assert "409" not in body
