"""The instrument the MoT phase-swap arms are measured with, on the CPU.

The quantity under test is a TRANSFER COST, and the two ways to get it wrong are
both bookkeeping rather than physics:

* ``drain_transfer_stats()`` RESETS the evictor's accumulators and ``base_trainer``
  already calls it once per step. A probe that drained a second time would read
  zeros on one side and blank the trainer's own charted series on the other, and
  nothing about the resulting numbers would look wrong. The recorder therefore
  observes what the trainer logged instead of pulling again, and that is what is
  pinned here -- with the evictor's drain count as the assertion.
* the first swaps of a run are not steady state (every pinned staging block is a
  fresh ``cudaHostAlloc``), and SENSENOVA_TRAINING_DESIGN.md 8.6 already retracted
  one number measured without excluding them. The warmup exclusion is arithmetic
  and is checked as arithmetic.

Nothing here loads a checkpoint or touches CUDA; ``_clock`` is substituted so the
step-boundary ``torch.cuda.synchronize`` never initializes a context.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.probes import sensenova_full_finetune as probe
from core.training.probes.sensenova_real_checkpoint import (
    EXIT_SMOKE_STEPS,
    EXIT_SMOKE_WIDTH,
    VRAM_GATE_FRACTION,
)


class _StubEvictor:
    """The accumulate-and-reset contract of the real ``drain_transfer_stats``."""

    def __init__(self):
        self.d2h_seconds = 0.0
        self.h2d_seconds = 0.0
        self.d2h_bytes = 0
        self.h2d_bytes = 0
        self.overlap_active = True
        self.drains = 0

    def charge(self, d2h_s, h2d_s, d2h_b, h2d_b):
        self.d2h_seconds += d2h_s
        self.h2d_seconds += h2d_s
        self.d2h_bytes += d2h_b
        self.h2d_bytes += h2d_b

    def drain_transfer_stats(self):
        self.drains += 1
        stats = {
            "d2h_seconds": self.d2h_seconds,
            "h2d_seconds": self.h2d_seconds,
            "d2h_bytes": self.d2h_bytes,
            "h2d_bytes": self.h2d_bytes,
            "overlap_active": self.overlap_active,
        }
        self.d2h_seconds = 0.0
        self.h2d_seconds = 0.0
        self.d2h_bytes = 0
        self.h2d_bytes = 0
        return stats


class _StubTrainer:
    """base_trainer's per-step drain block, reduced to the part that matters."""

    def __init__(self, evictor):
        self.sensenova_phase_evictor = evictor
        self.logged = []

    def log_extra_metric(self, name, value):
        self.logged.append((name, value))

    def run_step(self):
        drained = self.sensenova_phase_evictor.drain_transfer_stats()
        self.log_extra_metric("sn_d2h_s", drained["d2h_seconds"])
        self.log_extra_metric("sn_h2d_s", drained["h2d_seconds"])
        self.log_extra_metric("sn_d2h_gib", drained["d2h_bytes"] / 2 ** 30)
        self.log_extra_metric("sn_h2d_gib", drained["h2d_bytes"] / 2 ** 30)
        self.log_extra_metric(
            "sn_swap_overlap", 1.0 if drained["overlap_active"] else 0.0
        )


@pytest.fixture
def frozen_clock(monkeypatch):
    ticks = [0.0]

    def advance(seconds):
        ticks[0] += seconds

    monkeypatch.setattr(probe, "_clock", lambda: ticks[0])
    return advance


def test_the_recorder_reads_the_trainers_drain_instead_of_draining_again(frozen_clock):
    evictor = _StubEvictor()
    trainer = _StubTrainer(evictor)
    recorder = probe._StepTransferRecorder()
    recorder.install(trainer)
    recorder.start()

    for step in (1, 2, 3):
        # Two swaps per four-phase step, which is why the drain is once per step
        # and not once per transition.
        evictor.charge(0.10 * step, 0.20 * step, 2 ** 30, 2 * 2 ** 30)
        evictor.charge(0.10 * step, 0.20 * step, 2 ** 30, 2 * 2 ** 30)
        frozen_clock(1.0)
        trainer.run_step()
        recorder.close_step(step)

    assert evictor.drains == 3, "the evictor was drained more than once per step"
    assert [s["step"] for s in recorder.steps] == [1, 2, 3]
    assert [s["wall_s"] for s in recorder.steps] == [1.0, 1.0, 1.0]
    assert [round(s["sn_d2h_s"], 6) for s in recorder.steps] == [0.2, 0.4, 0.6]
    assert [round(s["sn_h2d_s"], 6) for s in recorder.steps] == [0.4, 0.8, 1.2]
    assert [s["sn_d2h_gib"] for s in recorder.steps] == [2.0, 2.0, 2.0]
    assert [s["sn_h2d_gib"] for s in recorder.steps] == [4.0, 4.0, 4.0]
    assert [s["sn_swap_overlap"] for s in recorder.steps] == [1.0, 1.0, 1.0]
    # The trainer still got everything it logs; the recorder is a tee, not a tap.
    assert [name for name, _v in trainer.logged].count("sn_d2h_s") == 3


def test_a_step_that_swapped_nothing_is_recorded_as_nothing(frozen_clock):
    """The eviction-OFF arm: no evictor, so no series, and the summary says so."""
    recorder = probe._StepTransferRecorder()
    recorder.start()
    for step in (1, 2):
        frozen_clock(0.5)
        recorder.close_step(step)
    assert [s["sn_d2h_s"] for s in recorder.steps] == [None, None]
    summary = recorder.summary(warmup=0)
    assert summary["d2h_s"]["n"] == 0
    assert summary["transfer_share_of_step"]["n"] == 0
    assert any("no evictor transfer series" in n for n in summary["notes"])


def test_the_steady_state_summary_excludes_the_warmup_steps(frozen_clock):
    evictor = _StubEvictor()
    trainer = _StubTrainer(evictor)
    recorder = probe._StepTransferRecorder()
    recorder.install(trainer)
    recorder.start()

    # Step 1 is the unrepresentative one: a whole second of cudaHostAlloc.
    walls = [3.0, 1.0, 1.0, 2.0]
    d2h = [2.0, 0.2, 0.4, 0.6]
    for index, (wall, seconds) in enumerate(zip(walls, d2h), start=1):
        evictor.charge(seconds, seconds, 2 ** 30, 2 ** 30)
        frozen_clock(wall)
        trainer.run_step()
        recorder.close_step(index)

    summary = recorder.summary(warmup=1)
    assert summary["warmup_steps"] == 1
    assert summary["steady_state_steps"] == 3
    assert summary["step_wall_s"] == {"n": 3, "median": 1.0, "min": 1.0, "max": 2.0}
    assert summary["d2h_s"]["median"] == pytest.approx(0.4)
    assert summary["d2h_s"]["max"] == pytest.approx(0.6)
    # (d2h + h2d) / step_wall, per step then medianed: 0.4/1, 0.8/1, 1.2/2.
    assert summary["transfer_share_of_step"]["median"] == pytest.approx(0.6)
    assert summary["overlap_active_all_steps"] is True
    assert summary["notes"] == []

    # A warmup that swallows the run reports nothing and SAYS it reports nothing,
    # rather than medianing an empty steady state into a null nobody reads.
    swallowed = recorder.summary(warmup=4)
    assert swallowed["steady_state_steps"] == 0
    assert any("no steady-state steps" in n for n in swallowed["notes"])


def test_a_step_that_straddled_a_downgrade_is_not_reported_as_overlapped(frozen_clock):
    evictor = _StubEvictor()
    trainer = _StubTrainer(evictor)
    recorder = probe._StepTransferRecorder()
    recorder.install(trainer)
    recorder.start()
    for step, overlapped in ((1, True), (2, False)):
        evictor.overlap_active = overlapped
        evictor.charge(0.1, 0.1, 2 ** 30, 2 ** 30)
        frozen_clock(1.0)
        trainer.run_step()
        recorder.close_step(step)
    summary = recorder.summary(warmup=0)
    assert summary["overlap_active_all_steps"] is False
    assert summary["overlap_active_any_step"] is True


def _args(*extra):
    argv = ["probe", "--arm", "train", "--model-path", "m", "--workdir", "w",
            "--out", "o", *extra]
    saved, sys.argv = sys.argv, argv
    try:
        return probe._parse_args()
    finally:
        sys.argv = saved


def test_an_invocation_that_names_none_of_the_new_flags_is_unchanged():
    args = _args()
    assert args.vram_fraction == VRAM_GATE_FRACTION
    assert args.overlap_transfer is None      # -> the key is never set at all
    assert args.phase_eviction is None        # -> follows --four-phase
    assert args.label is None
    assert args.steps == EXIT_SMOKE_STEPS
    assert args.resolution == EXIT_SMOKE_WIDTH
    assert probe._resolve_phase_eviction(args) is False
    assert probe._vram_fraction(args) == VRAM_GATE_FRACTION


def test_phase_eviction_follows_four_phase_until_it_is_named():
    assert probe._resolve_phase_eviction(_args("--four-phase")) is True
    assert probe._resolve_phase_eviction(_args("--phase-eviction")) is True
    assert probe._resolve_phase_eviction(_args("--no-phase-eviction")) is False
    with pytest.raises(ValueError, match="cannot be combined with --four-phase"):
        probe._resolve_phase_eviction(_args("--four-phase", "--no-phase-eviction"))


def test_the_vram_gate_override_is_an_override_and_not_a_new_default():
    assert probe._vram_fraction(_args("--vram-fraction", "0.98")) == pytest.approx(0.98)
    assert VRAM_GATE_FRACTION == 0.72


def test_the_stdout_summary_survives_a_result_that_has_none_of_it():
    text = probe._format_summary({"arm": "train", "failures": ["boom"]})
    assert "SENSENOVA SWAP MEASUREMENT" in text
    assert "n/a" in text
    assert "FAILURE: boom" in text
