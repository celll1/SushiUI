"""`445f5051` refused to let a run that trained nothing report success, but only
in `BaseTrainer.train`. The VAE fine-tuner and the tagger trainer have their
own loops; this file extends the guard to both.

TAGGER: the guard counts optimizer steps (`_optimizer_steps_completed`), not
backward passes. Four `continue`s skip a batch -- `batch is None`, NaN/Inf
loss, and a non-finite grad norm in each of the AMP/non-AMP branches -- and
all four must fail to advance the census, since a backward without a
following optimizer step leaves the weights untouched. `_epochs_entered`
exempts the one legitimate no-op: resuming past the final epoch.

VAE: the guard reads `global_step` alone (not a backward census), because
`global_step` only advances at the optimizer step, and a `.stop_training`
sentinel can land mid gradient-accumulation window -- backward(s) ran,
`global_step` did not.

CHECKPOINT POLICY (both loops): no checkpoint is written while nothing has
been optimized this run (or, for the VAE, ever this session's weights are
still the loaded ones). The tagger's counter is cumulative and never resets,
so one good batch anywhere in the run re-enables every later save.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/vae_and_tagger_nothing_trained_test.py -v
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

TAGGER_SRC = (BACKEND / "core" / "tagger" / "tagger_trainer.py").read_text(encoding="utf-8")
VAE_SRC = (BACKEND / "core" / "training" / "vae" / "vae_trainer.py").read_text(encoding="utf-8")


# ==========================================================================
# VAE: a runnable CPU stand-in for the loop.
#
# Everything that needs a real VAE, a database or a disk artifact is replaced;
# `train()`, `_train_micro_step`, `_clip_gradients` and `_finalize` are the
# shipped ones, so the counter's placement and the guard are what is under test.
# ==========================================================================

class _Dist:
    def __init__(self, t):
        self._t = t

    def mode(self):
        return self._t

    def sample(self):
        return self._t


class _TinyVae(torch.nn.Module):
    """One trainable scalar, differentiable through `decode`."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5))

    def encode(self, x):
        return SimpleNamespace(latent_dist=_Dist(x))

    def decode(self, z):
        return SimpleNamespace(sample=z * self.w)


def _tiny_loss(recon, pixels, posterior):
    loss = torch.nn.functional.mse_loss(recon, pixels)
    return loss, {"recon": float(loss.detach())}


class _StubVaeTrainer:
    """Mixin applied over the real VaeTrainer; see module docstring."""

    def _detect_resume_seq(self):
        pass

    def load_base_vae(self):
        self.vae = _TinyVae()

    def select_trainable(self):
        self.trainable_params = list(self.vae.parameters())
        self.trainable_names = ["w"]

    def build_optimizer(self):
        self.optimizer = torch.optim.SGD(self.trainable_params, lr=0.1)
        self.lr_scheduler = None

    def build_losses(self):
        self.loss_bank = _tiny_loss

    def init_ema(self):
        self.ema = None

    def _run_validation(self, step):
        self.validations.append(step)
        self._last_val_step = step

    def _log_step(self, *a, **k):
        pass

    def _queue_metrics(self, *a, **k):
        pass

    def _flush_metrics(self):
        self.flushed += 1

    def save_checkpoint(self, step, final=False):
        self.checkpoints.append((step, final))
        self._last_ckpt_step = step
        return self.checkpoints_dir / f"step_{step:06d}"

    def save_diffusers_vae(self, step):
        self.exports.append(step)
        return self.output_dir / "export"


def _make_vae_trainer(tmp_path, **overrides):
    from core.training.vae.vae_trainer import VaeTrainer

    cls = type("_TestVaeTrainer", (_StubVaeTrainer, VaeTrainer), {})
    cfg = {
        "seed": 1234,
        "dtype": "fp32",
        "resolution": 16,
        "validation_resolution": 16,
        "validation_num_images": 1,
        "crop_scale_policy": "downscale",
        "crop_scale_max_downscale": 0.0,
        "batch_size": 1,
        "num_workers": 0,
        "total_steps": 3,
        "gradient_accumulation_steps": 1,
        "save_every": 0,
        "validation_every": 0,
        "max_grad_norm": 1.0,
        "resume_from": "",
        "train_encoder": False,
        "acknowledge_latent_space_break": False,
        "train_decoder": True,
        "export_bare_ldm": False,
        "ema_enabled": False,
    }
    cfg.update(overrides)
    t = cls(cfg, output_dir=str(tmp_path / "run"), run_name="t", run_id=None)
    # The GPU is off-limits for tests and the loop reads self.device directly.
    t.device = torch.device("cpu")
    t.validations, t.checkpoints, t.exports, t.flushed = [], [], [], 0
    return t


def _dataset_items(tmp_path, n=4):
    from PIL import Image

    d = tmp_path / "images"
    d.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n):
        p = d / f"img_{i}.png"
        Image.new("RGB", (32, 32), (10 * i, 40, 200 - 10 * i)).save(p)
        items.append({"image_path": str(p)})
    return items


# --------------------------------------------------------------------------
# Negative control: what the VAE loop shipped for a zero-backward run.
# --------------------------------------------------------------------------

def _shipped_vae_tail(trainer):
    """The pre-fix tail of `VaeTrainer.train`, verbatim in behaviour."""
    if trainer.val_batch is not None and trainer._last_val_step != trainer.global_step:
        trainer._run_validation(trainer.global_step)
    if trainer._last_ckpt_step != trainer.global_step:
        trainer.save_checkpoint(trainer.global_step, final=True)
    trainer.save_diffusers_vae(trainer.global_step)
    trainer._flush_metrics()
    return trainer.stopped


def test_negative_control_vae_stop_at_step_zero_exported_untrained_weights(tmp_path):
    """A `.stop_training` sentinel present before the first micro-step left the
    loop with 0 backward passes and global_step 0 -- and the shipped tail then
    checkpointed and exported a full VAE of the untouched base weights."""
    t = _make_vae_trainer(tmp_path)
    t.val_batch = None
    t.stopped = True
    assert t.global_step == 0

    assert _shipped_vae_tail(t) is True
    assert t.checkpoints == [(0, True)]   # a checkpoint of weights nothing touched
    assert t.exports == [0]               # and a full diffusers VAE export of them


def test_vae_stop_at_step_zero_now_writes_nothing(tmp_path):
    t = _make_vae_trainer(tmp_path)
    (t.output_dir).mkdir(parents=True, exist_ok=True)
    (t.output_dir / ".stop_training").write_text("", encoding="utf-8")

    assert t.train(_dataset_items(tmp_path)) is True
    assert t.global_step == 0
    assert t.checkpoints == []
    assert t.exports == []
    assert t.flushed == 1


def test_vae_stop_mid_accumulation_window_writes_nothing(tmp_path):
    """Regression for the auditor's F1 reproduction: accum=2, the stop sentinel
    lands after the first micro-step of the window. One backward has run but
    no optimizer step has, so global_step is still 0 and nothing may be
    written."""
    t = _make_vae_trainer(tmp_path, total_steps=2, gradient_accumulation_steps=2)
    (t.output_dir).mkdir(parents=True, exist_ok=True)
    real_micro_step = t._train_micro_step
    calls = {"n": 0}

    def _stop_after_first_micro(batch, accum):
        result = real_micro_step(batch, accum)
        calls["n"] += 1
        if calls["n"] == 1:
            (t.output_dir / ".stop_training").write_text("", encoding="utf-8")
        return result

    t._train_micro_step = _stop_after_first_micro

    assert t.train(_dataset_items(tmp_path)) is True
    assert calls["n"] == 1
    assert t.global_step == 0
    assert t.checkpoints == []
    assert t.exports == []


def test_vae_run_that_trains_is_unchanged(tmp_path):
    t = _make_vae_trainer(tmp_path, total_steps=3)

    assert t.train(_dataset_items(tmp_path)) is False
    assert t.global_step == 3
    assert t.checkpoints == [(3, True)]
    assert t.exports == [3]
    # The optimizer really moved the one trainable parameter.
    assert float(t.vae.w.detach()) != 0.5


def test_vae_resume_at_target_is_still_a_legitimate_noop(tmp_path):
    """A resume at or past `total_steps` enters no iteration and trains nothing,
    exactly like BaseTrainer's empty epoch range. Its weights carry the earlier
    session's work, so it must still export -- and must not raise."""
    t = _make_vae_trainer(tmp_path, total_steps=3)
    original_load = t.load_base_vae

    def _load_and_resume():
        original_load()
        t.global_step = 3  # what load_checkpoint would have restored

    t.load_base_vae = _load_and_resume

    assert t.train(_dataset_items(tmp_path)) is False
    assert t.checkpoints == [(3, True)]
    assert t.exports == [3]


def test_vae_finalize_refuses_a_run_that_trained_nothing(tmp_path):
    from core.training.base_trainer import NothingTrainedError

    t = _make_vae_trainer(tmp_path)
    t.val_batch = None

    with pytest.raises(NothingTrainedError) as exc:
        t._finalize(total_steps=100)
    msg = str(exc.value)
    assert "no optimizer step" in msg
    assert "100 step(s)" in msg
    assert "No checkpoint or export was written" in msg
    assert t.checkpoints == []
    assert t.exports == []


def test_vae_finalize_never_discards_a_resumed_run(tmp_path):
    """global_step > 0 means an earlier session's work is in these weights, so
    the refusal must not fire however little this session trained."""
    t = _make_vae_trainer(tmp_path)
    t.val_batch = None
    t.global_step = 800

    assert t._finalize(total_steps=1000) is False
    assert t.checkpoints == [(800, True)]
    assert t.exports == [800]


def test_vae_does_not_count_backwards_it_never_reads():
    assert "_backwards_completed" not in VAE_SRC


# ==========================================================================
# Tagger
# ==========================================================================

class _HazardReferenceLoop:
    """A hand-written model of the "count the wrong event" hazard class, NOT an
    extraction of tagger_trainer.py (unlike `_shipped_vae_tail` below, which is
    verbatim). It exists to show the general shape of the bug in isolation;
    the actual tagger_trainer.py behaviour is pinned separately by the
    AST-structural tests further down and by `_tagger_stub`, both of which
    read the real source.

    The loop always skips `batch is None` and NaN/Inf loss. `skip="nonfinite_grad"`
    additionally skips negative-loss batches after their (would-be) backward,
    modelling F2's scenario: backward ran, the optimizer step was skipped.
    """

    def __init__(self, epochs=2, skip=None):
        self.epochs = epochs
        self.skip = skip
        self.global_step = 0
        self.checkpoints = []
        self.emitted = []
        self.epoch_losses = []

    def run(self, batches):
        for epoch in range(1, self.epochs + 1):
            epoch_loss, batches_processed = 0.0, 0
            for batch in batches:
                if batch is None:
                    continue                      # whole batch of corrupt images
                loss = batch
                if loss != loss:                  # NaN/Inf loss
                    continue
                if self.skip == "nonfinite_grad" and loss < 0:
                    continue                      # backward ran, step skipped
                self.global_step += 1
                epoch_loss += loss
                batches_processed += 1
            self.checkpoints.append(("latest", epoch))
            self.epoch_losses.append(epoch_loss / max(batches_processed, 1))
        self.emitted.append("completed")
        return {"total_steps": self.global_step}


def test_negative_control_tagger_every_batch_skipped_reported_success():
    nan = float("nan")
    loop = _HazardReferenceLoop(epochs=2)
    result = loop.run([None, nan, None, nan, None])

    assert loop.emitted == ["completed"]        # routes.py: status = "completed"
    assert result["total_steps"] == 0           # not one optimizer step
    assert loop.epoch_losses == [0.0, 0.0]      # a perfect-looking loss chart
    assert loop.checkpoints == [("latest", 1), ("latest", 2)]  # of untrained weights


def test_negative_control_tagger_every_batch_nonfinite_grad_reported_success():
    """F2's scenario: every batch backward()s fine but the grad norm is
    non-finite, so the optimizer step is skipped on every batch."""
    loop = _HazardReferenceLoop(epochs=1, skip="nonfinite_grad")
    result = loop.run([-1.0, -2.0, -3.0])
    assert loop.emitted == ["completed"]
    assert result["total_steps"] == 0
    assert loop.epoch_losses == [0.0]


def test_negative_control_tagger_partial_skips_are_a_normal_run():
    """The same reference loop is fine when anything at all trains -- the
    defect is specific to skipping everything."""
    loop = _HazardReferenceLoop(epochs=1)
    result = loop.run([None, 2.0, float("nan"), 4.0])
    assert result["total_steps"] == 2
    assert loop.epoch_losses == [3.0]


def _tagger_stub(epochs_entered=0, optimizer_steps_completed=0):
    from core.tagger.tagger_trainer import TaggerTrainer

    stub = SimpleNamespace(_epochs_entered=epochs_entered,
                           _optimizer_steps_completed=optimizer_steps_completed)
    stub._assert_trained_something = (
        lambda: TaggerTrainer._assert_trained_something(stub))
    return stub


def test_tagger_run_that_trained_nothing_is_failed_not_completed():
    from core.training.base_trainer import NothingTrainedError

    stub = _tagger_stub(epochs_entered=2, optimizer_steps_completed=0)
    with pytest.raises(NothingTrainedError) as exc:
        stub._assert_trained_something()
    msg = str(exc.value)
    assert "2 epoch(s)" in msg
    assert "no optimizer step" in msg
    assert "unreadable images" in msg and "NaN/Inf" in msg
    assert "no checkpoint" in msg


@pytest.mark.parametrize("optimizer_steps", [1, 7, 100000])
def test_tagger_run_that_trained_something_is_untouched(optimizer_steps):
    _tagger_stub(epochs_entered=3, optimizer_steps_completed=optimizer_steps)._assert_trained_something()


def test_tagger_resume_past_the_last_epoch_is_still_a_legitimate_noop():
    """`_save_training_state` records `epoch + 1` at every epoch boundary, so a
    resume after the final epoch skips every epoch and trains nothing. That is
    correct, and is the same exemption BaseTrainer makes."""
    _tagger_stub(epochs_entered=0, optimizer_steps_completed=0)._assert_trained_something()


def test_tagger_guard_uses_the_shared_exception_type():
    from core.training.base_trainer import NothingTrainedError

    assert issubclass(NothingTrainedError, RuntimeError)
    assert "NothingTrainedError" in TAGGER_SRC
    assert "NothingTrainedError" in VAE_SRC


# --------------------------------------------------------------------------
# Structural pins: the counter has to sit at the optimizer step (not the
# backward), the guard at the success exit, and no model write may happen
# while the counter is zero.
# --------------------------------------------------------------------------

def _tagger_train_ast():
    tree = ast.parse(TAGGER_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            for sub in ast.walk(node):
                if isinstance(sub, ast.For) and "loader_iter" in ast.dump(sub.iter):
                    return node
    raise AssertionError("TaggerTrainer.train not found")


def _parents(root):
    out = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _increment_lines(fn):
    lines = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Attribute)
                and node.target.attr == "_optimizer_steps_completed"):
            lines.add(node.lineno)
    return lines


def _optimizer_step_call_lines(fn):
    """`scaler.step(optimizer)` (AMP) and `optimizer.step()` (non-AMP) -- the
    two events that actually move the weights, as opposed to `.backward()`,
    which only computes gradients."""
    return sorted(
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "step" and isinstance(n.func.value, ast.Name)
        and n.func.value.id in ("scaler", "optimizer")
    )


def test_tagger_every_optimizer_step_counts_itself_and_nothing_else_does():
    """F2 regression: the census must count `optimizer.step()`/`scaler.step()`,
    not `.backward()` -- a non-finite grad norm backward()s and then skips the
    step via `continue`, and must not advance the counter."""
    fn = _tagger_train_ast()
    step_lines = _optimizer_step_call_lines(fn)
    increments = sorted(_increment_lines(fn))
    assert len(step_lines) == 2, step_lines   # AMP (scaler.step) and non-AMP (optimizer.step)
    assert len(increments) == 2, increments
    # Each increment is the statement immediately after its step call, so a
    # `continue` that skips the step (non-finite grad norm) cannot reach it.
    assert [s + 1 for s in step_lines] == increments


def test_tagger_epochs_entered_counts_only_epochs_the_loop_enters():
    assert TAGGER_SRC.count("self._epochs_entered += 1") == 1
    idx = TAGGER_SRC.index("self._epochs_entered += 1")
    before = TAGGER_SRC[:idx]
    # It sits after the resume skip, so a resume past the last epoch enters none.
    assert before.rstrip().endswith("continue")
    assert "if epoch < resume_epoch:" in before[-200:]


def test_tagger_guard_fires_before_the_completed_emit():
    assert TAGGER_SRC.count("self._assert_trained_something()") == 1
    guard = TAGGER_SRC.index("self._assert_trained_something()")
    emit = TAGGER_SRC.index('self._emit("completed"')
    assert guard < emit
    # The stop exit returns before the guard: a user stop is not a failure.
    stop_return = TAGGER_SRC.index('# event. This exit reports no success')
    assert stop_return < guard


def _census_guards(node, parents):
    """True when the census decides whether *node* runs: either it is inside an
    `if` that reads the counter, or an earlier sibling `if` on the counter left
    the block (break/return/continue/raise) before reaching it."""
    terminators = (ast.Break, ast.Return, ast.Continue, ast.Raise)
    cur = node
    while cur in parents:
        parent = parents[cur]
        if isinstance(parent, ast.If) and "_optimizer_steps_completed" in ast.dump(parent.test):
            return True
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(parent, field, None)
            if not isinstance(stmts, list) or cur not in stmts:
                continue
            for prior in stmts[:stmts.index(cur)]:
                if (isinstance(prior, ast.If)
                        and "_optimizer_steps_completed" in ast.dump(prior.test)
                        and prior.body and isinstance(prior.body[-1], terminators)):
                    return True
        cur = parent
    return False


_STATE_WRITE_FUNCS = {
    "_save_model_checkpoint", "_save_training_state", "_save_optimizer_state",
    "_save_vocabulary_snapshot", "_save_tag_metrics", "_save_ood_reference",
}


def test_tagger_writes_no_model_checkpoint_while_the_census_is_zero():
    """Every state-write call outside the batch loop must be decided by the
    census. The ones inside the batch loop are exempt: they are lexically
    dominated by the optimizer step in the same iteration."""
    fn = _tagger_train_ast()
    parents = _parents(fn)
    batch_loop = next(n for n in ast.walk(fn)
                      if isinstance(n, ast.For) and "loader_iter" in ast.dump(n.iter))

    sites, exempt = [], 0
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _STATE_WRITE_FUNCS):
            continue
        ancestors, cur = [], node
        while cur in parents:
            cur = parents[cur]
            ancestors.append(cur)
        if batch_loop in ancestors:
            exempt += 1
            continue
        sites.append((node.lineno, _census_guards(node, parents)))

    assert exempt == 6, "the step-based checkpoint block is the only in-loop write site"
    assert len(sites) == 26, sites
    assert all(guarded for _, guarded in sites), sites


def test_the_census_guard_test_can_fail():
    """Negative control for the check above: an unguarded write is detected."""
    fn = ast.parse(
        "def train(self):\n"
        "    for _x in loader_iter:\n"
        "        pass\n"
        "    _save_model_checkpoint(model, d, 'latest', m, mode)\n"
    ).body[0]
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_save_model_checkpoint")
    assert not _census_guards(call, _parents(fn))
