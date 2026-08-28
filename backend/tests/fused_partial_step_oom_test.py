"""An OOM inside a fused backward can leave a half-applied optimizer step.

Under the fused paths -- ``_setup_fused_backward_pass`` (Adafactor / AdamW8bit /
the two ring-buffer optimizers, installed for ANY architecture whenever Block
Swap is on, and for SenseNova's full fine-tune at ``blocks_to_swap=0``) and
``_setup_fused_optimizer_groups`` -- there is no ``optimizer.step()``. Each
parameter is updated from its own post-accumulate-grad hook the moment its
gradient exists. So an out-of-memory error raised partway through
``backward()`` leaves the parameters whose hooks had fired a step ahead of the
rest.

What shipped: ``_forward_backward_with_oom_recovery`` catches that OOM, skips
the batch (or retries it micro-batched, which applies those updates a SECOND
time and returns success), and the run continues on the mixture. No rollback
exists, and no detector: the updated-parameter census is opt-in and, by design,
is not asserted for an abandoned batch -- which is precisely this batch.

The fix does not roll back (there is no affordable snapshot of tens of GiB of
weights). It counts the updates this backward has applied, and refuses to
continue if the count is non-zero when the OOM arrives:
``PartialOptimizerStepError``, distinct from ``NothingTrainedError`` and
``BucketsExhaustedError``, writing no checkpoint.

NEGATIVE CONTROLS (the shipped behaviour, recorded so the fix is provably a
change): ``test_negative_control_*``.

CPU-only, no CUDA: the OOM is a real exception raised from a real autograd
backward, through the real hooks, and classified by the real
``_classify_cuda_error``.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/fused_partial_step_oom_test.py -v
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.base_trainer import (  # noqa: E402
    BaseTrainer,
    BucketsExhaustedError,
    NothingTrainedError,
    PartialOptimizerStepError,
)
from core.training.optimizers.update_census import (  # noqa: E402
    UpdateCensus,
    applied_updates,
    reset_applied_updates,
)

BASE_TRAINER_SRC = (BACKEND / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8"
)

def _function(name: str) -> ast.FunctionDef:
    """The named ``BaseTrainer`` method, as a syntax tree."""
    tree = ast.parse(BASE_TRAINER_SRC)
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


SEED = 20260825
DIM = 4
LR = 1e-1
OOM = "CUDA out of memory. Tried to allocate 20.00 MiB"


# --------------------------------------------------------------------------
# A three-layer model whose backward dies between the layers.
# --------------------------------------------------------------------------


_TRAP: list = [None]  # what the trap raises; None means the ordinary OOM


class _DieInBackward(torch.autograd.Function):
    """Identity forward; raises on the way back."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        raise _TRAP[0] or RuntimeError(OOM)


class _Model(torch.nn.Module):
    """``l3`` is the LAST layer, so its gradient is the FIRST to arrive."""

    def __init__(self, trap: bool):
        super().__init__()
        torch.manual_seed(SEED)
        self.l1 = torch.nn.Linear(DIM, DIM, bias=False)
        self.l2 = torch.nn.Linear(DIM, DIM, bias=False)
        self.l3 = torch.nn.Linear(DIM, DIM, bias=False)
        self.trap = trap

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        if self.trap:
            # Between l2 and l3: l3's hook fires, then this raises, so l2 and
            # l1 never receive a gradient.
            x = _DieInBackward.apply(x)
        return self.l3(x)


class _Rig:
    """A model + a real fused-backward optimizer, with the real hooks."""

    def __init__(self, mode: str = "fused_backward", trap: bool = True):
        self.model = _Model(trap)
        self.before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        stub = SimpleNamespace(
            log_prefix="[Rig]",
            sensenova_four_phase=None,
            _partial_step_taint=None,
            _last_periodic_checkpoint_step=None,
            _periodic_save_every=500,
            use_grad_scaler=False,
            optimizer_schedule_free=False,
            optimizer_stochastic_rounding=False,
            optimizer_update_census=False,
            _update_census=None,
            _fused_grad_norm=None,
            use_fused_backward=False,
            fused_optimizer_groups=None,
            blocks_to_swap=1,
            num_optimizer_groups=0,
            learning_rate=LR,
            optimizer_cautious=False,
            optimizer_beta1=None,
            optimizer_beta2=None,
            optimizer_epsilon=None,
            optimizer_weight_decay=None,
            optimizer_use_radam=False,
            config={},
        )
        stub._applied_updates_now = BaseTrainer._applied_updates_now
        stub._flush_fused_group_partials = (
            lambda: BaseTrainer._flush_fused_group_partials(stub))
        params = list(self.model.parameters())
        if mode == "fused_backward":
            from transformers import Adafactor

            stub.optimizer = Adafactor(
                params, lr=LR, scale_parameter=False, relative_step=False,
                warmup_init=False,
            )
            stub._fused_backward_target_module = lambda: self.model
            BaseTrainer._setup_fused_backward_pass(stub, "adafactor")
        elif mode == "fused_groups":
            from core.training.optimizers.fused_optimizer_groups import (
                FusedOptimizerGroups,
            )

            # One group per parameter, so a group completes (and applies) as
            # soon as its single gradient arrives -- the same mid-backward
            # partiality with the OTHER fused implementation.
            optimizers = [torch.optim.AdamW([p], lr=LR) for p in params]
            groups = FusedOptimizerGroups(optimizers=optimizers, max_grad_norm=0.0)
            groups.register_hooks()
            stub.fused_optimizer_groups = groups
            stub.optimizer = optimizers[0]
        else:  # "unfused"
            stub.optimizer = torch.optim.AdamW(params, lr=LR)
        self.trainer = stub

    def backward(self):
        """One backward, armed and bracketed as ``_execute_forward_backward``
        does it. Everything but the three lines of control flow is the real
        method; the control flow itself is pinned by
        ``test_the_taint_brackets_exactly_the_backward``."""
        reset_applied_updates()  # stands in for the recovery-call entry reset
        BaseTrainer._reset_fused_group_counters(self.trainer)
        before = BaseTrainer._applied_updates_now()
        out = self.model(torch.ones(2, DIM))
        try:
            out.sum().backward()
            BaseTrainer._flush_fused_group_partials(self.trainer)
        except BaseException as exc:
            BaseTrainer._note_partial_step_taint(self.trainer, before, exc)
            raise

    def moved(self):
        return sorted(
            name
            for name, p in self.model.named_parameters()
            if not torch.equal(p.detach(), self.before[name])
        )


# --------------------------------------------------------------------------
# (A) The finding: the hooks fire mid-backward, so an OOM leaves a prefix
#     updated. Reproduced through the real machinery.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fused_backward", "fused_groups"])
def test_oom_mid_backward_updates_a_prefix_of_the_parameters(mode):
    rig = _Rig(mode)
    with pytest.raises(RuntimeError) as exc:
        rig.backward()
    assert "out of memory" in str(exc.value)
    # l3's gradient arrives first, so l3 is a step ahead of l2 and l1.
    assert rig.moved() == ["l3.weight"]
    assert applied_updates() == 1


def test_a_complete_backward_updates_every_parameter():
    """The rig is only interesting because the untrapped run is complete."""
    rig = _Rig("fused_backward", trap=False)
    rig.backward()
    assert rig.moved() == ["l1.weight", "l2.weight", "l3.weight"]
    assert applied_updates() == 3


def test_the_unfused_path_applies_nothing_during_backward():
    """Without hooks, a dead backward leaves every weight where it was --
    which is why the batch-skip is correct there and only there."""
    rig = _Rig("unfused")
    with pytest.raises(RuntimeError):
        rig.backward()
    assert rig.moved() == []
    assert applied_updates() == 0


# --------------------------------------------------------------------------
# Negative control: the shipped recovery, spliced back in.
# --------------------------------------------------------------------------


class _ShippedRecovery:
    """The pre-fix OOM handler: classify, skip the batch, keep going."""

    def __init__(self, rig):
        self.rig = rig
        self.batches_skipped = 0
        self.batches_trained = 0

    def run(self, batches: int):
        for _ in range(batches):
            try:
                self.rig.backward()
                self.batches_trained += 1
            except RuntimeError as e:
                if BaseTrainer._classify_cuda_error(e) != "oom":
                    raise
                self.batches_skipped += 1
        return "complete"


def test_negative_control_partial_step_is_skipped_and_the_run_continues():
    rig = _Rig("fused_backward")
    loop = _ShippedRecovery(rig)
    assert loop.run(3) == "complete"
    assert (loop.batches_skipped, loop.batches_trained) == (3, 0)
    # Three abandoned batches, and l3 took a step in every one of them while
    # l1 and l2 took none: the run continued on weights from three different
    # steps of the schedule.
    assert rig.moved() == ["l3.weight"]


def test_negative_control_the_real_recovery_without_the_guard_skips_the_batch():
    """The shipped path exactly: the real ``_forward_backward_with_oom_recovery``
    with the new refusal neutralised returns "batch skipped" and leaves one of
    the three parameters a step ahead."""
    rig = _Rig("fused_backward")
    stub = _recovery_stub(rig)
    stub._refuse_partial_fused_step = lambda *a: None
    assert _recover(stub) == (0.0, 0.0, 0.0, True)
    assert rig.moved() == ["l3.weight"]
    assert applied_updates() == 1


def test_negative_control_the_census_is_blind_to_it():
    """(B) The census WOULD see the missing updates -- it is simply not asked
    on an abandoned batch, which is by design and is exactly this batch."""
    rig = _Rig("fused_backward")
    census = UpdateCensus()
    params = list(rig.model.parameters())
    census.expect(params, {id(p): n for n, p in rig.model.named_parameters()})
    census.begin_step(True)
    census.record(params[-1])
    with pytest.raises(RuntimeError, match="census failed"):
        census.assert_complete("negative control")

    # ...and the call site guards it with exactly the flag an abandoned batch
    # sets.
    idx = BASE_TRAINER_SRC.index("self._update_census.assert_complete(")
    assert (
        "if self._update_census is not None and not cuda_error_skip:"
        in BASE_TRAINER_SRC[idx - 400:idx]
    )


# --------------------------------------------------------------------------
# (C) The fix, driven through the real recovery method.
# --------------------------------------------------------------------------


def _recovery_stub(rig=None, fused=True, batch=1, on_forward_backward=None):
    """A trainer-shaped object carrying only what the recovery path reads."""
    calls = {"execute": 0, "microbatch": 0, "cleanup": 0}

    def _execute(**kwargs):
        calls["execute"] += 1
        if on_forward_backward is not None:
            return on_forward_backward(calls["execute"])
        rig.backward()
        return 1.0, 1.0, 0.0

    def _microbatch(micro_bs, eff_bs, b):
        calls["microbatch"] += 1
        return 1.0, 1.0, 0.0

    stub = SimpleNamespace(
        log_prefix="[Recovery]",
        use_fused_backward=fused,
        fused_optimizer_groups=None,
        activation_dispatcher=None,
        activation_dispatch_enable=False,
        _actdispatch_oom=False,
        _batch_was_unfittable=False,
        _last_periodic_checkpoint_step=None,
        _execute_forward_backward=_execute,
        _microbatch_two_stage=_microbatch,
        _oom_recovery_cleanup=lambda: calls.__setitem__("cleanup", calls["cleanup"] + 1),
        _activation_dispatch_begin=lambda latents: (None, None),
        _activation_dispatch_end=lambda cm, info: None,
        _classify_cuda_error=BaseTrainer._classify_cuda_error,
        # CPU-only file: never probe real CUDA. _cuda_is_available is pinned
        # True so _refuse_save_after_partial_step always consults
        # _cuda_context_alive below instead of the real torch.cuda.is_available()
        # -- which would make these tests invert on a machine with no GPU
        # (torch.cuda.is_available() False -> ctx_alive forced True regardless
        # of the stub). Default "dead" (no salvage attempted) so a test that
        # doesn't care about the quarantine branch gets the simpler,
        # allocation-free path; tests that DO care override these explicitly.
        _cuda_is_available=lambda: True,
        _cuda_context_alive=lambda: False,
        _save_quarantined_partial_step_checkpoint=lambda step, epoch: False,
        calls=calls,
        batch=batch,
    )
    for name in ("_refuse_partial_fused_step", "_partial_fused_step_message",
                 "_resume_point_sentence", "_refuse_save_after_partial_step"):
        setattr(stub, name, (lambda m: lambda *a: getattr(BaseTrainer, m)(stub, *a))(name))
    return stub


def _recover(stub):
    return BaseTrainer._forward_backward_with_oom_recovery(
        stub,
        mnt_latents=torch.zeros(stub.batch, 4, 8, 8),
        mnt_text_embeddings=None,
        mnt_attention_mask=None,
        mnt_pooled_embeddings=None,
        timesteps=torch.zeros(stub.batch),
        debug_save_path=None,
        batch_captions=None,
        batch_reference_paths=None,
        alphas_cumprod_cached=None,
        use_condition_images=False,
        condition_images_batch=None,
        reference_latents_nested=None,
        min_split_batch_size=1,
    )


def test_a_partial_step_stops_the_run_instead_of_skipping_the_batch():
    rig = _Rig("fused_backward")
    stub = _recovery_stub(rig)
    with pytest.raises(PartialOptimizerStepError) as exc:
        _recover(stub)
    msg = str(exc.value)
    assert "after 1 parameter update(s) had already been applied" in msg
    assert "mixture of two steps" in msg
    # No ORDINARY checkpoint/state/optimizer/EMA is asserted -- not "no
    # checkpoint at all", since the weights may still be salvaged separately
    # to a quarantined artefact (see _refuse_save_after_partial_step).
    assert "no ordinary checkpoint, training state, optimizer, or EMA file" in msg
    assert rig.moved() == ["l3.weight"]


def test_a_partial_step_is_not_retried_so_it_cannot_double_apply():
    """batch=4 would take the micro-batch rung, whose retry re-runs the same
    samples -- applying l3's update a second time and returning SUCCESS."""
    rig = _Rig("fused_backward")
    stub = _recovery_stub(rig, batch=4)
    with pytest.raises(PartialOptimizerStepError):
        _recover(stub)
    assert stub.calls["microbatch"] == 0
    assert stub.calls["execute"] == 1


def test_zero_updates_so_far_is_still_the_old_safe_batch_skip():
    """An OOM in the forward (the common case) has applied nothing, so the
    shipped skip stays exactly as it was. No hand-reset here: production has to
    open the window itself, at entry (see the MNT test below)."""
    def _forward_oom(_n):
        raise RuntimeError(OOM)

    stub = _recovery_stub(fused=True, on_forward_backward=_forward_oom)
    assert _recover(stub) == (0.0, 0.0, 0.0, True)
    assert stub._batch_was_unfittable is True
    assert stub._actdispatch_oom is True


def test_a_completed_iteration_does_not_poison_the_next_ones_forward():
    """MNT > 1 calls the recovery once per iteration. Iteration 1 completes (3
    updates on the ledger); iteration 2's FORWARD then OOMs, which is the most
    common OOM site and has applied nothing. It must skip, as it always did --
    a per-backward window would have opened too late and killed the run."""
    rig = _Rig("fused_backward", trap=False)
    stub = _recovery_stub(rig)
    assert _recover(stub) == (1.0, 1.0, 0.0, False)   # iteration 1: complete
    assert applied_updates() == 3

    def _forward_oom(_n):
        raise RuntimeError(OOM)

    stub._execute_forward_backward = lambda **kw: _forward_oom(1)
    assert _recover(stub) == (0.0, 0.0, 0.0, True)    # iteration 2: skipped
    assert stub._batch_was_unfittable is True


def test_a_micro_split_is_never_started_under_a_fused_path():
    """The chunk sequence Codex names: the full batch OOMs in its forward (zero
    updates, so the refusal passes), and the retry would then run chunk 1 to
    completion -- applying its updates -- before chunk 2 could OOM with the
    ledger freshly reset. The retry is not allowed to start."""
    def _forward_oom(_n):
        raise RuntimeError(OOM)

    stub = _recovery_stub(fused=True, batch=4, on_forward_backward=_forward_oom)
    assert _recover(stub) == (0.0, 0.0, 0.0, True)
    assert stub.calls["microbatch"] == 0
    assert stub._batch_was_unfittable is True


def test_the_non_fused_path_still_micro_splits():
    """C is a fused-path rule only: without hooks a micro-split IS gradient
    accumulation and remains the right recovery."""
    def _forward_oom(n):
        if n == 1:
            raise RuntimeError(OOM)
        return 1.0, 1.0, 0.0

    stub = _recovery_stub(fused=False, batch=4, on_forward_backward=_forward_oom)
    assert _recover(stub) == (1.0, 1.0, 0.0, False)
    assert stub.calls["microbatch"] == 1


def test_the_proactive_splitter_refuses_both_fused_flavours():
    """It already refused ``use_fused_backward``; fused optimizer GROUPS step
    from their hooks too and were being split."""
    fn = _function("_activation_dispatch_begin")
    body = ast.unparse(fn)
    assert "if fused_backward_active(self):" in body
    assert 'getattr(self, "use_fused_backward", False)' not in body


def test_the_non_fused_path_is_unchanged_even_with_updates_on_the_ledger():
    """Without hooks the updates on the ledger came from an optimizer.step()
    AFTER a completed backward, not from a half-finished one."""
    from core.training.optimizers.update_census import note_update_applied

    def _forward_oom(_n):
        note_update_applied(7)
        raise RuntimeError(OOM)

    stub = _recovery_stub(fused=False, on_forward_backward=_forward_oom)
    assert _recover(stub) == (0.0, 0.0, 0.0, True)


def test_a_fatal_cuda_error_still_wins_over_the_partial_check():
    from core.training.base_trainer import FatalCudaError
    from core.training.optimizers.update_census import note_update_applied

    def _fatal(_n):
        note_update_applied(1)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    stub = _recovery_stub(fused=True, on_forward_backward=_fatal)
    with pytest.raises(FatalCudaError):
        _recover(stub)


def test_a_non_cuda_error_is_still_raised_untouched():
    def _boom(_n):
        raise RuntimeError("shapes do not match")

    stub = _recovery_stub(fused=True, on_forward_backward=_boom)
    with pytest.raises(RuntimeError, match="shapes do not match"):
        _recover(stub)


# --------------------------------------------------------------------------
# The message, and where the failure is allowed to go.
# --------------------------------------------------------------------------


def _message(stub, applied=12):
    stub._resume_point_sentence = lambda: BaseTrainer._resume_point_sentence(stub)
    return BaseTrainer._partial_fused_step_message(stub, applied, RuntimeError(OOM))


def test_message_names_the_checkpoint_to_resume_from():
    stub = _recovery_stub(fused=True)
    stub._last_periodic_checkpoint_step = 4000
    msg = _message(stub)
    assert "Resume from the last periodic checkpoint (step 4000)" in msg
    assert "12 parameter update(s)" in msg


def test_message_points_a_resumed_run_at_the_checkpoint_actually_loaded():
    """`train()` takes its OWN resume_from_checkpoint, so the __init__ attribute
    can be None on a resumed run; and "latest" is a request, not a path. The
    label is resolved once, preferring the path that was really loaded."""
    stub = _recovery_stub(fused=True)
    stub._resume_checkpoint_label = "out/checkpoint-5000.safetensors"
    msg = _message(stub, 3)
    assert "resumed from (out/checkpoint-5000.safetensors)" in msg
    assert "must be started again" not in msg
    assert "latest" not in msg


def test_message_does_not_blame_save_every_when_the_interval_simply_had_not_come():
    """`save_every=500` and an OOM at step herd 137: naming save_every=0 as the
    cause would be naming a cause that is not the cause."""
    stub = _recovery_stub(fused=True)
    stub._periodic_save_every = 500
    msg = _message(stub, 3)
    assert "did not reach its first checkpoint interval (save_every=500)" in msg
    assert "save_every=0" not in msg


def test_message_says_so_when_save_every_zero_left_no_checkpoint():
    """`save_every=0` is legal (445f5051), so "resume from the last checkpoint"
    can be advice with nothing behind it."""
    stub = _recovery_stub(fused=True)
    stub._periodic_save_every = 0
    msg = _message(stub, 3)
    assert "save_every=0 disables periodic checkpointing" in msg
    assert "Resume from the last periodic checkpoint" not in msg


def test_the_run_records_what_the_resume_sentence_needs():
    """All three inputs are set once per invocation, next to the counters."""
    fn = ast.unparse(_function("train"))
    assert "self._periodic_save_every = save_every_n_steps" in fn
    assert "self._resume_checkpoint_label" in fn
    assert "self._partial_step_taint = None" in fn
    assert "self._last_periodic_checkpoint_step = global_step" in fn


def test_the_groups_class_refuses_a_clip_it_cannot_perform():
    """The in-hook per-parameter clip was dead code (BaseTrainer passes 0.0); a
    reader could believe clipping happened there."""
    from core.training.optimizers.fused_optimizer_groups import FusedOptimizerGroups

    src = (BACKEND / "core" / "training" / "optimizers"
           / "fused_optimizer_groups.py").read_text(encoding="utf-8")
    assert "clip_grad_norm_" not in src
    with pytest.raises(ValueError, match="cannot clip by global norm"):
        FusedOptimizerGroups(optimizers=[], max_grad_norm=1.0)
    assert FusedOptimizerGroups(optimizers=[], max_grad_norm=0.0).max_grad_norm == 0.0


def test_the_error_is_distinct_from_the_two_added_with_it():
    assert issubclass(PartialOptimizerStepError, RuntimeError)
    assert not issubclass(PartialOptimizerStepError, NothingTrainedError)
    assert not issubclass(PartialOptimizerStepError, BucketsExhaustedError)
    assert not issubclass(NothingTrainedError, PartialOptimizerStepError)
    assert not issubclass(BucketsExhaustedError, PartialOptimizerStepError)


def test_it_writes_no_ordinary_emergency_checkpoint():
    """The ordinary emergency save is skipped; whether a QUARANTINED one runs
    is decided inside _refuse_save_after_partial_step (tested separately)."""
    idx = BASE_TRAINER_SRC.index("if isinstance(e, PartialOptimizerStepError):")
    block = BASE_TRAINER_SRC[idx:idx + 600]
    assert "_refuse_save_after_partial_step" in block
    assert "save_checkpoint" not in block
    assert "save_optimizer_state" not in block
    assert idx < BASE_TRAINER_SRC.index(
        "[EMERGENCY] Attempting to save emergency checkpoint"
    )


def test_the_batch_safety_net_does_not_swallow_it():
    """It is raised FROM an OOM, so the outer `except Exception` would classify
    it as recoverable and skip the batch -- the behaviour it exists to stop."""
    idx = BASE_TRAINER_SRC.index("except PartialOptimizerStepError:")
    assert idx < BASE_TRAINER_SRC.index("except FatalCudaError:")
    assert idx < BASE_TRAINER_SRC.index("except Exception as batch_error:")


def test_every_oom_handler_in_the_recovery_path_checks_it():
    """The offload retry and the micro-batch retry each catch their own OOM and
    fall through to the same skip."""
    fn = _function("_forward_backward_with_oom_recovery")
    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
    assert len(handlers) == 3
    for handler in handlers:
        body = ast.unparse(handler)
        assert "_refuse_partial_fused_step" in body, body[:200]


def test_the_ledger_window_is_the_whole_recovery_call_and_only_that():
    """The window has to open at ENTRY -- before the forward, not before the
    backward -- and must not reopen inside. A line-number assertion says
    nothing about ordering across calls, which is how the too-narrow window
    survived its first test."""
    fn = _function("_forward_backward_with_oom_recovery")
    statements = [n for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and (
                      (isinstance(n.func, ast.Name) and n.func.id == "reset_applied_updates")
                      or getattr(n.func, "attr", None) in (
                          "reset_applied_updates", "_activation_dispatch_begin",
                          "_execute_forward_backward", "_microbatch_two_stage"))]
    order = [getattr(c.func, "id", None) or c.func.attr for c in statements]
    assert order[0] == "reset_applied_updates"
    assert order[1] == "_activation_dispatch_begin"
    assert order.count("reset_applied_updates") == 1

    # And nowhere else in the trainer: a second reset would be a narrower
    # window by the back door.
    tree = ast.parse(BASE_TRAINER_SRC)
    resets = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and getattr(n.func, "id", None) == "reset_applied_updates"]
    assert len(resets) == 1
    assert "reset_applied_updates" not in ast.unparse(_function("_reset_fused_group_counters"))


# --------------------------------------------------------------------------
# (D) The OOM route is not the only way out of a half-applied backward.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [
    RuntimeError("shapes do not match"),          # not_cuda: re-raised before the refusal
    KeyError("exp_avg"),                          # not a RuntimeError at all
    KeyboardInterrupt(),                          # the interrupt save
    RuntimeError("CUDA error: an illegal memory access was encountered"),
])
def test_any_exception_out_of_the_backward_taints_the_step(exc):
    rig = _Rig("fused_backward")
    _TRAP[0] = exc
    try:
        with pytest.raises(type(exc)):
            rig.backward()
    finally:
        _TRAP[0] = None
    taint = rig.trainer._partial_step_taint
    assert taint is not None
    assert taint["applied"] == 1
    assert taint["kind"] == type(exc).__name__


def test_a_clean_backward_leaves_no_taint():
    rig = _Rig("fused_backward", trap=False)
    rig.backward()
    assert rig.trainer._partial_step_taint is None


def test_an_unfused_backward_leaves_no_taint():
    rig = _Rig("unfused")
    with pytest.raises(RuntimeError):
        rig.backward()
    assert rig.trainer._partial_step_taint is None


def test_a_tainted_step_writes_no_ordinary_checkpoint_when_ctx_is_dead():
    """CUDA context dead -> nothing GPU-side can be read back, so not even the
    quarantine salvage is attempted (stub default: _cuda_context_alive=False)."""
    stub = _recovery_stub(fused=True)
    stub.log_prefix = "[T]"
    stub._partial_step_taint = {"applied": 42, "kind": "KeyboardInterrupt", "detail": ""}
    stub._periodic_save_every = 500
    stub._last_periodic_checkpoint_step = 2000
    stub._resume_point_sentence = lambda: BaseTrainer._resume_point_sentence(stub)
    quarantine_calls = []
    stub._save_quarantined_partial_step_checkpoint = (
        lambda step, epoch: quarantine_calls.append((step, epoch)) or True
    )
    assert BaseTrainer._refuse_save_after_partial_step(stub, "The interrupt landed", 2137, 3) is True
    assert quarantine_calls == []
    # No taint -> the ordinary emergency/interrupt save is untouched.
    stub._partial_step_taint = None
    assert BaseTrainer._refuse_save_after_partial_step(stub, "x", 2137, 3) is False


def test_a_tainted_step_salvages_a_quarantined_checkpoint_when_ctx_is_alive():
    """CUDA context alive -> the tainted weights ARE worth salvaging, under the
    marker resume scanning excludes (see QUARANTINE_ENTRY_MARKER)."""
    stub = _recovery_stub(fused=True)
    stub.log_prefix = "[T]"
    stub._partial_step_taint = {"applied": 7, "kind": "KeyError", "detail": "boom"}
    stub._periodic_save_every = 500
    stub._last_periodic_checkpoint_step = 2000
    stub._resume_point_sentence = lambda: BaseTrainer._resume_point_sentence(stub)
    stub._cuda_context_alive = lambda: True
    quarantine_calls = []
    stub._save_quarantined_partial_step_checkpoint = (
        lambda step, epoch: quarantine_calls.append((step, epoch)) or True
    )
    assert BaseTrainer._refuse_save_after_partial_step(stub, "The interrupt landed", 2137, 3) is True
    # The ordinary save is still refused (return True); the quarantine helper
    # was invoked with this call's own step/epoch, not the interval markers.
    assert quarantine_calls == [(2137, 3)]


def test_both_save_handlers_consult_the_taint_before_writing():
    for anchor, save in (("except KeyboardInterrupt:", "self.save_checkpoint(step=global_step"),
                         ("except Exception as e:", "[EMERGENCY] Attempting to save")):
        start = BASE_TRAINER_SRC.index(anchor)
        gate = BASE_TRAINER_SRC.index("_refuse_save_after_partial_step", start)
        assert gate < BASE_TRAINER_SRC.index(save, start)


def test_the_taint_brackets_exactly_the_backward():
    """Scoped, so an exception raised outside the backward keeps its ordinary
    emergency save -- the mistake the too-wide ledger window made. Kept inline
    rather than extracted into a helper, so the existing single-source ordering
    invariants (arm-before-backward, flush-after-backward) still hold in the
    method they were written against."""
    fn = _function("_execute_forward_backward")
    tries = [n for n in fn.body if isinstance(n, ast.Try)]
    assert len(tries) == 1
    guard = tries[0]
    body = ast.unparse(guard.body)
    assert ".backward()" in body
    assert "self._flush_fused_group_partials()" in body
    # The arming and the count snapshot are OUTSIDE, before it.
    assert "_reset_fused_group_counters" not in body
    preceding = ast.unparse(fn.body[fn.body.index(guard) - 2:fn.body.index(guard)])
    assert "self._reset_fused_group_counters()" in preceding
    assert "_applied_before = self._applied_updates_now()" in preceding

    assert len(guard.handlers) == 1
    assert ast.unparse(guard.handlers[0].type) == "BaseException"
    handler = ast.unparse(guard.handlers[0])
    assert "self._note_partial_step_taint(_applied_before, _exc)" in handler
    assert handler.rstrip().endswith("raise")


def test_partial_step_salvages_weights_only_not_optimizer_or_state():
    """The decision recorded: a weights-only quarantined artefact is
    attempted (via the dedicated quarantine save, not the ordinary
    checkpoint path), while optimizer state, training state and EMA --
    all half-applied too -- are never saved for this step."""
    fn = ast.unparse(_function("_refuse_save_after_partial_step"))
    assert "_save_quarantined_partial_step_checkpoint" in fn
    assert "save_optimizer_state" not in fn
    assert "save_training_state" not in fn
    assert "_save_ema_checkpoint" not in fn

    # The quarantine save itself is weights-only: it reuses the ordinary
    # save_checkpoint() (arch-specific formats) but never the optimizer/
    # training-state/EMA saves alongside it.
    quarantine_fn = ast.unparse(_function("_save_quarantined_partial_step_checkpoint"))
    assert "save_checkpoint" in quarantine_fn
    assert "save_optimizer_state" not in quarantine_fn
    assert "save_training_state" not in quarantine_fn


def test_vision_encoder_sibling_is_excluded_from_every_resume_scan():
    """A quarantined (or EMA) save's Vision Encoder sibling file is named
    "{run_name}{suffix}_vision_encoder_step_N.safetensors" -- it carries
    neither QUARANTINE_ENTRY_MARKER nor EMA_ENTRY_MARKER, because both markers
    require "_step_" to immediately follow the suffix, and the VE filename
    inserts "vision_encoder_" first. Every _list_checkpoint_entries() call
    site that decides what resume can pick up must therefore also exclude
    "vision_encoder", not just the rotation-cleanup call site -- otherwise a
    VE sibling with a higher step number than its own excluded main
    checkpoint is handed to resume as if it were a live checkpoint."""
    exclude_tuples = [
        m.group(1) for m in re.finditer(r"exclude_substr=\(([^)]*)\)", BASE_TRAINER_SRC)
    ]
    # 4 resume/rotation scans, plus the two free-space ones added with
    # disk-space-aware retention (checkpoint_space).
    assert len(exclude_tuples) == 6, (
        f"expected exactly 6 _list_checkpoint_entries(exclude_substr=(...)) call "
        f"sites, found {len(exclude_tuples)}"
    )
    for tup in exclude_tuples:
        assert "vision_encoder" in tup, f"exclude_substr={tup} is missing 'vision_encoder'"
        assert "QUARANTINE_ENTRY_MARKER" in tup
        assert "EMA_ENTRY_MARKER" in tup


# --------------------------------------------------------------------------
# (A) Which architectures are affected: all of them, because none of this is
#     architecture-specific.
# --------------------------------------------------------------------------


def test_the_seam_is_defined_once_and_overridden_by_no_trainer():
    names = {
        "_forward_backward_with_oom_recovery",
        "_execute_forward_backward",
        "_reset_fused_group_counters",
        "_refuse_partial_fused_step",
        "_setup_fused_backward_pass",
        "_setup_fused_optimizer_groups",
    }
    definitions = {name: [] for name in names}
    for path in sorted((BACKEND / "core" / "training").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                definitions[node.name].append(path.name)
    for name, files in definitions.items():
        assert files == ["base_trainer.py"], (name, files)


def test_the_fused_install_is_not_gated_on_any_architecture():
    """Block Swap installs the hooks for whichever architecture asked for it;
    SenseNova's full fine-tune is an ADDITIONAL entry point, not the only one."""
    idx = BASE_TRAINER_SRC.index("elif optimizer_type.lower() in FUSED_BACKWARD_OPTIMIZERS:")
    branch = BASE_TRAINER_SRC[idx:BASE_TRAINER_SRC.index(
        "elif optimizer_type.lower() in self._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS:"
    )]
    assert "self._setup_fused_backward_pass(optimizer_type)" in branch
    assert "is_sensenova" not in branch
    # And neither the recovery path nor the refusal asks about the architecture.
    for name in ("_forward_backward_with_oom_recovery",
                 "_refuse_partial_fused_step",
                 "_partial_fused_step_message"):
        body = ast.unparse(_function(name))
        assert "is_sensenova" not in body and "self.is_" not in body, name


def test_every_training_capable_architecture_reaches_it():
    """All 13 run BaseTrainer.train; the two subclasses that wrap it wrap it in
    a `finally`, with no handler that could swallow the refusal."""
    from core.training.arch import ARCH_REGISTRY

    assert len(ARCH_REGISTRY) == 13
    for module in ("lora_trainer.py", "full_parameter_trainer.py"):
        tree = ast.parse((BACKEND / "core" / "training" / module).read_text(encoding="utf-8"))
        overrides = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "train"
        ]
        for node in overrides:
            outer = [n for n in node.body if isinstance(n, ast.Try)]
            assert outer and outer[0].finalbody
            # The wrapper itself catches nothing; the teardown inside its
            # `finally` has its own try, which cannot see the refusal.
            assert outer[0].handlers == []
            assert ast.unparse(outer[0].body) == "return super().train(*args, **kwargs)"


@pytest.mark.parametrize("optimizer_type", [
    "adafactor", "adamw8bit", "adamw8bit_ringbuffer", "lion8bit_ringbuffer",
])
def test_every_fused_backward_optimizer_records_its_updates(optimizer_type):
    """The ledger is fed from the shared ``record_param_update`` seam, so every
    optimizer that can be installed as a fused backward feeds it."""
    from core.training.base_trainer import FUSED_BACKWARD_OPTIMIZERS

    assert optimizer_type in FUSED_BACKWARD_OPTIMIZERS
    module = {
        "adafactor": "adafactor_fused",
        "adamw8bit": "adamw8bit_fused",
        "adamw8bit_ringbuffer": "adamw8bit_ringbuffer",
        "lion8bit_ringbuffer": "lion8bit_ringbuffer",
    }[optimizer_type]
    src = (BACKEND / "core" / "training" / "optimizers" / f"{module}.py").read_text(
        encoding="utf-8"
    )
    assert "record_param_update" in src


def test_the_fused_group_path_records_its_updates_too():
    src = (BACKEND / "core" / "training" / "optimizers"
           / "fused_optimizer_groups.py").read_text(encoding="utf-8")
    idx = src.index("self.optimizers[i].step()")
    assert "note_update_applied" in src[idx:idx + 400]
