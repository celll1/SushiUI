"""Two architecture-independent training defects found by the U-2 resolution
campaign (`d1df3443`), and their fixes.

1. `save_every = 0` -- the obvious spelling of "never save" -- reached
   `global_step % save_every_n_steps` in `BaseTrainer.train` and raised
   `ZeroDivisionError` on the first step, whereupon the emergency handler wrote
   a full checkpoint. The run both failed and did the expensive thing it had
   been told not to do. 0 now means "never save periodically", which is what
   `tagger_trainer` and `vae_trainer` already meant by it, and every optional
   periodic interval goes through `periodic_intervals.due`.

2. An OOM inside bucket selection excluded the bucket, dropped every remaining
   batch, and let the run finish reporting success with nothing trained. There
   was no product-default detector for this: what caught it in the campaign was
   a probe's SHA-256 `moved_census`, and the trainer's own
   `optimizer_update_census` is both opt-in and skipped for abandoned batches --
   precisely the case that produces a no-op run. So the counter added here is a
   first line, not a backstop: the drop refuses when it empties the epoch, and a
   run that completes no backward pass is failed at BOTH exits (epoch
   exhaustion and the "reached target steps" early return, which skipped batches
   reach because they advance `global_step`). It is not OOM-specific -- the
   corrupted-image, no-valid-latents and missing-condition-image skips are
   covered by the same counter.

   The refusal is split by that counter. `_unfittable_buckets` grows DURING
   training, so losing the last fittable bucket after thousands of good steps is
   reachable; that raises `BucketsExhaustedError`, which falls through to the
   emergency save, because those weights are worth keeping and `save_every=0`
   (legalised above) may leave no other copy. Only a run that trained nothing
   raises `NothingTrainedError`, which writes no checkpoint.

NEGATIVE CONTROLS (the shipped behaviour, recorded so the fix is provably a
change): `test_negative_control_*`.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_periodic_and_nothing_trained_test.py -v
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from core.training.parameter_change_tracker import ParameterChangeTracker  # noqa: E402
from core.training.periodic_intervals import due, normalize_interval  # noqa: E402

BASE_TRAINER_SRC = (BACKEND / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------
# Negative control 1: what shipped for save_every = 0.
# --------------------------------------------------------------------------

class _ShippedLoop:
    """Replica of the shipped train-loop shape: an unguarded modulo inside the
    try whose `except Exception` writes an emergency checkpoint."""

    def __init__(self):
        self.checkpoints_written = []

    def run(self, save_every_n_steps, total_steps=3):
        try:
            for step in range(1, total_steps + 1):
                if step % save_every_n_steps == 0:          # the shipped line
                    self.checkpoints_written.append(("periodic", step))
        except Exception:
            self.checkpoints_written.append(("emergency", 0))
            raise


def test_negative_control_save_every_zero_crashed_then_saved_anyway():
    loop = _ShippedLoop()
    with pytest.raises(ZeroDivisionError):
        loop.run(save_every_n_steps=0)
    # Both halves of the defect: it died at step 1 ...
    assert loop.checkpoints_written == [("emergency", 0)]
    # ... and the only checkpoint written was the emergency one, i.e. the run
    # wrote a full checkpoint despite having been told never to save.


def test_negative_control_shipped_loop_is_fine_for_positive_intervals():
    loop = _ShippedLoop()
    loop.run(save_every_n_steps=2, total_steps=5)
    assert loop.checkpoints_written == [("periodic", 2), ("periodic", 4)]


# --------------------------------------------------------------------------
# Fix 1: 0 means never; positive intervals are unchanged.
# --------------------------------------------------------------------------

def test_due_never_fires_for_zero_and_never_divides_by_zero():
    for step in range(0, 200):
        assert due(step, 0) is False


def test_due_matches_the_shipped_modulo_for_every_positive_interval():
    """Nothing changes for a run that does not ask for 'never'.

    The scheduling code is architecture-independent -- one expression in
    `BaseTrainer.train` shared by every arch -- so equality over the interval
    and step ranges is the whole claim.
    """
    for interval in range(1, 65):
        for step in range(0, 400):
            assert due(step, interval) == (step % interval == 0), (step, interval)


def test_normalize_interval_optional_and_mandatory():
    assert normalize_interval(0) == 0
    assert normalize_interval(None) == 0
    assert normalize_interval(-7) == 0        # nonsense folds to "never"
    assert normalize_interval(500) == 500
    # Not-optional intervals (gradient accumulation): 0 means 1, not "off".
    assert normalize_interval(0, minimum=1) == 1
    assert normalize_interval(None, minimum=1) == 1
    assert normalize_interval(-3, minimum=1) == 1
    assert normalize_interval(4, minimum=1) == 4


def test_base_trainer_normalizes_every_interval_it_moduloes():
    """train() must clamp the four intervals before the loop reads them."""
    for name, minimum in (
        ("save_every_n_steps", ""),
        ("sample_every_n_steps", ""),
        ("debug_latents_every", ""),
        ("gradient_accumulation_steps", ", minimum=1"),
    ):
        assert f"{name} = normalize_interval({name}{minimum})" in BASE_TRAINER_SRC, name


def _named_divisors(src, ops):
    """Sorted unique divisor names for the given binary ops (literals skipped:
    a constant divisor cannot become zero)."""
    found = set()
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ops)):
            continue
        right = node.right
        if isinstance(right, ast.Name):
            found.add(right.id)
        elif isinstance(right, ast.Attribute):
            found.add(right.attr)
    return sorted(found)


def test_no_unguarded_periodic_division_remains_in_base_trainer():
    """Every division by a named interval in base_trainer is routed through
    due() or guarded by a positive check / clamped at construction.

    `//` counts, not only `%`: the next-interval arithmetic beside the sample
    and debug-latent modulos floor-divides by the same names, so a new interval
    introduced with `//` would be just as fatal and must not pass silently.
    """
    divisors = _named_divisors(BASE_TRAINER_SRC, (ast.Mod, ast.FloorDiv))
    # `%` and `//` by a name are the scheduling/grid operators. Pin the exact
    # set so a new interval added with EITHER fails this test.
    assert divisors == [
        "_danbooru_inj_interval",      # guarded by `self._danbooru_inj_interval > 0`
        "align",                       # bucket alignment, not a periodic interval
        "batch_size",                  # structural divisor
        "debug_latents_every",         # guarded by `debug_latents_every > 0`
        "ema_update_every",            # clamped max(1, ...) at construction
        "gradient_accumulation_steps",  # normalize_interval(..., minimum=1)
        "lh",                          # latent dims, guarded by `lh <= 0 or lw <= 0`
        "lw",
        "mnt",                         # multi_noise_timesteps, structural
        "sample_every_n_steps",        # guarded by `sample_every_n_steps > 0`
        "steps_per_epoch",             # structural divisor
        "vsf",                         # VAE scale factor, structural
    ], divisors


def test_no_interval_named_divisor_escapes_via_true_division():
    """`/` too, restricted to interval-shaped names.

    An exact-set pin is impossible for `/` because pathlib's `path / "name"`
    shares the operator, so this filters by name instead: nothing whose name
    reads as a periodic interval may be divided by outside the pinned set.
    """
    divisors = [
        n for n in _named_divisors(BASE_TRAINER_SRC,
                                   (ast.Mod, ast.FloorDiv, ast.Div))
        if any(tok in n for tok in ("every", "interval", "_freq"))
    ]
    assert divisors == [
        "_danbooru_inj_interval",
        "debug_latents_every",
        "ema_update_every",
        "sample_every_n_steps",
    ], divisors


def test_save_site_uses_due():
    assert "if interval_due(global_step, save_every_n_steps):" in BASE_TRAINER_SRC
    assert "if global_step % save_every_n_steps == 0:" not in BASE_TRAINER_SRC


def test_debug_latents_zero_is_guarded():
    assert (
        "if mnt_idx == 0 and debug_dir is not None and debug_latents_every > 0:"
        in BASE_TRAINER_SRC
    )


@pytest.mark.parametrize("interval", [0, None, -5])
def test_parameter_change_tracker_interval_zero_does_not_divide_by_zero(interval):
    """The real __init__ (no components, so the snapshot loop is a no-op)."""
    tracker = ParameterChangeTracker({}, interval)
    assert tracker.interval == 1
    assert tracker.compute(0) is None          # step 0 never reports
    assert tracker.compute(5) == {"update_norm": {}, "cumulative_drift": {}}


def test_parameter_change_tracker_keeps_a_positive_interval():
    assert ParameterChangeTracker({}, 100).interval == 100


def test_tagger_validate_every_zero_does_not_divide_by_zero():
    """The tagger loop guards its two save intervals but NOT `validate_every`,
    which was reachable as 0 through the API (the UI's min={1} is not a
    backstop) and raised at the end of the first epoch."""
    src = (BACKEND / "core" / "tagger" / "tagger_trainer.py").read_text(
        encoding="utf-8"
    )
    assert 'epoch % int(cfg.get("validate_every", 1)) == 0' not in src
    assert 'validate_every = max(1, int(cfg.get("validate_every", 1) or 1))' in src
    assert "if val_loader and epoch % validate_every == 0:" in src


def test_tagger_validate_every_is_bounded_at_the_api():
    from pydantic import ValidationError
    from api.routes import TaggerTrainingRunCreateRequest
    with pytest.raises(ValidationError):
        TaggerTrainingRunCreateRequest(validate_every=0)


def test_openapi_bounds_tagger_validate_every():
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    prop = (spec["components"]["schemas"]["TaggerTrainingRunCreateRequest"]
            ["properties"]["validate_every"])
    assert prop["minimum"] == 1


def test_relora_refuses_zero_merge_interval():
    src = (BACKEND / "core" / "training" / "relora_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "if int(relora_merge_every or 0) <= 0:" in src
    assert "relora_merge_every must be >= 1" in src


# --------------------------------------------------------------------------
# API surface: 0 is accepted and documented, negatives are refused.
# --------------------------------------------------------------------------

def _create_request_model():
    from api.routes import TrainingRunCreateRequest
    return TrainingRunCreateRequest


def _minimal_request(**overrides):
    body = {"training_method": "lora", "base_model_path": "x.safetensors"}
    body.update(overrides)
    return body


def test_save_every_zero_is_accepted_by_the_api():
    model = _create_request_model()
    req = model(**_minimal_request(save_every=0, sample_every=0))
    assert req.save_every == 0 and req.sample_every == 0


def test_negative_intervals_are_refused_at_run_creation():
    from pydantic import ValidationError
    model = _create_request_model()
    for field in ("save_every", "sample_every", "debug_latents_every"):
        with pytest.raises(ValidationError):
            model(**_minimal_request(**{field: -1}))
    for field in ("gradient_accumulation_steps", "param_tracking_interval",
                  "relora_merge_every"):
        with pytest.raises(ValidationError):
            model(**_minimal_request(**{field: 0}))


def test_openapi_documents_the_bounds():
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    assert props["save_every"]["minimum"] == 0
    assert "0 means never save periodically" in props["save_every"]["description"]
    assert props["sample_every"]["minimum"] == 0
    assert props["debug_latents_every"]["minimum"] == 0
    assert props["gradient_accumulation_steps"]["minimum"] == 1
    assert props["param_tracking_interval"]["minimum"] == 1
    assert props["relora_merge_every"]["minimum"] == 1


# --------------------------------------------------------------------------
# Negative control 2: what shipped when a bucket exclusion emptied the epoch.
# --------------------------------------------------------------------------

def _batch(w, h):
    return [({"bucket_width": w, "bucket_height": h}, object())]


class _ShippedBucketLoop:
    """Replica of the shipped drop-and-continue: filter, then loop, then
    report success regardless of how many batches survived."""

    def __init__(self, unfittable):
        self.unfittable = set(unfittable)
        self.trained = 0

    def run(self, batches, epochs=2):
        for _ in range(epochs):
            kept = [b for b in batches
                    if (b[0][0]["bucket_width"], b[0][0]["bucket_height"])
                    not in self.unfittable]
            for _b in kept:
                self.trained += 1
        return "complete"


def test_negative_control_all_batches_dropped_reported_success():
    loop = _ShippedBucketLoop({(1024, 1024)})
    assert loop.run([_batch(1024, 1024) for _ in range(4)]) == "complete"
    assert loop.trained == 0  # a run that trained nothing and said it finished


# --------------------------------------------------------------------------
# Fix 2: the drop refuses when it empties the epoch, and a run that completed
# no backward pass cannot report success.
# --------------------------------------------------------------------------

def _stub_trainer(unfittable=(), epochs_entered=0, backwards_completed=0,
                  batches_skipped=0):
    """A trainer-shaped object carrying only what the two guards read, so the
    real methods run without a model, dataset or optimizer."""
    from core.training.base_trainer import BaseTrainer
    stub = SimpleNamespace(
        _unfittable_buckets=set(unfittable),
        _epochs_entered=epochs_entered,
        _backwards_completed=backwards_completed,
        _batches_skipped=batches_skipped,
        log_prefix="[Test]",
    )
    for name in ("_nothing_trainable_message", "_buckets_exhausted_message"):
        setattr(stub, name, (lambda m: lambda n=0: getattr(BaseTrainer, m)(stub, n))(name))
    return stub


def _drop(stub, batches):
    from core.training.base_trainer import BaseTrainer
    return BaseTrainer._drop_unfittable_batches(stub, batches)


def _assert_trained(stub):
    from core.training.base_trainer import BaseTrainer
    return BaseTrainer._assert_trained_something(stub)


def test_dropping_every_batch_raises_instead_of_emptying_the_epoch():
    from core.training.base_trainer import NothingTrainedError
    stub = _stub_trainer(unfittable=[(1024, 1024)])
    with pytest.raises(NothingTrainedError) as exc:
        _drop(stub, [_batch(1024, 1024) for _ in range(4)])
    msg = str(exc.value)
    assert "1024x1024" in msg
    assert "All 4 batch(es)" in msg


def test_losing_the_last_bucket_after_training_does_not_discard_the_run():
    """`_unfittable_buckets` grows DURING training, so a run can lose its last
    fittable bucket after thousands of good steps. That is still a failure, but
    the weights are worth saving: it must NOT raise NothingTrainedError, whose
    handler deliberately writes no checkpoint -- with `save_every=0`, which this
    changeset made legal, that would destroy the whole run's work."""
    from core.training.base_trainer import BucketsExhaustedError, NothingTrainedError
    stub = _stub_trainer(unfittable=[(1024, 1024)], epochs_entered=4,
                         backwards_completed=500)
    with pytest.raises(BucketsExhaustedError) as exc:
        _drop(stub, [_batch(1024, 1024) for _ in range(4)])
    assert not isinstance(exc.value, NothingTrainedError)
    msg = str(exc.value)
    assert "500 backward pass(es) completed" in msg
    assert "emergency checkpoint is being written" in msg
    # The untrained-run message would be a lie here.
    assert "No parameter was updated" not in msg
    assert "no batch completed a backward" not in msg


def test_buckets_exhausted_falls_through_to_the_emergency_save():
    """Only NothingTrainedError short-circuits the emergency handler."""
    from core.training.base_trainer import BucketsExhaustedError, NothingTrainedError
    assert issubclass(BucketsExhaustedError, RuntimeError)
    assert not issubclass(BucketsExhaustedError, NothingTrainedError)
    idx = BASE_TRAINER_SRC.index("if isinstance(e, NothingTrainedError):")
    assert "BucketsExhaustedError" not in BASE_TRAINER_SRC[idx:idx + 700]


def test_partial_drop_still_returns_the_survivors():
    stub = _stub_trainer(unfittable=[(1024, 1024)])
    kept = _drop(stub, [_batch(1024, 1024), _batch(512, 512), _batch(768, 768)])
    assert [(b[0][0]["bucket_width"], b[0][0]["bucket_height"]) for b in kept] == [
        (512, 512), (768, 768)
    ]


def test_no_exclusions_is_a_pass_through():
    """Nothing changes for a run that never OOMs, on any architecture: the
    filter is skipped entirely and the same list object comes back."""
    stub = _stub_trainer()
    batches = [_batch(512, 512), _batch(1024, 1024)]
    assert _drop(stub, batches) is batches
    assert _drop(stub, []) == []


def test_exclusions_but_no_matching_batch_is_a_pass_through():
    stub = _stub_trainer(unfittable=[(1536, 1536)])
    batches = [_batch(512, 512), _batch(1024, 1024)]
    assert _drop(stub, batches) == batches


def test_run_that_trained_nothing_is_failed_not_completed():
    from core.training.base_trainer import NothingTrainedError
    stub = _stub_trainer(epochs_entered=3, backwards_completed=0)
    with pytest.raises(NothingTrainedError) as exc:
        _assert_trained(stub)
    assert "no batch completed a backward pass" in str(exc.value)


@pytest.mark.parametrize("skipped", [1, 7])
def test_non_oom_batch_skips_also_fail_the_run(skipped):
    """The guard is not OOM-specific: corrupted images, empty latent lists and
    missing condition images all `continue` past the backward, and a run made
    entirely of those must not report success either."""
    from core.training.base_trainer import NothingTrainedError
    stub = _stub_trainer(epochs_entered=1, backwards_completed=0,
                         batches_skipped=skipped)
    with pytest.raises(NothingTrainedError) as exc:
        _assert_trained(stub)
    msg = str(exc.value)
    assert f"{skipped} batch(es) were skipped" in msg
    assert "corrupted image" in msg and "missing condition" in msg


def test_run_that_trained_something_is_untouched():
    _assert_trained(_stub_trainer(epochs_entered=3, backwards_completed=1))
    _assert_trained(_stub_trainer(epochs_entered=100, backwards_completed=99999))
    # A run that trained AND skipped some batches is a normal run.
    _assert_trained(_stub_trainer(epochs_entered=2, backwards_completed=10,
                                  batches_skipped=3))


def test_empty_epoch_range_is_still_a_legitimate_noop():
    """A resume at or past the last epoch enters no epoch and must not fail."""
    _assert_trained(_stub_trainer(epochs_entered=0, backwards_completed=0))


@pytest.mark.parametrize(
    "method", ["_nothing_trainable_message", "_buckets_exhausted_message"]
)
def test_message_does_not_claim_the_card_is_exhausted(method):
    """The campaign's OOM was raised against a 0.72 memory fraction with
    9.95 GiB of the card free; neither message may say the hardware ran out."""
    from core.training.base_trainer import BaseTrainer
    stub = _stub_trainer(unfittable=[(1024, 1024)], backwards_completed=10)
    msg = getattr(BaseTrainer, method)(stub, 0)
    lowered = msg.lower()
    assert "set_per_process_memory_fraction" in msg
    assert "free vram on the card does not mean the budget was not reached" in lowered
    for claim in ("out of gpu memory", "gpu is out of memory", "card is out of",
                  "hardware is exhausted", "not enough vram"):
        assert claim not in lowered, claim
    # Actionable: says what to change.
    assert "blocks_to_swap" in msg and "resolution" in lowered


def test_train_calls_both_guards():
    assert "batches = self._drop_unfittable_batches(batches)" in BASE_TRAINER_SRC
    assert "self._backwards_completed += 1" in BASE_TRAINER_SRC
    assert "self._epochs_entered += 1" in BASE_TRAINER_SRC
    # Both exits: epoch exhaustion AND the "reached target steps" early return,
    # which skipped batches can reach because they advance global_step.
    assert BASE_TRAINER_SRC.count("self._assert_trained_something()") == 2
    idx = BASE_TRAINER_SRC.index("Reached target steps")
    assert "self._assert_trained_something()" in BASE_TRAINER_SRC[idx:idx + 700]


def test_counter_only_advances_for_a_completed_backward():
    """The increment must sit under `if not cuda_error_skip`, or an OOM-skipped
    batch would count as training."""
    idx = BASE_TRAINER_SRC.index("self._backwards_completed += 1")
    preceding = BASE_TRAINER_SRC[:idx].rsplit("\n", 2)[-2].strip()
    assert preceding == "if not cuda_error_skip:"


def test_every_whole_batch_skip_site_counts_itself():
    """Each `continue` that abandons a whole batch must bump _batches_skipped,
    so the failure message can say how many and why."""
    assert BASE_TRAINER_SRC.count("self._batches_skipped += 1") == 4


def test_nothing_trained_writes_no_emergency_checkpoint():
    """Defect 1's other half, generalised: a refusal must not trigger the
    expensive save it exists to avoid."""
    assert "if isinstance(e, NothingTrainedError):" in BASE_TRAINER_SRC
    idx = BASE_TRAINER_SRC.index("if isinstance(e, NothingTrainedError):")
    block = BASE_TRAINER_SRC[idx:idx + 700]
    assert "No checkpoint written: nothing was trained" in block
    assert "save_checkpoint" not in block
    # It must be reached before the emergency save path.
    assert idx < BASE_TRAINER_SRC.index("[EMERGENCY] Attempting to save emergency checkpoint")


def test_refusal_reaches_the_run_record_as_failed():
    """Every `trainer.train(...)` in train_runner is wrapped only by the
    top-level handler that marks the run failed, so the refusal is reported
    rather than swallowed into a green run."""
    src = (BACKEND / "core" / "training" / "train_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    handler_lines = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.tries = []

        def visit_Try(self, node):
            self.tries.append(node)
            self.generic_visit(node)
            self.tries.pop()

        def visit_Call(self, node):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "train"
                    and isinstance(func.value, ast.Name) and func.value.id == "trainer"):
                handler_lines.append(sorted(
                    h.lineno for t in self.tries for h in t.handlers
                ))
            self.generic_visit(node)

    Visitor().visit(tree)
    assert handler_lines, "no trainer.train() call found"
    # All diffusion train() calls share one handler set (KeyboardInterrupt +
    # Exception at module level); no per-call try swallows anything.
    assert len({tuple(h) for h in handler_lines}) == 1, handler_lines
    top = handler_lines[0]
    body = "\n".join(src.splitlines()[top[-1] - 1: top[-1] + 12])
    assert 'run.status = "failed"' in body
    assert "run.error_message = str(e)" in body


def test_nothing_trained_error_is_a_runtime_error():
    """Existing `except RuntimeError` sites keep catching it."""
    from core.training.base_trainer import NothingTrainedError
    assert issubclass(NothingTrainedError, RuntimeError)
