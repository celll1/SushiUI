"""U-2-4 follow-up: sharing ONE boundary cut across an MNT window (8.3.5).

    venv/Scripts/python.exe -m pytest backend/tests/sensenova_shared_prefix_window_test.py -v

Every claim here is paired with the shipped per-iteration behaviour as its
negative control, because the two are both correct and differ only in WHAT they
train. ``float64`` keeps the arithmetic checkable without a GPU.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.training.optimizers.update_census import UpdateCensus  # noqa: E402
from core.training.ops import sensenova_ops  # noqa: E402
from core.training.sensenova_four_phase import (  # noqa: E402
    SenseNovaFourPhaseBackward,
    install_four_phase_backward,
    understanding_deferred_parameters,
)

_LAYERS = 2
_WIDTH = 4


class UnderstandingStage(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.k = nn.ModuleList(
            [nn.Linear(_WIDTH, _WIDTH, bias=False, dtype=torch.float64)
             for _ in range(_LAYERS)]
        )
        self.v = nn.ModuleList(
            [nn.Linear(_WIDTH, _WIDTH, bias=False, dtype=torch.float64)
             for _ in range(_LAYERS)]
        )

    def forward(self, tokens):
        return [(k(tokens), v(tokens)) for k, v in zip(self.k, self.v)]


class GenerationStage(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.blocks = nn.ModuleList(
            [nn.Linear(_WIDTH, _WIDTH, bias=False, dtype=torch.float64)
             for _ in range(_LAYERS)]
        )

    def forward(self, image, cache):
        hidden = image
        for block, layer in zip(self.blocks, cache.layers):
            attention = (
                torch.softmax(hidden @ layer.keys.transpose(-1, -2), dim=-1)
                @ layer.values
            )
            hidden = torch.tanh(block(hidden) + attention)
        return hidden


class _Cache:
    def __init__(self, layers):
        self.layers = layers


class _Layer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class FakeTrainer:
    """Enough of the trainer for the four-phase context, plus a fused-hook stand-in.

    ``apply_updates`` is the post-accumulate-grad hook this route relies on:
    a parameter moves when, and only when, its own gradient arrives.
    """

    def __init__(self):
        self.und = UnderstandingStage()
        self.gen = GenerationStage()
        self.transformer = None
        self.log_prefix = "[Test]"
        self.und_update_count = {name: 0 for name, _ in self.und.named_parameters()}

    def apply_updates(self, lr=0.1):
        for name, p in self.und.named_parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)
                p.grad = None
                self.und_update_count[name] += 1
        for p in self.gen.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)
                p.grad = None

    def und_weights(self):
        return {n: p.detach().clone() for n, p in self.und.named_parameters()}


def _tokens_and_targets(n):
    g = torch.Generator().manual_seed(7)
    tokens = torch.randn(5, _WIDTH, generator=g, dtype=torch.float64)
    images = [torch.randn(6, _WIDTH, generator=g, dtype=torch.float64) for _ in range(n)]
    targets = [torch.randn(6, _WIDTH, generator=g, dtype=torch.float64) for _ in range(n)]
    return tokens, images, targets


@pytest.fixture
def patched_prefix(monkeypatch):
    """Route ``flush``'s phase-3 recompute at the synthetic understanding stage."""
    def build(trainer, _transformer, inputs):
        tokens = inputs[0]
        return _Cache([_Layer(k, v) for k, v in trainer.und(tokens)])

    monkeypatch.setattr(sensenova_ops, "_build_trainable_prefix", build)
    return build


def _run_window(trainer, context, tokens, images, targets, *, step_each=True):
    """Drive one batch: cut once (shared) or per iteration, N generation backwards."""
    n = len(images)
    if context.shared_window:
        cache = context.cut(_Cache([_Layer(k.detach(), v.detach())
                                    for k, v in trainer.und(tokens)]), (tokens,))
        context.begin_window(n)
    losses = []
    for i in range(n):
        if not context.shared_window:
            with torch.no_grad():
                built = trainer.und(tokens)
            cache = context.cut(_Cache([_Layer(k, v) for k, v in built]), (tokens,))
        loss = ((trainer.gen(images[i], cache) - targets[i]) ** 2).mean()
        loss.backward()
        losses.append(float(loss))
        context.after_generation_backward()
        if step_each:
            trainer.apply_updates()
    return losses


# --------------------------------------------------------------------------
# (A) the mechanism: one cut, one grad buffer, leaf reuse
# --------------------------------------------------------------------------


def test_shared_window_reuses_one_set_of_leaves_and_never_grows_a_pending_list(
    patched_prefix,
):
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=True)
    tokens, images, targets = _tokens_and_targets(4)

    cache = context.cut(_Cache([_Layer(k.detach(), v.detach())
                                for k, v in trainer.und(tokens)]), (tokens,))
    context.begin_window(4)
    seen_leaf_ids = set()
    pending_high_water = 0
    for i in range(4):
        seen_leaf_ids.add(id(cache.layers[0].keys))
        loss = ((trainer.gen(images[i], cache) - targets[i]) ** 2).mean()
        loss.backward()
        context.after_generation_backward()
        pending_high_water = max(pending_high_water, context.pending_count)

    # ONE leaf object served every iteration -- the boundary is one buffer, not N.
    assert len(seen_leaf_ids) == 1
    # capture()+flush() happen together, so nothing is ever queued.
    assert pending_high_water == 0
    assert context.pending_count == 0


def test_shared_window_cuts_once_per_window_and_per_iteration_cuts_n_times(
    patched_prefix,
):
    tokens, images, targets = _tokens_and_targets(3)

    for shared, expected_cuts in ((True, 1), (False, 3)):
        trainer = FakeTrainer()
        context = SenseNovaFourPhaseBackward(trainer, shared_window=shared)
        cuts = {"n": 0}
        real_cut = context.cut

        def counting_cut(cache, inputs, _real=real_cut, _c=cuts):
            _c["n"] += 1
            return _real(cache, inputs)

        context.cut = counting_cut
        _run_window(trainer, context, tokens, images, targets)
        assert cuts["n"] == expected_cuts, shared


# --------------------------------------------------------------------------
# (G) und invariance across the window, and its negative control
# --------------------------------------------------------------------------


def test_understanding_weights_are_bit_identical_until_the_window_flushes(
    patched_prefix,
):
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=True)
    tokens, images, targets = _tokens_and_targets(4)
    start = trainer.und_weights()

    cache = context.cut(_Cache([_Layer(k.detach(), v.detach())
                                for k, v in trainer.und(tokens)]), (tokens,))
    context.begin_window(4)
    for i in range(4):
        loss = ((trainer.gen(images[i], cache) - targets[i]) ** 2).mean()
        loss.backward()
        final = context.is_final_iteration()
        context.after_generation_backward()
        if not final:
            # Phase 3 has not run, so no understanding gradient exists and the
            # hook cannot have moved anything.
            assert all(p.grad is None for p in trainer.und.parameters())
        trainer.apply_updates()
        if not final:
            for name, before in start.items():
                after = dict(trainer.und.named_parameters())[name]
                assert torch.equal(after.detach(), before), name

    assert all(count == 1 for count in trainer.und_update_count.values())


def test_negative_control_per_iteration_moves_the_und_half_every_iteration(
    patched_prefix,
):
    """The shipped behaviour, as the control: N updates rather than one."""
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=False)
    tokens, images, targets = _tokens_and_targets(4)
    start = trainer.und_weights()

    _run_window(trainer, context, tokens, images, targets)

    assert all(count == 4 for count in trainer.und_update_count.values())
    for name, before in start.items():
        after = dict(trainer.und.named_parameters())[name]
        assert not torch.equal(after.detach(), before), name


# --------------------------------------------------------------------------
# (G) sum vs mean, and MNT=1 identity
# --------------------------------------------------------------------------


def test_sum_and_mean_differ_by_exactly_the_window_length(patched_prefix):
    tokens, images, targets = _tokens_and_targets(4)
    grads = {}
    for reduction in ("sum", "mean"):
        trainer = FakeTrainer()
        context = SenseNovaFourPhaseBackward(
            trainer, shared_window=True, reduction=reduction
        )
        cache = context.cut(_Cache([_Layer(k.detach(), v.detach())
                                    for k, v in trainer.und(tokens)]), (tokens,))
        context.begin_window(4)
        for i in range(4):
            loss = ((trainer.gen(images[i], cache) - targets[i]) ** 2).mean()
            loss.backward()
            context.after_generation_backward()
        grads[reduction] = {n: p.grad.clone() for n, p in trainer.und.named_parameters()}

    for name, summed in grads["sum"].items():
        assert torch.equal(grads["mean"][name] * 4.0, summed), name
        assert summed.abs().max() > 0


def test_mnt_one_is_arithmetically_identical_to_the_per_iteration_route(
    patched_prefix,
):
    tokens, images, targets = _tokens_and_targets(1)
    results = {}
    for shared in (True, False):
        trainer = FakeTrainer()
        context = SenseNovaFourPhaseBackward(trainer, shared_window=shared)
        _run_window(trainer, context, tokens, images, targets)
        results[shared] = trainer.und_weights()

    for name, weights in results[True].items():
        assert torch.equal(weights, results[False][name]), name


def test_shared_window_equals_a_single_backward_of_the_summed_window_loss(
    patched_prefix,
):
    """Exactness: deferring phase 3 backpropagates the sum at the START weights."""
    tokens, images, targets = _tokens_and_targets(3)

    shared_trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(shared_trainer, shared_window=True)
    cache = context.cut(_Cache([_Layer(k.detach(), v.detach())
                                for k, v in shared_trainer.und(tokens)]), (tokens,))
    context.begin_window(3)
    for i in range(3):
        loss = ((shared_trainer.gen(images[i], cache) - targets[i]) ** 2).mean()
        loss.backward()
        context.after_generation_backward()
        # The generation half moves between iterations; the understanding half
        # must not, which is what makes the deferred backward exact.
        for p in shared_trainer.gen.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-0.05)
                p.grad = None
    shared_grads = {n: p.grad.clone() for n, p in shared_trainer.und.named_parameters()}

    # Reference: the same three generation graphs, same generation-half updates,
    # against an understanding half whose weights never moved, summed.
    ref = FakeTrainer()
    ref.und.load_state_dict(shared_trainer.und.state_dict())
    for name, p in ref.und.named_parameters():
        p.data.copy_(dict(FakeTrainer().und.named_parameters())[name].data)
    ref_total = None
    for i in range(3):
        built = ref.und(tokens)
        ref_cache = _Cache([_Layer(k, v) for k, v in built])
        loss = ((ref.gen(images[i], ref_cache) - targets[i]) ** 2).mean()
        loss.backward()
        for p in ref.gen.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-0.05)
                p.grad = None
    ref_total = {n: p.grad.clone() for n, p in ref.und.named_parameters()}

    for name, expected in ref_total.items():
        assert torch.allclose(shared_grads[name], expected, rtol=0, atol=1e-12), name


# --------------------------------------------------------------------------
# (D) the replacement cut invariant and the discard decision
# --------------------------------------------------------------------------


def test_a_window_read_by_the_wrong_number_of_backwards_is_refused(patched_prefix):
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=True)
    tokens, images, targets = _tokens_and_targets(3)
    cache = context.cut(_Cache([_Layer(k.detach(), v.detach())
                                for k, v in trainer.und(tokens)]), (tokens,))
    context.begin_window(3)
    loss = ((trainer.gen(images[0], cache) - targets[0]) ** 2).mean()
    loss.backward()
    context.after_generation_backward()

    with pytest.raises(RuntimeError, match="declared 3"):
        context.capture()


def test_capturing_a_window_no_backward_read_is_refused():
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=True)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    context.begin_window(2)
    with pytest.raises(RuntimeError, match="without a single generation backward"):
        context.capture()


def test_a_second_cut_before_the_window_closes_is_still_refused():
    """The one-outstanding-boundary clause survives verbatim."""
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=True)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    with pytest.raises(RuntimeError, match="never captured"):
        context.cut(_Cache([_Layer(keys, keys)]), ("b",))


def test_a_generation_backward_before_begin_window_is_refused():
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=True)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    with pytest.raises(RuntimeError, match="before begin_window"):
        context.after_generation_backward()


def test_discard_reports_how_many_backwards_lose_their_understanding_gradient(
    patched_prefix,
):
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=True)
    tokens, images, targets = _tokens_and_targets(4)
    cache = context.cut(_Cache([_Layer(k.detach(), v.detach())
                                for k, v in trainer.und(tokens)]), (tokens,))
    context.begin_window(4)
    for i in range(2):
        loss = ((trainer.gen(images[i], cache) - targets[i]) ** 2).mean()
        loss.backward()
        context.after_generation_backward()
        trainer.apply_updates()

    assert context.discard() == 2
    assert context.dropped_backwards == 2
    # The remaining iterations of the abandoned batch keep counting rather than
    # silently doing nothing.
    context.after_generation_backward()
    assert context.dropped_backwards == 3
    assert all(count == 0 for count in trainer.und_update_count.values())


def test_per_iteration_discard_still_loses_only_its_own_iteration():
    """The shipped asymmetry, as the control for the widened one above."""
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=False)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    assert context.discard() == 0
    assert context.dropped_backwards == 0


def test_a_new_window_over_an_unfinished_one_is_refused():
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=True)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    context.begin_window(3)
    context._window_backwards = 1
    with pytest.raises(RuntimeError, match="uncaptured generation"):
        context.begin_window(3)


# --------------------------------------------------------------------------
# The step seam, against the REAL evictor
#
# The seam is four lines after `flush()` and asserts a HALF RESIDENT. Nothing
# above it catches, so getting it wrong kills the run on the first non-final
# iteration -- which is what shipped until this test existed. `flush()` being a
# correct no-op there is not the same as the seam being a no-op.
# --------------------------------------------------------------------------


class _Half(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))


class _EvictLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _Half()
        self.proj_mot_gen = _Half()


def _real_evictor():
    from core.training.sensenova_phase_eviction import SenseNovaTrainingPhaseEvictor

    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.model = nn.Module()
    root.language_model.model.layers = nn.ModuleList(
        [_EvictLayer() for _ in range(42)]
    )
    return SenseNovaTrainingPhaseEvictor(root, "cpu", four_phase=True)


class _SeamTrainer:
    """Only what ``_assert_sensenova_step_seam_residency`` reads."""

    def __init__(self, evictor):
        self.sensenova_phase_evictor = evictor

    def seam(self, four_phase):
        from core.training.base_trainer import BaseTrainer

        return BaseTrainer._assert_sensenova_step_seam_residency(self, four_phase)


def _recomputed_cache():
    """Phase 3's recomputed prefix: a real graph, so autograd.backward can run."""
    source = torch.zeros(2, 3, requires_grad=True)
    return _Cache([_Layer(source * 1.0, source * 1.0)])


def _deposit(leaf_cache):
    """What the generation backward leaves on the boundary leaves."""
    for layer in leaf_cache.layers:
        for leaf in (layer.keys, layer.values):
            leaf.grad = (
                torch.ones_like(leaf) if leaf.grad is None else leaf.grad + 1
            )


def _drive_window(context, evictor, n, *, abort_at=None):
    """One batch through the real evictor and the real seam.

    Mirrors the production order: enter_prefix + cut at batch prep, then per
    iteration enter_denoise (a no-op after the first), backward,
    after_generation_backward, flush at the seam, seam residency assertion.
    """
    evictor.enter_prefix()
    source = torch.randn(2, 3, requires_grad=True)
    leaf_cache = context.cut(_Cache([_Layer(source, source)]), ("tokens",))
    context.begin_window(n)
    states = []
    for i in range(n):
        evictor.enter_denoise()
        evictor.assert_generation_resident()
        if abort_at is not None and i == abort_at:
            context.discard()
            states.append("aborted")
            break
        # Stand in for the generation backward's deposit on the SHARED leaves --
        # the ones cut() returned, which every iteration accumulates into.
        _deposit(leaf_cache)
        context.after_generation_backward()
        context.flush()
        trainer = _SeamTrainer(evictor)
        trainer.seam(context)
        states.append(evictor.state)
    return states


def test_the_step_seam_survives_every_iteration_of_a_real_shared_window(
    patched_prefix, monkeypatch,
):
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=True)
    evictor = _real_evictor()
    trainer.sensenova_phase_evictor = evictor
    context.trainer = trainer
    monkeypatch.setattr(
        sensenova_ops, "_build_trainable_prefix",
        lambda _t, _tr, _i: _recomputed_cache(),
    )

    states = _drive_window(context, evictor, 4)

    # Non-final iterations leave the GENERATION half resident; only the last one
    # brings the understanding half back.
    assert states == ["denoise", "denoise", "denoise", "und_backward"]


def test_the_step_seam_asserts_the_understanding_half_on_the_per_iteration_route(
    patched_prefix, monkeypatch,
):
    """The shipped route, as the control: phase 3 runs every iteration."""
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=False)
    evictor = _real_evictor()
    trainer.sensenova_phase_evictor = evictor
    context.trainer = trainer
    monkeypatch.setattr(
        sensenova_ops, "_build_trainable_prefix",
        lambda _t, _tr, _i: _recomputed_cache(),
    )

    states = []
    for _ in range(3):
        evictor.enter_prefix()
        source = torch.randn(2, 3, requires_grad=True)
        leaf_cache = context.cut(_Cache([_Layer(source, source)]), ("tokens",))
        evictor.enter_denoise()
        _deposit(leaf_cache)
        context.after_generation_backward()
        context.flush()
        _SeamTrainer(evictor).seam(context)
        states.append(evictor.state)

    assert states == ["und_backward"] * 3


def test_phase_three_ran_is_not_is_final_iteration_at_the_seam():
    """The exact confusion that shipped: at iteration N-2 the two disagree."""
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=True)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    context.begin_window(3)
    keys.grad = torch.ones(2, 3)
    context.after_generation_backward()   # iteration 0 of 3
    context.after_generation_backward()   # iteration 1 of 3 -- N-2

    assert context.is_final_iteration() is True    # the NEXT backward closes it
    assert context.phase_three_ran is False        # but phase 3 has NOT run


def test_the_seam_is_inert_without_a_split_and_without_an_evictor():
    evictor = _real_evictor()
    evictor.enter_prefix()
    evictor.enter_denoise()
    _SeamTrainer(evictor).seam(None)

    class NoEvictor:
        sensenova_phase_evictor = None

    from core.training.base_trainer import BaseTrainer

    BaseTrainer._assert_sensenova_step_seam_residency(NoEvictor(), None)


def test_the_seam_still_catches_a_half_that_is_resident_when_it_must_not_be(
    patched_prefix, monkeypatch,
):
    """Window-awareness must not turn the assertion into a rubber stamp."""
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=True)
    evictor = _real_evictor()
    trainer.sensenova_phase_evictor = evictor
    context.trainer = trainer
    monkeypatch.setattr(
        sensenova_ops, "_build_trainable_prefix",
        lambda _t, _tr, _i: _recomputed_cache(),
    )
    evictor.enter_prefix()
    source = torch.randn(2, 3, requires_grad=True)
    leaf_cache = context.cut(_Cache([_Layer(source, source)]), ("tokens",))
    context.begin_window(1)
    evictor.enter_denoise()
    _deposit(leaf_cache)
    context.after_generation_backward()
    assert context.phase_three_ran is True
    # Phase 3 ran, so the seam wants the understanding half -- put the evictor
    # back in denoise behind its back and the seam must object.
    evictor.state = "denoise"
    with pytest.raises(RuntimeError, match="requires prefix or und_backward"):
        _SeamTrainer(evictor).seam(context)


def test_the_seam_catches_a_generation_half_that_is_wrongly_not_resident(
    patched_prefix, monkeypatch,
):
    """The non-final branch is an assertion too, not a way of skipping one."""
    trainer = FakeTrainer()
    context = SenseNovaFourPhaseBackward(trainer, shared_window=True)
    evictor = _real_evictor()
    trainer.sensenova_phase_evictor = evictor
    context.trainer = trainer
    monkeypatch.setattr(
        sensenova_ops, "_build_trainable_prefix",
        lambda _t, _tr, _i: _recomputed_cache(),
    )
    evictor.enter_prefix()
    source = torch.randn(2, 3, requires_grad=True)
    leaf_cache = context.cut(_Cache([_Layer(source, source)]), ("tokens",))
    context.begin_window(3)
    evictor.enter_denoise()
    _deposit(leaf_cache)
    context.after_generation_backward()
    assert context.phase_three_ran is False

    # Non-final iteration: the seam wants the GENERATION half. Put the evictor
    # into und_backward behind its back and it must object.
    evictor.state = "und_backward"
    with pytest.raises(RuntimeError, match="generation work requires denoise"):
        _SeamTrainer(evictor).seam(context)


def test_an_aborted_window_is_visible_to_the_loop_that_must_end_the_batch():
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=True)
    evictor = _real_evictor()
    states = _drive_window(context, evictor, 4, abort_at=2)

    assert states[-1] == "aborted"
    assert context.window_aborted is True
    assert context.dropped_backwards == 2
    # The next batch's cut clears it, so the break is per batch.
    context.cut(_Cache([_Layer(torch.randn(2, 3, requires_grad=True),
                               torch.randn(2, 3, requires_grad=True))]), ("b",))
    assert context.window_aborted is False


def test_the_per_iteration_route_never_reports_an_aborted_window():
    """The break must not fire for the shipped route, whose skips are per step."""
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=False)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    context.discard()
    assert context.window_aborted is False


def test_continuing_into_an_aborted_window_would_fail_the_census():
    """Why the break exists, rather than only that it exists.

    ``discard`` clears the window size, so ``is_final_iteration`` answers True
    and the census would demand the deferred group on an iteration that cannot
    produce it -- killing the run on the path whose whole purpose is keeping it
    alive.
    """
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=True)
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    context.begin_window(4)
    assert context.is_final_iteration() is False
    context.discard()
    assert context.is_final_iteration() is True

    census, params = _census_with()
    census.begin_step(True, expect_deferred=context.is_final_iteration())
    census.record(params["gen.a"])
    census.record(params["gen.b"])
    with pytest.raises(RuntimeError, match="received no optimizer"):
        census.assert_complete("iteration after a discard")


def test_the_mnt_loop_breaks_on_an_aborted_window():
    """The guard is at the top of the MNT loop, before any forward."""
    import inspect

    from core.training.base_trainer import BaseTrainer

    source = inspect.getsource(BaseTrainer.train)
    head = source.index("for mnt_idx in range(multi_noise_timesteps):")
    tail = source.index("timesteps = timestep_sampler.sample(", head)
    guard = source[head:tail]
    assert "window_aborted" in guard and "break" in guard


# --------------------------------------------------------------------------
# (C) the census, window-aware
# --------------------------------------------------------------------------


def _census_with(deferred_names=("und.a", "und.b")):
    params = {n: nn.Parameter(torch.zeros(1))
              for n in ("gen.a", "gen.b", "und.a", "und.b")}
    census = UpdateCensus()
    census.expect(params.values(), {id(p): n for n, p in params.items()})
    census.set_deferred([params[n] for n in deferred_names])
    return census, params


def test_census_excuses_the_deferred_group_only_on_non_final_backwards():
    census, params = _census_with()

    census.begin_step(True, expect_deferred=False)
    for n in ("gen.a", "gen.b"):
        census.record(params[n])
    census.assert_complete("mid-window")  # correct run, does not raise

    census.begin_step(True, expect_deferred=True)
    for n in ("gen.a", "gen.b", "und.a", "und.b"):
        census.record(params[n])
    census.assert_complete("window end")


def test_census_still_catches_an_understanding_half_that_never_updates():
    """THE guarantee: deferral moves the check, it does not remove it."""
    census, params = _census_with()

    # Three non-final backwards pass, exactly as a correct deferred run does.
    for _ in range(3):
        census.begin_step(True, expect_deferred=False)
        census.record(params["gen.a"])
        census.record(params["gen.b"])
        census.assert_complete("mid-window")

    # The backward that closes the window requires the deferred group in full.
    census.begin_step(True, expect_deferred=True)
    census.record(params["gen.a"])
    census.record(params["gen.b"])
    with pytest.raises(RuntimeError, match="received no optimizer"):
        census.assert_complete("window end")


def test_census_still_catches_a_generation_parameter_mid_window():
    """Deferral must not blind the census to the half that is NOT deferred."""
    census, params = _census_with()
    census.begin_step(True, expect_deferred=False)
    census.record(params["gen.a"])
    with pytest.raises(RuntimeError, match="gen.b"):
        census.assert_complete("mid-window")


def test_census_refuses_a_degenerate_deferred_group():
    params = {n: nn.Parameter(torch.zeros(1)) for n in ("a", "b")}
    census = UpdateCensus()
    census.expect(params.values(), {id(p): n for n, p in params.items()})
    with pytest.raises(ValueError, match="does not intersect"):
        census.set_deferred([nn.Parameter(torch.zeros(1))])
    with pytest.raises(ValueError, match="whole"):
        census.set_deferred(params.values())


def test_expect_clears_a_stale_deferred_group():
    census, params = _census_with()
    assert census.deferred_count == 2
    census.expect(params.values(), {id(p): n for n, p in params.items()})
    assert census.deferred_count == 0


def test_begin_step_defaults_to_requiring_everything():
    """Every other trainer calls begin_step(True); its behaviour is unchanged."""
    census, params = _census_with()
    census.begin_step(True)
    census.record(params["gen.a"])
    census.record(params["gen.b"])
    with pytest.raises(RuntimeError, match="und\\."):
        census.assert_complete()


# --------------------------------------------------------------------------
# Untouched paths
# --------------------------------------------------------------------------


def test_deferred_parameters_are_empty_without_the_shared_route():
    class Stub:
        sensenova_four_phase = None

    assert understanding_deferred_parameters(Stub()) == ()

    stub = Stub()
    stub.sensenova_four_phase = SenseNovaFourPhaseBackward(
        FakeTrainer(), shared_window=False
    )
    assert understanding_deferred_parameters(stub) == ()


def test_census_deferred_parameters_is_empty_for_every_other_architecture():
    from core.training.base_trainer import census_deferred_parameters

    class Stub:
        is_sensenova = False

    assert census_deferred_parameters(Stub()) == ()


def test_deferred_parameters_come_from_the_modules_the_evictor_stages_to_cpu():
    class Evictor:
        def __init__(self, modules):
            self._modules = modules

        @property
        def understanding_modules(self):
            return tuple(self._modules)

    linear = nn.Linear(2, 2)
    frozen = nn.Linear(2, 2)
    frozen.requires_grad_(False)

    class Stub:
        pass

    stub = Stub()
    stub.sensenova_four_phase = SenseNovaFourPhaseBackward(
        FakeTrainer(), shared_window=True
    )
    stub.sensenova_phase_evictor = Evictor([linear, frozen])
    deferred = understanding_deferred_parameters(stub)
    assert {id(p) for p in deferred} == {id(linear.weight), id(linear.bias)}


def test_frozen_understanding_branch_conditioning_is_unchanged():
    """The frozen path already reuses one prefix; the shared branch adds nothing."""
    from types import MethodType

    from core.training.base_trainer import BaseTrainer

    class Frozen:
        train_text_encoder = False
        sensenova_four_phase = None
        # What _encode_sensenova_batch_prefix stashes for a single item that
        # drew no null label, and the per-batch alternate-prefix memo slot.
        _sensenova_prefix_cfg_null = False
        _sensenova_alt_cfg_null_prefix = None

        def encode_caption(self, *_a, **_k):
            raise AssertionError("the frozen branch must not re-encode")

    frozen = Frozen()
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, frozen)
    sentinel = object()
    assert conditioning(sentinel, captions=["c"], mnt_index=3)[3] is sentinel


def test_a_trainable_branch_without_sharing_still_re_encodes_every_iteration():
    from types import MethodType

    from core.training.base_trainer import BaseTrainer

    calls = []

    class PerIteration:
        train_text_encoder = True
        sensenova_four_phase = SenseNovaFourPhaseBackward(
            FakeTrainer(), shared_window=False
        )

        def encode_caption(self, caption, **_k):
            calls.append(caption)
            return "fresh", None

    owner = PerIteration()
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, owner)
    assert conditioning("first", captions=["c"], mnt_index=1)[3] == "fresh"
    assert calls == ["c"]


def test_the_shared_branch_reuses_the_prefix_instead_of_re_encoding():
    from types import MethodType

    from core.training.base_trainer import BaseTrainer

    class Shared:
        train_text_encoder = True
        sensenova_four_phase = SenseNovaFourPhaseBackward(
            FakeTrainer(), shared_window=True
        )

        def encode_caption(self, *_a, **_k):
            raise AssertionError("the shared window must not re-encode the prefix")

    owner = Shared()
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, owner)
    sentinel = object()
    assert conditioning(sentinel, captions=["c"], mnt_index=2)[3] is sentinel


def test_begin_window_and_is_final_iteration_are_inert_off_the_shared_route():
    context = SenseNovaFourPhaseBackward(FakeTrainer(), shared_window=False)
    context.begin_window(8)
    assert context.is_final_iteration() is True


# --------------------------------------------------------------------------
# (B) the settings themselves
# --------------------------------------------------------------------------


def test_defaults_are_off_and_sum():
    assert TRAINING_DEFAULTS["sensenova_four_phase_shared_prefix"] is False
    assert TRAINING_DEFAULTS["sensenova_four_phase_grad_reduction"] == "sum"


def test_an_unknown_reduction_is_refused_at_construction():
    with pytest.raises(ValueError, match="sum, mean"):
        SenseNovaFourPhaseBackward(FakeTrainer(), reduction="median")


def test_install_reads_the_settings_off_the_trainer():
    trainer = FakeTrainer()
    trainer.sensenova_four_phase_shared_prefix = True
    trainer.sensenova_four_phase_grad_reduction = "mean"
    context = install_four_phase_backward(trainer)
    assert context.shared_window is True and context.reduction == "mean"
    assert trainer.sensenova_four_phase is context

    plain = FakeTrainer()
    assert install_four_phase_backward(plain).shared_window is False


def test_the_shared_flag_is_refused_without_the_split_it_shares():
    class Stub:
        config = {"sensenova_four_phase_shared_prefix": True}

    with pytest.raises(ValueError, match="requires sensenova_four_phase_eviction"):
        sensenova_ops.assert_shared_prefix_contract(Stub())

    class Armed:
        config = {
            "sensenova_four_phase_shared_prefix": True,
            "sensenova_four_phase_eviction": True,
        }

    sensenova_ops.assert_shared_prefix_contract(Armed())


@pytest.mark.parametrize("trainer_cls", ["full_finetune", "lora"])
def test_the_shared_flag_without_the_split_raises_through_the_trainers_own_setup(
    trainer_cls,
):
    """Not through a direct call: the contract has to be REACHED.

    Its call site used to sit inside the eviction gate, so the one configuration
    it exists for -- shared prefix set without the split -- fell straight through
    and the flag silently did nothing.
    """
    if trainer_cls == "full_finetune":
        from core.training.full_parameter_trainer import FullParameterTrainer as Cls
    else:
        from core.training.lora_trainer import LoRATrainer as Cls

    trainer = Cls.__new__(Cls)
    trainer.is_sensenova = True
    trainer.log_prefix = "[test]"
    trainer.config = {}
    trainer.train_unet = True
    # The gate this used to hide behind: eviction OFF, so the old call site was
    # unreachable.
    trainer.sensenova_mot_phase_eviction = False
    trainer.sensenova_four_phase_eviction = False
    trainer.sensenova_four_phase_shared_prefix = True

    with pytest.raises(ValueError, match="requires sensenova_four_phase_eviction"):
        trainer._setup_sensenova_phase_eviction()


@pytest.mark.parametrize("trainer_cls", ["full_finetune", "lora"])
def test_a_non_sensenova_trainer_and_an_unarmed_one_pass_the_hoisted_contract(
    trainer_cls,
):
    if trainer_cls == "full_finetune":
        from core.training.full_parameter_trainer import FullParameterTrainer as Cls
    else:
        from core.training.lora_trainer import LoRATrainer as Cls

    other_arch = Cls.__new__(Cls)
    other_arch.is_sensenova = False
    other_arch._setup_sensenova_phase_eviction()

    unarmed = Cls.__new__(Cls)
    unarmed.is_sensenova = True
    unarmed.log_prefix = "[test]"
    unarmed.config = {}
    unarmed.train_unet = True
    unarmed.sensenova_mot_phase_eviction = False
    unarmed.sensenova_four_phase_eviction = False
    unarmed.sensenova_four_phase_shared_prefix = False
    unarmed._setup_sensenova_phase_eviction()


def test_an_understanding_only_run_is_refused_before_the_census_can_complain():
    """train_runner refuses this in config terms; this is the trainer-side backstop.

    Without it a hand-built trainer meets the same configuration as
    ``set_deferred``'s "the deferred group is the whole expectation set".
    """
    class UndOnly:
        config = {
            "sensenova_four_phase_shared_prefix": True,
            "sensenova_four_phase_eviction": True,
            "train_unet": False,
        }

    with pytest.raises(ValueError, match="requires train_unet"):
        sensenova_ops.assert_shared_prefix_contract(UndOnly())


def test_the_runner_already_refuses_an_understanding_only_shared_run():
    from unittest.mock import patch

    from core.model_loader import ModelLoader
    from core.training import train_runner

    config = {
        "batch_size": 1, "blocks_to_swap": 0,
        "train_unet": False, "train_text_encoder": True,
        "sensenova_mot_phase_eviction": True,
        "sensenova_four_phase_eviction": True,
        "sensenova_four_phase_shared_prefix": True,
        "optimizer": "adafactor", "gradient_accumulation_steps": 1,
        "num_optimizer_groups": 0, "use_ema": False,
        "sensenova_full_finetune_save_format": "mixed",
    }
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="requires both train_unet"):
            train_runner._apply_sensenova_training_contract(
                "model", "full_finetune", config, {}
            )


def test_the_mnt_cost_notice_stops_announcing_a_cost_the_shared_route_removes():
    from core.training import training_events

    emitted = []
    original = training_events.emit_training_warning

    def capture(message, code=None, prefix="", console=True):
        emitted.append({"code": code, "message": message})
        return {}

    sensenova_ops.emit_training_warning = capture
    try:
        class Shared:
            config = {
                "multi_noise_timesteps": 4,
                "sensenova_four_phase_shared_prefix": True,
                "sensenova_four_phase_grad_reduction": "sum",
            }

        assert sensenova_ops.warn_four_phase_mnt_cost(Shared()) is True
        assert emitted[-1]["code"] == "sensenova_four_phase_shared_prefix"
        assert "ONE update per window" in emitted[-1]["message"]
        assert "paid 4 times per step" not in emitted[-1]["message"]

        class PerIteration:
            config = {"multi_noise_timesteps": 4}

        assert sensenova_ops.warn_four_phase_mnt_cost(PerIteration()) is True
        assert emitted[-1]["code"] == "sensenova_four_phase_mnt_cost"
    finally:
        sensenova_ops.emit_training_warning = original


def test_the_run_config_carries_both_keys():
    from core.training.training_config import _build_train_section

    section = _build_train_section(
        {
            "sensenova_four_phase_shared_prefix": True,
            "sensenova_four_phase_grad_reduction": "mean",
        },
        total_steps=None,
        epochs=1,
        train_unet=True,
        train_text_encoder=True,
        include_block_swap=True,
    )
    assert section["sensenova_four_phase_shared_prefix"] is True
    assert section["sensenova_four_phase_grad_reduction"] == "mean"

    default = _build_train_section(
        {}, total_steps=None, epochs=1, train_unet=True,
        train_text_encoder=True, include_block_swap=True,
    )
    assert default["sensenova_four_phase_shared_prefix"] is False
    assert default["sensenova_four_phase_grad_reduction"] == "sum"


def test_the_runner_refuses_the_shared_flag_without_the_split_and_a_bad_reduction():
    from unittest.mock import patch

    from core.model_loader import ModelLoader
    from core.training import train_runner

    def config(**overrides):
        base = {
            "batch_size": 1,
            "blocks_to_swap": 0,
            "train_unet": True,
            "train_text_encoder": True,
            "sensenova_mot_phase_eviction": True,
            "sensenova_four_phase_eviction": True,
            "optimizer": "adafactor",
            "gradient_accumulation_steps": 1,
            "num_optimizer_groups": 0,
            "use_ema": False,
            "sensenova_full_finetune_save_format": "mixed",
        }
        base.update(overrides)
        return base

    path = "model"
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        armed = config(sensenova_four_phase_shared_prefix=True)
        assert train_runner._apply_sensenova_training_contract(
            path, "full_finetune", armed, {}
        )
        # Normalized in place, so the trainer never has to re-parse it.
        assert armed["sensenova_four_phase_grad_reduction"] == "sum"

        with pytest.raises(ValueError, match="requires sensenova_four_phase_eviction"):
            train_runner._apply_sensenova_training_contract(
                path,
                "full_finetune",
                config(sensenova_four_phase_eviction=False,
                       sensenova_four_phase_shared_prefix=True,
                       train_text_encoder=False),
                {},
            )
        with pytest.raises(ValueError, match="must be 'sum' or 'mean'"):
            train_runner._apply_sensenova_training_contract(
                path,
                "full_finetune",
                config(sensenova_four_phase_grad_reduction="median"),
                {},
            )


def test_the_capability_table_hides_both_keys_where_the_mechanism_is_absent():
    from api.arch_capabilities import (
        TRAINING_FEATURE_PARAMS,
        TRAINING_FEATURE_UNSUPPORTED,
    )

    keys = TRAINING_FEATURE_PARAMS["sensenova_mot_eviction"]
    assert "sensenova_four_phase_shared_prefix" in keys
    assert "sensenova_four_phase_grad_reduction" in keys
    assert "sensenova_mot_eviction" in TRAINING_FEATURE_UNSUPPORTED["sdxl"]
    assert "sensenova_mot_eviction" not in TRAINING_FEATURE_UNSUPPORTED.get(
        "sensenova", {}
    )


def test_the_rest_api_and_spec_carry_both_settings():
    import yaml

    from api.routes import TrainingRunCreateRequest

    fields = TrainingRunCreateRequest.model_fields
    assert fields["sensenova_four_phase_shared_prefix"].default is False
    assert fields["sensenova_four_phase_grad_reduction"].default == "sum"

    spec = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "openapi.yaml").read_text(
            encoding="utf-8"
        )
    )
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    assert props["sensenova_four_phase_shared_prefix"]["default"] is False
    assert props["sensenova_four_phase_grad_reduction"]["enum"] == ["sum", "mean"]
    assert props["sensenova_four_phase_grad_reduction"]["default"] == "sum"


def test_the_frontend_exposes_both_settings_and_clears_them_with_the_split():
    root = Path(__file__).resolve().parents[2] / "frontend" / "src"
    tsx = (root / "components" / "training" / "TrainingConfig.tsx").read_text(
        encoding="utf-8"
    )
    api_ts = (root / "utils" / "api.ts").read_text(encoding="utf-8")

    assert "sensenova_four_phase_shared_prefix?: boolean;" in api_ts
    assert 'sensenova_four_phase_grad_reduction?: "sum" | "mean";' in api_ts
    assert "sensenova_four_phase_shared_prefix: false," in tsx
    assert 'sensenova_four_phase_grad_reduction: "sum",' in tsx
    assert ('sensenova_four_phase_shared_prefix: params.sensenova_four_phase_shared_prefix,'
            in tsx)
    assert 'updateParam("sensenova_four_phase_shared_prefix", false);' in tsx
    assert "sensenova-four-phase-shared-prefix" in tsx


def test_the_dropped_gradient_counter_has_a_chart_definition():
    from core.training.metric_registry import EXTRA_METRIC_DEFS

    assert "sn_und_grad_dropped" in EXTRA_METRIC_DEFS
