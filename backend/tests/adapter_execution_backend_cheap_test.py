"""Execution-backend selection for adapter algebra: registry, probe, latch,
warm-up and dispatch. CPU, ~5 s.

Every gate here runs against a FAKE backend registered by the test, never
against a real one: this build registers only ``reference``, and a gate that
could only be exercised by a backend nobody has written would be a gate that
never runs. The fakes are deliberately unpleasant -- one is numerically wrong,
one raises during its probe, one raises only after it has been admitted -- so
the refusal, the probe and the latch are each proved by the behaviour they
exist for.

Covers ``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` phase 4's mechanism bullets.
It does NOT cover, and cannot: any performance property. Nothing here measures
anything, and nothing in this repo claims a fused adapter path is faster.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_execution_backend_cheap_test.py -v
"""

import ast
import os
import pathlib
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from core.adapters import (DoRALinearLayer, LoHaLinearLayer,  # noqa: E402
                           LoKrLinearLayer, LoRALinearLayer,
                           MiniMaxH3LoRALinearLayer)
from core.adapters.execution import (BACKENDS, REFERENCE,  # noqa: E402
                                     AdapterBackend, clear_probe_cache,
                                     is_latched, latched_reason,
                                     probe_region, reference_backend,
                                     region_for, reset_execution_state,
                                     select_adapter_backend,
                                     selected_adapter_backend,
                                     warm_up_adapter_backend)
from core.adapters.execution.dispatch import (BACKEND_UNAVAILABLE_CODE,  # noqa: E402
                                              LATCH_CODE, active_backend)
from core.adapters.execution.selection import (BACKEND_ENV_VAR,  # noqa: E402
                                               apply_configured_backend)
from core.adapters.capability import ADAPTER_PAIRS  # noqa: E402
from core.adapters.session import AdapterRefusal  # noqa: E402

D_IN, D_OUT = 12, 10
RANK, ALPHA = 4, 8.0


@pytest.fixture(autouse=True)
def _clean_execution_state():
    """No test may leak a selection, a latch or a verdict into the next."""
    reset_execution_state()
    clear_probe_cache()
    registered = set(BACKENDS)
    yield
    for name in set(BACKENDS) - registered:
        del BACKENDS[name]
    reset_execution_state()
    clear_probe_cache()


# -- fixtures --------------------------------------------------------------

def _base(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(D_IN, D_OUT)


def _exercised(layer: nn.Module) -> nn.Module:
    """Randomise the factor that starts as a no-op.

    Without this the delta is exactly zero and every comparison below passes
    with the branch effectively absent -- the trap ``adapter_oracle_gate`` and
    the probe both guard.
    """
    with torch.no_grad():
        for tensor in layer.branch_tensors().values():
            if isinstance(tensor, nn.Parameter) and not bool(tensor.any()):
                tensor.normal_(0.0, 2.0)
    return layer


def _branches():
    """One live branch per shipped algebra, each with a non-zero delta."""
    lora = _exercised(LoRALinearLayer(_base(1), RANK, ALPHA, "lora"))
    loha = _exercised(LoHaLinearLayer(_base(2), RANK, ALPHA, "loha"))
    lokr = _exercised(LoKrLinearLayer(_base(3), RANK, ALPHA, "lokr"))
    inner = LoRALinearLayer(_base(4), RANK, ALPHA, "inner")
    dora = _exercised(DoRALinearLayer(inner.original_module, inner))
    h3 = _exercised(MiniMaxH3LoRALinearLayer(_base(5), RANK, ALPHA, "h3"))
    return {"lora": lora, "loha": loha, "lokr": lokr, "dora": dora, "h3": h3}


def _register(name, fn, **overrides) -> AdapterBackend:
    fields = dict(
        name=name,
        fn=fn,
        pairs=frozenset(ADAPTER_PAIRS),
        dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
        device_kinds=frozenset({"cpu", "cuda"}),
        trainable=True,
        requires_matching_dtypes=False,
        needs_probe=True,
        availability=lambda: None,
    )
    fields.update(overrides)
    backend = AdapterBackend(**fields)
    BACKENDS[name] = backend
    return backend


def _x(seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((3, D_IN), generator=generator)


# -- the reference backend -------------------------------------------------

class TestReferenceBackend:
    def test_the_reference_backend_is_registered_and_needs_no_probe(self):
        backend = reference_backend()
        assert backend.name == REFERENCE
        assert BACKENDS[REFERENCE] is backend
        assert backend.needs_probe is False
        assert backend.availability() is None
        assert backend.pairs == frozenset(ADAPTER_PAIRS), (
            "the reference path implements every algebra; a narrower "
            "declaration would make some branch unservable by anything")

    def test_nothing_is_selected_by_default(self):
        assert selected_adapter_backend() == REFERENCE
        assert active_backend() is None, (
            "None is the reference path; a descriptor here would put an extra "
            "call on every branch forward")

    def test_the_unselected_path_is_the_reference_backends_own_fn(self):
        x = _x()
        for name, branch in _branches().items():
            assert torch.equal(branch.forward_delta(x),
                               reference_backend().fn(branch, x)), name

    def test_selecting_reference_explicitly_changes_nothing(self):
        x = _x()
        before = {name: branch.forward_delta(x)
                  for name, branch in _branches().items()}
        assert select_adapter_backend(REFERENCE) == REFERENCE
        for name, branch in _branches().items():
            assert torch.equal(branch.forward_delta(x), before[name]), name

    def test_every_algebra_is_byte_identical_through_the_seam(self):
        """``forward`` and ``forward_delta`` must agree exactly: the composite
        drives branches through the latter, training installs a bare branch and
        uses the former."""
        x = _x()
        for name, branch in _branches().items():
            expected = branch.original_module(x) + branch.reference_delta(x)
            assert torch.equal(branch(x), expected), f"{name}: forward"
            assert torch.equal(branch.forward_delta(x),
                               branch.reference_delta(x)), f"{name}: delta"


# -- selection -------------------------------------------------------------

class TestSelection:
    def test_an_unknown_backend_is_refused_with_its_code(self):
        with pytest.raises(AdapterRefusal) as excinfo:
            select_adapter_backend("lycoris")
        assert excinfo.value.code == BACKEND_UNAVAILABLE_CODE
        assert "lycoris" in str(excinfo.value)
        assert selected_adapter_backend() == REFERENCE

    def test_an_unknown_backend_can_warn_instead_of_refusing(self):
        warnings = []
        assert select_adapter_backend("lycoris", warn=lambda m, c: warnings.append((m, c)),
                                      strict=False) == REFERENCE
        assert [code for _, code in warnings] == [BACKEND_UNAVAILABLE_CODE]

    def test_a_backend_that_cannot_run_here_is_refused_with_its_reason(self):
        _register("absent", lambda branch, x: None,
                  availability=lambda: "triton is not installed")
        with pytest.raises(AdapterRefusal) as excinfo:
            select_adapter_backend("absent")
        assert excinfo.value.code == BACKEND_UNAVAILABLE_CODE
        assert "triton is not installed" in str(excinfo.value)

    def test_the_environment_is_the_real_selection_path(self, monkeypatch):
        """What the trainer's warm-up hook actually calls."""
        monkeypatch.setenv(BACKEND_ENV_VAR, "lycoris")
        with pytest.raises(AdapterRefusal) as excinfo:
            apply_configured_backend()
        assert excinfo.value.code == BACKEND_UNAVAILABLE_CODE

    def test_an_unset_environment_selects_nothing(self, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        assert apply_configured_backend() == REFERENCE
        assert active_backend() is None


# -- the probe -------------------------------------------------------------

class TestProbe:
    def test_a_correct_backend_passes_and_then_runs(self):
        calls = []

        def fake(branch, x):
            calls.append(x.shape)
            return branch.reference_delta(x)

        _register("good", fake)
        select_adapter_backend("good")
        branch = _branches()["lora"]
        x = _x()
        assert torch.equal(branch.forward_delta(x), branch.reference_delta(x))
        assert len(calls) >= 2, "the probe ran it, then the dispatch did"
        assert not is_latched("good")

    def test_a_numerically_wrong_backend_is_not_usable_and_never_runs_again(self):
        calls = []

        def wrong(branch, x):
            calls.append(1)
            return branch.reference_delta(x) * 1.5

        _register("wrong", wrong)
        select_adapter_backend("wrong")
        branch = _branches()["lora"]
        x = _x()
        assert torch.equal(branch.forward_delta(x), branch.reference_delta(x))
        assert len(calls) == 1, "probed once, then refused; not called again"
        assert torch.equal(branch.forward_delta(x), branch.reference_delta(x))
        assert len(calls) == 1
        assert not is_latched("wrong"), (
            "a numerical disagreement is a region verdict, not a launch "
            "failure -- the backend stays selectable for other regions")

    def test_a_probe_verdict_does_not_generalise_across_regions(self):
        _register("good", lambda branch, x: branch.reference_delta(x))
        backend = BACKENDS["good"]
        branch = _branches()["lora"]
        fp32 = region_for(branch, torch.float32)
        bf16 = region_for(branch, torch.bfloat16)
        assert fp32 != bf16
        assert probe_region(backend, branch, fp32).usable
        from core.adapters.execution import cached_verdict
        assert cached_verdict("good", bf16) is None, (
            "a verdict for one dtype must not admit another")

    def test_a_zero_delta_branch_cannot_certify_anything_by_accident(self):
        """The probe randomises the no-op factor of its own copy; a branch whose
        delta is still zero would otherwise admit any backend at all."""
        _register("good", lambda branch, x: branch.reference_delta(x))
        fresh = LoRALinearLayer(_base(7), RANK, ALPHA, "fresh")
        assert not bool(fresh.lora_up.weight.any())
        result = probe_region(BACKENDS["good"], fresh,
                              region_for(fresh, torch.float32))
        assert result.usable, result.reason
        assert not bool(fresh.lora_up.weight.any()), (
            "the probe must not mutate the live branch")

    def test_a_dishonest_backend_cannot_pass_by_returning_zero(self):
        _register("zero", lambda branch, x: torch.zeros_like(branch.reference_delta(x)))
        fresh = LoRALinearLayer(_base(7), RANK, ALPHA, "fresh")
        result = probe_region(BACKENDS["zero"], fresh,
                              region_for(fresh, torch.float32))
        assert not result.usable
        assert "oracle" in (result.reason or "")

    def test_a_backend_with_a_wrong_backward_is_refused(self):
        class _BadBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, delta):
                return delta.clone()

            @staticmethod
            def backward(ctx, grad):
                return grad * 3.0

        _register("badgrad", lambda branch, x: _BadBackward.apply(branch.reference_delta(x)))
        branch = _branches()["lora"]
        result = probe_region(BACKENDS["badgrad"], branch,
                              region_for(branch, torch.float32))
        assert not result.usable
        assert "gradients disagree" in (result.reason or "")

    def test_the_oracle_budget_estimate_is_not_optimistic(self):
        """MEASURED at 512x512 rank 48, ``with_grad=False``: the oracle peaks at
        97.2 MiB, because ``_low_rank_product`` holds the rank-1 term list and
        its ``torch.stack`` at once. A ``rank + 3`` estimate said 51.0 MiB --
        1.91x optimistic, and an inference-only backend is exactly the arm that
        skips the doubling."""
        from core.adapters.execution.probe import _oracle_bytes
        region = region_for(_branches()["lora"], torch.float32)._replace(
            out_features=512, in_features=512)
        estimate = _oracle_bytes(region, rank=48, with_grad=False)
        assert estimate >= int(97.2 * 1024 ** 2), (
            f"{estimate / 1024 ** 2:.1f} MiB estimated against a 97.2 MiB peak")

    def test_a_region_too_large_to_check_is_not_admitted(self):
        _register("good", lambda branch, x: branch.reference_delta(x))
        huge = region_for(_branches()["lora"], torch.float32)._replace(
            out_features=100_000, in_features=100_000)
        result = probe_region(BACKENDS["good"], _branches()["lora"], huge)
        assert not result.usable
        assert "probe budget" in (result.reason or "")


# -- the latch -------------------------------------------------------------

class TestLatch:
    def test_a_backend_that_raises_during_its_probe_latches_off(self):
        def explode(branch, x):
            raise RuntimeError("kernel launch failed")

        _register("boom", explode)
        select_adapter_backend("boom")
        branch = _branches()["lora"]
        x = _x()
        assert torch.equal(branch.forward_delta(x), branch.reference_delta(x))
        assert is_latched("boom")
        assert "kernel launch failed" in latched_reason("boom")
        assert active_backend() is None
        assert selected_adapter_backend() == REFERENCE

    def test_a_backend_that_raises_after_admission_latches_and_falls_back(self):
        """The gate the design doc states: on a launch failure the backend is off
        for the process and every later call takes the reference path."""
        state = {"calls": 0}

        def flaky(branch, x):
            state["calls"] += 1
            if state["calls"] > 1:  # call 1 is the probe, which succeeds
                raise RuntimeError("CantSplit")
            return branch.reference_delta(x)

        _register("flaky", flaky)
        select_adapter_backend("flaky")
        branch = _branches()["lora"]
        x = _x()
        assert torch.equal(branch.forward_delta(x), branch.reference_delta(x))
        assert is_latched("flaky")
        calls_at_latch = state["calls"]
        for _ in range(3):
            assert torch.equal(branch.forward_delta(x), branch.reference_delta(x))
        assert state["calls"] == calls_at_latch, "a latched backend is never called again"

    def test_the_latch_is_per_process_and_re_selection_is_refused(self):
        _register("boom", lambda branch, x: (_ for _ in ()).throw(RuntimeError("nope")))
        select_adapter_backend("boom")
        branch = _branches()["lora"]
        branch.forward_delta(_x())
        assert is_latched("boom")
        with pytest.raises(AdapterRefusal) as excinfo:
            select_adapter_backend("boom")
        assert excinfo.value.code == BACKEND_UNAVAILABLE_CODE
        assert "latched off" in str(excinfo.value)

    def test_a_latch_after_real_work_says_work_used_it(self):
        state = {"calls": 0}

        def flaky(branch, x):
            state["calls"] += 1
            if state["calls"] > 2:  # 1 = the probe, 2 = one live result
                raise RuntimeError("CantSplit")
            return branch.reference_delta(x)

        warnings = []
        _register("flaky", flaky)
        select_adapter_backend("flaky", warn=lambda m, c: warnings.append((m, c)))
        branch = _branches()["lora"]
        warm_up_adapter_backend([branch], activation_dtypes=(torch.float32,),
                                warn=lambda m, c: warnings.append((m, c)),
                                log=lambda line: None)
        branch.forward_delta(_x())   # the live result
        branch.forward_delta(_x())   # raises, latches
        assert [code for _, code in warnings] == [LATCH_CODE]
        assert "after it had already produced results" in warnings[0][0]

    def test_a_latch_before_any_real_work_says_nothing_used_it(self):
        """The message must not claim work ran under the backend when the FIRST
        live call is the one that failed -- warm-up having run is not evidence
        that anything was computed with it, since the probe runs on a copy."""
        state = {"calls": 0}

        def flaky(branch, x):
            state["calls"] += 1
            if state["calls"] > 1:  # the probe passes, the first live call does not
                raise RuntimeError("CantSplit")
            return branch.reference_delta(x)

        warnings = []
        _register("flaky", flaky)
        select_adapter_backend("flaky", warn=lambda m, c: warnings.append((m, c)))
        branch = _branches()["lora"]
        warm_up_adapter_backend([branch], activation_dtypes=(torch.float32,),
                                warn=lambda m, c: warnings.append((m, c)),
                                log=lambda line: None)
        branch.forward_delta(_x())
        assert [code for _, code in warnings] == [LATCH_CODE]
        assert "before it produced any result" in warnings[0][0]

    def test_a_latch_during_warm_up_says_no_step_ran_on_it(self):
        warnings = []
        _register("boom", lambda branch, x: (_ for _ in ()).throw(RuntimeError("nope")))
        select_adapter_backend("boom")
        with pytest.raises(AdapterRefusal):
            warm_up_adapter_backend([_branches()["lora"]],
                                    activation_dtypes=(torch.float32,),
                                    warn=lambda m, c: warnings.append((m, c)),
                                    log=lambda line: None, strict=True)
        assert [code for _, code in warnings] == [LATCH_CODE]
        assert "before it produced any result" in warnings[0][0]


# -- warm-up ---------------------------------------------------------------

class TestWarmUp:
    def test_warm_up_probes_every_region_before_any_step(self):
        probed = []

        def fake(branch, x):
            probed.append(tuple(x.shape))
            return branch.reference_delta(x)

        _register("good", fake)
        select_adapter_backend("good")
        branches = list(_branches().values())
        report = warm_up_adapter_backend(branches,
                                         activation_dtypes=(torch.float32,),
                                         log=lambda line: None)
        assert report.backend == "good"
        assert report.usable == report.regions > 0
        assert not report.latched
        calls_after_warm_up = len(probed)
        for branch in branches:
            branch.forward_delta(_x())
        assert len(probed) == calls_after_warm_up + len(branches), (
            "a warmed region must not be probed again inside the step")

    def test_the_run_dtype_does_not_replace_a_branchs_own(self):
        """The MiniMax-H3 shape: fp32 I/O heads inside a bf16 run, no autocast.

        The run dtype alone fabricates a bf16 region this branch never sees --
        whose probe fails on a real dtype mismatch -- and leaves the fp32 region
        it does see unwarmed, which is the step-1 stall warm-up exists to
        prevent. Both halves are asserted here.
        """
        probed = []
        _register("good", lambda branch, x: (probed.append(1),
                                             branch.reference_delta(x))[1])
        select_adapter_backend("good")
        branch = _branches()["h3"]           # fp32 base, fp32 branch
        report = warm_up_adapter_backend([branch],
                                         activation_dtypes=(torch.bfloat16,),
                                         log=lambda line: None, strict=True)
        from core.adapters.execution import cached_verdict
        fp32 = cached_verdict("good", region_for(branch, torch.float32))
        bf16 = cached_verdict("good", region_for(branch, torch.bfloat16))
        assert report.regions == 2, "the union, not just the requested dtype"
        assert fp32 is not None and fp32.usable, "the region the branch really uses"
        assert bf16 is not None and not bf16.usable, (
            "the fabricated region fails on a genuine dtype mismatch")
        calls = len(probed)
        branch.forward_delta(_x())
        assert len(probed) == calls + 1, (
            "the fp32 region was warmed, so the first real forward must not "
            "probe inside the step")

    def test_strict_does_not_refuse_over_a_fabricated_region(self):
        """A branch served in one region of its union is served."""
        _register("good", lambda branch, x: branch.reference_delta(x))
        select_adapter_backend("good")
        report = warm_up_adapter_backend([_branches()["h3"]],
                                         activation_dtypes=(torch.bfloat16,),
                                         log=lambda line: None, strict=True)
        assert report.usable == 1 and report.regions == 2

    def test_a_run_with_no_adapters_is_not_refused(self):
        _register("good", lambda branch, x: branch.reference_delta(x))
        select_adapter_backend("good")
        report = warm_up_adapter_backend([], log=lambda line: None, strict=True)
        assert report.regions == 0 and report.usable == 0

    def test_warm_up_with_nothing_selected_is_a_no_op(self):
        report = warm_up_adapter_backend(list(_branches().values()),
                                         log=lambda line: None)
        assert report.backend == REFERENCE
        assert report.regions == 0

    def test_strict_warm_up_refuses_a_backend_that_can_serve_nothing(self):
        _register("narrow", lambda branch, x: branch.reference_delta(x),
                  pairs=frozenset({("lokr", False)}))
        select_adapter_backend("narrow")
        with pytest.raises(AdapterRefusal) as excinfo:
            warm_up_adapter_backend([_branches()["lora"]],
                                    activation_dtypes=(torch.float32,),
                                    log=lambda line: None, strict=True)
        assert excinfo.value.code == BACKEND_UNAVAILABLE_CODE

    def test_a_declaration_alone_admits_nothing(self):
        """``pairs``/``dtypes`` are the cheap half; the probe is the deciding one."""
        _register("liar", lambda branch, x: branch.reference_delta(x) * 2.0)
        select_adapter_backend("liar")
        report = warm_up_adapter_backend([_branches()["loha"]],
                                         activation_dtypes=(torch.float32,),
                                         log=lambda line: None)
        assert report.regions == 1 and report.usable == 0


# -- the dispatch point ----------------------------------------------------

class TestDispatchPoint:
    """A future fused backend must replace the delta computation without any
    architecture changing. These two assertions are that claim, checked."""

    def test_only_the_adapter_layers_define_or_call_the_branch_delta(self):
        allowed = {
            os.path.join("backend", "core", "adapters", "layers.py"),
            os.path.join("backend", "core", "adapters", "execution", "dispatch.py"),
            os.path.join("backend", "core", "adapters", "execution", "registry.py"),
            os.path.join("backend", "core", "adapters", "execution", "probe.py"),
        }
        offenders = []
        for path in sorted(pathlib.Path(_BACKEND).rglob("*.py")):
            if path.parent.name == "tests":
                continue
            relative = os.path.relpath(str(path), _REPO)
            if relative in allowed:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in (
                        "forward_delta", "reference_delta"):
                    offenders.append(f"{relative}:{node.lineno} defines {node.name}")
                elif (isinstance(node, ast.Attribute)
                      and node.attr in ("forward_delta", "reference_delta")):
                    offenders.append(f"{relative}:{node.lineno} calls {node.attr}")
        assert offenders == [], (
            "the branch delta is a core.adapters concern; an architecture that "
            "defines or calls it would have to change when a backend lands: "
            + "; ".join(offenders))

    def test_the_trainer_warms_the_backend_before_the_first_step(self):
        """The hook is called from ``train()``, before ``torch.compile`` and
        therefore before the loop -- read from the source, so a hook nobody
        calls fails here rather than passing silently."""
        source = pathlib.Path(_BACKEND, "core", "training", "base_trainer.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        train = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef) and node.name == "train")
        called = [node.func.attr for node in ast.walk(train)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr in ("_warm_up_adapter_execution_backend",
                                         "_maybe_compile_transformer")]
        assert called[:2] == ["_warm_up_adapter_execution_backend",
                             "_maybe_compile_transformer"]
