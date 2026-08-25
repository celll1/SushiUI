"""G-RB3: the per-step census of parameters an update actually reached.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/ringbuffer_update_census_test.py -v

THE FAILURE MODE
----------------
Under the fused backward pass ``optimizer.step()`` is never called, so every
update comes from that parameter's own post-accumulate-grad hook. A parameter
whose hook never fires -- or returns early -- is updated by nothing for the
whole run, and the loss falls normally regardless. Loss curves cannot see it;
counting the updates can. ``docs/guides/SENSENOVA_TRAINING_DESIGN.md`` 6.5
registers that count as gate G-RB3.

NEGATIVE CONTROL
----------------
``CensusDisabledTest`` records the shipped behaviour with the census switched
off: a parameter that receives no update is accepted in silence, and
``assert_complete`` says nothing. That is the state this gate exists to leave.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.optimizers.update_census import (  # noqa: E402
    CENSUS_ATTR,
    UpdateCensus,
    attach_update_census,
    enable_update_census,
    get_update_census,
    record_param_update,
    trainable_params_of,
)


class _FakeOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups
        self.state = {}


def _params(n, requires_grad=True):
    return [nn.Parameter(torch.zeros(2), requires_grad=requires_grad) for _ in range(n)]


class ExpectationSetTest(unittest.TestCase):
    def test_expectation_is_param_groups_deduplicated_and_trainable_only(self):
        shared = nn.Parameter(torch.zeros(2))
        frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
        a, b = _params(2)
        optimizer = _FakeOptimizer([
            {"params": [a, shared, frozen]},
            {"params": [shared, b]},
        ])
        expected = trainable_params_of(optimizer)
        # shared once, frozen never: a frozen parameter receives no gradient, so
        # neither a hook nor step() would move it and demanding an update for it
        # would fail on a correct run.
        self.assertEqual(len(expected), 3)
        self.assertEqual({id(p) for p in expected}, {id(a), id(b), id(shared)})

    def test_expect_returns_count_and_names_come_from_the_module(self):
        module = nn.Linear(3, 3, bias=True)
        census = UpdateCensus()
        names = {id(p): n for n, p in module.named_parameters()}
        count = census.expect(list(module.parameters()), names)
        self.assertEqual(count, 2)
        self.assertEqual(census.missing(), ["bias", "weight"])


class CensusCompletenessTest(unittest.TestCase):
    def test_complete_step_passes(self):
        ps = _params(4)
        optimizer = _FakeOptimizer([{"params": ps}])
        census = enable_update_census(optimizer)
        for p in ps:
            record_param_update(optimizer, p)
        census.assert_complete("test")
        self.assertEqual(census.updated_count, 4)
        self.assertEqual(census.expected_count, 4)
        self.assertEqual(census.missing(), [])

    def test_one_missing_parameter_raises_and_names_it(self):
        module = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
        ps = list(module.parameters())
        optimizer = _FakeOptimizer([{"params": ps}])
        census = enable_update_census(optimizer, module)
        record_param_update(optimizer, ps[0])
        with self.assertRaises(RuntimeError) as ctx:
            census.assert_complete("global_step=1")
        message = str(ctx.exception)
        self.assertIn("1 of 2", message)
        self.assertIn("1.weight", message)
        self.assertIn("global_step=1", message)

    def test_begin_step_clears_the_previous_step(self):
        ps = _params(2)
        optimizer = _FakeOptimizer([{"params": ps}])
        census = enable_update_census(optimizer)
        for p in ps:
            record_param_update(optimizer, p)
        census.assert_complete()
        census.begin_step(True)
        self.assertEqual(census.updated_count, 0)
        with self.assertRaises(RuntimeError):
            census.assert_complete()

    def test_updates_for_unowned_parameters_are_counted_separately(self):
        ps = _params(2)
        stranger = nn.Parameter(torch.zeros(2))
        optimizer = _FakeOptimizer([{"params": ps}])
        census = enable_update_census(optimizer)
        for p in ps:
            record_param_update(optimizer, p)
        record_param_update(optimizer, stranger)
        census.assert_complete()
        self.assertEqual(census.unexpected_count(), 1)


class CensusDisabledTest(unittest.TestCase):
    """Negative control: the behaviour without this gate."""

    def test_no_census_attached_records_nothing_and_raises_nothing(self):
        ps = _params(3)
        optimizer = _FakeOptimizer([{"params": ps}])
        self.assertIsNone(get_update_census(optimizer))
        # This is the shipped path: the update sites call record_param_update
        # unconditionally and it is a no-op when nothing is attached.
        for p in ps:
            record_param_update(optimizer, p)
        self.assertIsNone(getattr(optimizer, CENSUS_ATTR, None))

    def test_disarmed_census_accepts_a_skipped_parameter_in_silence(self):
        ps = _params(3)
        optimizer = _FakeOptimizer([{"params": ps}])
        census = UpdateCensus()
        attach_update_census(optimizer, census)
        census.expect(ps)
        census.begin_step(False)
        record_param_update(optimizer, ps[0])  # two parameters skipped
        self.assertEqual(census.updated_count, 0)  # recording is gated too
        census.assert_complete("disarmed")  # must not raise
        self.assertEqual(census.steps_checked, 0)


class UnreachableParameterExemptionTest(unittest.TestCase):
    """L3: parameters no gradient can reach must not fail a correct run.

    SenseNova's understanding branch has five, and U-2-5 asserts exactly that
    census -- so without an exemption this gate would raise every step on the
    route it was written for.
    """

    def _module(self):
        return nn.ModuleDict({
            "alive": nn.Linear(2, 2, bias=False),
            "dead": nn.Linear(2, 2, bias=True),
        })

    def test_exempt_by_module_path_covers_its_tensors(self):
        module = self._module()
        optimizer = _FakeOptimizer([{"params": list(module.parameters())}])
        names = {id(p): n for n, p in module.named_parameters()}
        census = UpdateCensus()
        # "dead" covers dead.weight AND dead.bias, without naming either.
        count = census.expect(list(module.parameters()), names, exempt=["dead"])
        self.assertEqual(count, 1)
        self.assertEqual(census.missing(), ["alive.weight"])

    def test_an_exempt_parameter_never_updated_still_passes(self):
        module = self._module()
        optimizer = _FakeOptimizer([{"params": list(module.parameters())}])
        census = enable_update_census(optimizer, module, exempt=["dead"])
        record_param_update(optimizer, module["alive"].weight)
        census.assert_complete("exempt")  # must not raise

    def test_a_non_exempt_parameter_is_still_demanded(self):
        module = self._module()
        optimizer = _FakeOptimizer([{"params": list(module.parameters())}])
        census = enable_update_census(optimizer, module, exempt=["dead"])
        with self.assertRaises(RuntimeError) as ctx:
            census.assert_complete("exempt")
        self.assertIn("alive.weight", str(ctx.exception))

    def test_a_prefix_does_not_match_a_longer_sibling_name(self):
        """``dead`` must not exempt ``deadweight``."""
        module = nn.ModuleDict({
            "dead": nn.Linear(2, 2, bias=False),
            "deadweight": nn.Linear(2, 2, bias=False),
        })
        names = {id(p): n for n, p in module.named_parameters()}
        census = UpdateCensus()
        census.expect(list(module.parameters()), names, exempt=["dead"])
        self.assertEqual(census.missing(), ["deadweight.weight"])

    def test_the_exemption_list_comes_from_the_existing_predictor(self):
        """It must not be a second hand-maintained copy of the five names."""
        from core.models.sensenova.sensenova_lora import und_gradient_unreachable_paths
        from core.training.base_trainer import census_exempt_names

        class Stub:
            is_sensenova = True
            log_prefix = "[test]"

        names = census_exempt_names(Stub())
        self.assertEqual(set(names), und_gradient_unreachable_paths())
        self.assertEqual(len(names), 5)

    def test_no_exemption_for_other_architectures(self):
        from core.training.base_trainer import census_exempt_names

        class Stub:
            is_sensenova = False
            log_prefix = "[test]"

        self.assertEqual(tuple(census_exempt_names(Stub())), ())


class CensusScopeIsParamGroupsTest(unittest.TestCase):
    """L2: what the gate guarantees, written down as a test.

    "Every parameter the optimizer OWNS was updated" -- not "every trainable
    parameter of the model was". The other direction is
    fused_backward_registration's orphan check.
    """

    def test_a_trainable_parameter_the_optimizer_does_not_own_is_invisible(self):
        module = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
        owned = [module[0].weight]  # module[1].weight deliberately left out
        optimizer = _FakeOptimizer([{"params": owned}])
        census = enable_update_census(optimizer, module)
        record_param_update(optimizer, module[0].weight)
        census.assert_complete("scope")  # passes, though module[1] is untrained
        self.assertEqual(census.expected_count, 1)

    def test_the_other_direction_is_covered_by_hook_registration(self):
        """The orphan check that DOES see it, so the pair is complete."""
        from core.training.optimizers.fused_backward_registration import (
            register_fused_backward_hooks,
        )
        module = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
        optimizer = _FakeOptimizer([{"params": [module[0].weight], "use_8bit": True}])
        with self.assertRaises(RuntimeError) as ctx:
            register_fused_backward_hooks(
                optimizer, module, "test", lambda p, g: (lambda param: None)
            )
        self.assertIn("in no param_group", str(ctx.exception))


class TrainerWiringTest(unittest.TestCase):
    def test_setup_update_census_is_off_by_default_and_arms_when_switched_on(self):
        from core.training.base_trainer import setup_update_census

        ps = _params(3)
        optimizer = _FakeOptimizer([{"params": ps}])

        class Stub:
            log_prefix = "[test]"
            optimizer_update_census = False
            _update_census = None

        stub = Stub()
        self.assertIsNone(setup_update_census(stub, [optimizer]))
        self.assertIsNone(get_update_census(optimizer))

        stub.optimizer_update_census = True
        census = setup_update_census(stub, [optimizer])
        self.assertIsNotNone(census)
        self.assertIs(get_update_census(optimizer), census)
        self.assertEqual(census.expected_count, 3)


if __name__ == "__main__":
    unittest.main()
