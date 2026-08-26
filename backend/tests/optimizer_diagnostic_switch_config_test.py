"""The two fused-backward diagnostic switches, and the channel that arms them.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/optimizer_diagnostic_switch_config_test.py -v

THE DEFECT
----------
``optimizer_update_census`` (G-RB3) and ``optimizer_state_host_resident`` (G-RB2)
were both assigned a bare ``False`` in ``BaseTrainer.__init__``, so nothing but
a hand-constructed trainer could set them. They had readers -- and a probe that
armed one by attribute -- but no config could. Every layer below them was
finished -- the census records every
fused update and raises on a shortfall, the allocator hands out pinned
per-parameter state -- but no run could switch either on. That matters for the
census in particular, because SENSENOVA_TRAINING_DESIGN.md 13.4 names it as
U-2-5's acceptance criterion ("step ごとの updated-param census == trainable 数"),
and the only way to arm it was to build a trainer object and set an attribute on
it by hand.

THE DECISION
------------
Both are read from the run's ``train_config`` -- the channel ``use_ema``,
``gradient_checkpointing`` and ``sensenova_full_finetune_save_format`` already
come through, and the one every trainer construction in ``train_runner`` already
passes. Neither gets a ``param_defaults`` -> ``routes`` -> ``openapi`` ->
frontend chain, deliberately:

* the census RAISES on a shortfall, so as a checkbox a false positive would take
  down a correct run, and its result is a stdout census with nothing for the UI
  to display;
* host-resident state only applies to the two ring-buffer optimizers, and the
  one route whose budget wants it allows ``adafactor`` alone
  (``SENSENOVA_FULL_FINETUNE_OPTIMIZERS``), so the option would be inert
  wherever it is selectable and unmeasured wherever it is not.

NEGATIVE CONTROLS
-----------------
``ShippedBehaviourTest`` reproduces the old assignment and shows it ignores the
config, and ``NoApiSurfaceTest`` records that neither key can be armed from an
API request, an OpenAPI schema or the generated YAML.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.base_trainer import BaseTrainer, setup_update_census  # noqa: E402

SWITCHES = ("optimizer_update_census", "optimizer_state_host_resident")

_BASE_TRAINER_SOURCE = (BACKEND_ROOT / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8"
)


def _init_assignment(attribute: str) -> ast.Assign:
    """The ``self.<attribute> = ...`` statement inside ``BaseTrainer.__init__``."""
    tree = ast.parse(_BASE_TRAINER_SOURCE)
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name == "BaseTrainer"):
            continue
        for fn in cls.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (isinstance(target, ast.Attribute)
                            and target.attr == attribute
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        return node
    raise AssertionError(f"no `self.{attribute} = ...` in BaseTrainer.__init__")


def _resolve(attribute: str, train_config: dict):
    """Evaluate the shipped right-hand side against a train_config.

    The expression is taken out of ``BaseTrainer.__init__`` rather than
    re-typed, so this cannot pass against a copy of the wiring while the real
    one is a literal.
    """
    expression = _init_assignment(attribute).value
    code = compile(ast.Expression(body=expression), "<base_trainer>", "eval")
    return eval(code, {}, {"_tc": train_config if train_config else {}})  # noqa: S307


class ConfigChannelTest(unittest.TestCase):
    def test_both_switches_default_off_with_no_config(self):
        for name in SWITCHES:
            with self.subTest(name):
                self.assertIs(_resolve(name, {}), False)
                self.assertIs(_resolve(name, None), False)

    def test_both_switches_are_armed_by_the_run_config(self):
        for name in SWITCHES:
            with self.subTest(name):
                self.assertIs(_resolve(name, {name: True}), True)

    def test_a_truthy_yaml_value_is_coerced_to_bool(self):
        # Hand-written YAML is the channel; `yes`/`1` must not reach the
        # optimizer kwargs as a string or an int.
        for name in SWITCHES:
            with self.subTest(name):
                self.assertIs(_resolve(name, {name: 1}), True)
                self.assertIs(_resolve(name, {name: 0}), False)

    def test_one_switch_does_not_arm_the_other(self):
        self.assertIs(_resolve("optimizer_state_host_resident",
                               {"optimizer_update_census": True}), False)
        self.assertIs(_resolve("optimizer_update_census",
                               {"optimizer_state_host_resident": True}), False)

    def test_train_config_is_the_channel_every_trainer_is_built_with(self):
        """The config the switches are read from is the one train_runner passes.

        Not an edit to train_runner: it already forwards ``train_config`` to all
        four trainer constructions, which is why the fix needed no change there.
        """
        runner = (BACKEND_ROOT / "core" / "training" / "train_runner.py").read_text(
            encoding="utf-8"
        )
        # LoRA, full-parameter, ControlNet and VAE, as of this commit. Asserted
        # as a floor rather than an equality so adding a fifth trainer does not
        # fail this file -- what matters is that the forwarding exists at all.
        self.assertGreaterEqual(runner.count("train_config=train_config,"), 4)


class CensusArmingTest(unittest.TestCase):
    """``setup_update_census`` is what the config value has to reach."""

    class _Stub:
        log_prefix = "[test]"
        _update_census = None

        def _fused_backward_target_module(self):
            raise RuntimeError("no module loaded")

    def test_off_takes_no_census(self):
        stub = self._Stub()
        stub.optimizer_update_census = _resolve("optimizer_update_census", {})
        self.assertIsNone(setup_update_census(stub, []))

    def test_on_arms_a_census(self):
        stub = self._Stub()
        stub.optimizer_update_census = _resolve(
            "optimizer_update_census", {"optimizer_update_census": True})
        census = setup_update_census(stub, [])
        self.assertIsNotNone(census)
        self.assertIs(census, stub._update_census)


class ShippedBehaviourTest(unittest.TestCase):
    """Negative control: the assignment as it shipped, evaluated the same way."""

    def test_a_literal_false_ignores_every_config(self):
        shipped = ast.parse("self.optimizer_update_census = False").body[0]
        code = compile(ast.Expression(body=shipped.value), "<shipped>", "eval")
        for config in ({}, {"optimizer_update_census": True},
                       {"optimizer_state_host_resident": True}):
            self.assertIs(eval(code, {}, {"_tc": config}), False)  # noqa: S307

    def test_the_literal_is_gone_from_both_switches(self):
        for name in SWITCHES:
            with self.subTest(name):
                self.assertNotIn(f"self.{name} = False", _BASE_TRAINER_SOURCE)


class NoApiSurfaceTest(unittest.TestCase):
    """Negative control: neither switch can be armed from the product API.

    Recorded rather than assumed, because "config-channel only" is the decision
    -- if a later change gives one of them a request field, this test fails and
    the reason above has to be revisited (and, per
    docs/guides/ADD_A_PARAMETER.md, the rest of the chain added).
    """

    # Prose may name them -- PUT /training/runs/{id} documents that it carries
    # them across a regeneration (config_edit_key_preservation_test.py). What is
    # asserted is that no layer *declares* either as a settable parameter.
    DECLARATION_FILES = (
        BACKEND_ROOT / "api" / "param_defaults.py",
        BACKEND_ROOT / "api" / "routes.py",
        BACKEND_ROOT / "core" / "training" / "training_config.py",
        REPO_ROOT / "openapi.yaml",
        REPO_ROOT / "frontend" / "src" / "utils" / "api.ts",
    )

    def test_no_layer_of_the_parameter_chain_declares_either_switch(self):
        declaration = re.compile(
            r"^\s*[\"']?(%s)[\"']?\s*[:=?]" % "|".join(SWITCHES))
        for path in self.DECLARATION_FILES:
            if not path.is_file():
                continue
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                with self.subTest(path=path.name, lineno=lineno):
                    self.assertIsNone(declaration.match(line), line.strip())

    def test_neither_switch_is_a_request_field(self):
        from api.routes import TrainingRunCreateRequest

        for name in SWITCHES:
            with self.subTest(name):
                self.assertNotIn(name, TrainingRunCreateRequest.model_fields)

    def test_no_generated_config_can_emit_either_switch(self):
        from core.training.training_config import train_section_key_vocabulary

        for name in SWITCHES:
            with self.subTest(name):
                self.assertNotIn(name, train_section_key_vocabulary())

    def test_training_defaults_has_neither_key(self):
        from api.param_defaults import TRAINING_DEFAULTS

        for name in SWITCHES:
            with self.subTest(name):
                self.assertNotIn(name, TRAINING_DEFAULTS)


class UnchangedElsewhereTest(unittest.TestCase):
    """Nothing about LoRA runs, or any other trainer, moved."""

    def test_the_census_is_still_fused_backward_only(self):
        # setup_update_census is called from the fused-backward setup, not from
        # setup_optimizer generally: a non-fused run gets the printed note, not
        # a census. Arming it from a config does not change where it applies.
        self.assertEqual(_BASE_TRAINER_SOURCE.count("setup_update_census(self,"), 1)
        self.assertIn("optimizer_update_census is set but this", _BASE_TRAINER_SOURCE)

    def test_setting_the_attribute_directly_still_works(self):
        # A probe holds a live trainer and sets the attribute after construction
        # (probes/sensenova_full_finetune.py:195). The readers stayed getattr-based,
        # so that path is unaffected by the config channel.
        for name in SWITCHES:
            with self.subTest(name):
                self.assertIn(f'getattr(self, "{name}", False)', _BASE_TRAINER_SOURCE)
        self.assertTrue(hasattr(BaseTrainer, "_ringbuffer_optimizer_kwargs"))

    def test_the_probe_still_arms_the_census_by_attribute(self):
        probe = (BACKEND_ROOT / "core" / "training" / "probes"
                 / "sensenova_full_finetune.py").read_text(encoding="utf-8")
        self.assertIn("trainer.optimizer_update_census = True", probe)


if __name__ == "__main__":
    unittest.main()
