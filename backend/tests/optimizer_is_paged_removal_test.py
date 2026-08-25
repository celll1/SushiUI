"""Guard: ``optimizer_is_paged`` is gone, and its absence breaks nothing.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/optimizer_is_paged_removal_test.py -v

WHAT WAS REMOVED AND WHY
------------------------
``optimizer_is_paged`` ran the full length of the stack -- param_defaults,
routes, openapi.yaml, a "Paged (CPU offload)" checkbox in the training panel,
the emitted YAML, and ``BaseTrainer.__init__`` -- and was then read by nothing.
``OptimizerFactory.create_optimizer`` selects ``paged_adamw`` /
``paged_adamw8bit`` / ``paged_lion8bit`` from the TYPE STRING alone.

It was removed rather than wired, for two reasons:

* The panel's optimizer dropdown offers no ``paged_*`` name, so paging is not
  reachable from the product; the checkbox was its only affordance and it was
  inert. The types themselves are kept -- they work, and a direct API call or a
  hand-written YAML can select them.
* Wiring it would have been dangerous. ``setup_optimizer`` refuses Block Swap +
  fused optimizer groups for the 8-bit optimizers because they cannot update a
  CPU-resident parameter. Mapping ``adamw8bit -> paged_adamw8bit`` on a ticked
  box would have walked a user straight through that refusal into the crash it
  exists to prevent -- and the paged names were themselves missing from the
  refusal list, which is fixed alongside the removal and pinned below.

WHAT EACH GROUP PINS
--------------------
* ``RemovalIsCompleteTest`` -- no read or write of the name survives in
  backend source (AST, so comments are allowed and code is not), it is gone
  from the openapi schema exactly once, and the frontend mentions it only in
  comments.
* ``BackwardCompatibilityTest`` -- the decision on old data: SILENTLY IGNORED.
  A request carrying the field, a YAML in ``training.db`` carrying the key, and
  the /params -> edit -> regenerate round trip over such a YAML all still work;
  the key is simply not surfaced and not re-emitted.
* ``PagedEightBitRefusalTest`` -- the refusal FIRES for ``paged_adamw8bit`` and
  ``paged_lion8bit`` under Block Swap + fused optimizer groups. Asserted as a
  raised exception from the real ``setup_optimizer``, not as a name present in
  a list.
"""

from __future__ import annotations

import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.training.base_trainer import BaseTrainer  # noqa: E402

DEAD = "optimizer_is_paged"


class RemovalIsCompleteTest(unittest.TestCase):
    """The name appears in no executable position anywhere in the backend."""

    # The tests are allowed to name it: this file does, and
    # optimizer_option_threading_test still checks that the VAE config gate --
    # which refuses by PREFIX -- keeps refusing it in a hand-merged YAML.
    SKIP_DIRS = {"tests", "__pycache__"}

    def _sources(self) -> List[Path]:
        files = []
        for path in _BACKEND.rglob("*.py"):
            rel = path.relative_to(_BACKEND)
            if rel.parts and rel.parts[0] in self.SKIP_DIRS:
                continue
            if "__pycache__" in rel.parts:
                continue
            files.append(path)
        return files

    def test_the_scan_actually_reads_the_backend(self):
        """A scan that finds no files would make every assertion below vacuous."""
        files = self._sources()
        self.assertGreater(len(files), 100, "backend source scan found almost nothing")
        names = {p.name for p in files}
        for expected in ("base_trainer.py", "train_runner.py", "training_config.py",
                         "param_defaults.py", "routes.py", "optimizer_factory.py"):
            self.assertIn(expected, names)

    def test_no_backend_source_reads_or_writes_the_name(self):
        """AST, not grep: a comment may mention it, no code may use it.

        Covers every executable shape it had -- ``self.optimizer_is_paged``
        (attribute), ``optimizer_is_paged=...`` (keyword / parameter),
        ``train_config.get('optimizer_is_paged', ...)`` and
        ``train['optimizer_is_paged'] = ...`` (string constant).
        """
        offenders: List[str] = []
        for path in self._sources():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:  # pragma: no cover - would be a real bug
                self.fail(f"{path} does not parse: {exc}")
            for node in ast.walk(tree):
                hit = (
                    (isinstance(node, ast.Attribute) and node.attr == DEAD)
                    or (isinstance(node, ast.Name) and node.id == DEAD)
                    or (isinstance(node, ast.keyword) and node.arg == DEAD)
                    or (isinstance(node, ast.arg) and node.arg == DEAD)
                    or (isinstance(node, ast.Constant) and node.value == DEAD)
                )
                if hit:
                    offenders.append(
                        f"{path.relative_to(_REPO)}:{getattr(node, 'lineno', '?')}")
        self.assertEqual(offenders, [], f"{DEAD} is still used by: {offenders}")

    def test_the_ast_scan_would_catch_each_shape(self):
        """The detector itself, mutation-tested against every removed form."""
        sources = (
            f"self.{DEAD} = flag",                 # attribute
            f"x = {DEAD}",                          # name
            f"Trainer({DEAD}=value)",               # keyword
            f"def f({DEAD}=False): pass",           # parameter
            f"train_config.get('{DEAD}', False)",   # string constant
            f"train['{DEAD}'] = True",              # string constant
        )
        for source in sources:
            with self.subTest(source=source):
                tree = ast.parse(source)
                found = any(
                    (isinstance(n, ast.Attribute) and n.attr == DEAD)
                    or (isinstance(n, ast.Name) and n.id == DEAD)
                    or (isinstance(n, ast.keyword) and n.arg == DEAD)
                    or (isinstance(n, ast.arg) and n.arg == DEAD)
                    or (isinstance(n, ast.Constant) and n.value == DEAD)
                    for n in ast.walk(tree)
                )
                self.assertTrue(found, source)
        # ...and does not fire on a comment, which is what the surviving
        # mentions in backend source are.
        tree = ast.parse(f"# {DEAD} was removed here\nx = 1\n")
        self.assertFalse(any(
            (isinstance(n, ast.Name) and n.id == DEAD)
            or (isinstance(n, ast.Constant) and n.value == DEAD)
            for n in ast.walk(tree)))

    def test_base_trainer_no_longer_accepts_the_keyword(self):
        parameters = BaseTrainer.__init__.__code__.co_varnames
        self.assertNotIn(DEAD, parameters)

    def test_it_is_gone_from_the_openapi_schema(self):
        spec_path = _REPO / "openapi.yaml"
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        schemas = spec["components"]["schemas"]
        request = schemas["TrainingRunCreateRequest"]["properties"]
        # Sanity: the schema is the one we think, so an empty/renamed schema
        # cannot make the absence check pass trivially.
        self.assertIn("optimizer", request)
        self.assertIn("optimizer_cautious", request)
        self.assertNotIn(DEAD, request)
        for name, schema in schemas.items():
            self.assertNotIn(DEAD, schema.get("properties", {}) or {}, name)
        # Raw text too: this file has had duplicate keys from concurrent edits,
        # and yaml.safe_load silently keeps only the last of a duplicated pair.
        self.assertNotIn(DEAD, spec_path.read_text(encoding="utf-8"))

    def test_the_paged_optimizer_types_are_kept(self):
        """Only the boolean went. The types it pretended to select still work."""
        from core.training.optimizer_factory import OptimizerFactory

        available = OptimizerFactory.get_available_optimizers()
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:  # pragma: no cover - bitsandbytes is installed here
            self.skipTest("bitsandbytes not installed; paged types are hidden")
        for name in ("paged_adamw", "paged_adamw8bit", "paged_lion8bit"):
            self.assertIn(name, available)
            # Still described, i.e. still a supported product surface.
            self.assertTrue(OptimizerFactory.get_optimizer_info(name))

    def test_the_frontend_mentions_it_only_in_comments(self):
        """No npm run here; a line-level check is the substitute."""
        for rel in ("frontend/src/components/training/TrainingConfig.tsx",
                    "frontend/src/utils/api.ts"):
            path = _REPO / rel
            self.assertTrue(path.exists(), rel)
            for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1):
                if DEAD not in line and "optimizerIsPaged" not in line:
                    continue
                stripped = line.strip()
                self.assertTrue(
                    stripped.startswith("//") or stripped.startswith("*")
                    or stripped.startswith("/*"),
                    f"{rel}:{number} still uses the removed flag: {stripped}",
                )


class BackwardCompatibilityTest(unittest.TestCase):
    """Old runs and old clients keep working; the key is silently ignored.

    The decision, stated: SILENTLY IGNORED, at every boundary that can see it.
    ``TrainingRunCreateRequest`` does not declare the field and Pydantic's
    default ``extra="ignore"`` drops it; ``_extract_request_params_from_yaml``
    iterates the model's fields, so a YAML key with no field is never read; the
    config generators never write it back. Nothing raises anywhere.
    """

    OLD_TRAIN_SECTION = {
        "steps": 100,
        "lr": 1e-4,
        "optimizer": "adamw8bit",
        # What every YAML written before the removal carries.
        DEAD: True,
        "optimizer_cautious": True,
        "optimizer_weight_decay": 0.02,
    }

    def test_a_request_still_carrying_the_field_is_accepted_and_dropped(self):
        from api.routes import TrainingRunCreateRequest

        request = TrainingRunCreateRequest(
            training_method="lora",
            base_model_path="model.safetensors",
            optimizer="adamw8bit",
            **{DEAD: True},
        )
        self.assertFalse(hasattr(request, DEAD))
        self.assertNotIn(DEAD, request.model_dump())
        # The neighbouring option is still honoured -- i.e. the request was
        # accepted, not merely "did not raise because everything was dropped".
        self.assertEqual(request.optimizer, "adamw8bit")

    def test_an_old_yaml_still_resolves_through_the_edit_form_extractor(self):
        from api.routes import _extract_request_params_from_yaml

        params = _extract_request_params_from_yaml(
            {"train": dict(self.OLD_TRAIN_SECTION), "network": {"type": "lora"}},
            job="lora",
        )
        self.assertNotIn(DEAD, params)
        # The rest of the section survives, so this is not "the extractor
        # returned nothing".
        self.assertEqual(params["optimizer"], "adamw8bit")
        self.assertTrue(params["optimizer_cautious"])
        self.assertEqual(params["optimizer_weight_decay"], 0.02)

    def test_the_edit_round_trip_over_an_old_yaml_regenerates_a_config(self):
        """/params -> TrainingRunCreateRequest -> generate -> parse, no raise."""
        from api.routes import TrainingRunCreateRequest, _extract_request_params_from_yaml
        from core.training.training_config import TrainingConfigGenerator

        params = _extract_request_params_from_yaml(
            {"train": dict(self.OLD_TRAIN_SECTION), "network": {"type": "lora"}},
            job="lora",
        )
        params.update(training_method="lora", base_model_path="model.safetensors",
                      total_steps=10)
        request = TrainingRunCreateRequest(**params)
        text = TrainingConfigGenerator.generate_lora_config(
            request.model_dump(), run_name="paged_removal",
            base_model_path="model.safetensors", output_dir="out",
            dataset_path="data",
        )
        train = yaml.safe_load(text)["config"]["process"][0]["train"]
        self.assertNotIn(DEAD, train)
        self.assertTrue(train["optimizer_cautious"])

    def test_the_generator_drops_the_key_even_when_handed_it_directly(self):
        from core.training.training_config import TrainingConfigGenerator

        text = TrainingConfigGenerator.generate_lora_config(
            {"total_steps": 10, "optimizer": "adamw8bit", DEAD: True},
            run_name="paged_removal_direct",
            base_model_path="model.safetensors", output_dir="out",
            dataset_path="data",
        )
        train = yaml.safe_load(text)["config"]["process"][0]["train"]
        self.assertNotIn(DEAD, train)
        self.assertEqual(train["optimizer"], "adamw8bit")

    def test_training_defaults_no_longer_publishes_it(self):
        from api.param_defaults import TRAINING_DEFAULTS

        self.assertNotIn(DEAD, TRAINING_DEFAULTS)
        # Nothing else moved.
        self.assertEqual(TRAINING_DEFAULTS["optimizer"], "adamw8bit")
        self.assertIs(TRAINING_DEFAULTS["optimizer_cautious"], False)


class _StubTrainer:
    """The smallest object ``setup_optimizer`` can run against.

    Same shape as ``optimizer_option_threading_test._StubTrainer``, minus the
    removed attribute -- which is itself part of what is under test here: the
    real method must not read it.
    """

    setup_optimizer = BaseTrainer.setup_optimizer
    _report_effective_component_lrs = BaseTrainer._report_effective_component_lrs
    _record_configured_group_lrs = BaseTrainer._record_configured_group_lrs
    _name_configured_groups = BaseTrainer._name_configured_groups
    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _resolved_optimizer_hyperparameters = BaseTrainer._resolved_optimizer_hyperparameters
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass
    _setup_fused_optimizer_groups = BaseTrainer._setup_fused_optimizer_groups
    _attach_stochastic_rounding = BaseTrainer._attach_stochastic_rounding
    _RINGBUFFER_ONLY_OPTIONS = BaseTrainer._RINGBUFFER_ONLY_OPTIONS
    _NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS = BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS

    def __init__(self, **overrides: Any):
        self.log_prefix = "[StubTrainer]"
        self.learning_rate = 1e-4
        self.weight_dtype = torch.bfloat16
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.use_ema = False
        self.config: Dict[str, Any] = {}
        self.optimizer_cautious = False
        self.optimizer_beta1 = None
        self.optimizer_beta2 = None
        self.optimizer_epsilon = None
        self.optimizer_weight_decay = None
        self.optimizer_schedule_free = False
        self.optimizer_warmup_steps = 0
        self.optimizer_schedule_free_r = 0.0
        self.optimizer_schedule_free_weight_lr_power = 2.0
        self.optimizer_use_radam = False
        self.optimizer_stochastic_rounding = False
        for key, value in overrides.items():
            setattr(self, key, value)
        self.param = torch.nn.Parameter(torch.zeros(4))

    def setup_trainable_parameters(self):
        return [{"params": [self.param], "lr": self.learning_rate}]

    def _setup_ema(self):
        pass


class PagedEightBitRefusalTest(unittest.TestCase):
    """Block Swap + fused optimizer groups refuses the PAGED 8-bit names too.

    The list named ``adamw8bit`` / ``lion8bit`` / ``adafactor8bit`` and both
    ring buffers, but not ``paged_adamw8bit`` / ``paged_lion8bit``. Those are
    the same 8-bit kernels; paging moves the optimizer STATE to host memory and
    does nothing about the PARAMETER that Block Swap moved to the CPU, which is
    what the refusal is about. So the identical crash was reachable under a
    name the check did not know.

    Asserted through the real ``setup_optimizer``: the refusal has to FIRE.
    """

    EIGHT_BIT = ("adamw8bit", "lion8bit",
                 "paged_adamw8bit", "paged_lion8bit",
                 "adamw8bit_ringbuffer", "lion8bit_ringbuffer")

    def _setup(self, optimizer_type: str, **overrides: Any):
        stub = _StubTrainer(blocks_to_swap=8, num_optimizer_groups=4, **overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            stub.setup_optimizer(optimizer_type=optimizer_type,
                                 lr_scheduler_type="constant", total_steps=10)
        return stub

    def test_every_eight_bit_name_including_the_paged_ones_is_refused(self):
        for optimizer_type in self.EIGHT_BIT:
            with self.subTest(optimizer_type=optimizer_type):
                with self.assertRaises(ValueError) as ctx:
                    self._setup(optimizer_type)
                message = str(ctx.exception)
                self.assertIn(optimizer_type, message)
                self.assertIn("8-bit", message)
                self.assertIn("num_optimizer_groups", message)

    def test_the_refusal_is_specific_to_this_combination(self):
        """Not "raises for everything": the escape hatches it names must work.

        Each of the three remedies the message offers is exercised, so a
        blanket ``raise`` in ``setup_optimizer`` would fail here even though it
        would pass the test above.
        """
        # (1) num_optimizer_groups=0 -- the ring-buffer/8bit path.
        stub = _StubTrainer(blocks_to_swap=8, num_optimizer_groups=0)
        with contextlib.redirect_stdout(io.StringIO()):
            stub.setup_optimizer(optimizer_type="adamw8bit",
                                 lr_scheduler_type="constant", total_steps=10)
        self.assertIsNotNone(stub.optimizer)

        # (3) Block Swap off.
        stub = _StubTrainer(blocks_to_swap=0, num_optimizer_groups=4)
        with contextlib.redirect_stdout(io.StringIO()):
            stub.setup_optimizer(optimizer_type="paged_adamw8bit",
                                 lr_scheduler_type="constant", total_steps=10)
        self.assertIsNotNone(stub.optimizer)

    def test_a_non_eight_bit_optimizer_still_runs_the_fused_group_path(self):
        """(2) "use a non-8bit optimizer": adamw under the same configuration."""
        import core.training.optimizers.fused_optimizer_groups as fog

        created: List[Dict[str, Any]] = []

        def _create(**kwargs):
            created.append(kwargs)
            return [torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))],
                                      lr=kwargs["learning_rate"])]

        class _Groups:
            def __init__(self, optimizers, max_grad_norm=0.0):
                pass

            def register_hooks(self):
                pass

        originals = (fog.create_optimizer_groups, fog.FusedOptimizerGroups)
        fog.create_optimizer_groups, fog.FusedOptimizerGroups = _create, _Groups
        try:
            self._setup("adamw")
        finally:
            fog.create_optimizer_groups, fog.FusedOptimizerGroups = originals

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["optimizer_type"], "adamw")

    def test_the_refusal_survives_a_capitalised_type_name(self):
        """The check lowercases; a YAML written by hand may not."""
        with self.assertRaises(ValueError):
            self._setup("Paged_AdamW8bit")


class NoWarningSurvivesTheFlagTest(unittest.TestCase):
    """The report added when the flag was known-dead went with the flag."""

    def test_setup_optimizer_says_nothing_about_paging(self):
        stub = _StubTrainer()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stub.setup_optimizer(optimizer_type="adamw",
                                 lr_scheduler_type="constant", total_steps=10)
        output = buffer.getvalue()
        self.assertNotIn(DEAD, output)
        self.assertNotIn("is not applied", output)
        # The stub really did run the method it is meant to run.
        self.assertIsNotNone(stub.optimizer)


if __name__ == "__main__":
    unittest.main()
