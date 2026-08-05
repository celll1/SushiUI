"""Guard: every optimizer option the API accepts must reach the optimizer.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/optimizer_option_threading_test.py -v

THE DEFECT
----------
``train_runner.main()`` builds four trainers (LoRA, ReLoRA, full-parameter,
ControlNet) and each construction re-reads the optimizer options out of the YAML
by hand. The full-parameter site read none of them: a full fine-tune ran with
``betas=(0.9, 0.999)``, ``eps=1e-8``, ``weight_decay=0.01``, no cautious
masking, no Schedule-Free and -- because ``optimizer_warmup_steps`` is also what
feeds the LR scheduler's ``num_warmup_steps`` -- no LR warmup at all, no matter
what the user set. The values were in the request, in the YAML and in the UI;
only the eleven lines that read them were missing.

Two more instances of the same shape were found next to it and are pinned here:

* ``_setup_fused_optimizer_groups`` re-created every optimizer with hardcoded
  ``weight_decay=0.01 / betas=(0.9, 0.999) / eps=1e-8``, discarding the
  configured values for exactly the Block-Swap runs that use it.
* ``lens_lora_scope`` / ``lens_img_lr_factor`` / ``lens_txt_lr_factor`` exist in
  ``param_defaults``, ``routes``, ``openapi.yaml`` and the training panel, but
  ``training_config`` never wrote them into the YAML, so ``lens_adapter`` and
  ``lora_trainer`` always read their fallbacks.

WHAT EACH GROUP PINS
--------------------
* ``TrainerCallSiteCensusTest`` -- the permanent anchor. Every optimizer option
  BaseTrainer accepts is passed at every trainer construction site in
  train_runner, and the value passed is the variable that was read from
  train_config under the SAME key, with the SAME default (so a hardcoded
  literal, a wrong key, a wrong ``.get()`` default and a later reassignment all
  fail).

  Its limitation, stated so this file is not read as a general
  parameter-plumbing guard: the anchor is keyed on ``BaseTrainer.__init__``'s
  parameter list. An option that reaches param_defaults, routes, openapi and
  the YAML but never becomes a BaseTrainer parameter is invisible to it -- that
  is the shape of the ``lens_*`` defect fixed alongside this work, and of
  ``optimizer_stochastic_rounding`` before 8547f93c. It also covers
  ``train_runner`` only: an option that arrives and is then ignored downstream
  needs an effect test, which is what the other groups here are.
* ``OptimizerOptionEffectTest`` -- the options change what the optimizer and the
  LR schedule actually do once they arrive.
* ``FusedPathTest`` -- the Block-Swap paths: configured hyperparameters survive
  the fused-optimizer-groups rebuild, and Schedule-Free is refused by the
  fused-backward path instead of dying with a KeyError inside backward().
* ``UnsupportedOptionIsNamedTest`` -- an option a given optimizer cannot honour
  is reported, never silently dropped.
* ``VaeOptimizerOptionTest`` -- the VAE trainer consumes ``optimizer`` and
  ``optimizer_weight_decay`` only; the rest are refused, not ignored.
* ``LensConfigKeyTest`` -- the Lens knobs reach the YAML the adapters read AND
  a narrowed scope actually removes LoRA targets. Getting the value there was
  only half of it: the consumer started from the all-true default scope and
  only ever set True, so a scope could widen but never narrow.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.base_trainer import BaseTrainer  # noqa: E402

# Every trainer in train_runner forwards **kwargs to BaseTrainer, so the set of
# options a trainer supports IS BaseTrainer's optimizer_* parameter list.
# Derived, not hand-listed: this repo has had hand-maintained option maps drift.
OPTIMIZER_OPTIONS = tuple(
    name for name in inspect.signature(BaseTrainer.__init__).parameters
    if name.startswith("optimizer_")
)

TRAINER_CALLS = ("LoRATrainer", "ReLoRATrainer", "FullParameterTrainer",
                 "ControlNetTrainer")


class _StubTrainer:
    """The smallest object BaseTrainer's optimizer methods can run against.

    Borrows the real methods; supplies only the attributes they read. No model,
    no CUDA, no dataset -- the code under test is optimizer construction.
    """

    setup_optimizer = BaseTrainer.setup_optimizer
    _resolved_optimizer_hyperparameters = BaseTrainer._resolved_optimizer_hyperparameters
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass
    _setup_fused_optimizer_groups = BaseTrainer._setup_fused_optimizer_groups
    _RINGBUFFER_ONLY_OPTIONS = BaseTrainer._RINGBUFFER_ONLY_OPTIONS

    def __init__(self, **overrides: Any):
        self.log_prefix = "[StubTrainer]"
        self.learning_rate = 1e-4
        self.weight_dtype = torch.bfloat16
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.use_ema = False
        self.config: Dict[str, Any] = {}
        self.optimizer_is_paged = False
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


class _FakeCudaTensor(torch.Tensor):
    """A CPU tensor that answers ``is_cuda``.

    Both ring-buffer optimizers skip parameters that are not on CUDA (Block Swap
    offloads them) and copy their state to the GPU when it is not already there.
    Their CUDA extensions take minutes to compile, so the update path is
    exercised here on CPU tensors that answer the residency check, with the
    extension replaced by a stand-in. Same device as
    ``bf16_stochastic_rounding_test._FakeCudaParameter``.
    """

    @property
    def is_cuda(self) -> bool:  # noqa: D401
        return True


def _fake_cuda(tensor: torch.Tensor, requires_grad: bool = False):
    return torch.Tensor._make_subclass(_FakeCudaTensor, tensor, requires_grad)


def _run_setup_optimizer(stub: _StubTrainer, **kwargs) -> str:
    """Call the real setup_optimizer, returning everything it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        stub.setup_optimizer(**kwargs)
    return buffer.getvalue()


class TrainerCallSiteCensusTest(unittest.TestCase):
    """Every trainer construction passes every optimizer option, from config.

    This is the anchor that stops the defect recurring: adding an
    ``optimizer_*`` parameter to BaseTrainer and forgetting one of the four call
    sites fails here instead of silently running with the fallback.
    """

    @classmethod
    def setUpClass(cls):
        import core.training.train_runner as train_runner

        cls.source_path = Path(inspect.getsourcefile(train_runner))
        cls.source = cls.source_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _call_sites(self) -> Dict[str, ast.Call]:
        sites: Dict[str, ast.Call] = {}
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in TRAINER_CALLS:
                continue
            self.assertNotIn(node.func.id, sites,
                             f"{node.func.id} is constructed more than once; "
                             f"this census assumes one site per trainer")
            sites[node.func.id] = node
        return sites

    def _assignments(self) -> List[ast.Assign]:
        """EVERY single-target `name = ...` assignment, in source order.

        Deliberately not filtered to ``train_config.get(...)``: collecting only
        the config reads would let a later plain reassignment
        (``optimizer_cautious = False`` after the read) pass unnoticed, because
        the config read would still be found. The nearest preceding assignment
        overall has to be the config read.
        """
        assignments = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ]
        return sorted(assignments, key=lambda n: n.lineno)

    @staticmethod
    def _config_read_key(node: ast.Assign):
        """The 'key' of `x = train_config.get('key', ...)`, else None."""
        value = node.value
        if not isinstance(value, ast.Call):
            return None
        func = value.func
        if (isinstance(func, ast.Attribute) and func.attr == "get"
                and isinstance(func.value, ast.Name)
                and func.value.id == "train_config"
                and value.args and isinstance(value.args[0], ast.Constant)):
            return value.args[0].value
        return None

    def test_the_option_list_is_the_one_we_think_it_is(self):
        """A derivation that silently returns nothing would pass every test."""
        self.assertEqual(len(OPTIMIZER_OPTIONS), 12, OPTIMIZER_OPTIONS)
        for name in ("optimizer_cautious", "optimizer_schedule_free",
                     "optimizer_warmup_steps", "optimizer_schedule_free_r",
                     "optimizer_schedule_free_weight_lr_power",
                     "optimizer_use_radam", "optimizer_stochastic_rounding",
                     "optimizer_beta1", "optimizer_beta2", "optimizer_epsilon",
                     "optimizer_weight_decay", "optimizer_is_paged"):
            self.assertIn(name, OPTIMIZER_OPTIONS)

    def test_every_trainer_call_site_passes_every_optimizer_option(self):
        sites = self._call_sites()
        self.assertEqual(set(sites), set(TRAINER_CALLS), sorted(sites))

        for trainer, call in sites.items():
            passed = {kw.arg for kw in call.keywords if kw.arg}
            missing = [name for name in OPTIMIZER_OPTIONS if name not in passed]
            self.assertEqual(
                missing, [],
                f"{trainer}(...) at line {call.lineno} does not pass "
                f"{missing}; those options are read from the YAML and dropped",
            )

    def test_each_option_carries_the_value_read_from_that_config_key(self):
        """Passing the keyword is not enough -- it must carry the config value.

        Catches the weaker failure mode where a call site passes the argument
        but hardcodes it (``optimizer_cautious=False``), which no
        keyword-presence check would notice.
        """
        assignments = self._assignments()
        sites = self._call_sites()

        for trainer, call in sites.items():
            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            for option in OPTIMIZER_OPTIONS:
                value = keywords[option]
                self.assertIsInstance(
                    value, ast.Name,
                    f"{trainer}: {option} is not passed as a variable "
                    f"(a literal here means the config value is ignored)",
                )
                # The nearest preceding assignment to that name is the one this
                # branch of main() uses (every branch re-reads its own copy),
                # and it has to BE the config read -- a plain reassignment
                # after the read would otherwise win at runtime and pass here.
                binding = [n for n in assignments
                           if n.targets[0].id == value.id and n.lineno < call.lineno]
                self.assertTrue(
                    binding,
                    f"{trainer}: {option}={value.id} is never assigned before "
                    f"the call",
                )
                key = self._config_read_key(binding[-1])
                self.assertEqual(
                    key, option,
                    f"{trainer}: the last assignment to {value.id} before line "
                    f"{call.lineno} is not train_config.get('{option}') "
                    f"(got {key!r} at line {binding[-1].lineno})",
                )

    def test_each_config_read_falls_back_to_the_trainer_default(self):
        """A wrong `.get()` default is a silent behaviour change.

        ``train_config.get('optimizer_beta1', 0.5)`` would pass every check
        above while changing what a config that omits the key does. Every read's
        default must equal the parameter default BaseTrainer declares. All
        twelve agree today; this is a guard for the next one.
        """
        parameters = inspect.signature(BaseTrainer.__init__).parameters
        checked = 0
        for node in self._assignments():
            key = self._config_read_key(node)
            if key not in OPTIMIZER_OPTIONS:
                continue
            if len(node.value.args) < 2:
                # No default given -> .get() returns None, which is the
                # declared default for the Optional[float] hyperparameters.
                self.assertIsNone(
                    parameters[key].default,
                    f"line {node.lineno}: train_config.get('{key}') has no "
                    f"default but BaseTrainer defaults to "
                    f"{parameters[key].default!r}",
                )
                checked += 1
                continue
            default = node.value.args[1]
            self.assertIsInstance(default, ast.Constant,
                                  f"line {node.lineno}: non-literal default")
            self.assertEqual(
                default.value, parameters[key].default,
                f"line {node.lineno}: train_config.get('{key}', "
                f"{default.value!r}) disagrees with BaseTrainer's default "
                f"{parameters[key].default!r}",
            )
            checked += 1
        # Four branches x twelve options; a zero here would mean the scan found
        # nothing and every assertion above was vacuous.
        self.assertGreaterEqual(checked, len(OPTIMIZER_OPTIONS) * len(TRAINER_CALLS))


class OptimizerOptionEffectTest(unittest.TestCase):
    """Once threaded, each option changes what actually runs."""

    def test_betas_epsilon_and_weight_decay_reach_the_optimizer(self):
        stub = _StubTrainer(optimizer_beta1=0.85, optimizer_beta2=0.95,
                            optimizer_epsilon=1e-6, optimizer_weight_decay=0.05)
        _run_setup_optimizer(stub, optimizer_type="adamw",
                             lr_scheduler_type="constant", total_steps=10)
        group = stub.optimizer.param_groups[0]
        self.assertEqual(group["betas"], (0.85, 0.95))
        self.assertEqual(group["eps"], 1e-6)
        self.assertEqual(group["weight_decay"], 0.05)

    def test_unset_hyperparameters_keep_the_documented_fallbacks(self):
        """What a dropped key produced -- pinned so the fix cannot move it."""
        stub = _StubTrainer()
        _run_setup_optimizer(stub, optimizer_type="adamw",
                             lr_scheduler_type="constant", total_steps=10)
        group = stub.optimizer.param_groups[0]
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-8)
        self.assertEqual(group["weight_decay"], 0.01)

    def test_warmup_steps_shape_the_learning_rate_schedule(self):
        """optimizer_warmup_steps is not ring-buffer-only: it IS the LR warmup.

        A full fine-tune that never read the key ran every step at the base LR
        with the UI reporting a warmup.
        """
        warm = _StubTrainer(optimizer_warmup_steps=10)
        _run_setup_optimizer(warm, optimizer_type="adamw",
                             lr_scheduler_type="constant_with_warmup",
                             total_steps=100)
        first = warm.lr_scheduler.get_last_lr()[0]
        self.assertLess(first, warm.learning_rate)
        for _ in range(10):
            warm.optimizer.step()
            warm.lr_scheduler.step()
        self.assertAlmostEqual(warm.lr_scheduler.get_last_lr()[0],
                               warm.learning_rate)

        flat = _StubTrainer(optimizer_warmup_steps=0)
        _run_setup_optimizer(flat, optimizer_type="adamw",
                             lr_scheduler_type="constant_with_warmup",
                             total_steps=100)
        self.assertAlmostEqual(flat.lr_scheduler.get_last_lr()[0],
                               flat.learning_rate)

    def test_ringbuffer_only_options_reach_the_factory(self):
        from core.training.optimizer_factory import OptimizerFactory

        recorded: Dict[str, Any] = {}
        original = OptimizerFactory.create_optimizer

        def _recorder(**kwargs):
            recorded.update(kwargs)
            return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-4)

        OptimizerFactory.create_optimizer = staticmethod(_recorder)
        try:
            stub = _StubTrainer(
                optimizer_cautious=True,
                optimizer_schedule_free=True,
                optimizer_warmup_steps=7,
                optimizer_schedule_free_r=0.5,
                optimizer_schedule_free_weight_lr_power=3.0,
                optimizer_use_radam=True,
                optimizer_stochastic_rounding=True,
            )
            _run_setup_optimizer(stub, optimizer_type="adamw8bit_ringbuffer",
                                 lr_scheduler_type="constant", total_steps=10)
        finally:
            OptimizerFactory.create_optimizer = original

        self.assertTrue(recorded["cautious"])
        self.assertTrue(recorded["schedule_free"])
        self.assertEqual(recorded["warmup_steps"], 7)
        self.assertEqual(recorded["r"], 0.5)
        self.assertEqual(recorded["weight_lr_power"], 3.0)
        self.assertTrue(recorded["use_radam"])
        self.assertTrue(recorded["stochastic_rounding"])

    def test_cautious_masking_changes_the_update(self):
        """The flag is honoured by the optimizer, not merely stored.

        Runs both ring-buffer optimizers' real ``step()`` against a recording
        stand-in for the CUDA extension (the kernel is what applies the mask),
        and pins that the flag arrives as the kernel's ``cautious`` argument.
        """
        import core.training.optimizers.adamw8bit_ringbuffer as adamw_mod
        import core.training.optimizers.lion8bit_ringbuffer as lion_mod

        seen: List[bool] = []

        class _Ext:
            def init_quantization_maps(self, *a, **k):
                pass

            def adamw_8bit_update(self, param, grad, s1, s2, a1, a2, beta1,
                                  beta2, eps, lr, wd, gnorm, step, cautious):
                seen.append(cautious)

            def lion_8bit_update(self, param, grad, exp_avg, absmax, beta1,
                                 beta2, eps, lr, wd, gnorm, step, cautious):
                seen.append(cautious)

        for module, cls_name in ((adamw_mod, "AdamW8bit_RingBuffer"),
                                 (lion_mod, "Lion8bit_RingBuffer")):
            for cautious in (True, False):
                seen.clear()
                original = module.get_extension
                module.get_extension = lambda: _Ext()
                try:
                    p = _fake_cuda(torch.zeros(8, dtype=torch.bfloat16), True)
                    p.grad = torch.ones(8, dtype=torch.bfloat16)
                    opt = getattr(module, cls_name)(
                        [p], lr=1e-4, weight_decay=0.0, use_8bit=True,
                        cautious=cautious,
                    )
                    opt.state[p]["is_8bit"] = True
                    for key in ("exp_avg", "exp_avg_sq"):
                        opt.state[p][key] = _fake_cuda(
                            torch.zeros(8, dtype=torch.uint8))
                    for key in ("absmax", "absmax1", "absmax2"):
                        opt.state[p][key] = _fake_cuda(torch.zeros(1))
                    with contextlib.redirect_stdout(io.StringIO()):
                        opt.step()
                finally:
                    module.get_extension = original
                self.assertEqual(seen, [cautious], cls_name)


class UnsupportedOptionIsNamedTest(unittest.TestCase):
    """An option the chosen optimizer cannot honour must be reported."""

    def test_non_ringbuffer_optimizer_names_the_options_it_drops(self):
        stub = _StubTrainer(optimizer_cautious=True,
                            optimizer_schedule_free=True,
                            optimizer_use_radam=True)
        output = _run_setup_optimizer(stub, optimizer_type="adamw",
                                      lr_scheduler_type="constant",
                                      total_steps=10)
        self.assertIn("optimizer_cautious", output)
        self.assertIn("optimizer_schedule_free", output)
        self.assertIn("optimizer_use_radam", output)
        self.assertIn("not supported by 'adamw'", output)

    def test_is_paged_is_reported_because_nothing_reads_it(self):
        """The flag reaches BaseTrainer and is read by no optimizer path.

        OptimizerFactory selects a paged optimizer from the type name
        (paged_adamw / paged_adamw8bit / paged_lion8bit); the boolean the UI
        offers alongside them has never done anything. Until it is either wired
        or removed, the run has to say so.
        """
        stub = _StubTrainer(optimizer_is_paged=True)
        output = _run_setup_optimizer(stub, optimizer_type="adamw",
                                      lr_scheduler_type="constant",
                                      total_steps=10)
        self.assertIn("optimizer_is_paged is not applied", output)

        paged = _StubTrainer(optimizer_is_paged=True)
        output = _run_setup_optimizer(paged, optimizer_type="paged_adamw",
                                      lr_scheduler_type="constant",
                                      total_steps=10)
        self.assertNotIn("optimizer_is_paged is not applied", output)

    def test_nothing_is_reported_when_nothing_was_requested(self):
        stub = _StubTrainer()
        output = _run_setup_optimizer(stub, optimizer_type="adamw",
                                      lr_scheduler_type="constant",
                                      total_steps=10)
        self.assertNotIn("not supported by", output)


class FusedPathTest(unittest.TestCase):
    """The Block-Swap optimizer paths honour the same options."""

    def test_fused_optimizer_groups_use_the_configured_hyperparameters(self):
        import core.training.optimizers.fused_optimizer_groups as fog

        recorded: Dict[str, Any] = {}

        def _create(**kwargs):
            recorded.update(kwargs)
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
            stub = _StubTrainer(optimizer_beta1=0.8, optimizer_beta2=0.9,
                                optimizer_epsilon=1e-7,
                                optimizer_weight_decay=0.03,
                                num_optimizer_groups=2)
            stub.optimizer = torch.optim.AdamW([stub.param], lr=1e-4)
            with contextlib.redirect_stdout(io.StringIO()):
                stub._setup_fused_optimizer_groups("adamw", 100, "constant")
        finally:
            fog.create_optimizer_groups, fog.FusedOptimizerGroups = originals

        self.assertEqual(recorded["weight_decay"], 0.03)
        self.assertEqual(recorded["betas"], (0.8, 0.9))
        self.assertEqual(recorded["eps"], 1e-7)

    def test_schedule_free_is_refused_by_the_fused_backward_path(self):
        """Block Swap + ring-buffer + Schedule-Free used to die inside backward.

        The per-parameter hooks apply the standard 8-bit update and read
        ``state['exp_avg']``, which Schedule-Free never allocates (see the
        sibling test), so the first backward raised KeyError. Refused up front
        instead, with a message naming what to change.
        """
        for optimizer_type in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
            stub = _StubTrainer(optimizer_schedule_free=True, blocks_to_swap=8)
            with self.assertRaises(ValueError) as ctx:
                with contextlib.redirect_stdout(io.StringIO()):
                    stub._setup_fused_backward_pass(optimizer_type)
            message = str(ctx.exception)
            self.assertIn("optimizer_schedule_free", message)
            self.assertIn(optimizer_type, message)

    def test_schedule_free_state_lacks_what_the_hook_reads(self):
        """Why the refusal above exists, pinned against the real allocator."""
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        original = mod.get_extension
        mod.get_extension = lambda: type("_E", (), {
            "init_quantization_maps": lambda self, *a, **k: None})()
        try:
            p = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
            with contextlib.redirect_stdout(io.StringIO()):
                opt = mod.AdamW8bit_RingBuffer([p], lr=1e-4, use_8bit=False,
                                               schedule_free=True)
                opt._init_param_state(p)
        finally:
            mod.get_extension = original

        self.assertIn("z", opt.state[p])
        self.assertNotIn("exp_avg", opt.state[p])


class VaeOptimizerOptionTest(unittest.TestCase):
    """The VAE trainer consumes two optimizer keys; the rest are refused.

    ``VaeTrainer.build_optimizer`` passes optimizer type / params / lr /
    weight_decay to OptimizerFactory and nothing else, and the VAE API surface
    (``VAE_TRAINING_DEFAULTS``) exposes exactly those two. A hand-written or
    hand-merged YAML can still carry the diffusion generators' other
    ``optimizer_*`` keys, and accepting one would mean a run whose own config
    claims cautious masking / Schedule-Free / stochastic rounding while doing
    none of it.
    """

    BASE_TRAIN = {"steps": 10, "lr": 1e-5, "optimizer": "adamw",
                  "optimizer_weight_decay": 0.001}

    def _resolve(self, train_extra: Dict[str, Any]):
        from core.training.vae.vae_config import resolve_vae_training_config

        train = dict(self.BASE_TRAIN)
        train.update(train_extra)
        return resolve_vae_training_config(
            {"train": train, "vae": {}}, base_model_path="vae.safetensors")

    def test_the_two_supported_keys_are_honoured(self):
        cfg = self._resolve({"optimizer": "adamw8bit",
                             "optimizer_weight_decay": 0.02})
        self.assertEqual(cfg["optimizer"], "adamw8bit")
        self.assertEqual(cfg["optimizer_weight_decay"], 0.02)

    def test_every_other_optimizer_key_is_refused_by_name(self):
        from core.training.vae.vae_config import VaeConfigError

        for key, value in (("optimizer_cautious", True),
                           ("optimizer_schedule_free", True),
                           ("optimizer_schedule_free_r", 0.5),
                           ("optimizer_schedule_free_weight_lr_power", 3.0),
                           ("optimizer_use_radam", True),
                           ("optimizer_warmup_steps", 100),
                           ("optimizer_stochastic_rounding", True),
                           ("optimizer_is_paged", True),
                           ("optimizer_beta1", 0.85),
                           ("optimizer_beta2", 0.95),
                           ("optimizer_epsilon", 1e-6)):
            with self.subTest(key=key):
                with self.assertRaises(VaeConfigError) as ctx:
                    self._resolve({key: value})
                self.assertIn(key, str(ctx.exception))

    def test_the_warmup_refusal_points_at_the_key_that_works(self):
        """Warmup IS available here, under the VAE trainer's own key.

        ``build_optimizer`` passes ``lr_warmup_steps`` to ``get_scheduler``, so
        a user who wrote the diffusion spelling should be told what to write
        instead of only being told to delete it.
        """
        from core.training.vae.vae_config import VaeConfigError

        with self.assertRaises(VaeConfigError) as ctx:
            self._resolve({"optimizer_warmup_steps": 100})
        self.assertIn("lr_warmup_steps", str(ctx.exception))

        with self.assertRaises(VaeConfigError) as ctx:
            self._resolve({"optimizer_cautious": True})
        self.assertNotIn("lr_warmup_steps", str(ctx.exception))

    def test_the_shipped_generator_still_resolves(self):
        """The gate must not refuse anything generate_vae_config writes."""
        from api.param_defaults import VAE_TRAINING_DEFAULTS
        from core.training.training_config import TrainingConfigGenerator
        from core.training.vae.vae_config import resolve_vae_training_config

        text = TrainingConfigGenerator.generate_vae_config(
            dict(VAE_TRAINING_DEFAULTS),
            run_name="vae_gate", base_model_path="vae.safetensors",
            output_dir="out", dataset_path="data",
        )
        process = yaml.safe_load(text)["config"]["process"][0]
        cfg = resolve_vae_training_config(process,
                                          base_model_path="vae.safetensors")
        self.assertEqual(cfg["optimizer"],
                         VAE_TRAINING_DEFAULTS["optimizer"])


class LensConfigKeyTest(unittest.TestCase):
    """Lens knobs have to be in the YAML the adapter reads them from."""

    KEYS = ("lens_lora_scope", "lens_img_lr_factor", "lens_txt_lr_factor")

    def _train_section(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.training.training_config import TrainingConfigGenerator

        text = TrainingConfigGenerator.generate_lora_config(
            {"total_steps": 10, **params},
            run_name="lens_keys", base_model_path="lens.safetensors",
            output_dir="out", dataset_path="data",
        )
        return yaml.safe_load(text)["config"]["process"][0]["train"]

    def test_the_user_value_survives_into_the_config(self):
        section = self._train_section({
            "lens_lora_scope": "img_attn",
            "lens_img_lr_factor": 0.25,
            "lens_txt_lr_factor": 4.0,
        })
        self.assertEqual(section["lens_lora_scope"], "img_attn")
        self.assertEqual(section["lens_img_lr_factor"], 0.25)
        self.assertEqual(section["lens_txt_lr_factor"], 4.0)

    def test_the_defaults_match_param_defaults(self):
        from api.param_defaults import TRAINING_DEFAULTS

        section = self._train_section({})
        for key in self.KEYS:
            self.assertEqual(section[key], TRAINING_DEFAULTS[key], key)


def _lens_transformer() -> torch.nn.Module:
    """The smallest module tree ``iter_lens_lora_targets`` recognises.

    One block: img_attn -> img_qkv + to_out.0 (2), txt_attn -> txt_qkv +
    to_add_out (2), img_mlp -> w1/w2/w3 (3), txt_mlp -> w1/w2/w3 (3),
    mod -> img_mod.1 + txt_mod.1 (2).
    """
    def _linear():
        return torch.nn.Linear(4, 4)

    attn = torch.nn.Module()
    attn.img_qkv = _linear()
    attn.txt_qkv = _linear()
    attn.to_out = torch.nn.ModuleList([_linear()])
    attn.to_add_out = _linear()

    def _mlp():
        mlp = torch.nn.Module()
        mlp.w1, mlp.w2, mlp.w3 = _linear(), _linear(), _linear()
        return mlp

    block = torch.nn.Module()
    block.attn = attn
    block.img_mlp = _mlp()
    block.txt_mlp = _mlp()
    block.img_mod = torch.nn.Sequential(torch.nn.SiLU(), _linear())
    block.txt_mod = torch.nn.Sequential(torch.nn.SiLU(), _linear())

    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([block])
    return transformer


class LensScopeEffectTest(unittest.TestCase):
    """The scope has to be able to NARROW, not only widen.

    Reaching ``self.config`` was half the fix. The consumer in
    ``lora_trainer._create_adapter`` started from the all-true ``DEFAULT_SCOPE``
    and only ever assigned ``True``, so "img_attn,txt_attn" -- what the panel
    emits when img_mlp and txt_mlp are unticked -- still trained all four
    groups. A YAML-only test cannot see that, so this one counts the LoRA
    modules the adapter actually wraps.
    """

    # img_qkv + to_out.0 + txt_qkv + to_add_out + 3 + 3, mod off by default.
    DEFAULT_TARGETS = 10

    def _wrapped(self, scope_csv: str) -> int:
        from core.training.adapters.lens_adapter import LensLoRAAdapter
        from core.training.lora_trainer import LoRATrainer

        class _Stub:
            is_zimage = False
            is_flux2 = False
            is_lens = True
            log_prefix = "[LensScopeStub]"
            lora_rank = 4
            lora_alpha = 4
            lora_dtype = torch.float32

        stub = _Stub()
        # The YAML value, exactly where train_config lands it.
        stub.config = {"lens_lora_scope": scope_csv} if scope_csv else {}
        stub.transformer = _lens_transformer()

        with contextlib.redirect_stdout(io.StringIO()):
            LoRATrainer._create_adapter(stub)
            self.assertIsInstance(stub.adapter, LensLoRAAdapter)
            return stub.adapter.apply_lora_to_unet({})

    def test_a_narrowed_scope_removes_targets(self):
        narrowed = self._wrapped("img_attn,txt_attn")
        self.assertEqual(narrowed, 4)
        self.assertLess(
            narrowed, self.DEFAULT_TARGETS,
            "unticking img_mlp/txt_mlp in the panel must remove those targets",
        )

    def test_the_default_scope_is_unchanged(self):
        self.assertEqual(self._wrapped(""), self.DEFAULT_TARGETS)
        self.assertEqual(self._wrapped("img_attn,txt_attn,img_mlp,txt_mlp"),
                         self.DEFAULT_TARGETS)

    def test_the_scope_can_still_widen(self):
        self.assertEqual(
            self._wrapped("img_attn,txt_attn,img_mlp,txt_mlp,mod"),
            self.DEFAULT_TARGETS + 2,
        )

    def test_an_unrecognised_scope_trains_the_default_rather_than_nothing(self):
        """A scope naming nothing valid must not produce a LoRA with 0 targets."""
        self.assertEqual(self._wrapped("nonsense,also_nonsense"),
                         self.DEFAULT_TARGETS)

    def test_lens_matches_the_sibling_arch_contract(self):
        from core.models.ideogram4.ideogram4_lora import (
            parse_scope_csv as ideogram4_parse,
        )
        from core.models.lens.lens_lora import (
            DEFAULT_SCOPE, parse_scope_csv as lens_parse,
        )

        # Both build from all-false, both fall back on empty/all-false input.
        self.assertEqual(lens_parse("img_attn"),
                         {k: (k == "img_attn") for k in DEFAULT_SCOPE})
        self.assertEqual(lens_parse(""), DEFAULT_SCOPE)
        self.assertEqual(lens_parse("nothing_here"), DEFAULT_SCOPE)
        self.assertEqual(ideogram4_parse(""),
                         ideogram4_parse("nothing_here"))


if __name__ == "__main__":
    unittest.main()
