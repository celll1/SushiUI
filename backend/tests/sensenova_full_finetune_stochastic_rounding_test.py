"""Phase U-2-3: this route runs with stochastic rounding, and a dropout guard.

Run with:
    venv/Scripts/python.exe -m pytest \
        backend/tests/sensenova_full_finetune_stochastic_rounding_test.py -v

THE DEFECT THIS RECORDS
-----------------------
U-2-2's contract refuses ``optimizer: adamw`` for SenseNova full FT because
``torch.optim.AdamW`` has no per-parameter seam and its bf16 updates are rounded
to nearest. The configuration the contract PERMITS -- Adafactor at the shipped
``optimizer_stochastic_rounding`` default of False -- reproduces the same defect
through a different door: ``_attach_stochastic_rounding`` returns immediately
when the flag is unset.

``NegativeControlTest`` drives the exact seam this route uses
(``adafactor_fused.step_param``, which the post-accumulate-grad hooks call) and
records the number. Measured here, on CPU, with no model:

    bf16, N=65536 ~ N(0, 0.02), constant gradient, lr 1e-5, Adafactor
      SR off, 20 steps    84.5% of elements never move; 18.3% of the drift
      SR off, 400 steps   84.5% never move (moved@1 == moved@400: frozen)
      SR on,  20 steps    11.6% never move; 100.1% of the drift
      SR on,  400 steps    0.0% never move; 100.0% of the drift

84.5% is the same number SENSENOVA_TRAINING_DESIGN.md 6.3 records from an
independent measurement over all optimizers.

WHY IT IS FORCED RATHER THAN DEFAULTED
--------------------------------------
The flag cannot carry "unset": ``routes.py`` declares it ``bool`` with a False
default and ``training_config`` writes the YAML key only when it is true, so an
omitted key and an explicit false are the same value by the time a trainer sees
it (``TransportCannotExpressUnsetTest`` proves this from the source rather than
asserting it). Refusing on False would refuse every request; honouring it would
run the route at 84.5% frozen. So it is a per-architecture route requirement in
``param_defaults`` -- applied, printed, and then VERIFIED at the seam.
"""

from __future__ import annotations

import ast
import io
import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.param_defaults import (  # noqa: E402
    FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH,
    TRAINING_DEFAULTS,
    full_finetune_forces_stochastic_rounding,
)
from core.training.adapters import SenseNovaFullParameterAdapter  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.optimizers.stochastic_rounding import (  # noqa: E402
    WRAPPED_ATTR,
)
from core.training.ops import sensenova_ops  # noqa: E402
from core.training.ops.sensenova_ops import (  # noqa: E402
    assert_full_finetune_dropout_free,
    assert_full_finetune_stochastic_rounding_attached,
    assert_understanding_training_supported,
    enforce_full_finetune_stochastic_rounding,
)

from sensenova_int8_materialize_test import _Decoder  # noqa: E402

LR = 1e-5          # the full fine-tune learning rate 6.3's closed form uses
N = 1 << 16
SEED = 20260825
SIGMA = 0.02       # DiT weights ~ N(0, 0.02)


def _weights() -> torch.Tensor:
    torch.manual_seed(SEED)
    return (torch.randn(N) * SIGMA).bfloat16()


def _adafactor(p: torch.nn.Parameter):
    """The optimizer SENSENOVA_FULL_FINETUNE_OPTIMIZERS admits, as configured."""
    from transformers.optimization import Adafactor

    return Adafactor(
        [p], lr=LR, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8,
        beta1=None, weight_decay=0.0, scale_parameter=False, relative_step=False,
        warmup_init=False,
    )


def _census(stochastic_rounding: bool, steps: int) -> Dict[str, float]:
    """Drive ``step_param`` for ``steps`` and report what fraction never moved."""
    from core.training.optimizers.adafactor_fused import patch_adafactor_fused
    from core.training.optimizers.stochastic_rounding import (
        attach_stochastic_rounding,
    )

    w0 = _weights()
    p = torch.nn.Parameter(w0.clone())
    optimizer = _adafactor(p)
    patch_adafactor_fused(optimizer)
    if stochastic_rounding:
        assert "step_param" in attach_stochastic_rounding(optimizer)
    group = optimizer.param_groups[0]

    ever_moved = torch.zeros(N, dtype=torch.bool)
    moved_after_first = None
    previous = p.detach().clone()
    for step in range(steps):
        p.grad = torch.full((N,), -1.0, dtype=torch.bfloat16)
        optimizer.step_param(p, group)
        ever_moved |= p.detach().ne(previous)
        previous = p.detach().clone()
        if step == 0:
            moved_after_first = ever_moved.float().mean().item()

    drift = (p.detach().float() - w0.float()).mean().item()
    return {
        "never_moved": 1.0 - ever_moved.float().mean().item(),
        "moved_at_1": moved_after_first,
        "moved_at_n": ever_moved.float().mean().item(),
        "drift_fraction": drift / (steps * LR),
    }


# ---------------------------------------------------------------------------
# The negative control: what the shipped configuration does
# ---------------------------------------------------------------------------

class NegativeControlTest(unittest.TestCase):
    """Adafactor at the shipped default, on this route's own seam.

    This is the permitted configuration, not a refused one: the contract accepts
    ``optimizer=adafactor``, and before this change nothing turned stochastic
    rounding on for it.
    """

    def test_the_shipped_default_freezes_84_percent_of_the_tensor(self):
        census = _census(stochastic_rounding=False, steps=20)
        # The design's independently measured figure, reproduced on this seam.
        self.assertAlmostEqual(census["never_moved"], 0.845, places=2)
        # 6.3's "18% of what the learning rate asks for".
        self.assertAlmostEqual(census["drift_fraction"], 0.183, places=2)

    def test_those_elements_are_frozen_rather_than_slow(self):
        """moved@1 == moved@N: no element joins the moving set later."""
        for steps in (20, 400):
            with self.subTest(steps=steps):
                census = _census(stochastic_rounding=False, steps=steps)
                self.assertEqual(census["moved_at_1"], census["moved_at_n"])
                self.assertAlmostEqual(census["never_moved"], 0.845, places=2)

    def test_the_route_with_this_change_moves_them_instead(self):
        twenty = _census(stochastic_rounding=True, steps=20)
        self.assertAlmostEqual(twenty["never_moved"], 0.116, places=2)
        self.assertAlmostEqual(twenty["drift_fraction"], 1.0, places=2)

        four_hundred = _census(stochastic_rounding=True, steps=400)
        self.assertEqual(four_hundred["never_moved"], 0.0)
        self.assertAlmostEqual(four_hundred["drift_fraction"], 1.0, places=2)


# ---------------------------------------------------------------------------
# The trainer stub: real BaseTrainer.setup_optimizer, no model
# ---------------------------------------------------------------------------

class _StubTrainer:
    """The smallest object ``BaseTrainer.setup_optimizer`` runs against.

    Same shape as the stub in ``bf16_stochastic_rounding_default_optimizer_test``:
    real methods, no model, no dataset, no CUDA.
    """

    setup_optimizer = BaseTrainer.setup_optimizer
    _resolved_optimizer_hyperparameters = BaseTrainer._resolved_optimizer_hyperparameters
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass
    _setup_fused_optimizer_groups = BaseTrainer._setup_fused_optimizer_groups
    _attach_stochastic_rounding = BaseTrainer._attach_stochastic_rounding
    _fused_backward_target_module = BaseTrainer._fused_backward_target_module
    _RINGBUFFER_ONLY_OPTIONS = BaseTrainer._RINGBUFFER_ONLY_OPTIONS
    _NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS = BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS
    _BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS = BaseTrainer._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS

    def __init__(self, **overrides: Any):
        self.log_prefix = "[StubTrainer]"
        self.learning_rate = LR
        self.weight_dtype = torch.bfloat16
        self.training_dtype = torch.bfloat16
        self.use_grad_scaler = False
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.use_ema = False
        self.config: Dict[str, Any] = {"optimizer": "adafactor"}
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
        self.max_grad_norm = 0.0
        # The route: SenseNova, full parameter.
        self.is_sensenova = True
        self.trains_base_weights = True
        self.train_unet = True
        self.train_text_encoder = False
        self.transformer = nn.Linear(4, 4).to(torch.bfloat16)
        for key, value in overrides.items():
            setattr(self, key, value)
        self.param = torch.nn.Parameter(_weights())

    def setup_trainable_parameters(self):
        return [{"params": [self.param], "lr": self.learning_rate}]

    def _setup_ema(self):
        pass


def _run_setup(**overrides):
    """setup_optimizer with stdout captured; returns (trainer, printed)."""
    optimizer_type = overrides.pop("optimizer_type", "adafactor")
    trainer = _StubTrainer(**overrides)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        trainer.setup_optimizer(optimizer_type=optimizer_type, total_steps=10)
    return trainer, buffer.getvalue()


class RouteEnforcementTest(unittest.TestCase):
    """The flag ends up on, and the SEAM ends up wrapped."""

    def test_setup_optimizer_turns_stochastic_rounding_on_for_this_route(self):
        trainer, printed = _run_setup()
        self.assertTrue(trainer.optimizer_stochastic_rounding)
        self.assertIn("optimizer_stochastic_rounding", printed)
        self.assertIn("84.5%", printed)

    def test_the_fused_step_param_is_the_thing_that_gets_wrapped(self):
        """Assert on the mechanism: the seam the hooks call, not the flag.

        ``_setup_fused_backward_pass`` patches ``step_param`` onto Adafactor and
        registers post-accumulate-grad hooks that call
        ``self.optimizer.step_param``; the interposer replaces that attribute
        afterwards, and the hooks resolve it at call time.
        """
        trainer, _ = _run_setup()
        self.assertTrue(trainer.use_fused_backward)
        step_param = trainer.optimizer.step_param
        self.assertTrue(callable(step_param))
        self.assertTrue(
            getattr(step_param, WRAPPED_ATTR, False),
            "the route's per-parameter update seam is not carrying stochastic rounding",
        )

    def test_a_real_backward_reaches_the_wrapped_seam(self):
        """End to end through autograd: the hook must run the wrapper.

        A hook that had captured the unwrapped ``step_param`` at registration
        time would pass every assertion above and still write round-to-nearest.
        """
        trainer = _StubTrainer()
        trainer.param = torch.nn.Parameter(torch.full((16,), 1.0, dtype=torch.bfloat16))
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.setup_optimizer(optimizer_type="adafactor", total_steps=10)

        seen = []
        wrapped = trainer.optimizer.step_param

        def spy(p, group):
            seen.append((p.dtype, getattr(wrapped, WRAPPED_ATTR, False)))
            return wrapped(p, group)

        trainer.optimizer.step_param = spy
        (trainer.param.float().sum() * -1.0).backward()
        self.assertEqual(seen, [(torch.bfloat16, True)],
                         "the post-accumulate-grad hook did not reach the wrapped seam")

    def test_the_verification_refuses_an_unwrapped_seam(self):
        """If the interposition ever stops applying, the run stops."""
        trainer, _ = _run_setup()
        from core.training.optimizers.adafactor_fused import adafactor_step_param

        # Exactly what a regression looks like: step_param present, unwrapped.
        trainer.optimizer.step_param = adafactor_step_param.__get__(trainer.optimizer)
        with self.assertRaises(RuntimeError) as caught:
            assert_full_finetune_stochastic_rounding_attached(trainer, "adafactor")
        self.assertIn("step_param", str(caught.exception))

    def test_with_the_change_disabled_the_route_ships_the_defect(self):
        """The negative control, in place: table entry False -> nothing happens."""
        table = FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH
        original = table["sensenova"]
        table["sensenova"] = False
        try:
            trainer, printed = _run_setup()
            self.assertFalse(trainer.optimizer_stochastic_rounding)
            self.assertFalse(getattr(trainer.optimizer.step_param, WRAPPED_ATTR, False))
            self.assertNotIn("84.5%", printed)
        finally:
            table["sensenova"] = original


class ExplicitUserSettingTest(unittest.TestCase):
    """What an explicitly-set value does, and why it cannot be honoured here."""

    def test_an_explicit_true_is_left_alone_and_not_announced(self):
        trainer = _StubTrainer(optimizer_stochastic_rounding=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            changed = enforce_full_finetune_stochastic_rounding(trainer)
        self.assertFalse(changed)
        self.assertEqual(buffer.getvalue(), "")
        self.assertTrue(trainer.optimizer_stochastic_rounding)

    def test_a_false_is_overridden_and_the_override_is_visible(self):
        """Not silent: it says what changed, why, and that it is not optional."""
        trainer = _StubTrainer(optimizer_stochastic_rounding=False)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            changed = enforce_full_finetune_stochastic_rounding(trainer)
        printed = buffer.getvalue()
        self.assertTrue(changed)
        self.assertTrue(trainer.optimizer_stochastic_rounding)
        self.assertIn("optimizer_stochastic_rounding", printed)
        self.assertIn("turned on", printed)
        self.assertIn("84.5%", printed)
        self.assertIn("not optional", printed)


class TransportCannotExpressUnsetTest(unittest.TestCase):
    """Why this is forced rather than resolved per-arch like train_text_encoder.

    ``resolve_full_finetune_train_text_encoder`` works because that value can
    arrive as None. This one cannot, and these assertions read the source rather
    than trusting the claim.
    """

    def test_the_request_model_declares_a_plain_bool(self):
        source = (Path(_BACKEND) / "api" / "routes.py").read_text(encoding="utf-8")
        found = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "optimizer_stochastic_rounding"
        ]
        self.assertTrue(found)
        for node in found:
            # `bool`, not `Optional[bool]`: absence is not representable.
            self.assertIsInstance(node.annotation, ast.Name)
            self.assertEqual(node.annotation.id, "bool")

    def test_the_yaml_writer_emits_the_key_only_when_it_is_true(self):
        source = (Path(_BACKEND) / "core" / "training" / "training_config.py").read_text(
            encoding="utf-8")
        self.assertIn('if p.get("optimizer_stochastic_rounding"):', source)

    def test_the_shipped_global_default_is_still_false(self):
        self.assertIs(TRAINING_DEFAULTS["optimizer_stochastic_rounding"], False)


class OtherArchitecturesUnchangedTest(unittest.TestCase):
    """Proven by driving them, and by reading the table -- not asserted."""

    def test_only_sensenova_is_in_the_table(self):
        from core.training.arch import ARCH_REGISTRY

        forced = {
            arch for arch in ARCH_REGISTRY
            if full_finetune_forces_stochastic_rounding(arch)
        }
        self.assertEqual(forced, {"sensenova"})
        self.assertIs(FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH["_default"], False)
        self.assertEqual(set(FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH),
                         {"_default", "sensenova"})

    def test_a_non_sensenova_full_finetune_keeps_round_to_nearest(self):
        """The same optimizer, the same flag, a different architecture."""
        trainer, printed = _run_setup(is_sensenova=False)
        self.assertFalse(trainer.optimizer_stochastic_rounding)
        step_param = getattr(trainer.optimizer, "step_param", None)
        self.assertFalse(getattr(step_param, WRAPPED_ATTR, False))
        self.assertNotIn("84.5%", printed)

    def test_a_non_sensenova_run_still_honours_an_explicit_true(self):
        trainer, _ = _run_setup(is_sensenova=False, optimizer_stochastic_rounding=True)
        self.assertTrue(getattr(trainer.optimizer.step_param, WRAPPED_ATTR, False))


class SenseNovaLoraUnchangedTest(unittest.TestCase):
    """LoRA on this architecture honours the flag, in both directions."""

    def _lora_trainer(self, **overrides):
        return _StubTrainer(trains_base_weights=False,
                            config={"training_method": "lora", "optimizer": "adamw"},
                            **overrides)

    def test_lora_does_not_get_stochastic_rounding_forced_on(self):
        trainer = self._lora_trainer()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            trainer.setup_optimizer(optimizer_type="adafactor", total_steps=10)
        self.assertFalse(trainer.optimizer_stochastic_rounding)
        self.assertNotIn("84.5%", buffer.getvalue())

    def test_lora_still_accepts_the_optimizers_full_ft_refuses(self):
        """The full-FT contract must not have leaked onto the LoRA route."""
        trainer = self._lora_trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.setup_optimizer(optimizer_type="adamw", total_steps=10)
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)

    def test_lora_honours_an_explicit_true(self):
        trainer = self._lora_trainer(optimizer_stochastic_rounding=True)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.setup_optimizer(optimizer_type="adafactor", total_steps=10)
        self.assertTrue(getattr(trainer.optimizer.step_param, WRAPPED_ATTR, False))


# ---------------------------------------------------------------------------
# The dropout guard
# ---------------------------------------------------------------------------

_BRANCH_FLAGS = {
    "gen": {"train_unet": True, "train_text_encoder": False},
    "und": {"train_unet": False, "train_text_encoder": True},
    "both": {"train_unet": True, "train_text_encoder": True},
}


def _decoder_with_dropout(dropout: float) -> nn.Module:
    from core.models.sensenova.loader import materialize_int8_decoder_linears

    transformer = _Decoder()
    materialize_int8_decoder_linears(transformer, branch="both")
    transformer.language_model.model.config = SimpleNamespace(
        attention_dropout=dropout, num_hidden_layers=42
    )
    return transformer


def _full_ft_trainer(branch: str, transformer: nn.Module):
    return SimpleNamespace(
        transformer=transformer,
        unet=None,
        text_encoder=None,
        trains_base_weights=True,
        is_sensenova=True,
        weight_dtype=torch.bfloat16,
        training_dtype=torch.bfloat16,
        use_grad_scaler=False,
        unet_lr=1e-6,
        text_encoder_1_lr=None,
        text_encoder_lr=None,
        config={"optimizer": "adafactor"},
        **_BRANCH_FLAGS[branch],
    )


class DropoutGuardTest(unittest.TestCase):
    """Fail-closed on a non-zero ``attention_dropout``, on EVERY branch.

    The vendor attention keeps ``dropout=0.0 if not self.training else
    self.attention_dropout`` and the full-FT adapter stamps ``train()`` on the
    whole MoT decoder. Both halves live in one ``language_model``, and the
    prompt prefix is built by the understanding half on every step even when
    only the generation half is trained, so the branch the guard used to be
    gated on is not the branch that needs it.
    """

    def test_negative_control_the_branch_gated_guard_never_runs_for_gen_only(self):
        """Records the shipped hole: gen-only reached train() unchecked.

        The understanding guard refuses this transformer when it is called -- it
        is the CALL that was branch-gated, so the default branch for this route
        was never checked at all. With the new guard removed, a dropout of 0.1
        prepares for training without any error.
        """
        transformer = _decoder_with_dropout(0.1)
        with self.assertRaises(RuntimeError):
            assert_understanding_training_supported(transformer)

        calls = []
        original_und = sensenova_ops.assert_understanding_training_supported
        original_ft = sensenova_ops.assert_full_finetune_dropout_free
        sensenova_ops.assert_understanding_training_supported = (
            lambda module: calls.append("und"))
        sensenova_ops.assert_full_finetune_dropout_free = lambda module: None
        try:
            adapter = SenseNovaFullParameterAdapter(_full_ft_trainer("gen", transformer))
            adapter.prepare_models_for_training()
        finally:
            sensenova_ops.assert_understanding_training_supported = original_und
            sensenova_ops.assert_full_finetune_dropout_free = original_ft
        self.assertEqual(calls, [], "the understanding guard is branch-gated")

    def test_gen_only_full_finetune_is_refused_with_a_non_zero_dropout(self):
        transformer = _decoder_with_dropout(0.1)
        adapter = SenseNovaFullParameterAdapter(_full_ft_trainer("gen", transformer))
        with self.assertRaises(RuntimeError) as caught:
            adapter.prepare_models_for_training()
        message = str(caught.exception)
        self.assertIn("attention_dropout=0.0", message)
        self.assertIn("full fine-tuning", message)

    def test_every_branch_is_refused(self):
        for branch in ("gen", "und", "both"):
            with self.subTest(branch=branch):
                transformer = _decoder_with_dropout(0.25)
                adapter = SenseNovaFullParameterAdapter(
                    _full_ft_trainer(branch, transformer))
                with self.assertRaises(RuntimeError):
                    adapter.prepare_models_for_training()

    def test_zero_dropout_refuses_nothing(self):
        """Upstream's default; today's configurations are unaffected."""
        for branch in ("gen", "und", "both"):
            with self.subTest(branch=branch):
                transformer = _decoder_with_dropout(0.0)
                adapter = SenseNovaFullParameterAdapter(
                    _full_ft_trainer(branch, transformer))
                adapter.prepare_models_for_training()

    def test_a_tree_without_the_vendor_decoder_stack_is_refused(self):
        with self.assertRaises(RuntimeError):
            assert_full_finetune_dropout_free(nn.Linear(4, 4))


if __name__ == "__main__":
    unittest.main()
