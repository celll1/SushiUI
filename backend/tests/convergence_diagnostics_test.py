"""Tests for convergence diagnostics instrumentation (Phase 1).

Validates statistical measures for symptoms A and B, registration in EXTRA_METRIC_DEFS,
and parameter default wiring without requiring a real GPU.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Importing trainer/routes must not take the GPU the owner's run holds.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from api.param_defaults import TRAINING_DEFAULTS
from core.training.diagnostics.convergence_stats import (
    cell_periodicity_power,
    channel_mean_std_gap,
    low_frequency_power_ratio,
    pixel_stats_gap,
    trajectory_gap,
)
from core.training.metric_registry import EXTRA_METRIC_DEFS


class TestConvergenceStats:
    def test_channel_mean_std_gap_identical(self):
        x = torch.randn(2, 4, 16, 16)
        mean_err, std_err = channel_mean_std_gap(x, x)
        assert mean_err == pytest.approx(0.0, abs=1e-6)
        assert std_err == pytest.approx(0.0, abs=1e-6)

    def test_channel_mean_std_gap_bias(self):
        x = torch.zeros(1, 4, 8, 8)
        y = torch.ones(1, 4, 8, 8) * 2.5
        mean_err, std_err = channel_mean_std_gap(x, y)
        assert mean_err == pytest.approx(2.5, abs=1e-5)
        assert std_err == pytest.approx(0.0, abs=1e-5)

    def test_low_frequency_power_ratio_smooth_vs_noise(self):
        # Smooth image (low frequency dominant)
        h, w = 32, 32
        y, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        smooth = torch.sin(x * math.pi) * torch.cos(y * math.pi)
        smooth_ratio = low_frequency_power_ratio(smooth, radius_fraction=0.25)

        # High frequency noise (checkerboard)
        checker = ((torch.arange(h).unsqueeze(1) + torch.arange(w).unsqueeze(0)) % 2).float() * 2.0 - 1.0
        checker_ratio = low_frequency_power_ratio(checker, radius_fraction=0.25)

        assert 0.0 <= smooth_ratio <= 1.0
        assert 0.0 <= checker_ratio <= 1.0
        assert smooth_ratio > checker_ratio

    def test_pixel_stats_gap_identical(self):
        img = torch.rand(1, 3, 32, 32)
        stats = pixel_stats_gap(img, img)
        assert stats["lum_err"] == pytest.approx(0.0, abs=1e-6)
        assert stats["sat_err"] == pytest.approx(0.0, abs=1e-6)
        assert stats["var_ratio"] == pytest.approx(1.0, abs=1e-5)

    def test_pixel_stats_gap_luminance_shift(self):
        img1 = torch.zeros(1, 3, 16, 16)
        img2 = torch.ones(1, 3, 16, 16) * 0.5
        stats = pixel_stats_gap(img1, img2)
        assert stats["lum_err"] == pytest.approx(0.5, abs=1e-5)

    def test_cell_periodicity_power_grid_detection(self):
        # Random uniform noise: no 8px periodicity
        torch.manual_seed(42)
        noise = torch.randn(1, 64, 64)
        noise_power = cell_periodicity_power(noise, period=8)
        assert abs(noise_power) < 0.2

        # 8px grid artifact: periodic pattern every 8 pixels
        grid = torch.zeros(1, 64, 64)
        grid[:, ::8, :] += 1.0
        grid[:, :, ::8] += 1.0
        grid_power = cell_periodicity_power(grid, period=8)
        assert grid_power > 0.5

    def test_trajectory_gap(self):
        gap = trajectory_gap(0.1, 0.4)
        assert gap == pytest.approx(0.3, abs=1e-6)


class TestMetricRegistryIntegration:
    EXPECTED_METRICS = [
        "diag_latent_mean_err",
        "diag_latent_std_err",
        "diag_low_freq_power_ratio",
        "diag_pixel_luminance_err",
        "diag_cell_periodicity_power",
        "diag_trajectory_gap",
    ]

    def test_all_diag_metrics_registered(self):
        for name in self.EXPECTED_METRICS:
            assert name in EXTRA_METRIC_DEFS, f"{name} missing from EXTRA_METRIC_DEFS"
            entry = EXTRA_METRIC_DEFS[name]
            assert entry.get("sampling") == "periodic"
            assert entry.get("family") == "bounded_diagnostic"


class TestParameterDefaults:
    def test_diagnostics_defaults_exist(self):
        assert "convergence_diagnostics_enable" in TRAINING_DEFAULTS
        assert TRAINING_DEFAULTS["convergence_diagnostics_enable"] is False
        assert "convergence_diagnostics_interval" in TRAINING_DEFAULTS
        assert TRAINING_DEFAULTS["convergence_diagnostics_interval"] == 100


class TestBaseTrainerDiagnosticsHook:
    def test_run_convergence_diagnostics_executes_safely(self, tmp_path):
        from core.training.base_trainer import BaseTrainer

        # Construct minimal mock trainer
        trainer = MagicMock(spec=BaseTrainer)
        trainer.output_dir = tmp_path
        trainer.log_prefix = "[TestTrainer]"
        trainer._sample_prompts = [{"positive": "test prompt"}]
        trainer._diag_gt_img = None
        trainer._diag_gt_latent = None
        trainer._last_predicted_latent = torch.randn(1, 4, 16, 16)
        trainer.extra_metrics = {}

        logged_metrics = {}
        def fake_log_extra_metric(name, value):
            logged_metrics[name] = value
        trainer.log_extra_metric = fake_log_extra_metric

        rollout_img = Image.new("RGB", (64, 64), color=(128, 128, 128))

        # Call real unbound method on trainer instance
        BaseTrainer._run_convergence_diagnostics(
            trainer,
            current_step=100,
            rollout_sample=rollout_img,
            reference_image_path=None,
        )

        # Confirm images saved
        assert (tmp_path / "samples" / "step_000100_diag_rollout.png").exists()

        # Confirm metrics logged
        assert "diag_low_freq_power_ratio" in logged_metrics
        assert "diag_cell_periodicity_power" in logged_metrics
        assert "diag_trajectory_gap" in logged_metrics
        assert not math.isnan(logged_metrics["diag_low_freq_power_ratio"])


# ---------------------------------------------------------------------------
# The x0 producer: ops/crop_decode_loss.compute_crop_decode_loss is the ONE
# place that fills _last_predicted_latent. Nothing below assigns it by hand --
# that is the defect these tests exist to catch.
# ---------------------------------------------------------------------------

import ast
import inspect
import subprocess
from types import SimpleNamespace

from core.models.components.vae_registry import denormalize
from core.training.arch import ARCH_REGISTRY
from core.training.arch.base_arch import ArchHandler
from core.training.base_trainer import BaseTrainer, predicted_latent_is_supplied
from core.training.ops.crop_decode_loss import (
    compute_crop_decode_loss,
    crop_decode_or_x0_capture_needed,
)

OPS_DIR = BACKEND_ROOT / "core" / "training" / "ops"
SCALING_FACTOR = 0.13025

# The commit whose gate behaviour the diagnostics-off path must reproduce
# EXACTLY. Pinned by hash on purpose: "HEAD" would start comparing this change
# against itself the moment it is committed.
GATE_BASELINE_REV = "032cf97722eaae4b10ad4b6866baaa9afd065902"

GATE_MODULES = ("anima_ops", "flux2_ops", "ideogram4_ops", "krea2_ops", "lens_ops",
                "sd_sdxl_ops", "sensenova_ops", "zimage_ops")


class _FakeVAE:
    """Enough VAE for shift_scale (de)normalisation and a round trip."""

    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(scaling_factor=SCALING_FACTOR, shift_factor=None)

    def encode(self, x):
        latent = torch.nn.functional.avg_pool2d(x, 8).repeat(1, 2, 1, 1)[:, :4]
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))

    def decode(self, z):
        return SimpleNamespace(sample=z[:, :3].tanh())


class _FakeTrainer:
    """Carries only what the producer and the consumer actually read."""

    log_prefix = "[test]"
    crop_decode_loss_enable = False
    crop_decode_loss_weight = 0.0
    convergence_diagnostics_enable = True
    use_condition_images = False
    wiring = None
    device = "cpu"
    capture_predicted_latent = BaseTrainer.capture_predicted_latent

    def __init__(self, output_dir=None):
        self.vae = _FakeVAE()
        self.output_dir = output_dir
        self._last_predicted_latent = None
        self._diag_gt_img = None
        self._diag_gt_latent = None
        self._sample_prompts = []
        self.logged = {}

    def log_extra_metric(self, name, value):
        self.logged[name] = value


def _flow_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(2, 4, 16, 16, generator=g)
    noise = torch.randn(2, 4, 16, 16, generator=g)
    t = torch.tensor([0.3, 0.7])
    noisy = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * noise
    return x0, noise, t, noisy


def _run_producer(trainer, *, predicted_latent=None, model_pred=None):
    x0, noise, t, noisy = _flow_batch()
    v = noise - x0 if model_pred is None else model_pred
    out = compute_crop_decode_loss(
        trainer=trainer,
        model_pred=v,
        noisy_latents=noisy,
        timesteps=t,
        clean_latents=x0,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="eps_minus_x0",
        predicted_latent=predicted_latent,
    )
    return out, (x0, noise, t, noisy, v)


class TestPredictedLatentCapture:
    def test_producer_fills_last_predicted_latent_in_the_decoder_domain(self):
        trainer = _FakeTrainer()
        out, (x0, noise, t, noisy, v) = _run_producer(trainer)

        assert out == (None, 0.0)  # crop decode is off: capture only
        got = trainer._last_predicted_latent
        assert got is not None, "the producer did not fill _last_predicted_latent"
        assert tuple(got.shape) == (1, 4, 16, 16) and got.device.type == "cpu"

        expected = denormalize(
            (noisy - t.view(-1, 1, 1, 1) * v)[:1], trainer.vae, None)
        assert torch.allclose(got, expected, atol=1e-5)

    def test_supplied_x0_is_used_as_given(self):
        trainer = _FakeTrainer()
        supplied = torch.full((2, 4, 16, 16), 0.25)
        _run_producer(trainer, predicted_latent=supplied)
        assert torch.allclose(
            trainer._last_predicted_latent,
            denormalize(supplied[:1], trainer.vae, None), atol=1e-6)

    def test_capture_keeps_no_graph(self):
        trainer = _FakeTrainer()
        x0, noise, t, noisy = _flow_batch()
        v = (noise - x0).requires_grad_(True)
        _run_producer(trainer, model_pred=v)
        got = trainer._last_predicted_latent
        assert got.requires_grad is False and got.grad_fn is None
        assert got.device.type == "cpu" and got.dtype is torch.float32

    def test_nothing_is_held_while_diagnostics_are_off(self):
        trainer = _FakeTrainer()
        trainer.convergence_diagnostics_enable = False
        out, _ = _run_producer(trainer)
        assert out == (None, 0.0)
        assert trainer._last_predicted_latent is None

        # ...and the method itself refuses too, whoever calls it.
        BaseTrainer.capture_predicted_latent(trainer, torch.ones(2, 4, 8, 8))
        assert trainer._last_predicted_latent is None

    def test_5d_single_frame_latent_is_squeezed(self):
        trainer = _FakeTrainer()
        supplied = torch.randn(2, 4, 1, 16, 16)
        _run_producer(trainer, predicted_latent=supplied)
        assert tuple(trainer._last_predicted_latent.shape) == (1, 4, 16, 16)


class TestDiagnosticsActuallyEmit:
    """End to end, with no hand-written _last_predicted_latent anywhere."""

    def test_latent_metrics_and_single_x0_companion_appear(self, tmp_path):
        ref = tmp_path / "ref.png"
        Image.new("RGB", (128, 128), color=(90, 120, 200)).save(ref)

        trainer = _FakeTrainer(output_dir=tmp_path)
        _run_producer(trainer)
        assert trainer._last_predicted_latent is not None

        BaseTrainer._run_convergence_diagnostics(
            trainer,
            current_step=200,
            rollout_sample=Image.new("RGB", (128, 128), color=(100, 100, 100)),
            reference_image_path=str(ref),
        )

        # _diag_gt_latent is filled by the production path, not by the test.
        assert trainer._diag_gt_latent is not None
        assert "diag_latent_mean_err" in trainer.logged
        assert "diag_latent_std_err" in trainer.logged
        assert not math.isnan(trainer.logged["diag_latent_mean_err"])
        assert (tmp_path / "samples" / "step_000200_diag_single_x0.png").exists()
        assert (tmp_path / "samples" / "step_000200_diag_gt_roundtrip.png").exists()

    def test_without_the_producer_the_latent_metrics_stay_absent(self, tmp_path):
        ref = tmp_path / "ref.png"
        Image.new("RGB", (128, 128), color=(90, 120, 200)).save(ref)
        trainer = _FakeTrainer(output_dir=tmp_path)

        BaseTrainer._run_convergence_diagnostics(
            trainer, current_step=200,
            rollout_sample=Image.new("RGB", (128, 128), color=(100, 100, 100)),
            reference_image_path=str(ref))

        assert "diag_latent_mean_err" not in trainer.logged
        assert not (tmp_path / "samples" / "step_000200_diag_single_x0.png").exists()


class TestGateInvarianceAgainstBaseline:
    """With diagnostics off the widened gate must reproduce the old one bit for
    bit -- read out of the ops sources at GATE_BASELINE_REV, not restated here."""

    @staticmethod
    def _baseline_gate_exprs(module: str):
        # Bytes, then utf-8: some ops modules carry Japanese comments and the
        # default locale decode is cp932 on this repo's host.
        src = subprocess.run(
            ["git", "show", GATE_BASELINE_REV + ":backend/core/training/ops/" + module + ".py"],
            cwd=REPO_ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        return [ast.unparse(n) for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.BoolOp)
                and "crop_decode_loss_enable" in ast.unparse(n)]

    def test_every_baseline_module_still_has_a_recognisable_gate(self):
        for module in GATE_MODULES:
            assert self._baseline_gate_exprs(module), module

    def test_new_gate_equals_the_baseline_gate_when_diagnostics_are_off(self):
        # float weight only: BaseTrainer stores float(crop_decode_loss_weight),
        # and the baseline expression raises on None rather than gating on it.
        cases = [(False, 0.0), (False, 1.0), (True, 0.0), (True, 1e-9), (True, 0.5)]
        for module in GATE_MODULES:
            for expr in self._baseline_gate_exprs(module):
                for enable, weight in cases:
                    trainer = _FakeTrainer()
                    trainer.convergence_diagnostics_enable = False
                    trainer.crop_decode_loss_enable = enable
                    trainer.crop_decode_loss_weight = weight
                    old = bool(eval(expr, {"getattr": getattr}, {"trainer": trainer}))
                    assert crop_decode_or_x0_capture_needed(trainer) is old, (
                        module + ": " + expr + " enable=" + str(enable)
                        + " weight=" + str(weight))

    def test_new_gate_opens_for_a_diagnostics_only_run(self):
        trainer = _FakeTrainer()
        assert crop_decode_or_x0_capture_needed(trainer) is True
        assert trainer.crop_decode_loss_enable is False


# ---------------------------------------------------------------------------
# Which archs supply an x0 is DERIVED from ops/, never listed.
# ---------------------------------------------------------------------------

def _ops_modules_of(arch_name: str) -> set:
    cls = ARCH_REGISTRY[arch_name]
    tree = ast.parse(Path(inspect.getsourcefile(cls)).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "train_step":
                    for call in ast.walk(fn):
                        if (isinstance(call, ast.Call)
                                and isinstance(call.func, ast.Attribute)
                                and call.func.attr == "train_step"
                                and isinstance(call.func.value, ast.Name)):
                            modules.add(call.func.value.id)
    assert modules, arch_name + ": no ops train_step dispatch found"
    return modules


def _reaches_the_x0_producer(path: Path) -> bool:
    """The module calls compute_crop_decode_loss AND gates it on the widened
    predicate. Both matter: the old crop-decode-only gate would leave the
    capture unreachable on every run that did not also buy the aux loss."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called.add(func.id if isinstance(func, ast.Name)
                       else getattr(func, "attr", None))
    return {"compute_crop_decode_loss", "crop_decode_or_x0_capture_needed"} <= called


class TestSuppliesDeclaration:
    def test_declaration_matches_ops_source(self):
        for arch_name, cls in sorted(ARCH_REGISTRY.items()):
            reaches = any(_reaches_the_x0_producer(OPS_DIR / (m + ".py"))
                          for m in _ops_modules_of(arch_name))
            assert cls.supplies_predicted_latent == reaches, (
                arch_name + ": supplies_predicted_latent="
                + str(cls.supplies_predicted_latent) + " but its ops module "
                + ("does" if reaches else "does not") + " reach the x0 producer")

    def test_every_handler_declares_it_in_its_own_body(self):
        for arch_name, cls in sorted(ARCH_REGISTRY.items()):
            assert "supplies_predicted_latent" in vars(cls), (
                arch_name + " inherits the declaration instead of making it")

    def test_base_default_supplies_nothing(self):
        assert ArchHandler.supplies_predicted_latent is False

    def test_predicate_matches_the_declarations(self):
        for arch, cls in sorted(ARCH_REGISTRY.items()):
            trainer = _FakeTrainer()
            assert predicted_latent_is_supplied(trainer, arch) is cls.supplies_predicted_latent

    def test_predicate_is_false_for_controlnet_on_every_arch(self):
        for arch in sorted(ARCH_REGISTRY):
            trainer = _FakeTrainer()
            trainer.use_condition_images = True
            assert predicted_latent_is_supplied(trainer, arch) is False, arch


def test_enabled_line_names_what_this_run_cannot_measure():
    """The ENABLED line must consult the same predicate, once, outside any loop."""
    tree = ast.parse((BACKEND_ROOT / "core" / "training" / "base_trainer.py")
                     .read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and "Convergence diagnostics: ENABLED" in n.value]
    assert len(hits) == 1, str(len(hits)) + " ENABLED lines"

    node, enclosing = hits[0], []
    while node in parents:
        node = parents[node]
        enclosing.append(node)
    guard = next(n for n in enclosing if isinstance(n, ast.If))
    assert "predicted_latent_is_supplied" in ast.unparse(guard), \
        "the ENABLED line does not consult predicted_latent_is_supplied"
    assert not any(isinstance(n, (ast.For, ast.While)) for n in enclosing), \
        "printed inside a loop: it would print every iteration"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "train" for n in enclosing)
