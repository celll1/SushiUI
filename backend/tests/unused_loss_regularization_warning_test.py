"""Coverage for BaseTrainer._warn_unused_loss_regularization_keys.

min_snr_gamma / snr_regularization_* / energy_regularization_* /
reconstruction_loss_weight can be set in the UI and training config for any
architecture, but only a subset of the per-architecture op modules ever read
them (verified against ops/sd_sdxl_ops.py, ops/flux2_ops.py, ops/zimage_ops.py,
and every other ops/*_ops.py). This warns once, in a single block, when a
configured key will have no effect -- and must never change what the loss
computes.

Which archs consume reconstruction_loss_weight is NOT a list here or in
base_trainer: each handler declares
``ArchHandler.consumes_reconstruction_loss_weight`` and the declaration test
below DERIVES the truth from the arch's ops module by AST, so an arch that
gains or loses the term fails until its declaration is revisited.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.arch import ARCH_REGISTRY
from core.training.arch.base_arch import ArchHandler
from core.training.base_trainer import BaseTrainer

BACKEND = Path(__file__).resolve().parents[1]
OPS_DIR = BACKEND / "core" / "training" / "ops"
RECON_KEY = "reconstruction_loss_weight"


def _fake_trainer(**overrides):
    base = dict(
        min_snr_gamma=0.0,
        snr_regularization_loss=None,
        energy_regularization_loss=None,
        reconstruction_loss_weight=0.0,
        use_condition_images=False,
        prediction_target="epsilon",
        log_prefix="[test]",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _warn(trainer, arch_name):
    BaseTrainer._warn_unused_loss_regularization_keys(trainer, arch_name)


# ---------------------------------------------------------------------------
# (a) SenseNova + min_snr_gamma: must warn (sensenova_ops.py never reads it).
# ---------------------------------------------------------------------------

def test_sensenova_min_snr_gamma_warns(capsys):
    trainer = _fake_trainer(min_snr_gamma=5.0)
    _warn(trainer, "sensenova")
    out = capsys.readouterr().out
    assert "min_snr_gamma=5.0" in out
    assert "sensenova" in out


def test_sensenova_snr_regularization_warns(capsys):
    trainer = _fake_trainer(snr_regularization_loss=object())
    _warn(trainer, "sensenova")
    out = capsys.readouterr().out
    assert "snr_regularization_*" in out


def test_sensenova_energy_regularization_warns(capsys):
    trainer = _fake_trainer(energy_regularization_loss=object())
    _warn(trainer, "sensenova")
    out = capsys.readouterr().out
    assert "energy_regularization_*" in out


def test_other_non_consuming_archs_warn_too(capsys):
    for arch in ("acestep", "anima", "ideogram4", "krea2", "lens", "ltx2",
                 "minimax_h3", "minit2i"):
        trainer = _fake_trainer(min_snr_gamma=5.0, snr_regularization_loss=object(),
                                 energy_regularization_loss=object())
        _warn(trainer, arch)
        out = capsys.readouterr().out
        assert "min_snr_gamma=5.0" in out, arch
        assert "snr_regularization_*" in out, arch
        assert "energy_regularization_*" in out, arch


# ---------------------------------------------------------------------------
# (b) Actually-consumed combinations must NOT warn.
# ---------------------------------------------------------------------------

def test_sd15_min_snr_gamma_epsilon_does_not_warn(capsys):
    trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="epsilon")
    _warn(trainer, "sd15")
    assert capsys.readouterr().out == ""


def test_sdxl_min_snr_gamma_epsilon_does_not_warn(capsys):
    trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="epsilon")
    _warn(trainer, "sdxl")
    assert capsys.readouterr().out == ""


def test_sd15_min_snr_gamma_velocity_warns(capsys):
    """min_snr_gamma is gated on prediction_target=='epsilon' even for sd15."""
    trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="velocity")
    _warn(trainer, "sd15")
    out = capsys.readouterr().out
    assert "prediction_target='epsilon'" in out
    assert "prediction_target='velocity'" in out


def test_sd_sdxl_flux2_zimage_regularization_does_not_warn(capsys):
    for arch in ("sd15", "sdxl", "flux2", "zimage"):
        trainer = _fake_trainer(snr_regularization_loss=object(), energy_regularization_loss=object())
        _warn(trainer, arch)
        assert capsys.readouterr().out == "", arch


def test_controlnet_min_snr_gamma_epsilon_applies_to_zimage_and_flux2(capsys):
    for arch in ("sd15", "sdxl", "zimage", "flux2"):
        trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="epsilon",
                                 use_condition_images=True)
        _warn(trainer, arch)
        assert capsys.readouterr().out == "", arch


def test_controlnet_regularization_never_applies(capsys):
    """train_step_controlnet never reads snr/energy_regularization_loss,
    regardless of architecture."""
    for arch in ("sd15", "sdxl", "zimage", "flux2"):
        trainer = _fake_trainer(snr_regularization_loss=object(),
                                 energy_regularization_loss=object(),
                                 use_condition_images=True)
        _warn(trainer, arch)
        out = capsys.readouterr().out
        assert "snr_regularization_*" in out, arch
        assert "energy_regularization_*" in out, arch
        assert "ControlNet training" in out, arch


def test_disabled_keys_never_warn(capsys):
    trainer = _fake_trainer()  # all defaults: 0.0 / None / None
    for arch in ("sensenova", "sd15", "sdxl", "flux2", "zimage", "acestep"):
        _warn(trainer, arch)
        assert capsys.readouterr().out == "", arch


# ---------------------------------------------------------------------------
# (c) The warning is print-only: it must not touch the trainer's loss state.
# ---------------------------------------------------------------------------

def test_warning_does_not_mutate_trainer_state(capsys):
    snr_module = object()
    energy_module = object()
    trainer = _fake_trainer(
        min_snr_gamma=5.0,
        snr_regularization_loss=snr_module,
        energy_regularization_loss=energy_module,
        prediction_target="velocity",
    )
    before = dict(vars(trainer))
    _warn(trainer, "sensenova")
    capsys.readouterr()
    after = dict(vars(trainer))
    assert before == after
    assert trainer.min_snr_gamma == 5.0
    assert trainer.snr_regularization_loss is snr_module
    assert trainer.energy_regularization_loss is energy_module


# ---------------------------------------------------------------------------
# (d) reconstruction_loss_weight: consumption is per-arch AND per-forward.
# ---------------------------------------------------------------------------

def _consuming_archs():
    return {name for name, cls in ARCH_REGISTRY.items()
            if cls.consumes_reconstruction_loss_weight}


def test_consuming_archs_do_not_warn(capsys):
    """The normal (non-ControlNet) forward of every consuming arch is silent."""
    consuming = _consuming_archs()
    assert consuming, "no arch declares consumes_reconstruction_loss_weight"
    for arch in sorted(consuming):
        trainer = _fake_trainer(reconstruction_loss_weight=0.5)
        _warn(trainer, arch)
        assert capsys.readouterr().out == "", arch


def test_non_consuming_archs_warn(capsys):
    for arch in sorted(set(ARCH_REGISTRY) - _consuming_archs()):
        trainer = _fake_trainer(reconstruction_loss_weight=0.5)
        _warn(trainer, arch)
        out = capsys.readouterr().out
        assert "reconstruction_loss_weight=0.5" in out, arch
        assert arch in out, arch


def test_minimax_h3_warns(capsys):
    """Pinned by name: minimax_h3 is today's only non-consuming arch."""
    trainer = _fake_trainer(reconstruction_loss_weight=0.5)
    _warn(trainer, "minimax_h3")
    out = capsys.readouterr().out
    assert "reconstruction_loss_weight=0.5" in out
    assert "minimax_h3" in out


def test_controlnet_never_consumes_it(capsys):
    """train_step_controlnet ignores the weight on every architecture, so a
    ControlNet run warns even for an arch whose own train_step consumes it."""
    for arch in sorted(ARCH_REGISTRY):
        trainer = _fake_trainer(reconstruction_loss_weight=0.5,
                                use_condition_images=True)
        _warn(trainer, arch)
        out = capsys.readouterr().out
        assert "reconstruction_loss_weight=0.5" in out, arch
        assert "train_step_controlnet" in out, arch


def test_zero_weight_never_warns(capsys):
    for use_condition_images in (False, True):
        for arch in sorted(ARCH_REGISTRY):
            trainer = _fake_trainer(reconstruction_loss_weight=0.0,
                                    use_condition_images=use_condition_images)
            _warn(trainer, arch)
            assert capsys.readouterr().out == "", arch


def test_bound_arch_handler_is_preferred_over_the_registry(capsys):
    """The run's own handler decides; the registry is only the fallback."""
    trainer = _fake_trainer(reconstruction_loss_weight=0.5,
                            arch=ARCH_REGISTRY["minimax_h3"](None))
    _warn(trainer, "sdxl")
    assert "reconstruction_loss_weight=0.5" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (e) The declaration is derived from ops/, not restated.
# ---------------------------------------------------------------------------

def _ops_modules_of(arch_name: str) -> set:
    """The ops modules this handler's ``train_step`` dispatches into."""
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
    assert modules, f"{arch_name}: no ops train_step dispatch found"
    return modules


def _reads_key(node: ast.AST, aliases: set) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == RECON_KEY:
            return True
        if isinstance(sub, ast.Constant) and sub.value == RECON_KEY:
            return True
        if isinstance(sub, ast.Name) and sub.id in aliases:
            return True
    return False


def _folds_key_into_arithmetic(path: Path) -> bool:
    """True when the weight (or a local bound to it) is an operand of an
    arithmetic expression -- i.e. it reaches the loss, rather than only being
    compared against zero to report that it is ignored (minimax_h3_ops)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _reads_key(node.value, aliases):
            aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Mult, ast.Add, ast.Sub, ast.Div)):
            if _reads_key(node.left, aliases) or _reads_key(node.right, aliases):
                return True
    return False


def test_declaration_matches_ops_source():
    for arch_name, cls in sorted(ARCH_REGISTRY.items()):
        folded = any(_folds_key_into_arithmetic(OPS_DIR / f"{module}.py")
                     for module in _ops_modules_of(arch_name))
        assert cls.consumes_reconstruction_loss_weight == folded, (
            f"{arch_name}: consumes_reconstruction_loss_weight="
            f"{cls.consumes_reconstruction_loss_weight} but its ops module "
            f"{'does' if folded else 'does not'} fold {RECON_KEY} into the loss"
        )


def test_every_handler_declares_it_in_its_own_body():
    for arch_name, cls in sorted(ARCH_REGISTRY.items()):
        assert "consumes_reconstruction_loss_weight" in vars(cls), (
            f"{arch_name} inherits the declaration instead of making it")


def test_base_default_is_not_consuming():
    """A handler that forgets to declare gets a warning, not a silent drop."""
    assert ArchHandler.consumes_reconstruction_loss_weight is False


# ---------------------------------------------------------------------------
# (f) The warning fires once per run, not per step.
# ---------------------------------------------------------------------------

def test_warning_is_called_once_outside_any_loop():
    tree = ast.parse((BACKEND / "core" / "training" / "base_trainer.py")
                     .read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    call_sites = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "_warn_unused_loss_regularization_keys"]
    assert len(call_sites) == 1, f"{len(call_sites)} call sites"

    node, enclosing = call_sites[0], []
    while node in parents:
        node = parents[node]
        enclosing.append(node)
    assert not any(isinstance(n, (ast.For, ast.While)) for n in enclosing), \
        "called inside a loop: it would print every iteration"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "train" for n in enclosing)
