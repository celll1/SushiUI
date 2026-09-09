"""``ArchHandler.velocity_sign`` is the single declaration of each arch's velocity.

A wrong sign is silent -- ``predict_x0`` returns a plausible tensor off by
``2*t*v`` -- so the declaration is worth nothing unless something checks it
against the code it describes. Every check here DERIVES the sign from the ops
module by AST rather than restating it, so a future edit to a target expression
or an inline x_0 recovery fails until the declaration is revisited:

  (1) every registered handler declares the attribute in its own body;
  (2) the declaration matches how ``train_step`` builds its velocity TARGET
      (``noise - latents`` vs ``latents - noise``, through get_target_unified
      where the arch uses it);
  (3) the declaration matches the INLINE x_0 recoveries that were deliberately
      left in place (``noisy -/+ sigma * v``) -- these carry the sign in the
      shape of the expression, and moving them would change bf16 arithmetic;
  (4) every ops call site passes its OWN arch's handler attribute -- not a
      literal, and not another arch's handler;
  (5) the declared string means, in ``predict_x0``, what the target says.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/velocity_sign_declaration_test.py -q
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
import unittest
from pathlib import Path
from typing import Optional, Set

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402

from core.training.arch import ARCH_REGISTRY  # noqa: E402
from core.training.ops.x0_recovery import VELOCITY_SIGNS, predict_x0  # noqa: E402

OPS_DIR = BACKEND / "core" / "training" / "ops"
BASE_TRAINER = BACKEND / "core" / "training" / "base_trainer.py"

CLEAN_NAMES = {"latents", "x0", "x0_a", "images", "x0_tokens"}
NOISE_NAMES = {"noise", "eps", "eps_v", "eps_a"}
TARGET_NAMES = {"target", "v_target", "target_v", "target_a"}
X0_NAME = re.compile(r"(^|_)pred_x0|predicted_latent")


def _names(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _handler_names(node: ast.AST) -> Set[str]:
    """Class names an attribute is read off; empty for a shape not recognised here."""
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.IfExp):  # sd_sdxl_ops picks SD15 or SDXL at runtime
        return _handler_names(node.body) | _handler_names(node.orelse)
    return set()


def _sign_of_difference(node: ast.BinOp) -> Optional[str]:
    """``<noise...> - <clean...>`` => eps_minus_x0, and the mirror; else None."""
    left, right = _names(node.left), _names(node.right)
    if left & NOISE_NAMES and right & CLEAN_NAMES and not (left & CLEAN_NAMES) \
            and not (right & NOISE_NAMES):
        return "eps_minus_x0"
    if left & CLEAN_NAMES and right & NOISE_NAMES and not (left & NOISE_NAMES) \
            and not (right & CLEAN_NAMES):
        return "x0_minus_eps"
    return None


def _unique(signs, where: str) -> Optional[str]:
    signs = {s for s in signs if s is not None}
    assert len(signs) <= 1, f"{where}: conflicting signs {signs}"
    return signs.pop() if signs else None


def _differences(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Sub):
            yield _sign_of_difference(sub)


def _get_target_unified_sign() -> Optional[str]:
    tree = ast.parse(BASE_TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_target_unified":
            return _unique(_differences(node), "get_target_unified")
    raise AssertionError("get_target_unified not found in base_trainer.py")


def target_sign(ops_path: Path) -> Optional[str]:
    """The sign of the velocity TARGET this ops module regresses onto."""
    tree = ast.parse(ops_path.read_text(encoding="utf-8"))
    signs, indirect = [], False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id in TARGET_NAMES for t in node.targets):
            continue
        signs.extend(_differences(node.value))
        indirect |= any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "get_target_unified"
            for c in ast.walk(node.value)
        )
    direct = _unique(signs, ops_path.name)
    if direct is not None:
        return direct
    return _get_target_unified_sign() if indirect else None


def inline_x0_sign(ops_path: Path) -> Optional[str]:
    """The sign carried by this module's inline ``noisy -/+ sigma * v`` recoveries."""
    tree = ast.parse(ops_path.read_text(encoding="utf-8"))
    signs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and X0_NAME.search(t.id) for t in node.targets):
            continue
        value = node.value
        if not (isinstance(value, ast.BinOp)
                and isinstance(value.op, (ast.Add, ast.Sub))
                and isinstance(value.right, ast.BinOp)
                and isinstance(value.right.op, ast.Mult)):
            continue
        signs.append("x0_minus_eps" if isinstance(value.op, ast.Add) else "eps_minus_x0")
    return _unique(signs, ops_path.name)


def ops_module_path(handler_cls) -> Path:
    """The ops module the handler's train_step delegates to, read from its source."""
    src = inspect.getsource(handler_cls.train_step)
    found = re.findall(r"from core\.training\.ops import (\w+)", src)
    assert len(set(found)) == 1, f"{handler_cls.__name__}: ops imports {found}"
    return OPS_DIR / f"{found[0]}.py"


def handlers_by_ops_module() -> dict:
    """ops module file name -> the handler class names whose train_step uses it."""
    owners: dict = {}
    for cls in ARCH_REGISTRY.values():
        owners.setdefault(ops_module_path(cls).name, set()).add(cls.__name__)
    return owners


class DeclarationTest(unittest.TestCase):
    def test_every_registered_arch_declares_a_sign(self):
        for name, cls in ARCH_REGISTRY.items():
            with self.subTest(arch=name):
                self.assertIn(
                    "velocity_sign", cls.__dict__,
                    f"{cls.__name__} inherits ArchHandler.velocity_sign instead of "
                    f"declaring one; a velocity arch would silently get None and a "
                    f"sample-prediction arch should say None on purpose",
                )
                self.assertIn(cls.velocity_sign, (None,) + tuple(VELOCITY_SIGNS))

    def test_declaration_matches_the_target_construction(self):
        for name, cls in ARCH_REGISTRY.items():
            with self.subTest(arch=name):
                self.assertEqual(target_sign(ops_module_path(cls)), cls.velocity_sign)

    def test_declaration_matches_the_inline_x0_recovery(self):
        """Only the modules that still recover x_0 inline; the rest derive None."""
        seen = set()
        for name, cls in ARCH_REGISTRY.items():
            derived = inline_x0_sign(ops_module_path(cls))
            if derived is None:
                continue
            seen.add(name)
            with self.subTest(arch=name):
                self.assertEqual(derived, cls.velocity_sign)
        self.assertEqual(
            seen,
            {"anima", "flux2", "ideogram4", "krea2", "lens", "ltx2",
             "minimax_h3", "acestep"},
            "the set of archs recovering x_0 inline changed; a new inline copy is a "
            "new place the sign can be dropped silently",
        )

    def test_no_ops_call_site_spells_a_sign_literal(self):
        """And each call site reads the handler of the arch THAT module trains."""
        owners = handlers_by_ops_module()
        for path in sorted(OPS_DIR.glob("*.py")):
            if path.name == "x0_recovery.py":
                continue
            allowed = owners.get(path.name)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg != "velocity_sign":
                        continue
                    with self.subTest(file=path.name, line=node.lineno):
                        self.assertNotIsInstance(
                            kw.value, ast.Constant,
                            "pass the arch handler's velocity_sign, not a literal",
                        )
                        if allowed is None:
                            continue  # shared helper: forwards its own parameter
                        self.assertTrue(
                            isinstance(kw.value, ast.Attribute)
                            and kw.value.attr == "velocity_sign",
                            f"{path.name}:{node.lineno}: pass <ArchHandler>.velocity_sign",
                        )
                        read = _handler_names(kw.value.value)
                        self.assertTrue(
                            read and read <= allowed,
                            f"{path.name}:{node.lineno} reads {sorted(read)}.velocity_sign, "
                            f"but this module trains {sorted(allowed)}; another arch's sign "
                            f"is as silent as a literal one -- x_0 off by 2*t*v",
                        )

    def test_declared_sign_round_trips_through_predict_x0(self):
        g = torch.Generator().manual_seed(3)
        x0 = torch.randn(2, 4, 8, 8, generator=g)
        eps = torch.randn(2, 4, 8, 8, generator=g)
        t = torch.tensor([0.2, 0.8]).view(-1, 1, 1, 1)
        noisy = (1.0 - t) * x0 + t * eps
        for name, cls in ARCH_REGISTRY.items():
            sign = cls.velocity_sign
            if sign is None:
                continue
            with self.subTest(arch=name):
                v = (x0 - eps) if sign == "x0_minus_eps" else (eps - x0)
                got = predict_x0(
                    noise_process="flow", prediction_target="velocity",
                    noisy_latents=noisy, model_pred=v, timesteps=t.view(-1),
                    velocity_sign=sign,
                )
                self.assertTrue(torch.allclose(got, x0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
