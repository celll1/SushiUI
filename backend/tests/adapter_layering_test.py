"""``core.adapters`` layering gate and canonical-object identity.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_layering_test.py -v -s

WHY THIS FILE EXISTS. ``core.training.adapters`` is a subpackage of
``core.training``, whose ``__init__`` reaches ``api.param_defaults`` and
transitively ``api.routes`` -- a ``core -> api`` back-edge. Twelve generation
modules across eleven architectures imported the leaf LoRA layer class from
there, so each inherited the whole API surface and a CUDA context. They now
import ``core.adapters``; ``ShimRemovalTest`` keeps it that way. Measured in a
fresh process on this machine (repo venv, cwd ``backend/``, warm cache):

    core.adapters                          1.26 s   1020 modules  CUDA False
    core.training.adapters.sd15_adapter    9.16 s   5806 modules  CUDA True

The layering test below re-measures the first arm every run and asserts the
constraint. The second arm is the BASELINE and is opt-in behind
``SUSHI_ADAPTER_LAYERING_BASELINE=1``: it costs about nine seconds and creates a
CUDA context, which is not something an otherwise CPU-only gate should do on a
machine that may have a training run in flight.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_PROBE = r"""
import json, sys, time
_before = len(sys.modules)
_t = time.perf_counter()
import {module}
_dt = time.perf_counter() - _t
import torch
print("__PROBE__" + json.dumps({{
    "seconds": _dt,
    "modules": len(sys.modules) - _before,
    "core_training": "core.training" in sys.modules,
    "api": "api" in sys.modules,
    "api_routes": "api.routes" in sys.modules,
    "api_param_defaults": "api.param_defaults" in sys.modules,
    "cuda_initialized": torch.cuda.is_initialized(),
}}))
"""


def _probe(module: str) -> dict:
    """Import ``module`` in a FRESH interpreter and report what it dragged in.

    Fresh is the whole point: inside the running backend ``api.routes`` is
    already loaded, which is exactly what masks the back-edge.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        cwd=_BACKEND,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"probe importing {module} failed ({proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("__PROBE__"):
            return json.loads(line[len("__PROBE__"):])
    raise AssertionError(f"probe importing {module} printed no result:\n{proc.stdout[-4000:]}")


def _report(label: str, result: dict) -> None:
    print(
        f"[adapter-layering] {label}: {result['seconds']:.2f} s, "
        f"{result['modules']} modules, core.training={result['core_training']}, "
        f"api.routes={result['api_routes']}, "
        f"cuda_initialized={result['cuda_initialized']}"
    )


class AdapterPackageLayeringTest(unittest.TestCase):
    """``core.adapters`` must not reach into ``core.training`` or ``api``."""

    def test_core_adapters_imports_without_training_api_or_cuda(self):
        new = _probe("core.adapters")
        _report("core.adapters", new)

        if os.environ.get("SUSHI_ADAPTER_LAYERING_BASELINE") == "1":
            old = _probe("core.training.adapters.sd15_adapter")
            _report("core.training.adapters.sd15_adapter", old)
        else:
            print("[adapter-layering] baseline arm skipped; set "
                  "SUSHI_ADAPTER_LAYERING_BASELINE=1 to re-measure it "
                  "(it costs ~9 s and creates a CUDA context)")

        self.assertFalse(new["core_training"],
                         "core.adapters must not import core.training")
        self.assertFalse(new["api"], "core.adapters must not import api")
        self.assertFalse(new["api_routes"],
                         "core.adapters must not import api.routes")
        self.assertFalse(new["api_param_defaults"],
                         "core.adapters must not import api.param_defaults")
        self.assertFalse(new["cuda_initialized"],
                         "importing core.adapters must not initialise CUDA")


class CanonicalObjectIdentityTest(unittest.TestCase):
    """Every module-scope binding of a moved name must be THE SAME object.

    ``isinstance`` checks in the loaders and
    ``type(m).__name__ == "LoRALinearLayer"`` gates in
    ``core.models.common.int8_runtime_quantize`` /
    ``core.vram_optimization`` all break silently if some module ever rebinds a
    second class object under the same name.
    """

    BINDINGS = [
        ("LoRALinearLayer", "core.training.adapters.sd15_adapter"),
        ("LoRALinearLayer", "core.training.adapters.minimax_h3_adapter"),
        ("MiniMaxH3LoRALinearLayer", "core.training.adapters.minimax_h3_adapter"),
        ("is_lora_wrappable_linear", "core.training.adapters.zimage_adapter"),
        ("count_quantized_linears", "core.training.adapters.base_adapter"),
        ("lora_branch_dtype", "core.adapters.targets"),
    ]

    def test_every_module_scope_binding_is_the_canonical_object(self):
        import importlib

        import core.adapters as canonical

        for name, module_name in self.BINDINGS:
            with self.subTest(symbol=name, module=module_name):
                module = importlib.import_module(module_name)
                self.assertIs(getattr(module, name), getattr(canonical, name))

    def test_minimax_subclass_relationship_survives_the_move(self):
        from core.adapters import LoRALinearLayer, MiniMaxH3LoRALinearLayer

        self.assertTrue(issubclass(MiniMaxH3LoRALinearLayer, LoRALinearLayer))
        self.assertEqual(LoRALinearLayer.__name__, "LoRALinearLayer")
        self.assertEqual(MiniMaxH3LoRALinearLayer.__name__, "MiniMaxH3LoRALinearLayer")


MOVED_NAMES = frozenset({
    "LoRALinearLayer",
    "MiniMaxH3LoRALinearLayer",
    "count_quantized_linears",
    "is_lora_wrappable_linear",
    "lora_branch_dtype",
})

_LEAF_MODULES = frozenset({
    "core.training.adapters.base_adapter",
    "core.training.adapters.sd15_adapter",
    "core.training.adapters.minimax_h3_adapter",
})


def _resolved_module(node: ast.ImportFrom, path: str) -> str:
    """The absolute module an ``ImportFrom`` names, relative imports included."""
    if not node.level:
        return node.module or ""
    package = os.path.relpath(path, _BACKEND).replace("\\", "/").split("/")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


class ShimRemovalTest(unittest.TestCase):
    """No module may reach the moved names through ``core.training.adapters``.

    The re-export shims are gone; this asserts nobody grows a new path back to
    them, which is what would quietly reintroduce the back-edge measured above
    for whichever generation module did it.
    """

    def test_no_module_imports_a_moved_name_from_the_training_package(self):
        offenders = []
        for path in sorted(pathlib.Path(_BACKEND).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if _resolved_module(node, str(path)) not in _LEAF_MODULES:
                    continue
                hits = sorted({a.name for a in node.names} & MOVED_NAMES)
                if hits:
                    rel = os.path.relpath(path, _REPO)
                    offenders.append(f"{rel}:{node.lineno}: {', '.join(hits)}")
        self.assertEqual(
            offenders, [],
            "import these from core.adapters instead: " + "; ".join(offenders))

    def test_base_adapter_no_longer_re_exports_the_target_helpers(self):
        from core.training.adapters import base_adapter

        for name in ("is_lora_wrappable_linear", "lora_branch_dtype"):
            self.assertFalse(hasattr(base_adapter, name),
                             f"{name} moved to core.adapters.targets")


if __name__ == "__main__":
    unittest.main()
