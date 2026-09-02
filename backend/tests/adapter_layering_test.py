"""``core.adapters`` layering gate and Phase 1 re-export identity.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_layering_test.py -v -s

WHY THIS FILE EXISTS. ``core.training.adapters`` is a subpackage of
``core.training``, whose ``__init__`` reaches ``api.param_defaults`` and
transitively ``api.routes`` -- a ``core -> api`` back-edge. Twelve generation
modules across eleven architectures imported the leaf LoRA layer class from
there, so each inherited the whole API surface and a CUDA context. Measured in a
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

import json
import os
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


class ReexportIdentityTest(unittest.TestCase):
    """Old path and new path must yield THE SAME object, not an equal one.

    ``isinstance`` checks in the loaders and
    ``type(m).__name__ == "LoRALinearLayer"`` gates in
    ``core.models.common.int8_runtime_quantize`` /
    ``core.vram_optimization`` all break silently if a shim ever rebinds a
    second class object under the same name.
    """

    def test_moved_symbols_are_the_same_object_on_both_paths(self):
        import core.adapters as new
        from core.adapters import layers as new_layers
        from core.adapters import targets as new_targets
        from core.training.adapters import base_adapter as old_base
        from core.training.adapters import sd15_adapter as old_sd15
        from core.training.adapters import minimax_h3_adapter as old_h3

        pairs = [
            ("LoRALinearLayer", old_sd15.LoRALinearLayer, new_layers.LoRALinearLayer),
            ("LoRALinearLayer", old_h3.LoRALinearLayer, new_layers.LoRALinearLayer),
            ("MiniMaxH3LoRALinearLayer", old_h3.MiniMaxH3LoRALinearLayer,
             new_layers.MiniMaxH3LoRALinearLayer),
            ("is_lora_wrappable_linear", old_base.is_lora_wrappable_linear,
             new_targets.is_lora_wrappable_linear),
            ("lora_branch_dtype", old_base.lora_branch_dtype,
             new_targets.lora_branch_dtype),
            ("count_quantized_linears", old_base.count_quantized_linears,
             new_targets.count_quantized_linears),
        ]
        for name, old_obj, new_obj in pairs:
            with self.subTest(symbol=name):
                self.assertIs(old_obj, new_obj)
                self.assertIs(getattr(new, name), new_obj)

    def test_minimax_subclass_relationship_survives_the_move(self):
        from core.adapters import LoRALinearLayer, MiniMaxH3LoRALinearLayer

        self.assertTrue(issubclass(MiniMaxH3LoRALinearLayer, LoRALinearLayer))
        self.assertEqual(LoRALinearLayer.__name__, "LoRALinearLayer")
        self.assertEqual(MiniMaxH3LoRALinearLayer.__name__, "MiniMaxH3LoRALinearLayer")


if __name__ == "__main__":
    unittest.main()
