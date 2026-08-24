"""Guard: ``adamw8bit`` + Block Swap must keep REAL 8-bit optimizer state.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adamw8bit_fused_bnb_state_test.py -v

THE DEFECT
----------
Selecting ``optimizer: adamw8bit`` with ``blocks_to_swap > 0`` routes every
update through ``adamw8bit_fused.step_param``, because bitsandbytes' ``step()``
runs after Block Swap has moved parameters to the CPU. That ``step_param`` was a
hand-written AdamW that allocated its moments with ``torch.zeros_like(p)``, so
the run was named 8-bit but carried dense state. Measured by
``core/training/probes/optimizer_bf16_and_vram.py`` (``--arm vram``):

    adamw8bit, step()  path   2.031250 B/param   uint8 x2 + absmax
    adamw8bit, fused   path   4.000000 B/param   bf16 x2

and the two state formats were mutually unreadable, so a checkpoint could not
move between Block Swap on and off.

THE FIX
-------
``step_param`` delegates to ``Optimizer8bit.init_state`` / ``update_step``, the
per-parameter seam bitsandbytes' own ``step()`` drives. Same kernels, same state,
one format. What this file pins:

* the fused path's state really is bitsandbytes' 8-bit layout;
* the two paths produce bitwise-identical parameters and interchangeable
  ``state_dict``s;
* stochastic rounding still reaches the fused update (the BF16 rounding defect
  must not come back through the delegation);
* ``step()`` itself is untouched;
* ``(gindex, pindex)`` are resolved correctly -- a ``GlobalOptimManager``
  override applies through the hook exactly as it does through ``step()``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.optimizers.adamw8bit_fused import patch_adamw8bit_fused  # noqa: E402
from core.training.optimizers.stochastic_rounding import (  # noqa: E402
    NATIVE_ATTR,
    NATIVE_STEP_PARAM,
    WRAPPED_ATTR,
    attach_stochastic_rounding,
)

LR = 1e-5
# Above bitsandbytes' min_8bit_size (4096); below it the state is FP32 by design,
# in both paths.
N = 1 << 13
SEED = 20260825
STEPS = 8


def _weights(device="cpu") -> torch.Tensor:
    torch.manual_seed(SEED)
    return ((torch.randn(N) * 0.02).bfloat16()).to(device)


def _bnb_adamw8bit(params, lr=LR):
    import bitsandbytes as bnb

    return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0)


@unittest.skipUnless(torch.cuda.is_available(), "the bitsandbytes 8-bit kernels require CUDA")
class FusedPathKeeps8BitStateTest(unittest.TestCase):
    """The headline: the Block Swap path allocates bitsandbytes' own state."""

    def _drive(self, fused: bool, stochastic_rounding: bool = False, w0=None,
               steps=STEPS, lr=LR):
        w0 = _weights("cuda") if w0 is None else w0
        p = torch.nn.Parameter(w0.clone())
        optimizer = _bnb_adamw8bit([p], lr=lr)
        if fused:
            patch_adamw8bit_fused(optimizer, stochastic_rounding)
        elif stochastic_rounding:
            attach_stochastic_rounding(optimizer)

        for _ in range(steps):
            p.grad = torch.full_like(p, -1.0)
            if fused:
                optimizer.step_param(p, optimizer.param_groups[0])
            else:
                optimizer.step()
            p.grad = None
        return p, optimizer

    def test_the_fused_state_is_bitsandbytes_8bit_not_dense(self):
        p, optimizer = self._drive(fused=True)
        state = optimizer.state[p]

        self.assertEqual(
            set(state) - {"step"},
            {"state1", "state2", "qmap1", "qmap2", "absmax1", "absmax2"},
            "the fused path is not using bitsandbytes' blockwise 8-bit state",
        )
        self.assertEqual(state["state1"].dtype, torch.uint8)
        self.assertEqual(state["state2"].dtype, torch.uint8)
        self.assertNotIn("exp_avg", state, "dense moments are the defect this pins")

        # 2 uint8 moments + 2 fp32 absmax per 256-element block = 2.03125 B/param,
        # against 4.0 for the bf16 moments the old implementation allocated.
        state_bytes = sum(
            v.numel() * v.element_size()
            for k, v in state.items()
            if torch.is_tensor(v) and k not in ("qmap1", "qmap2")  # shared, not per-param
        )
        self.assertAlmostEqual(state_bytes / p.numel(), 2.03125, places=5)

    def test_both_paths_produce_the_same_parameter(self):
        w0 = _weights("cuda")
        p_step, _ = self._drive(fused=False, w0=w0)
        p_fused, _ = self._drive(fused=True, w0=w0)
        self.assertTrue(
            torch.equal(p_step.data, p_fused.data),
            "the fused seam must run the same update as step()",
        )

    def test_the_step_counter_advances_per_parameter(self):
        _, optimizer = self._drive(fused=True)
        self.assertEqual(next(iter(optimizer.state.values()))["step"], STEPS)

    def test_a_state_dict_saved_by_step_is_loadable_by_the_fused_path(self):
        w0 = _weights("cuda")
        p_step, opt_step = self._drive(fused=False, w0=w0)

        p = torch.nn.Parameter(p_step.data.clone())
        optimizer = _bnb_adamw8bit([p])
        patch_adamw8bit_fused(optimizer, False)
        optimizer.load_state_dict(opt_step.state_dict())

        p.grad = torch.full_like(p, -1.0)
        optimizer.step_param(p, optimizer.param_groups[0])
        p_step.grad = torch.full_like(p_step, -1.0)
        opt_step.step()

        self.assertEqual(optimizer.state[p]["step"], STEPS + 1)
        self.assertTrue(torch.equal(p.data, p_step.data),
                        "resuming a step() checkpoint on the fused path diverged")

    def test_a_state_dict_saved_by_the_fused_path_is_loadable_by_step(self):
        w0 = _weights("cuda")
        p_fused, opt_fused = self._drive(fused=True, w0=w0)

        p = torch.nn.Parameter(p_fused.data.clone())
        optimizer = _bnb_adamw8bit([p])
        optimizer.load_state_dict(opt_fused.state_dict())

        p.grad = torch.full_like(p, -1.0)
        optimizer.step()
        p_fused.grad = torch.full_like(p_fused, -1.0)
        opt_fused.step_param(p_fused, opt_fused.param_groups[0])

        self.assertTrue(torch.equal(p.data, p_fused.data),
                        "resuming a Block Swap checkpoint without Block Swap diverged")

    def test_stochastic_rounding_still_reaches_the_fused_update(self):
        """The delegation must not resurrect the BF16 rounding defect.

        bitsandbytes writes the parameter inside the kernel, so coverage now comes
        from making the parameter FP32 for the call rather than from arithmetic in
        Python. An Adam step peaks at 10*lr, so weights above 512*10*lr cannot move
        under round-to-nearest -- at lr 1e-6 that is most of a ~N(0, 0.02) tensor.
        """
        lr, steps = 1e-6, 200
        w0 = _weights("cuda")
        frozen = w0.abs().float() >= 512 * 10 * lr
        self.assertGreater(frozen.float().mean().item(), 0.5)

        p_rtn, _ = self._drive(fused=True, stochastic_rounding=False, w0=w0,
                               steps=steps, lr=lr)
        self.assertTrue(
            torch.equal(p_rtn.data[frozen], w0[frozen]),
            "round-to-nearest is expected to leave these weights bitwise unchanged",
        )

        p_sr, optimizer = self._drive(fused=True, stochastic_rounding=True, w0=w0,
                                      steps=steps, lr=lr)
        moved = p_sr.data[frozen].ne(w0[frozen]).float().mean().item()
        self.assertGreater(moved, 0.5, "stochastic rounding did not reach the fused update")
        drift = (p_sr.data[frozen].float() - w0[frozen].float()).mean().item()
        self.assertGreater(drift, 0.5 * steps * lr, "the sub-ULP updates were not carried")

        # And it did not cost the 8-bit state, which is the whole point of the path.
        state = optimizer.state[p_sr]
        self.assertEqual(state["state1"].dtype, torch.uint8)
        self.assertEqual(state["state2"].dtype, torch.uint8)

    def test_the_parameter_and_gradient_are_bf16_again_after_a_rounded_step(self):
        p = torch.nn.Parameter(_weights("cuda"))
        optimizer = _bnb_adamw8bit([p])
        patch_adamw8bit_fused(optimizer, True)
        grad = torch.full_like(p, -1.0)
        p.grad = grad
        optimizer.step_param(p, optimizer.param_groups[0])
        self.assertEqual(p.data.dtype, torch.bfloat16)
        self.assertIs(p.grad, grad)
        self.assertEqual(p.grad.dtype, torch.bfloat16)

    def test_a_global_override_reaches_the_fused_update(self):
        """Pins the ``(gindex, pindex)`` the hook has to reconstruct.

        They exist only so ``get_config`` can find
        ``GlobalOptimManager.index2config[(gindex, pindex)]``. Overriding the
        second parameter to 32-bit state is therefore a direct read-out of whether
        the hook passed the same indices ``step()`` would have.
        """
        from bitsandbytes.optim import GlobalOptimManager

        manager = GlobalOptimManager.get_instance()
        try:
            a = torch.nn.Parameter(_weights("cuda"))
            b = torch.nn.Parameter(_weights("cuda"))
            optimizer = _bnb_adamw8bit([a, b])
            manager.override_config(b, "optim_bits", 32)
            manager.register_parameters(optimizer.param_groups)
            patch_adamw8bit_fused(optimizer, False)

            for p in (a, b):
                p.grad = torch.full_like(p, -1.0)
                optimizer.step_param(p, optimizer.param_groups[0])

            self.assertEqual(optimizer.state[a]["state1"].dtype, torch.uint8)
            self.assertEqual(optimizer.state[b]["state1"].dtype, torch.float32,
                             "the (gindex, pindex) override did not reach the hook")
        finally:
            manager.initialize()  # drop the override; the manager is a singleton


class PatchContractTest(unittest.TestCase):
    """What ``patch_adamw8bit_fused`` installs, and what it refuses. No CUDA."""

    def _optimizer(self):
        return _bnb_adamw8bit([torch.nn.Parameter(_weights())])

    def test_step_is_left_as_bitsandbytes_own(self):
        optimizer = self._optimizer()
        patch_adamw8bit_fused(optimizer, False)
        self.assertNotIn("step", vars(optimizer),
                         "the non-Block-Swap path must keep bitsandbytes' own step()")
        self.assertIs(optimizer.step.__func__, type(optimizer).step)

    def test_a_non_bitsandbytes_optimizer_is_refused(self):
        """The update delegates, so an optimizer without the seam cannot be patched.

        It used to be accepted -- the patch installed a hand-written AdamW on
        anything -- which is exactly how the dense-state path stayed invisible.
        """
        optimizer = torch.optim.SGD([torch.nn.Parameter(_weights())], lr=LR)
        with self.assertRaises(TypeError):
            patch_adamw8bit_fused(optimizer, False)

    def test_the_generic_interposer_leaves_this_step_param_alone(self):
        """It applies stochastic rounding itself; wrapping would round twice."""
        optimizer = self._optimizer()
        patch_adamw8bit_fused(optimizer, True)
        self.assertTrue(getattr(optimizer.step_param, NATIVE_ATTR, False))
        self.assertEqual(attach_stochastic_rounding(optimizer), (NATIVE_STEP_PARAM,))
        self.assertFalse(getattr(optimizer.step_param, WRAPPED_ATTR, False))
        self.assertFalse(getattr(optimizer.update_step, WRAPPED_ATTR, False))

    def test_the_fallback_step_dispatches_through_the_instance_attribute(self):
        """``adamw8bit_step`` is not installed today, so nothing else pins it.

        A batch ``step()`` that called the MODULE-LEVEL ``..._step_param`` would
        bypass every interposition, which rebinds the instance attribute. Recorded
        without running the update, so this stays CPU-only.
        """
        from core.training.optimizers.adamw8bit_fused import adamw8bit_step

        optimizer = self._optimizer()
        patch_adamw8bit_fused(optimizer, True)
        p = optimizer.param_groups[0]["params"][0]
        p.grad = torch.full_like(p, -1.0)

        seen = []
        optimizer.step_param = lambda param, group: seen.append(param)
        adamw8bit_step(optimizer)
        self.assertEqual(seen, [p], "step() did not dispatch through self.step_param")

    def test_a_parameter_without_a_gradient_is_skipped(self):
        optimizer = self._optimizer()
        patch_adamw8bit_fused(optimizer, False)
        p = optimizer.param_groups[0]["params"][0]
        optimizer.step_param(p, optimizer.param_groups[0])
        self.assertEqual(len(optimizer.state.get(p, {})), 0)


if __name__ == "__main__":
    unittest.main()
