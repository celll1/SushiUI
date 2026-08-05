"""Guard: the Schedule-Free ``z`` sequence must not be frozen by round-to-nearest.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/schedulefree_z_sequence_test.py -v

THE DEFECTS
-----------
``optimizer_stochastic_rounding`` was made to reach the PARAMETER update
(8547f93c, 836a6376) and both commits recorded that the Schedule-Free ``z``
sequence was still written with round-to-nearest. z is not a side state: it is
the sequence the algorithm optimizes, ``p`` is derived from it, and it is read
back and rewritten on every step -- so an update it cannot represent is
discarded permanently, exactly like a sub-half-ULP parameter update.

Four distinct failures were found underneath that, all measured here:

1. **8-bit z was read back as zero, always.** ``__constant__`` symbols are per
   translation unit. ``adamw8bit_kernel.cu`` / ``lion8bit_kernel.cu`` declare
   ``d_qmap_signed`` and their ``init_quantization_maps()`` fills only their own
   copy; the Schedule-Free kernels declare their own copies, which nothing ever
   filled. Every ``dequantize_value()`` in the Schedule-Free kernels therefore
   returned 0. Measured before the fix: one step collapsed ``absmax_z`` from
   6.8e-2 to 1.0e-5 (= lr, the update applied to a z of zero) and drove all
   16384 z codes to 255.

2. **z's initial quantization used a different map from the kernel's decoder.**
   ``quantize_blockwise_inplace`` wrote LINEAR codes while the kernel decodes
   through the dynamic map: mean |z - p| was 2.34e-2 against mean |p| 3.32e-2 on
   a real Krea 2 tensor (70% relative error), where the 8-bit grid can represent
   it to 7.1e-4 (2%).

3. **z's SCALE decayed 0.7031% per step, compounding.** The signed dynamic map
   is not symmetric: it ends at +1.000000000 but at -0.992968738. A block whose
   extreme element is negative normalizes to exactly -1.0, can only be stored as
   -0.992968738, and so hands the next step an ``absmax_z`` 0.7031% smaller --
   which is then recomputed from that smaller value, and so on. Over 3000 steps
   with zero-mean gradients, mean|z| fell to 0.485 of its bf16 reference (0.254
   with stochastic rounding, which tracks the sinking scale instead of hiding it
   behind frozen codes). Fixing it needs BOTH halves:

   * quantize z with symmetric headroom (``absmax = max|z| / 0.992968738``), so
     the extreme element is exactly representable in either sign and the
     dequantize -> recompute-absmax -> requantize round trip is idempotent;
   * store the element that DEFINES absmax without stochastic rounding, so the
     block's scale never inherits rounding noise. absmax is a maximum over the
     block's own stored values, so noise there feeds back into the scale: with
     headroom alone it ran AWAY (+0.63%/step, 1.5e8x over 3000 steps), and
     clamping that element down one code instead inverted the same feedback and
     sank the block to 0.37x.

4. **z's writes were round-to-nearest.** Measured on a real Krea 2 slice, 300
   steps at lr 1e-5 with a constant gradient, once (1)-(3) were fixed:

       storage      rounding   z elements that moved   realized drift
       8-bit codes  RTN                        0.54%            1.0%
       8-bit codes  stochastic                86.08%           99.8%
       bf16         RTN                       14.43%           11.7%
       bf16         stochastic                99.60%           96.8%

   ``absmax`` moved by 0.15-0.20% over those 300 steps, in both 8-bit rows -- the
   figures above are the rounding, not a scale artefact.

   The 8-bit stochastic row was 75.98% / 86.6% until the exemption in (3) was
   fixed to compare against a BROADCAST block maximum:
   ``cub::BlockReduce::Reduce`` returns the aggregate in thread 0 only, so every
   other thread was comparing against its own warp/raking partial and exempting
   itself, which left 12.7% of the tensor -- the large-magnitude end of it -- on
   round-to-nearest.

   Survival over 20000 steps with zero-mean gradients, as a ratio to the bf16
   reference: z at 1.12 (RTN) and 1.12 (SR), the parameter at 1.00 and 1.05. The
   excess grows with the step count rather than saturating, and is the same in
   both rounding modes, so it is the 8-bit grid's own noise rather than the
   rounding rule; 20000 steps is the longest horizon measured.

   Schedule-Free + stochastic rounding also could not complete a single step in
   the unquantized path: with the parameter lifted to FP32 and z left BF16,
   ``y.lerp_(end=z)`` raised "expected dtype float for `end`".

And on resume: a checkpoint written before (1) was fixed holds a z that decodes
to a constant, and z is not inert -- ``y = (1 - ckp1) * y + ckp1 * z`` pulls the
weights to it. Measured, 300 zero-gradient steps after such a resume took mean|p|
from 1.63e-2 to 5.21e-5.

WHAT EACH GROUP PINS
--------------------
* ``ZInitQuantizationTest``    -- (2), on CPU: the codes must decode back through
  the map the kernels use, with the headroom the kernels expect.
* ``Bf16ZSequenceTest``        -- (4) for a parameter-dtype z, through the real
  ``step()`` with BF16 params and BF16 grads.
* ``TrainEvalSequenceTest``    -- the same for the ``p.lerp_(end=z)`` writes in
  train()/eval(), plus that a quantized z can be lerped at all. NOTE: nothing in
  the trainer calls ``optimizer.eval()`` today, so that path is exercised here
  and nowhere else.
* ``EightBitZKernelTest``      -- (1), (3) and (4) through the REAL CUDA kernel,
  including the long-horizon survival that (3) is about and the per-block count
  of elements exempted from stochastic rounding.
* ``LionScheduleFreeRefusalTest`` -- Lion + Schedule-Free is REFUSED: its kernel
  stores Lion's momentum EMA into the parameter instead of the position
  sequence. The kernel's own fixes are pinned there too, on the direct call.
* ``ScheduleFreeResumeGuardTest`` -- the resume repair.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.optimizers.quantization_map import create_quantization_map  # noqa: E402

SEED = 20260805
BLOCKSIZE = 256


def _weights(n: int, scale: float = 0.02) -> torch.Tensor:
    torch.manual_seed(SEED)
    return (torch.randn(n) * scale).bfloat16()


def _dequantize(codes: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
    """Decode 8-bit blockwise codes exactly as ``dequantize_value()`` does."""
    qmap = create_quantization_map(signed=True)
    flat = codes.detach().cpu().reshape(-1).long()
    scales = absmax.detach().cpu().float().repeat_interleave(BLOCKSIZE)[: flat.numel()]
    return qmap[flat] * scales


class _FakeCudaParameter(torch.nn.Parameter):
    """A CPU parameter that reports ``is_cuda`` (both optimizers skip CPU params)."""

    @property
    def is_cuda(self) -> bool:  # noqa: D401
        return True


def _fake_param(tensor: torch.Tensor) -> _FakeCudaParameter:
    return torch.Tensor._make_subclass(_FakeCudaParameter, tensor, True)


class _NullExtension:
    """Stands in for the compiled extension so the unquantized path can run on CPU."""

    def init_quantization_maps(self, *args, **kwargs):
        pass


class ZInitQuantizationTest(unittest.TestCase):
    """z starts as a quantized copy of p -- through the map the kernel decodes with."""

    def test_codes_decode_back_to_the_parameter(self):
        from core.training.optimizers.adamw8bit_ringbuffer import quantize_blockwise_inplace

        p = _weights(4096).float()
        codes, absmax = quantize_blockwise_inplace(p, BLOCKSIZE)
        error = (_dequantize(codes, absmax) - p).abs().mean().item()

        # The 8-bit dynamic grid represents a N(0, 0.02) tensor to ~1e-3 of its
        # own scale. Writing linear codes and decoding them through the dynamic
        # map gave 0.70 * mean|p|.
        self.assertLess(
            error, 0.05 * p.abs().mean().item(),
            "z is initialised through a different quantization map than the kernel decodes with",
        )

    def test_the_codes_are_the_nearest_grid_point(self):
        """The kernel's quantize_value() picks the nearer of the two neighbours."""
        from core.training.optimizers.adamw8bit_ringbuffer import quantize_blockwise_inplace

        qmap = create_quantization_map(signed=True)
        p = _weights(BLOCKSIZE).float()
        codes, absmax = quantize_blockwise_inplace(p, BLOCKSIZE)

        normalized = (p / absmax[0]).clamp(-1.0, 1.0)
        brute = (normalized.unsqueeze(1) - qmap.unsqueeze(0)).abs().argmin(dim=1)
        self.assertTrue(torch.equal(codes.long(), brute))

    def test_absmax_carries_symmetric_headroom(self):
        """absmax = max|z| / 0.992968738, so the extreme is representable either sign."""
        from core.training.optimizers.adamw8bit_ringbuffer import quantize_blockwise_inplace

        qmap = create_quantization_map(signed=True)
        qmax_symmetric = min(-qmap.min().item(), qmap.max().item())
        self.assertLess(qmax_symmetric, 1.0, "the signed map is expected to be asymmetric")

        p = _weights(3 * BLOCKSIZE).float()
        _, absmax = quantize_blockwise_inplace(p, BLOCKSIZE)
        expected = p.view(3, BLOCKSIZE).abs().amax(dim=1) / qmax_symmetric
        self.assertTrue(torch.allclose(absmax, expected))

    def test_the_extreme_element_round_trips_exactly(self):
        """The whole point of the headroom: no 0.7% shrink per step.

        With a NEGATIVE block extreme (which the map cannot represent at full
        scale), decoding used to give back 0.9930 of the block maximum, and the
        kernel adopted that as the next absmax.
        """
        from core.training.optimizers.adamw8bit_ringbuffer import quantize_blockwise_inplace

        p = _weights(BLOCKSIZE).float()
        p[0] = -p.abs().max() * 1.5  # force the extreme to be negative
        codes, absmax = quantize_blockwise_inplace(p, BLOCKSIZE)
        decoded = _dequantize(codes, absmax)
        self.assertAlmostEqual(
            decoded.abs().max().item(), p.abs().max().item(), places=6,
            msg="the block maximum is not recoverable, so absmax shrinks every step",
        )


class _UnquantizedScheduleFree:
    """Drives the real ``step()`` on the unquantized (parameter-dtype z) path."""

    N = 1 << 13
    STEPS = 200
    LR = 1e-6

    def _optimizer(self, stochastic_rounding: bool):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        original = mod.get_extension
        mod.get_extension = lambda: _NullExtension()
        try:
            p = _fake_param(_weights(self.N).clone())
            opt = mod.AdamW8bit_RingBuffer(
                [p], lr=self.LR, weight_decay=0.0, use_8bit=False,
                schedule_free=True, stochastic_rounding=stochastic_rounding,
            )
        finally:
            mod.get_extension = original
        opt.train()
        return p, opt

    def _run(self, stochastic_rounding: bool):
        p, opt = self._optimizer(stochastic_rounding)
        opt._init_param_state(p)
        z0 = opt.state[p]["z"].detach().clone()
        for _ in range(self.STEPS):
            # BF16 gradient for a BF16 parameter: what autograd actually hands back.
            p.grad = torch.full((self.N,), -1.0, dtype=torch.bfloat16)
            opt.step()
        return p, z0, opt.state[p]["z"].detach()


class Bf16ZSequenceTest(unittest.TestCase, _UnquantizedScheduleFree):
    """z in the parameter's dtype: round-to-nearest freezes it, stochastic rounding does not."""

    def setUp(self):
        # The Schedule-Free step moves z by ~lr per step; an element of z can only
        # move under round-to-nearest when half a BF16 ULP is below that, i.e.
        # |z| <= 512*lr.
        self.frozen = _weights(self.N).abs().float() >= 512 * self.LR
        self.assertGreater(self.frozen.float().mean().item(), 0.5)

    def test_round_to_nearest_freezes_the_z_sequence(self):
        _, z0, z1 = self._run(stochastic_rounding=False)
        self.assertEqual(z0.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(z1[self.frozen], z0[self.frozen]),
            "round-to-nearest is expected to leave these elements of z bitwise unchanged",
        )

    def test_stochastic_rounding_carries_the_z_sequence(self):
        _, z0, z1 = self._run(stochastic_rounding=True)

        moved = z1[self.frozen].ne(z0[self.frozen]).float().mean().item()
        self.assertGreater(moved, 0.5, "stochastic rounding did not reach the z sequence")

        drift = (z1[self.frozen].float() - z0[self.frozen].float()).mean().item()
        self.assertGreater(
            drift, 0.5 * self.STEPS * self.LR,
            "z drifted by far less than the updates it was given",
        )
        self.assertLess(
            drift, 2.0 * self.STEPS * self.LR,
            "z drifted by far more than the updates it was given (rounding must be unbiased)",
        )

    def test_schedule_free_with_stochastic_rounding_completes_a_step(self):
        """It raised "expected dtype float for `end`" in y.lerp_(end=z) at step 1."""
        p, opt = self._optimizer(stochastic_rounding=True)
        p.grad = torch.full((self.N,), -1.0, dtype=torch.bfloat16)
        opt.step()  # must not raise
        self.assertEqual(p.dtype, torch.bfloat16)
        self.assertEqual(opt.state[p]["z"].dtype, torch.bfloat16)

    def test_the_z_state_is_not_promoted_to_fp32(self):
        """The fix must stay scratch-based: no persistent 4-byte-per-element master."""
        _, z0, z1 = self._run(stochastic_rounding=True)
        self.assertEqual(z1.dtype, torch.bfloat16)
        self.assertEqual(z1.numel(), self.N)


class TrainEvalSequenceTest(unittest.TestCase):
    """train()/eval() write ``p = lerp(p, z)`` -- the same rounding question."""

    N = 1 << 12
    CALLS = 300
    BETA1 = 0.99          # train() lerps p toward z with weight 1 - beta1
    WEIGHT = 1 - BETA1
    # z sits 1e-3 above p, so a single lerp moves p by 1e-5 -- an order of
    # magnitude under half a BF16 ULP (~7.6e-5 at |p| ~ 0.02) -- while the total
    # available movement is several ULPs. Round-to-nearest discards all of it.
    OFFSET = 1e-3

    def _optimizer(self, stochastic_rounding: bool, quantized_z: bool = False):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        original = mod.get_extension
        mod.get_extension = lambda: _NullExtension()
        try:
            p = _fake_param(_weights(self.N).clone())
            opt = mod.AdamW8bit_RingBuffer(
                [p], lr=1e-6, betas=(self.BETA1, 0.999), weight_decay=0.0, use_8bit=False,
                schedule_free=True, stochastic_rounding=stochastic_rounding,
            )
        finally:
            mod.get_extension = original

        state = opt.state[p]
        z = (p.detach().float() + self.OFFSET)
        if quantized_z:
            from core.training.optimizers.adamw8bit_ringbuffer import quantize_blockwise_inplace

            codes, absmax = quantize_blockwise_inplace(z, BLOCKSIZE)
            state["z"] = codes
            state["absmax_z"] = absmax
            state["is_8bit"] = True
        else:
            state["z"] = z.bfloat16()
            state["is_8bit"] = False
        return p, opt

    def _unrepresentable(self, before: torch.Tensor) -> torch.Tensor:
        """Elements whose lerp step (``WEIGHT * OFFSET``) is under half a BF16 ULP."""
        mask = before.abs().float() >= 512 * self.WEIGHT * self.OFFSET
        self.assertGreater(mask.float().mean().item(), 0.5)
        return mask

    def test_round_to_nearest_freezes_the_train_mode_write(self):
        p, opt = self._optimizer(stochastic_rounding=False)
        before = p.detach().clone()
        mask = self._unrepresentable(before)
        for _ in range(self.CALLS):
            opt.train()
        self.assertTrue(
            torch.equal(p.detach()[mask], before[mask]),
            "round-to-nearest is expected to discard every one of these lerps",
        )

    def test_stochastic_rounding_carries_the_train_mode_write(self):
        p, opt = self._optimizer(stochastic_rounding=True)
        before = p.detach().clone()
        mask = self._unrepresentable(before)
        for _ in range(self.CALLS):
            opt.train()
        moved = p.detach()[mask].ne(before[mask]).float().mean().item()
        self.assertGreater(moved, 0.5, "stochastic rounding did not reach train()")
        # Every lerp moves p TOWARD z, which is above it.
        self.assertGreater((p.detach().float() - before.float()).mean().item(), 0.0)
        self.assertEqual(p.dtype, torch.bfloat16)

    def test_eval_moves_the_parameter_the_other_way(self):
        p, opt = self._optimizer(stochastic_rounding=True)
        before = p.detach().clone()
        for _ in range(self.CALLS):
            opt.eval()
        self.assertLess((p.detach().float() - before.float()).mean().item(), 0.0)

    def test_a_quantized_z_is_decoded_before_the_lerp(self):
        """state['z'] holds UINT8 CODES in 8-bit mode.

        ``p.lerp_(end=codes)`` raised "got dtype unsigned char"; casting the codes
        to the parameter dtype instead would drag p toward integers in [0, 255].
        p must converge on the DEQUANTIZED z.
        """
        p, opt = self._optimizer(stochastic_rounding=True, quantized_z=True)
        before = p.detach().clone()
        opt.train()  # must not raise
        self.assertEqual(p.dtype, torch.bfloat16)

        for _ in range(self.CALLS):
            opt.train()

        state = opt.state[p]
        z = _dequantize(state["z"], state["absmax_z"])
        # 300 lerps at weight 0.01 close 95% of the gap to z.
        self.assertLess(
            (p.detach().float() - z).abs().mean().item(),
            0.25 * self.OFFSET + 0.02 * z.abs().mean().item(),
            "p did not converge on the dequantized z",
        )
        self.assertLess(p.detach().abs().max().item(), 1.0,
                        "p was lerped toward the raw quantization codes")


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class EightBitZKernelTest(unittest.TestCase):
    """The shipping Schedule-Free path: the REAL 8-bit CUDA kernel.

    Small tensors on the real device (<1 MB of VRAM); no model is loaded.
    """

    N = 1 << 13
    STEPS = 200
    LR = 1e-5

    def _optimizer(self, stochastic_rounding: bool, weights: torch.Tensor):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        p = torch.nn.Parameter(weights.clone().cuda())
        opt = mod.AdamW8bit_RingBuffer(
            [p], lr=self.LR, weight_decay=0.0, use_8bit=True,
            schedule_free=True, stochastic_rounding=stochastic_rounding,
        )
        opt.train()
        return p, opt

    def test_the_kernel_reads_back_the_z_it_was_given(self):
        """Its __constant__ quantization map was never filled, so z decoded to 0."""
        torch.manual_seed(SEED)
        weights = _weights(self.N)
        p, opt = self._optimizer(False, weights)
        opt._init_param_state(p)
        state = opt.state[p]

        before = _dequantize(state["z"], state["absmax_z"])
        # A zero gradient leaves z untouched: z_new = z - lr * (0/denom).
        p.grad = torch.zeros_like(p)
        opt.step()
        after = _dequantize(state["z"], state["absmax_z"])

        self.assertLess(
            (after - before).abs().mean().item(), 0.05 * before.abs().mean().item(),
            "one kernel step destroyed z -- the Schedule-Free constant map is not initialised",
        )
        self.assertGreater(
            state["absmax_z"].min().item(), 10 * self.LR,
            "absmax_z collapsed to ~lr, which is what a z of zero produces",
        )

    def _drift(self, stochastic_rounding: bool):
        torch.manual_seed(SEED)
        weights = _weights(self.N)
        p, opt = self._optimizer(stochastic_rounding, weights)
        p.grad = torch.full_like(p, -1.0)
        opt.step()
        state = opt.state[p]
        codes0 = state["z"].clone()
        z0 = _dequantize(codes0, state["absmax_z"])
        for _ in range(self.STEPS):
            p.grad = torch.full_like(p, -1.0)
            opt.step()
        z1 = _dequantize(state["z"], state["absmax_z"])
        moved = codes0.ne(state["z"]).float().mean().item()
        return moved, (z1 - z0).mean().item() / (self.STEPS * self.LR)

    def test_round_to_nearest_pins_the_z_codes(self):
        moved, realized = self._drift(stochastic_rounding=False)
        self.assertLess(moved, 0.25, "expected the 8-bit z codes to stay put under RTN")
        self.assertLess(realized, 0.75, "expected most of the intended drift to be discarded")

    def test_stochastic_quantization_carries_the_z_sequence(self):
        moved, realized = self._drift(stochastic_rounding=True)
        # The thresholds are tight on purpose: comparing the block maximum
        # against cub::BlockReduce's RETURN VALUE (valid in thread 0 only) rather
        # than a broadcast of it exempted an eighth of the tensor from stochastic
        # rounding -- biased toward the large magnitudes -- and showed up here as
        # 76% moved / 87% realized instead of 86% / 100%.
        self.assertGreater(moved, 0.85, "stochastic rounding did not reach the 8-bit z codes")
        self.assertGreater(realized, 0.92, "the z sequence still lost part of its drift")
        self.assertLess(realized, 1.6, "z drifted far past its updates -- rounding must be unbiased")

    def test_the_flag_is_what_changes_the_behaviour(self):
        """Same seed, same data: only optimizer_stochastic_rounding differs."""
        off_moved, _ = self._drift(False)
        on_moved, _ = self._drift(True)
        self.assertGreater(on_moved, off_moved + 0.25)

    def test_the_default_is_still_round_to_nearest(self):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        p = torch.nn.Parameter(_weights(256).cuda())
        opt = mod.AdamW8bit_RingBuffer([p], lr=self.LR, use_8bit=True, schedule_free=True)
        self.assertFalse(opt.param_groups[0]["stochastic_rounding"])

    def test_the_kernel_ignores_the_seed_when_the_flag_is_off(self):
        """The kernel's own flag, pinned independently of Python's seed choice.

        Python passes seed=0 when the flag is off, and ``sr_mix(0) == 0`` makes
        ``u`` a fixed function of ``tid`` -- which incidentally pins the codes.
        So a kernel that ignored ``stochastic_z`` and always rounded
        stochastically would still look deterministic from Python. Calling it
        directly with a NONZERO seed and the flag off is what separates them.
        """
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        def run(seed):
            torch.manual_seed(SEED)
            p = torch.nn.Parameter(_weights(self.N).cuda())
            opt = mod.AdamW8bit_RingBuffer(
                [p], lr=self.LR, weight_decay=0.0, use_8bit=True, schedule_free=True,
            )
            opt.train()
            opt._init_param_state(p)
            state = opt.state[p]
            for _ in range(20):
                opt.ext.adamw_8bit_schedulefree_update(
                    p, torch.full_like(p, -1.0), state["z"], state["exp_avg_sq"],
                    state["absmax_z"], state["absmax2"],
                    0.9, 0.999, 1e-8, self.LR, 0.0, 0.0, 1.0, 1e-3,
                    False,  # stochastic_z OFF
                    seed,
                )
            return state["z"].clone()

        self.assertTrue(
            torch.equal(run(0), run(0x5DEECE66)),
            "the kernel is rounding stochastically even with stochastic_z=False",
        )

    def test_exactly_one_element_per_block_is_exempt_from_stochastic_rounding(self):
        """Only the element that DEFINES absmax may be rounded to nearest.

        ``cub::BlockReduce::Reduce`` returns the aggregate in thread 0 only, so
        comparing against its return value in every thread exempted whichever
        element was the largest of its warp/raking segment as well -- 12.7% of
        the tensor, biased toward the large magnitudes, silently kept on
        round-to-nearest. Nothing pinned the count, which is how it survived.

        Fixture: every element is given a gradient that moves z by exactly half a
        grid gap, so a stochastically-rounded element flips its code with
        probability 1/2 and survives 32 seeds with probability 2^-31. The block
        extreme gets a zero gradient so absmax cannot move (which would shift
        everyone else's rounding fraction). Seed-invariant codes are then the
        exempt set.
        """
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        seeds = 32
        lr = 1e-4
        beta2 = 0.999
        torch.manual_seed(SEED)
        p = torch.nn.Parameter(_weights(self.N).cuda())
        opt = mod.AdamW8bit_RingBuffer(
            [p], lr=lr, betas=(0.9, beta2), weight_decay=0.0, use_8bit=True,
            schedule_free=True, stochastic_rounding=True,
        )
        opt.train()
        gen = torch.Generator(device="cuda").manual_seed(SEED + 2)
        for _ in range(20):
            p.grad = torch.randn(self.N, generator=gen, device="cuda", dtype=torch.bfloat16)
            opt.step()

        state = opt.state[p]
        snapshot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
        p_snapshot = p.detach().clone()

        qsigned = create_quantization_map(signed=True).cuda()
        qunsigned = create_quantization_map(signed=False).cuda()
        scale = state["absmax_z"].repeat_interleave(BLOCKSIZE)[: self.N]
        codes = state["z"].long()
        z = qsigned[codes] * scale
        v = qunsigned[state["exp_avg_sq"].long()] * \
            state["absmax2"].repeat_interleave(BLOCKSIZE)[: self.N]
        denom = (v / (1 - beta2 ** (opt.k + 1))).sqrt() + 1e-8
        gap = (qsigned[(codes + 1).clamp(max=255)] - qsigned[codes]).abs() * scale

        grad = 0.5 * gap * denom / lr
        extreme = z.abs().view(-1, BLOCKSIZE).argmax(dim=1) + \
            torch.arange(self.N // BLOCKSIZE, device="cuda") * BLOCKSIZE
        grad[extreme] = 0.0
        grad = grad.bfloat16()

        seen = []
        for seed in range(seeds):
            for key, value in snapshot.items():
                if torch.is_tensor(value):
                    state[key].copy_(value)
            p.data.copy_(p_snapshot)
            torch.manual_seed(seed)
            p.grad = grad.clone()
            opt.step()
            seen.append(state["z"].clone())

        invariant = (torch.stack(seen) == seen[0]).all(dim=0)
        per_block = invariant.view(-1, BLOCKSIZE).sum(1).float()

        self.assertTrue(
            bool(invariant[extreme].all()),
            "the element that defines absmax must be stored deterministically",
        )
        self.assertLessEqual(
            per_block.max().item(), 2,
            "more than one element per block is exempt from stochastic rounding",
        )
        self.assertLess(
            per_block.mean().item(), 1.2,
            "more than one element per block is exempt from stochastic rounding",
        )

    def _survival(self, stochastic_rounding: bool, steps: int = 2000):
        """mean|z| after ``steps`` of zero-mean gradients, over its starting value."""
        torch.manual_seed(SEED)
        p = torch.nn.Parameter(_weights(self.N).cuda())
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        opt = mod.AdamW8bit_RingBuffer(
            [p], lr=self.LR, weight_decay=0.0, use_8bit=True,
            schedule_free=True, stochastic_rounding=stochastic_rounding,
        )
        opt.train()
        opt._init_param_state(p)
        state = opt.state[p]
        z0 = _dequantize(state["z"], state["absmax_z"]).abs().mean().item()
        gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
        for _ in range(steps):
            p.grad = torch.randn(self.N, generator=gen, device="cuda", dtype=torch.bfloat16)
            opt.step()
        z1 = _dequantize(state["z"], state["absmax_z"]).abs().mean().item()
        return z1 / z0

    def test_the_z_sequence_survives_a_long_run(self):
        """The scale must not decay (or run away) on its own.

        Before the symmetric headroom, a block with a negative extreme lost
        0.7031% of its scale per step: 0.992969^2000 = 7.7e-7. With headroom but
        stochastic rounding on the element that defines absmax, the same feedback
        ran the other way at +0.63%/step: 1.0063^2000 = 3e5.
        """
        for stochastic_rounding in (False, True):
            with self.subTest(stochastic_rounding=stochastic_rounding):
                ratio = self._survival(stochastic_rounding)
                self.assertGreater(ratio, 0.7, "the z sequence decayed toward zero")
                self.assertLess(ratio, 1.4, "the z sequence ran away")

    def test_a_zero_gradient_leaves_the_scale_alone(self):
        """dequantize -> recompute absmax -> requantize has to be idempotent."""
        torch.manual_seed(SEED)
        p = torch.nn.Parameter(_weights(self.N).cuda())
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        opt = mod.AdamW8bit_RingBuffer(
            [p], lr=self.LR, weight_decay=0.0, use_8bit=True, schedule_free=True,
        )
        opt.train()
        opt._init_param_state(p)
        state = opt.state[p]
        codes0 = state["z"].clone()
        absmax0 = state["absmax_z"].clone()
        for _ in range(500):
            p.grad = torch.zeros_like(p)
            opt.step()

        self.assertTrue(
            torch.equal(codes0, state["z"]),
            "z drifted with no gradient at all",
        )
        self.assertLess(
            (state["absmax_z"] / absmax0 - 1).abs().max().item(), 1e-4,
            "the block scale moved with no gradient at all",
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class LionScheduleFreeRefusalTest(unittest.TestCase):
    """Lion + Schedule-Free is refused, and why.

    ``lion8bit_schedulefree_kernel.cu`` uses z for Lion's momentum EMA and then
    writes ``x = (1 - ckp1) * z + ckp1 * y`` into the parameter. Schedule-Free's z
    is a POSITION sequence, and ckp1 is ~1/k, so the parameter becomes the
    momentum buffer within a few steps: measured with random gradients,
    corr(p, z) = 0.994 at step 5 and 0.9996 at step 20, with mean|p| leaving its
    initial 1.6e-2 for the momentum's scale (and falling to 2.5e-5 under a
    constant gradient). A correct implementation needs a position sequence AND a
    momentum EMA -- a second 8-bit state this mode does not allocate -- so the
    configuration is refused rather than patched.
    """

    N = 1 << 12

    def test_the_constructor_refuses_schedule_free(self):
        import core.training.optimizers.lion8bit_ringbuffer as mod

        p = torch.nn.Parameter(_weights(256).cuda())
        with self.assertRaises(RuntimeError) as ctx:
            mod.Lion8bit_RingBuffer([p], lr=1e-5, use_8bit=True, schedule_free=True)
        self.assertIn("momentum", str(ctx.exception))
        self.assertIn("AdamW8bit_RingBuffer", str(ctx.exception))

    def test_the_refusal_is_not_a_valueerror(self):
        """BaseTrainer catches ValueError from optimizer construction and falls
        back to AdamW -- a ValueError here would substitute an optimizer silently."""
        import core.training.optimizers.lion8bit_ringbuffer as mod

        p = torch.nn.Parameter(_weights(256).cuda())
        try:
            mod.Lion8bit_RingBuffer([p], lr=1e-5, use_8bit=True, schedule_free=True)
        except RuntimeError:
            pass
        except ValueError:  # pragma: no cover - the point of the test
            self.fail("the refusal would be swallowed by BaseTrainer's fallback")

    def test_plain_lion_is_unaffected(self):
        import core.training.optimizers.lion8bit_ringbuffer as mod

        torch.manual_seed(SEED)
        p = torch.nn.Parameter(_weights(self.N).cuda())
        before = p.detach().float().abs().mean().item()
        opt = mod.Lion8bit_RingBuffer([p], lr=1e-5, weight_decay=0.0, use_8bit=True)
        for _ in range(50):
            p.grad = torch.full_like(p, -1.0)
            opt.step()
        after = p.detach().float().abs().mean().item()
        self.assertLess(abs(after - before) / before, 0.5)

    def test_the_kernel_still_writes_the_momentum_into_the_parameter(self):
        """Pins the defect the refusal names, so the refusal cannot outlive it.

        Driven through the kernel directly, since the optimizer refuses to build.
        If someone fixes the kernel, this fails -- and the refusal above (and in
        BaseTrainer) is what they must then remove.
        """
        import core.training.optimizers.lion8bit_ringbuffer as mod
        from core.training.optimizers.adamw8bit_ringbuffer import quantize_blockwise_inplace

        torch.manual_seed(SEED)
        p = torch.nn.Parameter(_weights(self.N).cuda())
        opt = mod.Lion8bit_RingBuffer([p], lr=1e-5, weight_decay=0.0, use_8bit=True)
        blocks = self.N // BLOCKSIZE
        state_z = torch.zeros(self.N, dtype=torch.uint8, device="cuda")
        absmax_z = torch.zeros(blocks, dtype=torch.float32, device="cuda")

        gen = torch.Generator(device="cuda").manual_seed(SEED + 3)
        for step in range(1, 21):
            grad = torch.randn(self.N, generator=gen, device="cuda", dtype=torch.bfloat16)
            opt.ext.lion_8bit_schedulefree_update(
                p, grad, state_z, absmax_z,
                0.9, 0.99, 0.0, 1e-5, 0.0, 1.0 / step, 1.0, False, False, 0,
            )

        z = _dequantize(state_z, absmax_z)
        correlation = torch.corrcoef(torch.stack([p.detach().float().cpu(), z]))[0, 1]
        self.assertGreater(
            correlation.item(), 0.9,
            "the parameter no longer tracks the momentum -- the kernel may be fixed, "
            "in which case lift the Schedule-Free refusals in Lion8bit_RingBuffer "
            "and BaseTrainer.setup_optimizer",
        )

    def test_the_kernels_constant_map_is_still_initialised(self):
        """The per-TU constant-map fix, pinned on the (otherwise unreachable) kernel."""
        import core.training.optimizers.lion8bit_ringbuffer as mod

        torch.manual_seed(SEED)
        p = torch.nn.Parameter(_weights(self.N).cuda())
        opt = mod.Lion8bit_RingBuffer([p], lr=1e-5, weight_decay=0.0, use_8bit=True)
        blocks = self.N // BLOCKSIZE
        state_z = torch.zeros(self.N, dtype=torch.uint8, device="cuda")
        absmax_z = torch.zeros(blocks, dtype=torch.float32, device="cuda")

        grad = torch.full((self.N,), -1.0, dtype=torch.bfloat16, device="cuda")
        opt.ext.lion_8bit_schedulefree_update(
            p, grad, state_z, absmax_z, 0.9, 0.99, 0.0, 1e-5, 0.0, 1.0, 1.0, False, False, 0,
        )
        first = _dequantize(state_z, absmax_z).abs().mean().item()
        opt.ext.lion_8bit_schedulefree_update(
            p, grad, state_z, absmax_z, 0.9, 0.99, 0.0, 1e-5, 0.0, 0.5, 1.0, False, False, 0,
        )
        second = _dequantize(state_z, absmax_z).abs().mean().item()

        # An EMA read back as zero cannot accumulate: it would return to
        # (1-beta2)*|g| on every step instead of growing toward |g|.
        self.assertGreater(
            second, 1.5 * first,
            "the momentum did not accumulate -- the Schedule-Free constant map is not initialised",
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ScheduleFreeResumeGuardTest(unittest.TestCase):
    """Resuming a checkpoint written before the constant map was initialised.

    Such a checkpoint holds a z that decodes to a constant (in the observed case
    every code 255 with absmax_z at ~0). It is not inert: y is lerped toward z
    every step, so the weights are dragged to it -- measured, mean|p| fell from
    1.63e-2 to 5.21e-5 over 300 zero-gradient steps.
    """

    N = 1 << 12

    def _fresh(self):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        torch.manual_seed(SEED)
        p = torch.nn.Parameter(_weights(self.N).cuda())
        opt = mod.AdamW8bit_RingBuffer(
            [p], lr=1e-5, weight_decay=0.0, use_8bit=True, schedule_free=True,
        )
        opt.train()
        p.grad = torch.zeros_like(p)
        opt.step()
        return p, opt

    def _pre_fix_state_dict(self, absmax: float = 1e-5):
        """A checkpoint as the broken kernel actually left it.

        ``absmax_z`` collapsed to ~lr rather than exactly zero (measured: 1.0e-5
        from a 6.8e-2 start), and every code went to 255. The all-zero variant is
        exercised too -- a guard that only recognises one of the two signatures
        misses real checkpoints.
        """
        import copy

        _, opt = self._fresh()
        state_dict = copy.deepcopy(opt.state_dict())
        key = next(iter(state_dict["state"]))
        state_dict["state"][key]["z"] = torch.full((self.N,), 255, dtype=torch.uint8)
        state_dict["state"][key]["absmax_z"] = torch.full((self.N // BLOCKSIZE,), absmax)
        return state_dict

    def test_resuming_a_degenerate_z_does_not_destroy_the_weights(self):
        for absmax in (0.0, 1e-5):
            with self.subTest(absmax_z=absmax):
                p, opt = self._fresh()
                before = p.detach().float().abs().mean().item()

                opt.load_state_dict(self._pre_fix_state_dict(absmax))
                for _ in range(300):
                    p.grad = torch.zeros_like(p)
                    opt.step()

                after = p.detach().float().abs().mean().item()
                self.assertGreater(
                    after, 0.5 * before,
                    "resuming a pre-fix Schedule-Free checkpoint pulled the weights to zero",
                )

    def test_a_zero_scale_is_repaired_even_when_the_codes_vary(self):
        """The second signature, on its own.

        ``absmax_z`` all zero decodes z to zero whatever the codes say, which is
        just as destructive and is not covered by the constant-code test.
        """
        import copy

        _, donor = self._fresh()
        state_dict = copy.deepcopy(donor.state_dict())
        key = next(iter(state_dict["state"]))
        state_dict["state"][key]["z"] = torch.randint(
            0, 256, (self.N,), dtype=torch.uint8
        )
        state_dict["state"][key]["absmax_z"] = torch.zeros(self.N // BLOCKSIZE)

        p, opt = self._fresh()
        before = p.detach().float().abs().mean().item()
        opt.load_state_dict(state_dict)
        for _ in range(300):
            p.grad = torch.zeros_like(p)
            opt.step()
        self.assertGreater(p.detach().float().abs().mean().item(), 0.5 * before)

    def test_the_repaired_z_is_the_parameter(self):
        """z_0 = p is the Schedule-Free initial condition."""
        p, opt = self._fresh()
        opt.load_state_dict(self._pre_fix_state_dict())
        state = opt.state[p]
        z = _dequantize(state["z"], state["absmax_z"]).cuda()
        self.assertLess(
            (z - p.detach().float()).abs().mean().item(),
            0.05 * p.detach().float().abs().mean().item(),
        )

    def test_a_healthy_checkpoint_is_left_alone(self):
        """The guard must not fire on state it is not meant to repair."""
        import copy

        _, donor = self._fresh()
        for _ in range(20):
            donor.param_groups[0]["params"][0].grad = torch.full(
                (self.N,), -1.0, dtype=torch.bfloat16, device="cuda"
            )
            donor.step()
        healthy = copy.deepcopy(donor.state_dict())
        key = next(iter(healthy["state"]))

        p, opt = self._fresh()
        opt.load_state_dict(healthy)
        self.assertTrue(
            torch.equal(opt.state[p]["z"].cpu(), healthy["state"][key]["z"].cpu()),
            "the guard rewrote a healthy z",
        )

    def test_a_constant_valued_parameter_is_not_repaired(self):
        """An all-ones RMSNorm weight, a zero-init LoRA B, a zero bias.

        Their z legitimately IS a single repeated code, so "every code identical"
        alone would fire the repair -- and its loud warning -- on hundreds of
        healthy tensors in any early checkpoint. The decoded z also has to
        disagree with the parameter.
        """
        import copy

        import core.training.optimizers.adamw8bit_ringbuffer as mod

        import contextlib
        import io

        for label, value in (("ones (RMSNorm)", 1.0), ("zeros (bias / LoRA B)", 0.0)):
            with self.subTest(parameter=label):
                p = torch.nn.Parameter(
                    torch.full((self.N,), value, dtype=torch.bfloat16, device="cuda")
                )
                opt = mod.AdamW8bit_RingBuffer(
                    [p], lr=1e-5, weight_decay=0.0, use_8bit=True, schedule_free=True,
                )
                opt.train()
                p.grad = torch.zeros_like(p)
                opt.step()
                healthy = copy.deepcopy(opt.state_dict())
                key = next(iter(healthy["state"]))

                # The repair is a no-op on these (it would write back the z they
                # already hold), so comparing state cannot see it fire -- what a
                # user sees is the warning, once per tensor, on a checkpoint with
                # nothing wrong with it.
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    opt.load_state_dict(healthy)
                self.assertNotIn(
                    "Re-seeded z", log.getvalue(),
                    f"the guard fired on a healthy constant parameter ({label})",
                )
                self.assertTrue(
                    torch.equal(opt.state[p]["z"].cpu(), healthy["state"][key]["z"].cpu()),
                    f"the guard rewrote the z of a healthy constant parameter ({label})",
                )

    def test_the_second_moment_and_counters_survive_the_repair(self):
        """Repairing z must not throw away the rest of the resumed state."""
        p, opt = self._fresh()
        state_dict = self._pre_fix_state_dict()
        state_dict["k"] = 4321
        key = next(iter(state_dict["state"]))
        marker = state_dict["state"][key]["exp_avg_sq"].clone()

        opt.load_state_dict(state_dict)
        self.assertEqual(opt.k, 4321)
        self.assertTrue(torch.equal(opt.state[p]["exp_avg_sq"].cpu(), marker.cpu()))


if __name__ == "__main__":
    unittest.main()
