"""Round-trip coverage for ``ops/x0_recovery.predict_x0``.

x0 -> z_t (real scheduler) -> predict_x0 -> x0, for every (noise_process,
prediction_target, velocity_sign) combination the repo trains, plus the
non-obvious properties the later aux-loss phases depend on: gradient survives,
fp16/bf16/fp32 stay finite, unsupported values raise.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/x0_recovery_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402
from diffusers import DDPMScheduler  # noqa: E402

from core.training.base_trainer import add_noise_unified, get_target_unified  # noqa: E402
from core.training.ops.x0_recovery import predict_x0, snr_band_mask  # noqa: E402

# Flow: t=1 is pure noise, so eps-prediction recovery divides by (1-t); 0.9 caps
# the amplification at 10x and keeps this an accuracy test, not a conditioning one.
FLOW_TIMESTEPS = [0.05, 0.25, 0.5, 0.75, 0.9]
DDPM_TIMESTEPS = [0, 250, 500, 750, 999]


def _fixtures(dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(5, 4, 8, 8, generator=g, dtype=torch.float32).to(dtype)
    noise = torch.randn(5, 4, 8, 8, generator=g, dtype=torch.float32).to(dtype)
    return x0, noise


def _timesteps(noise_process, dtype=torch.float32):
    if noise_process == "flow":
        return torch.tensor(FLOW_TIMESTEPS, dtype=dtype)
    return torch.tensor(DDPM_TIMESTEPS, dtype=torch.long)


class PredictX0RoundTripTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    def _roundtrip(self, noise_process, prediction_target, velocity_sign):
        x0, noise = _fixtures()
        t = _timesteps(noise_process)
        z = add_noise_unified(noise_process, self.scheduler, x0, noise, t)
        target = get_target_unified(
            noise_process, prediction_target, self.scheduler, x0, noise, t
        )
        if prediction_target == "velocity" and velocity_sign == "x0_minus_eps":
            target = -target
        recovered = predict_x0(
            noise_process=noise_process,
            prediction_target=prediction_target,
            noisy_latents=z,
            model_pred=target,
            timesteps=t,
            noise_scheduler=self.scheduler,
            velocity_sign=velocity_sign,
        )
        self.assertTrue(
            torch.allclose(recovered.float(), x0.float(), atol=1e-4, rtol=1e-3),
            f"{noise_process}/{prediction_target}/{velocity_sign}: "
            f"max err {(recovered.float() - x0.float()).abs().max().item():.3e}",
        )

    def test_flow(self):
        for prediction_target in ("epsilon", "velocity", "sample"):
            for velocity_sign in ("eps_minus_x0", "x0_minus_eps"):
                with self.subTest(target=prediction_target, sign=velocity_sign):
                    self._roundtrip("flow", prediction_target, velocity_sign)

    def test_ddpm(self):
        for prediction_target in ("epsilon", "velocity", "sample"):
            for velocity_sign in ("eps_minus_x0", "x0_minus_eps"):
                with self.subTest(target=prediction_target, sign=velocity_sign):
                    self._roundtrip("ddpm", prediction_target, velocity_sign)

    def test_velocity_signs_are_not_interchangeable(self):
        """A wrong sign must be visibly wrong, not a rounding difference."""
        x0, noise = _fixtures()
        t = _timesteps("flow")
        z = add_noise_unified("flow", self.scheduler, x0, noise, t)
        v = get_target_unified("flow", "velocity", self.scheduler, x0, noise, t)
        standard = predict_x0(
            noise_process="flow", prediction_target="velocity", noisy_latents=z,
            model_pred=v, timesteps=t, velocity_sign="eps_minus_x0",
        )
        flipped = predict_x0(
            noise_process="flow", prediction_target="velocity", noisy_latents=z,
            model_pred=v, timesteps=t, velocity_sign="x0_minus_eps",
        )
        self.assertGreater((standard - flipped).abs().max().item(), 1e-2)


class PredictX0GradientTest(unittest.TestCase):
    """The aux-loss phases put this on the autograd path; no detach/no_grad inside."""

    def test_gradient_reaches_model_pred_and_noisy_latents(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        for noise_process, prediction_target in (
            ("flow", "epsilon"), ("flow", "velocity"),
            ("ddpm", "epsilon"), ("ddpm", "velocity"),
        ):
            with self.subTest(process=noise_process, target=prediction_target):
                x0, noise = _fixtures()
                t = _timesteps(noise_process)
                z = add_noise_unified(noise_process, scheduler, x0, noise, t).requires_grad_(True)
                pred = torch.zeros_like(x0, requires_grad=True)
                out = predict_x0(
                    noise_process=noise_process, prediction_target=prediction_target,
                    noisy_latents=z, model_pred=pred, timesteps=t,
                    noise_scheduler=scheduler, velocity_sign="eps_minus_x0",
                )
                out.square().mean().backward()
                self.assertIsNotNone(pred.grad)
                self.assertGreater(pred.grad.abs().sum().item(), 0.0)
                self.assertIsNotNone(z.grad)
                self.assertGreater(z.grad.abs().sum().item(), 0.0)


class PredictX0DtypeTest(unittest.TestCase):
    def test_production_schedule_never_reaches_the_alpha_floor(self):
        """Pins the 6.8e-2 the clamp_min(1e-3) comment in base_trainer cites."""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear",
        )
        self.assertGreater(scheduler.alphas_cumprod.sqrt().min().item(), 1e-3)

    def test_finite_in_every_training_dtype(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for noise_process in ("flow", "ddpm"):
                for prediction_target in ("epsilon", "velocity", "sample"):
                    with self.subTest(dtype=dtype, process=noise_process, target=prediction_target):
                        x0, noise = _fixtures(dtype=dtype)
                        t = _timesteps(noise_process, dtype=dtype)
                        z = add_noise_unified(noise_process, scheduler, x0, noise, t)
                        target = get_target_unified(
                            noise_process, prediction_target, scheduler, x0, noise, t
                        )
                        out = predict_x0(
                            noise_process=noise_process, prediction_target=prediction_target,
                            noisy_latents=z, model_pred=target.to(dtype), timesteps=t,
                            noise_scheduler=scheduler, velocity_sign="eps_minus_x0",
                        )
                        self.assertTrue(torch.isfinite(out).all())

    def test_zero_terminal_snr_stays_finite(self):
        """alpha_bar_T is exactly 0 under rescale_betas_zero_snr; the guard bounds it."""
        scheduler = DDPMScheduler(num_train_timesteps=1000, rescale_betas_zero_snr=True)
        self.assertEqual(float(scheduler.alphas_cumprod[-1]), 0.0)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x0, noise = _fixtures(dtype=dtype)
                t = torch.full((5,), 999, dtype=torch.long)
                z = add_noise_unified("ddpm", scheduler, x0, noise, t)
                out = predict_x0(
                    noise_process="ddpm", prediction_target="epsilon",
                    noisy_latents=z, model_pred=noise, timesteps=t,
                    noise_scheduler=scheduler,
                )
                self.assertTrue(torch.isfinite(out).all())


class UnsupportedValueTest(unittest.TestCase):
    def test_raises(self):
        x0, noise = _fixtures()
        t = _timesteps("flow")
        base = dict(noisy_latents=x0, model_pred=noise, timesteps=t)
        with self.assertRaises(ValueError):
            predict_x0(noise_process="edm", prediction_target="epsilon", **base)
        with self.assertRaises(ValueError):
            predict_x0(noise_process="flow", prediction_target="score", **base)
        with self.assertRaises(ValueError):
            predict_x0(
                noise_process="flow", prediction_target="velocity",
                velocity_sign="whichever", **base,
            )
        with self.assertRaises(ValueError):
            # Omitting the sign for a velocity target must raise, not pick one:
            # the wrong sign is silent, off by 2*t*v.
            predict_x0(noise_process="flow", prediction_target="velocity", **base)
        with self.assertRaises(ValueError):
            # ddpm without a scheduler has no alphas_cumprod to read.
            predict_x0(
                noise_process="ddpm", prediction_target="epsilon",
                noisy_latents=x0, model_pred=noise, timesteps=_timesteps("ddpm"),
            )


class SnrBandMaskTest(unittest.TestCase):
    def test_flow_band(self):
        # flow SNR = ((1-t)/t)^2: t=0.5 -> 1.0, t=0.25 -> 9.0, t=0.75 -> 1/9.
        t = torch.tensor([0.25, 0.5, 0.75])
        mask = snr_band_mask("flow", t, snr_min=0.5, snr_max=2.0)
        self.assertEqual(mask.tolist(), [False, True, False])
        self.assertEqual(snr_band_mask("flow", t, snr_min=0.5).tolist(), [True, True, False])
        self.assertEqual(snr_band_mask("flow", t).tolist(), [True, True, True])

    def test_ddpm_band_is_monotone_in_timestep(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        t = torch.tensor(DDPM_TIMESTEPS, dtype=torch.long)
        mask = snr_band_mask("ddpm", t, noise_scheduler=scheduler, snr_min=1.0)
        # SNR falls monotonically with the ddpm timestep, so a min-SNR band keeps a prefix.
        self.assertEqual(mask.tolist(), sorted(mask.tolist(), reverse=True))
        self.assertTrue(mask[0].item())
        self.assertFalse(mask[-1].item())


if __name__ == "__main__":
    unittest.main()
