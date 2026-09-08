"""``predict_x0`` must reproduce the inline x_0 it replaced, bit for bit.

Z-Image's four inline copies were migrated because their expression is the
helper's expression, fp32 promotion included. The archs that were NOT migrated
(anima / acestep / ltx2 and the crop-decode pre-computations in
flux2 / ideogram4 / krea2 / lens) form their sigma view in the *model* dtype, so
the product rounds to bf16 before the subtraction; the second test pins that
difference so a later reader does not "finish the job" and silently move a run.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/x0_recovery_call_site_identity_test.py -q
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.ops.x0_recovery import predict_x0  # noqa: E402
from core.training.ops.zimage_ops import _predict_x0_zimage  # noqa: E402

DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _fixtures(dtype):
    g = torch.Generator().manual_seed(7)
    z = torch.randn(3, 4, 8, 8, generator=g).to(dtype)
    pred = torch.randn(3, 4, 8, 8, generator=g).to(dtype)
    t = torch.tensor([0.05, 0.5, 0.93], dtype=torch.float32)
    return z, pred, t


class ZImageIdentityTest(unittest.TestCase):
    """Pre-migration Z-Image expression (git 62080e47:ops/zimage_ops.py)."""

    @staticmethod
    def _legacy(noisy_latents, model_pred, timesteps):
        t = timesteps.float()
        while t.dim() < noisy_latents.dim():
            t = t.unsqueeze(-1)
        return noisy_latents + t * model_pred

    def test_bit_identical_in_every_training_dtype(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                z, pred, t = _fixtures(dtype)
                legacy = self._legacy(z, pred, t)
                migrated = _predict_x0_zimage(z, pred, t)
                self.assertEqual(legacy.dtype, migrated.dtype)
                self.assertTrue(torch.equal(legacy, migrated))

    def test_sign_is_the_inverted_one(self):
        z, pred, t = _fixtures(torch.float32)
        wrong = predict_x0(
            noise_process="flow", prediction_target="velocity", noisy_latents=z,
            model_pred=pred, timesteps=t, velocity_sign="eps_minus_x0",
        )
        self.assertFalse(torch.equal(wrong, _predict_x0_zimage(z, pred, t)))


class InModelDtypeCallSiteIsNotIdenticalTest(unittest.TestCase):
    """Why anima/acestep/ltx2 (and the crop-decode pre-computations) stayed inline."""

    def test_bf16_sigma_view_differs_from_the_fp32_helper(self):
        z, pred, t = _fixtures(torch.bfloat16)
        sigma_view = t.view(-1, 1, 1, 1).to(z.dtype)
        legacy = z - sigma_view * pred
        migrated = predict_x0(
            noise_process="flow", prediction_target="velocity", noisy_latents=z,
            model_pred=pred, timesteps=t, velocity_sign="eps_minus_x0",
        )
        self.assertEqual(legacy.dtype, torch.bfloat16)
        self.assertEqual(migrated.dtype, torch.float32)
        self.assertFalse(torch.equal(legacy.float(), migrated))


if __name__ == "__main__":
    unittest.main()
