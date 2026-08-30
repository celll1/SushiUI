"""Per-architecture timestep convention (t=0 clean vs t=1 clean).

Regression coverage for the confirmed inconsistency: `timestep_sampler.py`
documented only the "t=0 clean" convention, but SenseNova (and MiniT2I, and
SD1.5/SDXL under noise_process="ddpm") use the inverse ("t=1 clean"). This
file pins:

  (a) every ArchHandler's declared `resolve_timestep_convention()` against a
      literal transcription of its `train_step` noise-mixing formula (not the
      convention constant itself -- so a typo in the constant cannot pass by
      agreeing with itself), and
  (b) that adding the convention constant/logging did not touch the sampling
      MATH: TimestepSampler outputs are byte-identical (same seed) to a
      reference re-implementation of the pre-existing formulas.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/timestep_convention_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402

from core.training.arch.acestep import AceStepArchHandler  # noqa: E402
from core.training.arch.anima import AnimaArchHandler  # noqa: E402
from core.training.arch.flux2 import Flux2ArchHandler  # noqa: E402
from core.training.arch.ideogram4 import Ideogram4ArchHandler  # noqa: E402
from core.training.arch.krea2 import Krea2ArchHandler  # noqa: E402
from core.training.arch.lens import LensArchHandler  # noqa: E402
from core.training.arch.ltx2 import Ltx2ArchHandler  # noqa: E402
from core.training.arch.minimax_h3 import MiniMaxH3ArchHandler  # noqa: E402
from core.training.arch.minit2i import MiniT2IArchHandler  # noqa: E402
from core.training.arch.sd15 import SD15ArchHandler  # noqa: E402
from core.training.arch.sdxl import SDXLArchHandler  # noqa: E402
from core.training.arch.sensenova import SenseNovaArchHandler  # noqa: E402
from core.training.arch.zimage import ZImageArchHandler  # noqa: E402
from core.training.timestep_sampler import (  # noqa: E402
    LogitNormalTimestepSampler,
    NormalTimestepSampler,
    UniformTimestepSampler,
)


class ArchTimestepConventionTest(unittest.TestCase):
    """(a): declared convention == the arch's real noise-mixing formula.

    Each expected value is transcribed directly from the train_step formula
    cited in the ArchHandler source comment (file:line noted per case), not
    copied from the constant under test.
    """

    def test_t0_clean_archs(self):
        # noisy = (1-t)*latents + t*noise -- t=0 is clean, for every one of
        # these (acestep_ops.py:590, anima_ops.py:391, add_noise_unified's
        # "flow" branch used by flux2_ops.py/zimage_ops.py, ideogram4_ops.py:285,
        # krea2_ops.py:251, lens_ops.py:311, ltx2_ops.py:607,
        # minimax_h3_ops.py:31).
        t0_handlers = [
            AceStepArchHandler(), AnimaArchHandler(), Flux2ArchHandler(),
            Ideogram4ArchHandler(), Krea2ArchHandler(), LensArchHandler(),
            Ltx2ArchHandler(), MiniMaxH3ArchHandler(), ZImageArchHandler(),
        ]
        for handler in t0_handlers:
            with self.subTest(arch=handler.name):
                self.assertEqual(handler.resolve_timestep_convention(None), "t0")
                self.assertEqual(handler.timestep_convention, "t0")

    def test_t1_clean_archs(self):
        # SenseNova: z_image = t*x0 + (1-t)*noise (sensenova_ops.py:1638) -- t=1
        # is clean.
        # MiniT2I: x_t = images*t + noise*(1-t) (minit2i_ops.py:319, explicitly
        # documented in-line as "t=1 data, t=0 noise") -- t=1 is clean.
        t1_handlers = [SenseNovaArchHandler(), MiniT2IArchHandler()]
        for handler in t1_handlers:
            with self.subTest(arch=handler.name):
                self.assertEqual(handler.resolve_timestep_convention(None), "t1")
                self.assertEqual(handler.timestep_convention, "t1")

    def test_sd_sdxl_convention_depends_on_noise_process(self):
        # ops/sd_sdxl_ops.py train_step: ddpm scales the sampler draw via
        # (1-t)*num_train_timesteps (t=0 -> near-T = noisy, t=1 -> near-0 =
        # clean) => "t1"; flow feeds t straight into add_noise_unified's flow
        # branch (t=0 clean, t=1 noise) => "t0".
        for handler in (SD15ArchHandler(), SDXLArchHandler()):
            with self.subTest(arch=handler.name):
                trainer_ddpm = SimpleNamespace(noise_process="ddpm")
                trainer_flow = SimpleNamespace(noise_process="flow")
                self.assertEqual(handler.resolve_timestep_convention(trainer_ddpm), "t1")
                self.assertEqual(handler.resolve_timestep_convention(trainer_flow), "t0")
                # No trainer / no noise_process attr: matches ops/sd_sdxl_ops.py's
                # own fallback default ("ddpm", for backward compatibility).
                self.assertEqual(handler.resolve_timestep_convention(None), "t1")
                self.assertEqual(
                    handler.resolve_timestep_convention(SimpleNamespace()), "t1"
                )


class TimestepSamplerMathUnchangedTest(unittest.TestCase):
    """(b): the sampling MATH is untouched by the docs/logging change.

    Re-implements each sampler's formula independently (not by calling the
    sampler class) and checks bit-for-bit agreement under a fixed seed, so a
    future accidental edit to the sampler body -- e.g. "fixing" the sign to
    match one architecture's convention -- is caught even though it would not
    change any docstring.
    """

    def test_uniform_unchanged(self):
        torch.manual_seed(1234)
        sampler = UniformTimestepSampler(min_timestep=0.1, max_timestep=0.9)
        actual = sampler.sample(8, torch.device("cpu"))

        torch.manual_seed(1234)
        raw = torch.rand(8, device=torch.device("cpu"))
        expected = raw * (0.9 - 0.1) + 0.1
        self.assertTrue(torch.equal(actual, expected))

    def test_logit_normal_unchanged(self):
        torch.manual_seed(5678)
        sampler = LogitNormalTimestepSampler(min_timestep=0.0, max_timestep=1.0, mean=-0.8, std=0.8)
        actual = sampler.sample(16, torch.device("cpu"))

        torch.manual_seed(5678)
        u = torch.randn(16, device=torch.device("cpu")) * 0.8 + (-0.8)
        expected = torch.sigmoid(u)
        self.assertTrue(torch.equal(actual, expected))

    def test_normal_unchanged(self):
        torch.manual_seed(999)
        sampler = NormalTimestepSampler(min_timestep=0.0, max_timestep=1.0, mean=0.5, std=0.2)
        actual = sampler.sample(16, torch.device("cpu"))

        torch.manual_seed(999)
        raw = torch.randn(16, device=torch.device("cpu")) * 0.2 + 0.5
        expected = torch.clamp(raw, 0.0, 1.0)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
