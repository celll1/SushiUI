"""RNG-safety of the timestep-distribution-median diagnostic log line.

Addendum to 61e0b415: the diagnostic probe added to base_trainer.train()
originally called `timestep_sampler.sample(4096, ...)` unguarded, consuming
4096 draws from the process-shared torch CPU RNG at training startup -- a
side effect on the training random sequence, which this repo has specifically
worked to keep reproducible (batch-shuffle RNG fix). This pins that the probe
(a) never perturbs `torch.random.get_rng_state()` and (b) never raises out of
training even if the sampler implementation is broken.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/timestep_diagnostic_rng_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402

from core.training.base_trainer import log_timestep_distribution_median  # noqa: E402
from core.training.timestep_sampler import UniformTimestepSampler  # noqa: E402


class _RaisingSampler:
    def sample(self, batch_size, device):
        raise RuntimeError("simulated sampler failure")


class TimestepDiagnosticRngTest(unittest.TestCase):
    def test_cpu_rng_state_unchanged(self):
        sampler = UniformTimestepSampler(min_timestep=0.0, max_timestep=1.0)
        torch.manual_seed(42)
        before = torch.random.get_rng_state()
        log_timestep_distribution_median("[test]", sampler, "t0")
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))

    def test_rng_stream_after_probe_matches_stream_without_probe(self):
        # Stronger than state-equality alone: prove a draw taken AFTER the
        # probe is identical to a draw taken with no probe at all.
        sampler = UniformTimestepSampler(min_timestep=0.0, max_timestep=1.0)

        torch.manual_seed(7)
        log_timestep_distribution_median("[test]", sampler, "t0")
        draw_with_probe = torch.rand(4)

        torch.manual_seed(7)
        draw_without_probe = torch.rand(4)

        self.assertTrue(torch.equal(draw_with_probe, draw_without_probe))

    def test_sampler_exception_does_not_propagate(self):
        # Must not raise -- a broken diagnostic must not fail training.
        try:
            log_timestep_distribution_median("[test]", _RaisingSampler(), "t0")
        except Exception as e:  # pragma: no cover - failure path under test
            self.fail(f"diagnostic probe raised out of the log helper: {e}")

    def test_sampler_exception_leaves_rng_state_untouched(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        log_timestep_distribution_median("[test]", _RaisingSampler(), "t0")
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
