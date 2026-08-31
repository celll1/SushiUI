"""Guard: stratified MNT timesteps must not change the sampled DISTRIBUTION.

Why this file exists
--------------------
An MNT window is a `multi_noise_timesteps`-sample Monte-Carlo estimate of an
integral over t on ONE image. At batch size 1 the timestep draw is very nearly
the only source of within-window variance, so stratifying it is the textbook
variance reduction. The whole value of the change rests on it being *unbiased*:
if it moved the marginal density it would silently retune the architecture's
timestep distribution, which for SenseNova is upstream's own pretraining
setting (logit-normal mu=-0.8, sigma=0.8) and must not drift.

So these tests pin two things:
  1. every stratified draw has the SAME marginal law as the IID draw it
     replaces (quantile agreement, per sampler);
  2. it actually reduces the variance of the window mean.

Plus the operational details that are easy to get wrong: the stratum ORDER must
be permuted (MNT iteration 0 is not an ordinary iteration -- SenseNova reuses
the batch prefix there and debug latents are saved there, so a monotone order
would bind those to the noisiest stratum forever), and a sampler without a
closed-form quantile must fall back instead of raising into the training loop.
"""

from __future__ import annotations

import os
import statistics
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import BaseTrainer
from core.training.timestep_sampler import (
    BetaTimestepSampler,
    CustomTimestepSampler,
    LogitNormalTimestepSampler,
    NormalTimestepSampler,
    TimestepSampler,
    UniformTimestepSampler,
)

CPU = torch.device("cpu")

# The SenseNova production setting, plus the other quantile-bearing samplers.
SAMPLERS = {
    "uniform": UniformTimestepSampler(0.0, 1.0),
    "logit_normal(-0.8,0.8)": LogitNormalTimestepSampler(0.0, 1.0, -0.8, 0.8),
    "logit_normal(0,1)": LogitNormalTimestepSampler(0.0, 1.0, 0.0, 1.0),
    "normal": NormalTimestepSampler(0.0, 1.0, 0.5, 0.2),
    "custom": CustomTimestepSampler(0.0, 0.85, [3.0, 2.0, 1.5, 1.0, 0.5]),
}


def _quantiles(values, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    s = sorted(values)
    return [s[int(p * (len(s) - 1))] for p in ps]


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_stratified_marginal_matches_iid(name):
    """The estimator must stay unbiased: same law, not merely a similar range."""
    sampler = SAMPLERS[name]
    torch.manual_seed(0)
    iid = torch.cat([sampler.sample(256, CPU) for _ in range(80)]).tolist()
    strat = torch.cat(
        [sampler.sample_stratified(16, 16, CPU).flatten() for _ in range(80)]
    ).tolist()

    for q_iid, q_str in zip(_quantiles(iid), _quantiles(strat)):
        assert q_str == pytest.approx(q_iid, abs=0.02), name
    assert statistics.mean(strat) == pytest.approx(statistics.mean(iid), abs=0.01)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_icdf_is_the_inverse_of_the_sampled_cdf(name):
    """Direct check of the quantile function against the empirical CDF."""
    sampler = SAMPLERS[name]
    torch.manual_seed(1)
    drawn = sorted(sampler.sample(200000, CPU).tolist())
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        empirical = drawn[int(p * (len(drawn) - 1))]
        analytic = sampler.icdf(torch.tensor([p])).item()
        assert analytic == pytest.approx(empirical, abs=0.01), f"{name} @ p={p}"


def test_stratification_reduces_window_variance():
    """The point of the change. 16 strata vs 16 IID draws, same marginal."""
    sampler = SAMPLERS["logit_normal(-0.8,0.8)"]
    torch.manual_seed(2)

    def window_means(stratified):
        out = []
        for _ in range(400):
            if stratified:
                t = sampler.sample_stratified(16, 1, CPU).flatten()
            else:
                t = sampler.sample(16, CPU)
            out.append(t.mean().item())
        return statistics.pvariance(out)

    v_strat = window_means(True)
    v_iid = window_means(False)
    assert v_strat < v_iid / 3, f"stratified={v_strat:.3e} iid={v_iid:.3e}"


def test_each_stratum_appears_exactly_once():
    """A permuted order is still a permutation -- no stratum lost or doubled."""
    sampler = SAMPLERS["uniform"]
    torch.manual_seed(3)
    for _ in range(50):
        t = sampler.sample_stratified(16, 1, CPU).flatten().tolist()
        buckets = sorted(min(int(x * 16), 15) for x in t)
        assert buckets == list(range(16))


def test_stratum_order_is_permuted():
    """MNT iteration 0 must not be permanently bound to the lowest stratum:
    SenseNova reuses the batch prefix there and debug latents are saved there."""
    sampler = SAMPLERS["uniform"]
    torch.manual_seed(4)
    firsts = [sampler.sample_stratified(16, 1, CPU)[0].item() for _ in range(200)]
    assert max(firsts) > 0.75, "row 0 never drew a high stratum -- order is sorted"
    assert min(firsts) < 0.25
    # And row 0 should look like a plain draw from the marginal.
    assert statistics.mean(firsts) == pytest.approx(0.5, abs=0.06)


def test_beta_has_no_quantile_and_says_so():
    """torch.distributions.Beta has no icdf; the trainer must fall back, not die."""
    with pytest.raises(NotImplementedError):
        BetaTimestepSampler(0.0, 1.0, 2.0, 2.0).sample_stratified(16, 1, CPU)


def test_min_max_range_is_respected():
    sampler = LogitNormalTimestepSampler(0.2, 0.8, -0.8, 0.8)
    torch.manual_seed(5)
    t = sampler.sample_stratified(16, 4, CPU)
    assert t.shape == (16, 4)
    assert float(t.min()) >= 0.2 and float(t.max()) <= 0.8


# ---------------------------------------------------------------- trainer glue

class Harness:
    _stratified_mnt_timesteps = BaseTrainer._stratified_mnt_timesteps

    def __init__(self, config=None):
        self.config = config if config is not None else {}
        self.log_prefix = "[Test]"
        self.device = CPU


def test_trainer_returns_none_when_mnt_is_one():
    """MNT=1 has nothing to stratify; the per-iteration draw stands."""
    h = Harness()
    assert h._stratified_mnt_timesteps(SAMPLERS["uniform"], 1, 1) is None


def test_trainer_returns_block_shaped_for_the_mnt_loop():
    h = Harness()
    torch.manual_seed(6)
    block = h._stratified_mnt_timesteps(SAMPLERS["logit_normal(-0.8,0.8)"], 16, 1)
    assert block is not None and block.shape == (16, 1)
    # The loop indexes block[mnt_idx] and hands it straight to train_step.
    assert block[0].shape == (1,)


def test_trainer_honours_the_config_switch():
    h = Harness({"stratified_timesteps": False})
    assert h._stratified_mnt_timesteps(SAMPLERS["uniform"], 16, 1) is None


def test_trainer_default_is_on_and_comes_from_param_defaults():
    from api.param_defaults import TRAINING_DEFAULTS

    assert TRAINING_DEFAULTS["stratified_timesteps"] is True
    h = Harness({})
    assert h._stratified_mnt_timesteps(SAMPLERS["uniform"], 16, 1) is not None


def test_trainer_falls_back_for_beta_without_raising():
    """A missing quantile costs variance, not the run."""
    h = Harness()
    assert h._stratified_mnt_timesteps(BetaTimestepSampler(0.0, 1.0, 2.0, 2.0), 16, 1) is None
    assert h._stratified_timesteps_warned is True


def test_trainer_swallows_an_unexpected_sampler_failure():
    class Broken(TimestepSampler):
        def sample(self, batch_size, device):
            raise AssertionError("unused")

        def icdf(self, u):
            raise RuntimeError("boom")

    h = Harness()
    assert h._stratified_mnt_timesteps(Broken(0.0, 1.0), 16, 1) is None
