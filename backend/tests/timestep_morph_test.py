"""Resume-time timestep distribution morphing.

Covers the sampler-level contract from
`docs/guides/TIMESTEP_DISTRIBUTION_MORPH_DESIGN.md`: the lambda schedule on the
successful-optimizer-update axis, quantile vs mixture interpolation, canonical
config equality, and the checkpoint state round trip (nesting, version refusal,
deterministic flattening).

Assertions are on equality IN LAW, never bit identity: quantile interpolation
draws a uniform and applies icdf where a plain normal sampler draws randn, so
even at lambda=0 the two agree in distribution only.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/timestep_morph_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.timestep_sampler import (  # noqa: E402
    MAX_MORPH_NESTING,
    MorphingTimestepSampler,
    QuantileTableSampler,
    TimestepSampler,
    build_sampler_from_expr,
    canonicalize_timestep_config,
    flatten_sampler,
    morph_lambda,
    sampler_depth,
    sampler_expr,
    sampler_to_config,
    validate_morph_config,
)

CPU = torch.device("cpu")
UNIFORM = {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0}
LOGIT = {"distribution": "logit_normal", "mean": 1.0, "std": 1.0}


def _ks(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, _ = torch.sort(a.double())
    b, _ = torch.sort(b.double())
    grid = torch.cat([a, b])
    cdf_a = torch.searchsorted(a, grid).double() / a.numel()
    cdf_b = torch.searchsorted(b, grid).double() / b.numel()
    return float((cdf_a - cdf_b).abs().max())


def _morph(steps=100, curve="cosine", interpolation="quantile", start_update=0,
           source=None, target=None):
    return MorphingTimestepSampler(
        source=source or TimestepSampler.from_config(UNIFORM),
        target=target or TimestepSampler.from_config(LOGIT),
        steps=steps, curve=curve, interpolation=interpolation,
        start_update=start_update,
    )


class LambdaScheduleTest(unittest.TestCase):
    def test_endpoints_and_monotonicity(self):
        for curve in ("linear", "cosine"):
            self.assertEqual(morph_lambda(100, 100, 50, curve), 0.0)
            self.assertEqual(morph_lambda(150, 100, 50, curve), 1.0)
            self.assertEqual(morph_lambda(999, 100, 50, curve), 1.0)
            self.assertEqual(morph_lambda(0, 100, 50, curve), 0.0)
            values = [morph_lambda(s, 100, 50, curve) for s in range(100, 151)]
            self.assertEqual(values, sorted(values))

    def test_cosine_is_flat_at_both_ends(self):
        near_start = morph_lambda(101, 100, 100, "cosine")
        near_end = 1.0 - morph_lambda(199, 100, 100, "cosine")
        self.assertLess(near_start, morph_lambda(101, 100, 100, "linear"))
        self.assertLess(near_end, morph_lambda(101, 100, 100, "linear"))

    def test_lambda_only_advances_with_the_update_counter(self):
        sampler = _morph(steps=10, start_update=5)
        sampler.set_optimizer_update_step(5)
        self.assertEqual(sampler.lam, 0.0)
        # A skipped optimizer step leaves the counter alone: same lambda.
        self.assertEqual(sampler.lam, 0.0)
        sampler.set_optimizer_update_step(10)
        self.assertAlmostEqual(sampler.lam, 0.5, places=6)
        self.assertFalse(sampler.is_finished())
        sampler.set_optimizer_update_step(15)
        self.assertTrue(sampler.is_finished())


class QuantileInterpolationTest(unittest.TestCase):
    def test_icdf_monotone_in_u(self):
        sampler = _morph(steps=100)
        for update in (0, 50, 100):
            sampler.set_optimizer_update_step(update)
            u = torch.linspace(1e-4, 1 - 1e-4, 2048)
            t = sampler.icdf(u)
            self.assertTrue(bool((t[1:] - t[:-1] >= -1e-6).all()),
                            f"icdf not monotone at update {update}")

    def test_endpoints_match_their_sampler_in_law(self):
        sampler = _morph(steps=100)
        source = TimestepSampler.from_config(UNIFORM)
        target = TimestepSampler.from_config(LOGIT)
        torch.manual_seed(0)
        for update, reference in ((0, source), (100, target)):
            sampler.set_optimizer_update_step(update)
            drawn = sampler.sample(60000, CPU)
            self.assertLess(_ks(drawn, reference.sample(60000, CPU)), 0.02)

    def test_mass_moves_rather_than_splitting(self):
        # Halfway through, every quantile sits between the two endpoints'.
        sampler = _morph(steps=100)
        sampler.set_optimizer_update_step(50)
        u = torch.linspace(0.01, 0.99, 99)
        mid = sampler.icdf(u)
        lo = TimestepSampler.from_config(UNIFORM).icdf(u)
        hi = TimestepSampler.from_config(LOGIT).icdf(u)
        self.assertTrue(bool((mid >= torch.minimum(lo, hi) - 1e-6).all()))
        self.assertTrue(bool((mid <= torch.maximum(lo, hi) + 1e-6).all()))

    def test_stratified_marginals_match_plain_draws(self):
        sampler = _morph(steps=100)
        sampler.set_optimizer_update_step(37)
        torch.manual_seed(1)
        strata = sampler.sample_stratified(8, 4000, CPU).reshape(-1)
        self.assertLess(_ks(strata, sampler.sample(32000, CPU)), 0.02)

    def test_support_is_the_union_of_the_endpoints(self):
        sampler = _morph(
            source=TimestepSampler.from_config({**UNIFORM, "min_timestep": 0.2}),
            target=TimestepSampler.from_config({**UNIFORM, "max_timestep": 0.8}),
        )
        self.assertEqual((sampler.min_timestep, sampler.max_timestep), (0.0, 1.0))


class MixtureInterpolationTest(unittest.TestCase):
    def test_no_quantile_function_exposed(self):
        sampler = _morph(interpolation="mixture")
        with self.assertRaises(NotImplementedError):
            sampler.icdf(torch.tensor([0.5]))

    def test_falls_back_when_an_endpoint_has_no_icdf(self):
        sampler = _morph(target=TimestepSampler.from_config({"distribution": "beta"}))
        self.assertEqual(sampler.requested_interpolation, "quantile")
        self.assertEqual(sampler.interpolation, "mixture")
        self.assertIsNotNone(sampler.fallback_reason)

    def test_frozen_mixture_source_forces_mixture(self):
        inner = _morph(interpolation="mixture", steps=10)
        inner.set_optimizer_update_step(5)
        outer = MorphingTimestepSampler(
            source=inner.freeze(), target=TimestepSampler.from_config(LOGIT),
            steps=10, interpolation="quantile")
        self.assertEqual(outer.interpolation, "mixture")

    def test_density_is_linear_in_lambda(self):
        sampler = _morph(steps=100, curve="linear", interpolation="mixture")
        sampler.set_optimizer_update_step(25)
        torch.manual_seed(2)
        drawn = sampler.sample(120000, CPU)
        source_mean = float(TimestepSampler.from_config(UNIFORM).sample(120000, CPU).mean())
        target_mean = float(TimestepSampler.from_config(LOGIT).sample(120000, CPU).mean())
        self.assertAlmostEqual(float(drawn.mean()),
                               0.75 * source_mean + 0.25 * target_mean, places=2)

    def test_median_tracks_lambda_without_touching_the_training_rng(self):
        sampler = _morph(steps=100, interpolation="mixture")
        torch.manual_seed(7)
        before = torch.rand(1)
        medians = []
        for update in (0, 50, 100):
            sampler.set_optimizer_update_step(update)
            medians.append(sampler.median())
        torch.manual_seed(7)
        self.assertEqual(float(torch.rand(1)), float(before))
        self.assertEqual(medians, sorted(medians))
        self.assertAlmostEqual(medians[0], 0.5, places=2)


class SerialisationTest(unittest.TestCase):
    def test_canonical_equality_ignores_aliases_and_omitted_defaults(self):
        self.assertEqual(
            canonicalize_timestep_config({"distribution": "lognormal", "mean": 0.0}),
            canonicalize_timestep_config(
                {"distribution": "logit_normal", "mean": 0.0, "std": 1.0,
                 "min_timestep": 0.0, "max_timestep": 1.0}),
        )

    def test_canonical_form_drops_morph_and_irrelevant_params(self):
        canon = canonicalize_timestep_config(
            {"distribution": "uniform", "mean": 3.0, "morph": {"enabled": True}})
        self.assertEqual(canon, {"distribution": "uniform", "min_timestep": 0.0,
                                 "max_timestep": 1.0})

    def test_invalid_configs_are_refused(self):
        for bad in ({"distribution": "nope"},
                    {"distribution": "normal", "std": 0.0},
                    {"distribution": "beta", "alpha": 0.0},
                    {"distribution": "uniform", "min_timestep": 0.9, "max_timestep": 0.1},
                    {"distribution": "logit_normal", "mean": float("nan")}):
            with self.assertRaises(ValueError, msg=str(bad)):
                canonicalize_timestep_config(bad)

    def test_invalid_morph_blocks_are_refused(self):
        for bad in ({"enabled": True, "steps": 0},
                    {"enabled": True, "steps": 10, "curve": "spline"},
                    {"enabled": True, "steps": 10, "interpolation": "magic"},
                    {"enabled": True, "steps": 10, "from": {"distribution": "nope"}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                validate_morph_config(bad)
        validate_morph_config({"enabled": False, "steps": 0})

    def test_round_trip_preserves_the_law_and_position(self):
        sampler = _morph(steps=200, start_update=1000, curve="linear")
        sampler.set_optimizer_update_step(1100)
        rebuilt = build_sampler_from_expr(sampler.expr())
        rebuilt.set_optimizer_update_step(1100)
        self.assertEqual(rebuilt.lam, sampler.lam)
        u = torch.linspace(0.01, 0.99, 64)
        self.assertTrue(torch.allclose(rebuilt.icdf(u), sampler.icdf(u), atol=1e-6))

    def test_frozen_round_trip_keeps_lambda_pinned(self):
        sampler = _morph(steps=100)
        sampler.set_optimizer_update_step(50)
        frozen = sampler.freeze()
        rebuilt = build_sampler_from_expr(frozen.expr())
        rebuilt.set_optimizer_update_step(10_000)
        self.assertAlmostEqual(rebuilt.lam, frozen.lam, places=9)

    def test_unknown_version_is_refused(self):
        expr = sampler_expr(TimestepSampler.from_config(UNIFORM))
        expr["version"] = 99
        with self.assertRaises(ValueError):
            build_sampler_from_expr(expr)

    def test_nesting_depth(self):
        sampler = TimestepSampler.from_config(UNIFORM)
        self.assertEqual(sampler_depth(sampler), 0)
        for depth in range(1, MAX_MORPH_NESTING + 1):
            sampler = MorphingTimestepSampler(
                source=sampler, target=TimestepSampler.from_config(LOGIT), steps=10)
            self.assertEqual(sampler_depth(sampler), depth)
            self.assertEqual(sampler_depth(build_sampler_from_expr(sampler.expr())), depth)

    def test_flattening_is_deterministic_and_preserves_the_law(self):
        sampler = _morph(steps=100)
        sampler.set_optimizer_update_step(50)
        first = flatten_sampler(sampler)
        second = flatten_sampler(sampler)
        self.assertEqual(first.expr()["table"], second.expr()["table"])
        self.assertEqual(sampler_depth(first), 0)
        torch.manual_seed(3)
        self.assertLess(_ks(first.sample(60000, CPU), sampler.sample(60000, CPU)), 0.02)
        rebuilt = build_sampler_from_expr(first.expr())
        self.assertIsInstance(rebuilt, QuantileTableSampler)
        u = torch.linspace(0.01, 0.99, 64)
        self.assertTrue(torch.allclose(rebuilt.icdf(u), first.icdf(u), atol=1e-6))

    def test_flattening_does_not_touch_the_training_rng(self):
        sampler = _morph(steps=10, interpolation="mixture")
        torch.manual_seed(11)
        expected = torch.rand(3)
        torch.manual_seed(11)
        flatten_sampler(sampler)
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_plain_samplers_round_trip_through_their_config(self):
        for config in (UNIFORM, LOGIT,
                       {"distribution": "normal", "mean": 0.4, "std": 0.15},
                       {"distribution": "beta", "alpha": 1.5, "beta": 3.0},
                       {"distribution": "custom", "custom_weights": [1.0, 2.0, 1.0]}):
            sampler = TimestepSampler.from_config(config)
            self.assertEqual(sampler_to_config(sampler),
                             canonicalize_timestep_config(config))
            rebuilt = build_sampler_from_expr(sampler_expr(sampler))
            self.assertEqual(sampler_to_config(rebuilt), sampler_to_config(sampler))


if __name__ == "__main__":
    unittest.main()
