"""Config-layer plumbing for the resume-time timestep morph.

`stamp_timestep_morph_source` is the only place the PREVIOUS distribution is
still readable: PUT /training/runs/{id} regenerates the YAML and the trainer
only ever sees what it is handed.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/timestep_morph_config_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.training_config import stamp_timestep_morph_source  # noqa: E402


def _yaml(sampling):
    config = {"config": {"process": [{"train": {"timestep_sampling": sampling}}]}}
    return yaml.dump(config, sort_keys=False)


def _train(text):
    return yaml.safe_load(text)["config"]["process"][0]["train"]


MORPH = {"enabled": True, "steps": 500, "curve": "cosine",
         "interpolation": "quantile", "from": None}


class StampMorphSourceTest(unittest.TestCase):
    def test_previous_distribution_becomes_the_source(self):
        old = _yaml({"distribution": "uniform"})
        new = _yaml({"distribution": "logit_normal", "mean": 0.5, "morph": dict(MORPH)})
        stamped, source = stamp_timestep_morph_source(old, new)
        self.assertEqual(source["distribution"], "uniform")
        self.assertEqual(
            _train(stamped)["timestep_sampling"]["morph"]["from"], source)

    def test_alias_only_change_is_not_a_morph(self):
        old = _yaml({"distribution": "lognormal", "mean": 0.5})
        new = _yaml({"distribution": "logit_normal", "mean": 0.5, "std": 1.0,
                     "morph": dict(MORPH)})
        stamped, source = stamp_timestep_morph_source(old, new)
        self.assertIsNone(source)
        self.assertIsNone(_train(stamped)["timestep_sampling"]["morph"]["from"])

    def test_explicit_source_is_left_alone(self):
        explicit = {"distribution": "normal", "mean": 0.2, "std": 0.1}
        new = _yaml({"distribution": "uniform",
                     "morph": {**MORPH, "from": explicit}})
        stamped, source = stamp_timestep_morph_source(
            _yaml({"distribution": "beta"}), new)
        self.assertIsNone(source)
        self.assertEqual(
            _train(stamped)["timestep_sampling"]["morph"]["from"], explicit)

    def test_no_previous_config_falls_back_to_the_arch_default(self):
        new = _yaml({"distribution": "uniform", "morph": dict(MORPH)})
        stamped, source = stamp_timestep_morph_source(
            None, new, arch_default={"distribution": "logit_normal", "mean": -0.8,
                                     "std": 0.8})
        self.assertEqual(source["mean"], -0.8)
        # ... and with nothing to fall back on, the morph stays a no-op.
        _, none_source = stamp_timestep_morph_source(None, new)
        self.assertIsNone(none_source)

    def test_disabled_or_absent_morph_is_untouched(self):
        for sampling in ({"distribution": "uniform"},
                         {"distribution": "uniform",
                          "morph": {**MORPH, "enabled": False}}):
            new = _yaml(sampling)
            stamped, source = stamp_timestep_morph_source(
                _yaml({"distribution": "beta"}), new)
            self.assertIsNone(source)
            self.assertEqual(stamped, new)

    def test_unparseable_or_invalid_input_is_passed_through(self):
        new = _yaml({"distribution": "uniform", "morph": dict(MORPH)})
        self.assertEqual(stamp_timestep_morph_source("::: not yaml :::", new)[1], None)
        self.assertEqual(
            stamp_timestep_morph_source(_yaml({"distribution": "nope"}), new)[1], None)


class DefaultsTest(unittest.TestCase):
    def test_training_defaults_carry_a_disabled_morph_block(self):
        from api.param_defaults import (
            TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH, TRAINING_DEFAULTS,
        )
        morph = TRAINING_DEFAULTS["timestep_sampling"]["morph"]
        self.assertFalse(morph["enabled"])
        self.assertEqual(morph["curve"], "cosine")
        self.assertEqual(morph["interpolation"], "quantile")
        self.assertIsNone(morph["from"])
        # The per-arch defaults stay morph-free: a morph is never implied by
        # picking a model.
        for name, config in TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH.items():
            self.assertNotIn("morph", config, name)


if __name__ == "__main__":
    unittest.main()
