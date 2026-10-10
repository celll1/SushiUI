"""Grid choices must not perturb exposure selection or timestep draws."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from api.param_defaults import DIT_HYBRID_INITIALIZATION_DEFAULTS
from core.rewiring import provenance
from core.rewiring.resolution_schedule import (
    ResolutionSchedule, calibration_identity, exposure_noise,
    initialization_resolutions, resolution_recipe_fields, timestep_generator,
)
from core.rewiring.spec import digest
from hybrid_provenance_test import record


def test_single_grid_preserves_rng_and_calibration_hash():
    settings = deepcopy(DIT_HYBRID_INITIALIZATION_DEFAULTS)
    latent = torch.zeros(1, 16, 8, 8)
    expected_rng = torch.Generator().manual_seed(718)
    expected_noise = torch.randn(latent.shape, generator=expected_rng)
    expected_sigma = .01 + .98 * torch.rand(1, generator=expected_rng)
    actual_rng = torch.Generator().manual_seed(718)
    noise, sigma = exposure_noise(latent, actual_rng)
    assert torch.equal(noise, expected_noise) and torch.equal(sigma, expected_sigma)
    assert torch.equal(actual_rng.get_state(), expected_rng.get_state())
    fingerprints = [{"image_sha256": "a" * 64, "role": "fit"}]
    assert digest(calibration_identity(fingerprints, settings)) == digest(
        dict(items=fingerprints, resolution=settings["resolution"]))
    assert resolution_recipe_fields(settings) == {}
    schedule = ResolutionSchedule(settings)
    assert [schedule.next() for _ in range(9)] == [settings["resolution"]] * 9


def test_mixed_grids_have_balanced_reproducible_shape_independent_timesteps():
    settings = dict(resolution=512, additional_resolutions=[256, 1024], seed=718)
    schedule = ResolutionSchedule(settings)
    assert [schedule.next() for _ in range(9)] == [512, 256, 1024] * 3
    observed = []
    for shapes in ([8] * 9, [8, 4, 16] * 3):
        noise_rng = torch.Generator().manual_seed(718)
        sigma_rng = timestep_generator(settings,"cpu")
        assert sigma_rng.initial_seed() == 719
        observed.append([exposure_noise(torch.zeros(1, 16, side, side), noise_rng, sigma_rng)[1]
                         for side in shapes])
    assert all(torch.equal(left, right) for left, right in zip(*observed))
    assert ResolutionSchedule(settings).next() == 512
    assert timestep_generator(dict(resolution=512,seed=718),"cpu") is None


@pytest.mark.parametrize("extra", [[512], [256, 256], [True], [65], [0], [1088],
                                  [128, 256, 384, 768], (256,), None, [256.0]])
def test_invalid_resolution_lists_refused(extra):
    with pytest.raises(ValueError):
        initialization_resolutions(dict(resolution=512, additional_resolutions=extra))


def test_mixed_recipe_roundtrip_preserves_old_lineage_and_distinguishes_grids():
    value = record()
    settings = {key: deepcopy(DIT_HYBRID_INITIALIZATION_DEFAULTS[key])
                for key in provenance.RECIPE_FIELDS}
    value["recipe"] = dict(schema_version=1, settings=settings)
    spec = value["origin_spec"]
    legacy = provenance.checkpoint_metadata(value, spec)
    assert provenance.read_checkpoint_metadata(legacy, spec) == value
    mixed = deepcopy(value)
    mixed["recipe"]["settings"].update(resolution_recipe_fields(
        dict(resolution=settings["resolution"], additional_resolutions=[256])))
    metadata = provenance.checkpoint_metadata(mixed, spec)
    assert metadata[provenance.META_HASH] != legacy[provenance.META_HASH]
    assert provenance.read_checkpoint_metadata(metadata, spec) == mixed
    assert provenance.checkpoint_metadata(value, spec) == legacy
    for mutate in (lambda s: s.pop("resolution_sampling"),
                   lambda s: s.update(resolution_sampling="random"),
                   lambda s: s.update(additional_resolutions=[]),
                   lambda s: s.update(additional_resolutions=[512])):
        broken = deepcopy(mixed)
        mutate(broken["recipe"]["settings"])
        with pytest.raises(ValueError):
            provenance.validate_record(broken, spec)
