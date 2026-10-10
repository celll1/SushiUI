"""Bounded resolution curricula with shape-independent exposure selection."""
from copy import deepcopy

RESOLUTION_SAMPLING = "round_robin_v1"


def initialization_resolutions(settings):
    base = settings["resolution"]
    extra = settings.get("additional_resolutions", [])
    if not isinstance(extra, list) or len(extra) > 3:
        raise ValueError("Initialization supports at most three additional resolutions")
    values = (base, *extra)
    if any(type(side) is not int or not 64 <= side <= 1024 or side % 64 for side in values):
        raise ValueError("Initialization resolutions must be multiples of 64 from 64 to 1024")
    if len(set(values)) != len(values):
        raise ValueError("Initialization resolutions must be distinct")
    return values


def resolution_recipe_fields(settings):
    resolutions = initialization_resolutions(settings)
    if len(resolutions) == 1:
        return {}
    return {"additional_resolutions": list(resolutions[1:]),
            "resolution_sampling": RESOLUTION_SAMPLING}


def resolution_items(items, settings):
    for side in initialization_resolutions(settings):
        for index, item in enumerate(items):
            yield side, index, item


def calibration_identity(fingerprints, settings):
    return dict(items=deepcopy(fingerprints), resolution=settings["resolution"],
                **resolution_recipe_fields(settings))


class ResolutionSchedule:
    """Each fitting phase starts at the anchor and visits every grid equally."""

    def __init__(self, settings):
        self.resolutions = initialization_resolutions(settings)
        self.position = 0

    def next(self):
        side = self.resolutions[self.position % len(self.resolutions)]
        self.position += 1
        return side


def timestep_generator(settings, device):
    import torch

    if len(initialization_resolutions(settings)) == 1:
        return None
    return torch.Generator(device=device).manual_seed(settings["seed"] + 1)


def exposure_noise(latent, noise_rng, sigma_rng=None):
    import torch

    # Single-grid callers retain the historical RNG order; mixed grids use a separate sigma stream.
    if sigma_rng is None:
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=noise_rng)
        sigma = .01 + .98 * torch.rand(1, device=latent.device, generator=noise_rng)
    else:
        sigma = .01 + .98 * torch.rand(1, device=latent.device, generator=sigma_rng)
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=noise_rng)
    return noise, sigma
