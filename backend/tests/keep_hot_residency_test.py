from types import SimpleNamespace

import torch

from core.keep_hot import (
    additional_residency_nbytes,
    component_nbytes,
    mark_resident,
)


def test_component_nbytes_sums_grouped_components():
    first = torch.nn.Linear(3, 2)
    second = torch.nn.Linear(2, 1, bias=False)

    assert component_nbytes((first, second)) == (
        component_nbytes(first) + component_nbytes(second)
    )


def test_residency_budget_counts_only_new_components_for_same_model():
    manager = SimpleNamespace()
    text_encoders = (torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    denoiser = torch.nn.Linear(5, 4)
    vae = torch.nn.Linear(4, 3)
    components = {
        "text_encoder": text_encoders,
        "unet": denoiser,
        "vae": vae,
    }

    mark_resident(manager, "unet", "model-a")
    mark_resident(manager, "vae", "model-a")

    assert additional_residency_nbytes(manager, "model-a", components) == component_nbytes(
        text_encoders
    )


def test_residency_budget_counts_every_component_after_identity_change():
    manager = SimpleNamespace()
    denoiser = torch.nn.Linear(5, 4)
    vae = torch.nn.Linear(4, 3)
    components = {"unet": denoiser, "vae": vae}

    mark_resident(manager, "unet", "model-a")
    mark_resident(manager, "vae", "model-a")

    assert additional_residency_nbytes(manager, "model-b", components) == sum(
        component_nbytes(component) for component in components.values()
    )
