"""CPU tests for FoveaTag masked reconstruction components."""

import torch

from core.tagger.foveatag.reconstruction import (
    PatchReconstructionHead,
    expand_patch_mask,
    mask_image_patches,
    random_patch_mask,
)


def test_reconstruction_head_decodes_one_rgb_patch_per_feature_cell():
    head = PatchReconstructionHead(feature_dim=8, patch_size=4)
    reconstruction = head(torch.randn(2, 8, 3, 5))
    assert reconstruction.shape == (2, 3, 12, 20)


def test_random_mask_has_exact_count_and_is_reproducible():
    first = random_patch_mask(
        2,
        4,
        5,
        0.25,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )
    second = random_patch_mask(
        2,
        4,
        5,
        0.25,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(first, second)
    assert first.sum(dim=(1, 2)).tolist() == [5, 5]


def test_image_masking_changes_only_selected_patch():
    images = torch.ones(1, 3, 4, 4)
    patch_mask = torch.tensor([[[True, False], [False, False]]])
    masked = mask_image_patches(images, patch_mask, patch_size=2, fill=-1.0)

    assert torch.all(masked[:, :, :2, :2] == -1)
    assert torch.all(masked[:, :, :2, 2:] == 1)
    assert torch.all(masked[:, :, 2:, :] == 1)
    assert expand_patch_mask(patch_mask, 2).shape == (1, 4, 4)
