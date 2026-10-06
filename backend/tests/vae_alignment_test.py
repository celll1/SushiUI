import pytest
import torch
from core.experiments.vae_alignment import AffineMap, SpatialMap, approved_image_paths, fit_affine, fit_orthogonal, fit_spatial, fold_input, ming_patch_order, pack, rows, unpack


def test_spatial_packing_is_lossless():
    value = torch.arange(2 * 16 * 8 * 10).reshape(2, 16, 8, 10).float()
    packed = pack(value)
    assert packed.shape == (2, 64, 4, 5)
    assert torch.equal(unpack(packed), value)
    with pytest.raises(ValueError):
        pack(value[:, :, :7])


def test_affine_fit_generalizes_to_unseen_images_and_folds_input():
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(5, 8, 10, 10, generator=generator)
    mapping = AffineMap(torch.randn(8, 8, generator=generator).double(), torch.randn(8, generator=generator).double())
    fitted = fit_affine(source[:4], mapping(source[:4]), ridge=0)
    torch.testing.assert_close(fitted(source[4:]), mapping(source[4:]), atol=1e-5, rtol=1e-5)
    old_input = torch.nn.Linear(8, 20).double()
    weight, bias = fold_input(old_input, fitted)
    actual = rows(source[4:]).double() @ weight.T + bias
    expected = old_input(rows(fitted(source[4:])).double())
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_rgb_bridge_loses_nullspace_information():
    to_rgb = AffineMap(torch.randn(64, 3).double(), torch.zeros(3).double())
    from_rgb = AffineMap(torch.linalg.pinv(to_rgb.weight), torch.zeros(64).double())
    assert to_rgb.then(from_rgb).diagnostics()["rank"] == 3


def test_orthogonal_map_preserves_noise_covariance():
    source = torch.randn(4, 8, 10, 10)
    target = torch.randn_like(source)
    result = fit_orthogonal(source, target)
    assert result.diagnostics()["noise_covariance_relative_error"] < 1e-12


def test_mismatched_grid_is_rejected():
    with pytest.raises(ValueError):
        fit_affine(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))


def test_flow_bias_is_scaled_by_clean_fraction():
    mapping = AffineMap(torch.eye(4).double(), torch.arange(4).double())
    state = torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(mapping.flow_state(state, 0), mapping(state))
    torch.testing.assert_close(mapping.flow_state(state, 1), state)
    value = mapping.flow_state(state, torch.tensor([0., 1.]))
    torch.testing.assert_close(value[:1], mapping(state[:1]))
    torch.testing.assert_close(value[1:], state[1:])


@pytest.mark.parametrize("record", ["unverified.png", {"path": "x", "rating": "general", "tags": []},
                                    {"path": "x", "rating": "explicit", "tags": ["explicit"]},
                                    {"path": "x", "rating": "general", "tags": ["general", "sensitive"]}])
def test_unapproved_images_are_rejected_without_opening_files(record):
    with pytest.raises(ValueError):
        approved_image_paths([record])


def test_general_rating_is_required():
    assert approved_image_paths([{"path": "approved.png", "rating": "general", "tags": ["general", "solo"]}]) == ["approved.png"]


def test_spatial_alignment_recovers_neighborhood_transform():
    source = torch.randn(5, 4, 12, 12)
    mapping = SpatialMap(torch.randn(36, 4).double(), torch.randn(4).double())
    fitted = fit_spatial(source[:4], mapping(source[:4]), ridge=0)
    torch.testing.assert_close(fitted(source[4:]), mapping(source[4:]), atol=1e-5, rtol=1e-5)


def test_ming_patch_order_matches_native_patchification():
    latent = torch.randn(1, 16, 8, 8)
    packed = pack(latent)
    mapping = ming_patch_order(AffineMap(torch.eye(64).double(), torch.zeros(64).double()))
    actual = rows(mapping(packed))
    native = latent[0].view(16, 4, 2, 4, 2).permute(1, 3, 2, 4, 0).reshape(16, 64)
    torch.testing.assert_close(actual, native)
