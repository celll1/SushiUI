"""CPU tests for the from-scratch FoveaTag encoder."""

import torch

from core.tagger.foveatag.encoder import (
    FoveaEncoder,
    FoveaEncoderConfig,
    apply_2d_rope,
    source_coordinates,
)


def _tiny_config(**overrides):
    values = {
        "widths": (8, 16, 32, 48),
        "depths": (1, 1, 2, 1),
        "window_size": 4,
        "attention_heads": (2, 3),
        "mlp_ratio": 2.0,
        "scale_dropout": 0.0,
    }
    values.update(overrides)
    return FoveaEncoderConfig(**values)


def test_encoder_accepts_non_square_non_multiple_canvas_and_preserves_stride_16_map():
    model = FoveaEncoder(_tiny_config()).eval()
    images = torch.randn(2, 3, 65, 49)
    geometry = torch.tensor([
        [0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [0.25, 0.75, 0.2, 0.4, -2.0, -1.0, 0.0, 0.0, 1.0, 1.0],
    ])

    output = model(images, geometry)

    assert output.local_map.shape == (2, 32, 5, 4)
    assert output.local_coordinates.shape == (2, 5, 4, 2)
    assert output.global_tokens.shape == (2, 6, 48)
    assert output.global_summary.shape == (2, 48)
    assert torch.isfinite(output.global_summary).all()


def test_source_coordinates_follow_view_center_and_field_size():
    geometry = torch.tensor([[0.5, 0.25, 0.4, 0.2, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    coordinates = source_coordinates(
        geometry,
        2,
        2,
        device=geometry.device,
        dtype=geometry.dtype,
    )

    torch.testing.assert_close(coordinates[0, 0, 0], torch.tensor([0.4, 0.2]))
    torch.testing.assert_close(coordinates[0, 1, 1], torch.tensor([0.6, 0.3]))


def test_source_coordinates_account_for_letterbox_content_box():
    geometry = torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.25, 0.0, 0.5, 1.0]])
    coordinates = source_coordinates(
        geometry,
        2,
        4,
        device=geometry.device,
        dtype=geometry.dtype,
    )

    torch.testing.assert_close(coordinates[0, 0, 1], torch.tensor([0.25, 0.25]))
    torch.testing.assert_close(coordinates[0, 1, 2], torch.tensor([0.75, 0.75]))


def test_rope_is_identity_at_zero_coordinates_and_changes_nonzero_positions():
    query = torch.randn(1, 2, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    zeros = torch.zeros(1, 3, 2)
    rotated_query, rotated_key = apply_2d_rope(query, key, zeros)
    torch.testing.assert_close(rotated_query, query)
    torch.testing.assert_close(rotated_key, key)

    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]]])
    moved_query, _ = apply_2d_rope(query, key, positions)
    assert not torch.allclose(moved_query[:, :, 1:], query[:, :, 1:])


def test_default_encoder_stays_in_first_capacity_band():
    parameter_count = FoveaEncoder().parameter_count()
    assert 25_000_000 <= parameter_count <= 50_000_000


def test_invalid_attention_geometry_is_rejected():
    try:
        _tiny_config(widths=(8, 16, 30, 48))
    except ValueError as error:
        assert "divisible by four" in str(error)
    else:
        raise AssertionError("expected invalid RoPE dimensions to fail")
