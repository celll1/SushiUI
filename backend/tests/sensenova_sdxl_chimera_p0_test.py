"""P0 contracts for SenseNova SDXL Chimera (CPU, tiny models)."""

import pytest
import torch

from core.models.sensenova_sdxl_chimera import (
    ChimeraBridgeConfig,
    ConditioningBridge,
    apply_sensenova_rope,
    build_donor_equal_unet,
    flow_euler_step,
    flow_noising,
    flow_velocity_target,
    parameter_census,
    selected_layer_indices,
    spatial_query_positions,
)


def _tiny_sdxl_unet():
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        norm_num_groups=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=2048,
        attention_head_dim=2,
        transformer_layers_per_block=1,
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=1328,
    )


def test_donor_equal_unet_scratch_and_transplant_contracts():
    torch.manual_seed(123)
    donor = _tiny_sdxl_unet()
    scratch, scratch_report = build_donor_equal_unet(donor, initialization="scratch", seed=9)
    transplanted, transplant_report = build_donor_equal_unet(
        donor, initialization="sdxl_transplant", seed=10
    )

    assert parameter_census(scratch) == parameter_census(donor)
    assert scratch_report.parameter_count == transplant_report.parameter_count
    assert torch.count_nonzero(scratch.conv_out.weight) == 0
    for name, tensor in donor.state_dict().items():
        assert torch.equal(transplanted.state_dict()[name], tensor), name
    assert not torch.equal(scratch.conv_in.weight, donor.conv_in.weight)


def test_donor_equal_unet_rejects_incompatible_contracts():
    donor = _tiny_sdxl_unet()
    donor.register_to_config(cross_attention_dim=16)
    with pytest.raises(ValueError, match="cross_attention_dim=2048"):
        build_donor_equal_unet(donor)


def test_selected_layers_are_relative_unique_and_include_last():
    assert selected_layer_indices(42) == (10, 20, 31, 41)
    assert selected_layer_indices(2) == (0, 1)
    assert selected_layer_indices(1) == (0,)
    with pytest.raises(ValueError, match="positive"):
        selected_layer_indices(0)


def test_bridge_shapes_positions_mask_and_gradients():
    config = ChimeraBridgeConfig(
        hidden_size=16,
        kv_width=8,
        selected_layers=(1, 3),
        context_tokens=5,
        context_dim=12,
        pooled_dim=7,
        bridge_dim=16,
        num_heads=4,
    )
    bridge = ConditioningBridge(config)
    hidden = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    positions = torch.randn(2, 6, 3)
    kv = {
        layer: (torch.randn(2, 2, 6, 4), torch.randn(2, 2, 6, 4))
        for layer in config.selected_layers
    }
    out = bridge(hidden, kv, mask, positions)

    assert out.encoder_hidden_states.shape == (2, 5, 12)
    assert out.pooled_text_embeds.shape == (2, 7)
    assert out.context_positions.shape == (2, 5, 3)
    assert out.resampler_weights.shape == (2, 5, 6)
    assert out.position_variance.shape == (2, 5, 3)
    assert torch.equal(out.resampler_weights[0, :, 4:6], torch.zeros(5, 2))
    out.encoder_hidden_states.square().mean().backward()
    assert bridge.context_projection.weight.grad is not None
    assert bridge.hidden_projection.weight.grad is not None


def test_spatial_positions_match_physical_points_and_crop_shift():
    coarse = spatial_query_positions(2, 2, target_height=64, target_width=64)
    fine = spatial_query_positions(4, 4, target_height=64, target_width=64)
    # Coarse cell (0,0) and the mean of its four fine children share a center.
    fine_grid = fine.reshape(1, 4, 4, 3)
    assert torch.allclose(coarse[0, 0], fine_grid[0, :2, :2].mean(dim=(0, 1)))

    shifted = spatial_query_positions(
        2, 2, target_height=64, target_width=64, crop_top=32, crop_left=64
    )
    delta = shifted - coarse
    assert torch.allclose(delta[..., 0], torch.zeros_like(delta[..., 0]))
    assert torch.allclose(delta[..., 1], torch.ones_like(delta[..., 1]))
    assert torch.allclose(delta[..., 2], torch.full_like(delta[..., 2], 2.0))


def test_rope_matches_rotate_half_reference_and_constant_t_is_relative_invariant():
    tensor = torch.randn(1, 2, 5, 16)
    positions = torch.randn(1, 5, 3)
    actual = apply_sensenova_rope(tensor, positions, rope_theta=10000, rope_theta_hw=1000)

    def reference_axis(part, coordinate, base):
        dim = part.shape[-1]
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freq = coordinate.float().unsqueeze(-1) * inv
        emb = torch.cat((freq, freq), dim=-1).unsqueeze(1)
        left, right = part.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return part * emb.cos() + rotated * emb.sin()

    t, h, w = torch.split(tensor, (8, 4, 4), dim=-1)
    expected = torch.cat((
        reference_axis(t, positions[..., 0], 10000),
        reference_axis(h, positions[..., 1], 1000),
        reference_axis(w, positions[..., 2], 1000),
    ), dim=-1)
    assert torch.allclose(actual, expected)

    shifted = positions.clone()
    shifted[..., 0] += 123.0
    q1 = apply_sensenova_rope(tensor, positions)
    q2 = apply_sensenova_rope(tensor, shifted)
    assert torch.allclose(
        torch.matmul(q1, q1.transpose(-1, -2)),
        torch.matmul(q2, q2.transpose(-1, -2)),
        atol=2e-5,
        rtol=2e-5,
    )


def test_flow_endpoints_and_straight_path():
    clean = torch.randn(3, 4, 5, 6)
    noise = torch.randn_like(clean)
    t = torch.tensor([0.0, 0.4, 1.0])
    sample = flow_noising(clean, noise, t)
    velocity = flow_velocity_target(clean, noise)
    assert torch.equal(sample[0], noise[0])
    assert torch.equal(sample[2], clean[2])
    reached = flow_euler_step(noise, velocity, 0.0, 1.0)
    assert torch.allclose(reached, clean, atol=5e-7, rtol=1e-6)
