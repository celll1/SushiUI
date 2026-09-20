import math

import torch
from diffusers import UNet2DConditionModel

from core.models.sensenova_sdxl_chimera.artifact import (
    V4_FORMAT_VERSION,
    prediction_contract,
    validated_prediction_contract,
)
from core.models.sensenova_sdxl_chimera.flow import (
    FLOW_V4_PREDICTION,
    conditioning_reliability,
    destruction_coordinate,
    destruction_coordinate_polar_flow_target,
    destruction_coordinate_recover_clean,
    gate_conditioning_tensor,
    mixed_coordinate_polar_exp_euler_step,
    polar_flow_target,
    polar_tangent_projection,
)
from core.models.sensenova_sdxl_chimera.attention_processor import (
    install_chimera_attention_processors,
)
from core.models.sensenova_sdxl_chimera.pipeline_ops import (
    ChimeraConditioning,
    _unet_polar,
)
from core.models.sensenova_sdxl_chimera.unet import install_polar_radial_head


def _orthogonal_pair(dtype=torch.float64):
    noise = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]], dtype=dtype)
    clean = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]], dtype=dtype)
    return clean, noise


def test_v4_uses_the_v3_beta32_state_path():
    clean, noise = _orthogonal_pair()
    times = torch.tensor([0.37], dtype=clean.dtype)
    v3 = polar_flow_target(
        clean,
        noise,
        times,
        latent_mean=[0.0],
        angular_schedule="confidence_gated_beta_3_2_v1",
    )
    v4 = destruction_coordinate_polar_flow_target(
        clean, noise, times, latent_mean=[0.0]
    )
    torch.testing.assert_close(v4.sample, v3.sample)
    torch.testing.assert_close(v4.direction, v3.direction)
    torch.testing.assert_close(v4.radius, v3.radius)
    torch.testing.assert_close(v4.radial_velocity, v3.radial_velocity)


def test_v4_noise_endpoint_tangent_is_finite_and_nonzero():
    clean, noise = _orthogonal_pair()
    target = destruction_coordinate_polar_flow_target(
        clean, noise, 0.0, latent_mean=[0.0]
    )
    assert torch.isfinite(target.tangent_velocity).all()
    torch.testing.assert_close(
        target.tangent_velocity.square().mean().sqrt(),
        torch.tensor(math.pi / 2, dtype=clean.dtype),
    )
    torch.testing.assert_close(
        (target.direction * target.tangent_velocity).mean(),
        torch.tensor(0.0, dtype=clean.dtype),
        atol=1e-12,
        rtol=0,
    )


def test_v4_tangent_matches_finite_difference_in_destruction_coordinate():
    clean, noise = _orthogonal_pair()
    t = 0.41
    epsilon = 1e-5
    center = destruction_coordinate_polar_flow_target(
        clean, noise, t, latent_mean=[0.0]
    )
    before = destruction_coordinate_polar_flow_target(
        clean, noise, t - epsilon, latent_mean=[0.0]
    )
    after = destruction_coordinate_polar_flow_target(
        clean, noise, t + epsilon, latent_mean=[0.0]
    )
    d_before, _ = destruction_coordinate(t - epsilon, clean)
    d_after, _ = destruction_coordinate(t + epsilon, clean)
    finite_difference = (after.sample - before.sample) / (d_after - d_before)
    tangent_difference = polar_tangent_projection(
        finite_difference, center.direction
    )
    torch.testing.assert_close(
        tangent_difference,
        center.tangent_velocity,
        atol=2e-5,
        rtol=2e-5,
    )


def test_v4_exact_target_recovers_clean_endpoint():
    clean, noise = _orthogonal_pair()
    target = destruction_coordinate_polar_flow_target(
        clean, noise, 0.43, latent_mean=[0.0]
    )
    recovered = destruction_coordinate_recover_clean(
        target.sample,
        target.radial_velocity,
        target.tangent_velocity,
        0.43,
        latent_mean=[0.0],
        latent_centered_second_moment=1.0,
    )
    torch.testing.assert_close(recovered, clean, atol=1e-10, rtol=1e-10)


def test_mixed_coordinate_solver_uses_delta_d_for_angle():
    clean, noise = _orthogonal_pair()
    current_t = 0.2
    next_t = 0.2001
    current = destruction_coordinate_polar_flow_target(
        clean, noise, current_t, latent_mean=[0.0]
    )
    expected = destruction_coordinate_polar_flow_target(
        clean, noise, next_t, latent_mean=[0.0]
    )
    result = mixed_coordinate_polar_exp_euler_step(
        current.sample,
        current.radial_velocity,
        current.tangent_velocity,
        current_t,
        next_t,
        latent_mean=[0.0],
    )
    torch.testing.assert_close(result.direction, expected.direction, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        result.direction.square().mean().sqrt(),
        torch.tensor(1.0, dtype=clean.dtype),
        atol=1e-12,
        rtol=0,
    )


def test_conditioning_reliability_has_exact_endpoints():
    sample = torch.zeros(2, 1, 1, 1)
    gate = conditioning_reliability(torch.tensor([0.0, 1.0]), sample)
    torch.testing.assert_close(gate.flatten(), torch.tensor([0.0, 1.0]))


def test_noise_endpoint_removes_arbitrary_conditioning_difference():
    sample = torch.randn(2, 4, 2, 2)
    conditional = torch.randn(2, 9, 32)
    unconditional = torch.randn(2, 9, 32)
    conditional = gate_conditioning_tensor(conditional, 0.0, sample)
    unconditional = gate_conditioning_tensor(unconditional, 0.0, sample)
    torch.testing.assert_close(conditional, unconditional, atol=0, rtol=0)


def test_noise_endpoint_unet_outputs_are_context_invariant():
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=6,
        attention_head_dim=1,
        norm_num_groups=4,
        addition_embed_type="text_time",
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=29,
    ).eval()
    install_polar_radial_head(unet)
    install_chimera_attention_processors(unet, backend="normal")
    sample = torch.randn(1, 4, 8, 8)
    time_ids = torch.tensor([[64, 64, 0, 0, 64, 64]], dtype=sample.dtype)
    prediction = prediction_contract(
        FLOW_V4_PREDICTION,
        latent_mean=[0.0, 0.0, 0.0, 0.0],
        latent_centered_second_moment=1.0,
    )

    def run(seed):
        generator = torch.Generator().manual_seed(seed)
        conditioning = ChimeraConditioning(
            encoder_hidden_states=torch.randn(1, 3, 6, generator=generator),
            pooled_text_embeds=torch.randn(1, 5, generator=generator),
            context_positions=torch.zeros(1, 3, 3),
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            fingerprint=str(seed),
            key_lengths=(3,),
        )
        return _unet_polar(
            unet,
            sample,
            torch.tensor(0.0),
            conditioning,
            time_ids,
            cache_metadata=(64, 64, 0, 0, 0.0, "test"),
            prediction=prediction,
        )[:2]

    radial_a, tangent_a = run(1)
    radial_b, tangent_b = run(2)
    torch.testing.assert_close(radial_a, radial_b, atol=0, rtol=0)
    torch.testing.assert_close(tangent_a, tangent_b, atol=0, rtol=0)


def test_v4_artifact_contract_is_strict_and_format_owned():
    contract = prediction_contract(
        FLOW_V4_PREDICTION,
        latent_mean=[0.0, 0.0, 0.0, 0.0],
        latent_centered_second_moment=1.0,
    )
    assert contract["tangent_units"] == "beta_3_2_d"
    assert contract["conditioning_gate"] == "clean_alignment_squared_v1"
    assert contract["integrator"] == "mixed_coordinate_polar_exp_euler_v1"
    assert validated_prediction_contract(
        {"format_version": V4_FORMAT_VERSION, "prediction": contract}
    ) == contract
