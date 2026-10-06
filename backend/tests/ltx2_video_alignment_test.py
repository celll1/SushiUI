"""Real small native AV transformers; synthetic inputs only."""

import pytest
import torch

from core.models.components.latent_adapter import ImageCodecContract, SpatialAlignmentPair
from core.models.components.video_latent_adapter import VideoCodecContract, map_video_tokens


@pytest.mark.parametrize("old,new", [(4, 2), (2, 4)])
@pytest.mark.parametrize("rewired", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_native_av_boundary_gradient_and_unet_coexistence(old, new, rewired, per_token):
    from diffusers import LTX2VideoTransformer3DModel
    from core.models.ltx2.video_alignment import install_video_alignment
    from core.models.ltx2.residual_unet import native_config, wrap_transformer
    from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
    from core.rewiring.spec import build_spec, OPERATORS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        inner = LTX2VideoTransformer3DModel(in_channels=old, out_channels=old,
            num_attention_heads=2, attention_head_dim=16, cross_attention_dim=32,
            audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
            audio_attention_head_dim=8, audio_cross_attention_dim=16, num_layers=4,
            use_prompt_embeddings=False, cross_attn_mod=True, audio_cross_attn_mod=True,
            gated_attn=True, audio_gated_attn=True)
        model = inner
        if rewired:
            graph = build_spec("ltx2", [{"name": "main", "block_count": 4}], [
                dict(id="region", stream="main", parent=None, start_block=1, end_block=3,
                     down_factor=[2, 2], **OPERATORS)], native_config(inner.config))
            model = wrap_transformer(inner, graph)
        image = ImageCodecContract(new, 16, 5, "identity", 1, 3)
        contract = VideoCodecContract(image, 2, "causal_first_frame_v1")
        declaration = image.declaration(source="synthetic", source_identity="old", target_identity="new",
                                        old_channels=old, old_scale=32)
        pair = SpatialAlignmentPair(declaration)
        for bridge in (pair.input, pair.output):
            weight = torch.randn(bridge.source_channels*9, bridge.target_channels)/9
            bias = torch.randn(bridge.target_channels)/10
            bridge.initialize(weight, bias)
        aligned = install_video_alignment(model, pair, contract)
        assert aligned.config.in_channels == new
        assert inner.config.in_channels == max(old, new)
        assert tuple(inner.rope.scale_factors) == (2, 16, 16)
        assert tuple(inner.audio_rope.scale_factors) != (2, 16, 16)
        tokens = torch.randn(1, 3*4*6, new, requires_grad=True)
        args = dict(audio_hidden_states=torch.randn(1, 5, 4), encoder_hidden_states=torch.randn(1, 4, 32),
            audio_encoder_hidden_states=torch.randn(1, 4, 16), timestep=torch.tensor([500.]),
            audio_timestep=torch.tensor([600.]), sigma=torch.tensor([.8]), audio_sigma=torch.tensor([.6]),
            num_frames=3, height=4, width=6, fps=24., audio_num_frames=5, return_dict=False)
        state_sigma = .5
        if per_token:
            args["timestep"] = torch.linspace(200., 800., 72).reshape(1, 72)
            state_sigma = args["timestep"]/1000
        actual = aligned(hidden_states=tokens, **args)
        # Prompt sigma deliberately differs from the state timestep.
        backbone_input = map_video_tokens(pair.input, tokens, grid=(3, 4, 6), sigma=state_sigma)
        reference = Ltx2BlockLoopWrapper(inner)(hidden_states=backbone_input, **args)
        expected_video = map_video_tokens(pair.output, reference[0], grid=(3, 4, 6), velocity=True)
        torch.testing.assert_close(actual[0], expected_video, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
        assert actual[0].shape == tokens.shape and actual[1].shape == (1, 5, 4)
        output = aligned(hidden_states=tokens, **{**args, "return_dict": True})
        torch.testing.assert_close(output.sample, actual[0], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(aligned.parameters(), lr=1e-3)
        before = pair.output.projection.weight.detach().clone()
        (actual[0]-torch.randn_like(actual[0])).square().mean().backward()
        assert pair.input.projection.weight.grad.abs().sum() > 0
        assert pair.output.projection.weight.grad.abs().sum() > 0
        assert tokens.grad.isfinite().all()
        if new > old:
            assert inner.proj_out.weight.grad[old:].abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, pair.output.projection.weight)
        assert any(key.startswith("_video_latent_alignment.") for key in aligned.state_dict())
        if rewired:
            assert inner._residual_unet_last_grid["video_grid"] == [3, 4, 6]
            assert inner._residual_unet_last_grid["audio_tokens"] == 5
