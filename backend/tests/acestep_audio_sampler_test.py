"""Real small ACE samplers on target-native audio grids, using synthetic latents."""

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from acestep_vae_boundary_test import small_owner, alignment


def args():
    return dict(text_hidden_states=torch.randn(1, 3, 8), text_attention_mask=torch.ones(1, 3),
        lyric_hidden_states=torch.randn(1, 4, 8), lyric_attention_mask=torch.ones(1, 4),
        refer_audio_acoustic_hidden_states_packed=torch.randn(1, 16, 2),
        refer_audio_order_mask=torch.zeros(1, dtype=torch.long), src_latents=torch.randn(1, 48, 2),
        chunk_masks=torch.ones(1, 48, 2), is_covers=torch.zeros(1, dtype=torch.bool),
        silence_latent=torch.randn(1, 16, 2), seed=73, infer_steps=2, dcw_enabled=False)


@pytest.mark.parametrize("new,rewired,mode", [(1, False, "plain"), (4, True, "cover"),
    (4, False, "repaint"), (1, True, "sde")])
def test_real_sampler_keeps_decoder_on_target_grid(new, rewired, mode):
    from core.models.acestep.vae_boundary import install_decoder_alignment
    from core.models.acestep.residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec, OPERATORS
    torch.manual_seed(51)
    owner, pair = small_owner().eval(), alignment(new)
    if rewired:
        graph = build_spec("acestep", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
                 down_factor=[2, 1], **OPERATORS)], native_config(owner.config))
        restore_topology(owner, graph)
    install_decoder_alignment(owner, pair)
    inputs = args()
    if mode == "cover":
        inputs.update(is_covers=torch.ones(1, dtype=torch.bool), audio_cover_strength=.5,
            cover_noise_strength=.4, retake_variance=.3, retake_seed=74, sampler_mode="heun",
            non_cover_text_hidden_states=inputs["text_hidden_states"],
            non_cover_text_attention_mask=inputs["text_attention_mask"])
    if mode == "repaint":
        mask = torch.zeros(1, 48, dtype=torch.bool)
        mask[:, 12:36] = True
        inputs.update(repaint_mask=mask, clean_src_latents=inputs["src_latents"].clone(),
                      repaint_crossfade_frames=4)
    if mode == "sde":
        inputs["infer_method"] = "sde"
    observed = []
    forward = owner.decoder.forward
    def decoder(*a, **kw):
        observed.append((kw["hidden_states"].shape, kw["attention_mask"].shape))
        return forward(*a, **kw)
    with patch.object(owner.decoder, "forward", decoder), torch.no_grad():
        result = owner.generate_audio(**inputs)
    assert observed and all(x == torch.Size([1, 24, new]) and m == torch.Size([1, 24]) for x, m in observed)
    assert result["target_latents"].shape == (1, 24, new)
    assert torch.isfinite(result["target_latents"]).all()
    assert inputs["src_latents"].shape == (1, 48, 2)


def test_sampler_clean_and_masks_match_independent_projection():
    from core.models.acestep.vae_boundary import sampler_target_inputs
    pair = alignment(4)
    source = torch.randn(2, 47, 2)
    clean = torch.randn_like(source)
    mask = torch.zeros(2, 47, dtype=torch.bool)
    mask[:, 10:30] = True
    attention = torch.ones(2, 47)
    attention[:, 40:] = 0
    output = sampler_target_inputs(pair, source, attention, clean, mask, 9)
    teacher = F.interpolate(clean.transpose(1, 2), size=24, mode="area")
    teacher = F.pad(teacher, (0, 0, 0, 2))
    expected = F.conv1d(teacher, pair.output.projection.weight, pair.output.projection.bias, padding=1).transpose(1, 2)
    torch.testing.assert_close(output["clean"], expected)
    torch.testing.assert_close(output["mask"], F.interpolate(mask[:, None].float(), size=24, mode="nearest")[:, 0].bool())
    assert output["crossfade"] == 5
    assert output["source"].shape == (2, 24, 4)
    with pytest.raises(ValueError, match="both"):
        sampler_target_inputs(pair, source, attention, clean, None, 9)
