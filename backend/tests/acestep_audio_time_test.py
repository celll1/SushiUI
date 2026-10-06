"""Physical audio time grids and explicitly preserved legacy position policy."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from acestep_vae_boundary_test import small_decoder, small_owner, alignment
from acestep_audio_artifact_test import codec


@pytest.mark.parametrize("rewired,scale", [(False, 2.), (True, 2.), (False, 1.5), (True, 1.5)])
def test_rope_and_local_masks_preserve_source_time_across_grids(rewired, scale):
    from core.models.acestep.vendor.modeling_acestep_v15_turbo import AceStepDiTModel
    from core.models.acestep.vae_boundary import install_decoder_alignment
    from core.models.acestep.residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec, OPERATORS
    config = deepcopy(small_decoder().config)
    config.use_sliding_window = True
    config.layer_types = ["sliding_attention"] * 7
    model = AceStepDiTModel(config).eval()
    if rewired:
        graph = build_spec("acestep", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
                 down_factor=[2, 1], **OPERATORS)], native_config(config))
        restore_topology(model, graph)
    pair = alignment(4)
    if scale == 1.5:
        from core.models.components.audio_latent_adapter import AudioCodecContract, AudioAlignmentPair, PairedAudioCalibration
        pair = AudioAlignmentPair(AudioCodecContract(2, 48000, 2, 8, "aligned_hop_v1"),
                                  AudioCodecContract(4, 48000, 2, 12, "aligned_hop_v1"))
        fit = PairedAudioCalibration(pair, ridge=1e-3)
        fit.add(torch.randn(2, 2, 96), torch.randn(2, 4, 64))
        fit.finish()
    install_decoder_alignment(model, pair)
    assert all(layer.self_attn.sliding_window == int(3 / scale) for layer in model.layers)
    rope, masks = [], []
    hooks = [model.rotary_emb.register_forward_pre_hook(lambda m, a: rope.append(a[1].detach().clone()))]
    for layer in model.layers:
        hooks.append(layer.register_forward_pre_hook(lambda m, a: masks.append((a[3].detach().clone(), a[4].detach().clone()))))
    try:
        with torch.no_grad():
            context = torch.cat((torch.randn(1, 46, 4), torch.ones(1, 46, 4)), dim=-1)
            result = model(hidden_states=torch.randn(1, 46, 4), context_latents=context,
                timestep=torch.tensor([.5]), timestep_r=torch.tensor([.5]), attention_mask=torch.ones(1, 46),
                encoder_hidden_states=torch.randn(1, 5, 16), encoder_attention_mask=torch.ones(1, 5), use_cache=False)[0]
    finally:
        for hook in hooks:
            hook.remove()
    assert result.shape == (1, 46, 4)
    assert rope and masks
    for mask, positions in masks:
        allowed = (positions[:, :, None] - positions[:, None, :]).abs() <= 3
        torch.testing.assert_close(mask[:, 0] == 0, allowed)
    torch.testing.assert_close(rope[0], scale * torch.arange(24 if rewired else 23)[None].float())
    if rewired:
        torch.testing.assert_close(rope[1], torch.arange(12)[None].float() * 2 * scale + scale / 2)


def test_legacy_artifact_preserves_index_positions_when_restored_and_resaved(tmp_path):
    from core.models.acestep.vae_boundary import install_decoder_alignment
    from core.models.acestep.audio_artifact import export_bundle, KEY, DECODER_KEY
    from core.models.acestep.residual_unet import export_metadata, AUDIO_TIME_KEY
    from core.models.acestep.loader import _build_dit
    owner, source, target = small_owner(), codec(2, [2, 2]), codec(4, [2, 4])
    install_decoder_alignment(owner, alignment(4), position_policy="target_index_v1")
    meta, codec_state = export_bundle(owner, target, source)
    record = json.loads(meta[KEY])
    assert record["version"] == 2 and record["position_policy"] == "target_index_v1"
    record["version"] = 1
    del record["position_policy"]
    meta[KEY] = json.dumps(record)
    assert AUDIO_TIME_KEY not in json.loads(meta[DECODER_KEY])
    path = tmp_path / "legacy.safetensors"
    save_file({**{"net."+k: v.contiguous() for k, v in owner.state_dict().items()}, **codec_state},
              str(path), metadata=export_metadata(owner, audio_metadata=meta))
    restored, _ = _build_dit(str(path), torch.float32)
    assert restored.decoder._audio_latent_alignment.position_policy == "target_index_v1"
    assert not hasattr(restored.decoder.config, AUDIO_TIME_KEY)
    rewritten, _ = export_bundle(restored, restored._audio_restored_codec)
    assert json.loads(rewritten[KEY])["position_policy"] == "target_index_v1"


def test_time_policy_mismatch_is_rejected_before_allocating_model(tmp_path):
    from acestep_audio_artifact_test import artifact
    from core.models.acestep.loader import _build_dit
    from core.models.acestep.audio_artifact import KEY
    from safetensors import safe_open
    from safetensors.torch import load_file
    _, _, _, path = artifact(tmp_path, 4, True)
    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata()
    record = json.loads(meta[KEY])
    record["position_policy"] = "target_index_v1"
    meta[KEY] = json.dumps(record)
    save_file(load_file(str(path)), str(tmp_path / "bad.safetensors"), metadata=meta)
    with patch("core.models.acestep.vendor.AceStepConditionGenerationModel", side_effect=AssertionError("No allocation")):
        with pytest.raises(ValueError, match="decoder config"):
            _build_dit(str(tmp_path / "bad.safetensors"), torch.float32)
