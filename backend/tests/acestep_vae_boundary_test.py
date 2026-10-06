"""Real small ACE decoder boundaries; conditioning owners use synthetic stubs."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from core.models.components.audio_latent_adapter import AudioCodecContract, AudioAlignmentPair, PairedAudioCalibration


def small_decoder():
    from core.models.acestep.vendor.configuration_acestep_v15 import AceStepConfig
    from core.models.acestep.vendor.modeling_acestep_v15_turbo import AceStepDiTModel
    config = AceStepConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=7,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, patch_size=2,
        in_channels=6, audio_acoustic_hidden_dim=2, encoder_hidden_size=16,
        sliding_window=3, attention_dropout=0)
    config._attn_implementation = "eager"
    return AceStepDiTModel(config)


def alignment(new):
    old = AudioCodecContract(2, 48000, 2, 4, "aligned_hop_v1")
    target = AudioCodecContract(new, 48000, 2, 8, "aligned_hop_v1")
    pair = AudioAlignmentPair(old, target)
    fit = PairedAudioCalibration(pair, ridge=1e-3)
    fit.add(torch.randn(2, 2, 96), torch.randn(2, new, 48))
    fit.finish()
    return pair


@pytest.mark.parametrize("new,rewired", [(1, False), (4, False), (1, True), (4, True)])
def test_real_decoder_target_grid_independent_oracle_and_optimizer(new, rewired, tmp_path):
    from core.models.acestep.vae_boundary import install_decoder_alignment
    from core.models.acestep.residual_unet import native_config, restore_topology
    from core.rewiring.spec import OPERATORS, build_spec
    torch.manual_seed(62)
    model, pair = small_decoder(), alignment(new)
    old_io = model.proj_in[1].weight.detach().clone()
    if rewired:
        graph = build_spec("acestep", [{"name":"main", "block_count":7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6, down_factor=[2,1], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5, down_factor=[2,1], **OPERATORS)], native_config(model.config))
        restore_topology(model, graph)
    rng = torch.get_rng_state().clone()
    install_decoder_alignment(model, pair)
    assert torch.equal(rng, torch.get_rng_state())
    width = max(2, new)
    for group in range(3):
        torch.testing.assert_close(model.proj_in[1].weight[:,group*width:group*width+2], old_io[:,group*2:group*2+2], rtol=0, atol=0)
    if rewired and new > 2:
        assert model._residual_unet_spec["base_config_sha256"] != graph["base_config_sha256"]
    reference = deepcopy(model)
    del reference._audio_latent_alignment
    x, source = torch.randn(2, 48, new), torch.randn(2, 48, new)
    mask = torch.ones_like(source)
    sigma = torch.tensor([.3, .8])
    args = dict(timestep=sigma, timestep_r=sigma, attention_mask=torch.ones(2,48),
        encoder_hidden_states=torch.randn(2,5,16), encoder_attention_mask=torch.ones(2,5), use_cache=False)
    internal_x = F.conv1d(x.transpose(1,2), pair.input.projection.weight, padding=1)
    internal_x += (1-sigma[:,None,None])*pair.input.projection.bias[None,:,None]
    internal_source = F.conv1d(source.transpose(1,2), pair.input.projection.weight, pair.input.projection.bias, padding=1)
    internal_context = torch.cat((internal_source.transpose(1,2), torch.ones(2,48,width)), -1)
    expected_v = reference(hidden_states=internal_x.transpose(1,2), context_latents=internal_context, **args)[0]
    expected = F.conv1d(expected_v.transpose(1,2), pair.output.projection.weight, padding=1)-pair.output.projection.bias[None,:,None]
    result = model(hidden_states=x, context_latents=torch.cat((source,mask),-1), **args)[0]
    assert result.shape == (2,48,new)
    torch.testing.assert_close(result, expected.transpose(1,2), atol=2e-6, rtol=2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result.square().mean().backward()
    assert pair.input.projection.weight.grad is not None
    assert pair.output.projection.weight.grad is not None
    if new > 2:
        assert torch.count_nonzero(model.proj_out[1].weight.grad[:,2:]) > 0
    optimizer.step()
    path = tmp_path/"state.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), path)
    restored = small_decoder()
    if rewired:
        restore_topology(restored, graph)
    install_decoder_alignment(restored, alignment(new))
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state["model"])
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(state["optimizer"])
    torch.testing.assert_close(restored(hidden_states=x, context_latents=torch.cat((source,mask),-1), **args)[0], model(hidden_states=x, context_latents=torch.cat((source,mask),-1), **args)[0], rtol=0, atol=0)
    assert len(resumed_optimizer.state) == len(optimizer.state)


def test_actual_prepare_condition_preserves_native_rvq_domain():
    from core.models.acestep.vae_boundary import install_decoder_alignment
    from core.models.acestep.vendor.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
    class Encoder(nn.Module):
        def forward(self, **kwargs):
            assert kwargs["refer_audio_acoustic_hidden_states_packed"].shape == (2,48,2)
            return torch.zeros(2,5,16), torch.ones(2,5)
    class Tokenizer(nn.Module):
        def forward(self, hidden):
            assert hidden.shape == (2,24,2,2)
            return hidden.reshape(2,48,2), None
    owner = AceStepConditionGenerationModel.__new__(AceStepConditionGenerationModel)
    nn.Module.__init__(owner)
    owner.decoder, owner.encoder, owner.tokenizer, owner.detokenizer = small_decoder(), Encoder(), Tokenizer(), nn.Identity()
    owner.config = deepcopy(owner.decoder.config)
    owner.config.pool_window_size = 2
    pair = alignment(4)
    install_decoder_alignment(owner.decoder, pair)
    assert owner.config.audio_acoustic_hidden_dim == 2
    assert owner.decoder.config.audio_acoustic_hidden_dim == 4
    src, hints = torch.randn(2,48,2), torch.randn(2,48,2)
    class Quantizer(nn.Module):
        def get_output_from_indices(self, codes):
            assert codes.shape == (2,24)
            return hints
    owner.tokenizer.quantizer = Quantizer()
    kwargs = dict(text_hidden_states=torch.zeros(2,3,16), text_attention_mask=torch.ones(2,3),
        lyric_hidden_states=torch.zeros(2,3,16), lyric_attention_mask=torch.ones(2,3),
        refer_audio_acoustic_hidden_states_packed=torch.zeros(2,48,2), refer_audio_order_mask=torch.arange(2),
        hidden_states=hints, attention_mask=torch.ones(2,48), silence_latent=torch.zeros(1,48,2),
        src_latents=src, chunk_masks=torch.ones(2,48,2), is_covers=torch.tensor([False,True]))
    for precomputed, codes in ((None, None), (hints, None), (None, torch.zeros(2,24,dtype=torch.long))):
        _, _, context = owner.prepare_condition(**kwargs, precomputed_lm_hints_25Hz=precomputed, audio_codes=codes)
        assert context.shape == (2,24,8)
        selected = torch.stack((src[0], hints[1])).transpose(1,2)
        teacher = F.avg_pool1d(selected,2)
        teacher = F.pad(teacher,(0,0,0,2))
        expected = pair.output(teacher).transpose(1,2)
        torch.testing.assert_close(context[...,:4], expected)
        assert torch.equal(context[...,4:], torch.ones(2,24,4))


def test_context_validation_keeps_mask_semantics():
    from core.models.acestep.vae_boundary import native_context_to_target, decoder_inputs
    pair = alignment(4)
    source = torch.randn(1,48,2)
    mask = torch.ones_like(source)
    mask[:,16:32] = 0
    result = native_context_to_target(pair, torch.cat((source,mask),-1))
    assert torch.equal(result[:,8:16,4:], torch.zeros(1,8,4))
    mask[...,0] = 0
    with pytest.raises(ValueError, match="uniform"):
        native_context_to_target(pair, torch.cat((source,mask),-1))
    assert native_context_to_target(pair, torch.zeros(1,47,4)).shape == (1,24,8)
    with pytest.raises(ValueError, match="duration"):
        native_context_to_target(pair, torch.zeros(1,47,4), target_frames=24)
    with pytest.raises(ValueError, match="geometry"):
        decoder_inputs(pair, torch.zeros(1,24,4), torch.zeros(1,48,8), .5)


def test_installer_rejects_inconsistent_patch_before_mutation():
    from core.models.acestep.vae_boundary import install_decoder_alignment
    model, pair = small_decoder(), alignment(4)
    config = deepcopy(model.config.to_dict())
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.proj_in[1].stride = (1,)
    with pytest.raises(ValueError, match="patch"):
        install_decoder_alignment(model, pair)
    assert model.config.to_dict() == config
    assert not hasattr(model, "_audio_latent_alignment")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def small_owner():
    from core.models.acestep.vendor import AceStepConditionGenerationModel
    config = deepcopy(small_decoder().config)
    for key, value in dict(num_lyric_encoder_hidden_layers=1, num_timbre_encoder_hidden_layers=1,
            num_attention_pooler_hidden_layers=1, num_audio_decoder_hidden_layers=1,
            text_hidden_dim=8, timbre_hidden_dim=2, fsq_dim=16, fsq_input_levels=[2,2],
            pool_window_size=2).items():
        setattr(config, key, value)
    return AceStepConditionGenerationModel(config)


def test_owner_separates_rvq_config_from_decoder_topology_and_resume(tmp_path):
    from types import SimpleNamespace
    from core.models.acestep.vae_boundary import install_decoder_alignment
    from core.models.acestep.residual_unet import (native_config, restore_topology,
        decoder_identity_config, configure_training_topology, resolve_generation_topology,
        generation_metadata, model_topology_metadata, export_metadata)
    from core.rewiring.spec import OPERATORS, build_spec, read_checkpoint_metadata
    from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
    owner = small_owner()
    source_config = deepcopy(owner.config.to_dict())
    source_state = {k: v.clone() for k, v in owner.state_dict().items() if not k.startswith("decoder.")}
    graph = build_spec("acestep", [{"name":"main", "block_count":7}], [
        dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,1],**OPERATORS)], native_config(owner.config))
    restore_topology(owner, graph)
    install_decoder_alignment(owner, alignment(4))
    assert owner.config.to_dict() == source_config
    for k, v in source_state.items():
        torch.testing.assert_close(owner.state_dict()[k], v, rtol=0, atol=0)
    actual = owner._residual_unet_spec
    assert actual == owner.decoder._residual_unet_spec
    assert actual["base_config_sha256"] != graph["base_config_sha256"]
    assert decoder_identity_config(owner)["audio_acoustic_hidden_dim"] == 4
    assert read_checkpoint_metadata(model_topology_metadata(owner)) == actual
    config = {}
    assert configure_training_topology(owner, config) == actual
    assert configure_training_topology(owner, {}, resuming=True, resume_spec=actual, install=False) == actual
    assert resolve_generation_topology(owner, []) == actual
    assert generation_metadata(owner, 24)["native_token_grid"] == [12,1]
    with pytest.raises(ValueError, match="identity|config|topology"):
        configure_training_topology(owner, {}, resuming=True, resume_spec=graph, install=False)
    with pytest.raises(NotImplementedError, match="bundled codec"):
        export_metadata(owner)
    path = tmp_path/"unbundled.safetensors"
    with pytest.raises(ValueError, match="mismatched"):
        AceStepFullParameterAdapter(SimpleNamespace(transformer=owner)).write_checkpoint(1,1,path)
    assert not path.exists()
    owner.config.intermediate_size += 8
    with pytest.raises(ValueError, match="owner/decoder"):
        model_topology_metadata(owner)
