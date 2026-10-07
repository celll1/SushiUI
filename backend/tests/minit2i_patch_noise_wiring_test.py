import pytest
import torch

from backend.tests.minit2i_spatial_boundaries_test import model_for, no_cuda
from backend.tests.latent_adapter_contract_test import declaration, identity
from core.models.components.spatial_swap import install, artifact_metadata, read_declaration
from core.models.components.vae_alignment import SpatialLatentBridge
from core.models.components.image_noise_context import noise_scope, exposure_noise
from core.models.components.patch_noise_alignment import POLICY


def aligned_model():
    model = model_for(8)
    spec = declaration(8, 4)
    spec['patch_noise'] = dict(policy=POLICY, channels=4, bridge_channels=8, hidden_channels=8, patch_size=2)
    install(model, spec)
    pair = model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(4, 8), .2), identity(SpatialLatentBridge(8, 4), .3))
    pair.patch_noise.capture(model.model.net.img_embedder.proj1.weight, pair.input.projection,
                            model.model.net.img_embedder.proj1)
    assert pair.ready
    return model


def test_nested_exposure_noise_restores_outer_scope_on_failure():
    first, second = torch.randn(1, 4, 8, 12), torch.randn(1, 4, 8, 12)
    with noise_scope(first):
        with pytest.raises(RuntimeError):
            with noise_scope(second):
                assert torch.equal(exposure_noise(second), second)
                raise RuntimeError('reference failed')
        assert torch.equal(exposure_noise(first.expand(3, -1, -1, -1)), first.repeat(3, 1, 1, 1))
    with pytest.raises(ValueError, match='exposure'):
        exposure_noise(first)


def test_metadata_restore_prediction_and_training_gradient():
    model = aligned_model()
    restored = model_for(8)
    install(restored, read_declaration(artifact_metadata(model)))
    restored.load_state_dict(model.state_dict(), strict=True)
    image = torch.randn(2, 4, 8, 12, requires_grad=True)
    noise = torch.randn_like(image)
    t, text, mask = torch.tensor([.2, .8]), torch.randn(2, 5, 16), torch.ones(2, 5)
    with noise_scope(noise):
        actual = model(image, t, text, mask)
        torch.testing.assert_close(restored(image, t, text, mask), actual)
    actual.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert model.model.net.img_embedder.proj1.weight.grad.abs().sum() > 0
    assert not model.model.net.img_embedder.proj1._forward_hooks
    assert all(not value.requires_grad for value in model.vae_latent_bridge.patch_noise.buffers())


@pytest.mark.parametrize('mode', ['text', 'image', 'inpaint'])
def test_generation_modes_supply_noise_and_release_scope(mode):
    from core.models.minit2i.minit2i_pipeline_ops import denoise_loop, denoise_loop_img2img, denoise_loop_inpaint
    model = aligned_model()
    text, mask = torch.randn(1, 5, 16), torch.ones(1, 5)
    common = dict(transformer=model, text=text, mask=mask, num_inference_steps=2, cfg_scale=2.,
                  cfg_interval=(0., 1.), device='cpu', dtype=torch.float32, seed=63, clamp_output=False)
    if mode == 'text':
        result = denoise_loop(height=8, width=12, channels=4, **common)
    else:
        common.update(init_image=torch.randn(1, 4, 8, 12), denoising_strength=.8)
        result = (denoise_loop_img2img(**common) if mode == 'image' else
                  denoise_loop_inpaint(mask_latent=torch.ones(1, 1, 8, 12), **common))
    assert result.shape == (1, 4, 8, 12) and torch.isfinite(result).all()
    with pytest.raises(ValueError, match='exposure'):
        exposure_noise(result)


@pytest.mark.parametrize('multiple', [False, True])
def test_style_captures_use_each_reference_noise(multiple):
    from core.models.minit2i.minit2i_pipeline_ops import denoise_loop
    from core.inference.reference_style import StyleTransferConfig
    model = aligned_model()
    cfg = StyleTransferConfig(axes_dims=(8, 8), rope_layout='rotate_half', block_range=(0, 2))
    references = [(cfg, torch.randn(1, 4, 8, 12), torch.randn(1, 4, 8, 12)) for _ in range(2)]
    captured = []
    hook = model.register_forward_pre_hook(lambda module, args: captured.append(exposure_noise(args[0]).clone()))
    kwargs = (dict(style_refs=references) if multiple else
              dict(style_cfg=cfg, style_ref_x0=references[0][1], style_eps_ref=references[0][2]))
    try:
        result = denoise_loop(model, torch.randn(1, 5, 16), torch.ones(1, 5), 8, 12, 2,
            2., (0., 1.), 'cpu', torch.float32, channels=4, seed=63, clamp_output=False, **kwargs)
    finally:
        hook.remove()
    assert torch.isfinite(result).all()
    assert any(torch.equal(noise, references[0][2]) for noise in captured)
    if multiple:
        assert any(torch.equal(noise, references[1][2]) for noise in captured)
    assert all(getattr(block, '_style_ctx', None) is None for block in model.model.net.double_blocks)
    with pytest.raises(ValueError, match='exposure'):
        exposure_noise(result)


def test_sampler_exception_releases_noise_and_projection_hook():
    from core.models.minit2i.minit2i_pipeline_ops import denoise_loop
    model = aligned_model()
    def fail(*args):
        raise RuntimeError('cancelled test')
    with pytest.raises(RuntimeError, match='cancelled test'):
        denoise_loop(model, torch.randn(1, 5, 16), torch.ones(1, 5), 8, 12, 2,
            2., (0., 1.), 'cpu', torch.float32, channels=4, seed=63,
            clamp_output=False, progress_callback=fail)
    assert not model.model.net.img_embedder.proj1._forward_hooks
    with pytest.raises(ValueError, match='exposure'):
        exposure_noise(torch.zeros(1, 4, 8, 12))


@pytest.mark.parametrize('rewired', [False, True])
@pytest.mark.parametrize('sharded', [False, True])
def test_directory_restores_patch_alignment_and_optional_topology(tmp_path, rewired, sharded):
    import json
    from core.models.minit2i.residual_unet import (
        load_directory_transformer, native_config, restore_topology, model_topology_metadata,
    )
    from core.models.common.single_file_format import save_single_file_state
    from core.rewiring.spec import build_spec, OPERATORS
    model = aligned_model()
    config = native_config(model)
    if rewired:
        spec = build_spec('minit2i', [{'name': 'main', 'block_count': config['depth_double']}],
            [dict(id='s', stream='main', parent=None, start_block=1, end_block=2,
                  down_factor=[2, 2], **OPERATORS)], config)
        restore_topology(model, spec)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    metadata = dict(artifact_metadata(model), **model_topology_metadata(model))
    save_single_file_state(model.state_dict(), metadata,
        str(tmp_path / 'diffusion_pytorch_model.safetensors'),
        max_shard_bytes=20000 if sharded else 10**9)
    restored = load_directory_transformer(tmp_path, torch.float32)
    assert restored.vae_latent_bridge.ready
    assert getattr(restored, '_residual_unet_spec', None) == getattr(model, '_residual_unet_spec', None)
    image, noise = torch.randn(1, 4, 8, 12), torch.randn(1, 4, 8, 12)
    t, text, mask = torch.tensor([.3]), torch.randn(1, 5, 16), torch.ones(1, 5)
    with noise_scope(noise):
        torch.testing.assert_close(restored(image, t, text, mask), model(image, t, text, mask))
    for key, value in model.vae_latent_bridge.patch_noise.state_dict().items():
        assert torch.equal(restored.vae_latent_bridge.patch_noise.state_dict()[key], value)


def test_directory_refuses_disagreeing_alignment_before_allocation(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from core.models.minit2i.residual_unet import load_directory_transformer
    from core.models.minit2i.vendor.transformer import MiniT2IMMJiTModel
    metadata = artifact_metadata(aligned_model())
    save_file({'placeholder': torch.ones(1)}, str(tmp_path / 'first.safetensors'), metadata=metadata)
    save_file({'placeholder': torch.ones(1)}, str(tmp_path / 'second.safetensors'))
    monkeypatch.setattr(MiniT2IMMJiTModel, '__init__', lambda *a, **k: pytest.fail('model allocation'))
    with pytest.raises(ValueError, match='alignment declarations disagree'):
        load_directory_transformer(tmp_path, torch.float32)
