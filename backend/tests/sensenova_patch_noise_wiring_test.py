from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sensenova_latent_migration_test import _LatentTree, _vision
from sensenova_spatial_boundaries_test import _pair
from sensenova_training_core_test import _Cache
from core.models.components.patch_noise_alignment import POLICY, PatchNoiseAlignment
from core.models.components.spatial_swap import artifact_metadata, install, read_declaration, validate_loaded
from core.models.sensenova import sensenova_pipeline_ops as ops
from core.models.sensenova.vae_alignment import patch_projection_transform, validate_patch_alignment
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel
from core.training.ops.sensenova_ops import SenseNovaTrainingPrefix, train_step


class AlignedTree(_LatentTree):
    extract_feature = NEOChatModel.extract_feature
    _apply_time_schedule = NEOChatModel._apply_time_schedule


def aligned_model(dtype=torch.float32):
    torch.manual_seed(82)
    model = AlignedTree(channels=16, scale=16)
    model.fm_modules['vision_model_mot_gen'] = _vision(16, 2)
    nn.init.normal_(model.fm_modules['fm_head'].conv2.weight, std=.03)
    pair = _pair(3, 16)
    pair.declaration['old_scale'] = 1
    declaration = dict(policy=POLICY, channels=16, bridge_channels=16, hidden_channels=32, patch_size=2)
    pair.declaration.update(patch_noise=declaration, patch_noise_source_stride=32)
    pair.patch_noise = PatchNoiseAlignment(declaration)
    projection = model.fm_modules['vision_model_mot_gen'].embeddings.patch_embedding
    pair.patch_noise.capture(torch.randn(32, 3, 16, 16) * .02, pair.input.projection, projection)
    model.vae_latent_bridge = pair
    return model.to(dtype=dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_real_projection_corrects_noise_before_gelu_and_preserves_gradient(dtype):
    model = aligned_model(dtype)
    image = torch.randn(2, 16, 8, 12).to(dtype).requires_grad_()
    noise = torch.randn_like(image)
    t = torch.tensor([.25, .75])
    transform = patch_projection_transform(model, image, t, 1., noise)
    pair = model.vae_latent_bridge
    mapped = pair.input.diffusion_state(image, alpha=t)
    projection = model.fm_modules['vision_model_mot_gen'].embeddings.patch_embedding
    patches = model.patchify(mapped, 2, channel_first=True).reshape(-1, 16, 2, 2)
    raw = projection(patches)
    corrected = transform(raw).reshape(2, 4, 6, 32).permute(0, 3, 1, 2)
    current = F.conv2d(F.conv2d(noise.float(), pair.patch_noise.incoming, padding=1),
                       pair.patch_noise.projection, stride=2)
    desired = F.conv2d(noise.float(), pair.patch_noise.white, stride=2)
    source_scale = ops.compute_noise_scale(model, 4., 6., 1)
    expected = (raw.reshape(2, 4, 6, 32).permute(0, 3, 1, 2).float()
                + (1-t)[:, None, None, None] * (source_scale * desired-current)).to(dtype)
    torch.testing.assert_close(corrected, expected)
    shape = SimpleNamespace(batch_size=2, grid_h=4, grid_w=6, token_h=2, token_w=3)
    z, embedding, _ = ops._build_step_context(model, shape, image, t, 1.,
                                             enable_grad=True, exposure_noise=noise)
    assert z.shape == (2, 6, 16*16)
    assert embedding.shape == (2, 6, 64)
    embedding.float().square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert projection.weight.grad.abs().sum() > 0
    assert not projection._forward_hooks


@pytest.mark.parametrize('mode,cap', [('resolution', 16.), ('dynamic_sqrt', 16.), ('resolution', .01)])
def test_source_gain_uses_original_stride_and_noise_cap(mode, cap):
    model = aligned_model()
    model.noise_scale_mode, model.noise_scale_max_value = mode, cap
    image = torch.randn(1, 16, 8, 12)
    noise = torch.randn_like(image)
    target = ops.compute_noise_scale(model, 4, 6, 2)
    source = ops.compute_noise_scale(model, 4, 6, 1)
    transform = patch_projection_transform(model, image, torch.tensor(0.), target, noise)
    patch = model.vae_latent_bridge.patch_noise
    current = F.conv2d(F.conv2d(noise, patch.incoming, padding=1), patch.projection, stride=2)
    projected = current.permute(0, 2, 3, 1).reshape(-1, 32, 1, 1)
    actual = transform(projected).reshape(1, 4, 6, 32).permute(0, 3, 1, 2)
    expected = F.conv2d(noise, patch.white, stride=2) * (source/target)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_checkpoint_restores_patch_coefficients_and_rejects_geometry():
    model = aligned_model()
    restored = aligned_model()
    declaration = read_declaration(artifact_metadata(model))
    install(restored, declaration)
    restored.load_state_dict(model.state_dict(), strict=True)
    validate_loaded(restored)
    validate_patch_alignment(restored)
    for name, value in model.vae_latent_bridge.patch_noise.state_dict().items():
        assert torch.equal(value, restored.vae_latent_bridge.patch_noise.state_dict()[name])
    image, noise = torch.randn(2, 16, 8, 12), torch.randn(2, 16, 8, 12)
    shape = SimpleNamespace(batch_size=2, grid_h=4, grid_w=6, token_h=2, token_w=3)
    args = (shape, image, torch.tensor([.2, .8]), 1.)
    torch.testing.assert_close(ops._build_step_context(model, *args, exposure_noise=noise)[1],
                               ops._build_step_context(restored, *args, exposure_noise=noise)[1], rtol=0, atol=0)
    restored.vae_latent_bridge.declaration.pop('patch_noise_source_stride')
    with pytest.raises(ValueError, match='source token stride'):
        validate_patch_alignment(restored)
    restored.vae_latent_bridge.declaration['patch_noise_source_stride'] = 32
    restored.vae_latent_bridge.patch_noise.declaration['hidden_channels'] += 1
    with pytest.raises(ValueError, match='generation projection'):
        validate_patch_alignment(restored)


def test_training_passes_scaled_exposure_noise_and_trains_bridges(monkeypatch):
    model = aligned_model()
    image, noise = torch.randn(1, 16, 8, 12), torch.randn(1, 16, 8, 12)
    monkeypatch.setattr('core.training.mnt.training_noise_like', lambda trainer, value: noise)
    original = ops._build_step_context
    seen = []
    def record(*args, **kwargs):
        seen.append(kwargs['exposure_noise'].clone())
        return original(*args, **kwargs)
    monkeypatch.setattr(ops, '_build_step_context', record)
    trainer = SimpleNamespace(transformer=model, device=torch.device('cpu'), training_dtype=torch.float32,
                              gradient_checkpointing=False, defer_extra_metric=lambda *args: None)
    cache = _Cache()
    loss, _, _ = train_step(trainer, images=image, prefix=SenseNovaTrainingPrefix(cache, text_length=3),
                            timesteps=torch.tensor([.25]))
    loss.backward()
    torch.testing.assert_close(seen[0], noise * ops.compute_noise_scale(model, 4, 6, 2), rtol=0, atol=0)
    assert torch.isfinite(loss)
    for bridge in (model.vae_latent_bridge.input, model.vae_latent_bridge.output):
        assert bridge.projection.weight.grad.abs().sum() > 0
    assert all(not buffer.requires_grad for buffer in model.vae_latent_bridge.patch_noise.buffers())


def test_missing_exposure_is_refused_and_clean_legacy_needs_no_noise():
    model = aligned_model()
    image = torch.randn(1, 16, 8, 12)
    with pytest.raises(ValueError, match='exposure'):
        patch_projection_transform(model, image, torch.tensor(.5), 1., None)
    legacy = deepcopy(model)
    del legacy.vae_latent_bridge.patch_noise
    legacy.vae_latent_bridge.declaration.pop('patch_noise')
    legacy.vae_latent_bridge.declaration.pop('patch_noise_source_stride')
    assert patch_projection_transform(legacy, image, torch.tensor(.5), 1., None) is None


def prefix_for(model):
    return ops.SenseNovaPrefix(device=torch.device('cpu'), dtype=torch.float32, batch_size=1,
        token_h=2, token_w=3, grid_h=4, grid_w=6, merge_size=2, image_size=(192, 128),
        gen_size=(12, 8), cond_past_key_values=_Cache(), cond_indexes_image=torch.zeros(3, 6),
        cond_attention_mask={}, uncond_past_key_values=_Cache(), encode_cfg_scale=2.)


@pytest.mark.parametrize('mode', ['text', 'image', 'inpaint'])
def test_generation_modes_pass_the_drawn_noise_not_the_noised_image(monkeypatch, mode):
    model = aligned_model()
    prefix = prefix_for(model)
    noise, clean = torch.randn(1, 16, 8, 12), torch.randn(1, 16, 8, 12)
    monkeypatch.setattr(ops, 'prepare_noise', lambda *a, **k: noise)
    monkeypatch.setattr(ops, 'to_gen_space', lambda *a, **k: clean)
    monkeypatch.setattr(ops, 'prepare_mask', lambda *a, **k: torch.ones(1, 1, 8, 12))
    seen = []
    def run(owner, shape, image, ts, start, *args, **kwargs):
        assert kwargs['exposure_noise'] is noise
        if mode == 'text':
            assert image is noise
        else:
            torch.testing.assert_close(image, clean*ts[start]+noise*(1-ts[start]))
        seen.append(True)
        return image
    monkeypatch.setattr(ops, '_euler_run', run)
    common = dict(cfg_scale=2., timestep_shift=1., num_inference_steps=4, seed=63)
    if mode == 'text':
        ops.denoise_loop(model, prefix, **common)
    elif mode == 'image':
        ops.denoise_loop_img2img(model, prefix, object(), .5, **common)
    else:
        ops.denoise_loop_inpaint(model, prefix, object(), object(), .5, **common)
    assert seen == [True]


@pytest.mark.parametrize('failure', [False, True])
def test_euler_cfg_reuses_one_embedding_and_exception_leaves_no_hook(monkeypatch, failure):
    model, prefix = aligned_model(), prefix_for(None)
    noise = torch.randn(1, 16, 8, 12)
    seen = []
    def predict(owner, shape, embedding, time_embedding, z, t, scale, branch='cond'):
        seen.append((branch, embedding))
        if failure:
            raise RuntimeError('cancelled')
        return torch.zeros_like(z)
    monkeypatch.setattr(ops, '_predict_v_branch', predict)
    args = (model, prefix, noise, torch.tensor([0., .5, 1.]), 0, 2., (0., 1.), 'none', 1.)
    if failure:
        with pytest.raises(RuntimeError, match='cancelled'):
            ops._euler_run(*args, exposure_noise=noise)
    else:
        result = ops._euler_run(*args, exposure_noise=noise)
        assert torch.isfinite(result).all()
        assert [entry[0] for entry in seen] == ['cond', 'uncond', 'cond', 'uncond']
        assert seen[0][1] is seen[1][1] and seen[2][1] is seen[3][1]
    assert prefix.consumed
    assert not model.fm_modules['vision_model_mot_gen'].embeddings.patch_embedding._forward_hooks


def test_each_style_reference_passes_its_own_noise(monkeypatch):
    model, prefix = aligned_model(), prefix_for(None)
    noises = [torch.randn(1, 16, 8, 12) for _ in range(2)]
    references = [(SimpleNamespace(is_step_active=lambda *a: True, step_progress=lambda *a: .5),
                   torch.randn(1, 16, 8, 12), noise) for noise in noises]
    original, seen = ops._build_step_context, []
    def build(*args, **kwargs):
        seen.append(kwargs['exposure_noise'])
        return original(*args, **kwargs)
    monkeypatch.setattr(ops, '_build_step_context', build)
    monkeypatch.setattr(ops, '_predict_v_branch', lambda *a, **k: torch.zeros_like(a[4]))
    ops._style_capture_multi(model, prefix, references, 0, 2, torch.tensor(.5), 1.)
    assert len(seen) == 2 and all(a is b for a, b in zip(seen, noises))
    from core.models.sensenova.vendor.modeling_qwen3 import Qwen3Attention
    assert Qwen3Attention._style_ctx is None
