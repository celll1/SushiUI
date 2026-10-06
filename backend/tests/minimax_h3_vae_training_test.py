"""Actual tiny H3/target codecs, synthetic calibration and Full continuation."""

from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from ltx2_vae_training_test import Trainer, images
from minimax_h3_video_artifact_test import forward


def setup(tmp_path, channels, scale, rewired=True, dtype=torch.float32):
    from diffusers import AutoencoderKL
    from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from core.models.minimax_h3.vendor.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from core.models.minimax_h3.residual_unet import native_config, restore_topology
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
    from core.models.common.vae_source import resolve_vae_source
    from core.rewiring.spec import build_spec, OPERATORS
    from core.training.arch.minimax_h3 import MiniMaxH3ArchHandler
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16, hidden_size=16,
        num_layers=7, num_refiner_layers=1, ffn_dim=32, in_channels=2, audio_in_channels=6, patch_size=(1, 2, 2),
        text_dim=10, freq_dim=8, time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=2, adaln_curve_grid=33)
    if rewired:
        graph = build_spec("minimax_h3", [dict(name="main", block_count=7)], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
                down_factor=[2, 2], **OPERATORS)], native_config(model.config))
        restore_topology(model, graph)
    source = AutoencoderKLMiniMaxH3(latent_channels=2, block_out_channels=(4,)*4, layers_per_block=1,
        spatial_downsample_factors=(2,)*4, temporal_downsample_factors=(1, 1, 2, 2), norm_num_groups=1,
        decoder_num_layers=1, decoder_num_attention_heads=1, decoder_attention_head_dim=8,
        decoder_num_register_tokens=1, latents_mean=(.2, .3), latents_std=(.8, 1.2))
    source.register_to_config(_class_name="AutoencoderKLMiniMaxH3")
    source.disable_tiling()
    depth = scale.bit_length()
    codec = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=channels, block_out_channels=(4,)*depth,
        down_block_types=("DownEncoderBlock2D",)*depth, up_block_types=("UpDecoderBlock2D",)*depth,
        norm_num_groups=1, scaling_factor=2., shift_factor=.25).to(dtype)
    codec.save_pretrained(tmp_path/"codec")
    resolved = resolve_vae_source("file:"+str(tmp_path/"codec"), arch="minimax_h3", download=False, load_weights=True)
    trainer = Trainer(transformer=MiniMaxH3BlockLoopWrapper(model), transformer_original=model,
        vae=source.eval().requires_grad_(False),
        config={"vae_swap_new_channel_init":"auto", "vae_swap_calibration_samples":2,
                "vae_swap_calibration_resolution":128, "training_method":"full_finetune"},
        train_unet=True, bundle_vae=True, unet_lr=1e-3, vae_dtype=torch.float32,
        device=torch.device("cpu"), events=[], checks=0, stop_at=None,
        text_encoder=None, audio_vae=None, optimizer=None, is_minimax_h3=True,
        training_dtype=torch.float32, minimax_h3_components={"audio_latent_channels":6},
        log_extra_metric=lambda name,value:None)
    trainer.arch = MiniMaxH3ArchHandler(trainer)
    if rewired:
        trainer.config.update(dit_rewiring_mode="configure", dit_rewiring_spec=deepcopy(graph))
    return trainer, resolved


@pytest.mark.parametrize("channels,scale,rewired,dtype", [
    (1, 32, False, torch.float32), (4, 32, True, torch.bfloat16), (4, 8, True, torch.float32),
])
def test_calibration_optimizer_ownership_and_saved_continuation(tmp_path, channels, scale, rewired, dtype):
    from core.training.vae_swap import apply_latent_space
    from core.training.adapters.minimax_h3_adapter import MiniMaxH3AlignedFullParameterAdapter
    from core.models.minimax_h3.vae_training import PENDING
    from core.models.minimax_h3.video_artifact import load_checkpoint
    trainer, resolved = setup(tmp_path, channels, scale, rewired, dtype)
    trainer.config["vae_swap_source"] = resolved.source
    original_config = deepcopy(trainer.config)
    apply_latent_space(trainer, None)
    if rewired:
        from core.models.minimax_h3.residual_unet import configure_training_topology
        configure_training_topology(trainer.transformer, trainer.config)
        configure_training_topology(trainer.transformer, deepcopy(original_config), resuming=True)
    assert trainer.vae.dtype == dtype
    assert trainer.vae_identity.latent_hash == resolved.latent_hash
    assert trainer.wiring.latent_channels == channels and trainer.wiring.vae_scale_factor == scale
    assert trainer.arch.pixel_align == scale*2*(2 if rewired else 1)
    rebound = type(trainer.arch)(trainer)
    assert rebound.pixel_align == trainer.arch.pixel_align
    assert rebound.temporal.frame_multiple == 1 and rebound.temporal.latent_frames(5) == 5
    pair = trainer.transformer_original._video_latent_alignment
    assert not pair.ready
    adapter = MiniMaxH3AlignedFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    trainer.optimizer = torch.optim.AdamW(adapter.setup_trainable_parameters())
    owned = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    with pytest.raises(ValueError, match="calibrated"):
        forward(trainer.transformer)
    with pytest.raises(ValueError, match="calibrated"):
        adapter.save_checkpoint(0, 0, tmp_path/"unready")
    progress = []
    trainer._vae_alignment_progress_callback = lambda **kw: progress.append(kw)
    data = images(tmp_path)
    teacher_shapes, target_shapes = [], []
    old_encode = getattr(trainer, PENDING)[0].encode
    target_encode = trainer.vae.encode
    def record_teacher(value, *args, **kwargs):
        teacher_shapes.append(tuple(value.shape[-2:]))
        return old_encode(value, *args, **kwargs)
    def record_target(value, *args, **kwargs):
        target_shapes.append(tuple(value.shape[-2:]))
        return target_encode(value, *args, **kwargs)
    getattr(trainer, PENDING)[0].encode = record_teacher
    trainer.vae.encode = record_target
    rng = torch.random.get_rng_state().clone()
    trainer.arch.calibrate_before_training(trainer, data, None)
    trainer.vae.encode = target_encode
    expected_side = 64 if scale == 32 else 128
    assert teacher_shapes == [(expected_side, expected_side)]*2
    assert target_shapes == [(128, 128)]*2
    assert pair.ready and not hasattr(trainer, PENDING)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert trainer.events == ["park", "restore"]
    assert [p["step"] for p in progress] == [1, 2, 3, 4]
    assert owned == [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    from core.training.ops.minimax_h3_ops import vae_encode, vae_encode_clip, train_step
    pixels = torch.randn(1,3,128,128)
    expected = trainer.vae.encode(pixels[:,:,None].to(dtype)).latent_dist.mode().float()
    actual = vae_encode(trainer,pixels)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(vae_encode_clip(trainer,pixels[0][None]),expected,rtol=0,atol=0)
    loss,_,_ = train_step(trainer,actual,torch.randn(1,3,10),
        {"num_text_tokens":torch.tensor([3])},timesteps=torch.tensor([.3]))
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pair.parameters())
    trainer.optimizer.step()
    path = adapter.save_checkpoint(1, 1, tmp_path/"full")
    restored, vae, identity = load_checkpoint(path, torch.float32)
    if rewired:
        configure_training_topology(restored, original_config, resuming=True)
    assert vae.dtype == dtype and identity.latent_hash == resolved.latent_hash
    for key, value in trainer.transformer.transformer.state_dict().items():
        torch.testing.assert_close(value, restored.transformer.state_dict()[key], rtol=0, atol=0)
    restored.train(trainer.transformer.training)
    with torch.no_grad():
        for left, right in zip(forward(trainer.transformer), forward(restored)):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    resumed = Trainer(transformer=restored, transformer_original=restored.transformer, vae=vae,
        config=trainer.config.copy(), vae_dtype=dtype, bundle_vae=True, is_minimax_h3=True,
        train_unet=True, unet_lr=1e-3)
    resumed.arch = type(trainer.arch)(resumed)
    apply_latent_space(resumed, identity)
    assert not hasattr(resumed, PENDING)
    resumed.arch.calibrate_before_training(resumed, [], None)
    resumed_adapter = MiniMaxH3AlignedFullParameterAdapter(resumed)
    resumed_adapter.prepare_models_for_training()
    optimizer = torch.optim.AdamW(resumed_adapter.setup_trainable_parameters())
    optimizer.load_state_dict(deepcopy(trainer.optimizer.state_dict()))
    trainer.optimizer.zero_grad(set_to_none=True)
    for model, opt in ((trainer.transformer, trainer.optimizer), (restored, optimizer)):
        sum(value.float().square().mean() for value in forward(model)).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    for (_, left), (_, right) in zip(trainer.transformer.named_parameters(), restored.named_parameters()):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("stop", [1, 3])
def test_calibration_cancellation_restores_residency_and_retries(tmp_path, stop):
    from core.models.minimax_h3.vae_training import PENDING
    trainer, resolved = setup(tmp_path, 4, 32)
    trainer.arch.apply_vae_swap(trainer, resolved)
    trainer.stop_at = stop
    data = images(tmp_path)
    with pytest.raises(RuntimeError, match="stop requested"):
        trainer.arch.calibrate_before_training(trainer, data, None)
    assert trainer.events == ["park", "restore"]
    assert hasattr(trainer, PENDING) and not trainer.transformer_original._video_latent_alignment.ready
    assert trainer.vae.device.type == "cpu" and getattr(trainer, PENDING)[0].device.type == "cpu"
    trainer.stop_at = None
    trainer.arch.calibrate_before_training(trainer, data, None)
    assert not hasattr(trainer, PENDING) and trainer.transformer_original._video_latent_alignment.ready


def test_non_full_or_invalid_target_refused_before_resize(tmp_path):
    trainer, resolved = setup(tmp_path, 4, 32)
    original = trainer.transformer
    trainer.config["training_method"] = "lora"
    with pytest.raises(ValueError, match="full_finetune"):
        trainer.arch.apply_vae_swap(trainer, resolved)
    assert trainer.transformer is original
    trainer.config["training_method"] = "full_finetune"
    wrong = replace(resolved, content_hash="wrong")
    with pytest.raises(ValueError, match="resolved identity"):
        trainer.arch.apply_vae_swap(trainer, wrong)
    assert trainer.transformer is original


def test_checkpoint_as_base_dispatches_h3_with_resume_contract(monkeypatch):
    from core.training.base_trainer import BaseTrainer
    from core.model_loader import ModelLoader
    from core.training.ops import minimax_h3_ops
    trainer = SimpleNamespace(log_prefix="[synthetic H3]")
    seen = []
    monkeypatch.setattr(ModelLoader,"detect_model_type",staticmethod(lambda path:"minimax_h3"))
    monkeypatch.setattr(minimax_h3_ops,"load_components",lambda owner,**kw:seen.append((owner,kw)))
    BaseTrainer._load_checkpoint_as_base(trainer,"synthetic_full.safetensors")
    assert seen == [(trainer,{"resuming":True})]
    assert trainer.model_path == "synthetic_full.safetensors" and trainer.is_minimax_h3
