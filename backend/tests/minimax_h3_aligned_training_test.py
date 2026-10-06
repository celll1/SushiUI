"""Synthetic aligned H3 trainer ownership, codec precision and Full export."""

from types import SimpleNamespace

import pytest
import torch

from minimax_h3_video_artifact_test import fixture, forward


@pytest.mark.parametrize("channels", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aligned_full_ownership_save_and_resume(tmp_path, channels, dtype):
    from core.training.adapters.minimax_h3_adapter import MiniMaxH3AlignedFullParameterAdapter
    from core.training.ops.minimax_h3_ops import bind_video_codec
    from core.training.arch.minimax_h3 import MiniMaxH3ArchHandler
    from core.models.minimax_h3.video_artifact import load_checkpoint
    trainer = fixture(tmp_path, channels, dtype)
    trainer.train_unet = True
    trainer.unet_lr = 1e-3
    trainer.text_encoder = torch.nn.Linear(3, 3)
    trainer.audio_vae = torch.nn.Linear(3, 3)
    trainer.arch = MiniMaxH3ArchHandler()
    bind_video_codec(trainer, {"vae_identity": trainer.vae_identity})
    assert trainer.vae_dtype == dtype
    assert next(trainer.vae.parameters()).dtype == dtype
    assert trainer.wiring.latent_channels == channels
    assert trainer.wiring.vae_scale_factor == 2
    assert trainer.wiring.vae_norm == "identity"
    assert trainer.arch.temporal.latent_frames(5) == 5
    assert trainer.arch.pixel_align >= 4
    adapter = MiniMaxH3AlignedFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    groups = adapter.setup_trainable_parameters()
    assert len(groups) == 1 and groups[0]["lr"] == 1e-3
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in trainer.transformer.parameters()}
    for module in (trainer.vae, trainer.text_encoder, trainer.audio_vae):
        assert not module.training and not any(p.requires_grad for p in module.parameters())
    optimizer = torch.optim.AdamW(groups)
    sum(value.float().square().mean() for value in forward(trainer.transformer)).backward()
    pair = trainer.transformer.transformer._video_latent_alignment
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pair.parameters())
    optimizer.step()
    path = adapter.save_checkpoint(3, 1, tmp_path/"full")
    restored, vae, identity = load_checkpoint(path, dtype)
    assert identity.latent_hash == trainer.vae_identity.latent_hash
    assert next(vae.parameters()).dtype == dtype
    for left, right in zip(forward(trainer.transformer), forward(restored)):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    trainer.train_unet = False
    adapter.prepare_models_for_training()
    assert adapter.arch_param_groups() == []


def test_quantized_or_uncalibrated_base_refused_before_mutation(tmp_path):
    from core.training.adapters.minimax_h3_adapter import MiniMaxH3AlignedFullParameterAdapter
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    trainer = fixture(tmp_path, 4)
    trainer.transformer.requires_grad_(False)
    trainer.transformer.transformer.quantized_probe = Fp8Linear(4, 4, False, torch.float32)
    adapter = MiniMaxH3AlignedFullParameterAdapter(trainer)
    for method in (adapter.prepare_models_for_training, adapter.arch_param_groups):
        with pytest.raises(NotImplementedError, match="weight-only quantized"):
            method()
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    del trainer.transformer.transformer.quantized_probe
    trainer.transformer.transformer._video_latent_alignment.input._ready = False
    with pytest.raises(ValueError, match="calibrated"):
        adapter.prepare_models_for_training()


def test_codec_binding_rejects_changed_identity_and_clock(tmp_path):
    from dataclasses import replace
    from core.training.ops.minimax_h3_ops import bind_video_codec
    trainer = fixture(tmp_path, 4)
    with pytest.raises(ValueError, match="identity"):
        bind_video_codec(trainer, {"vae_identity": replace(trainer.vae_identity, content_hash="wrong")})
    trainer.transformer.transformer._video_codec_contract = None
    with pytest.raises(ValueError, match="contracts disagree"):
        bind_video_codec(trainer, {"vae_identity": trainer.vae_identity})


def test_native_precision_policy_stays_fp16():
    from core.training.ops.minimax_h3_ops import bind_video_codec
    trainer = SimpleNamespace(vae=torch.nn.Linear(2, 2).float())
    bind_video_codec(trainer, {})
    assert trainer.vae_dtype == torch.float16
    assert next(trainer.vae.parameters()).dtype == torch.float16
