"""Native tiny codecs and UNet; calibration images are synthetic noise only."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("channels", [2, 6])
@pytest.mark.parametrize("kind", ["flow_velocity", "endpoint_observable_velocity", "endpoint_observable_residual"])
def test_calibration_preserves_source_contract_and_trains_target_grid(tmp_path, channels, kind):
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.wiring import SENSENOVA_SDXL_CHIMERA_WIRING
    from core.models.sensenova_sdxl_chimera.vae_alignment import unet_forward, PENDING_KEY
    from core.training.arch.sensenova_sdxl_chimera import SenseNovaSDXLChimeraArchHandler
    from core.training.vae_swap import validate_latent_io

    torch.manual_seed(94)
    def codec(c, blocks):
        return AutoencoderKL(latent_channels=c, block_out_channels=(4,)*blocks,
            down_block_types=("DownEncoderBlock2D",)*blocks,
            up_block_types=("UpDecoderBlock2D",)*blocks,
            layers_per_block=1, norm_num_groups=4, scaling_factor=.7, shift_factor=.2)
    old = codec(4, 4)
    target = codec(channels, 5)
    directory = tmp_path/"target"
    target.save_pretrained(directory)
    resolved = resolve_vae_source(f"file:{directory}", arch="sensenova_sdxl_chimera")
    unet = UNet2DConditionModel(in_channels=4, out_channels=4,
        block_out_channels=(8, 16), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), layers_per_block=1,
        cross_attention_dim=6, attention_head_dim=2, norm_num_groups=4)
    original_input = unet.conv_in.weight.detach().clone()
    native_prediction = dict(type=kind)
    if kind != "flow_velocity":
        native_prediction.update(latent_mean=[.1, -.2, .3, -.4], latent_centered_second_moment=.6)
    trainer = SimpleNamespace(
        unet=unet, vae=old, chimera_prediction=native_prediction,
        condition_bridge=torch.nn.Linear(6, 6),
        chimera_teacher=SimpleNamespace(text_encoder=torch.nn.Linear(6, 6), text_encoder_2=torch.nn.Linear(6, 6)),
        wiring=SENSENOVA_SDXL_CHIMERA_WIRING, config=dict(
            training_method="full_finetune", vae_swap_new_channel_init="paired_spatial",
            vae_swap_calibration_samples=2, vae_swap_calibration_resolution=128),
        weight_dtype=torch.float32, vae_dtype=torch.float32, device="cpu",
        text_encoder=None, text_encoder_2=None, _check_stop_requested=lambda: None,
    )
    trainer.arch = SenseNovaSDXLChimeraArchHandler(trainer)
    trainer._main_model_module = lambda: trainer.unet
    trainer.move_main_model_to_cpu = lambda: trainer.unet.to("cpu")
    trainer.move_main_model_to_gpu = lambda: trainer.unet.to("cpu")
    trainer.move_vae_to_cpu = lambda: trainer.vae.to("cpu")
    trainer.arch.apply_vae_swap(trainer, resolved)
    assert trainer.wiring.latent_channels == channels
    assert trainer.wiring.vae_scale_factor == 16 and trainer.wiring.vae_norm == "identity"
    assert trainer.arch.pixel_align == 32
    assert unet.config.in_channels == unet.config.out_channels == max(4, channels)
    torch.testing.assert_close(unet.conv_in.weight[:, :4], original_input)
    images = []
    for i in range(2):
        path = tmp_path/f"synthetic{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(path)
        images.append(dict(image_path=str(path)))
    rng_before = torch.random.get_rng_state().clone()
    def cancel():
        raise InterruptedError("synthetic cancellation")
    trainer._check_stop_requested = cancel
    with pytest.raises(InterruptedError):
        trainer.arch.calibrate_before_training(trainer, [SimpleNamespace(items=images)], None)
    assert hasattr(trainer, PENDING_KEY) and not unet.vae_latent_bridge.ready
    assert next(trainer.vae.parameters()).device.type == "cpu"
    assert next(getattr(trainer, PENDING_KEY)[0].parameters()).device.type == "cpu"
    trainer._check_stop_requested = lambda: None
    trainer.arch.calibrate_before_training(trainer, [SimpleNamespace(items=images)], None)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert not hasattr(trainer, PENDING_KEY)
    assert unet.vae_latent_bridge.ready
    assert validate_latent_io(trainer) == []
    source = unet._chimera_source_prediction
    if kind != "flow_velocity":
        assert source["latent_mean"][:4] == native_prediction["latent_mean"]
        assert source["latent_centered_second_moment"] == .6
        assert len(source["latent_mean"]) == max(4, channels)
        assert len(trainer.chimera_prediction["latent_mean"]) == channels
        assert trainer.chimera_prediction["latent_centered_second_moment"] > 0
    assert trainer.chimera_vae_calibration["images"] == 2
    assert trainer.chimera_vae_calibration["positions_per_channel"] == 128
    encoded = trainer.vae.encode(torch.randn(1, 3, 128, 192)).latent_dist.mode()
    assert encoded.shape == (1, channels, 8, 12)
    prediction = unet_forward(unet, encoded, torch.tensor([.4]),
        prediction=trainer.chimera_prediction, encoder_hidden_states=torch.randn(1, 3, 6))
    assert prediction.shape == encoded.shape and torch.isfinite(prediction).all()
    optimizer = torch.optim.SGD(unet.parameters(), lr=.01)
    before = unet.vae_latent_bridge.output.projection.weight.detach().clone()
    prediction.square().mean().backward()
    for bridge in (unet.vae_latent_bridge.input, unet.vae_latent_bridge.output):
        assert bridge.projection.weight.grad is not None
        assert torch.isfinite(bridge.projection.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, unet.vae_latent_bridge.output.projection.weight)
    calibrated = {key: value.clone() for key, value in unet.vae_latent_bridge.state_dict().items()}
    trainer.arch.calibrate_before_training(trainer, [], None)
    for key, value in calibrated.items():
        torch.testing.assert_close(unet.vae_latent_bridge.state_dict()[key], value, rtol=0, atol=0)


def test_streaming_statistics_weight_positions_and_are_numerically_stable():
    from core.models.sensenova_sdxl_chimera.vae_alignment import TargetLatentMoments
    moments = TargetLatentMoments(2)
    a = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4) + 1e8
    b = torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2) + 1e8
    moments.add(a)
    moments.add(b)
    values = torch.cat([value.movedim(1, 0).flatten(1) for value in (a, b)], dim=1)
    mean, moment = moments.finish()
    torch.testing.assert_close(torch.tensor(mean, dtype=torch.float64), values.mean(1))
    assert moment == pytest.approx(float(values.var(1, unbiased=False).mean()))
    assert moments.images == 3 and moments.count == 20


def test_polar_swap_refuses_before_loading_codec_or_mutating_unet():
    from core.training.arch.sensenova_sdxl_chimera import SenseNovaSDXLChimeraArchHandler
    trainer = SimpleNamespace(unet=torch.nn.Linear(2, 2), config={},
        chimera_prediction=dict(type="polar_tangent_flow"))
    handler = SenseNovaSDXLChimeraArchHandler(trainer)
    with pytest.raises(ValueError, match="polar"):
        handler.apply_vae_swap(trainer, object())
    assert not hasattr(trainer.unet, "vae_latent_bridge")
