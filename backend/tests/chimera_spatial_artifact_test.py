"""Aligned full artifacts with native tiny codecs/UNet and synthetic images."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.mark.parametrize("channels", [2, 6])
@pytest.mark.parametrize("kind", ["flow_velocity", "endpoint_observable_velocity", "endpoint_observable_residual"])
def test_full_adapter_export_restore_and_optimizer_continuation(tmp_path, channels, kind):
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL
    from sensenova_sdxl_chimera_p1_test import _build
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.wiring import SENSENOVA_SDXL_CHIMERA_WIRING
    from core.models.sensenova_sdxl_chimera.loader import load_chimera_artifact, preflight_chimera_artifact
    from core.models.sensenova_sdxl_chimera.artifact import prediction_contract
    from core.models.sensenova_sdxl_chimera.vae_alignment import unet_forward
    from core.training.arch.sensenova_sdxl_chimera import SenseNovaSDXLChimeraArchHandler
    from core.training.adapters.sensenova_sdxl_chimera_adapter import SenseNovaSDXLChimeraFullParameterAdapter
    from core.training.ops.sensenova_sdxl_chimera_ops import load_components

    torch.manual_seed(32)
    _und, _donor, _vae, artifact = _build(tmp_path, initialization="sdxl_transplant")
    native = load_chimera_artifact(artifact["directory"])
    target = AutoencoderKL(latent_channels=channels, block_out_channels=(4,)*5,
        down_block_types=("DownEncoderBlock2D",)*5, up_block_types=("UpDecoderBlock2D",)*5,
        layers_per_block=1, norm_num_groups=4, scaling_factor=.7, shift_factor=.2)
    directory = tmp_path/"target"
    target.save_pretrained(directory)
    resolved = resolve_vae_source(f"file:{directory}", arch="sensenova_sdxl_chimera")
    prediction = prediction_contract(kind, latent_mean=[.1, -.2, .3, -.4], latent_centered_second_moment=.6)
    trainer = SimpleNamespace(
        unet=native["unet"], vae=native["vae"], condition_bridge=native["condition_bridge"],
        chimera_manifest=native["manifest"], chimera_runtime_config=native["config"],
        chimera_frozen_vae_state=native["frozen_vae_state"], chimera_prediction=prediction,
        config=dict(training_method="full_finetune", chimera_training_stage="unet",
            vae_swap_calibration_samples=2, vae_swap_calibration_resolution=128,
            chimera_max_shard_bytes=100_000 if channels == 6 else 10*1024**3),
        weight_dtype=torch.float32, vae_dtype=torch.float32, device="cpu", bundle_vae=True,
        wiring=SENSENOVA_SDXL_CHIMERA_WIRING, text_encoder=None, text_encoder_2=None,
        _check_stop_requested=lambda: None,
    )
    trainer.arch = SenseNovaSDXLChimeraArchHandler(trainer)
    trainer.chimera_manifest["prediction"] = prediction
    # Only a fixture marker; this test does not measure conditioning quality.
    trainer.chimera_manifest["conditioning"]["bridge_state"] = "aligned"
    trainer._main_model_module = lambda: trainer.unet
    trainer.move_main_model_to_cpu = lambda: trainer.unet.to("cpu")
    trainer.move_main_model_to_gpu = lambda: trainer.unet.to("cpu")
    trainer.move_vae_to_cpu = lambda: trainer.vae.to("cpu")
    trainer.arch.apply_vae_swap(trainer, resolved)
    images = []
    for i in range(2):
        path = tmp_path/f"synthetic{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(path)
        images.append(dict(image_path=str(path)))
    trainer.arch.calibrate_before_training(trainer, [SimpleNamespace(items=images)], None)
    latent = trainer.vae.encode(torch.randn(1, 3, 128, 192)).latent_dist.mode().detach()
    timestep = torch.tensor([.4])
    kwargs = dict(encoder_hidden_states=torch.randn(1, 3, 2048),
        added_cond_kwargs=dict(text_embeds=torch.randn(1, 1280), time_ids=torch.ones(1, 6)))
    def step(model, contract, optimizer):
        optimizer.zero_grad()
        output = unet_forward(model, latent, timestep, prediction=contract, **kwargs)
        (output - 1).square().mean().backward()
        optimizer.step()
        return output
    trainer.optimizer = torch.optim.AdamW(trainer.unet.parameters(), lr=1e-4)
    step(trainer.unet, trainer.chimera_prediction, trainer.optimizer)
    optimizer_path = tmp_path/"optimizer.pt"
    torch.save(trainer.optimizer.state_dict(), optimizer_path)
    adapter = SenseNovaSDXLChimeraFullParameterAdapter(trainer)
    saved = adapter.write_checkpoint(1, 0, tmp_path/"full.safetensors")
    header = preflight_chimera_artifact(str(saved))
    assert header["vae_alignment"]["new_channels"] == channels
    restored = load_chimera_artifact(str(saved))
    assert restored["prediction"] == trainer.chimera_prediction
    assert restored["unet"]._chimera_source_prediction == trainer.unet._chimera_source_prediction
    assert restored["unet"].vae_latent_bridge.ready
    for name, value in trainer.unet.state_dict().items():
        torch.testing.assert_close(restored["unet"].state_dict()[name], value, rtol=0, atol=0)
    a = trainer.vae.encode(torch.ones(1, 3, 128, 192)).latent_dist.mode()
    b = restored["vae"].encode(torch.ones(1, 3, 128, 192)).latent_dist.mode()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert restored["vae"].decode(b).sample.shape == (1, 3, 128, 192)
    from core.models.sensenova_sdxl_chimera import pipeline_ops
    conditioning = pipeline_ops.ChimeraConditioning(
        encoder_hidden_states=kwargs["encoder_hidden_states"],
        pooled_text_embeds=kwargs["added_cond_kwargs"]["text_embeds"],
        context_positions=torch.zeros(1, 3, 3), attention_mask=torch.ones(1, 3, dtype=torch.bool),
        fingerprint="synthetic", key_lengths=(3,),
    )
    observed = []
    original_processors = dict(restored["unet"].attn_processors)
    def record_grid(model, args, forward_kwargs):
        observed.append((tuple(args[0].shape), forward_kwargs["added_cond_kwargs"]["time_ids"].clone()))
    handle = restored["unet"].register_forward_pre_hook(record_grid, with_kwargs=True)
    try:
        generated = pipeline_ops.sample_txt2img_latents(restored["unet"], conditioning, conditioning,
            height=128, width=192, steps=2, cfg_scale=3., cfg_mode="batched", seed=14,
            prediction=restored["prediction"])
        assert generated.shape == (1, channels, 8, 12)
        assert pipeline_ops.decode_latents(restored["vae"], generated).size == (192, 128)
        edited = pipeline_ops.sample_img2img_latents(restored["unet"], conditioning, conditioning, b,
            steps=2, denoising_strength=1., cfg_scale=3., cfg_mode="sequential", seed=14,
            prediction=restored["prediction"])
        assert edited.shape == generated.shape and torch.isfinite(edited).all()
        if channels == 2 and kind == "flow_velocity":
            from core.pipeline_backends.sensenova_sdxl_chimera import SenseNovaSDXLChimeraMixin
            components = dict(restored, understanding=dict(transformer=torch.nn.Linear(2, 2), tokenizer=object()))
            manager = SimpleNamespace(sensenova_sdxl_chimera_components=components, device="cpu")
            with patch.object(pipeline_ops, "build_conditioning", return_value=conditioning), \
                 patch.object(pipeline_ops, "prepare_generate_mask", wraps=pipeline_ops.prepare_generate_mask) as mask:
                source_image = Image.new("RGB", (192, 128), (20, 30, 40))
                result, _, _ = SenseNovaSDXLChimeraMixin._generate_edit_sensenova_sdxl_chimera(
                    manager, dict(width=192, height=128, steps=2, cfg_scale=1., seed=14),
                    source_image, Image.new("L", source_image.size, 0))
                assert result.size == (192, 128)
                assert mask.call_args.kwargs["latent_height"] == 8
                assert mask.call_args.kwargs["latent_width"] == 12
            with patch.object(pipeline_ops, "build_conditioning", side_effect=AssertionError("must fail before caption encoding")):
                with pytest.raises(ValueError, match="divisible by 32"):
                    SenseNovaSDXLChimeraMixin._generate_txt2img_sensenova_sdxl_chimera(
                        manager, dict(width=192, height=144, steps=2, cfg_scale=1., seed=14))
    finally:
        handle.remove()
        restored["unet"].set_attn_processor(dict(original_processors))
    assert observed
    for shape, time_ids in observed:
        assert shape[1:] == (max(4, channels), 8, 12)
        assert torch.equal(time_ids[:, :2], torch.tensor([[128., 192.]]).expand(shape[0], -1))
    with pytest.raises(ValueError, match="divisible by 32"):
        pipeline_ops.sample_txt2img_latents(restored["unet"], conditioning, None,
            height=144, width=192, steps=1, cfg_scale=1., seed=14, prediction=restored["prediction"])
    optimizer = torch.optim.AdamW(restored["unet"].parameters(), lr=1e-4)
    optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
    step(trainer.unet, trainer.chimera_prediction, trainer.optimizer)
    step(restored["unet"], restored["prediction"], optimizer)
    for name, value in trainer.unet.state_dict().items():
        torch.testing.assert_close(restored["unet"].state_dict()[name], value)
    reloaded_trainer = SimpleNamespace(model_path=str(saved), weight_dtype=torch.float32,
        vae_dtype=torch.float32, config=dict(chimera_training_stage="unet", chimera_flow_version="v2",
            chimera_v2_latent_mean=[99.]*4, chimera_v2_latent_centered_second_moment=99.))
    with patch("core.models.sensenova_sdxl_chimera.understanding.load_understanding_only",
               return_value=dict(transformer=torch.nn.Linear(2, 2), tokenizer=object())):
        load_components(reloaded_trainer)
    assert reloaded_trainer.chimera_prediction == restored["prediction"]
    assert reloaded_trainer.vae_identity is not None
    # Re-save after a runtime cast must retain the frozen original target codec.
    trainer.unet = restored["unet"]
    trainer.vae = restored["vae"].to(dtype=torch.float16)
    trainer.vae_identity = restored["vae_identity"]
    trainer.chimera_manifest = restored["manifest"]
    trainer.chimera_runtime_config = restored["config"]
    trainer.chimera_frozen_vae_state = restored["frozen_vae_state"]
    second = adapter.write_checkpoint(2, 0, tmp_path/"resaved")
    loaded_again = load_chimera_artifact(str(second))
    assert loaded_again["manifest"]["vae"]["content_hash"] == restored["manifest"]["vae"]["content_hash"]
    from copy import deepcopy
    from core.models.sensenova_sdxl_chimera.artifact import checkpoint_headers
    from core.models.sensenova_sdxl_chimera.vae_artifact import validate_metadata
    _shapes, metadata = checkpoint_headers(header["weights_path"])
    changed_manifest, changed_config = deepcopy(restored["manifest"]), deepcopy(restored["config"])
    changed_manifest["vae_alignment"]["calibration"]["images"] += 1
    changed_config["vae_alignment"] = changed_manifest["vae_alignment"]
    with pytest.raises(ValueError, match="weight metadata"):
        validate_metadata(changed_manifest, changed_config, metadata)
    changed_metadata = dict(metadata, **{"component.vae.hash":"bad"})
    with pytest.raises(ValueError, match="latent identity"):
        validate_metadata(restored["manifest"], restored["config"], changed_metadata)
    # Source/target contract drift is rejected before tensor loading.
    config_path = saved/"config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vae_alignment"]["source_prediction"]["type"] = "polar_tangent_flow"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="documents disagree"):
        preflight_chimera_artifact(str(saved))
