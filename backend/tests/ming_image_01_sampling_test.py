"""CPU flow and native RGBA sampling oracles."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Sampler CPU test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.conditioning import MingConditioning
from core.models.ming_image_01.flow import flow_training_pair, native_guidance, scheduler_prediction
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.sampling import MingSampler, MingGenerationCancelled
from core.models.ming_image_01.vendor.diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage


def make_sampler(profile):
    padding = "learned" if profile == "layer_decompose" else "zero_masked"
    config = {"dim": 24, "n_layers": 2, "n_refiner_layers": 1, "n_heads": 3,
              "n_kv_heads": 3, "cap_feat_dim": 8, "in_channels": 16,
              "axes_dims": [2, 2, 4], "axes_lens": [128, 32, 32],
              "alignment_padding_mode": padding, "multi_frame_output": profile == "layer_decompose"}
    transformer = create_transformer(config, device="cpu", dtype=torch.float32).eval()
    if profile == "layer_decompose":
        torch.nn.init.constant_(transformer.x_pad_token, 0.02)
        torch.nn.init.constant_(transformer.cap_pad_token, 0.03)
    vae = AutoencoderKLQwenImage(base_dim=32, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                                input_channels=4, z_dim=16, latents_mean=None, latents_std=None).eval()
    vae.input_channels = 4
    vae.register_to_config(scaling_factor=8.0064, shift_factor=0.0)
    artifact = SimpleNamespace(profile=profile, configs={"transformer": config, "vae": {"input_channels": 4}})
    components = SimpleNamespace(artifact=artifact, transformer=transformer, vae=vae,
                                 scheduler=FlowMatchEulerDiscreteScheduler(shift=6.0))
    return MingSampler(components)


def condition(batch=1):
    return MingConditioning(torch.randn(batch, 4, 8), torch.randn(batch, 3, 24))


def options():
    return {"width": 32, "height": 32, "steps": 2, "cfg": 1.0,
            "seed": 53, "requested_layers": 1}


def test_flow_sign_time_and_scheduler_one_step_oracle():
    clean, noise = torch.tensor([[2.0, 3.0]]), torch.tensor([[5.0, 7.0]])
    noisy, target, time = flow_training_pair(clean, noise, torch.ones(1))
    assert torch.equal(noisy, noise) and torch.equal(time, torch.zeros(1))
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(1)
    recovered = scheduler.step(scheduler_prediction(target), scheduler.timesteps[0], noisy).prev_sample
    torch.testing.assert_close(recovered, clean)
    clean_endpoint, _, time_endpoint = flow_training_pair(clean, noise, torch.zeros(1))
    assert torch.equal(clean_endpoint, clean) and torch.equal(time_endpoint, torch.ones(1))


def test_cfg_matches_native_positive_plus_extrapolation():
    positive, zero = torch.tensor([2.0]), torch.tensor([0.5])
    assert torch.equal(native_guidance(positive, zero, 2.0), torch.tensor([5.0]))
    assert torch.equal(native_guidance(positive, zero, 1.0), positive)
    with pytest.raises(ValueError):
        native_guidance(positive, zero, float("nan"))


def test_native_design_rgba_and_seed_reproducibility():
    sampler = make_sampler("generation_edit")
    conditioning = condition()
    first = sampler.run(task="text-to-image", conditioning=conditioning, **options())[0]
    second = sampler.run(task="text-to-image", conditioning=conditioning, **options())[0]
    assert first.values[0].mode == "RGBA" and first.values[0].size == (32, 32)
    assert first.values[0].tobytes() == second.values[0].tobytes()
    assert first.roles == ["image"]
    assert not torch.cuda.is_initialized()


def test_native_design_reference_edit():
    sampler = make_sampler("generation_edit")
    result = sampler.run(task="image-edit", conditioning=condition(),
                         reference_pixels=torch.randn(1, 4, 32, 32), **options())[0]
    assert result.values[0].mode == "RGBA"
    assert result.target_frames == 1


def test_native_layer_rgba_group_roles_and_frame_count():
    sampler = make_sampler("layer_decompose")
    args = {**options(), "requested_layers": 2, "cfg": 2.0}
    result = sampler.run(task="layer-decompose", conditioning=condition(),
                         reference_pixels=torch.randn(1, 4, 32, 32), **args)[0]
    assert result.roles == ["auxiliary_canvas", "layer_1", "layer_2"]
    assert len(result.values) == result.target_frames == 3
    assert all(image.mode == "RGBA" for image in result.values)
    assert not torch.cuda.is_initialized()


def test_batch_layer_outputs_use_frame_major_grouping():
    sampler = make_sampler("layer_decompose")
    conditioning = condition(2)
    result = sampler.run(task="layer-decompose", conditioning=conditioning,
                         reference_pixels=torch.randn(2, 4, 32, 32), output_type="latent", **options())
    assert len(result) == 2 and all(len(group.values) == 2 for group in result)
    assert all(value.shape == (16, 4, 4) for group in result for value in group.values)
    assert result[0].roles == ["auxiliary_canvas", "layer_1"]
    assert not torch.equal(result[0].values[0], result[1].values[0])


def test_cancel_stops_next_native_forward_and_removes_hook():
    sampler = make_sampler("generation_edit")
    state, progress = {"cancel": False}, []
    def callback(index, total, timestep, latents):
        progress.append(index)
        state["cancel"] = True
    with pytest.raises(MingGenerationCancelled):
        sampler.run(task="text-to-image", conditioning=condition(), cancelled=lambda: state["cancel"],
                    step_callback=callback, **options())
    assert progress == [0]
    assert not sampler.components.transformer._forward_pre_hooks
    assert not sampler.pipe.interrupt
    assert sampler.run(task="text-to-image", conditioning=condition(), **options())


@pytest.mark.parametrize("task, negative, count, reference", [
    ("layer-decompose", "", 2, True), ("text-to-image", "bad", 1, False),
    ("image-edit", "", 1, False), ("text-to-image", "", True, False),
])
def test_task_profile_and_negative_prompt_refusals(task, negative, count, reference):
    sampler = make_sampler("generation_edit")
    with pytest.raises(ValueError):
        sampler.run(task=task, conditioning=condition(), negative_prompt=negative,
                    reference_pixels=torch.randn(1, 4, 32, 32) if reference else None,
                    **{**options(), "requested_layers": count})



def test_frame_major_regrouping_oracle():
    sampler = make_sampler("layer_decompose")
    class OrderedPipe:
        _interrupt = False
        def __call__(self, **kwargs):
            assert kwargs["num_frames_per_prompt"] == 2
            assert len(kwargs["prompt_embeds"]) == 2
            assert all(torch.count_nonzero(value) == 0 for value in kwargs["negative_prompt_embeds"])
            return SimpleNamespace(images=torch.tensor([10, 20, 11, 21]))
        def maybe_free_model_hooks(self):
            pass
    sampler.pipe = OrderedPipe()
    groups = sampler.run(task="layer-decompose", conditioning=condition(2), output_type="latent",
                         reference_pixels=torch.zeros(2, 4, 32, 32), **options())
    assert [value.item() for value in groups[0].values] == [10, 11]
    assert [value.item() for value in groups[1].values] == [20, 21]


@pytest.mark.parametrize("patch", [
    {"steps": None}, {"cfg": None}, {"width": True}, {"seed": -1},
])
def test_invalid_sampling_parameters_fail_before_native_compute(patch):
    sampler = make_sampler("generation_edit")
    with pytest.raises(ValueError):
        sampler.run(task="text-to-image", conditioning=condition(), **{**options(), **patch})
