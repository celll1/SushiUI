"""Native dual-stream gradients through shared backward and two-stage microbatches."""

import copy
import sys
from pathlib import Path
from types import MethodType

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming shared step attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.training_data import prepare_training_batch, shared_step_inputs, native_step_batch
from core.training.base_trainer import BaseTrainer
from core.training.arch import get_arch_handler
from ming_image_01_training_batch_test import components, trainer


def shared_owner(value, accumulation=1):
    owner = trainer(value)
    owner.is_ming_image_01 = True
    owner.is_qwen_image_21 = owner.is_sensenova = owner.is_sensenova_sdxl_chimera = False
    owner.use_grad_scaler = False
    owner._grad_accum_steps = accumulation
    owner._concept_sample_loss_chunks = []
    for name in ("_begin_repa_profile_call", "_reset_fused_group_counters", "_flush_fused_group_partials",
                 "_flush_deferred_predicted_latent", "_flush_deferred_controlnet_metrics", "_flush_deferred_extra_metrics",
                 "_finish_repa_profile_call"):
        setattr(owner, name, lambda *a, **k: None)
    owner._applied_updates_now = lambda: 0
    owner._flush_repa_loss_metric_after_backward = lambda: 0
    owner._note_partial_step_taint = lambda *a: None
    owner._execute_forward_backward = MethodType(BaseTrainer._execute_forward_backward, owner)
    owner._microbatch_two_stage = MethodType(BaseTrainer._microbatch_two_stage, owner)
    owner._flush_concept_sample_loss = MethodType(BaseTrainer._flush_concept_sample_loss, owner)
    owner._slice_aux = BaseTrainer._slice_aux
    owner.arch = get_arch_handler(owner)
    return owner


def step_args(batch, noise):
    packed, aux, refs = shared_step_inputs(batch)
    return dict(mnt_latents=batch.clean, mnt_text_embeddings=packed, mnt_attention_mask=aux,
        mnt_pooled_embeddings=None, timesteps=torch.tensor([.2, .5, .8]),
        debug_save_path=None, batch_captions=["word5"]*3, batch_reference_paths=None,
        alphas_cumprod_cached=None, use_condition_images=False, condition_images_batch=None,
        reference_latents_nested=refs, lens_latent_shape=None, mnt_repa_pixels=None, mnt_time_ids=None,
        cfg_drop_mask=torch.tensor([False, True, False]), mnt_noise=noise)


def prepared(value, pixels):
    return prepare_training_batch(value, prompts=["word5", "word6", "word7"],
        references=[[Image.new("RGBA", (32, 32), (50, 60, 70, 100))]]*3,
        target_pixels=pixels, frame_weights=torch.tensor([[0, 1, 2]]*3, dtype=torch.float32)
            if pixels.shape[2] == 3 else torch.ones(3, 1), resolution=32)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_shared_full_and_microbatch_all_component_gradients_match(profile):
    torch.manual_seed(37)
    full = components(profile)
    full.conditioner.set_trainable({"language", "vision", "connector", "queries", "projections"})
    micro = copy.deepcopy(full)
    frames = 1 if profile == "generation_edit" else 3
    pixels = torch.rand(3, 4, frames, 32, 32)*2-1
    batch_full, batch_micro = prepared(full, pixels), prepared(micro, pixels)
    noise = torch.randn_like(batch_full.clean)
    full_owner, micro_owner = shared_owner(full, 2), shared_owner(micro, 2)
    reported_full = full_owner._execute_forward_backward(**step_args(batch_full, noise))
    reported_micro = micro_owner._microbatch_two_stage(1, 3, step_args(batch_micro, noise))
    torch.testing.assert_close(torch.tensor(reported_full), torch.tensor(reported_micro), rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(torch.tensor(full_owner._concept_sample_loss_chunks),
                               torch.tensor(micro_owner._concept_sample_loss_chunks), rtol=3e-5, atol=1e-6)
    compared = 0
    for first_module, second_module in ((full.transformer, micro.transformer), (full.conditioner, micro.conditioner)):
        for first, second in zip(first_module.parameters(), second_module.parameters()):
            assert (first.grad is None) == (second.grad is None)
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, rtol=2e-4, atol=2e-6)
                compared += 1
    assert compared > 40
    for name in ("language", "vision", "connector", "queries", "projections"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for module in micro.conditioner.component_groups()[name] for p in module.parameters())
    before = micro.transformer.layers[0].attention.to_q.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for module in (micro.transformer, micro.conditioner)
                                  for p in module.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    assert not torch.equal(before, micro.transformer.layers[0].attention.to_q.weight)
    assert not torch.cuda.is_initialized()


def test_transport_roundtrip_preserves_ragged_references_and_both_streams():
    value = components("generation_edit")
    batch = prepared(value, torch.zeros(3, 4, 1, 32, 32))
    batch.reference_latents[1] = None
    packed, aux, refs = shared_step_inputs(batch)
    restored = native_step_batch(batch.clean, packed, aux, refs)
    torch.testing.assert_close(restored.conditioning.queries, batch.conditioning.queries, rtol=0, atol=0)
    torch.testing.assert_close(restored.conditioning.direct, batch.conditioning.direct, rtol=0, atol=0)
    assert restored.reference_latents[1] is None and restored.conditioning.direct.requires_grad
    chunk = native_step_batch(batch.clean[1:], packed[1:], BaseTrainer._slice_aux(aux, 1, 3), refs[1:])
    assert chunk.clean.shape[0] == 2 and chunk.frame_weights.shape == (2, 1)


@pytest.mark.parametrize("problem", ["layout", "batch", "frames", "refs"])
def test_incompatible_transport_refused(problem):
    value = components("generation_edit")
    batch = prepared(value, torch.zeros(3, 4, 1, 32, 32))
    packed, aux, refs = shared_step_inputs(batch)
    if problem == "layout": aux["ming_layout"] = (True, 8, 24)
    elif problem == "batch": packed = packed[:1]
    elif problem == "frames": aux["ming_frame_weights"] = torch.ones(3, 2)
    else: refs = refs[:1]
    with pytest.raises(ValueError): native_step_batch(batch.clean, packed, aux, refs)



@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_common_record_assembly_reencodes_trainable_conditioning_for_each_mnt(tmp_path, profile):
    from core.models.ming_image_01.dataset import annotate_datasets, prepare_shared_latents, shared_inputs_for_mnt
    from ming_image_01_dataset_test import shared_dataset, sidecar_path
    dataset = shared_dataset(tmp_path, 3)
    if profile == "generation_edit": sidecar_path(tmp_path / "canvas.png").unlink()
    annotate_datasets([dataset], profile=profile)
    # Design's extra RGBAs are ordinary separate samples; use one logical sample.
    pairs = [(dataset.items[0], dataset)]
    pairs[0][0]["width"] = pairs[0][0]["height"] = 32
    value = components(profile)
    value.conditioner.set_trainable({"language", "vision", "connector", "queries", "projections"})
    owner = shared_owner(value)
    owner.bucket_strategy = "crop"
    owner.ming_prompt_resolution = 32
    owner.ming_image_01_components = value
    prepared_batch = prepare_shared_latents(owner, pairs)
    assert prepared_batch.clean.shape == (1, 16, 1 if profile == "generation_edit" else 3, 4, 4)
    assert prepared_batch.frozen_conditioning is None and value.vae.device.type == "cpu"
    optimizer = torch.optim.AdamW([p for module in (value.transformer, value.conditioner)
                                  for p in module.parameters() if p.requires_grad], lr=.001)
    previous_packed = None
    for step in range(2):
        packed, aux, refs = shared_inputs_for_mnt(owner, prepared_batch)
        assert packed.grad_fn is not None and packed is not previous_packed
        assert aux["ming_frame_weights"].shape[1] == prepared_batch.clean.shape[2]
        args = step_args(native_step_batch(prepared_batch.clean, packed, aux, refs), torch.ones_like(prepared_batch.clean))
        args["timesteps"] = torch.tensor([.2 + .5*step])
        args["cfg_drop_mask"] = torch.tensor([False])
        args["batch_captions"] = ["word5"]
        owner._execute_forward_backward(**args)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in value.conditioner.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        previous_packed = packed
    assert prepared_batch.frozen_conditioning is None and not torch.cuda.is_initialized()


def test_frozen_record_conditioning_reused_only_until_trainability_changes(tmp_path):
    from core.models.ming_image_01.dataset import annotate_datasets, prepare_shared_latents, shared_inputs_for_mnt
    from ming_image_01_dataset_test import shared_dataset
    dataset = shared_dataset(tmp_path)
    annotate_datasets([dataset], profile="layer_decompose")
    item = dataset.items[0]
    item["width"] = item["height"] = 32
    value = components("layer_decompose")
    value.conditioner.set_trainable(set())
    owner = shared_owner(value)
    owner.bucket_strategy, owner.ming_prompt_resolution = "resize", 32
    owner.ming_image_01_components = value
    prepared_batch = prepare_shared_latents(owner, [(item, dataset)])
    first, _, _ = shared_inputs_for_mnt(owner, prepared_batch)
    cached = prepared_batch.frozen_conditioning
    assert cached is not None and not first.requires_grad
    second, _, _ = shared_inputs_for_mnt(owner, prepared_batch)
    assert prepared_batch.frozen_conditioning is cached
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    value.conditioner.set_trainable({"queries"})
    third, _, _ = shared_inputs_for_mnt(owner, prepared_batch)
    assert third.requires_grad and prepared_batch.frozen_conditioning is None
    assert not torch.cuda.is_initialized()
