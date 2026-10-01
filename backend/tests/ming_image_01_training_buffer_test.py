"""Native CPU swap windows, live conditioning and source-drift boundaries."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def forbidden_cuda(*a, **k): raise AssertionError("Buffer test attempted CUDA initialization")
torch.cuda.get_device_capability=lambda *a, **k:(8,9)
torch.cuda.is_available=lambda:False
torch.cuda._lazy_init=forbidden_cuda
torch._C._cuda_init=forbidden_cuda

from core.models.ming_image_01.training_buffer import MingBatchBuffer, close_training_buffer
from core.models.ming_image_01.training_residency import MingTrainingResidency
from ming_image_01_cache_test import cached_record_setup
from ming_image_01_training_batch_test import trainer


def setup(tmp_path, *, trainable=False):
    value, item, _ = cached_record_setup(tmp_path)
    value.conditioner.set_trainable({"language", "queries", "connector", "projections"} if trainable else set())
    owner = trainer(value)
    owner.ming_image_01_components, owner.ming_cache_context = value, None
    owner.bucket_strategy, owner.ming_prompt_resolution, owner.run_seed = "resize", 32, 651
    owner._check_stop_requested = lambda: None
    policy = MingTrainingResidency(value, device="cpu", latent_mode="swap_onthefly", text_mode="swap_onthefly")
    events = []
    class Trace:
        modes = policy.modes
        @contextmanager
        def __call__(self, phase):
            events.append(("enter", phase))
            with policy(phase):
                try: yield
                finally: events.append(("exit", phase))
    owner.ming_training_residency = Trace()
    batches = [[({"_ming_record": item.record, "width": 32, "height": 32,
                  "caption": "word5" if index % 2 == 0 else "word5 word6"},
                 SimpleNamespace(unique_id=f"dataset-{index}"))] for index in range(3)]
    buffer = MingBatchBuffer(owner, batches, epoch=2, batch_offset=5, latent_interval=2, text_interval=3)
    owner.ming_batch_buffer = buffer
    return value, owner, batches, buffer, events, policy


def test_independent_windows_are_cpu_bounded_and_group_transfers(tmp_path):
    value, owner, batches, buffer, events, policy = setup(tmp_path)
    first = buffer.prepare(0, batches[0])
    assert len(buffer.latents) == 1 and len(buffer.conditions) == 2
    assert first.clean.device.type == "cpu" and first.frozen_conditioning.direct.device.type == "cpu"
    assert not first.frozen_conditioning.direct.requires_grad
    assert events == [("enter", "vae"), ("exit", "vae"), ("enter", "conditioner"), ("exit", "conditioner")]
    events.clear()
    second = buffer.prepare(1, batches[1])
    assert events == [] and not buffer.latents and len(buffer.conditions) == 1
    assert first.for_step().conditioning.direct.shape[1] < second.for_step().conditioning.direct.shape[1]
    third = buffer.prepare(2, batches[2])
    assert events == [("enter", "vae"), ("exit", "vae")]
    assert not buffer.latents and not buffer.conditions and policy.active is None
    close_training_buffer(owner)
    assert owner.ming_batch_buffer is None and buffer.closed
    with pytest.raises(ValueError, match="cursor"): buffer.prepare(0, batches[0])
    assert not torch.cuda.is_initialized()


def test_trainable_conditioner_is_not_prefilled_and_replays_live_mnt_graph(tmp_path):
    value, _, batches, buffer, events, _ = setup(tmp_path, trainable=True)
    prepared = buffer.prepare(0, batches[0])
    assert not buffer.conditions and prepared.frozen_conditioning is None
    assert all(phase != "conditioner" for _action, phase in events)
    for _ in range(2):
        value.conditioner.zero_grad(set_to_none=True)
        condition = prepared.for_step().conditioning
        (condition.direct.square().mean()+condition.queries.square().mean()).backward()
        assert next(value.conditioner.native.proj_in.parameters()).grad.abs().sum() > 0
    assert prepared.frozen_conditioning is None
    buffer.close()


def test_window_stop_restores_phase_and_cleanup_drops_partial_buffers(tmp_path):
    _, owner, batches, buffer, _, policy = setup(tmp_path)
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 2: raise KeyboardInterrupt("stop")
    owner._check_stop_requested = stop
    with pytest.raises(KeyboardInterrupt): buffer.prepare(0, batches[0])
    assert policy.active is None
    close_training_buffer(owner)
    assert not buffer.latents and not buffer.conditions


def test_caption_change_invalidates_prefilled_conditioning(tmp_path):
    _, _, batches, buffer, events, _ = setup(tmp_path)
    buffer.prepare(0, batches[0])
    events.clear()
    previous_length = buffer.conditions[1][1].direct.shape[1]
    batches[1][0][0]["caption"] = "word5 word6 word5"
    second = buffer.prepare(1, batches[1])
    assert ("enter", "conditioner") in events
    assert second.for_step().conditioning.direct.shape[1] > previous_length
    buffer.close()


def test_source_edit_after_prefill_is_refused_before_consumption(tmp_path):
    _, _, batches, buffer, _, _ = setup(tmp_path)
    buffer.prepare(0, batches[0])
    record = batches[1][0][0]["_ming_record"]
    record.targets[0].path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"): buffer.prepare(1, batches[1])
    buffer.close()


def test_buffered_frozen_conditioning_runs_native_dit_backward_and_update(tmp_path):
    from core.models.ming_image_01.training import flow_loss
    value, _, batches, buffer, _, _ = setup(tmp_path)
    value.transformer.requires_grad_(True)
    optimizer = torch.optim.AdamW(value.transformer.parameters(), lr=0.001)
    prepared = buffer.prepare(0, batches[0])
    batch = prepared.for_step()
    before = {name: parameter.detach().clone() for name, parameter in value.transformer.named_parameters()}
    result = flow_loss(value.transformer, clean=batch.clean, noise=torch.randn_like(batch.clean),
        sigma=torch.tensor([0.4]), conditioning=batch.conditioning,
        frame_weights=batch.frame_weights, reference_latents=batch.reference_latents)
    result.loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in value.transformer.parameters())
    assert all(parameter.grad is None for parameter in value.conditioner.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], parameter) for name, parameter in value.transformer.named_parameters())
    buffer.close()
    assert not torch.cuda.is_initialized()
