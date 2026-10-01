"""Symbolic residency ordering and native CPU cache/gradient boundaries."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def forbidden_cuda(*a, **k):
    raise AssertionError("Residency test attempted CUDA initialization")

torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda

from core.models.ming_image_01.training_residency import MingTrainingResidency


class Spy:
    def __init__(self, name, events, *, cacheable=True, fail=None):
        self.name, self.events, self.cacheable, self.fail = name, events, cacheable, fail

    def to(self, *, device):
        device = str(device)
        self.events.append((self.name, device))
        if self.fail == device:
            self.fail = None
            raise RuntimeError("transfer failed")
        return self


def setup(*, mode="swap_onthefly", device="cuda", trainable=False):
    events = []
    components = SimpleNamespace(transformer=Spy("dit", events), vae=Spy("vae", events),
        conditioner=Spy("te", events, cacheable=not trainable))
    return components, events, MingTrainingResidency(components, device=device, latent_mode=mode, text_mode=mode)


@pytest.mark.parametrize("mode", ["swap_onthefly", "pre_encoded_cache", "onthefly_gpu"])
@pytest.mark.parametrize("phase,name", [("vae", "vae"), ("conditioner", "te")])
def test_selected_mode_controls_main_model_transfers(mode, phase, name):
    components, events, residency = setup(mode=mode)
    with residency(phase):
        events.append(("encode", phase))
    expected = [(name, "cuda"), ("encode", phase), (name, "cpu")]
    if mode != "onthefly_gpu": expected = [("dit", "cpu"), *expected, ("dit", "cuda")]
    assert events == expected and residency.active is None
    assert not torch.cuda.is_initialized()


def test_trainable_conditioner_stays_resident_for_backward():
    _, events, residency = setup(trainable=True)
    with residency("conditioner"): pass
    assert events == [("dit", "cpu"), ("te", "cuda"), ("dit", "cuda")]


@pytest.mark.parametrize("failure", ["encode", "stage", "release"])
def test_failure_restores_main_model_and_releases_phase(failure):
    components, events, residency = setup()
    if failure == "stage": components.vae.fail = "cuda"
    if failure == "release": components.vae.fail = "cpu"
    with pytest.raises(RuntimeError):
        with residency("vae"):
            if failure == "encode": raise RuntimeError("encode failed")
    assert events[-1] == ("dit", "cuda") and ("vae", "cpu") in events
    assert residency.active is None
    with residency("vae"): pass


def test_cpu_execution_has_no_main_model_swap_and_nested_phases_refuse():
    _, events, residency = setup(device="cpu")
    with residency("vae"):
        with pytest.raises(RuntimeError, match="overlap"):
            with residency("conditioner"): pass
    assert events == [("vae", "cpu"), ("vae", "cpu")]
    with pytest.raises(ValueError, match="Unknown"):
        with residency("other"): pass


def test_native_cache_hits_skip_encoder_residency(tmp_path):
    from ming_image_01_cache_test import cached_record_setup
    from core.models.ming_image_01.dataset import prepare_record_latents
    value, item, context = cached_record_setup(tmp_path)
    events = []
    policy = MingTrainingResidency(value, device="cpu", latent_mode="pre_encoded_cache", text_mode="pre_encoded_cache")
    class RecordPhase:
        def __call__(self, phase):
            events.append(phase)
            return policy(phase)
    first = prepare_record_latents(value, [item], resolution=32, cache_context=context, residency=RecordPhase()).for_step()
    assert events == ["vae", "conditioner"]
    events.clear()
    second = prepare_record_latents(value, [item], resolution=32, cache_context=context, residency=RecordPhase()).for_step()
    assert events == []
    torch.testing.assert_close(first.clean, second.clean, rtol=0, atol=0)
    torch.testing.assert_close(first.conditioning.direct, second.conditioning.direct, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_native_trainable_conditioner_retains_live_graph(tmp_path):
    from ming_image_01_cache_test import cached_record_setup
    from core.models.ming_image_01.dataset import prepare_record_latents
    value, item, _ = cached_record_setup(tmp_path)
    value.conditioner.set_trainable({"language", "vision", "connector", "queries", "projections"})
    policy = MingTrainingResidency(value, device="cpu", latent_mode="swap_onthefly", text_mode="swap_onthefly")
    prepared = prepare_record_latents(value, [item], resolution=32, residency=policy)
    for _ in range(2):
        value.conditioner.zero_grad(set_to_none=True)
        condition = prepared.for_step().conditioning
        (condition.direct.square().mean() + condition.queries.square().mean()).backward()
        assert next(value.conditioner.native.proj_in.parameters()).grad.abs().sum() > 0
    assert policy.active is None and not torch.cuda.is_initialized()
