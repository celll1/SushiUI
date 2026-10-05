"""Optimizer state ownership and exception-safe encoder residency, without CUDA."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.training import optimizer_residency as residency


class DeviceTensor:
    def __init__(self, device, events, label):
        self.device, self.events, self.label = torch.device(device), events, label

    def to(self, device):
        self.events.append((self.label, str(device)))
        return DeviceTensor(device, self.events, self.label)


def symbolic_torch(monkeypatch):
    monkeypatch.setattr(residency, "torch", SimpleNamespace(
        Tensor=(torch.Tensor, DeviceTensor), cuda=SimpleNamespace(empty_cache=lambda: None)))


@pytest.mark.parametrize("fail", [False, True])
def test_parking_restores_only_owned_cuda_state(monkeypatch, fail):
    symbolic_torch(monkeypatch)
    events = []
    owned, other = object(), object()
    step = torch.tensor(2.)
    state = {"step": step, "nested": [{"moment": DeviceTensor("cuda:1", events, "moment")}]}
    opt = SimpleNamespace(state={owned: state, other: {"moment": DeviceTensor("cuda:0", events, "other")}})
    try:
        with residency.park_optimizer_state([opt], [owned]):
            assert state["nested"][0]["moment"].device.type == "cpu"
            assert state["step"] is step
            if fail:
                raise RuntimeError("encoding failed")
    except RuntimeError:
        assert fail
    assert events == [("moment", "cpu"), ("moment", "cuda:1")]
    assert state["nested"][0]["moment"].device == torch.device("cuda:1")


def test_cpu_adamw_state_and_next_update_are_unchanged():
    a = torch.nn.Parameter(torch.tensor([1., 2.]))
    b = torch.nn.Parameter(a.detach().clone())
    opts = [torch.optim.AdamW([p], lr=.01) for p in (a, b)]
    for parameter, opt in zip((a, b), opts):
        parameter.grad = torch.ones_like(parameter)
        opt.step()
    step = opts[0].state[a]["step"]
    with residency.park_optimizer_state([opts[0]], [a]):
        assert opts[0].state[a]["step"] is step
    for opt in opts:
        opt.step()
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_shared_refill_releases_encoders_before_restoring_state(monkeypatch, fail):
    symbolic_torch(monkeypatch)
    events = []
    parameter = object()
    opt = SimpleNamespace(state={parameter: {"moment": DeviceTensor("cuda", events, "moment")}})
    trainer = SimpleNamespace(
        optimizer=opt,
        _main_model_module=lambda: SimpleNamespace(parameters=lambda: [parameter]),
        move_main_model_to_cpu=lambda: events.append(("main", "cpu")),
        move_main_model_to_gpu=lambda: events.append(("main", "cuda")),
        move_text_encoder_to_cpu=lambda: events.append(("text", "cpu")),
        move_vae_to_cpu=lambda: events.append(("vae", "cpu")))
    try:
        with residency.encoder_refill_residency(trainer, text=True, vae=True):
            events.append(("encode", "both"))
            if fail:
                raise RuntimeError("encode failed")
    except RuntimeError:
        assert fail
    assert events == [("main", "cpu"), ("moment", "cpu"), ("encode", "both"),
                      ("text", "cpu"), ("vae", "cpu"), ("main", "cuda"), ("moment", "cuda")]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fail", [False, True])
def test_ming_coincident_refill_parks_all_fused_optimizers_once(monkeypatch, fail):
    from core.models.ming_image_01.training_residency import MingTrainingResidency
    symbolic_torch(monkeypatch)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    events = []
    parameters = [object(), object()]
    optimizers = [SimpleNamespace(state={p: {"moment": DeviceTensor("cuda", events, str(i))}})
                  for i, p in enumerate(parameters)]
    def component(name):
        return SimpleNamespace(to=lambda *, device: events.append((name, str(device))),
                               cacheable=True, parameters=lambda: parameters)
    components = SimpleNamespace(transformer=component("main"), vae=component("vae"),
                                 conditioner=component("text"))
    trainer = SimpleNamespace(fused_optimizer_groups=SimpleNamespace(optimizers=optimizers))
    policy = MingTrainingResidency(components, device="cuda", latent_mode="swap_onthefly",
                                  text_mode="swap_onthefly", trainer=trainer)
    policy._report_memory = lambda *args: None
    try:
        with policy.continuous_refill():
            with policy("vae"):
                pass
            with policy("conditioner"):
                if fail:
                    raise RuntimeError("caption failed")
    except RuntimeError:
        assert fail
    for i in range(2):
        assert events.count((str(i), "cpu")) == 1
        assert events.count((str(i), "cuda")) == 1
    assert events.index(("text", "cpu")) < events.index(("main", "cuda")) < events.index(("1", "cuda"))
    assert events.count(("main", "cuda")) == 1
    assert policy.active is None and policy.optimizer_stack is None
