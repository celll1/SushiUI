"""Training-model prediction must not overlap native updates or lifecycle changes."""

import os
import threading

import pytest
import torch

from foveatag_live_prediction_test import handle, image_bytes
from foveatag_shared_export_test import model, batch
from core.tagger.foveatag.processor import FoveaProcessor
from core.tagger.foveatag.views import ViewConfig


def thread_call(target, errors):
    def run():
        try:
            target()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def finish(threads, errors):
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    if errors:
        raise errors[0]


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("device,dtype", [("cpu", None), ("cuda", torch.float16), ("cuda", torch.bfloat16)])
def test_prediction_waits_for_whole_native_optimizer_step(monkeypatch, method, device, dtype):
    if device == "cuda" and os.environ.get("FOVEATAG_CONCURRENCY_GPU_TEST") != "1":
        pytest.skip("explicit GPU concurrency test")
    instance = model(method).to(device)
    instance.vision_encoder.set_sdpa(True, allow_flash=device == "cuda", external_flash=device == "cuda")
    processor = FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=2),
                              seed=42, training=False)
    owner = handle(instance, processor)
    owner._inference_amp_dtype = dtype
    monkeypatch.setattr(owner, "can_predict", lambda: True)
    inputs = batch().to(device)
    optimizer = torch.optim.AdamW(instance.trainable_parameters(), lr=.001)
    entered, release, predicting, predicted = [threading.Event() for _ in range(4)]
    errors = []
    outputs = []
    def train():
        with owner.model_access():
            instance.train()
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                loss = instance(inputs).square().mean()
            entered.set()
            assert release.wait(timeout=5)
            assert instance.training
            loss.backward()
            assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                       for parameter in instance.parameters())
            optimizer.step()
    def predict():
        predicting.set()
        outputs.append(owner.predict(image_bytes(), .5))
        predicted.set()
    training_thread = thread_call(train, errors)
    assert entered.wait(timeout=5)
    inference_thread = thread_call(predict, errors)
    try:
        assert predicting.wait(timeout=5)
        assert not predicted.wait(timeout=.1)
        assert instance.training
    finally:
        release.set()
    finish([training_thread, inference_thread], errors)
    assert instance.training and outputs[0]["source"] == "training_model"


def test_detach_waits_for_inference_and_failure_releases_lock(monkeypatch):
    instance = model("full")
    processor = FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=2),
                              seed=42, training=False)
    owner = handle(instance, processor)
    monkeypatch.setattr(owner, "can_predict", lambda: True)
    entered, release, detaching, detached = [threading.Event() for _ in range(4)]
    errors = []
    original = instance.forward
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(instance, "forward", paused)
    inference_thread = thread_call(lambda: owner.predict(image_bytes(), .5), errors)
    assert entered.wait(timeout=5)
    def detach():
        detaching.set()
        owner.detach()
        detached.set()
    detach_thread = thread_call(detach, errors)
    try:
        assert detaching.wait(timeout=5)
        assert not detached.wait(timeout=.1)
        assert owner._model is instance
    finally:
        release.set()
    finish([inference_thread, detach_thread], errors)
    assert owner._model is None
    owner.attach(instance, None, None, processor=processor, vocabulary=None)
    monkeypatch.setattr(instance, "forward", lambda *a, **k: (_ for _ in ()).throw(ValueError("synthetic failure")))
    with pytest.raises(ValueError, match="synthetic failure"):
        owner.predict(image_bytes(), .5)
    completed = threading.Event()
    def acquire():
        with owner.model_access():
            completed.set()
    finish([thread_call(acquire, errors)], errors)
    assert completed.is_set()


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("mode", ["dram", "disk"])
def test_gpu_offload_waits_for_prediction_and_restores_state(monkeypatch, tmp_path, method, mode):
    if os.environ.get("FOVEATAG_CONCURRENCY_GPU_TEST") != "1":
        pytest.skip("explicit GPU lifecycle test")
    from core.gpu_coordinator import OffloadDecision
    instance = model(method).cuda()
    optimizer = torch.optim.AdamW(instance.trainable_parameters(), lr=.001)
    instance(batch().to("cuda")).square().mean().backward()
    optimizer.step()
    weights = {key: value.detach().cpu().clone() for key, value in instance.state_dict().items()}
    state = {key: {name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                   for name, value in values.items()} for key, values in optimizer.state_dict()["state"].items()}
    identities = [id(parameter) for parameter in instance.parameters()]
    processor = FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=2),
                              seed=42, training=False)
    owner = handle(instance, processor)
    owner._optimizer = optimizer
    entered, release, requested = [threading.Event() for _ in range(3)]
    original = instance.forward
    errors = []
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(instance, "forward", paused)
    prediction = thread_call(lambda: owner.predict(image_bytes(), .5), errors)
    assert entered.wait(timeout=5)
    def check_waiting():
        try:
            assert requested.wait(timeout=5)
            assert not release.wait(timeout=.1)
            assert next(instance.parameters()).device.type == "cuda"
        finally:
            release.set()
    watcher = thread_call(check_waiting, errors)
    requested.set()
    owner.offload(OffloadDecision(mode=mode, swap_dir=str(tmp_path)))
    finish([prediction, watcher], errors)
    assert next(instance.parameters()).device.type == "cpu" and not owner.can_predict()
    assert not optimizer.state
    owner.restore()
    assert owner.can_predict() and instance.training
    assert identities == [id(parameter) for parameter in instance.parameters()]
    for key, value in instance.state_dict().items():
        torch.testing.assert_close(value.cpu(), weights[key], rtol=0, atol=0)
    for key, values in optimizer.state_dict()["state"].items():
        for name, value in values.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value.cpu(), state[key][name], rtol=0, atol=0)
            else:
                assert value == state[key][name]
    assert owner._swap_path is None and not (tmp_path / "tagger_optim_state.pt").exists()
