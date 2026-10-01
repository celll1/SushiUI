"""CPU native trainer previews, ownership restoration and RGBA publication."""
import sys
from pathlib import Path
import pytest
import torch
from PIL import Image, PngImagePlugin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming training preview attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)
from ming_image_01_conditioning_test import write_tiny_package, inputs
from core.training.full_parameter_trainer import FullParameterTrainer
from core.training.arch.base_arch import SampleContext
from api.arch_capabilities import TRAINING_UNSUPPORTED
from core.models.ming_image_01.training_sampling import publish_training_sample


@pytest.mark.parametrize("profile,reference", [("generation_edit", False), ("generation_edit", True), ("layer_decompose", True)])
def test_native_trainer_preview_restores_training_and_saves_all_rgba(tmp_path, monkeypatch, profile, reference):
    root = tmp_path / "package"
    root.mkdir()
    write_tiny_package(root, profile=profile)
    monkeypatch.setitem(TRAINING_UNSUPPORTED, "ming_image_01", {})
    owner = FullParameterTrainer(model_path=str(root), output_dir=str(tmp_path / "run"),
        device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
        mixed_precision=False, attention_backend="native",
        train_config={"ming_conditioner_groups": ["vision", "projections"], "ming_image_01_sample_layers": 2})
    reference_path = tmp_path / "reference.png"
    Image.new("RGBA", (64, 64), (100, 80, 60, 100)).save(reference_path)
    modules = (owner.transformer, owner.text_encoder, owner.vae)
    modes = [m.training for root in modules for m in root.modules()]
    parameters = {id(p): p.detach().clone() for root in modules for p in root.parameters()}
    scheduler_timesteps = owner.noise_scheduler.timesteps.clone()
    progress = []
    ctx = SampleContext(prompt="word5", width=64, height=64, num_inference_steps=1,
        guidance_scale=1., seed=17, reference_image_path=str(reference_path) if reference else None,
        step_progress_callback=lambda completed, total: progress.append((completed, total)))
    sample = owner.arch.sample(owner, ctx)
    assert sample.mode == "RGBA" and sample.size == (64, 64)
    assert progress == [(1, 1)]
    assert modes == [m.training for root in modules for m in root.modules()]
    assert all(torch.equal(parameters[id(p)], p) for root in modules for p in root.parameters())
    torch.testing.assert_close(owner.noise_scheduler.timesteps, scheduler_timesteps, rtol=0, atol=0)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", "word5")
    paths = publish_training_sample(sample, tmp_path / "preview.png", metadata)
    expected = ["auxiliary_canvas", "layer_1", "layer_2"] if profile == "layer_decompose" else ["image"]
    assert len(paths) == len(expected)
    for path, role in zip(paths, expected):
        with Image.open(path) as image:
            assert image.mode == "RGBA" and image.info["ming_frame_role"] == role
            assert image.info["prompt"] == "word5"
    def cancel_progress(*args):
        raise KeyboardInterrupt("test preview cancellation")
    ctx.step_progress_callback = cancel_progress
    with pytest.raises(KeyboardInterrupt):
        owner.arch.sample(owner, ctx)
    assert modes == [m.training for root in modules for m in root.modules()]
    assert all(p.device.type == "cpu" for root in modules for p in root.parameters())
    assert not torch.cuda.is_initialized()


def test_prefetch_suspend_releases_forward_and_resumes_next_batch(tmp_path):
    from ming_image_01_cpu_prefetch_test import owner_setup
    from core.models.ming_image_01.training_buffer import MingBatchBuffer
    value, owner, batches, _ = owner_setup(tmp_path)
    buffer = MingBatchBuffer(owner, batches, epoch=2, batch_offset=5,
        latent_interval=2, text_interval=3, prefetch_depth=1)
    try:
        buffer.prepare(0, batches[0]).for_step()
        old = buffer.prefetcher
        with pytest.raises(ValueError, match="test failure"):
            with buffer.suspend_prefetch():
                assert not old.thread.is_alive()
                value.conditioner(**inputs())
                raise ValueError("test failure")
        assert buffer.prefetcher is not old and buffer.prefetcher.start == 1
        buffer.prepare(1, batches[1]).for_step()
        buffer.prepare(2, batches[2]).for_step()
        with buffer.suspend_prefetch():
            value.conditioner(**inputs())
        assert buffer.prefetcher is None
    finally:
        buffer.close()


def test_layer_publication_rolls_back_all_previous_frames(tmp_path, monkeypatch):
    import os
    from core.models.ming_image_01.results import MingResultGroup
    frames = [Image.new("RGBA", (16,16), (i * 60,0,0,100)) for i in range(3)]
    frames[0]._ming_result_group = MingResultGroup(frames, ["auxiliary_canvas", "layer_1", "layer_2"],
        "layer_decompose", 2, 3, 17, 16, 16, "pil")
    metadata = PngImagePlugin.PngInfo()
    path = tmp_path / "preview.png"
    paths = publish_training_sample(frames[0], path, metadata)
    previous = {p: p.read_bytes() for p in paths}
    actual = os.replace
    def failed(source, destination):
        if Path(source).name == "1":
            raise OSError("injected publication failure")
        return actual(source, destination)
    monkeypatch.setattr(os, "replace", failed)
    with pytest.raises(OSError, match="injected"):
        publish_training_sample(frames[0], path, metadata)
    assert all(p.read_bytes() == value for p, value in previous.items())
    assert not list(tmp_path.glob(".ming-sample-*"))


def test_prefetch_pause_does_not_hide_worker_failure(tmp_path):
    from types import SimpleNamespace
    from ming_image_01_training_buffer_test import setup
    _, _, _, buffer, _, _ = setup(tmp_path)
    error = RuntimeError("failed native producer")
    stopped = []
    buffer.prefetcher = SimpleNamespace(stop=lambda: stopped.append(True), error=error)
    try:
        with pytest.raises(RuntimeError, match="failed native producer"):
            with buffer.suspend_prefetch():
                pytest.fail("A failed producer must not reach generation")
        assert stopped == [True] and buffer.prefetcher is None
    finally:
        buffer.close()
