"""VRAM retention lifecycle: training-start release, failed-generation offload,
ControlNet/LLLite offload, and the double-start refusal.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/vram_release_lifecycle_test.py -q

No CUDA and no real weights: components are fakes that report GPU residency and
record every `.to()` call, which is exactly what the code under test reads
(`component_cuda_bytes` -> `parameters()/buffers()` -> `.is_cuda`).

Background: a training run started while the backend held ~10 GiB of generation
VRAM; nothing in the backend released it, and the "unloading generate pipeline"
block that looked like it did lives in the TRAINER's process, where every
attribute it checks is None.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import core.pipeline as pipeline_module
from core.pipeline import (
    DiffusionPipelineManager,
    component_cuda_bytes,
    offload_component_to_cpu,
)
from core.extensions.controlnet_manager import ControlNetManager
from core.training.training_process import TrainingProcessManager


class _FakeTensor:
    def __init__(self, nbytes=1024, cuda=True):
        self._nbytes = nbytes
        self.is_cuda = cuda

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1

    def to(self, device):
        return _FakeTensor(self._nbytes, cuda=(str(device) != "cpu"))


class _FakeComponent:
    """nn.Module stand-in: `parameters()` drive the residency check, `.to()` is
    recorded and flips residency the way a real move does."""

    def __init__(self, name, nbytes=1024, cuda=True, fail_to=False):
        self.name = name
        self._param = _FakeTensor(nbytes, cuda)
        self.moves = []
        self.fail_to = fail_to

    def parameters(self):
        return [self._param]

    def buffers(self):
        return []

    def to(self, device):
        self.moves.append(str(device))
        if self.fail_to:
            raise RuntimeError("simulated mid-move failure")
        self._param.is_cuda = str(device) != "cpu"
        return self


def _bare_manager():
    """A DiffusionPipelineManager without running __init__ (which touches disk
    and model state) -- only the attributes the release path reads are set."""
    manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
    manager.txt2img_pipeline = None
    manager.img2img_pipeline = None
    manager.inpaint_pipeline = None
    manager.vision_encoder = None
    for attr, _label, flag in pipeline_module.ARCH_COMPONENT_SETS:
        setattr(manager, attr, None)
        setattr(manager, flag, False)
    return manager


# ---------------------------------------------------------------- inventory


def test_cuda_bytes_only_counts_gpu_residents():
    """MUTANT THIS EXISTS FOR: counting every parameter instead of the CUDA ones.
    A CPU-resident component must report 0 so the offload never calls `.to()` on
    it -- MiniMax-H3's memory-mapped 48 GiB text encoder is CPU-resident and a
    `.to()` on it copies the file into anonymous memory."""
    assert component_cuda_bytes(_FakeComponent("gpu", nbytes=4096)) == 4096
    assert component_cuda_bytes(_FakeComponent("cpu", nbytes=4096, cuda=False)) == 0
    assert component_cuda_bytes(None) == 0


def test_offload_skips_cpu_component_and_records_failure(capsys):
    """MUTANT: swallowing the `.to()` failure silently (`except Exception: pass`).
    The component NAME must reach the log -- its absence is what made the
    original retention incident undiagnosable."""
    cpu_comp = _FakeComponent("already_cpu", cuda=False)
    released = []
    assert offload_component_to_cpu("te", cpu_comp, released) == 0
    assert cpu_comp.moves == []

    broken = _FakeComponent("broken", nbytes=2048, fail_to=True)
    assert offload_component_to_cpu("unet", broken, released) == 0
    assert released == []
    assert "unet" in capsys.readouterr().out


# ------------------------------------------------- F3 + F4: training release


def test_release_gpu_memory_offloads_every_arch_and_clears_keep_hot():
    """MUTANT: releasing only the SD pipelines (what the dead trainer-side block
    tried to do), or clearing keep-hot bookkeeping WITHOUT offloading first.
    clear_resident is bookkeeping-only by design, so a mutant that only clears
    leaves the VRAM held while claiming it is free."""
    from core.keep_hot import mark_resident, resident_components

    manager = _bare_manager()
    unet = _FakeComponent("unet", nbytes=8192)
    vae = _FakeComponent("vae", nbytes=1024)
    manager.txt2img_pipeline = types.SimpleNamespace(
        unet=unet, text_encoder=None, text_encoder_2=None, vae=vae)
    # Same modules behind a second pipeline: must be offloaded once, not twice.
    manager.img2img_pipeline = types.SimpleNamespace(
        unet=unet, text_encoder=None, text_encoder_2=None, vae=vae)
    transformer = _FakeComponent("transformer", nbytes=4096)
    manager.sensenova_components = {"transformer": transformer}
    manager.vision_encoder = _FakeComponent("vision_encoder", nbytes=512)

    mark_resident(manager, "unet", "model-key")
    mark_resident(manager, "vae", "model-key")

    result = manager.release_gpu_memory(reason="unit test")

    assert unet.moves == ["cpu"]
    assert vae.moves == ["cpu"]
    assert transformer.moves == ["cpu"]
    assert manager.vision_encoder.moves == ["cpu"]
    assert result["freed_bytes"] == 8192 + 1024 + 4096 + 512
    assert sorted(result["keep_hot_cleared"]) == ["unet", "vae"]
    assert resident_components(manager) == set()
    # The model stays LOADED -- this is a release, not an unload.
    assert manager.txt2img_pipeline is not None
    assert manager.sensenova_components is not None


def test_release_gpu_memory_also_offloads_controlnets(monkeypatch):
    """MUTANT: leaving the ControlNet/LLLite caches out of the training-start
    release. They are process-global and never evicted."""
    manager = _bare_manager()
    calls = []
    monkeypatch.setattr(manager, "_offload_controlnets_after_generation",
                        lambda: calls.append("cn"))
    manager.release_gpu_memory()
    assert calls == ["cn"]


# ------------------------------------------ F1: failure before the inner try


def test_failed_generation_offloads_staged_components():
    """MUTANT: restricting the offload to the denoise try/finally. Text encoders
    and the U-Net are staged to the GPU BEFORE it, so a failure in between (OOM,
    cancel, ControlNet setup) leaked them for the process lifetime."""
    from core.keep_hot import mark_resident, resident_components

    manager = _bare_manager()
    unet = _FakeComponent("unet", nbytes=4096)
    te = _FakeComponent("text_encoder", nbytes=2048)
    pipeline = types.SimpleNamespace(unet=unet, text_encoder=te, text_encoder_2=None, vae=None)
    manager.txt2img_pipeline = pipeline
    mark_resident(manager, "unet", "model-key")

    manager._offload_after_failed_generation(pipeline, "txt2img")

    assert unet.moves == ["cpu"]
    assert te.moves == ["cpu"]
    # A FAILED generation must not leave anything marked resident, even with
    # keep_models_hot on.
    assert resident_components(manager) == set()


def test_generate_txt2img_guard_offloads_when_body_raises(monkeypatch):
    """MUTANT: dropping the outer except/finally from generate_txt2img (the
    pre-staging hole) -- the body raises before its own try/finally is entered."""
    manager = _bare_manager()
    unet = _FakeComponent("unet", nbytes=4096)
    manager.txt2img_pipeline = types.SimpleNamespace(
        unet=unet, text_encoder=None, text_encoder_2=None, vae=None)

    controlnet_calls = []
    monkeypatch.setattr(manager, "_offload_controlnets_after_generation",
                        lambda: controlnet_calls.append("cn"))

    def _boom(params, progress_callback=None, step_callback=None):
        raise RuntimeError("CUDA out of memory during ControlNet setup")

    monkeypatch.setattr(manager, "_generate_txt2img_sd", _boom)

    with pytest.raises(RuntimeError):
        manager.generate_txt2img({"prompt": "x"})

    assert unet.moves == ["cpu"]
    assert controlnet_calls == ["cn"]


def test_generate_txt2img_guard_offloads_controlnets_on_success(monkeypatch):
    """MUTANT: putting the ControlNet offload only on the failure path. The
    caches must come off the GPU after a SUCCESSFUL generation too (keep-hot
    covers the model's own components, never the ControlNet caches)."""
    manager = _bare_manager()
    manager.txt2img_pipeline = types.SimpleNamespace(
        unet=None, text_encoder=None, text_encoder_2=None, vae=None)
    controlnet_calls = []
    monkeypatch.setattr(manager, "_offload_controlnets_after_generation",
                        lambda: controlnet_calls.append("cn"))
    monkeypatch.setattr(manager, "_generate_txt2img_sd",
                        lambda params, progress_callback=None, step_callback=None: ("img", 1, 2))

    assert manager.generate_txt2img({"prompt": "x"}) == ("img", 1, 2)
    assert controlnet_calls == ["cn"]


# ------------------------------------------------- F2: ControlNet and LLLite


def test_offload_controlnets_covers_lllite_state_dicts():
    """MUTANT: iterating `loaded_controlnets` only. LLLite state dicts are loaded
    with device="cuda" into a second cache that nothing ever moved off the GPU.
    `device` in the record is the COMPUTE device for the next module build and
    must NOT be rewritten to cpu."""
    manager = ControlNetManager()
    cn = _FakeComponent("cn", nbytes=1024)
    manager.loaded_controlnets = {"cn.safetensors": cn}
    weight = _FakeTensor(2048, cuda=True)
    manager.loaded_lllites = {
        "lllite.safetensors": {
            "state_dict": {"w": weight},
            "device": "cuda",
            "control_image": object(),
        }
    }

    manager.offload_controlnets_to_cpu()

    assert cn.moves == ["cpu"]
    record = manager.loaded_lllites["lllite.safetensors"]
    assert record["state_dict"]["w"].is_cuda is False
    assert record["device"] == "cuda"
    assert "control_image" not in record


# ---------------------------------------------------- double-start refusal


class _FakeChild:
    def __init__(self, returncode=None, pid=4242):
        self.returncode = returncode
        self.pid = pid


def test_create_process_refuses_to_overwrite_a_live_process():
    """MUTANT: `self.processes[run_id] = process` unconditionally. Two
    train_runner children for one run orphan the first -- the registry entry
    that could stop it is gone."""
    manager = TrainingProcessManager()
    existing = types.SimpleNamespace(process=_FakeChild(returncode=None), is_running=True)
    manager.processes[7] = existing

    assert manager.is_live(7) is True
    with pytest.raises(RuntimeError) as excinfo:
        manager.create_process(run_id=7, config_path="c.yaml", output_dir="out")
    assert "already has a live training process" in str(excinfo.value)
    assert manager.processes[7] is existing


def test_is_live_is_false_for_an_exited_or_unspawned_process():
    """MUTANT: reading `is_running` instead of the child's returncode. The flag
    is cleared only once the monitor task observes the exit, so it is stale
    exactly during the window a restart is attempted."""
    manager = TrainingProcessManager()
    manager.processes[1] = types.SimpleNamespace(process=_FakeChild(returncode=0), is_running=True)
    manager.processes[2] = types.SimpleNamespace(process=None, is_running=True)
    assert manager.is_live(1) is False
    assert manager.is_live(2) is False
    assert manager.is_live(99) is False
