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


def test_arch_component_sets_covers_every_components_attribute():
    """MUTANT: deleting any row from ARCH_COMPONENT_SETS. The inventory is the
    only thing the reload cleanup and release_gpu_memory iterate, so a missing
    row is an architecture whose weights are silently never released."""
    manager = DiffusionPipelineManager()
    assert {a for a in vars(manager) if a.endswith("_components")} == \
        {attr for attr, _label, _flag in pipeline_module.ARCH_COMPONENT_SETS}


def test_unmeasurable_component_is_offloaded_not_skipped():
    """MUTANT: returning 0 instead of None for a component with no
    `parameters()`, which collapses "CPU-resident" and "residency unknown" into
    one answer and skips the offload. Live instance:
    `ltx2_components["pipeline"]` is a DiffusionPipeline -- the per-arch cleanup
    blocks this replaced used `hasattr(comp, 'to')`, which is strictly wider."""
    class _Pipelineish:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    assert component_cuda_bytes(_Pipelineish()) is None
    comp = _Pipelineish()
    released = []
    assert offload_component_to_cpu("ltx2.pipeline", comp, released) == 0
    assert comp.moves == ["cpu"]
    assert released == [("ltx2.pipeline", 0)]


def test_unknown_residency_fallback_never_touches_an_nn_module():
    """The MiniMax-H3 invariant: its 48 GiB memory-mapped text encoder IS an
    nn.Module, so it is MEASURED at 0 CUDA bytes and skipped -- the unknown
    fallback above must not be able to reach it. Also skips bare tensors, whose
    `.to()` returns a copy and would free nothing."""
    import torch

    te = _FakeComponent("h3_text_encoder", nbytes=48 * 1024 ** 3, cuda=False)
    released = []
    assert offload_component_to_cpu("MiniMax-H3.text_encoder", te, released) == 0
    assert te.moves == []
    assert released == []

    tensor = torch.zeros(4)
    assert component_cuda_bytes(tensor) is None
    assert offload_component_to_cpu("latents_mean", tensor, released) == 0
    assert released == []


def test_pid_decoder_hook_runs_on_both_offload_branches():
    """MUTANT: dropping the `_stage_pid_cpu` hook from either branch. The PiD
    wrapper stages its decoder net independently of `.to()`, so a wrapper that
    reports 0 CUDA bytes (or is unmeasurable) can still be holding ~6 GB."""
    class _PidWrapper(_FakeComponent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pid_offloads = 0

        def _stage_pid_cpu(self):
            self.pid_offloads += 1

    on_gpu = _PidWrapper("vae", nbytes=2048)
    on_cpu = _PidWrapper("vae", nbytes=2048, cuda=False)
    released = []
    offload_component_to_cpu("vae", on_gpu, released)
    offload_component_to_cpu("vae", on_cpu, released)
    assert on_gpu.pid_offloads == 1
    assert on_cpu.pid_offloads == 1


def test_component_cleanup_resets_the_ltx2_and_h3_special_cases(monkeypatch):
    """MUTANT: deleting the `ltx2`/`minimax_h3` special cases from the shared
    cleanup loop. LTX-2.3's offload guard would stay True and the next load would
    never re-attach its cpu-offload hooks; H3's prompt cache would answer for a
    model that is no longer loaded."""
    manager = _bare_manager()
    manager._ltx2_offload_enabled = True
    manager.ltx2_components = {"pipeline": _FakeComponent("pipeline")}
    manager.minimax_h3_components = {"transformer": _FakeComponent("transformer")}
    manager.is_ltx2_model = True

    cleared = []
    from core.models.minimax_h3 import prompt_cache
    monkeypatch.setattr(prompt_cache, "clear", lambda: cleared.append("cleared"))

    manager._cleanup_component_architectures()

    assert manager.ltx2_components is None
    assert manager.minimax_h3_components is None
    assert manager.is_ltx2_model is False
    assert manager._ltx2_offload_enabled is False
    assert cleared == ["cleared"]


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


def test_release_gpu_memory_offloads_taesd(monkeypatch):
    """MUTANT: deleting the taesd_manager.offload_to_cpu() call. The cheap-decode
    models are a process-global cache that no generation owns."""
    from core.utils import taesd as taesd_module

    manager = _bare_manager()
    calls = []
    monkeypatch.setattr(taesd_module.taesd_manager, "offload_to_cpu",
                        lambda: calls.append("taesd"))
    monkeypatch.setattr(manager, "_offload_controlnets_after_generation", lambda: None)
    manager.release_gpu_memory()
    assert calls == ["taesd"]

    calls.clear()
    manager._offload_after_failed_generation(
        types.SimpleNamespace(unet=None, text_encoder=None, text_encoder_2=None, vae=None),
        "txt2img")
    assert calls == ["taesd"]


def test_release_sweeps_the_backends_other_gpu_holders(monkeypatch):
    """MUTANT: releasing only the loaded model. TIPO (an fp16 causal LM on cuda
    with an unload_model() nothing called), the tagger's ONNX CUDA session
    (auto_unload is a per-request parameter) and the cached spandrel upscaler are
    all resident in THIS process and reproduce the incident with an identical
    "release succeeded" log."""
    from core.extensions import tipo_manager as tipo_module
    from core.extensions import tagger_manager as tagger_module
    from core import upscaler as upscaler_module

    manager = _bare_manager()
    monkeypatch.setattr(manager, "_offload_controlnets_after_generation", lambda: None)

    tipo_calls = []
    monkeypatch.setattr(tipo_module.tipo_manager, "model", object(), raising=False)
    monkeypatch.setattr(tipo_module.tipo_manager, "unload_model",
                        lambda: tipo_calls.append("tipo"))

    tagger_calls = []
    monkeypatch.setattr(tagger_module.tagger_manager, "session", object(), raising=False)
    monkeypatch.setattr(tagger_module.tagger_manager, "unload_model",
                        lambda: tagger_calls.append("tagger"))

    upscaler_model = _FakeComponent("spandrel", nbytes=1024)
    monkeypatch.setitem(upscaler_module._spandrel_cache, "model", upscaler_model)

    result = manager.release_gpu_memory()

    assert tipo_calls == ["tipo"]
    assert tagger_calls == ["tagger"]
    assert upscaler_model.moves == ["cpu"]
    assert result["auxiliary"] == ["tipo", "tagger_onnx_session", "upscaler.spandrel"]


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


@pytest.mark.parametrize("exc", [KeyboardInterrupt, GeneratorExit,
                                 __import__("asyncio").CancelledError])
def test_generate_guards_catch_base_exceptions(monkeypatch, exc):
    """MUTANT: narrowing `except BaseException` to `except Exception`. A user
    cancel arrives as CancelledError and a Ctrl-C as KeyboardInterrupt -- neither
    derives from Exception, and both leave the U-Net staged."""
    manager = _bare_manager()
    unet = _FakeComponent("unet", nbytes=4096)
    manager.txt2img_pipeline = types.SimpleNamespace(
        unet=unet, text_encoder=None, text_encoder_2=None, vae=None)
    monkeypatch.setattr(manager, "_offload_controlnets_after_generation", lambda: None)

    def _boom(params, progress_callback=None, step_callback=None):
        raise exc()

    monkeypatch.setattr(manager, "_generate_txt2img_sd", _boom)

    with pytest.raises(exc):
        manager.generate_txt2img({"prompt": "x"})
    assert unet.moves == ["cpu"]


def test_img2img_guard_covers_the_pipeline_construction_staging(monkeypatch):
    """MUTANT: opening the try below the img2img/inpaint construction block. Its
    `.to(self.device)` stages the U-Net, both text encoders and the VAE, so an
    OOM there leaks exactly what the guard exists to cover."""
    manager = _bare_manager()
    unet = _FakeComponent("unet", nbytes=4096)
    te = _FakeComponent("text_encoder", nbytes=2048)
    manager.device = "cuda"
    manager.txt2img_pipeline = types.SimpleNamespace(
        unet=unet, text_encoder=te, text_encoder_2=None, vae=None,
        components={"unet": unet, "text_encoder": te})
    manager.img2img_pipeline = None
    for flag in ("is_zimage_model", "is_flux2_model", "is_anima_model", "is_lens_model",
                 "is_ideogram4_model", "is_minit2i_model", "is_krea2_model",
                 "is_sensenova_model", "is_ltx2_model", "is_minimax_h3_model"):
        setattr(manager, flag, False)
    monkeypatch.setattr(manager, "_offload_controlnets_after_generation", lambda: None)

    def _explode(**kwargs):
        raise RuntimeError("CUDA out of memory staging the img2img pipeline")

    monkeypatch.setattr(pipeline_module, "StableDiffusionImg2ImgPipeline", _explode)

    with pytest.raises(RuntimeError):
        manager.generate_img2img({"prompt": "x"}, init_image=None)

    # img2img_pipeline is still None here, so the offload has to fall back to the
    # txt2img pipeline that owns the same modules.
    assert unet.moves == ["cpu"]
    assert te.moves == ["cpu"]


# --------------------------------------------------------- PiD decoder staging


def test_pid_stage_sets_the_device_flag_before_moving():
    """MUTANT: reverting pid_vae_wrapper to `net.to("cuda")` then flag=cuda.
    nn.Module.to() moves parameters one at a time, so a mid-move OOM leaves some
    on the GPU while the flag still reads "cpu"."""
    from core.models.pid.pid_vae_wrapper import PidVaeWrapper

    wrapper = PidVaeWrapper.__new__(PidVaeWrapper)

    class _Net:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            if str(device) == "cuda":
                raise RuntimeError("CUDA out of memory mid-move")
            return self

    net = _Net()
    wrapper._pid_model = types.SimpleNamespace(net=net)
    wrapper._pid_device = "cpu"

    with pytest.raises(RuntimeError):
        wrapper._stage_pid_gpu()
    assert wrapper._pid_device == "cuda"

    # And the offload is unconditional, so the strand is recoverable.
    wrapper._stage_pid_cpu()
    assert net.moves == ["cuda", "cpu"]
    assert wrapper._pid_device == "cpu"


def test_pid_offload_is_unconditional_even_when_the_flag_says_cpu():
    """MUTANT: restoring the `if self._pid_device != "cpu"` guard around the
    offload. After a partial stage the flag cannot be trusted."""
    from core.models.pid.pid_vae_wrapper import PidVaeWrapper

    wrapper = PidVaeWrapper.__new__(PidVaeWrapper)

    class _Net:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    net = _Net()
    wrapper._pid_model = types.SimpleNamespace(net=net)
    wrapper._pid_device = "cpu"
    wrapper._stage_pid_cpu()
    assert net.moves == ["cpu"]


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
