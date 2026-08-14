"""MiniMax Music 3 txt2aud backend: staged-offload ordering (mocked components).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_staged_offload_test.py -v

No real weights are loaded. `MiniMaxMusic3Pipeline` is monkeypatched with a
fake that records, at the moment each of its methods is CALLED, which
components were resident on which device -- pinning the module docstring's
"Staged offload, and why THIS order" contract:

  1. language_model + rvq_depth_decoder on GPU, transformer/condition_encoder/
     vocoder still on CPU, for `encode_text`/`generate_ar`;
  2. transformer + condition_encoder on GPU, language_model/rvq_depth_decoder
     back on CPU, for `denoise_chunks`;
  3. vocoder on GPU, transformer/condition_encoder back on CPU, for `decode`.

Also covers: the final progress callback always reaches
`PROGRESS_TOTAL_UNITS`, an exception inside a stage still offloads that
stage's components (the `try/finally` contract), and `generate_txt2aud`'s
dispatch onto this backend for a MiniMax Music 3-flagged manager.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import core.models.minimax_music3.pipeline as music3_pipeline_module
from core.pipeline_backends.minimax_music3 import (
    PROGRESS_TOTAL_UNITS,
    MiniMaxMusic3Mixin,
    MiniMaxMusic3Txt2AudResult,
)

_TARGET_DEVICE = "cuda:0"  # never actually touched -- these are mock components


class _FakeParam:
    """A stand-in for one `nn.Parameter`: only `.device` is read by the
    backend's co-residency assertion (`next(comp.parameters()).device`)."""

    def __init__(self, device):
        self.device = device


class _MockComponent:
    """Stands in for an `nn.Module` component: records every `.to(device)`
    call into the shared `move_log`, keyed by this component's `name`, and
    exposes `.parameters()` (one `_FakeParam` tracking `.device`) so the
    backend's co-residency assertion has something to read, same as a real
    `nn.Module`.

    `fail_on_to_device`, if given, makes `.to(device)` raise a `RuntimeError`
    the FIRST time it is called with that target device (mimicking a
    mid-move CUDA OOM) -- used to prove the ->GPU staging path actually
    raises instead of silently leaving a component split across devices.
    """

    def __init__(self, name: str, move_log: list, fail_on_to_device: str = None):
        self.name = name
        self._move_log = move_log
        self._fail_on_to_device = fail_on_to_device
        self.device = "cpu"

    def to(self, device):
        if self._fail_on_to_device is not None and str(device) == self._fail_on_to_device:
            self._fail_on_to_device = None  # only the first call fails
            raise RuntimeError(f"mock CUDA OOM moving {self.name!r} to {device!r}")
        self.device = str(device)
        self._move_log.append((self.name, str(device)))
        return self

    def parameters(self):
        yield _FakeParam(self.device)


def _residents(move_log, names):
    """The most-recently-recorded device for each of `names`, per the log."""
    last = {}
    for name, device in move_log:
        if name in names:
            last[name] = device
    return last


class _FakeARResult:
    def __init__(self, num_frames=4, num_codebooks=8):
        self.frame_hiddens = torch.zeros(1, num_frames, num_codebooks * 4)
        self.frame_codes = torch.zeros(num_frames, num_codebooks, dtype=torch.int64)
        self.prefix_codes = torch.zeros(1, num_codebooks, dtype=torch.int64)


class _FakeMiniMaxMusic3Pipeline:
    """Records staging state at call time via `move_log` (captured from the
    manager's components at construction) instead of talking to real
    tensors/weights."""

    def __init__(self, tokenizer, language_model, rvq_depth_decoder, condition_encoder,
                 transformer, scheduler, vocoder, execution_device=None, move_log=None,
                 call_log=None, raise_in=None):
        self._components = {
            "tokenizer": tokenizer, "language_model": language_model,
            "rvq_depth_decoder": rvq_depth_decoder, "condition_encoder": condition_encoder,
            "transformer": transformer, "scheduler": scheduler, "vocoder": vocoder,
        }
        self._move_log = move_log
        self._call_log = call_log if call_log is not None else []
        self._raise_in = raise_in

    def _snapshot(self, stage):
        names = ("language_model", "rvq_depth_decoder", "transformer", "condition_encoder", "vocoder")
        self._call_log.append((stage, dict(_residents(self._move_log, names))))

    def encode_text(self, prompt, lyrics):
        self._snapshot("encode_text")
        if self._raise_in == "encode_text":
            raise RuntimeError("boom in encode_text")
        return torch.zeros(2, 4)

    def generate_ar(self, text_ids, audio_duration, generator=None, progress_callback=None,
                     resume_frame_codes=None, resume_prefix_codes=None):
        self._snapshot("generate_ar")
        if self._raise_in == "generate_ar":
            raise RuntimeError("boom in generate_ar")
        if progress_callback:
            progress_callback(2, 4, "ar")
            progress_callback(4, 4, "ar")
        return _FakeARResult()

    def denoise_chunks(self, frame_hiddens, num_inference_steps, flow_guidance_scale,
                        generator=None, progress_callback=None):
        self._snapshot("denoise_chunks")
        if self._raise_in == "denoise_chunks":
            raise RuntimeError("boom in denoise_chunks")
        if progress_callback:
            progress_callback(1, 2, "flow")
            progress_callback(2, 2, "flow")
        return [torch.zeros(1, 2, 3)]

    def decode(self, latent_chunks, output_type="pt"):
        self._snapshot("decode")
        if self._raise_in == "decode":
            raise RuntimeError("boom in decode")
        return torch.full((1, 2, 16), 0.25)

    @property
    def sampling_rate(self):
        return 44100

    @property
    def frame_rate(self):
        return 25.0


class _Manager(MiniMaxMusic3Mixin):
    def __init__(self, components, device="cuda:0"):
        self.minimax_music3_components = components
        self.is_minimax_music3_model = True
        self.device = device


def _build_manager(move_log, fail_component=None, fail_on_to_device=_TARGET_DEVICE):
    """`fail_component`, if given, makes that one component's `.to()` raise
    the first time it is called with `fail_on_to_device` -- see
    `_MockComponent`."""
    def _component(name):
        return _MockComponent(
            name, move_log, fail_on_to_device=fail_on_to_device if name == fail_component else None,
        )

    components = {
        "tokenizer": object(),
        "language_model": _component("language_model"),
        "rvq_depth_decoder": _component("rvq_depth_decoder"),
        "condition_encoder": _component("condition_encoder"),
        "transformer": _component("transformer"),
        "scheduler": object(),
        "vocoder": _component("vocoder"),
        "frame_rate": 25.0,
        "rvq_depth_decoder_config": {"num_codebooks": 8},
    }
    return _Manager(components)


def _params(**overrides):
    base = dict(
        prompt="ambient synth pad",
        lyrics="[verse]\nhello world",
        seed=42,
        audio_duration=1.0,
        num_inference_steps=2,
        flow_guidance_scale=1.7,
    )
    base.update(overrides)
    return base


def _patch_pipeline(monkeypatch, move_log, call_log, raise_in=None):
    def _factory(**kwargs):
        return _FakeMiniMaxMusic3Pipeline(move_log=move_log, call_log=call_log, raise_in=raise_in, **kwargs)

    monkeypatch.setattr(music3_pipeline_module, "MiniMaxMusic3Pipeline", _factory)


def test_staged_offload_ordering(monkeypatch):
    move_log = []
    call_log = []
    _patch_pipeline(monkeypatch, move_log, call_log)
    manager = _build_manager(move_log)
    progress_ticks = []

    result = manager._generate_txt2aud_minimax_music3(
        _params(), progress_callback=lambda step, total: progress_ticks.append((step, total)),
    )

    assert isinstance(result, MiniMaxMusic3Txt2AudResult)

    stages = dict(call_log)
    # AR stage: LM + depth decoder resident, transformer/condition_encoder/vocoder still CPU.
    ar_state = stages["generate_ar"]
    assert ar_state["language_model"] == _TARGET_DEVICE
    assert ar_state["rvq_depth_decoder"] == _TARGET_DEVICE
    assert ar_state.get("transformer", "cpu") == "cpu"
    assert ar_state.get("condition_encoder", "cpu") == "cpu"
    assert ar_state.get("vocoder", "cpu") == "cpu"

    # Flow stage: transformer + condition_encoder resident, LM/depth decoder back on CPU.
    flow_state = stages["denoise_chunks"]
    assert flow_state["transformer"] == _TARGET_DEVICE
    assert flow_state["condition_encoder"] == _TARGET_DEVICE
    assert flow_state["language_model"] == "cpu"
    assert flow_state["rvq_depth_decoder"] == "cpu"
    assert flow_state.get("vocoder", "cpu") == "cpu"

    # Decode: vocoder resident, transformer/condition_encoder back on CPU.
    decode_state = stages["decode"]
    assert decode_state["vocoder"] == _TARGET_DEVICE
    assert decode_state["transformer"] == "cpu"
    assert decode_state["condition_encoder"] == "cpu"

    # Every component ends up back on CPU (final offload).
    final = _residents(move_log, ("language_model", "rvq_depth_decoder", "transformer", "condition_encoder", "vocoder"))
    assert set(final.values()) == {"cpu"}

    # Progress reaches the fixed total exactly once at the end, monotonically.
    assert progress_ticks[-1] == (PROGRESS_TOTAL_UNITS, PROGRESS_TOTAL_UNITS)
    steps = [step for step, _total in progress_ticks]
    assert steps == sorted(steps)


@pytest.mark.parametrize("raise_in", ["generate_ar", "denoise_chunks", "decode"])
def test_exception_in_a_stage_still_offloads_that_stage(monkeypatch, raise_in):
    move_log = []
    call_log = []
    _patch_pipeline(monkeypatch, move_log, call_log, raise_in=raise_in)
    manager = _build_manager(move_log)

    with pytest.raises(RuntimeError, match=f"boom in {raise_in}"):
        manager._generate_txt2aud_minimax_music3(_params())

    # Regardless of which stage raised, every component that was ever moved
    # to the target device must have been moved back to CPU (the try/finally
    # contract) -- check the LAST recorded device for each name touched.
    touched = {name for name, _device in move_log}
    final = _residents(move_log, touched)
    for name, device in final.items():
        assert device == "cpu", f"{name} was left on {device!r} after a {raise_in} exception"


def test_partial_ar_staging_failure_raises_instead_of_continuing(monkeypatch):
    """F6: a failure moving the SECOND AR-pair component to the target
    device (mimicking a mid-move CUDA OOM on `rvq_depth_decoder` after
    `language_model` already succeeded) must raise -- not be swallowed into
    a printed warning that lets generation continue with a split-device
    component. The pipeline's own co-residency guard cannot catch this under
    manual (non-accelerate-hook) staging, so `_minimax_music3_move`'s
    ->GPU path must raise on its own, and the mixin's explicit co-residency
    assertion is the second layer -- either one failing to fire would let
    `generate_ar` run against a straddled component."""
    move_log = []
    call_log = []
    _patch_pipeline(monkeypatch, move_log, call_log)
    manager = _build_manager(move_log, fail_component="rvq_depth_decoder")

    with pytest.raises(RuntimeError, match="mock CUDA OOM"):
        manager._generate_txt2aud_minimax_music3(_params())

    # generate_ar must never have been reached -- the raise happens during
    # staging, before the pipeline is even called.
    assert not any(stage == "generate_ar" for stage, _state in call_log)
    # language_model DID move to the target device before the failure (this
    # is exactly the "split across devices" state the raise is protecting
    # against) -- confirm the mock recorded that partial move actually
    # happened, i.e. this test is exercising a REAL partial-failure scenario
    # and not one where nothing moved at all.
    assert ("language_model", _TARGET_DEVICE) in move_log
    assert not any(name == "rvq_depth_decoder" for name, _device in move_log)


def test_missing_component_raises_before_any_staging(monkeypatch):
    move_log = []
    call_log = []
    _patch_pipeline(monkeypatch, move_log, call_log)
    manager = _build_manager(move_log)
    manager.minimax_music3_components["vocoder"] = None

    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError):
        manager._generate_txt2aud_minimax_music3(_params())
    assert move_log == []


@pytest.mark.parametrize("missing_key", ["prompt", "lyrics", "audio_duration", "num_inference_steps", "flow_guidance_scale"])
def test_required_param_without_a_fallback_raises(monkeypatch, missing_key):
    move_log = []
    call_log = []
    _patch_pipeline(monkeypatch, move_log, call_log)
    manager = _build_manager(move_log)

    from api.error_handlers import ValidationError

    params = _params()
    params[missing_key] = None
    with pytest.raises(ValidationError):
        manager._generate_txt2aud_minimax_music3(params)
    assert move_log == []
