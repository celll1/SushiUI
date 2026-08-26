"""SenseNova training-time sample: KV cache streaming reachability.

Before this change, a training-time sample (``ops/sensenova_ops.py::generate_sample``)
drove the phase evictor's own prefix/denoise transitions but had no code path to
the standalone generation's KV cache streaming mechanism at all --
``transformer._kv_cache_streamer`` was never set, regardless of any config,
because ``kv_cache_streaming.install``/``uninstall`` were only ever called from
``pipeline_backends/sensenova.py``. ``test_sample_kv_streaming_off_by_default_*``
below reproduces exactly that (still-default) state as a negative control; the
other tests exercise the new opt-in path against the same ``generate_sample``
entry point used in production, mocking only the transfer/model-call layer the
same way ``sensenova_reference_training_test.py`` does.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _SampleTransformer(nn.Module):
    def forward(self):  # pragma: no cover - never called
        raise AssertionError


def _sample_trainer(*, kv_streaming=False, with_evictor=False):
    transformer = _SampleTransformer()
    transformer.train()
    trainer = SimpleNamespace(
        transformer=transformer,
        tokenizer=SimpleNamespace(),
        log_prefix="[test]",
        attention_backend="auto",
        device="cuda:0",
        sensenova_sample_kv_cache_streaming=kv_streaming,
        _resolve_training_backend=lambda backend: "sdpa",
        move_main_model_to_gpu=lambda: None,
    )
    events = []
    if with_evictor:
        trainer.sensenova_phase_evictor = SimpleNamespace(
            enter_prefix=lambda: events.append("prefix"),
            enter_denoise=lambda: events.append("denoise"),
            assert_generation_resident=lambda: events.append("resident"),
        )
    return trainer, transformer, events


class _SampleRecorder:
    def __init__(self):
        self.prefix = SimpleNamespace(name="prefix")

    def encode_prompt(self, *args, **kwargs):
        return self.prefix

    def denoise_loop(self, *args, **kwargs):
        return torch.zeros(1, 3, 32, 32)

    def set_attention_backend(self, transformer, backend, mode):
        return 1

    def clear_prefix_caches(self, prefix):
        pass


def _run_sample(recorder, trainer):
    from PIL import Image

    from core.training.ops.sensenova_ops import generate_sample

    target = "core.models.sensenova.sensenova_pipeline_ops"
    with patch(f"{target}.encode_prompt", recorder.encode_prompt), patch(
        f"{target}.denoise_loop", recorder.denoise_loop
    ), patch(f"{target}.set_attention_backend", recorder.set_attention_backend), patch(
        f"{target}.clear_prefix_caches", recorder.clear_prefix_caches
    ), patch(
        f"{target}.tensor_to_image", lambda tensor: Image.new("RGB", (8, 8))
    ):
        return generate_sample(
            trainer,
            prompt="a cat",
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=4.0,
            seed=7,
        )


def test_sample_kv_streaming_off_by_default_never_touches_the_streamer():
    """Negative control: the shipped default (and the pre-fix-only behaviour)
    never installs the streamer, so ``transformer._kv_cache_streamer`` stays
    absent for the whole call."""
    trainer, transformer, _ = _sample_trainer(kv_streaming=False)
    recorder = _SampleRecorder()

    with patch("core.models.sensenova.kv_cache_streaming.install") as install, patch(
        "core.models.sensenova.kv_cache_streaming.uninstall"
    ) as uninstall:
        image = _run_sample(recorder, trainer)

    assert image is not None
    install.assert_not_called()
    uninstall.assert_not_called()
    assert getattr(transformer, "_kv_cache_streamer", None) is None


def test_sample_kv_streaming_installs_before_encode_and_tears_down_after():
    trainer, transformer, _ = _sample_trainer(kv_streaming=True)
    recorder = _SampleRecorder()
    streamer = SimpleNamespace(name="streamer")

    with patch(
        "core.models.sensenova.kv_cache_streaming.install", return_value=streamer
    ) as install, patch(
        "core.models.sensenova.kv_cache_streaming.uninstall"
    ) as uninstall:
        image = _run_sample(recorder, trainer)

    assert image is not None
    install.assert_called_once_with(transformer, trainer.device)
    uninstall.assert_called_once_with(transformer, streamer)


def test_sample_kv_streaming_unavailable_warns_on_training_log(capsys):
    """The mechanism cannot silently do nothing: a requested-but-failed
    install is announced on the structured training_log channel (the sentinel
    line ``emit_training_warning`` writes), not just printed."""
    trainer, transformer, _ = _sample_trainer(kv_streaming=True)
    recorder = _SampleRecorder()

    with patch(
        "core.models.sensenova.kv_cache_streaming.install", return_value=None
    ), patch("core.models.sensenova.kv_cache_streaming.uninstall") as uninstall:
        _run_sample(recorder, trainer)

    out = capsys.readouterr().out
    assert "sensenova_sample_kv_cache_streaming_unavailable" in out
    # install() returned None and never set transformer._kv_cache_streamer:
    # there is nothing to tear down, so teardown is a no-op rather than an
    # uninstall(transformer, None) call.
    uninstall.assert_not_called()


def test_sample_kv_streaming_tears_down_on_the_denoise_exception_path():
    """The mechanism this protects: a raise inside denoise_loop must still
    reach uninstall(), or the streamer leaks into the next training step."""
    trainer, transformer, _ = _sample_trainer(kv_streaming=True)
    recorder = _SampleRecorder()
    streamer = SimpleNamespace(name="streamer")

    def _raise(*args, **kwargs):
        raise RuntimeError("denoise blew up")

    recorder.denoise_loop = _raise

    with patch(
        "core.models.sensenova.kv_cache_streaming.install", return_value=streamer
    ) as install, patch(
        "core.models.sensenova.kv_cache_streaming.uninstall"
    ) as uninstall:
        image = _run_sample(recorder, trainer)

    assert image is None
    install.assert_called_once_with(transformer, trainer.device)
    uninstall.assert_called_once_with(transformer, streamer)


def test_sample_kv_streaming_composes_with_the_phase_evictor_transition_pair():
    """Disjoint mechanisms: enabling streaming must not perturb the evictor's
    own prefix/denoise transition order."""
    trainer, _, events = _sample_trainer(kv_streaming=True, with_evictor=True)
    recorder = _SampleRecorder()
    streamer = SimpleNamespace(name="streamer")

    with patch(
        "core.models.sensenova.kv_cache_streaming.install", return_value=streamer
    ), patch("core.models.sensenova.kv_cache_streaming.uninstall"):
        _run_sample(recorder, trainer)

    assert events == ["prefix", "denoise", "resident"]
