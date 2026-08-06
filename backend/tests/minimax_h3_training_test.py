"""MiniMax-H3 training-path tests (Phase 6b + the post-audit fixes).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_training_test.py -v

Everything here is STATIC: no model is loaded, no GPU is touched. That is not a
convenience, it is the point -- each defect below is a config-time or
collation-time mistake that a 22 GB training run would only reveal minutes in
(or, in F1's case, never reveal at all).

Every test names the defect it pins and how it FAILS IF THE FIX IS REVERTED, so
a future reader can tell a real regression from a cosmetic edit:

  F1  the dtype normalization is UNCONDITIONAL. The pre-fix code only corrected
      torch.float16, so a UI-started run arrived with fp32 -- and fp32 is the
      dtype the FP8 codes then dequantize into inside every forward.
      Revert -> `test_weight_and_training_dtype_are_forced_to_bf16` fails,
      because its input is fp32 and the old condition leaves fp32 alone.
  F4  `audio_loss_weight` cannot be negative (it would INVERT the audio
      gradient). Revert to a bare `float` -> the negative case is accepted.
  F5  the per-clip audio collation normalizes device before stacking. A cache
      HIT returns the latent on the training device and a MISS returns it on the
      CPU, so a batch mixing them used to raise inside torch.stack.
      Revert -> the sentinel below is never moved and the assertion fails.
  F8  batch_size > 1 is refused at CONFIG time, not at the first train step.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.training.ops import minimax_h3_ops as OPS  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.train_runner import _is_bf16_native_base_model  # noqa: E402


# ===========================================================================
# helpers
# ===========================================================================

class _FakeTrainer:
    """The narrow surface `normalize_dtypes` / the batch guard read."""

    def __init__(self, weight_dtype, training_dtype, use_grad_scaler=False, batch_size=1,
                 model_path="M:/model/minimax_h3"):
        self.log_prefix = "[test]"
        self.model_path = model_path
        self.weight_dtype = weight_dtype
        self.training_dtype = training_dtype
        self.dtype = weight_dtype
        self.use_grad_scaler = use_grad_scaler
        self.grad_scaler = object() if use_grad_scaler else None
        self.config = {"batch_size": batch_size}


_MOVED_TO_CPU = []


class _DeviceSentinel(torch.Tensor):
    """A tensor that RECORDS that `.cpu()` was called somewhere on its lineage.

    Stands in for a cache-hit latent that arrives on the training device. A real
    CUDA tensor would prove the same thing, but this test must not allocate on
    the GPU (and must pass on a CPU-only machine), and what is being pinned is
    the CALL, not the device string: the collation has to normalize every input
    before stacking, whatever produced it.

    The flag is module-level rather than per-instance because `detach()` on a
    tensor subclass returns a NEW subclass instance, so a per-instance flag set
    inside `.cpu()` would be set on the detached copy the caller then discards.
    """

    @staticmethod
    def make(rows, cols):
        t = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        return t.as_subclass(_DeviceSentinel)

    def cpu(self, *args, **kwargs):  # noqa: D102
        _MOVED_TO_CPU.append(True)
        return torch.Tensor(self).clone()


# ===========================================================================
# F1 -- unconditional bf16 normalization
# ===========================================================================

@pytest.mark.parametrize("weight,training", [
    (torch.float32, torch.float16),   # the UI's non-bf16-native preset
    (torch.float32, torch.float32),
    (torch.float16, torch.float16),   # the only case the pre-fix code caught
])
def test_weight_and_training_dtype_are_forced_to_bf16(weight, training):
    t = _FakeTrainer(weight, training)
    OPS.normalize_dtypes(t)
    assert t.weight_dtype is torch.bfloat16
    assert t.training_dtype is torch.bfloat16
    # `dtype` is the legacy alias and must track weight_dtype.
    assert t.dtype is torch.bfloat16


def test_bf16_config_is_left_exactly_alone():
    """Negative control for the normalization: it must not fire when it has
    nothing to correct (an unconditional `setattr` would pass the test above
    even if it also clobbered a correct config, and a spurious GradScaler
    teardown would break an unrelated run)."""
    t = _FakeTrainer(torch.bfloat16, torch.bfloat16, use_grad_scaler=True)
    scaler = t.grad_scaler
    OPS.normalize_dtypes(t)
    assert t.weight_dtype is torch.bfloat16 and t.training_dtype is torch.bfloat16
    assert t.use_grad_scaler is True and t.grad_scaler is scaler


def test_grad_scaler_is_dropped_when_training_dtype_changes():
    """A scaler configured from an fp16 training_dtype raises 'Attempting to
    unscale FP16 gradients' once the run is actually bf16."""
    t = _FakeTrainer(torch.float32, torch.float16, use_grad_scaler=True)
    OPS.normalize_dtypes(t)
    assert t.use_grad_scaler is False
    assert t.grad_scaler is None


def test_train_runner_treats_minimax_h3_as_bf16_native():
    """The config that REACHES the trainer must already be bf16; the ops-level
    normalization is the second line of defence, not the only one.

    Path-name branch only -- no checkpoint is read, so this is a pure string
    test. Reverting the `_is_bf16_native_base_model` change makes it False and
    the UI/runner hand the trainer fp32."""
    assert _is_bf16_native_base_model("M:/model/minimax_h3") is True
    assert _is_bf16_native_base_model("/models/MiniMax-H3/fl2va.safetensors") is True
    # Negative control: an unrelated path must not be swept in by the substrings.
    assert _is_bf16_native_base_model("/models/sdxl/base.safetensors") is False


# ===========================================================================
# F8 -- batch_size > 1 refused at config time
# ===========================================================================

def test_batch_size_guard_is_config_time():
    """The refusal lives in `load_components`, before the model, the latent
    cache and the caption cache exist. Exercised directly here because loading
    the real 21 GB DiT is exactly what the guard exists to avoid."""
    t = _FakeTrainer(torch.bfloat16, torch.bfloat16, batch_size=2)
    with pytest.raises(ValueError) as exc:
        OPS.load_components(t)
    msg = str(exc.value)
    assert "batch_size=1" in msg
    assert "gradient_accumulation_steps" in msg  # actionable remedy, not just a no


def test_batch_size_one_passes_the_guard():
    """Negative control: batch 1 must get PAST the guard -- otherwise the test
    above would pass against a guard that refuses everything.

    Pointed at a NONEXISTENT path on purpose: getting past the guard must be
    observable without loading 21 GB of DiT, so the failure that follows is the
    loader's "layout not found", which is proof the guard let it through."""
    t = _FakeTrainer(torch.bfloat16, torch.bfloat16, batch_size=1,
                     model_path="Z:/no/such/minimax_h3_model")
    with pytest.raises(Exception) as exc:
        OPS.load_components(t)
    assert "batch_size" not in str(exc.value)
    assert "layout not found" in str(exc.value).lower()


# ===========================================================================
# F5 -- per-clip audio collation normalizes device before stacking
# ===========================================================================

def _batch(*latents):
    return [({"_clip_audio_latent": lat}, None) for lat in latents]


def test_audio_collation_moves_every_input_to_cpu_before_stacking():
    _MOVED_TO_CPU.clear()
    hit = _DeviceSentinel.make(74, 32)          # cache hit: arrives on the training device
    miss = torch.ones(74, 32)                    # cache miss: already CPU
    out = BaseTrainer._minimax_h3_batch_audio(_batch(hit, miss))
    assert _MOVED_TO_CPU, ("the collation must normalize device before stacking; a batch "
                           "mixing a cache hit with a miss otherwise raises inside torch.stack")
    assert out["audio_latents"].shape == (2, 74, 32)
    assert out["audio_latents"].device.type == "cpu"
    assert out["audio_present"].tolist() == [True, True]


def test_audio_collation_handles_a_silent_item():
    """A source with no audio track contributes zero rows and a False flag; its
    filler must match the real items' shape AND device."""
    out = BaseTrainer._minimax_h3_batch_audio(_batch(torch.ones(74, 32), None))
    assert out["audio_present"].tolist() == [True, False]
    assert out["audio_latents"].shape == (2, 74, 32)
    assert out["audio_latents"].device.type == "cpu"
    assert torch.count_nonzero(out["audio_latents"][1]) == 0


def test_audio_collation_pads_a_short_window():
    """A window at the very end of a source can yield a short audio read (the
    clip's SPAN and its DURATION differ by one frame). The short item is padded
    to the batch shape and still reported as present."""
    out = BaseTrainer._minimax_h3_batch_audio(_batch(torch.ones(74, 32), torch.ones(70, 32)))
    assert out["audio_latents"].shape == (2, 74, 32)
    assert out["audio_present"].tolist() == [True, True]
    assert torch.count_nonzero(out["audio_latents"][1][70:]) == 0


def test_audio_collation_with_no_audio_at_all_emits_no_tensor():
    out = BaseTrainer._minimax_h3_batch_audio(_batch(None, None))
    assert "audio_latents" not in out
    assert out["audio_present"].tolist() == [False, False]


# ===========================================================================
# F4 -- audio_loss_weight bound
# ===========================================================================

def test_audio_loss_weight_rejects_a_negative_value():
    """A negative weight inverts the audio gradient: the audio head would be
    trained AWAY from its target while the video half trains toward it. The
    openapi schema has always said `minimum: 0.0`; the Pydantic model must agree."""
    from pydantic import ValidationError

    from api.routes import TrainingRunCreateRequest

    base = dict(training_method="lora", base_model_path="x")
    assert TrainingRunCreateRequest(**base).audio_loss_weight == 1.0
    assert TrainingRunCreateRequest(**base, audio_loss_weight=0.0).audio_loss_weight == 0.0
    assert TrainingRunCreateRequest(**base, audio_loss_weight=2.5).audio_loss_weight == 2.5
    with pytest.raises(ValidationError):
        TrainingRunCreateRequest(**base, audio_loss_weight=-0.001)


def test_audio_loss_weight_default_matches_the_ssot_and_the_spec():
    import yaml

    from api.param_defaults import TRAINING_DEFAULTS

    assert TRAINING_DEFAULTS["audio_loss_weight"] == 1.0
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, "openapi.yaml"), encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    prop = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]["audio_loss_weight"]
    assert prop["default"] == TRAINING_DEFAULTS["audio_loss_weight"]
    assert prop["minimum"] == 0.0


# ===========================================================================
# F2 -- the full-FT refusal fires from the checkpoint, before any load
# ===========================================================================

def test_full_finetune_refusal_reads_the_capability_table(monkeypatch):
    """`_refuse_unsupported_full_finetune` must raise from DETECTION alone --
    that is what makes it cheaper than the 21 GB DiT + 48 GiB text-encoder load
    the constructor would otherwise perform first."""
    from core.training.full_parameter_trainer import FullParameterTrainer
    import core.model_loader as ML

    monkeypatch.setattr(ML.ModelLoader, "detect_model_type",
                        staticmethod(lambda p: "minimax_h3"))
    with pytest.raises(ValueError) as exc:
        FullParameterTrainer._refuse_unsupported_full_finetune("any/path")
    assert "minimax_h3" in str(exc.value)
    assert "lora" in str(exc.value).lower()

    # Negative control: an architecture that DOES offer full FT is untouched.
    monkeypatch.setattr(ML.ModelLoader, "detect_model_type", staticmethod(lambda p: "sdxl"))
    FullParameterTrainer._refuse_unsupported_full_finetune("any/path")


def test_the_h3_adapter_module_exports_no_full_parameter_adapter():
    """Layer 2 of the three-layer refusal: there is deliberately no
    MiniMaxH3FullParameterAdapter to fall back to."""
    from core.training.adapters import minimax_h3_adapter as A

    assert [n for n in dir(A) if "FullParameter" in n] == []


def test_training_unsupported_table_declares_it():
    """Layer 1: the table a client filters its method dropdown from."""
    from api.arch_capabilities import TRAINING_UNSUPPORTED

    assert "full_finetune" in TRAINING_UNSUPPORTED.get("minimax_h3", {})


# ===========================================================================
# timestep composition + audio-row geometry
# ===========================================================================

def test_minimax_h3_registers_a_uniform_timestep_default():
    """For this arch the sampler's output is the PRE-SHIFT draw u, which
    train_step then puts through shift 12 / shift 3. Uniform u is what
    reproduces the sigma distribution the released model is sampled at, so the
    per-arch default is registered explicitly rather than left to fall through."""
    from api.param_defaults import TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH as T

    assert T["minimax_h3"]["distribution"] == "uniform"
    assert (T["minimax_h3"]["min_timestep"], T["minimax_h3"]["max_timestep"]) == (0.0, 1.0)


def test_expected_audio_rows_matches_the_measured_closed_form():
    """`2 * round(T/24*40)`: 22 -> 74, 39 -> 130 (measured). Used to RECOGNISE a
    short audio read instead of letting it silently become the batch's row
    count."""
    from core.models.components.wiring import LTX2_TEMPORAL, MINIMAX_H3_TEMPORAL
    from core.training.video_loader import expected_audio_rows

    assert expected_audio_rows(22, 1, MINIMAX_H3_TEMPORAL) == 74
    assert expected_audio_rows(39, 1, MINIMAX_H3_TEMPORAL) == 130
    # An arch with no fixed frame rate has no window-level audio latent at all.
    assert expected_audio_rows(9, 1, LTX2_TEMPORAL) == 0


def test_clip_span_docstring_matches_clip_span():
    """F6: the worked example in the docstring said 28 where the code, the test
    suite and the commit message all say 27."""
    from core.models.components.wiring import MINIMAX_H3_TEMPORAL
    from core.training.bucketing import clip_span

    assert clip_span(22, 1, MINIMAX_H3_TEMPORAL, 30.0) == 27
    assert "27" in clip_span.__doc__
    assert "28 frames of a 30 fps" not in clip_span.__doc__
