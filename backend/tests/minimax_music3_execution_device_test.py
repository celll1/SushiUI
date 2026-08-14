"""``MiniMaxMusic3Pipeline.execution_device`` / ``.flow_execution_device`` offload-hook resolution.

Under the group-offload configurations the design doc targets (``M:/model/minimax-music3``'s ~8GB leaf-level
group-offload path, or plain ``accelerate`` sequential offload), a component's WEIGHTS rest on CPU/meta between
calls while an offload hook onloads them to the accelerator only for the forward. Reading
``next(component.parameters()).device`` in that state returns the resting device, not the device the next forward
will run on -- creating new tensors there either raises a device-mismatch error inside the forward, or (if the
resting device is ``meta``) runs silently with no error until something calls ``.item()``.

These tests fake the ``accelerate``-hook half of that resolution (an object with an ``execution_device`` attribute
on ``module._hf_hook``, exactly the shape ``diffusers.pipelines.pipeline_utils.DiffusionPipeline._execution_device``
reads) since it does not require a real ``accelerate`` hook to be attached -- just its documented attribute shape.
No GPU is needed: the fake "devices" are ``meta``/``cpu``, chosen only to be distinguishable and constructible
without CUDA.
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline


class _HookedLinear(nn.Module):
    """A tiny module with a fake `_hf_hook.execution_device`, mimicking an accelerate-hooked submodule."""

    def __init__(self, execution_device: torch.device):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self._hf_hook = SimpleNamespace(execution_device=execution_device)


def _pipeline_with_hooked_lm_and_transformer(lm_device: torch.device, transformer_device: torch.device):
    language_model = _HookedLinear(lm_device)
    transformer = _HookedLinear(transformer_device)
    return MiniMaxMusic3Pipeline(
        tokenizer=None,
        language_model=language_model,
        rvq_depth_decoder=None,
        condition_encoder=None,
        transformer=transformer,
        scheduler=None,
        vocoder=None,
        # No `execution_device=` override: exercise the real resolution logic.
    )


def test_execution_device_reads_the_language_models_offload_hook():
    pipeline = _pipeline_with_hooked_lm_and_transformer(
        lm_device=torch.device("meta"), transformer_device=torch.device("cpu")
    )
    assert pipeline.execution_device == torch.device("meta")


def test_flow_execution_device_reads_the_transformers_offload_hook_not_the_language_models():
    # Deliberately different hook devices on LM vs transformer: flow_execution_device must resolve to the
    # TRANSFORMER's, proving it probes its own stage's components rather than reusing `execution_device`'s
    # LM-first order.
    pipeline = _pipeline_with_hooked_lm_and_transformer(
        lm_device=torch.device("meta"), transformer_device=torch.device("cpu")
    )
    assert pipeline.flow_execution_device == torch.device("cpu")
    assert pipeline.flow_execution_device != pipeline.execution_device


def test_execution_device_falls_back_to_parameter_device_with_no_offload_hooks():
    language_model = nn.Linear(2, 2)  # no `_hf_hook`: nothing group-offload- or accelerate-hooked
    transformer = nn.Linear(2, 2)
    pipeline = MiniMaxMusic3Pipeline(
        tokenizer=None,
        language_model=language_model,
        rvq_depth_decoder=None,
        condition_encoder=None,
        transformer=transformer,
        scheduler=None,
        vocoder=None,
    )
    assert pipeline.execution_device == next(language_model.parameters()).device
    assert pipeline.flow_execution_device == next(transformer.parameters()).device


def test_explicit_execution_device_override_wins_over_hooks():
    pipeline = _pipeline_with_hooked_lm_and_transformer(
        lm_device=torch.device("meta"), transformer_device=torch.device("cpu")
    )
    pipeline._execution_device = torch.device("cpu")
    assert pipeline.execution_device == torch.device("cpu")
    assert pipeline.flow_execution_device == torch.device("cpu")
