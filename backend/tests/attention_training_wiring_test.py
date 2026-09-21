"""CPU checks for training architectures that delegate attention dispatch."""

import os
import sys
from types import SimpleNamespace

import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.ops import acestep_ops, ltx2_ops, qwen_image_21_ops


class _DiffusersTransformer:
    def __init__(self):
        self.backend = None

    def set_attention_backend(self, backend):
        self.backend = backend


class _TransformersModel:
    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation="eager")

    def set_attn_implementation(self, implementation):
        self.config._attn_implementation = implementation


def _trainer(transformer):
    return SimpleNamespace(
        transformer=transformer,
        log_prefix="[test]",
        _resolve_training_backend=lambda backend: backend,
    )


@pytest.mark.parametrize("backend", ["native", "flash"])
def test_ltx2_training_backend_reaches_diffusers(backend):
    trainer = _trainer(_DiffusersTransformer())
    ltx2_ops.setup_attention_backend(trainer, backend)
    assert trainer.transformer.backend == backend


@pytest.mark.parametrize(
    ("backend", "implementation"),
    [("native", "sdpa"), ("flash", "flash_attention_2")],
)
def test_acestep_training_backend_reaches_transformers(backend, implementation):
    trainer = _trainer(_TransformersModel())
    acestep_ops.setup_attention_backend(trainer, backend)
    assert trainer.transformer.config._attn_implementation == implementation


@pytest.mark.parametrize("setup", [ltx2_ops.setup_attention_backend, acestep_ops.setup_attention_backend])
def test_external_dispatchers_refuse_conduit_only_backend(setup):
    with pytest.raises(ValueError, match="supported backends are 'native' and 'flash'"):
        setup(_trainer(_TransformersModel()), "tq")


def test_acestep_vendor_supports_transformers_dynamic_backend_api():
    from core.models.acestep.vendor import AceStepConditionGenerationModel

    assert AceStepConditionGenerationModel._can_set_attn_implementation()


@pytest.mark.parametrize("backend", ["native", "flash"])
def test_qwen_image_21_training_backend_reaches_segmented_processors(backend):
    processors = [SimpleNamespace(_attention_backend=None) for _ in range(3)]
    transformer = SimpleNamespace(
        transformer_blocks=[
            SimpleNamespace(attn=SimpleNamespace(processor=processor))
            for processor in processors
        ]
    )
    qwen_image_21_ops.setup_attention_backend(_trainer(transformer), backend)
    assert [processor._attention_backend for processor in processors] == [backend] * 3


def test_qwen_image_21_refuses_backend_without_exact_segmented_path():
    transformer = SimpleNamespace(transformer_blocks=[])
    with pytest.raises(ValueError, match="supported backends are 'native' and 'flash'"):
        qwen_image_21_ops.setup_attention_backend(_trainer(transformer), "tq")
