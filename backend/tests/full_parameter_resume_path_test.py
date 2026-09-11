"""The unused full-parameter checkpoint reader refuses actionable input."""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.full_parameter_trainer import FullParameterTrainer


class _Stub:
    log_prefix = "[test]"


def test_full_parameter_checkpoint_reader_names_the_supported_resume_path():
    checkpoint = "run_step_000100.safetensors"
    with pytest.raises(NotImplementedError) as caught:
        FullParameterTrainer.load_checkpoint(_Stub(), checkpoint)

    message = str(caught.value)
    assert "resume_from_checkpoint" in message
    assert checkpoint in message
