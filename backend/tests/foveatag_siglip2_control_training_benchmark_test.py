"""CPU-side validation for the Phase-1 training benchmark."""

import pytest

from core.tagger.foveatag.siglip2_control_training_benchmark import (
    benchmark_control_training,
)


def test_control_training_benchmark_rejects_invalid_work_before_cuda():
    with pytest.raises(ValueError, match="image_paths"):
        benchmark_control_training("model", "vocab", [], local_views=4, warmup_steps=1, measured_steps=1)
    with pytest.raises(ValueError, match="dimensions"):
        benchmark_control_training("model", "vocab", ["x"], local_views=4, warmup_steps=1, measured_steps=0)
