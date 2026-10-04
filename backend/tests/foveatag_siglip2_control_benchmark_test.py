"""CPU validation for the Phase-1 control benchmark inputs."""

import pytest

from core.tagger.foveatag.siglip2_control_benchmark import benchmark_control_views


def test_control_benchmark_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark_control_views("x", "y", [], local_view_counts=[0], warmups=0)
    with pytest.raises(ValueError, match="non-negative"):
        benchmark_control_views("x", "y", ["a"], local_view_counts=[-1], warmups=0)
