"""CPU contract tests for the standalone FoveaTag GPU benchmark."""

import torch

from core.tagger.foveatag.artifact import FoveaArtifactConfig, InferenceMode
from core.tagger.foveatag.gpu_benchmark import _percentile, synthetic_inputs


def test_synthetic_inputs_follow_mode_and_geometry_contract():
    config = FoveaArtifactConfig(num_tags=3)
    mode = InferenceMode(overview_size=32, local_views=2)
    inputs = synthetic_inputs(config, mode, device=torch.device("cpu"), dtype=torch.float32)

    assert inputs["overview_images"].shape == (1, 3, 32, 32)
    assert inputs["local_images"].shape == (1, 2, 3, config.views.local_size, config.views.local_size)
    assert inputs["overview_geometry"].shape == (1, 10)
    assert inputs["local_geometry"].shape == (1, 2, 10)
    assert inputs["local_geometry"][0, 0, 8:].tolist() == [1.0, 1.0]


def test_percentile_uses_sorted_nearest_rank_index():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 3.0
