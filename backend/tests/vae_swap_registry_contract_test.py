"""New training architectures must explicitly declare codec alignment."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.arch import ARCH_REGISTRY
from core.training.vae_swap import VAE_SWAP_ALIGNMENT, initialization_mode


def test_registry_has_no_implicit_zero_alignment():
    assert set(VAE_SWAP_ALIGNMENT) == set(ARCH_REGISTRY)
    for arch, alignment in VAE_SWAP_ALIGNMENT.items():
        if alignment is not None:
            assert initialization_mode({}, arch) == alignment


@pytest.mark.parametrize("mode", ["auto", "zero", "paired_spatial"])
def test_undeclared_architecture_cannot_silently_initialize(mode):
    with pytest.raises(ValueError, match="no declared VAE swap alignment"):
        initialization_mode({"vae_swap_new_channel_init": mode}, "future_arch")


def test_planner_training_has_no_codec_alignment():
    with pytest.raises(ValueError, match="no trainable codec path"):
        initialization_mode({}, "yue2")
