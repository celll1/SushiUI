"""Non-finite decoder values must not become successful black PNGs."""

from types import SimpleNamespace

import pytest
import torch

from core.models.sensenova_sdxl_chimera.pipeline_ops import decode_latents


class Codec(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
        self.output = output
        self.calls = 0

    def decode(self, value, *, return_dict):
        self.calls += 1
        return (self.output,)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_latents_never_reach_codec(invalid):
    codec = Codec(torch.zeros(1, 3, 2, 2))
    with pytest.raises(ValueError, match="received non-finite latents"):
        decode_latents(codec, torch.full((1, 4, 2, 2), invalid))
    assert codec.calls == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_pixels_are_not_hidden_by_clipping(invalid):
    pixels = torch.zeros(1, 3, 2, 2)
    pixels[0, 1, 1, 1] = invalid
    with pytest.raises(ValueError, match="produced non-finite pixels"):
        decode_latents(Codec(pixels), torch.zeros(1, 4, 2, 2))


def test_finite_saturated_pixels_still_clip_normally():
    image = decode_latents(Codec(torch.full((1, 3, 2, 2), -2.0)), torch.zeros(1, 4, 2, 2))
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (0, 0, 0)
