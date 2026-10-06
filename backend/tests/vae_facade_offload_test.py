"""Codec method calls must trigger the owner's CPU-offload stage."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.models.components.latent_adapter import NormalizedFrameVAE, NormalizedImageVAE, ImageCodecContract, _Config
from core.models.components.video_latent_adapter import NormalizedVideoVAE, VideoCodecContract


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = _Config(scaling_factor=1., shift_factor=0.)
        self.staged = False

    def encode(self, pixels, **kwargs):
        assert self.staged, "offload hook must stage the codec before encode"
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels))

    def decode(self, latents, return_dict=False, **kwargs):
        assert self.staged, "offload hook must stage the codec before decode"
        return (latents,)


@pytest.mark.parametrize("kind", ["frame", "image", "video"])
def test_encode_decode_trigger_parent_offload_hook_once(kind):
    module = Codec()
    declaration = dict(version=1, method="paired_spatial", token_policy="target_native", source="synthetic",
        old_channels=3, new_channels=3, old_scale=1, new_scale=1,
        input_channels=3, ndim=4, norm="identity", norm_pack=1)
    if kind == "video":
        wrapper = NormalizedVideoVAE(module, VideoCodecContract(ImageCodecContract(3,1,4,"identity",1,3),
            1,"independent_frames_v1"))
        pixels = torch.randn(1,3,2,4,4)
    elif kind == "image":
        wrapper = NormalizedImageVAE(module, declaration)
        pixels = torch.randn(1,3,4,4)
    else:
        wrapper = NormalizedFrameVAE(module, declaration)
        pixels = torch.randn(1,4,2,4,4)
    staged = []
    def pre_forward(owner):
        assert owner is wrapper
        staged.append(owner)
        module.staged = True
    wrapper._hf_hook = SimpleNamespace(pre_forward=pre_forward)
    latents = wrapper.encode(pixels).latent_dist.mode()
    module.staged = False
    result = wrapper.decode(latents, return_dict=False)[0]
    assert len(staged) == 2
    torch.testing.assert_close(result[:,:3], pixels[:,:3])
