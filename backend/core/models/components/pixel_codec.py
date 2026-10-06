"""Identity RGB teacher for calibrating pixel backbones into a VAE space."""
from types import SimpleNamespace

from torch import nn


class _PixelPosterior:
    def __init__(self, pixels):
        self.mean = pixels

    def mode(self):
        return self.mean

    def sample(self, generator=None):
        return self.mean


class RGBPixelCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = dict(in_channels=3, latent_channels=3, scaling_factor=1., shift_factor=0.)

    def encode(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Pixel calibration requires BCHW RGB images")
        return SimpleNamespace(latent_dist=_PixelPosterior(pixels))
