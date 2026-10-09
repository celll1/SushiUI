"""Experimental Anima IO coefficients for packed native latents on a reduced grid."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def packed_io_weights(in_weight, out_weight, channels):
    """Average four native patch embeddings; repeat each native output patch.

    Channels follow pixel_unshuffle order. Padding must be constant inside each
    native patch. Codec calibration and noise covariance are separate gates.
    """
    if type(channels) is not int or channels < 1 or in_weight.ndim != 2:
        raise ValueError("Packed Anima IO requires positive channels and matrix weights")
    hidden = in_weight.shape[0]
    if in_weight.shape != (hidden, (channels + 1) * 4) or out_weight.shape != (channels * 4, hidden):
        raise ValueError("Packed Anima IO requires native 2x2 patches and one mask channel")
    if not torch.isfinite(in_weight).all() or not torch.isfinite(out_weight).all():
        raise ValueError("Packed Anima IO requires finite source weights")
    incoming = in_weight.new_zeros(hidden, channels * 4 + 1, 4)
    source = in_weight.reshape(hidden, channels + 1, 4)
    incoming[:, :channels * 4] = source[:, :channels].reshape(hidden, channels * 4, 1) / 4
    incoming[:, channels * 4] = source[:, channels].mean(-1, keepdim=True)
    outgoing = out_weight.reshape(4, channels, hidden).permute(1, 0, 2).reshape(channels * 4, hidden)
    outgoing = outgoing.unsqueeze(0).expand(4, -1, -1).reshape(channels * 16, hidden).clone()
    return incoming.flatten(1), outgoing


@torch.no_grad()
def packed_head_noise(noise):
    """Noise projection of the initial average-input/repeated-output boundaries."""
    if noise.ndim != 4 or any(size % 2 for size in noise.shape[-2:]):
        raise ValueError("Packed head noise requires an even BCHW grid")
    return F.interpolate(F.avg_pool2d(noise, 2), scale_factor=2, mode="nearest")
