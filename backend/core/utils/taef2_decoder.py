"""TAEF2 decoder — tiny autoencoder preview for the FLUX.2 (AutoencoderKLFlux2) latent space.

Independently implemented (architecture from madebyollin's TAESD, MIT) so the
project carries no runtime dependency on the upstream taesd package. Used as an
optional, higher-fidelity live-preview decoder for models that share the FLUX.2
VAE latent space (FLUX.2, Lens, Ideogram 4) — selectable alongside the existing
linear RGB-projection ("matrix") preview.

Weights: ``madebyollin/taef2`` (``taef2.safetensors``), 32 latent channels,
``flux_2`` architecture variant (mid-block group-norm in the first three blocks).
Only the decoder is built/loaded (preview is decode-only).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_midblock_gn: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in, n_out), nn.ReLU(), _conv(n_out, n_out), nn.ReLU(), _conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            n_gn = n_in * 4
            self.pool = nn.Sequential(
                nn.Conv2d(n_in, n_gn, 1, bias=False),
                nn.GroupNorm(4, n_gn),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_gn, n_in, 1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


def _build_decoder(latent_channels: int = 32, use_midblock_gn: bool = True) -> nn.Sequential:
    """The TAEF2 (flux_2) decoder: 32-ch latent -> RGB, 8x spatial upscale.

    Layer 0 is a Clamp (no params); the saved checkpoint indices are offset by +1
    to account for it (see TAEF2Decoder.load_from_file).
    """
    mb = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        _Clamp(), _conv(latent_channels, 64), nn.ReLU(),
        _Block(64, 64, **mb), _Block(64, 64, **mb), _Block(64, 64, **mb),
        nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _Block(64, 64), _Block(64, 64),
        nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _Block(64, 64), _Block(64, 64),
        nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _conv(64, 3),
    )


class TAEF2Decoder(nn.Module):
    """Decode-only TAEF2 for live preview. Input: 32-channel FLUX.2-VAE latent."""

    def __init__(self):
        super().__init__()
        self.decoder = _build_decoder(latent_channels=32, use_midblock_gn=True)

    @classmethod
    def from_hub(cls, repo_id: str = "madebyollin/taef2") -> "TAEF2Decoder":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, "taef2.safetensors")
        model = cls()
        model.load_from_file(path)
        return model

    def load_from_file(self, path: str) -> None:
        """Load the decoder weights from taef2.safetensors.

        The checkpoint stores ``decoder.layers.<i>.<suffix>``; map to this module's
        ``decoder`` Sequential by dropping the ``layers`` segment and shifting the
        index by +1 (the Sequential's index 0 is the param-less Clamp).
        """
        from safetensors.torch import load_file

        sd = load_file(path)
        converted: dict = {}
        for k, v in sd.items():
            parts = k.split(".")
            if parts[0] != "decoder" or len(parts) < 3:
                continue  # skip encoder.* (preview is decode-only)
            index = int(parts[2])
            suffix = parts[3:]
            converted[".".join([str(index + 1), *suffix])] = v
        self.decoder.load_state_dict(converted, strict=True)

    @torch.no_grad()
    def decode(self, latent_32ch: torch.Tensor) -> torch.Tensor:
        """Decode a (B, 32, H/8, W/8) latent to a (B, 3, H, W) image in [0, 1]."""
        dtype = next(self.decoder.parameters()).dtype
        device = next(self.decoder.parameters()).device
        x = latent_32ch.to(device=device, dtype=dtype)
        out = self.decoder(x)
        return out.float().clamp(0.0, 1.0)
