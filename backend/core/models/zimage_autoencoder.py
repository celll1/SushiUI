"""
Z-Image AutoencoderKL wrapper for SushiUI

This module provides a simple wrapper around diffusers' AutoencoderKL
to maintain compatibility with Z-Image loading code.
"""

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

__all__ = ["AutoencoderKL"]
