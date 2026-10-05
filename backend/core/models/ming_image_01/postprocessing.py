"""Shared chroma smoothing over decoded Ming RGB, preserving native alpha."""

import torch
from diffusers.image_processor import VaeImageProcessor
from api.param_defaults import GENERATION_DEFAULTS
from core.inference.color_flatten import flatten_chroma


class MingImageProcessor(VaeImageProcessor):
    def _denormalize_conditionally(self, image, do_denormalize=None):
        image = super()._denormalize_conditionally(image, do_denormalize)
        strength = getattr(self, "color_flatten_strength", GENERATION_DEFAULTS["color_flatten_strength"])
        if strength <= 0:
            return image
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError("Ming Color Flatten requires decoded BCHW RGBA")
        rgb = flatten_chroma(image[:, :3], strength).clamp(0, 1)
        return torch.cat((rgb, image[:, 3:]), dim=1)
