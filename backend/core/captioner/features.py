"""Request-local visual features from an evaluation-mode SigLIP2 tagger."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CaptionFeatures:
    spatial_tokens: torch.Tensor
    spatial_mask: torch.Tensor
    spatial_xy: torch.Tensor
    global_features: torch.Tensor
    tag_logits: torch.Tensor


class SigLIP2FeatureProvider:
    def __init__(self, tagger):
        if not callable(getattr(tagger, "forward_features", None)):
            raise ValueError("Caption features require a PyTorch SigLIP2 tagger")
        self.tagger = tagger

    @torch.no_grad()
    def extract(self, pixel_values, pixel_attention_mask=None, spatial_shapes=None):
        if any(module.training for module in self.tagger.modules()):
            raise ValueError("Caption features require an evaluation-mode tagger")
        if self.tagger.is_naflex:
            if pixel_values.ndim != 3 or spatial_shapes is None or pixel_attention_mask is None:
                raise ValueError("NaFlex requires patch values, mask, and spatial_shapes")
            batch, count = pixel_values.shape[:2]
            if spatial_shapes.shape != (batch, 2):
                raise ValueError("spatial_shapes must be [batch, 2]")
            shapes = spatial_shapes.detach().cpu().tolist()
            mask = pixel_attention_mask.to(device=pixel_values.device, dtype=torch.bool)
            if mask.shape != (batch, count):
                raise ValueError("patch mask shape mismatch")
        else:
            if pixel_values.ndim != 4:
                raise ValueError("Fixed-resolution SigLIP2 requires BCHW pixel values")
            batch = pixel_values.shape[0]
            patch_size = self.tagger.vision_encoder.config.patch_size
            if not isinstance(patch_size, int):
                raise ValueError("Unsupported fixed-resolution patch geometry")
            height, width = pixel_values.shape[-2:]
            if height % patch_size or width % patch_size:
                raise ValueError("Image dimensions must be divisible by patch size")
            shapes = [(height // patch_size, width // patch_size)] * batch
            count = shapes[0][0] * shapes[0][1]
            mask = torch.ones(batch, count, device=pixel_values.device, dtype=torch.bool)
        xy = torch.zeros(batch, count, 2, device=pixel_values.device, dtype=torch.float32)
        for row, (height, width) in enumerate(shapes):
            if int(height) != height or int(width) != width or min(height, width) <= 0:
                raise ValueError("Patch grid dimensions must be positive integers")
            height, width = int(height), int(width)
            valid_count = height * width
            if valid_count > count or not mask[row, :valid_count].all() or mask[row, valid_count:].any():
                raise ValueError("NaFlex patches must form a valid row-major prefix")
            yy, xx = torch.meshgrid(
                (torch.arange(height, device=xy.device) + 0.5) / height,
                (torch.arange(width, device=xy.device) + 0.5) / width,
                indexing="ij",
            )
            xy[row, :valid_count] = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        output = self.tagger.forward_features(pixel_values, pixel_attention_mask, spatial_shapes)
        if output.patch_features.shape[:2] != (batch, count):
            raise ValueError("Encoder patch output does not match input geometry")
        return CaptionFeatures(
            output.patch_features.detach(), mask, xy,
            output.pooled_features.detach(), output.tag_logits.detach(),
        )
