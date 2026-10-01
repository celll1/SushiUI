"""Native RGBA supervision and independent-frame VAE training batches."""

from dataclasses import dataclass

import torch

from .conditioning import MingConditioning
from .prompting import prepare_inputs, encode_inputs


@dataclass
class MingTrainingBatch:
    clean: torch.Tensor
    conditioning: MingConditioning
    frame_weights: torch.Tensor
    reference_latents: list[torch.Tensor | None] | None

    def sliced(self, start, end):
        return MingTrainingBatch(self.clean[start:end],
            MingConditioning(self.conditioning.queries[start:end], self.conditioning.direct[start:end]),
            self.frame_weights[start:end],
            None if self.reference_latents is None else self.reference_latents[start:end])


def shared_step_inputs(batch):
    """One differentiable transport tensor keeps both streams in shared graph replay."""
    queries, direct = batch.conditioning.queries, batch.conditioning.direct
    if queries.ndim != 3 or direct.ndim != 3 or queries.shape[0] != direct.shape[0]:
        raise ValueError("Ming conditioning requires aligned [B, sequence, width] streams")
    if queries.device != direct.device or queries.dtype != direct.dtype or min(*queries.shape, *direct.shape) < 1:
        raise ValueError("Ming conditioning streams require matching device/dtype and nonempty axes")
    width = max(queries.shape[2], direct.shape[2])
    packed = torch.cat((torch.nn.functional.pad(queries, (0, width-queries.shape[2])),
                        torch.nn.functional.pad(direct, (0, width-direct.shape[2]))), dim=1)
    aux = {"ming_layout": (queries.shape[1], queries.shape[2], direct.shape[2]),
           "ming_frame_weights": batch.frame_weights}
    return packed, aux, batch.reference_latents


def native_step_batch(clean, packed, aux, references):
    if not isinstance(aux, dict) or set(aux) != {"ming_layout", "ming_frame_weights"}:
        raise ValueError("Ming shared step requires explicit native layout and frame weights")
    layout = aux["ming_layout"]
    if not isinstance(layout, (tuple, list)) or len(layout) != 3 or any(type(v) is not int or v < 1 for v in layout):
        raise ValueError("Invalid Ming conditioning transport layout")
    length, query_width, direct_width = layout
    if (not isinstance(packed, torch.Tensor) or packed.ndim != 3 or packed.shape[0] != clean.shape[0]
            or packed.shape[1] <= length or packed.shape[2] != max(query_width, direct_width)):
        raise ValueError("Ming shared step tensor disagrees with its native layout")
    weights = aux["ming_frame_weights"]
    if clean.ndim != 5 or not isinstance(weights, torch.Tensor) or weights.shape != (clean.shape[0], clean.shape[2]):
        raise ValueError("Ming shared step requires native frames and aligned supervision weights")
    if references is not None and len(references) != clean.shape[0]:
        raise ValueError("Ming shared reference list disagrees with its batch")
    return MingTrainingBatch(clean,
        MingConditioning(packed[:, :length, :query_width], packed[:, length:, :direct_width]),
        weights, references)


@torch.no_grad()
def encode_frames(vae, pixels):
    """The Layer F axis is independent images, never VAE temporal compression."""
    if pixels.ndim != 5 or pixels.shape[1] != 4 or min(pixels.shape) < 1:
        raise ValueError("Ming RGBA targets require [B, 4, F, H, W]")
    if pixels.shape[-2] % 16 or pixels.shape[-1] % 16:
        raise ValueError("Ming RGBA pixels require a 16-pixel aligned grid")
    if not pixels.is_floating_point() or not torch.isfinite(pixels).all() or (pixels.abs() > 1).any():
        raise ValueError("Ming RGBA pixels must be finite floats in [-1, 1]")
    encoded = []
    for frame in pixels.unbind(2):
        value = vae.encode(frame.to(device=vae.device, dtype=vae.dtype).unsqueeze(2)).latent_dist.mode()
        if value.ndim != 5 or value.shape[2] != 1:
            raise ValueError("Ming VAE must preserve each independent target frame")
        shift = torch.as_tensor(vae.config.shift_factor, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(vae.config.scaling_factor, device=value.device, dtype=value.dtype)
        encoded.append(((value - shift) * scale).squeeze(2))
    return torch.stack(encoded, dim=2)


def prepare_training_batch(components, *, prompts, references, target_pixels, frame_weights, resolution):
    """Explicit supervision owns slot zero; references never become loss targets."""
    if target_pixels.ndim != 5:
        raise ValueError("Ming target pixels require native batch/channel/frame axes")
    batch, channels, frames, height, width = target_pixels.shape
    profile = components.artifact.profile
    if profile not in {"generation_edit", "layer_decompose"}:
        raise ValueError("Unknown Ming training profile")
    if channels != 4 or batch < 1 or len(prompts) != batch or len(references) != batch:
        raise ValueError("Ming prompts/references and RGBA targets must share a batch")
    if (profile == "generation_edit" and frames != 1) or (profile == "layer_decompose" and frames < 2):
        raise ValueError("Ming profile and target frame roles disagree")
    if profile == "layer_decompose" and any(not images for images in references):
        raise ValueError("Ming Layer supervision requires a reference image per item")
    if (frame_weights.shape != (batch, frames) or not torch.isfinite(frame_weights).all()
            or (frame_weights < 0).any() or (frame_weights.sum(1) <= 0).any()):
        raise ValueError("Ming frame weights require nonnegative [B, F] with supervised items")
    prepared = prepare_inputs(components.processor, prompts, references=references,
                              width=width, height=height, resolution=resolution)
    clean = encode_frames(components.vae, target_pixels)
    latent_references = []
    any_reference = any(value is not None for value in prepared.references)
    if any_reference:
        for pixels in prepared.references:
            if pixels is None:
                latent_references.append(None)
                continue
            images = pixels.reshape(-1, 4, height, width).permute(1, 0, 2, 3).unsqueeze(0)
            latent_references.append(encode_frames(components.vae, images)[0])
    condition = encode_inputs(components.conditioner, prepared,
                              requires_grad=not components.conditioner.cacheable)
    return MingTrainingBatch(clean, condition, frame_weights.detach(),
                             latent_references if any_reference else None)
