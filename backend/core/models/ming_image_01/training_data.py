"""Native RGBA supervision and independent-frame VAE training batches."""

from dataclasses import dataclass
from contextlib import nullcontext

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


def prepare_latent_batch(components, *, prompts, references, target_pixels, frame_weights, resolution,
                         tensor_cache=None, latent_key=None, conditioning_key=None, vae_device=None, residency=None):
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
    any_reference = any(value is not None for value in prepared.references)
    latent_channels = int(components.vae.config.z_dim)
    spatial_scale = int(getattr(components.vae, "spatial_scale_factor", 8))
    expected = {"clean": ((batch, latent_channels, frames, height//spatial_scale, width//spatial_scale), components.vae.dtype)}
    for index, pixels in enumerate(prepared.references):
        if pixels is not None:
            expected[f"reference_{index}"] = ((latent_channels, pixels.shape[0]//4,
                height//spatial_scale, width//spatial_scale), components.vae.dtype)
    cached = (tensor_cache.read("latents", latent_key, expected=expected)
              if tensor_cache is not None and latent_key is not None else None)
    if cached is not None:
        clean = cached["clean"]
        latent_references = [cached.get(f"reference_{index}") for index in range(batch)]
    else:
        with residency("vae") if residency is not None else nullcontext():
            try:
                if residency is None and vae_device is not None:
                    components.vae.to(device=vae_device)
                clean = encode_frames(components.vae, target_pixels)
                latent_references = []
                for pixels in prepared.references:
                    if pixels is None:
                        latent_references.append(None)
                        continue
                    images = pixels.reshape(-1, 4, height, width).permute(1, 0, 2, 3).unsqueeze(0)
                    latent_references.append(encode_frames(components.vae, images)[0])
            finally:
                if residency is None and vae_device is not None:
                    components.vae.to(device="cpu")
        if tensor_cache is not None and latent_key is not None:
            payload = {"clean": clean.detach().cpu()}
            payload.update({f"reference_{index}": value.detach().cpu()
                            for index, value in enumerate(latent_references) if value is not None})
            tensor_cache.write("latents", latent_key, payload)
    # Conditioning uses native processor inputs, not the already-encoded RGBA copies.
    prepared.references = [None] * batch
    return MingLatentBatch(clean, prepared, frame_weights.detach(),
                           latent_references if any_reference else None, components.conditioner,
                           tensor_cache=tensor_cache, conditioning_key=conditioning_key, residency=residency)



@dataclass
class MingLatentBatch:
    clean: torch.Tensor
    prepared: object
    frame_weights: torch.Tensor
    reference_latents: list[torch.Tensor | None] | None
    conditioner: object
    frozen_conditioning: MingConditioning | None = None
    tensor_cache: object = None
    conditioning_key: str | None = None
    residency: object = None

    def for_step(self, *, encoder_device=None):
        condition = resolve_conditioning(self, encoder_device=encoder_device)
        return MingTrainingBatch(self.clean, condition, self.frame_weights, self.reference_latents)


@dataclass
class MingConditioningBatch:
    prepared: object
    conditioner: object
    frozen_conditioning: MingConditioning | None = None
    tensor_cache: object = None
    conditioning_key: str | None = None
    residency: object = None

    def for_step(self, *, encoder_device=None):
        return resolve_conditioning(self, encoder_device=encoder_device)


def resolve_conditioning(source, *, encoder_device=None):
    trainable = not source.conditioner.cacheable
    if trainable:
        source.frozen_conditioning = None
        source.conditioning_key = None
    condition = source.frozen_conditioning
    if condition is None and source.tensor_cache is not None and source.conditioning_key is not None:
        from .cache import expected_conditioning
        cached = source.tensor_cache.read("conditioning", source.conditioning_key,
            expected=expected_conditioning(source.conditioner, source.prepared), conditioner=source.conditioner)
        if cached is not None:
            condition = MingConditioning(cached["queries"], cached["direct"])
    if condition is None:
        with source.residency("conditioner") if source.residency is not None else nullcontext():
            try:
                if source.residency is None and encoder_device is not None:
                    source.conditioner.to(device=encoder_device)
                condition = encode_inputs(source.conditioner, source.prepared, requires_grad=trainable)
            finally:
                if source.residency is None and encoder_device is not None and not trainable:
                    source.conditioner.to(device="cpu")
        if not trainable and source.tensor_cache is not None and source.conditioning_key is not None:
            source.tensor_cache.write("conditioning", source.conditioning_key,
                {"queries": condition.queries.detach().cpu(), "direct": condition.direct.detach().cpu()},
                conditioner=source.conditioner)
    if not trainable and source.frozen_conditioning is None:
        source.frozen_conditioning = condition.detached()
    return condition

def prepare_training_batch(components, **kwargs):
    return prepare_latent_batch(components, **kwargs).for_step()
