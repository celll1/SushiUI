"""Text-to-image inference operations for SenseNova SDXL Chimera."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .attention_processor import (
    ChimeraAttentionContext,
    clear_chimera_attention_caches,
    install_chimera_attention_processors,
    set_chimera_attention_context,
)
from .flow import flow_euler_step
from .prefix import encode_chimera_conditioning


@dataclass(frozen=True)
class ChimeraConditioning:
    encoder_hidden_states: torch.Tensor
    pooled_text_embeds: torch.Tensor
    context_positions: torch.Tensor
    fingerprint: str


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().float().cpu().contiguous().numpy()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_conditioning(transformer, tokenizer, bridge, prompt: str) -> ChimeraConditioning:
    output = encode_chimera_conditioning(transformer, tokenizer, bridge, prompt)
    return ChimeraConditioning(
        encoder_hidden_states=output.encoder_hidden_states.detach(),
        pooled_text_embeds=output.pooled_text_embeds.detach(),
        context_positions=output.context_positions.detach(),
        fingerprint=_tensor_digest(
            output.encoder_hidden_states,
            output.pooled_text_embeds,
            output.context_positions,
        ),
    )


def shifted_timesteps(steps: int, shift: float, *, device: torch.device | str) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if shift <= 0:
        raise ValueError(f"timestep shift must be positive, got {shift}")
    time = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
    sigma = 1.0 - time
    sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return 1.0 - sigma


def sdxl_time_ids(
    batch: int,
    *,
    original_height: int,
    original_width: int,
    crop_top: int,
    crop_left: int,
    target_height: int,
    target_width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [
        original_height,
        original_width,
        crop_top,
        crop_left,
        target_height,
        target_width,
    ]
    return torch.tensor([values], device=device, dtype=dtype).repeat(batch, 1)


def _unet_velocity(
    unet,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    conditioning: ChimeraConditioning,
    time_ids: torch.Tensor,
    *,
    cache_metadata: tuple,
) -> torch.Tensor:
    context = ChimeraAttentionContext(
        context_positions=conditioning.context_positions,
        target_height=int(cache_metadata[0]),
        target_width=int(cache_metadata[1]),
        crop_top=int(cache_metadata[2]),
        crop_left=int(cache_metadata[3]),
        prefix_terminal_t=float(cache_metadata[4]),
        cache_key=(conditioning.fingerprint, *cache_metadata),
    )
    set_chimera_attention_context(unet, context)
    added = {
        "text_embeds": conditioning.pooled_text_embeds,
        "time_ids": time_ids,
    }
    return unet(
        sample,
        timestep.expand(sample.shape[0]),
        encoder_hidden_states=conditioning.encoder_hidden_states,
        added_cond_kwargs=added,
        return_dict=False,
    )[0]


def sample_txt2img_latents(
    unet,
    positive: ChimeraConditioning,
    negative: ChimeraConditioning | None,
    *,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    timestep_shift: float = 1.0,
    cfg_mode: str = "sequential",
    original_height: int | None = None,
    original_width: int | None = None,
    crop_top: int = 0,
    crop_left: int = 0,
    attention_backend: str = "normal",
    progress_callback: Callable[[int, int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    if height % 8 or width % 8:
        raise ValueError("Chimera width and height must be divisible by 8")
    if cfg_mode not in {"sequential", "batched"}:
        raise ValueError(f"unsupported Chimera CFG mode: {cfg_mode}")
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sample = torch.randn((1, 4, height // 8, width // 8), generator=generator, device=device, dtype=dtype)
    times = shifted_timesteps(steps, timestep_shift, device=device)
    original_height = int(original_height or height)
    original_width = int(original_width or width)
    time_ids = sdxl_time_ids(
        1,
        original_height=original_height,
        original_width=original_width,
        crop_top=crop_top,
        crop_left=crop_left,
        target_height=height,
        target_width=width,
        device=device,
        dtype=dtype,
    )
    cache_metadata = (height, width, crop_top, crop_left, 0.0, "text-only-v1")
    needs_cfg = negative is not None and float(cfg_scale) > 1.0
    install_chimera_attention_processors(unet, backend=attention_backend)
    try:
        with torch.inference_mode():
            for index in range(steps):
                timestep = times[index].to(dtype=dtype)
                if not needs_cfg:
                    velocity = _unet_velocity(
                        unet, sample, timestep, positive, time_ids, cache_metadata=cache_metadata
                    )
                elif cfg_mode == "batched":
                    combined = ChimeraConditioning(
                        encoder_hidden_states=torch.cat(
                            (negative.encoder_hidden_states, positive.encoder_hidden_states)
                        ),
                        pooled_text_embeds=torch.cat(
                            (negative.pooled_text_embeds, positive.pooled_text_embeds)
                        ),
                        context_positions=torch.cat(
                            (negative.context_positions, positive.context_positions)
                        ),
                        fingerprint=f"{negative.fingerprint}:{positive.fingerprint}",
                    )
                    pair = _unet_velocity(
                        unet,
                        sample.repeat(2, 1, 1, 1),
                        timestep,
                        combined,
                        time_ids.repeat(2, 1),
                        cache_metadata=cache_metadata,
                    )
                    uncond, cond = pair.chunk(2)
                    velocity = uncond + float(cfg_scale) * (cond - uncond)
                else:
                    uncond = _unet_velocity(
                        unet, sample, timestep, negative, time_ids, cache_metadata=cache_metadata
                    )
                    cond = _unet_velocity(
                        unet, sample, timestep, positive, time_ids, cache_metadata=cache_metadata
                    )
                    velocity = uncond + float(cfg_scale) * (cond - uncond)
                sample = flow_euler_step(sample, velocity, times[index], times[index + 1])
                if progress_callback is not None:
                    progress_callback(index + 1, steps, sample)
        return sample
    finally:
        clear_chimera_attention_caches(unet)


def decode_latents(vae, latents: torch.Tensor) -> Image.Image:
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    decoded = vae.decode(latents / scaling + shift, return_dict=False)[0]
    pixels = (decoded.float() / 2.0 + 0.5).clamp(0, 1)
    array = (
        pixels[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    )
    return Image.fromarray(np.asarray(array))
