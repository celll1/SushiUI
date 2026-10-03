"""A frozen-SDXL denoising step for training the style IP-Adapter bridge."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from core.attention import AttentionMode
from core.inference.attention_processors import UnifiedAttnProcessor, set_attention_processor, set_style_ip_context
from .style_dataset import TRAIN


def choose_style_reference_rows(index, target_rows, rng: np.random.Generator) -> np.ndarray:
    """Pair each target with another train image by its artist, preferring distinct content."""
    references = []
    for target in target_rows:
        target = int(target)
        if index.item_split[target] != TRAIN:
            raise ValueError("SDXL style bridge targets must come from the train split")
        artist = int(index.item_artist[target])
        start, end = index.artist_indptr[artist:artist + 2]
        candidates = index.artist_items[start:end]
        candidates = candidates[(candidates != target) & (index.item_split[candidates] == TRAIN)]
        if not candidates.size:
            raise ValueError(f"Artist {artist} has no distinct training reference")
        target_content = set(index.content_ids(target).tolist())
        distinct_content = np.asarray(
            [row for row in candidates if target_content.isdisjoint(index.content_ids(int(row)))],
            dtype=np.int64,
        )
        if distinct_content.size:
            candidates = distinct_content
        references.append(int(rng.choice(candidates)))
    return np.asarray(references, dtype=np.int64)


def install_style_ip_training_processors(unet, attention_type: str = "normal") -> dict:
    """Install the autograd-safe attention path and return processors for restoration."""
    return set_attention_processor(unet, attention_type, mode=AttentionMode.TRAINING)


def sdxl_style_ip_denoising_loss(
    unet, scheduler, adapter, *, clean_latents: torch.Tensor,
    style_embedding: torch.Tensor, prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict[str, torch.Tensor], timesteps: torch.Tensor,
    noise: torch.Tensor, drop_style: bool = False,
) -> torch.Tensor:
    """Predict the SDXL diffusion target from a different same-artist reference.

    The caller owns reference/target pairing, frozen style encoding, timestep
    sampling, optimizer stepping, and the train/validation split.
    """
    if any(parameter.requires_grad for parameter in unet.parameters()):
        raise ValueError("Freeze SDXL U-Net parameters before training the style bridge")
    if not all(isinstance(proc, UnifiedAttnProcessor) and proc.mode == AttentionMode.TRAINING
               for name, proc in unet.attn_processors.items() if name.endswith("attn2.processor")):
        raise ValueError("Install training attention processors before training the style bridge")
    if clean_latents.shape != noise.shape or clean_latents.shape[0] != style_embedding.shape[0]:
        raise ValueError("Latents, noise, and style embedding batch sizes must match")
    if timesteps.ndim != 1 or timesteps.shape[0] != clean_latents.shape[0]:
        raise ValueError("Timesteps must contain one entry per target latent")
    noisy = scheduler.add_noise(clean_latents, noise, timesteps)
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = scheduler.get_velocity(clean_latents, noise, timesteps)
    else:
        raise ValueError(f"Unsupported SDXL prediction type: {prediction_type}")
    context = None if drop_style else adapter.context(style_embedding)
    set_style_ip_context(unet, context)
    try:
        predicted = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds,
                         added_cond_kwargs=added_cond_kwargs).sample
    finally:
        set_style_ip_context(unet, None)
    return F.mse_loss(predicted.float(), target.float())
