"""Custom sampling loop for advanced prompt control

This module provides a custom sampling loop that allows:
- Prompt editing (changing prompts mid-generation)
- Fine-grained control over each denoising step
- Access to intermediate latents

Based on diffusers' pipeline implementation but with added flexibility.
"""

import torch
from typing import Optional, Callable, Dict, Any, Union, List
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
import numpy as np


def custom_sampling_loop(
    pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds_callback: Optional[Callable[[int], tuple]] = None,
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    step_callback: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
) -> Image.Image:
    """Custom sampling loop with prompt editing support

    Args:
        pipeline: The diffusers pipeline (SD or SDXL)
        prompt_embeds: Initial prompt embeddings [batch, seq_len, hidden_size]
        negative_prompt_embeds: Initial negative prompt embeddings
        pooled_prompt_embeds: Pooled prompt embeds (SDXL only)
        negative_pooled_prompt_embeds: Negative pooled embeds (SDXL only)
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        width: Output width
        height: Output height
        generator: Random generator for reproducibility
        latents: Initial latents (optional)
        prompt_embeds_callback: Callback to get new embeddings at each step
            Called with (step_index) -> (prompt_embeds, negative_prompt_embeds, pooled, neg_pooled)
        progress_callback: Callback for progress updates (step, total, latents)
        step_callback: Callback after each step for custom processing

    Returns:
        Generated PIL Image
    """
    device = pipeline.device
    # Use unet's dtype for consistency with diffusers
    dtype = pipeline.unet.dtype
    is_sdxl = isinstance(pipeline, StableDiffusionXLPipeline)

    # Get components
    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Prepare latents
    if latents is None:
        latent_channels = unet.config.in_channels
        latent_height = height // 8
        latent_width = width // 8

        latents = torch.randn(
            (1, latent_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype
        )
        latents = latents * scheduler.init_noise_sigma

    # Current prompt embeds (will be updated by callback)
    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    print(f"[CustomSampling] Starting sampling loop with {num_inference_steps} steps")
    print(f"[CustomSampling] Latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"[CustomSampling] Prompt embeds shape: {prompt_embeds.shape}")

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Check if prompt should be updated at this step
        if prompt_embeds_callback is not None:
            new_embeds = prompt_embeds_callback(i)
            if new_embeds is not None:
                current_prompt_embeds, current_negative_prompt_embeds, current_pooled_prompt_embeds, current_negative_pooled_prompt_embeds = new_embeds
                print(f"[CustomSampling] Step {i}: Updated prompt embeddings")

        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Prepare added conditions for SDXL
        added_cond_kwargs = {}
        if is_sdxl:
            # SDXL requires time_ids
            # Create add_time_ids: [original_height, original_width, crop_top, crop_left, target_height, target_width]
            original_size = (height, width)
            crops_coords_top_left = (0, 0)
            target_size = (height, width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)

            # Duplicate for CFG (negative + positive)
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)

            # Concatenate pooled embeddings for CFG (negative first, then positive)
            if current_pooled_prompt_embeds is not None and current_negative_pooled_prompt_embeds is not None:
                add_text_embeds = torch.cat([current_negative_pooled_prompt_embeds, current_pooled_prompt_embeds], dim=0)
            else:
                add_text_embeds = None

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids
            }

        # Concatenate prompt embeddings for CFG (negative first, then positive)
        prompt_embeds_input = torch.cat([current_negative_prompt_embeds, current_prompt_embeds])

        # Predict noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_input,
                **added_cond_kwargs
            ).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Progress callback
        if progress_callback is not None:
            progress_callback(i, num_inference_steps, latents)

        # Step callback
        if step_callback is not None:
            callback_kwargs = {"latents": latents}
            callback_kwargs = step_callback(pipeline, i, t, callback_kwargs)
            latents = callback_kwargs.get("latents", latents)

    print(f"[CustomSampling] Sampling complete, decoding latents")

    # Decode latents to image
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])

    return image
