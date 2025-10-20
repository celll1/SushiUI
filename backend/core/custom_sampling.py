"""Custom sampling loop for advanced prompt control

This module provides a custom sampling loop that allows:
- Prompt editing (changing prompts mid-generation)
- Fine-grained control over each denoising step
- Access to intermediate latents

Based on diffusers' pipeline implementation but with added flexibility.
"""

import torch
from typing import Optional, Callable, Dict, Any, Union, List
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image
import numpy as np


def rescale_noise_cfg(noise_cfg: torch.Tensor, noise_pred_text: torch.Tensor, guidance_rescale: float = 0.0) -> torch.Tensor:
    """
    Rescale noise predictions to fix overexposure and improve image quality.

    Based on Section 3.4 from "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    https://arxiv.org/abs/2305.08891

    This is particularly important for v-prediction models to avoid washed out or blurry images.

    Args:
        noise_cfg: The predicted noise tensor after CFG (classifier-free guidance)
        noise_pred_text: The predicted noise tensor from text conditioning only (before CFG)
        guidance_rescale: Rescale factor (0.0 = no rescaling, 0.7 = recommended for v-pred)

    Returns:
        Rescaled noise prediction tensor
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # Rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # Mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def custom_sampling_loop(
    pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    width: int = 512,
    height: int = 512,
    generator: Optional[torch.Generator] = None,
    ancestral_generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds_callback: Optional[Callable[[int], tuple]] = None,
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    step_callback: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
    controlnet_images: Optional[List[Image.Image]] = None,
    controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
    control_guidance_start: Optional[Union[float, List[float]]] = None,
    control_guidance_end: Optional[Union[float, List[float]]] = None,
) -> Image.Image:
    """Custom sampling loop with prompt editing and ControlNet support

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
        generator: Random generator for initial latent generation
        ancestral_generator: Separate generator for stochastic samplers (Euler a, etc.). If None, uses generator.
        latents: Initial latents (optional)
        prompt_embeds_callback: Callback to get new embeddings at each step
            Called with (step_index) -> (prompt_embeds, negative_prompt_embeds, pooled, neg_pooled)
        progress_callback: Callback for progress updates (step, total, latents)
        step_callback: Callback after each step for custom processing
        controlnet_images: List of control images for ControlNet
        controlnet_conditioning_scale: Strength of ControlNet conditioning (float or list)
        control_guidance_start: When to start ControlNet guidance (0.0-1.0, float or list)
        control_guidance_end: When to end ControlNet guidance (0.0-1.0, float or list)

    Returns:
        Generated PIL Image
    """
    device = pipeline.device
    # Use unet's dtype for consistency with diffusers
    dtype = pipeline.unet.dtype

    # Check if SDXL by checking if text_encoder_2 exists (more reliable than isinstance for ControlNet pipelines)
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None
    print(f"[CustomSampling] Pipeline type: {type(pipeline).__name__}, is_sdxl: {is_sdxl}")

    # Use ancestral_generator for stochastic samplers, fallback to generator if not provided
    step_generator = ancestral_generator if ancestral_generator is not None else generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using separate ancestral generator for stochastic sampler")

    # Get components
    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Check if ControlNet is present
    controlnet = getattr(pipeline, 'controlnet', None)
    has_controlnet = controlnet is not None and controlnet_images is not None

    if has_controlnet:
        print(f"[CustomSampling] ControlNet detected, preparing control images")
        # Prepare control images
        if not isinstance(controlnet_images, list):
            controlnet_images = [controlnet_images]

        # Convert PIL images to tensors
        control_image_tensors = []
        for img in controlnet_images:
            if isinstance(img, Image.Image):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                img = torch.from_numpy(np.array(img)).float() / 255.0
                if img.ndim == 2:  # Grayscale
                    img = img.unsqueeze(-1).repeat(1, 1, 3)
                img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            control_image_tensors.append(img.to(device=device, dtype=dtype))

        # Normalize conditioning scales
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.0
        if not isinstance(controlnet_conditioning_scale, list):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(control_image_tensors)

        # Normalize guidance ranges
        if control_guidance_start is None:
            control_guidance_start = 0.0
        if not isinstance(control_guidance_start, list):
            control_guidance_start = [control_guidance_start] * len(control_image_tensors)

        if control_guidance_end is None:
            control_guidance_end = 1.0
        if not isinstance(control_guidance_end, list):
            control_guidance_end = [control_guidance_end] * len(control_image_tensors)

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
    print(f"[CustomSampling] Actual timesteps: {len(timesteps)} (some schedulers like DPM2 use 2x steps)")
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

        # Get ControlNet residuals if present
        down_block_res_samples = None
        mid_block_res_sample = None

        if has_controlnet:
            # Check if this step is within the guidance range
            current_fraction = i / num_inference_steps

            # Calculate active ControlNet scales for this step
            active_scales = []
            for idx, (start, end, scale) in enumerate(zip(control_guidance_start, control_guidance_end, controlnet_conditioning_scale)):
                if start <= current_fraction <= end:
                    active_scales.append(scale)
                else:
                    active_scales.append(0.0)  # Disable ControlNet outside guidance range

            # Only run ControlNet if at least one is active
            if any(s > 0 for s in active_scales):
                with torch.no_grad():
                    # Get ControlNet conditioning
                    if isinstance(controlnet, list):
                        # Multiple ControlNets
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for cn, ctrl_img, scale in zip(controlnet, control_image_tensors, active_scales):
                            if scale > 0:
                                controlnet_kwargs = {
                                    "encoder_hidden_states": prompt_embeds_input,
                                    "controlnet_cond": ctrl_img.repeat(2, 1, 1, 1),
                                    "conditioning_scale": scale,
                                    "return_dict": False,
                                }
                                # Add SDXL-specific conditioning to ControlNet
                                if is_sdxl and added_cond_kwargs:
                                    controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                                ctrl_result = cn(
                                    latent_model_input,
                                    t,
                                    **controlnet_kwargs
                                )
                                down_samples, mid_sample = ctrl_result
                                down_block_res_samples_list.append(down_samples)
                                mid_block_res_sample_list.append(mid_sample)

                        # Sum all ControlNet outputs
                        if down_block_res_samples_list:
                            down_block_res_samples = [
                                sum(samples) for samples in zip(*down_block_res_samples_list)
                            ]
                            mid_block_res_sample = sum(mid_block_res_sample_list)
                    else:
                        # Single ControlNet
                        if active_scales[0] > 0:
                            controlnet_kwargs = {
                                "encoder_hidden_states": prompt_embeds_input,
                                "controlnet_cond": control_image_tensors[0].repeat(2, 1, 1, 1),
                                "conditioning_scale": active_scales[0],
                                "return_dict": False,
                            }
                            # Add SDXL-specific conditioning to ControlNet
                            if is_sdxl and added_cond_kwargs:
                                controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                            down_block_res_samples, mid_block_res_sample = controlnet(
                                latent_model_input,
                                t,
                                **controlnet_kwargs
                            )

        # Predict noise residual
        with torch.no_grad():
            unet_kwargs = {
                "encoder_hidden_states": prompt_embeds_input,
            }
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            # Add SDXL-specific conditioning as a nested dict
            if is_sdxl and added_cond_kwargs:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            noise_pred = unet(
                latent_model_input,
                t,
                **unet_kwargs
            ).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Apply guidance rescale if specified (important for v-prediction models)
        if guidance_rescale > 0.0:
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # Compute previous noisy sample
        # Pass step_generator to ensure reproducibility with stochastic samplers (e.g., Euler a)
        latents = scheduler.step(noise_pred, t, latents, generator=step_generator).prev_sample

        # Progress callback
        # Note: Some schedulers (DPM2, DPM2a) create more timesteps than num_inference_steps
        # so we pass len(timesteps) as the total to avoid showing progress > 100%
        if progress_callback is not None:
            progress_callback(i, len(timesteps), latents)

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


def custom_img2img_sampling_loop(
    pipeline: Union[StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline],
    init_image: Image.Image,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    ancestral_generator: Optional[torch.Generator] = None,
    t_start_override: Optional[int] = None,
    prompt_embeds_callback: Optional[Callable[[int], tuple]] = None,
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    step_callback: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
    controlnet_images: Optional[List[Image.Image]] = None,
    controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
    control_guidance_start: Optional[Union[float, List[float]]] = None,
    control_guidance_end: Optional[Union[float, List[float]]] = None,
) -> Image.Image:
    """Custom img2img sampling loop with prompt editing and ControlNet support

    Args:
        pipeline: The diffusers img2img pipeline
        init_image: Initial PIL image
        prompt_embeds: Initial prompt embeddings
        negative_prompt_embeds: Initial negative prompt embeddings
        pooled_prompt_embeds: Pooled prompt embeds (SDXL only)
        negative_pooled_prompt_embeds: Negative pooled embeds (SDXL only)
        num_inference_steps: Number of denoising steps
        strength: Denoising strength (0.0 to 1.0)
        guidance_scale: CFG scale
        generator: Random generator for reproducibility
        prompt_embeds_callback: Callback to get new embeddings at each step
        progress_callback: Callback for progress updates
        step_callback: Callback after each step
        controlnet_images: List of control images for ControlNet
        controlnet_conditioning_scale: Strength of ControlNet conditioning
        control_guidance_start: When to start ControlNet guidance (0.0-1.0)
        control_guidance_end: When to end ControlNet guidance (0.0-1.0)

    Returns:
        Generated PIL Image
    """
    device = pipeline.device
    dtype = pipeline.unet.dtype

    # Check if SDXL by checking if text_encoder_2 exists
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

    # Use ancestral_generator for stochastic samplers, fallback to generator if not provided
    step_generator = ancestral_generator if ancestral_generator is not None else generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using separate ancestral generator for stochastic sampler")

    # Get components
    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Get image dimensions (save before converting to tensor)
    original_width, original_height = init_image.size

    # Check if ControlNet is present
    controlnet = getattr(pipeline, 'controlnet', None)
    has_controlnet = controlnet is not None and controlnet_images is not None

    if has_controlnet:
        print(f"[CustomSampling] ControlNet detected in img2img, preparing control images")
        # Prepare control images
        if not isinstance(controlnet_images, list):
            controlnet_images = [controlnet_images]

        # Convert PIL images to tensors
        control_image_tensors = []
        for img in controlnet_images:
            if isinstance(img, Image.Image):
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                img = torch.from_numpy(np.array(img)).float() / 255.0
                if img.ndim == 2:  # Grayscale
                    img = img.unsqueeze(-1).repeat(1, 1, 3)
                img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            control_image_tensors.append(img.to(device=device, dtype=dtype))

        # Normalize conditioning scales
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.0
        if not isinstance(controlnet_conditioning_scale, list):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(control_image_tensors)

        # Normalize guidance ranges
        if control_guidance_start is None:
            control_guidance_start = 0.0
        if not isinstance(control_guidance_start, list):
            control_guidance_start = [control_guidance_start] * len(control_image_tensors)

        if control_guidance_end is None:
            control_guidance_end = 1.0
        if not isinstance(control_guidance_end, list):
            control_guidance_end = [control_guidance_end] * len(control_image_tensors)

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Calculate timestep to start from
    if t_start_override is not None:
        # Use explicit t_start (for "Do full steps" mode)
        t_start = t_start_override
        print(f"[CustomSampling] Using explicit t_start={t_start} (Do full steps mode)")
    else:
        # Calculate from strength (standard img2img)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

    timesteps = timesteps[t_start:]

    # Encode initial image to latents
    # Convert PIL image to tensor if needed
    if isinstance(init_image, Image.Image):
        init_image = torch.from_numpy(np.array(init_image)).float() / 255.0
        init_image = init_image.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        init_image = init_image * 2.0 - 1.0  # Normalize to [-1, 1]

    with torch.no_grad():
        init_latents = vae.encode(
            init_image.to(device=device, dtype=dtype)
        ).latent_dist.sample(generator)
        init_latents = init_latents * vae.config.scaling_factor

    # Add noise to latents based on timestep
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = scheduler.add_noise(init_latents, noise, timesteps[0:1])

    # Current prompt embeds
    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    print(f"[CustomSampling] Starting img2img loop with {len(timesteps)} steps (strength={strength})")
    print(f"[CustomSampling] Latents shape: {latents.shape}, dtype: {latents.dtype}")

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Check if prompt should be updated
        if prompt_embeds_callback is not None:
            new_embeds = prompt_embeds_callback(t_start + i)
            if new_embeds is not None:
                current_prompt_embeds, current_negative_prompt_embeds, current_pooled_prompt_embeds, current_negative_pooled_prompt_embeds = new_embeds
                print(f"[CustomSampling] Step {t_start + i}: Updated prompt embeddings")

        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Prepare added conditions for SDXL
        added_cond_kwargs = None
        if is_sdxl:
            original_size = (original_height, original_width)
            crops_coords_top_left = (0, 0)
            target_size = (original_height, original_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)

            if current_pooled_prompt_embeds is not None and current_negative_pooled_prompt_embeds is not None:
                add_text_embeds = torch.cat([current_negative_pooled_prompt_embeds, current_pooled_prompt_embeds], dim=0)
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }
            else:
                print(f"[CustomSampling] Warning: SDXL detected but pooled embeddings are None, skipping added_cond_kwargs")

        # Concatenate prompt embeddings for CFG
        prompt_embeds_input = torch.cat([current_negative_prompt_embeds, current_prompt_embeds])

        # Get ControlNet residuals if present
        down_block_res_samples = None
        mid_block_res_sample = None

        if has_controlnet:
            # Check if this step is within the guidance range
            current_fraction = (t_start + i) / num_inference_steps

            # Calculate active ControlNet scales for this step
            active_scales = []
            for idx, (start, end, scale) in enumerate(zip(control_guidance_start, control_guidance_end, controlnet_conditioning_scale)):
                if start <= current_fraction <= end:
                    active_scales.append(scale)
                else:
                    active_scales.append(0.0)

            # Only run ControlNet if at least one is active
            if any(s > 0 for s in active_scales):
                with torch.no_grad():
                    if isinstance(controlnet, list):
                        # Multiple ControlNets
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for cn, ctrl_img, scale in zip(controlnet, control_image_tensors, active_scales):
                            if scale > 0:
                                controlnet_kwargs = {
                                    "encoder_hidden_states": prompt_embeds_input,
                                    "controlnet_cond": ctrl_img.repeat(2, 1, 1, 1),
                                    "conditioning_scale": scale,
                                    "return_dict": False,
                                }
                                if is_sdxl and added_cond_kwargs:
                                    controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                                ctrl_result = cn(
                                    latent_model_input,
                                    t,
                                    **controlnet_kwargs
                                )
                                down_samples, mid_sample = ctrl_result
                                down_block_res_samples_list.append(down_samples)
                                mid_block_res_sample_list.append(mid_sample)

                        # Sum all ControlNet outputs
                        if down_block_res_samples_list:
                            down_block_res_samples = [
                                sum(samples) for samples in zip(*down_block_res_samples_list)
                            ]
                            mid_block_res_sample = sum(mid_block_res_sample_list)
                    else:
                        # Single ControlNet
                        if active_scales[0] > 0:
                            controlnet_kwargs = {
                                "encoder_hidden_states": prompt_embeds_input,
                                "controlnet_cond": control_image_tensors[0].repeat(2, 1, 1, 1),
                                "conditioning_scale": active_scales[0],
                                "return_dict": False,
                            }
                            if is_sdxl and added_cond_kwargs:
                                controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                            down_block_res_samples, mid_block_res_sample = controlnet(
                                latent_model_input,
                                t,
                                **controlnet_kwargs
                            )

        # Predict noise residual
        with torch.no_grad():
            unet_kwargs = {
                "encoder_hidden_states": prompt_embeds_input,
            }
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            # Add SDXL-specific conditioning as a nested dict
            if is_sdxl and added_cond_kwargs:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            noise_pred = unet(
                latent_model_input,
                t,
                **unet_kwargs
            ).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Apply guidance rescale if specified (important for v-prediction models)
        if guidance_rescale > 0.0:
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # Compute previous noisy sample
        # Pass step_generator to ensure reproducibility with stochastic samplers (e.g., Euler a)
        latents = scheduler.step(noise_pred, t, latents, generator=step_generator).prev_sample

        # Progress callback
        if progress_callback is not None:
            progress_callback(i, len(timesteps), latents)

        # Step callback
        if step_callback is not None:
            callback_kwargs = {"latents": latents}
            callback_kwargs = step_callback(pipeline, t_start + i, t, callback_kwargs)
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


def custom_inpaint_sampling_loop(
    pipeline: Union[StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline],
    init_image: Image.Image,
    mask_image: Image.Image,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    ancestral_generator: Optional[torch.Generator] = None,
    t_start_override: Optional[int] = None,
    prompt_embeds_callback: Optional[Callable[[int], tuple]] = None,
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    step_callback: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
    controlnet_images: Optional[List[Image.Image]] = None,
    controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
    control_guidance_start: Optional[Union[float, List[float]]] = None,
    control_guidance_end: Optional[Union[float, List[float]]] = None,
) -> Image.Image:
    """Custom inpaint sampling loop with prompt editing and ControlNet support"""
    device = pipeline.device
    dtype = pipeline.unet.dtype

    # Check if SDXL by checking if text_encoder_2 exists
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

    # Use ancestral_generator for stochastic samplers, fallback to generator if not provided
    step_generator = ancestral_generator if ancestral_generator is not None else generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using separate ancestral generator for stochastic sampler")

    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Check if this is an inpaint-specific UNet (9 channels) or regular UNet (4 channels)
    # Regular UNets cannot accept concatenated mask+image, so we'll use img2img-style masking
    is_inpaint_unet = unet.config.in_channels == 9
    print(f"[CustomSampling] UNet in_channels: {unet.config.in_channels}, is_inpaint_unet: {is_inpaint_unet}")

    # Get image dimensions (save before converting to tensor)
    original_width, original_height = init_image.size

    # Check if ControlNet is present
    controlnet = getattr(pipeline, 'controlnet', None)
    has_controlnet = controlnet is not None and controlnet_images is not None

    if has_controlnet:
        print(f"[CustomSampling] ControlNet detected in inpaint, preparing control images")
        if not isinstance(controlnet_images, list):
            controlnet_images = [controlnet_images]

        control_image_tensors = []
        for img in controlnet_images:
            if isinstance(img, Image.Image):
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                img = torch.from_numpy(np.array(img)).float() / 255.0
                if img.ndim == 2:
                    img = img.unsqueeze(-1).repeat(1, 1, 3)
                img = img.permute(2, 0, 1).unsqueeze(0)
            control_image_tensors.append(img.to(device=device, dtype=dtype))

        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.0
        if not isinstance(controlnet_conditioning_scale, list):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(control_image_tensors)

        if control_guidance_start is None:
            control_guidance_start = 0.0
        if not isinstance(control_guidance_start, list):
            control_guidance_start = [control_guidance_start] * len(control_image_tensors)

        if control_guidance_end is None:
            control_guidance_end = 1.0
        if not isinstance(control_guidance_end, list):
            control_guidance_end = [control_guidance_end] * len(control_image_tensors)

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Calculate timestep to start from
    if t_start_override is not None:
        # Use explicit t_start (for "Do full steps" mode)
        t_start = t_start_override
        print(f"[CustomSampling] Using explicit t_start={t_start} (Do full steps mode)")
    else:
        # Calculate from strength (standard inpaint)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

    timesteps = timesteps[t_start:]

    # Prepare images
    if isinstance(init_image, Image.Image):
        init_image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
        init_image_tensor = init_image_tensor.permute(2, 0, 1).unsqueeze(0)
        init_image_tensor = init_image_tensor * 2.0 - 1.0
    else:
        init_image_tensor = init_image

    if isinstance(mask_image, Image.Image):
        mask_tensor = torch.from_numpy(np.array(mask_image.convert("L"))).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    else:
        mask_tensor = mask_image

    with torch.no_grad():
        init_latents = vae.encode(
            init_image_tensor.to(device=device, dtype=dtype)
        ).latent_dist.sample(generator)
        init_latents = init_latents * vae.config.scaling_factor

    mask_latent = torch.nn.functional.interpolate(
        mask_tensor.to(device=device, dtype=dtype),
        size=(init_latents.shape[-2], init_latents.shape[-1]),
        mode="nearest"
    )

    # Store original image latents for mask blending (before adding noise)
    image_latents = init_latents.clone()

    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = scheduler.add_noise(init_latents, noise, timesteps[0:1])

    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    print(f"[CustomSampling] Starting inpaint loop with {len(timesteps)} steps")

    for i, t in enumerate(timesteps):
        if prompt_embeds_callback is not None:
            new_embeds = prompt_embeds_callback(t_start + i)
            if new_embeds is not None:
                current_prompt_embeds, current_negative_prompt_embeds, current_pooled_prompt_embeds, current_negative_pooled_prompt_embeds = new_embeds

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Only concatenate mask and masked image for inpaint-specific UNets
        # Regular UNets use post-processing masking instead (see after scheduler.step)
        if is_inpaint_unet:
            # Use original clean image latents, masked to show only non-inpaint regions
            masked_image_latents = image_latents * (1 - mask_latent)
            latent_model_input = torch.cat([latent_model_input, mask_latent.repeat(2, 1, 1, 1), masked_image_latents.repeat(2, 1, 1, 1)], dim=1)

        added_cond_kwargs = None
        if is_sdxl:
            original_size = (original_height, original_width)
            crops_coords_top_left = (0, 0)
            target_size = (original_height, original_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)

            if current_pooled_prompt_embeds is not None and current_negative_pooled_prompt_embeds is not None:
                add_text_embeds = torch.cat([current_negative_pooled_prompt_embeds, current_pooled_prompt_embeds], dim=0)
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }
            else:
                print(f"[CustomSampling] Warning: SDXL detected but pooled embeddings are None, skipping added_cond_kwargs")

        prompt_embeds_input = torch.cat([current_negative_prompt_embeds, current_prompt_embeds])

        # Get ControlNet residuals if present
        down_block_res_samples = None
        mid_block_res_sample = None

        if has_controlnet:
            current_fraction = (t_start + i) / num_inference_steps
            active_scales = []
            for idx, (start, end, scale) in enumerate(zip(control_guidance_start, control_guidance_end, controlnet_conditioning_scale)):
                if start <= current_fraction <= end:
                    active_scales.append(scale)
                else:
                    active_scales.append(0.0)

            if any(s > 0 for s in active_scales):
                with torch.no_grad():
                    if isinstance(controlnet, list):
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for cn, ctrl_img, scale in zip(controlnet, control_image_tensors, active_scales):
                            if scale > 0:
                                controlnet_kwargs = {
                                    "encoder_hidden_states": prompt_embeds_input,
                                    "controlnet_cond": ctrl_img.repeat(2, 1, 1, 1),
                                    "conditioning_scale": scale,
                                    "return_dict": False,
                                }
                                if is_sdxl and added_cond_kwargs:
                                    controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                                ctrl_result = cn(
                                    latent_model_input,
                                    t,
                                    **controlnet_kwargs
                                )
                                down_samples, mid_sample = ctrl_result
                                down_block_res_samples_list.append(down_samples)
                                mid_block_res_sample_list.append(mid_sample)

                        if down_block_res_samples_list:
                            down_block_res_samples = [
                                sum(samples) for samples in zip(*down_block_res_samples_list)
                            ]
                            mid_block_res_sample = sum(mid_block_res_sample_list)
                    else:
                        if active_scales[0] > 0:
                            controlnet_kwargs = {
                                "encoder_hidden_states": prompt_embeds_input,
                                "controlnet_cond": control_image_tensors[0].repeat(2, 1, 1, 1),
                                "conditioning_scale": active_scales[0],
                                "return_dict": False,
                            }
                            if is_sdxl and added_cond_kwargs:
                                controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                            down_block_res_samples, mid_block_res_sample = controlnet(
                                latent_model_input,
                                t,
                                **controlnet_kwargs
                            )

        with torch.no_grad():
            unet_kwargs = {
                "encoder_hidden_states": prompt_embeds_input,
            }
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            # Add SDXL-specific conditioning as a nested dict
            if is_sdxl and added_cond_kwargs:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            noise_pred = unet(
                latent_model_input,
                t,
                **unet_kwargs
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Pass step_generator to ensure reproducibility with stochastic samplers (e.g., Euler a)
        latents = scheduler.step(noise_pred, t, latents, generator=step_generator).prev_sample

        # Apply mask blending ONLY for 4-channel UNets (regular models)
        # 9-channel inpaint UNets handle masking internally via concatenation
        if not is_inpaint_unet:
            init_latents_proper = image_latents  # Use clean original image latents

            # Re-noise original to match the noise level of denoised latents
            # Use NEXT timestep (where denoised latents are), not current timestep
            # Skip re-noising on the last step
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = scheduler.add_noise(
                    init_latents_proper,
                    noise,
                    noise_timestep.unsqueeze(0) if noise_timestep.dim() == 0 else noise_timestep
                )

            # Blend: preserve original outside mask (mask=0), use generated inside mask (mask=1)
            latents = (1 - mask_latent) * init_latents_proper + mask_latent * latents

        if progress_callback is not None:
            progress_callback(i, len(timesteps), latents)

        if step_callback is not None:
            callback_kwargs = step_callback(pipeline, t_start + i, t, {"latents": latents})
            latents = callback_kwargs.get("latents", latents)

    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image[0])
